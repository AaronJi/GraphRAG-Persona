"""Run native Qwen3.5 image-to-answer passes with all 43 source principles."""

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import fmean
from time import perf_counter

from generate_answer import FirstTokenTimer, ROOT
from infer_user_profile import parse_profile
from principles import PRINCIPLES, read_principles


DATASET = ROOT / "data/elderlytech_vqa"
MODEL_PATH = Path("D:/Models/Qwen3.5-4B")
SYSTEM_PROMPT = "Return the requested JSON. Write the profile and response entirely in English."


def build_prompt(question, principles):
    profile_prompt = (ROOT / "prompt/user_profile.txt").read_text(encoding="utf-8-sig")
    profile_definition, separator, _ = profile_prompt.partition("\nReturn only a JSON object")
    if not separator:
        raise ValueError("Could not separate the profile definition from its output instructions")
    catalog = "\n\n".join(
        f"{key} {item['category']} / {item['label']}\n"
        f"When: {item['when']}\nAct: {item['act']}"
        for key, item in sorted(principles.items())
    )
    values = {"profile_definition": profile_definition.strip(), "principles": catalog, "question": question}
    template = (ROOT / "prompt/qwen35_direct_baseline.txt").read_text(encoding="utf-8-sig")
    return re.sub(r"\{\{(\w+)\}\}", lambda match: values[match[1]], template).strip()


def load_model(model_path, torch_packages):
    if torch_packages is not None:
        sys.path.insert(0, str(torch_packages.resolve()))
    try:
        import torch
        import torchvision  # Load the matching vision backend with torch.
    finally:
        if torch_packages is not None:
            sys.path.pop(0)
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    print(f"加载 {model_path}（{device}, {dtype}）", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=dtype, device_map={"": device}, local_files_only=True,
    ).eval()
    return torch, device, dtype, processor, model


def run_model(runtime, image_path, prompt, max_new_tokens):
    torch, device, dtype, processor, model = runtime
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt},
        ]},
    ]
    prepare_started = perf_counter()
    inputs = processor.apply_chat_template(
        messages, enable_thinking=False, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(device)
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise RuntimeError("The native processor did not pass visual inputs")
    if device == "cuda":
        torch.cuda.synchronize()
    input_tokens = inputs["input_ids"].shape[1]
    image_tokens = int((inputs["input_ids"] == model.config.image_token_id).sum())
    if not image_tokens:
        raise RuntimeError("The prepared prompt contains no image tokens")
    preprocessing_seconds = perf_counter() - prepare_started
    tokenizer = processor.tokenizer
    eos_ids = list(dict.fromkeys([tokenizer.eos_token_id, model.config.text_config.eos_token_id]))
    with torch.inference_mode():
        started = perf_counter()
        timer = FirstTokenTimer(started)
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id, streamer=timer,
        )
    generated = output[0, input_tokens:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
    timing = {
        "ttft_seconds": timer.seconds,
        "generation_seconds": perf_counter() - started,
        "preprocessing_seconds": preprocessing_seconds,
    }
    return {
        "raw_output": raw, "timing": timing, "device": device, "dtype": str(dtype),
        "torch_version": torch.__version__, "input_tokens": input_tokens, "image_tokens": image_tokens,
        "output_tokens": len(generated),
        "reached_token_limit": len(generated) == max_new_tokens and int(generated[-1]) not in eos_ids,
    }


def parse_output(raw, principles):
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw.strip(), flags=re.S)
    payload = fenced[1] if fenced else raw.strip()
    # The model sometimes emits unquoted zero-padded IDs, which are invalid JSON numbers.
    payload = re.sub(
        r'(^[ \t]*"selected_principles"\s*:\s*\[)([\d,\s]+)(\])',
        lambda match: match[1] + re.sub(r"\d+", r'"\g<0>"', match[2]) + match[3],
        payload, flags=re.M,
    )
    result, end = json.JSONDecoder().raw_decode(payload)
    if not re.fullmatch(r"[}\s]*", payload[end:]):
        raise ValueError("Unexpected content after the generated JSON object")
    parse_profile(json.dumps(result["profile"], ensure_ascii=False), allow_open_role=True)
    selected = result["selected_principles"]
    if not isinstance(selected, list) or not all(
        re.fullmatch(r"[0-9]{1,2}", str(key)) and str(key).zfill(2) in principles
        for key in selected
    ):
        raise ValueError(f"Unknown selected principle identifiers: {selected}")
    result["selected_principles"] = [str(key).zfill(2) for key in selected]
    if not isinstance(result["response"], str) or not result["response"].strip():
        raise ValueError("Missing final response")
    return result


def run_sample(runtime, args, manifest, question, principles):
    sample_id = manifest["sample_id"]
    image = (DATASET / manifest["image_path"]).resolve()
    if not image.is_relative_to(DATASET.resolve()) or not image.is_file():
        raise ValueError(f"Invalid sample image {image}")
    prompt = build_prompt(question, principles)
    (args.output / f"{sample_id}_prompt.txt").write_text(
        f"System:\n{SYSTEM_PROMPT}\n\nUser image:\n{image}\n\nUser text:\n{prompt}\n",
        encoding="utf-8",
    )
    run = run_model(runtime, image, prompt, args.max_new_tokens)
    report = {
        "sample_id": sample_id, "image": str(image), "question": question,
        "model": str(args.model), "principles_source": str(args.principles),
        "principle_count": len(principles), **run,
    }
    save_report(args.output, report, principles)
    return report


def save_report(directory, report, principles):
    sample_id = report["sample_id"]
    report_file = directory / f"{sample_id}_baseline.json"
    try:
        if report["reached_token_limit"]:
            raise ValueError("Generation hit the token limit")
        result = parse_output(report["raw_output"], principles)
    except (ValueError, KeyError, TypeError) as error:
        report["format_error"] = str(error)
    else:
        report.pop("format_error", None)
        report.update(result)
        report["selected_principle_contents"] = {key: principles[key] for key in result["selected_principles"]}
        (directory / f"{sample_id}_answer.txt").write_text(result["response"] + "\n", encoding="utf-8")
    temporary = report_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(report_file)


def load_completed(directory, sample_id, principles):
    path = directory / f"{sample_id}_baseline.json"
    answer = directory / f"{sample_id}_answer.txt"
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["sample_id"] != sample_id:
            raise ValueError(f"Unexpected sample ID in {path}")
        if "format_error" in result and not result["reached_token_limit"]:
            save_report(directory, result, principles)
            if "format_error" not in result:
                print(f"{sample_id}: recovered saved output without rerunning", flush=True)
        if (answer.is_file() and result.get("response")
                and result["response"].strip() == answer.read_text(encoding="utf-8").strip()
                and not result["reached_token_limit"] and "format_error" not in result):
            return result
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="elderlytech_vqa_139")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--principles", type=Path, default=PRINCIPLES)
    parser.add_argument("--output", type=Path, default=ROOT / "output/qwen35-direct-baseline")
    parser.add_argument("--torch-site-packages", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.sample_id) or args.max_new_tokens < 1:
        parser.error("Invalid sample ID or token limit")

    manifests = [json.loads(line) for line in (DATASET / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    references = {
        row["sample_id"]: row["question"] for row in map(
            json.loads, (DATASET / "references.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    samples = manifests if args.all else [row for row in manifests if row["sample_id"] == args.sample_id]
    ids = [row["sample_id"] for row in samples]
    if not ids or len(ids) != len(set(ids)) or (args.all and len(ids) != 144):
        raise ValueError("Unexpected sample selection")
    args.output.mkdir(parents=True, exist_ok=True)
    principles = read_principles(args.principles)
    runtime = None
    results = []
    failures = []
    for manifest in samples:
        sample_id = manifest["sample_id"]
        result = load_completed(args.output, sample_id, principles) if args.resume else None
        if result is None:
            if runtime is None:
                runtime = load_model(args.model, args.torch_site_packages)
            result = run_sample(runtime, args, manifest, references[sample_id], principles)
            timing = result["timing"]
            print(f"{sample_id} TTFT={timing['ttft_seconds']:.3f}s "
                  f"generation={timing['generation_seconds']:.3f}s", flush=True)
        if not args.all:
            print(result["raw_output"], flush=True)
        if "format_error" in result:
            failures.append(sample_id)
            print(f"{sample_id} format error: {result['format_error']}", flush=True)
        else:
            results.append(result)
    if failures:
        raise SystemExit(f"Invalid model outputs saved for {len(failures)} samples: {', '.join(failures)}")
    if args.all:
        summary = {
            "sample_count": len(results),
            "ttft_mean_seconds": fmean(row["timing"]["ttft_seconds"] for row in results),
            "generation_mean_seconds": fmean(row["timing"]["generation_seconds"] for row in results),
        }
        text = json.dumps(summary, indent=2)
        (args.output / "latency_summary.json").write_text(text + "\n", encoding="utf-8")
        print(text, flush=True)


if __name__ == "__main__":
    main()
