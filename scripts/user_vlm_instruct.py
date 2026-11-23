"""Run the local User-VLM-3B-Instruct checkpoint on ElderlyTech-VQA."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from generate_answer import FirstTokenTimer, ROOT, read_sample


BASE = Path("D:/Models/User-VLM-3B-base")
ADAPTER = Path("D:/Models/User-VLM-3B-Instruct")
DATASET = ROOT / "data/elderlytech_vqa"
OUTPUT = ROOT / "output/user-vlm-3b-instruct"


def load_model(base, adapter, torch_packages, peft_packages):
    if torch_packages is not None:
        path = str(torch_packages.resolve())
        sys.path.insert(0, path)
    try:
        import torch
        if torch_packages is not None:
            import torchvision  # Load the vision backend paired with the alternate PyTorch build.
    finally:
        if torch_packages is not None:
            sys.path.remove(path)
    if peft_packages is not None:
        sys.path.append(str(peft_packages.resolve()))
    from peft import PeftModel
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Loading {base} + {adapter} on {device} ({dtype})", flush=True)
    processor = PaliGemmaProcessor.from_pretrained(adapter, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        base, dtype=dtype, device_map={"": device}, local_files_only=True,
    )
    model = PeftModel.from_pretrained(
        model, adapter, is_trainable=False, local_files_only=True,
    ).eval()
    return torch, device, dtype, processor, model


def run_sample(runtime, sample, question, base, adapter, max_new_tokens, profile=None):
    torch, device, dtype, processor, model = runtime

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    started = perf_counter()
    if profile is None:
        from PIL import Image

        image_path = DATASET / sample["image_path"]
        prompt = f"<image> <|im_start|>USER: {question}<|im_end|> ASSISTANT: "
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs.pop("labels", None)
        inputs = inputs.to(device=device, dtype=dtype)
        source_fields = {"input_mode": "image", "image": str(image_path)}
        expected_image_tokens = processor.image_seq_length
    else:
        tokenizer = processor.tokenizer
        prompt = (
            f"{tokenizer.bos_token} <|im_start|>USER: User profile: {profile}\n"
            f"Question: {question}<|im_end|> ASSISTANT: \n"
        )
        inputs = tokenizer(
            prompt, add_special_tokens=False, return_token_type_ids=False, return_tensors="pt",
        ).to(device)
        # Keep PaliGemma's bidirectional attention over the input prefix in both modes.
        inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
        source_fields = {"input_mode": "text-profile", "user_profile": profile}
        expected_image_tokens = 0
    image_tokens = int((inputs["input_ids"] == processor.image_token_id).sum())
    if image_tokens != expected_image_tokens:
        raise ValueError(f"Unexpected number of image tokens: {image_tokens}")
    input_tokens = inputs["input_ids"].shape[1]
    synchronize()
    preprocessing_seconds = perf_counter() - started

    with torch.inference_mode():
        started = perf_counter()
        ttft = FirstTokenTimer(started)
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=ttft,
        )
        synchronize()
        generation_seconds = perf_counter() - started
    generated = output[0, input_tokens:].cpu()
    answer = processor.decode(generated, skip_special_tokens=True).strip()
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    return {
        "sample_id": sample["sample_id"],
        **source_fields,
        "question": question,
        "prompt": prompt,
        "base": str(base),
        "adapter": str(adapter),
        "answer": answer,
        "device": str(device),
        "dtype": str(dtype),
        "input_tokens": input_tokens,
        "image_tokens": image_tokens,
        "max_new_tokens": max_new_tokens,
        "output_tokens": len(generated),
        "reached_token_limit": len(generated) == max_new_tokens and int(generated[-1]) not in eos_ids,
        "preprocessing_seconds": preprocessing_seconds,
        "ttft_seconds": ttft.seconds,
        "generation_seconds": generation_seconds,
    }


def select_samples(sample_id, run_all):
    manifest = DATASET / "manifest.jsonl"
    references = DATASET / "references.jsonl"
    if not run_all:
        return [(read_sample(manifest, sample_id), read_sample(references, sample_id))]

    samples = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines()]
    refs = [json.loads(line) for line in references.read_text(encoding="utf-8-sig").splitlines()]
    references_by_id = {row["sample_id"]: row for row in refs}
    expected_ids = {f"elderlytech_vqa_{index:03d}" for index in range(144)}
    if len(samples) != 144 or len(refs) != 144 or {row["sample_id"] for row in samples} != expected_ids:
        raise ValueError("Expected exactly 144 distinct ElderlyTech-VQA test samples")
    if set(references_by_id) != expected_ids or any(not row["question"].strip() for row in refs):
        raise ValueError("Test references must provide a question for every sample")
    return [(sample, references_by_id[sample["sample_id"]]) for sample in samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-id", default="elderlytech_vqa_139")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--adapter", type=Path, default=ADAPTER)
    parser.add_argument("--input-mode", choices=("image", "text-profile"), default="image")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--torch-site-packages", type=Path)
    parser.add_argument("--peft-site-packages", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    samples = select_samples(args.sample_id, args.all)
    for sample, reference in samples:
        if args.input_mode == "image":
            image_path = DATASET / sample["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
        elif not reference["user_profile"].strip():
            raise ValueError(f"Missing text profile for {sample['sample_id']}")
    base, adapter = args.base.resolve(), args.adapter.resolve()
    mode_suffix = "" if args.input_mode == "image" else "-text-profile"
    output_dir = OUTPUT.with_name(f"{OUTPUT.name}{mode_suffix}-{args.max_new_tokens}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = load_model(base, adapter, args.torch_site_packages, args.peft_site_packages)
    results = []
    for index, (sample, reference) in enumerate(samples, 1):
        profile = reference["user_profile"] if args.input_mode == "text-profile" else None
        result = run_sample(runtime, sample, reference["question"], base, adapter, args.max_new_tokens, profile)
        results.append(result)
        path = output_dir / f"{sample['sample_id']}.json"
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(path)
        if args.all:
            print(f"[{index}/{len(samples)}] {sample['sample_id']}  "
                  f"{result['output_tokens']} tokens  {result['generation_seconds']:.2f}s", flush=True)
        else:
            print(payload, flush=True)

    if args.all:
        combined = output_dir / "results.jsonl"
        temporary = combined.with_suffix(".tmp")
        temporary.write_text(
            "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results),
            encoding="utf-8",
        )
        temporary.replace(combined)
        print(f"Saved {len(results)} samples to {combined}", flush=True)


if __name__ == "__main__":
    main()
