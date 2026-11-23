"""Generate User-VLM replies for the full converted Synthetic-Persona-Chat test set."""

import argparse
import json
import re
from pathlib import Path
from time import perf_counter

from user_vlm_instruct import ADAPTER, BASE, ROOT, load_model


TEST_SET = ROOT / "data/synthetic_persona_chat/test.jsonl"
OUTPUT = ROOT / "output/user-vlm-3b-instruct-spc"
EXPECTED_SIZE = 25_550
OPENING_SPEAKER = re.compile(r"^User [12]:[ \t]*")
NEXT_SPEAKER = re.compile(r"\n[ \t]*User [12]:")


def read_test_set():
    with TEST_SET.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != EXPECTED_SIZE or len({row["id"] for row in rows}) != EXPECTED_SIZE:
        raise ValueError(f"Expected {EXPECTED_SIZE} distinct test examples")
    return rows


def format_prompt(row, tokenizer):
    instructions, history = row["prompt"]
    if instructions["role"] != "system" or history["role"] != "user":
        raise ValueError(f"Unexpected prompt roles for {row['id']}")
    return (
        f"{tokenizer.bos_token} <|im_start|>USER: {instructions['content']}\n\n"
        f"{history['content']}<|im_end|> ASSISTANT: \n"
    )


def completed_ids(path, expected_ids, limit):
    completed = set()
    with path.open("a+b") as stream:
        stream.seek(0)
        while raw := stream.readline():
            if not raw.endswith(b"\n"):
                stream.truncate(stream.tell() - len(raw))
                break
            row = json.loads(raw)
            sample_id = row["id"]
            if sample_id not in expected_ids or sample_id in completed or row["max_new_tokens"] != limit:
                raise ValueError(f"Unexpected saved result for {sample_id}")
            completed.add(sample_id)
    return completed


def first_reply(text):
    body = OPENING_SPEAKER.sub("", text.lstrip(), count=1)
    boundary = NEXT_SPEAKER.search(body)
    return (body[:boundary.start()] if boundary else body).strip(), boundary is not None


def score_first_reply(result, reused=False):
    answer, stopped = first_reply(result["answer"])
    return {
        **result,
        "raw_answer": result["answer"],
        "answer": answer,
        "finish_reason": "next_speaker" if stopped else ("length" if result["reached_token_limit"] else "eos"),
        "reached_token_limit": result["reached_token_limit"] and not stopped,
        "reused_full_generation": reused,
    }


def speaker_stop(marker, tokenizer, prompt_width):
    def stop(input_ids, scores, **kwargs):
        generated = input_ids[:, prompt_width:]
        candidates = marker(generated, scores, **kwargs)
        hits = candidates.nonzero().flatten()
        if hits.numel():
            decoded = tokenizer.batch_decode(generated[hits].tolist(), skip_special_tokens=True)
            candidates[hits] = candidates.new_tensor([first_reply(text)[1] for text in decoded])
        return candidates

    return stop


def generate_batch(runtime, examples, max_new_tokens, marker):
    torch, device, _, processor, model = runtime
    tokenizer = processor.tokenizer
    inputs = tokenizer.pad(
        {"input_ids": [tokens for _, tokens in examples]},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    ).to(device)
    # Match the image and text-profile runs' attention over the prompt prefix.
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    prompt_width = inputs["input_ids"].shape[1]
    generation_options = {}
    if marker is not None:
        from transformers import StoppingCriteriaList

        generation_options["stopping_criteria"] = StoppingCriteriaList([
            speaker_stop(marker, tokenizer, prompt_width),
        ])
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, **generation_options)
    generated = output[:, prompt_width:].cpu().tolist()
    eos = model.generation_config.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    results = []
    for (row, prompt_ids), token_ids in zip(examples, generated, strict=True):
        end = next((index + 1 for index, token in enumerate(token_ids) if token in eos_ids), None)
        actual_ids = token_ids[:end]
        if marker is not None and end is None:
            while actual_ids and actual_ids[-1] == tokenizer.pad_token_id:
                actual_ids.pop()
        result = {
            "id": row["id"],
            "source_row": row["source_row"],
            "target_turn": row["target_turn"],
            "target_speaker": row["target_speaker"],
            "answer": tokenizer.decode(actual_ids, skip_special_tokens=True).strip(),
            "input_tokens": len(prompt_ids),
            "output_tokens": len(actual_ids),
            "max_new_tokens": max_new_tokens,
            "reached_token_limit": end is None and len(actual_ids) == max_new_tokens,
        }
        results.append(score_first_reply(result) if marker is not None else result)
    return results


def reuse_full_generations(source, destination, completed, expected_ids, limit):
    if not source.is_file():
        return
    with source.open(encoding="utf-8") as stream, destination.open("a", encoding="utf-8", newline="\n") as output:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            sample_id = row["id"]
            if sample_id not in expected_ids or row["max_new_tokens"] != limit:
                raise ValueError(f"Unexpected full generation for {sample_id}")
            if sample_id not in completed:
                output.write(json.dumps(score_first_reply(row, reused=True), ensure_ascii=False) + "\n")
                completed.add(sample_id)
    print(f"{len(completed)} first replies available from saved results", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--first-reply", action="store_true", help="Stop at the next speaker label and score the first reply.")
    parser.add_argument("--torch-site-packages", type=Path)
    parser.add_argument("--peft-site-packages", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.max_new_tokens) < 1:
        parser.error("Batch size and generation limit must be positive")

    rows = read_test_set()
    expected_ids = {row["id"] for row in rows}
    mode = "-first-reply" if args.first_reply else ""
    output_dir = OUTPUT.with_name(f"{OUTPUT.name}{mode}-{args.max_new_tokens}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.jsonl"
    completed = completed_ids(path, expected_ids, args.max_new_tokens)
    if args.first_reply:
        raw = OUTPUT.with_name(f"{OUTPUT.name}-{args.max_new_tokens}") / "results.jsonl"
        reuse_full_generations(raw, path, completed, expected_ids, args.max_new_tokens)
    if len(completed) == len(rows):
        print(f"All {len(rows)} results already saved to {path}", flush=True)
        return

    torch, device, dtype, processor, model = load_model(
        BASE, ADAPTER, args.torch_site_packages, args.peft_site_packages,
    )
    # Collapse the adapter into the base in memory to avoid LoRA matmuls on every decode step.
    model = model.merge_and_unload().eval()
    runtime = torch, device, dtype, processor, model
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    marker = None
    if args.first_reply:
        from transformers import StopStringCriteria

        marker = StopStringCriteria(tokenizer, ["User 1:", "User 2:"])
    pending = [row for row in rows if row["id"] not in completed]
    inputs = tokenizer(
        [format_prompt(row, tokenizer) for row in pending], add_special_tokens=False,
        return_attention_mask=False, return_token_type_ids=False,
    )["input_ids"]
    if not inputs or any(processor.image_token_id in row for row in inputs):
        raise ValueError("Expected a fully text-only prompt")
    examples = sorted(zip(pending, inputs, strict=True), key=lambda example: len(example[1]), reverse=True)
    # Finish short replies in a first pass. Regroup only unfinished prompts so
    # a long continuation cannot keep all the other rows of its batch decoding.
    deferred = []
    if args.first_reply and args.max_new_tokens > 48:
        passes = [(48, examples), (args.max_new_tokens, deferred)]
    else:
        passes = [(args.max_new_tokens, examples)]
    batch_size = args.batch_size
    written = len(completed)
    started = perf_counter()
    with path.open("a", encoding="utf-8", newline="\n", buffering=1) as stream:
        for limit, queue in passes:
            position = 0
            while position < len(queue):
                batch = queue[position:position + batch_size]
                try:
                    results = generate_batch(runtime, batch, limit, marker)
                except torch.cuda.OutOfMemoryError:
                    if batch_size == 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    print(f"Retrying with batch size {batch_size}", flush=True)
                else:
                    for example, result in zip(batch, results, strict=True):
                        if limit < args.max_new_tokens and result["reached_token_limit"]:
                            deferred.append(example)
                            continue
                        result["max_new_tokens"] = args.max_new_tokens
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        written += 1
                    stream.flush()
                    position += len(results)
                    speed = (written - len(completed)) / (perf_counter() - started)
                    print(f"[{written}/{len(rows)}] {speed:.2f} samples/s", flush=True)
                    continue
                torch.cuda.empty_cache()
    print(f"Saved {written} test results to {path}", flush=True)


if __name__ == "__main__":
    main()
