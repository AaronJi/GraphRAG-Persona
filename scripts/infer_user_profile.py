"""Infer user profiles from a JSONL manifest of local images."""

import argparse
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("D:/Models/Qwen3-VL-2B-Instruct")
EXPLICIT_LABELS = {
    "age_group": {
        "preschool child", "school-age child", "adolescent", "young adult",
        "middle-aged adult", "older adult",
    },
    "gender_presentation": {"male", "female"},
    "role": {
        "student", "teacher / educator", "employee / office worker", "manager / executive",
        "healthcare worker", "engineer / developer", "freelancer / entrepreneur",
        "artist / designer / creative professional", "athlete / coach / sports professional",
        "researcher / scholar", "finance / legal / consulting professional", "service industry worker",
        "military / police / law enforcement", "retired person", "unemployed / job seeker",
    },
}


def read_samples(manifest, offset, limit):
    with manifest.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    selected = rows[offset:offset + limit]
    if not selected:
        raise ValueError(f"No samples at offset {offset} in {manifest}")
    root = manifest.resolve().parent
    samples = []
    for row in selected:
        image = (root / row["image_path"]).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError(f"Invalid image path for {row['sample_id']}: {image}")
        samples.append((row["sample_id"], row["image_path"], image))
    return samples


def parse_profile(raw, *, allow_open_role=False):
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw.strip(), flags=re.S)
    profile = json.loads(fenced[1] if fenced else raw)
    if not isinstance(profile, dict) or profile.keys() != {"explicit", "implicit"}:
        raise ValueError("Expected explicit and implicit objects")
    explicit = profile["explicit"]
    if not isinstance(explicit, dict) or explicit.keys() != EXPLICIT_LABELS.keys():
        raise ValueError("Invalid explicit fields")
    for field, labels in EXPLICIT_LABELS.items():
        value = explicit[field]
        if field == "role" and allow_open_role and isinstance(value, str) and value.strip():
            continue
        if value is not None and (not isinstance(value, str) or value not in labels):
            raise ValueError(f"Invalid label for {field}")
    implicit = profile["implicit"]
    fields = {"personality", "values", "communication_preferences"}
    if not isinstance(implicit, dict) or implicit.keys() != fields:
        raise ValueError("Invalid implicit fields")
    for field, value in implicit.items():
        if value is not None and not (
            isinstance(value, str) and value.strip()
            or isinstance(value, list) and value
            and all(isinstance(item, str) and item.strip() for item in value)
        ):
            raise ValueError(f"Invalid implicit value for {field}")
    return profile


class ProfileModel:
    def __init__(self, model_path, requested_device, threads):
        import torch
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
            Qwen2VLImageProcessorPil,
        )

        self.torch = torch
        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if requested_device == "auto" else requested_device
        if self.device == "cpu":
            torch.set_num_threads(threads)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.image_processor = Qwen2VLImageProcessorPil.from_pretrained(
            model_path, local_files_only=True,
        )
        self.chat_template = json.loads(
            (model_path / "chat_template.json").read_text(encoding="utf-8")
        )["chat_template"]
        print(f"Loading {model_path} on {self.device} ({dtype})", flush=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, device_map={"": self.device}, local_files_only=True,
        ).eval()

    def infer(self, image_path, prompt, max_new_tokens):
        from PIL import Image

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.tokenizer.apply_chat_template(
            messages, chat_template=self.chat_template, tokenize=False, add_generation_prompt=True,
        )
        # Use the PIL image path directly so this image-only workflow needs no video backend.
        with Image.open(image_path) as image:
            vision = self.image_processor(images=image.convert("RGB"), return_tensors="pt")
        image_token = "<|image_pad|>"
        merge_area = self.image_processor.merge_size ** 2
        image_tokens = int(vision["image_grid_thw"][0].prod()) // merge_area
        if text.count(image_token) != 1:
            raise ValueError("Expected one image placeholder in the model chat template")
        text = text.replace(image_token, image_token * image_tokens, 1)
        tokens = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")
        tokens["mm_token_type_ids"] = (
            tokens["input_ids"] == self.model.config.image_token_id
        ).long()
        tokens.update(vision)
        tokens["pixel_values"] = tokens["pixel_values"].to(self.model.dtype)
        tokens = tokens.to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **tokens, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
            )
        answer = generated[0, tokens["input_ids"].shape[1]:]
        return self.tokenizer.decode(answer, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/elderlytech_vqa/manifest.jsonl")
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompt/user_profile.txt")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=ROOT / "input/elderlytech_vqa_profiles.jsonl")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.offset < 0 or min(args.limit, args.threads, args.max_new_tokens) < 1:
        parser.error("offset must be nonnegative and limit, threads and max-new-tokens must be positive")
    if args.output.resolve() in {args.prompt.resolve(), args.manifest.resolve()}:
        parser.error("output must be distinct from the manifest and prompt")

    samples = read_samples(args.manifest, args.offset, args.limit)
    prompt = args.prompt.read_text(encoding="utf-8-sig").strip()
    if not prompt:
        parser.error("prompt must contain text")
    model = ProfileModel(args.model, args.device, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=args.output.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for sample_id, image_path, image in samples:
                print(f"Running {sample_id}", flush=True)
                raw = model.infer(image, prompt, args.max_new_tokens)
                result = {"sample_id": sample_id, "image_path": image_path}
                try:
                    result["profile"] = parse_profile(raw)
                except ValueError as exc:
                    failures += 1
                    result.update(profile=None, parse_error=str(exc), raw_output=raw)
                    print(f"{sample_id}: {exc}", flush=True)
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {len(samples)} results ({failures} invalid) to {args.output}", flush=True)
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
