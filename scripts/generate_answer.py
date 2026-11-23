"""Run retrieval and final generation for one sample with an existing inferred profile."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ID = "elderlytech_vqa_000"


def read_sample(path, sample_id):
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row["sample_id"] == sample_id:
                    return row
    raise ValueError(f"{sample_id} not found in {path}")


class FirstTokenTimer:
    def __init__(self, started):
        self.started = started
        self.seconds = None
        self.prompt_seen = False

    def put(self, _tokens):
        # Transformers sends the full prompt before sending the first generated token.
        if not self.prompt_seen:
            self.prompt_seen = True
        elif self.seconds is None:
            self.seconds = perf_counter() - self.started

    def end(self):
        pass


class AnswerModel:
    def __init__(self, settings):
        from openai import OpenAI

        self.model = settings["model"]
        self.client = OpenAI(
            base_url=settings["api_base"], api_key=settings["api_key"],
            timeout=settings["request_timeout"], max_retries=0,
        )

    def generate(self, prompt, max_new_tokens, answer_language):
        messages = [
            {"role": "system", "content": f"Write the final answer in {answer_language}."},
            {"role": "user", "content": prompt},
        ]
        parts = []
        first_token_seconds = None
        finish_reason = None
        started = perf_counter()
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0,
            max_tokens=max_new_tokens, stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                content = choice.delta.content
                if content:
                    if first_token_seconds is None:
                        first_token_seconds = perf_counter() - started
                    parts.append(content)
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason
            finished = perf_counter()
        finally:
            stream.close()
        if finish_reason not in {"stop", "length"}:
            raise RuntimeError(f"Chat completion ended with finish_reason={finish_reason!r}.")
        answer = "".join(parts).strip()
        if not answer:
            raise RuntimeError("The model returned an empty answer.")
        timing = {"generation_seconds": finished - started, "ttrt_seconds": first_token_seconds}
        return answer, finish_reason == "length", timing


def render_prompt(template, question, profile, selected, answer_language):
    selected = [{key: item[key] for key in ("name", "score", "when", "act")} for item in selected]
    fields = {
        "user_question": question,
        "user_profile": json.dumps(profile, ensure_ascii=False, indent=2),
        "retrieved_skills": json.dumps(selected, ensure_ascii=False, indent=2),
    }
    placeholder = r"\{\{(\w+)\}\}"
    if set(re.findall(placeholder, template)) != fields.keys():
        raise ValueError("Generation template placeholders do not match the supplied fields")
    prompt = re.sub(placeholder, lambda match: fields[match[1]], template)
    return f"{prompt.rstrip()}\n\nAnswer the question entirely in {answer_language}."


def main():
    if __package__:
        from .query_settings import load_query_settings
    else:
        from query_settings import load_query_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default=SAMPLE_ID)
    parser.add_argument("--settings", type=Path, default=ROOT / "query_settings.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--answer-language", default="English", help="Answer language for the selected sample.")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.sample_id):
        parser.error("sample ID must contain only letters, digits, underscores or hyphens")
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    settings = load_query_settings(args.settings)

    trace = []

    def print_step(title, value):
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)
        step = f"{title}\n{text}"
        trace.append(step)
        print(f"\n{step}", flush=True)

    dataset = ROOT / "data/elderlytech_vqa"
    profiles = ROOT / "input/elderlytech_vqa_profiles.jsonl"
    sample = read_sample(dataset / "manifest.jsonl", args.sample_id)
    reference = read_sample(dataset / "references.jsonl", args.sample_id)
    question = reference["question"]
    print_step("1. 初始样本", {
        "sample_id": args.sample_id, "image": str(dataset / sample["image_path"]), "question": question,
    })
    profile_record = read_sample(profiles, args.sample_id)
    profile = profile_record["profile"]
    if not isinstance(profile, dict):
        raise ValueError(f"No valid inferred profile for {args.sample_id}")
    print_step("2. Qwen3-VL 画像推理记录", profile_record)
    print_step("3. 数据集参考画像（仅供对照）", reference["user_profile"])

    # Use the same endpoint configuration for retrieval and generation.
    retrieval = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", str(ROOT / "scripts/retrieve_skills.py"),
         "--sample-id", args.sample_id, "--settings", str(args.settings.resolve())],
        stdout=subprocess.PIPE, encoding="utf-8", check=True,
    )
    result = json.loads(retrieval.stdout)
    print_step("4. BGE-M3 检索画像文本", result["query"])
    print_step("BGE-M3 在线检索耗时（毫秒）", result["timing"])
    print_step("5. BGE-M3 排序", "\n".join(
        f"{item['rank']:02d}  {item['score']:7.3f}  {item['name']}"
        for item in result["ranked"]
    ))
    selected = [
        {key: item[key] for key in ("name", "score", "when", "act")}
        for item in result["selected"]
    ]
    print_step(f"6. 入选技能（k={result['k']}, epsilon={result['epsilon']}）", selected)

    template = (ROOT / "prompt/final_generation.txt").read_text(encoding="utf-8-sig")
    prompt = render_prompt(template, question, profile, selected, args.answer_language)
    print_step("7. 填充后的生成提示词", f"System: Write the final answer in {args.answer_language}.\n\n{prompt}")

    model = AnswerModel(settings["llm"])
    answer, reached_limit, timing = model.generate(prompt, args.max_new_tokens, args.answer_language)
    print(f"生成耗时 {timing['generation_seconds']:.3f} s | TTRT {timing['ttrt_seconds']:.3f} s", flush=True)
    if reached_limit:
        raise RuntimeError("Generation reached max-new-tokens. Increase the limit and retry.")
    print_step("8. 最终回答", answer)
    output = ROOT / "output" / f"{args.sample_id}_answer.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(answer + "\n", encoding="utf-8")
    trace_output = output.with_name(f"{args.sample_id}_trace.txt")
    trace_output.write_text("\n\n".join(trace) + "\n", encoding="utf-8")
    print(f"\n已保存 {output}", flush=True)
    print(f"已保存 {trace_output}", flush=True)


if __name__ == "__main__":
    main()
