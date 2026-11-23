"""Wire the existing model and dataset functions together for main.py."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from time import perf_counter

from .generate_answer import AnswerModel, render_prompt
from .infer_user_profile import MODEL_PATH as VL_PATH, ProfileModel, parse_profile, read_samples
from .query_settings import load_query_settings
from .retrieve_skills import BGEM3, INDEX, SKILLS, prepare_skill_index, retrieve_profile


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/elderlytech_vqa"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("elderlytech-vqa", "synthetic-persona-chat"), default="elderlytech-vqa")
    parser.add_argument("--data", type=Path, help="Dataset directory or user-side Synthetic-Persona-Chat JSONL.")
    parser.add_argument("--skills", type=Path, default=SKILLS)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--settings", type=Path, default=ROOT / "query_settings.yaml")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--torch-site-packages", type=Path)
    parser.add_argument("--vl-device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--answer-language", default="English")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compact", action="store_true", help="Print timings and progress; save full traces as usual.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    if args.offset < 0 or min(args.limit, args.threads, args.max_new_tokens) < 1:
        parser.error("offset must be nonnegative, and limit and threads must be positive")
    args.profile_prompt = (ROOT / "prompt/user_profile.txt").read_text(encoding="utf-8-sig").strip()
    args.text_profile_prompt = (ROOT / "prompt/text_user_profile.txt").read_text(encoding="utf-8-sig").strip()
    args.generation_template = (ROOT / "prompt/final_generation.txt").read_text(encoding="utf-8-sig")
    args.query_settings = load_query_settings(args.settings)
    args.data = args.data or (
        DATASET if args.dataset == "elderlytech-vqa"
        else ROOT / "data/synthetic_persona_chat_user_side/test.jsonl"
    )
    return args


def initialize_embedder(args):
    return BGEM3(args.query_settings["embeddings"])


def prepare_index(embedder, args):
    index = prepare_skill_index(embedder, args.skills, args.index)
    args.skills_digest = hashlib.sha256(
        "".join(skill.path.read_text(encoding="utf-8-sig") for skill in index[0]).encode("utf-8"),
    ).hexdigest()
    print(f"已就绪 {len(index[0])} 条 GraphRAG Skill 的离线向量 {args.index}", flush=True)
    return index


def initialize_vl(args):
    if args.dataset == "synthetic-persona-chat":
        return None
    if args.torch_site_packages is not None:
        packages = str(args.torch_site_packages.resolve())
        if not (Path(packages) / "torch").is_dir():
            raise ValueError(f"No torch package in {packages}")
        sys.path.insert(0, packages)
        try:
            __import__("torch")
        finally:
            sys.path.remove(packages)
    return ProfileModel(VL_PATH, args.vl_device, args.threads)


def initialize_llm(args):
    return AnswerModel(args.query_settings["llm"])


def run_settings(args):
    return {
        "dataset": args.dataset,
        "data": str(args.data.resolve()),
        "profile_model": str(VL_PATH) if args.dataset == "elderlytech-vqa" else args.query_settings["llm"]["model"],
        "answer_model": args.query_settings["llm"]["model"],
        "embedding_model": args.query_settings["embeddings"]["model"],
        "inference_backend": "openai-compatible",
        "skills_digest": args.skills_digest,
        "profile_prompt": args.profile_prompt if args.dataset == "elderlytech-vqa" else args.text_profile_prompt,
        "generation_template": args.generation_template,
        "query_settings": {
            "k": args.query_settings["k"], "epsilon": args.query_settings["epsilon"],
            **{name: {key: args.query_settings[name][key] for key in ("api_base", "model", "request_timeout")}
               for name in ("llm", "embeddings")},
        },
        "answer_language": args.answer_language,
        "max_new_tokens": args.max_new_tokens,
    }


def completed_sample(sample, args):
    if not args.resume:
        return False
    report = args.output / f"{sample['sample_id']}_pipeline.json"
    answer = args.output / f"{sample['sample_id']}_answer.txt"
    if not report.is_file() or not answer.is_file():
        return False
    try:
        trace = json.loads(report.read_text(encoding="utf-8"))
        saved_answer = trace["generation"]["answer"]
        return (
            trace["sample"] == sample and trace.get("run_settings") == run_settings(args)
            and isinstance(saved_answer, str) and bool(saved_answer.strip())
            and answer.read_text(encoding="utf-8").strip() == saved_answer.strip()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load_samples(args):
    if args.dataset == "synthetic-persona-chat":
        path = args.data / "test.jsonl" if args.data.is_dir() else args.data
        with path.open(encoding="utf-8-sig") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs in {path}")
        selected = rows[args.offset:args.offset + args.limit]
        if not selected:
            raise ValueError(f"No samples at offset {args.offset} in {path}")
        for row in selected:
            history = row["history"]
            if (
                not isinstance(row["user_persona"], str) or not row["user_persona"].strip()
                or not isinstance(history, list) or not history
                or any(item.get("role") not in {"user", "assistant"} or not isinstance(item.get("content"), str)
                       for item in history)
                or history[-1] != {"role": "user", "content": row["user_query"]}
            ):
                raise ValueError(f"Invalid user-side sample {row['id']}")
            sample = {
                "sample_id": row["id"], "dataset": args.dataset, "split": row["split"],
                "source_row": row["source_row"], "target_turn": row["target_turn"],
                "user_persona": row["user_persona"], "history": history,
                "question": row["user_query"], "reference_answer": row["target_response"],
                "leading_context": row.get("leading_context", ""),
            }
            if not re.fullmatch(r"[a-zA-Z0-9_-]+", sample["sample_id"]):
                raise ValueError(f"Invalid sample ID: {sample['sample_id']}")
            if not completed_sample(sample, args):
                yield sample
        return

    references = {}
    with (args.data / "references.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                references[row["sample_id"]] = row

    for sample_id, _, image in read_samples(args.data / "manifest.jsonl", args.offset, args.limit):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", sample_id):
            raise ValueError(f"Invalid sample ID: {sample_id}")
        reference = references[sample_id]
        sample = {
            "sample_id": sample_id, "dataset": args.dataset, "image": str(image),
            "question": reference["question"], "reference_profile": reference["user_profile"],
            "reference_answer": reference["answer_gpt4o"],
        }
        if not completed_sample(sample, args):
            yield sample


def run_profile_inference(model, sample, args, llm=None):
    started = perf_counter()
    if "image" in sample:
        prompt = args.profile_prompt

        def infer(text):
            return model.infer(Path(sample["image"]), text, 320)
    else:
        if llm is None:
            raise ValueError("The generation client is required for text profile inference")
        evidence = {
            "user_persona": sample["user_persona"],
            "user_utterances": [item["content"] for item in sample["history"] if item["role"] == "user"],
        }
        prompt = args.text_profile_prompt.replace("{{user_evidence}}", json.dumps(evidence, ensure_ascii=False))

        def infer(text):
            raw, reached_limit, _ = llm.generate(text, 512, "English")
            if reached_limit:
                raise ValueError("Text profile generation reached the token limit")
            return raw

    for attempt in range(2):
        try:
            profile = parse_profile(infer(prompt), allow_open_role=True)
            break
        except ValueError as error:
            if attempt:
                raise
            print(f"{sample['sample_id']}: profile format validation failed; retrying once ({error})", flush=True)
            prompt += (
                f"\n\nThe previous output failed format validation ({error}). Output a complete JSON object. "
                "Use the full English labels from the explicit-field enumerations, preserving slashes "
                "and the spaces around them. Use English strings for all populated profile fields."
            )
    return profile, perf_counter() - started


def run_retrieval(embedder, index, profile, args):
    skills, embeddings = index
    return retrieve_profile(
        embedder, skills, embeddings, profile, args.query_settings["k"], args.query_settings["epsilon"],
    )


def run_generation(model, sample, profile, retrieval, args):
    question = sample["question"]
    if "history" in sample:
        question = (
            f"<conversation_history>\n{json.dumps(sample['history'][:-1], ensure_ascii=False)}\n"
            f"</conversation_history>\n\n<current_user_query>\n{question}\n</current_user_query>"
        )
        if sample.get("leading_context"):
            question = f"<conversation_context>\n{sample['leading_context']}\n</conversation_context>\n\n{question}"
    prompt = render_prompt(
        args.generation_template, question, profile, retrieval["selected"], args.answer_language,
    )
    answer, reached_limit, timing = model.generate(prompt, args.max_new_tokens, args.answer_language)
    return {
        "messages": [
            {"role": "system", "content": f"Write the final answer in {args.answer_language}."},
            {"role": "user", "content": prompt},
        ],
        "answer": answer,
        "reached_token_limit": reached_limit,
        "timing": timing,
    }


def save_sample_results(sample, profile, vl_seconds, retrieval, generation, args):
    trace = {
        "sample": sample,
        "run_settings": run_settings(args),
        "inferred_profile": profile,
        "vl_seconds": vl_seconds,
        "retrieval": retrieval,
        "generation": generation,
    }
    text = json.dumps(trace, ensure_ascii=False, indent=2)
    args.output.mkdir(parents=True, exist_ok=True)
    sample_id = sample["sample_id"]
    (args.output / f"{sample_id}_answer.txt").write_text(generation["answer"] + "\n", encoding="utf-8")
    temporary = args.output / f".{sample_id}_pipeline.tmp"
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(args.output / f"{sample_id}_pipeline.json")
    if args.compact:
        print(sample_id, f"VL={vl_seconds:.3f}s", retrieval["timing"], generation["timing"], flush=True)
    else:
        print(text, flush=True)
