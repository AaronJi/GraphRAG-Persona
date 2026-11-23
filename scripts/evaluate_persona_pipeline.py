"""Evaluate saved GraphRAG pipeline replies on the user-side Synthetic-Persona-Chat test set."""

import argparse
import json
from pathlib import Path

from evaluate_user_vlm import evaluate, read_rows
from generate_answer import ROOT, render_prompt


REFERENCES = ROOT / "data/synthetic_persona_chat_user_side/test.jsonl"
TEST_SIZE = 13_041


def expected_messages(sample, trace):
    """Rebuild the generation input from the persona, past messages and retrieved skills."""
    settings = trace["run_settings"]
    question = (
        f"<conversation_history>\n{json.dumps(sample['history'][:-1], ensure_ascii=False)}\n"
        f"</conversation_history>\n\n<current_user_query>\n{sample['question']}\n</current_user_query>"
    )
    if sample["leading_context"]:
        question = (
            f"<conversation_context>\n{sample['leading_context']}\n</conversation_context>\n\n{question}"
        )
    language = settings["answer_language"]
    prompt = render_prompt(
        settings["generation_template"], question, trace["inferred_profile"],
        trace["retrieval"]["selected"], language,
    )
    return [
        {"role": "system", "content": f"Write the final answer in {language}."},
        {"role": "user", "content": prompt},
    ]


def load_predictions(output_dir, sources):
    results = {}
    common_settings = None
    missing = [
        sample_id for sample_id in sources
        if not (output_dir / f"{sample_id}_pipeline.json").is_file()
    ]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} required pipeline results. First missing IDs: {missing[:10]}"
        )
    for sample_id, source in sources.items():
        path = output_dir / f"{sample_id}_pipeline.json"
        trace = json.loads(path.read_text(encoding="utf-8"))
        sample = trace["sample"]
        expected = {
            "sample_id": sample_id,
            "dataset": "synthetic-persona-chat",
            "split": "test",
            "source_row": source["source_row"],
            "target_turn": source["target_turn"],
            "user_persona": source["user_persona"],
            "history": source["history"],
            "question": source["user_query"],
            "reference_answer": source["target_response"],
            "leading_context": source.get("leading_context", ""),
        }
        if sample != expected:
            raise ValueError(f"Saved sample metadata differs from the user-side test source for {sample_id}")
        settings = trace["run_settings"]
        if settings["dataset"] != "synthetic-persona-chat":
            raise ValueError(f"Wrong pipeline dataset for {sample_id}")
        if common_settings is None:
            common_settings = settings
        elif settings != common_settings:
            raise ValueError(f"Mixed pipeline settings in the selected results at {sample_id}")
        generation = trace["generation"]
        if generation["messages"] != expected_messages(expected, trace):
            raise ValueError(
                f"Generation input differs from the persona/history/query inputs for {sample_id}"
            )
        if not isinstance(generation["answer"], str) or type(generation["reached_token_limit"]) is not bool:
            raise ValueError(f"Invalid saved generation for {sample_id}")
        results[sample_id] = {
            "answer": generation["answer"],
            "reached_token_limit": generation["reached_token_limit"],
        }
    if results.keys() != sources.keys():
        raise ValueError("The selected results and test sample IDs differ")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory containing *_pipeline.json files.")
    parser.add_argument("--references", type=Path, default=REFERENCES)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Evaluate this many consecutive test samples for a bounded run.")
    args = parser.parse_args()
    if args.offset < 0 or (args.limit is not None and args.limit < 1):
        parser.error("Offset must be nonnegative and limit must be positive")

    sources = read_rows(args.references, id_field="id", expected_ids=None)
    if len(sources) != TEST_SIZE:
        raise ValueError(f"Expected {TEST_SIZE:,} user-side test examples, found {len(sources):,}")
    selected_ids = list(sources)[args.offset:None if args.limit is None else args.offset + args.limit]
    if not selected_ids or (args.limit is not None and len(selected_ids) != args.limit):
        raise ValueError("The requested sample range is outside the test set")
    selected = {sample_id: sources[sample_id] for sample_id in selected_ids}
    for sample_id, row in selected.items():
        if (
            row["split"] != "test" or row["user_speaker"] != "User 1" or row["assistant_speaker"] != "User 2"
            or not isinstance(row["user_persona"], str) or not row["user_persona"].strip()
            or not row["history"] or row["history"][-1] != {"role": "user", "content": row["user_query"]}
        ):
            raise ValueError(f"Invalid user-side test example {sample_id}")

    predictions = load_predictions(args.output_dir, selected)
    metrics = evaluate(selected, predictions, reference_field="target_response")
    report = {
        "dataset": "synthetic-persona-chat",
        "persona_owner": "user",
        "reference_file": str(args.references.resolve()),
        "reference_field": "target_response",
        "offset": args.offset,
        "selected_samples": len(selected),
        "full_test_samples": TEST_SIZE,
        "metrics_definition": {
            "bleu_2": "Mean of sentence BLEU-2 with NLTK method3 smoothing and case-sensitive whitespace tokenization",
            "rouge_l_f1": "Mean of ROUGE-L F1 with lowercase ASCII alphanumeric tokenization and Porter stemming",
            "distinct_2": "Unique / all within-answer bigrams across predictions using case-sensitive whitespace tokenization",
        },
        "metrics": metrics,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    path = args.output_dir / "metrics.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    print(payload, end="")


if __name__ == "__main__":
    main()
