"""Score complete ElderlyTech-VQA or Synthetic-Persona-Chat runs."""

import argparse
import json
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path
from statistics import fmean

from nltk.stem import PorterStemmer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "data/elderlytech_vqa/references.jsonl"
EXPECTED_IDS = {f"elderlytech_vqa_{index:03d}" for index in range(144)}
SPC_REFERENCES = ROOT / "data/synthetic_persona_chat/test.jsonl"
SPC_SAMPLE_COUNT = 25_550
STEMMER = PorterStemmer()
SMOOTH = SmoothingFunction().method3
ALPHANUMERIC = re.compile(r"[a-z0-9]+")


def read_rows(path, id_field="sample_id", expected_ids=EXPECTED_IDS):
    rows = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            row = json.loads(line)
            sample_id = row[id_field]
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Missing or invalid {id_field} in {path}")
            if sample_id in rows:
                raise ValueError(f"Duplicate sample {sample_id} in {path}")
            rows[sample_id] = row
    if expected_ids is not None and rows.keys() != expected_ids:
        missing = sorted(expected_ids - rows.keys())
        extra = sorted(rows.keys() - expected_ids)
        raise ValueError(
            f"Incorrect samples in {path}. Missing {len(missing)}: {missing[:10]}. "
            f"Extra {len(extra)}: {extra[:10]}."
        )
    return rows


def read_spc_references(path):
    rows = read_rows(path, id_field="id", expected_ids=None)
    if len(rows) != SPC_SAMPLE_COUNT:
        raise ValueError(f"Expected {SPC_SAMPLE_COUNT:,} test samples in {path}, found {len(rows):,}.")
    references = {}
    for key, row in rows.items():
        completion = row["completion"]
        if len(completion) != 1 or completion[0]["role"] != "assistant":
            raise ValueError(f"Expected one assistant completion for {key}")
        references[key] = {"completion[0].content": completion[0]["content"]}
    return references


def rouge_tokens(text):
    return [
        STEMMER.stem(word) if len(word) > 3 else word
        for word in ALPHANUMERIC.findall(text.lower())
    ]


def rouge_l_f1(reference, prediction):
    if not reference or not prediction:
        return 0.0
    previous = [0] * (len(prediction) + 1)
    for ref_token in reference:
        current = [0]
        for index, pred_token in enumerate(prediction, 1):
            if ref_token == pred_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return 2 * previous[-1] / (len(reference) + len(prediction))


def evaluate(references, results, reference_field="answer_gpt4o"):
    bleu_scores, rouge_scores = [], []
    answer_bigrams = Counter()
    capped_ids, empty_ids = [], []
    for key in sorted(references):
        row = results[key]
        reference = references[key][reference_field]
        prediction = row["answer"]
        if reference_field == "answer_gpt4o" and row["question"] != references[key]["question"]:
            raise ValueError(f"Question differs from reference for {key}")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"Missing {reference_field} for {key}")
        if not isinstance(prediction, str):
            raise ValueError(f"Prediction answer must be a string for {key}")
        if row["reached_token_limit"]:
            capped_ids.append(key)
        pred_words = prediction.split()
        if not pred_words:
            empty_ids.append(key)
        bleu_scores.append(sentence_bleu(
            [reference.split()], pred_words, weights=(0.5, 0.5), smoothing_function=SMOOTH,
        ))
        rouge_scores.append(rouge_l_f1(rouge_tokens(reference), rouge_tokens(prediction)))
        answer_bigrams.update(pairwise(pred_words))

    total_bigrams = answer_bigrams.total()
    return {
        "samples": len(results),
        "bleu_2": fmean(bleu_scores),
        "rouge_l_f1": fmean(rouge_scores),
        "distinct_2": len(answer_bigrams) / total_bigrams if total_bigrams else 0.0,
        "distinct_bigrams": len(answer_bigrams),
        "total_bigrams": total_bigrams,
        "reached_token_limit": len(capped_ids),
        "limited_sample_ids": capped_ids,
        "empty_answers": len(empty_ids),
        "empty_sample_ids": empty_ids,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", metavar=("LABEL", "JSONL"), required=True)
    parser.add_argument("--dataset", choices=("elderlytech-vqa", "synthetic-persona-chat"), default="elderlytech-vqa")
    parser.add_argument("--references", type=Path, help="Override the dataset's default reference file.")
    parser.add_argument("--output", type=Path, help="Optional destination for the JSON report.")
    args = parser.parse_args()
    if len({label for label, _ in args.run}) != len(args.run):
        parser.error("Each --run label must be unique")
    if args.dataset == "synthetic-persona-chat":
        reference_field = "completion[0].content"
        id_field = "id"
        references = read_spc_references(args.references or SPC_REFERENCES)
    else:
        reference_field = "answer_gpt4o"
        id_field = "sample_id"
        references = read_rows(args.references or REFERENCES)
    report = {
        "reference_field": reference_field,
        "metrics_definition": {
            "bleu_2": "Mean of sentence BLEU-2 with NLTK method3 smoothing and case-sensitive whitespace tokenization",
            "rouge_l_f1": "Mean of ROUGE-L F1 with lowercase ASCII alphanumeric tokenization and Porter stemming",
            "distinct_2": "Unique / all within-answer bigrams across predictions using case-sensitive whitespace tokenization",
        },
        "runs": {
            label: evaluate(references, read_rows(Path(path), id_field, references.keys()), reference_field)
            for label, path in args.run
        },
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
