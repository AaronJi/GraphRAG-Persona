"""Verify a complete ElderlyTech-VQA run and summarize per-sample latency."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/elderlytech_vqa/manifest.jsonl"


def read_manifest(path):
    with path.open(encoding="utf-8-sig") as stream:
        samples = [json.loads(line) for line in stream if line.strip()]
    ids = {sample["sample_id"] for sample in samples}
    if len(samples) < 2 or len(ids) != len(samples):
        raise ValueError("The manifest must have at least two distinct sample IDs")
    return samples


def sample_latencies(trace):
    vl = trace["vl_seconds"]
    retrieval = trace["retrieval"]["timing"]
    query = retrieval["query_embedding_ms"] / 1000
    ranking = retrieval["dot_product_and_ranking_ms"] / 1000
    generation = trace["generation"]["timing"]
    metrics = {
        "vl": vl,
        "query_embedding": query,
        "dot_product_and_ranking": ranking,
        "retrieval_total": query + ranking,
        "generation": generation["generation_seconds"],
        "ttrt": generation["ttrt_seconds"],
    }
    for name, value in metrics.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid duration for {name}: {value!r}")
    if metrics["ttrt"] > metrics["generation"]:
        raise ValueError("TTRT exceeds total generation time")
    return metrics


def summarize(manifest, output_dir):
    samples = read_manifest(manifest)
    expected = {sample["sample_id"] for sample in samples}
    trace_paths = {
        path.name.removesuffix("_pipeline.json"): path
        for path in output_dir.glob("*_pipeline.json")
    }
    answer_paths = {
        path.name.removesuffix("_answer.txt"): path
        for path in output_dir.glob("*_answer.txt")
    }
    for kind, paths in (("traces", trace_paths), ("answers", answer_paths)):
        missing = sorted(expected - paths.keys())
        unexpected = sorted(paths.keys() - expected)
        if missing or unexpected:
            raise ValueError(
                f"{kind}: {len(missing)} missing (first IDs: {missing[:5]}), "
                f"{len(unexpected)} unexpected (first IDs: {unexpected[:5]})"
            )

    measurements = {}
    for sample in samples:
        sample_id = sample["sample_id"]
        try:
            trace = json.loads(trace_paths[sample_id].read_text(encoding="utf-8-sig"))
            answer = answer_paths[sample_id].read_text(encoding="utf-8-sig").rstrip("\n")
            expected_image = (manifest.parent / sample["image_path"]).resolve()
            if trace["sample"]["sample_id"] != sample_id:
                raise ValueError("trace sample ID does not match the manifest")
            if Path(trace["sample"]["image"]).resolve() != expected_image:
                raise ValueError("trace image does not match the manifest")
            if not answer.strip() or trace["generation"]["answer"] != answer:
                raise ValueError("answer file is empty or differs from the trace")
            ranked = trace["retrieval"]["ranked"]
            if len({row["name"] for row in ranked}) != 43 or len(ranked) != 43:
                raise ValueError("trace does not rank all 43 skills exactly once")
            for name, duration in sample_latencies(trace).items():
                measurements.setdefault(name, []).append(duration)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{sample_id}: {exc}") from exc

    return {
        "sample_count": len(samples),
        "time_unit": "seconds",
        "stddev": "sample (n-1)",
        "metrics": {
            name: {"mean": mean(values), "sample_stddev": stdev(values)}
            for name, values in measurements.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    report = summarize(args.manifest, args.output_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    destination = args.output_dir / "latency_summary.json"
    destination.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"Saved {destination.resolve()}")


if __name__ == "__main__":
    main()
