"""Associate user-side examples, run native GraphRAG 0.5.0, and export skills."""

import argparse
import asyncio
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml
from datashaper import NoopVerbCallbacks
from graphrag.config.load_config import load_config
from graphrag.index.cache.json_pipeline_cache import JsonPipelineCache
from graphrag.index.create_pipeline_config import create_pipeline_config
from graphrag.index.emit.types import TableEmitterType
from graphrag.index.llm.load_llm import load_llm
from graphrag.index.run import run_pipeline_with_config
from graphrag.index.storage.file_pipeline_storage import FilePipelineStorage
from graphrag.index.utils.hashing import gen_md5_hash
from graphrag.logging.factories import create_progress_reporter
from graphrag.logging.types import ReporterType

from scripts.principles import PRINCIPLES, read_principles
from scripts.query_settings import load_query_settings
from scripts.retrieve_skills import BGEM3, INDEX, SKILLS, prepare_skill_index


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "data/synthetic_persona_chat_user_side/train_sample_1pct.jsonl"


def load_samples(path, limit):
    with path.open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    rows = rows[:limit] if limit is not None else rows
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Training examples must have distinct IDs")
    for row in rows:
        if row["split"] != "train" or row["history"][-1] != {
            "role": "user", "content": row["user_query"],
        }:
            raise ValueError(f"Invalid user-side training example: {row['id']}")
    return rows


def cached_llm(config, namespace):
    # Reuse GraphRAG's own client, JSON handling, rate limiting and cache.
    cache = JsonPipelineCache(FilePipelineStorage(str(Path(config.root_dir) / "cache")))
    return load_llm(namespace, config.llm.type, NoopVerbCallbacks(), cache,
                    config.llm.model_dump(), chat_only=True)


async def associate(config, samples, principles, folder):
    llm = cached_llm(config, "principle_association")
    prompt = (ROOT / "prompt/principle_association.txt").read_text(encoding="utf-8")
    documents = []

    def valid(value):
        ids = value.get("related_principles") if isinstance(value, dict) else None
        return (isinstance(ids, list) and all(isinstance(i, str) and i in principles for i in ids)
                and len(ids) == len(set(ids)))

    with (folder / "associations.jsonl").open("w", encoding="utf-8") as stream:
        for number, sample in enumerate(samples, 1):
            evidence = {key: sample[key] for key in (
                "id", "user_persona", "history", "user_query", "target_response",
            )}
            result = await llm(
                prompt, json=True, name=sample["id"], is_response_valid=valid,
                variables={"principles": json.dumps(principles, ensure_ascii=False),
                           "example": json.dumps(evidence, ensure_ascii=False)},
                model_parameters={"max_tokens": 256},
            )
            ids = result.json["related_principles"]
            stream.write(json.dumps({"sample_id": sample["id"], "related_principles": ids},
                                    ensure_ascii=False) + "\n")
            stream.flush()
            if ids:
                text = json.dumps({
                    **evidence, "leading_context": sample.get("leading_context", ""),
                    "relevant_principles": {key: principles[key] for key in ids},
                }, ensure_ascii=False)
                documents.append({"id": gen_md5_hash({"text": text}, ["text"]),
                                  "title": sample["id"] + ".txt", "text": text})
            print(f"Association {number}/{len(samples)}: {sample['id']} -> {ids}", flush=True)
    if not documents:
        raise ValueError("No semantically related principles; no index documents to process")
    return pd.DataFrame(documents)


async def compile_skills(config, artifacts, directory, level):
    reports = pd.read_parquet(artifacts / "create_final_community_reports.parquet")
    communities = pd.read_parquet(artifacts / "create_final_communities.parquet")
    assigned = {(int(row.community), int(row.level)) for row in communities.itertuples()}
    selected = reports.loc[
        (reports["level"].astype(int) == level)
        & (reports["community"].astype(int) >= 0)
        & reports.apply(lambda row: (int(row["community"]), int(row["level"])) in assigned, axis=1)
    ].sort_values("community")
    expected = {key for key in assigned if key[0] >= 0 and key[1] == level}
    actual = {(int(row.community), int(row.level)) for row in selected.itertuples()}
    if (not expected or actual != expected or selected.duplicated(["community", "level"]).any()
            or not selected["full_content"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()):
        raise ValueError("Expected one complete report for every assigned community at the selected level")
    prompt = (ROOT / "prompt/community_skill.txt").read_text(encoding="utf-8")
    profile = (ROOT / "prompt/user_profile.txt").read_text(encoding="utf-8-sig").split("Return only a JSON object")[0]
    llm = cached_llm(config, "community_to_skill")
    directory.mkdir(parents=True, exist_ok=True)
    records = []

    def valid(value):
        return isinstance(value, dict) and all(
            isinstance(value.get(key), str) and value[key].strip() for key in ("title", "when", "act")
        )

    for number, report in enumerate(selected.itertuples(), 1):
        response = await llm(
            prompt, json=True, name=str(report.id), is_response_valid=valid,
            variables={"profile_definition": profile, "community_report": report.full_content},
            model_parameters={"max_tokens": 512},
        )
        skill = {key: response.json[key] for key in ("title", "when", "act")}
        name = f"graph-c{level}-{int(report.community)}"
        metadata = {
            "name": name, "description": skill["when"],
            "metadata": {"community": int(report.community), "level": level,
                         "report-id": str(report.id), "source-index": str(artifacts.resolve())},
        }
        target = directory / name / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "---\n" + yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False) + "---\n\n"
            + f"# {skill['title']}\n\n## When\n\n{skill['when']}\n\n## Act\n\n{skill['act']}\n",
            encoding="utf-8",
        )
        records.append({"name": name, "report_id": str(report.id),
                        "community": int(report.community), "level": level, **skill})
        print(f"Skill {number}/{len(selected)}: {name}", flush=True)
    (directory / "catalog.json").write_text(
        json.dumps({"skills": [row["name"] for row in records]}, indent=2) + "\n", encoding="utf-8",
    )
    (artifacts / "skills.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8",
    )
    return len(records)


async def run(args):
    if version("graphrag") != "0.5.0":
        raise RuntimeError("This project requires graphrag==0.5.0")
    config = load_config(args.root.resolve())
    raw = yaml.safe_load((args.root / "settings.yaml").read_text(encoding="utf-8"))
    # GraphRAG 0.5.0's loader drops strategy and replaces zero overlap with its default.
    config.cluster_graph.strategy = raw["cluster_graph"]["strategy"]
    config.chunks.overlap = raw["chunks"]["overlap"]
    folder = args.root.resolve() / "output"
    artifacts = folder
    config.storage.base_dir = str(artifacts)
    config.reporting.base_dir = str(folder / "reports")
    pipeline = create_pipeline_config(config)
    samples = load_samples(args.train, args.limit)
    if args.dry_run:
        print(json.dumps({"graphrag": version("graphrag"), "samples": len(samples),
                          "artifacts": str(artifacts),
                          "workflows": [workflow.name for workflow in pipeline.workflows]}, indent=2))
        return
    folder.mkdir(parents=True, exist_ok=True)
    documents = await associate(config, samples, read_principles(PRINCIPLES), folder)
    reporter = create_progress_reporter(ReporterType("print"))
    try:
        # All extraction, merging, description summarization, Leiden and CS are official workflows.
        async for result in run_pipeline_with_config(
            pipeline, dataset=documents, run_id="persona", emit=[TableEmitterType.Parquet],
            progress_reporter=reporter, is_resume_run=args.resume,
        ):
            if result.errors:
                raise RuntimeError(f"Official GraphRAG failed: {result.workflow}; see {folder / 'reports'}")
            print(f"GraphRAG completed: {result.workflow}", flush=True)
    finally:
        reporter.stop()
    skill_count = await compile_skills(config, artifacts, args.skills, args.community_level)
    embedding_settings = load_query_settings(args.query_settings)["embeddings"]
    prepare_skill_index(BGEM3(embedding_settings), args.skills, args.embedding_index)
    summary = {"graphrag": version("graphrag"), "status": "complete", "persona_side": "user",
               "source": str(args.train.resolve()), "principles": str(PRINCIPLES), "language": "English",
               "samples": len(samples), "documents": len(documents),
               "official_workflows": [workflow.name for workflow in pipeline.workflows], "artifacts": str(artifacts),
               "community_level": args.community_level, "skills": skill_count,
               "embedding_service": {key: embedding_settings[key] for key in ("api_base", "model")},
               "skill_directory": str(args.skills.resolve()), "when_embeddings": str(args.embedding_index.resolve())}
    summary["parquet_tables"] = {
        path.name: {"path": str(path), "rows": len(pd.read_parquet(path))}
        for path in sorted(artifacts.glob("*.parquet"))
    }
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "index")
    parser.add_argument("--train", type=Path, default=TRAIN)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skills", type=Path, default=SKILLS)
    parser.add_argument("--embedding-index", type=Path, default=INDEX)
    parser.add_argument("--query-settings", type=Path, default=ROOT / "query_settings.yaml")
    parser.add_argument("--community-level", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Reuse native workflow outputs; requires identical inputs/settings")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.community_level < 0:
        parser.error("Invalid limit or community level")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
