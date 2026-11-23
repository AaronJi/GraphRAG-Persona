"""Rank skill When clauses against inferred profiles and emit JSONL to stdout."""

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter_ns

import numpy as np
import yaml

if __package__:
    from .query_settings import load_query_settings
else:
    from query_settings import load_query_settings


ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skill/graphrag"
INDEX = ROOT / "data/graphrag_skill_when_embeddings.npz"
PROFILE_FIELDS = {
    "explicit": {"age_group": "Age group", "gender_presentation": "Gender presentation", "role": "Role"},
    "implicit": {"personality": "Personality", "values": "Values", "communication_preferences": "Communication preferences"},
}


@dataclass(frozen=True)
class Skill:
    name: str
    source_rule: str
    path: Path
    when: str
    act: str
    metadata: dict = field(default_factory=dict)


def load_skills(directory):
    skills = []
    names = set()
    catalog = directory / "catalog.json"
    if catalog.is_file():
        listed = json.loads(catalog.read_text(encoding="utf-8-sig"))["skills"]
        if (
            not isinstance(listed, list) or any(
                not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)
                for name in listed
            ) or len(listed) != len(set(listed))
        ):
            raise ValueError(f"Invalid skill catalog {catalog}")
        paths = [directory / name / "SKILL.md" for name in listed]
    else:
        paths = sorted(directory.glob("*/SKILL.md"))
    for path in paths:
        text = path.read_text(encoding="utf-8-sig")
        document = re.fullmatch(r"---\n(.*?)\n---\n(.*)", text, flags=re.S)
        if not document:
            raise ValueError(f"Missing YAML frontmatter in {path}")
        metadata = yaml.safe_load(document[1])
        sections = dict(re.findall(
            r"^## (When|Act|How)[ \t]*\n(.*?)(?=^#{1,2} |\Z)", document[2], flags=re.M | re.S,
        ))
        if "How" in sections:
            if "Act" in sections:
                raise ValueError(f"Expected one behavioral guidance section in {path}")
            sections["Act"] = sections.pop("How")
        if set(sections) != {"When", "Act"} or not all(s.strip() for s in sections.values()):
            raise ValueError(f"Expected nonempty When and Act sections in {path}")
        name = metadata["name"]
        if name != path.parent.name or name in names:
            raise ValueError(f"Invalid or duplicate skill name in {path}")
        names.add(name)
        sources = metadata.get("metadata", {})
        source_rule = sources.get("source-rule", "")
        if isinstance(source_rule, list):
            source_rule = ",".join(str(rule) for rule in source_rule)
        skills.append(Skill(name, str(source_rule), path,
                            sections["When"].strip(), sections["Act"].strip(), sources))
    if not skills:
        raise ValueError(f"No skills found in {directory}")
    return skills


def profile_query(profile):
    """Use only populated fields from the validated profile object."""
    parts = []
    for section, fields in PROFILE_FIELDS.items():
        group = profile[section]
        for field, label in fields.items():
            value = group.get(field)
            if isinstance(value, str) and value.strip():
                parts.append(f"{label}: {value.strip()}")
            elif isinstance(value, list):
                values = [item.strip() for item in value if isinstance(item, str) and item.strip()]
                if values:
                    parts.append(f"{label}: {', '.join(values)}")
    return ". ".join(parts)


class BGEM3:
    def __init__(self, settings):
        from openai import OpenAI

        self.settings = dict(settings)
        self.dimension = None
        self.client = OpenAI(
            base_url=settings["api_base"], api_key=settings["api_key"],
            timeout=settings["request_timeout"], max_retries=0,
        )

    def encode(self, texts, batch_size=8):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding input must contain nonempty strings")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Embedding batch size must be a positive integer")
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            response = self.client.embeddings.create(
                model=self.settings["model"], input=batch, encoding_format="float",
            )
            data = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in data] != list(range(len(batch))):
                raise ValueError("Embedding response indices do not match the input batch")
            matrix = np.asarray([item.embedding for item in data], dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(batch) or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
                raise ValueError("The embedding service returned an invalid matrix")
            if self.dimension is not None and matrix.shape[1] != self.dimension:
                raise ValueError("Embedding dimension changed; rebuild the skill index with the current service")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("The embedding service returned a zero or invalid vector")
            self.dimension = matrix.shape[1]
            vectors.append(matrix / norms)
        return np.concatenate(vectors)


def rank_skills(skills, similarities, k, epsilon):
    scores = np.clip(similarities, -1.0, 1.0) * 100.0
    order = sorted(range(len(skills)), key=lambda i: (-float(scores[i]), skills[i].name))
    ranked = []
    selected = []
    for rank, i in enumerate(order, 1):
        skill = skills[i]
        item = {"rank": rank, "name": skill.name, "source_rule": skill.source_rule,
                "path": str(skill.path), "score": float(scores[i]), "metadata": skill.metadata}
        ranked.append(item)
        if item["score"] > epsilon and len(selected) < k:
            selected.append({**item, "when": skill.when, "act": skill.act})
    return ranked, selected


def index_signature(skills, settings):
    # Include only the skill data used for embedding. Act edits take effect online.
    payload = ["openai-embeddings-float-l2-v1", settings["api_base"].rstrip("/"), settings["model"],
               [(skill.name, skill.when) for skill in skills]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def save_index(path, signature, embeddings):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez(stream, signature=signature, embeddings=embeddings)
        except BaseException:
            stream.close()
            temporary.unlink()
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_skill_index(retriever, skill_dir, index_path):
    """Reuse the current When matrix, or rebuild it after corpus/model changes."""
    skills = load_skills(skill_dir.resolve())
    signature = index_signature(skills, retriever.settings)
    if index_path.is_file():
        try:
            embeddings = load_skill_index(index_path, signature, len(skills))
            if retriever.dimension is not None and embeddings.shape[1] != retriever.dimension:
                raise ValueError("Cached embedding dimension differs from the current service")
        except (OSError, ValueError, KeyError):
            pass
        else:
            retriever.dimension = embeddings.shape[1]
            return skills, embeddings
    embeddings = retriever.encode([skill.when for skill in skills])
    if embeddings.ndim != 2 or embeddings.shape[0] != len(skills) or embeddings.shape[1] < 1 or not np.isfinite(embeddings).all():
        raise ValueError("Unexpected BGE-M3 embedding matrix")
    save_index(index_path, signature, embeddings)
    return skills, embeddings


def load_skill_index(path, signature, count):
    with np.load(path, allow_pickle=False) as index:
        if index["signature"].item() != signature:
            raise ValueError("Skill index is stale; rebuild it with --build-index")
        embeddings = index["embeddings"]
    if embeddings.ndim != 2 or embeddings.shape[0] != count or embeddings.shape[1] < 1 or not np.isfinite(embeddings).all():
        raise ValueError("Unexpected BGE-M3 embedding matrix; rebuild it with --build-index")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Cached embeddings must be unit vectors; rebuild with --build-index")
    return embeddings


def retrieve_profile(retriever, skills, embeddings, profile, k, epsilon):
    if not isinstance(profile, dict) or set(profile) != set(PROFILE_FIELDS) or any(
        not isinstance(profile[section], dict) or set(profile[section]) != set(fields)
        for section, fields in PROFILE_FIELDS.items()
    ):
        raise ValueError("Invalid profile structure")
    query = profile_query(profile)
    if not query and all(value is None for group in profile.values() for value in group.values()):
        return {
            "query": "", "k": k, "epsilon": epsilon, "ranked": [], "selected": [],
            "timing": {"query_embedding_ms": 0.0, "dot_product_and_ranking_ms": 0.0},
        }
    if not query:
        raise ValueError("Profile has no populated fields")

    started = perf_counter_ns()
    vector = retriever.encode([query])[0]
    encoded = perf_counter_ns()
    if embeddings.ndim != 2 or embeddings.shape != (len(skills), len(vector)):
        raise ValueError("Query and skill embedding dimensions differ; rebuild the skill index")
    scores = embeddings @ vector
    ranked, selected = rank_skills(skills, scores, k, epsilon)
    finished = perf_counter_ns()
    return {
        "query": query,
        "k": k,
        "epsilon": epsilon,
        "ranked": ranked,
        "selected": selected,
        "timing": {
            "query_embedding_ms": (encoded - started) / 1e6,
            "dot_product_and_ranking_ms": (finished - encoded) / 1e6,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=ROOT / "input/elderlytech_vqa_profiles.jsonl")
    parser.add_argument("--skills", type=Path, default=SKILLS)
    parser.add_argument("--settings", type=Path, default=ROOT / "query_settings.yaml")
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--build-index", action="store_true", help="Encode all When clauses offline and exit.")
    parser.add_argument("--sample-id", action="append", help="Repeat to select multiple samples.")
    args = parser.parse_args()
    try:
        settings = load_query_settings(args.settings)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if args.build_index:
        retriever = BGEM3(settings["embeddings"])
        skills, _ = prepare_skill_index(retriever, args.skills, args.index)
        print(f"Indexed {len(skills)} skills in {args.index}")
        return
    skills = load_skills(args.skills.resolve())
    signature = index_signature(skills, settings["embeddings"])
    if not args.index.is_file():
        parser.error(f"missing skill index {args.index}; run with --build-index first")
    try:
        when_embeddings = load_skill_index(args.index, signature, len(skills))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    k, epsilon = settings["k"], settings["epsilon"]

    profiles = [json.loads(line) for line in args.profiles.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()]
    if args.sample_id:
        requested = set(args.sample_id)
        profiles = [row for row in profiles if row["sample_id"] in requested]
        missing = requested - {row["sample_id"] for row in profiles}
        if missing:
            parser.error(f"unknown sample IDs: {', '.join(sorted(missing))}")
    if not profiles:
        parser.error("no profiles found")
    for row in profiles:
        if not isinstance(row.get("profile"), dict) or not profile_query(row["profile"]):
            parser.error(f"sample {row['sample_id']} has no populated profile fields")

    retriever = BGEM3(settings["embeddings"])
    retriever.dimension = when_embeddings.shape[1]
    for row in profiles:
        result = retrieve_profile(retriever, skills, when_embeddings, row["profile"], k, epsilon)
        print(json.dumps({"sample_id": row["sample_id"], **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
