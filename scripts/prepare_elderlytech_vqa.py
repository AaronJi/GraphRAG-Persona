"""Extract the original ElderlyTech-VQA images and write an image-only manifest."""

import argparse
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"D:\Data\ElderlyTech-VQA")
PARQUET_NAME = "train-00000-of-00001.parquet"
REFERENCE_COLUMNS = (
    "user_profile",
    "question",
    "answer_gpt4o",
    "answer_gpt4o_1paragraph",
    "answer_gpt4o_1sentence",
)


def write_converted_dataset(source: Path, output: Path) -> int:
    table = pq.ParquetFile(source).read(columns=["image", *REFERENCE_COLUMNS])
    missing = set(("image", *REFERENCE_COLUMNS)) - set(table.column_names)
    if missing or not table.num_rows:
        raise ValueError(f"Invalid source Parquet: {table.num_rows} rows, missing {sorted(missing)}")

    (output / "images").mkdir(parents=True)
    with (
        (output / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as manifest,
        (output / "references.jsonl").open("w", encoding="utf-8", newline="\n") as references,
    ):
        for row_index, row in enumerate(table.to_pylist()):
            sample_id = f"elderlytech_vqa_{row_index:03d}"
            image = row["image"]
            raw_bytes = image["bytes"] if image else None
            if not raw_bytes:
                raise ValueError(f"Missing image bytes for {sample_id}")
            with Image.open(BytesIO(raw_bytes)) as photo:
                if photo.format != "JPEG":
                    raise ValueError(f"Expected JPEG for {sample_id}, got {photo.format}")
                photo.load()

            image_path = Path("images") / f"{sample_id}.jpg"
            (output / image_path).write_bytes(raw_bytes)
            manifest.write(json.dumps({
                "sample_id": sample_id,
                "row_index": row_index,
                "image_path": image_path.as_posix(),
            }, ensure_ascii=False) + "\n")
            references.write(json.dumps({
                "sample_id": sample_id,
                "row_index": row_index,
                "source_image_path": image["path"],
                **{key: row[key] for key in REFERENCE_COLUMNS},
            }, ensure_ascii=False) + "\n")
    return table.num_rows


def prepare(source: Path, output: Path) -> int:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".elderlytech-vqa-", dir=output.parent) as staging_dir:
        staging_root = Path(staging_dir).resolve()
        if staging_root.parent != output.parent:
            raise ValueError(f"Unexpected staging directory: {staging_root}")
        staging = staging_root / "converted"
        count = write_converted_dataset(source, staging)
        if output.exists():
            for generated in staging.rglob("*"):
                if generated.is_file():
                    existing = output / generated.relative_to(staging)
                    if not existing.is_file() or existing.read_bytes() != generated.read_bytes():
                        raise FileExistsError(f"Existing output differs at {existing}. Use a new --output-dir.")
        else:
            staging.rename(output)
    return count


def main() -> None:
    source = DATASET_ROOT / PARQUET_NAME
    if not source.is_file():
        source = DATASET_ROOT / "data" / PARQUET_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=source, help="Source ElderlyTech-VQA Parquet")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "elderlytech_vqa",
        help="Directory for native JPEGs, manifest.jsonl and references.jsonl",
    )
    args = parser.parse_args()
    count = prepare(args.source, args.output_dir)
    print(f"Prepared {count} samples in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
