#!/usr/bin/env python
"""Build a deduplicated YOLO dataset with a deterministic 7:2:1 split.

The source dataset is left untouched. Exact duplicate images are grouped by
SHA-256 content hash. For duplicate groups with conflicting annotations, the
valid label file containing the most rows is retained and the conflict is
recorded in the manifest/report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SOURCE_SPLIT_ORDER = ("train", "val", "test")
OUTPUT_SPLIT_ORDER = ("train", "test", "val")
SPLIT_RATIOS = {"train": 0.7, "test": 0.2, "val": 0.1}


@dataclass(frozen=True)
class ImageRecord:
    image: Path
    label: Path
    source_split: str
    digest: str
    label_rows: tuple[str, ...]


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment setup issue
        raise RuntimeError("PyYAML is required to read data.yaml.") from error

    with path.open("r", encoding="utf-8-sig") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("data.yaml must contain a mapping at its root.")
    return payload


def normalize_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        try:
            return [str(raw_names[key]) for key in sorted(raw_names, key=lambda item: int(item))]
        except (TypeError, ValueError) as error:
            raise ValueError("names mapping keys must be numeric class IDs.") from error
    raise ValueError("data.yaml must define names as a list or numeric mapping.")


def resolve_split(source_root: Path, config: dict[str, Any], split: str) -> Path:
    value = config.get(split)
    if not isinstance(value, str):
        raise ValueError(f"data.yaml must define a string path for '{split}'.")
    candidate = Path(value)
    root = source_root / candidate if not candidate.is_absolute() else candidate
    return root.resolve()


def image_label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / f"{image.stem}.txt"


def parse_label(label: Path, class_count: int) -> tuple[str, ...]:
    rows: list[str] = []
    for line_number, raw_line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label}:{line_number}: expected 5 YOLO fields")
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"{label}:{line_number}: non-numeric YOLO value") from error
        if not 0 <= class_id < class_count:
            raise ValueError(f"{label}:{line_number}: class {class_id} outside 0..{class_count - 1}")
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{label}:{line_number}: normalized coordinate outside 0..1")
        rows.append(" ".join(fields))
    return tuple(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_records(source_root: Path, config: dict[str, Any], class_count: int) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for split in SOURCE_SPLIT_ORDER:
        images_dir = resolve_split(source_root, config, split)
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
        for image in sorted(images_dir.iterdir(), key=lambda item: item.name.lower()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = image_label_path(image)
            if not label.is_file():
                raise FileNotFoundError(f"Missing label for image: {image}")
            rows = parse_label(label, class_count)
            records.append(
                ImageRecord(
                    image=image,
                    label=label,
                    source_split=split,
                    digest=sha256(image),
                    label_rows=rows,
                )
            )
    if not records:
        raise ValueError("No supported images were found in the source dataset.")
    return records


def split_counts(total: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    allocated = 0
    for split in OUTPUT_SPLIT_ORDER:
        raw = total * SPLIT_RATIOS[split]
        base = int(raw)
        counts[split] = base
        allocated += base
        remainders.append((raw - base, split))
    for _, split in sorted(remainders, key=lambda item: (-item[0], OUTPUT_SPLIT_ORDER.index(item[1])))[
        : total - allocated
    ]:
        counts[split] += 1
    return counts


def safe_output_name(record: ImageRecord, used_names: set[str]) -> str:
    name = record.image.name
    if name.lower() not in used_names:
        used_names.add(name.lower())
        return name
    suffix = record.digest[:12]
    candidate = f"{record.image.stem}_{suffix}{record.image.suffix.lower()}"
    used_names.add(candidate.lower())
    return candidate


def write_data_yaml(output_root: Path, names: list[str]) -> None:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - environment setup issue
        raise RuntimeError("PyYAML is required to write data.yaml.") from error

    payload = {
        "path": ".",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    output_root.joinpath("data.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/self-driving-cars-v6"))
    parser.add_argument("--output", type=Path, default=Path("datasets/self-driving-cars-v6_deduplicated"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output directory.")
    args = parser.parse_args()

    source_root = args.source.resolve()
    output_root = args.output.resolve()
    data_yaml = source_root / "data.yaml"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source_root}")
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Source data.yaml does not exist: {data_yaml}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}; use --overwrite explicitly.")
        shutil.rmtree(output_root)

    config = load_yaml(data_yaml)
    names = normalize_names(config.get("names"))
    records = collect_records(source_root, config, len(names))

    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.digest].append(record)

    unique_records: list[ImageRecord] = []
    conflict_groups = 0
    for digest in sorted(groups):
        group = groups[digest]
        if len({record.label_rows for record in group}) > 1:
            conflict_groups += 1
        selected = min(
            group,
            key=lambda record: (
                -len(record.label_rows),
                SOURCE_SPLIT_ORDER.index(record.source_split),
                record.image.name.lower(),
            ),
        )
        unique_records.append(selected)

    random.Random(args.seed).shuffle(unique_records)
    counts = split_counts(len(unique_records))
    assignments: dict[str, list[ImageRecord]] = {split: [] for split in OUTPUT_SPLIT_ORDER}
    offset = 0
    for split in OUTPUT_SPLIT_ORDER:
        assignments[split] = unique_records[offset : offset + counts[split]]
        offset += counts[split]

    for split in OUTPUT_SPLIT_ORDER:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    class_counts: dict[str, Counter[str]] = {split: Counter() for split in OUTPUT_SPLIT_ORDER}
    for split in OUTPUT_SPLIT_ORDER:
        used_names: set[str] = set()
        for record in assignments[split]:
            output_name = safe_output_name(record, used_names)
            output_image = output_root / split / "images" / output_name
            output_label = output_root / split / "labels" / f"{Path(output_name).stem}.txt"
            shutil.copy2(record.image, output_image)
            output_label.write_text("\n".join(record.label_rows) + ("\n" if record.label_rows else ""), encoding="utf-8")
            for row in record.label_rows:
                class_counts[split][row.split()[0]] += 1
            manifest_rows.append(
                {
                    "split": split,
                    "image": output_name,
                    "label": output_label.name,
                    "sha256": record.digest,
                    "source_split": record.source_split,
                    "source_image": record.image.name,
                    "duplicate_group_size": len(groups[record.digest]),
                    "label_row_count": len(record.label_rows),
                    "label_conflict": len({item.label_rows for item in groups[record.digest]}) > 1,
                }
            )

    write_data_yaml(output_root, names)
    with (output_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "split_order": "train:test:val",
        "class_count": len(names),
        "source_image_count": len(records),
        "unique_image_count": len(unique_records),
        "duplicate_groups": sum(len(group) > 1 for group in groups.values()),
        "duplicate_files_removed": len(records) - len(unique_records),
        "duplicate_groups_with_label_conflicts": conflict_groups,
        "label_selection_policy": "retain valid label file with the most rows; tie-break by train, val, test and filename",
        "split_counts": counts,
        "object_counts_by_class_id": {split: dict(sorted(counter.items())) for split, counter in class_counts.items()},
    }
    (output_root / "dedup_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
