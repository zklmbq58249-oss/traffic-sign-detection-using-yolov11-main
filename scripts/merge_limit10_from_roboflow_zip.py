#!/usr/bin/env python
"""Merge Speed Limit 10 images from a Roboflow YOLO zip into a dataset.

Only images containing the source Speed Limit 10 class are considered. Exact
duplicate image bytes are skipped across every destination split and within
the source archive. All annotations on a selected image are retained and
mapped by class name so that no other annotated object becomes unlabeled.
YOLO segmentation rows are converted to their normalized bounding boxes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def canonical_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def normalized_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        return [str(raw_names[key]) for key in sorted(raw_names, key=lambda item: int(item))]
    raise ValueError("data.yaml must define names as a list or numeric mapping.")


def load_config(path: Path) -> tuple[dict[str, Any], list[str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: data.yaml must contain a mapping.")
    return payload, normalized_names(payload.get("names"))


def resolve_root(data_yaml: Path, config: dict[str, Any]) -> Path:
    raw_root = config.get("path", ".")
    root = Path(raw_root)
    return (data_yaml.parent / root).resolve() if not root.is_absolute() else root.resolve()


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_label(text: str, class_count: int, source_label: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5 and (len(fields) < 7 or (len(fields) - 1) % 2 != 0):
            raise ValueError(f"{source_label}:{line_number}: expected a YOLO box or polygon row")
        try:
            class_id = int(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as error:
            raise ValueError(f"{source_label}:{line_number}: non-numeric YOLO value") from error
        if not 0 <= class_id < class_count:
            raise ValueError(f"{source_label}:{line_number}: class {class_id} outside 0..{class_count - 1}")
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{source_label}:{line_number}: normalized coordinate outside 0..1")
        if len(fields) == 5:
            rows.append(fields)
            continue
        x_values = coordinates[0::2]
        y_values = coordinates[1::2]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        rows.append(
            [
                fields[0],
                f"{(x_min + x_max) / 2:.12g}",
                f"{(y_min + y_max) / 2:.12g}",
                f"{x_max - x_min:.12g}",
                f"{y_max - y_min:.12g}",
            ]
        )
    return rows


def destination_image_hashes(root: Path) -> tuple[dict[str, Path], set[str]]:
    by_digest: dict[str, Path] = {}
    used_names: set[str] = set()
    for split in ("train", "val", "test"):
        images_dir = root / split / "images"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Destination image directory does not exist: {images_dir}")
        for image in images_dir.iterdir():
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            digest = image_hash(image.read_bytes())
            by_digest.setdefault(digest, image)
            used_names.add(image.name.lower())
    return by_digest, used_names


def label_member_for(image_member: str) -> str:
    path = PurePosixPath(image_member)
    parts = list(path.parts)
    try:
        index = parts.index("images")
    except ValueError as error:
        raise ValueError(f"Source image is not under an images directory: {image_member}") from error
    parts[index] = "labels"
    return str(PurePosixPath(*parts).with_suffix(".txt"))


def output_name(source_name: str, digest: str, used_names: set[str]) -> str:
    path = PurePosixPath(source_name)
    candidate = path.name
    if candidate.lower() in used_names:
        candidate = f"{path.stem}_{digest[:12]}{path.suffix.lower()}"
    if candidate.lower() in used_names:
        raise ValueError(f"Could not create a unique destination filename for {source_name}")
    used_names.add(candidate.lower())
    return candidate


def append_manifest_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "split",
        "image",
        "label",
        "sha256",
        "source_split",
        "source_image",
        "duplicate_group_size",
        "label_row_count",
        "label_conflict",
    ]
    existing = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not existing:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_zip", type=Path)
    parser.add_argument("target_data_yaml", type=Path)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--report", type=Path, help="Write a JSON merge report to this path.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without writing images or labels.")
    args = parser.parse_args()

    source_zip = args.source_zip.resolve()
    target_yaml = args.target_data_yaml.resolve()
    if not source_zip.is_file():
        raise FileNotFoundError(f"Source zip does not exist: {source_zip}")
    if not target_yaml.is_file():
        raise FileNotFoundError(f"Target data.yaml does not exist: {target_yaml}")

    target_config, target_names = load_config(target_yaml)
    target_root = resolve_root(target_yaml, target_config)
    target_by_name = {canonical_name(name): index for index, name in enumerate(target_names)}
    source_limit_key = canonical_name("Speed Limit 10")
    if source_limit_key not in target_by_name:
        raise ValueError("Target data.yaml has no Speed Limit 10 class.")
    target_limit_id = target_by_name[source_limit_key]

    destination_hashes, used_names = destination_image_hashes(target_root)
    source_candidates: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    duplicate_existing = 0
    duplicate_source = 0
    source_class_names: list[str] | None = None
    mapped_counts: Counter[str] = Counter()

    with zipfile.ZipFile(source_zip) as archive:
        try:
            source_config = yaml.safe_load(archive.read("data.yaml").decode("utf-8-sig"))
        except KeyError as error:
            raise ValueError("Source zip is missing data.yaml") from error
        source_names = normalized_names(source_config.get("names") if isinstance(source_config, dict) else None)
        source_class_names = source_names
        source_by_id = {index: canonical_name(name) for index, name in enumerate(source_names)}
        source_limit_ids = {index for index, name in source_by_id.items() if name == source_limit_key}
        if not source_limit_ids:
            raise ValueError("Source zip has no Speed Limit 10 class (accepted forms include Speed Limit -10-).")
        source_to_target: dict[int, int] = {}
        for source_id, source_key in source_by_id.items():
            if source_key not in target_by_name:
                raise ValueError(f"Source class '{source_names[source_id]}' has no target class mapping.")
            source_to_target[source_id] = target_by_name[source_key]

        members = sorted(
            name
            for name in archive.namelist()
            if "/images/" in name and PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES
        )
        for image_member in members:
            label_member = label_member_for(image_member)
            try:
                raw_label = archive.read(label_member).decode("utf-8-sig")
            except KeyError as error:
                raise ValueError(f"Source image has no matching label: {image_member}") from error
            rows = parse_label(raw_label, len(source_names), label_member)
            if not any(int(row[0]) in source_limit_ids for row in rows):
                continue
            image_bytes = archive.read(image_member)
            digest = image_hash(image_bytes)
            if digest in destination_hashes:
                duplicate_existing += 1
                continue
            if digest in source_hashes:
                duplicate_source += 1
                continue
            source_hashes.add(digest)
            mapped_rows: list[str] = []
            for row in rows:
                target_id = source_to_target[int(row[0])]
                mapped_rows.append(" ".join([str(target_id), *row[1:]]))
                mapped_counts[str(target_id)] += 1
            source_candidates.append(
                {
                    "member": image_member,
                    "bytes": image_bytes,
                    "digest": digest,
                    "rows": mapped_rows,
                }
            )

    added_rows: list[dict[str, Any]] = []
    destination_images = target_root / args.split / "images"
    destination_labels = target_root / args.split / "labels"
    if not args.dry_run:
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)
        for item in source_candidates:
            name = output_name(item["member"], item["digest"], used_names)
            image_path = destination_images / name
            label_path = destination_labels / f"{Path(name).stem}.txt"
            image_path.write_bytes(item["bytes"])
            label_path.write_text("\n".join(item["rows"]) + "\n", encoding="utf-8")
            added_rows.append(
                {
                    "split": args.split,
                    "image": name,
                    "label": label_path.name,
                    "sha256": item["digest"],
                    "source_split": "roboflow_crv",
                    "source_image": PurePosixPath(item["member"]).name,
                    "duplicate_group_size": 1,
                    "label_row_count": len(item["rows"]),
                    "label_conflict": False,
                }
            )
        append_manifest_rows(target_root / "split_manifest.csv", added_rows)

    report = {
        "source_zip": str(source_zip),
        "target_data_yaml": str(target_yaml),
        "target_split": args.split,
        "source_classes": source_class_names,
        "target_speed_limit_10_class_id": target_limit_id,
        "source_speed_limit_10_images_considered": len(source_candidates) + duplicate_existing + duplicate_source,
        "duplicate_images_skipped_against_destination": duplicate_existing,
        "duplicate_images_skipped_within_source": duplicate_source,
        "images_added": len(source_candidates),
        "mapped_object_counts_by_target_class_id": dict(sorted(mapped_counts.items(), key=lambda item: int(item[0]))),
        "dry_run": args.dry_run,
    }
    report_path = args.report.resolve() if args.report else None
    if report_path and not args.dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
