#!/usr/bin/env python
"""Validate the exported Roboflow YOLO dataset before baseline evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
EXPECTED_COUNTS = {"train": 3530, "val": 801, "test": 638}


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is required. Run scripts/setup_yolo11_baseline.ps1 first."
        ) from error

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("data.yaml must contain a mapping at its root.")
    return payload


def resolve_dataset_root(data_yaml: Path, config: dict[str, Any]) -> Path:
    raw_path = config.get("path", ".")
    root = Path(raw_path)
    return (data_yaml.parent / root).resolve() if not root.is_absolute() else root.resolve()


def resolve_split(root: Path, value: Any, split: str) -> Path:
    if isinstance(value, list):
        raise ValueError(f"{split} must point to one image directory; lists are not supported by this validator.")
    if not isinstance(value, str):
        raise ValueError(f"data.yaml is missing a string path for the '{split}' split.")
    candidate = Path(value)
    return (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = parts.index("images")
    except ValueError as error:
        raise ValueError(f"Cannot infer label path because 'images' is absent from {image_path}") from error
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def normalized_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        try:
            return [str(raw_names[key]) for key in sorted(raw_names, key=lambda item: int(item))]
        except (TypeError, ValueError) as error:
            raise ValueError("names mapping keys must be numeric class IDs.") from error
    raise ValueError("data.yaml must define class names as a list or numeric mapping.")


def validate_split(images_dir: Path, class_count: int) -> dict[str, Any]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")

    images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    missing_labels: list[str] = []
    invalid_labels: list[str] = []
    labels_seen = 0

    for image in images:
        label = label_path(image)
        if not label.is_file():
            missing_labels.append(str(image))
            continue
        labels_seen += 1
        for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.split()
            if len(fields) != 5:
                invalid_labels.append(f"{label}:{line_number}: expected 5 YOLO fields")
                continue
            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                invalid_labels.append(f"{label}:{line_number}: non-numeric value")
                continue
            if not 0 <= class_id < class_count:
                invalid_labels.append(f"{label}:{line_number}: class {class_id} outside 0..{class_count - 1}")
            elif any(value < 0.0 or value > 1.0 for value in coordinates):
                invalid_labels.append(f"{label}:{line_number}: normalized coordinates outside 0..1")

    return {
        "images_dir": str(images_dir),
        "image_count": len(images),
        "label_count": labels_seen,
        "missing_label_count": len(missing_labels),
        "missing_label_examples": missing_labels[:10],
        "invalid_label_count": len(invalid_labels),
        "invalid_label_examples": invalid_labels[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Path to exported Roboflow data.yaml")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--skip-count-check", action="store_true", help="Do not require the expected 3530/801/638 split sizes")
    args = parser.parse_args()

    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"data.yaml does not exist: {data_yaml}")

    config = load_yaml(data_yaml)
    names = normalized_names(config.get("names"))
    root = resolve_dataset_root(data_yaml, config)
    report: dict[str, Any] = {
        "data_yaml": str(data_yaml),
        "dataset_root": str(root),
        "class_count": len(names),
        "class_names": names,
        "splits": {},
        "errors": [],
    }

    if len(names) != 15:
        report["errors"].append(f"Expected 15 classes, found {len(names)}.")

    for split in ("train", "val", "test"):
        try:
            images_dir = resolve_split(root, config.get(split), split)
            split_report = validate_split(images_dir, len(names))
            if not args.skip_count_check and split_report["image_count"] != EXPECTED_COUNTS[split]:
                report["errors"].append(
                    f"{split}: expected {EXPECTED_COUNTS[split]} images, found {split_report['image_count']}."
                )
            if split_report["missing_label_count"]:
                report["errors"].append(f"{split}: {split_report['missing_label_count']} images have no label file.")
            if split_report["invalid_label_count"]:
                report["errors"].append(f"{split}: {split_report['invalid_label_count']} invalid label rows.")
            report["splits"][split] = split_report
        except (FileNotFoundError, ValueError) as error:
            report["errors"].append(f"{split}: {error}")

    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")

    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
