"""Fine-tune the bundled traffic-sign detector on the local YOLO dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
# Keep Ultralytics state in the repository instead of the user profile.
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")

from ultralytics import YOLO  # noqa: E402


def repo_path(value: str) -> Path:
    """Resolve command-line paths relative to this repository."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def class_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        return [str(raw_names[key]) for key in sorted(raw_names, key=lambda key: int(key))]
    raise ValueError("data.yaml must define 'names' as a list or numeric mapping.")


def load_data_names(data_yaml: Path) -> list[str]:
    with data_yaml.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("data.yaml must contain a mapping at its root.")
    return class_names(config.get("names"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model/traffic_sign_detector.pt", help="Starting checkpoint path")
    parser.add_argument("--data", default="datasets/self-driving-cars-v6/data.yaml", help="YOLO data.yaml path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device, e.g. 0; use cpu for CPU training")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/my_training", help="Training output directory")
    parser.add_argument("--name", default="traffic_sign_custom", help="Experiment name")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing output directory")
    parser.add_argument("--test-after-train", action="store_true", help="Evaluate best.pt on the test split after training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = repo_path(args.model).resolve()
    data_path = repo_path(args.data).resolve()
    project_path = repo_path(args.project).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Starting checkpoint not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch == 0:
        raise ValueError("epochs and imgsz must be positive; batch cannot be 0.")

    model = YOLO(str(model_path), task="detect")
    dataset_names = load_data_names(data_path)
    model_names = [str(model.names[index]) for index in sorted(model.names)]
    if dataset_names != model_names:
        raise ValueError(
            "Dataset class names do not exactly match the starting checkpoint. "
            f"\nmodel:   {model_names}\ndataset: {dataset_names}\n"
            "Use a matching checkpoint or start from yolo11n.pt for a different class mapping."
        )

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        exist_ok=args.exist_ok,
    )

    save_dir = Path(model.trainer.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    print(f"Training output: {save_dir}")
    print(f"Best checkpoint: {best_path}")

    if args.test_after_train:
        if not best_path.is_file():
            raise FileNotFoundError(f"Training ended without best.pt: {best_path}")
        YOLO(str(best_path), task="detect").val(
            data=str(data_path),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(project_path),
            name=f"{args.name}_test",
            exist_ok=args.exist_ok,
            plots=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
