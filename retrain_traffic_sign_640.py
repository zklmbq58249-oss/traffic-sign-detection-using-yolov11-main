"""Fine-tune the traffic-sign detector with a fixed 640-pixel input size.

This script starts from the best ``traffic_sign_custom2`` checkpoint, writes a
separate run directory, and evaluates the resulting best checkpoint on both
the validation and test splits. A temporary dataset YAML with an absolute
``path`` is used because Ultralytics resolves relative dataset paths through
its global datasets directory on Windows.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 640
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")

from ultralytics import YOLO  # noqa: E402


def repo_path(value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def class_names(raw_names: Any) -> list[str]:
    """Normalize YOLO list/dictionary class-name formats."""
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        return [str(raw_names[key]) for key in sorted(raw_names, key=lambda key: int(key))]
    raise ValueError("data.yaml must define 'names' as a list or numeric mapping.")


def load_data_config(data_yaml: Path) -> dict[str, Any]:
    """Load and validate the dataset configuration."""
    with data_yaml.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("data.yaml must contain a mapping at its root.")
    class_names(config.get("names"))
    return config


@contextmanager
def temporary_runtime_yaml(data_yaml: Path) -> Iterator[str]:
    """Yield a temporary YAML whose dataset root is absolute."""
    config = load_data_config(data_yaml)
    config["path"] = str(data_yaml.parent.resolve())
    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        encoding="utf-8",
        delete=False,
    )
    try:
        yaml.safe_dump(config, temporary, allow_unicode=True, sort_keys=False)
        temporary.close()
        yield temporary.name
    finally:
        try:
            os.unlink(temporary.name)
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="runs/my_training/traffic_sign_custom2/weights/best.pt",
        help="Checkpoint to fine-tune.",
    )
    parser.add_argument("--data", default="datasets/self-driving-cars-v6/data.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device, or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--project", default="runs/my_training")
    parser.add_argument("--name", default="traffic_sign_custom2_finetune_640")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def print_metrics(split: str, metrics: Any, names: list[str]) -> None:
    """Print aggregate and Speed Limit 10 metrics for one split."""
    box = metrics.box
    print(
        f"{split}: P={box.mp:.4f} R={box.mr:.4f} "
        f"mAP50={box.map50:.4f} mAP50-95={box.map:.4f}"
    )
    speed_limit_10 = names.index("Speed Limit 10")
    print(
        "Speed Limit 10: "
        f"P={box.p[speed_limit_10]:.4f} "
        f"R={box.r[speed_limit_10]:.4f} "
        f"AP50={box.ap50[speed_limit_10]:.4f} "
        f"AP50-95={box.ap[speed_limit_10]:.4f}"
    )


def main() -> int:
    args = parse_args()
    model_path = repo_path(args.model).resolve()
    data_path = repo_path(args.data).resolve()
    project_path = repo_path(args.project).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    if args.epochs <= 0 or args.batch <= 0 or args.workers < 0 or args.lr0 <= 0:
        raise ValueError("epochs, batch and lr0 must be positive; workers cannot be negative.")

    model = YOLO(str(model_path), task="detect")
    dataset_names = class_names(load_data_config(data_path).get("names"))
    model_names = [str(model.names[index]) for index in range(len(model.names))]
    if dataset_names != model_names:
        raise ValueError(
            "Dataset class names do not exactly match the checkpoint."
            f"\nmodel:   {model_names}\ndataset: {dataset_names}"
        )

    with temporary_runtime_yaml(data_path) as runtime_data:
        model.train(
            data=runtime_data,
            epochs=args.epochs,
            imgsz=IMAGE_SIZE,
            batch=args.batch,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            workers=args.workers,
            lr0=args.lr0,
            project=str(project_path),
            name=args.name,
            exist_ok=args.exist_ok,
        )

        save_dir = Path(model.trainer.save_dir)
        best_path = save_dir / "weights" / "best.pt"
        if not best_path.is_file():
            raise FileNotFoundError(f"Training ended without best.pt: {best_path}")
        print(f"Training output: {save_dir}")
        print(f"Best checkpoint: {best_path}")

        best_model = YOLO(str(best_path), task="detect")
        for split in ("val", "test"):
            metrics = best_model.val(
                data=runtime_data,
                split=split,
                imgsz=IMAGE_SIZE,
                batch=args.batch,
                device=args.device,
                workers=0,
                project=str(project_path),
                name=f"{args.name}_{split}",
                exist_ok=args.exist_ok,
                plots=True,
            )
            print_metrics(split, metrics, model_names)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
