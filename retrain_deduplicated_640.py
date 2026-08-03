"""Train YOLO11 on the deduplicated traffic-sign dataset at 640 pixels.

The script starts from the project's 15-class traffic-sign checkpoint, uses the
deduplicated 7:2:1 dataset by default, and evaluates the resulting best
checkpoint on both validation and test splits.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


ROOT = Path(__file__).resolve().parent
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")

from ultralytics import YOLO  # noqa: E402


IMAGE_SIZE = 640
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 16
DEFAULT_SEED = 42
# Windows multiprocessing workers can exit unexpectedly during Ultralytics
# training. Use the stable single-process loader by default on Windows; users
# can override this with --workers after a successful run.
DEFAULT_WORKERS = 0 if os.name == "nt" else 4


def repo_path(value: str) -> Path:
    """Resolve a command-line path relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalize_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list):
        return [str(name) for name in raw_names]
    if isinstance(raw_names, dict):
        return [str(raw_names[key]) for key in sorted(raw_names, key=lambda key: int(key))]
    raise ValueError("data.yaml must define 'names' as a list or numeric mapping.")


def load_data_config(data_yaml: Path) -> dict[str, Any]:
    with data_yaml.open("r", encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("data.yaml must contain a mapping at its root.")
    config["names"] = normalize_names(config.get("names"))
    config["nc"] = len(config["names"])
    return config


@contextmanager
def temporary_runtime_yaml(data_yaml: Path) -> Iterator[str]:
    """Give Ultralytics an absolute dataset root on Windows."""
    config = load_data_config(data_yaml)
    config["path"] = str(data_yaml.parent.resolve())
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8", delete=False) as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        runtime_path = handle.name
    try:
        yield runtime_path
    finally:
        Path(runtime_path).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="model/traffic_sign_detector.pt", help="Starting 15-class checkpoint")
    parser.add_argument(
        "--data",
        default="datasets/self-driving-cars-v6_deduplicated/data.yaml",
        help="Deduplicated YOLO data.yaml path",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device", default="0", help="CUDA device, or cpu")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="DataLoader workers (0 is the Windows-safe default)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--project", default="runs/my_training")
    parser.add_argument("--name", default="traffic_sign_dedup_640")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reuse of an existing output directory")
    return parser.parse_args()


def validate_inputs(model_path: Path, data_path: Path) -> list[str]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Starting checkpoint not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    config = load_data_config(data_path)
    if len(config["names"]) != 15:
        raise ValueError(f"Expected 15 classes for the custom checkpoint, found {len(config['names'])}.")
    return config["names"]


def load_checkpoint_with_fallback(
    best_path: Path,
    last_path: Path,
    training_started_at: float,
) -> tuple[YOLO, Path]:
    """Load the newest usable checkpoint, tolerating Windows file locks."""
    candidates = [best_path, last_path]
    if best_path.is_file() and last_path.is_file() and best_path.stat().st_mtime < training_started_at:
        # A reused output directory can contain a stale best.pt from an earlier
        # interrupted run. Prefer the checkpoint written by this run.
        candidates = [last_path, best_path]

    errors: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for attempt in range(3):
            try:
                with candidate.open("rb") as handle:
                    handle.read(1)
                return YOLO(str(candidate), task="detect"), candidate
            except (OSError, RuntimeError) as error:
                errors.append(f"{candidate} (attempt {attempt + 1}): {error}")
                time.sleep(1)
    details = "\n".join(errors) if errors else "no checkpoint file was found"
    raise RuntimeError(f"Unable to load a usable training checkpoint:\n{details}")


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch == 0 or args.workers < 0 or args.lr0 <= 0:
        raise ValueError("epochs must be positive, batch cannot be 0, workers cannot be negative, and lr0 must be positive.")

    model_path = repo_path(args.model).resolve()
    data_path = repo_path(args.data).resolve()
    project_path = repo_path(args.project).resolve()
    dataset_names = validate_inputs(model_path, data_path)

    model = YOLO(str(model_path), task="detect")
    model_names = [str(model.names[index]) for index in sorted(model.names)]
    if dataset_names != model_names:
        raise ValueError(
            "Dataset class names do not match the starting checkpoint."
            f"\nmodel:   {model_names}\ndata:    {dataset_names}"
        )

    training_started_at = time.time()
    with temporary_runtime_yaml(data_path) as runtime_data:
        model.train(
            data=runtime_data,
            epochs=args.epochs,
            imgsz=IMAGE_SIZE,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            seed=args.seed,
            deterministic=True,
            lr0=args.lr0,
            project=str(project_path),
            name=args.name,
            exist_ok=args.exist_ok,
            plots=True,
        )

        save_dir = Path(model.trainer.save_dir)
        best_path = save_dir / "weights" / "best.pt"
        last_path = save_dir / "weights" / "last.pt"
        best_model, evaluation_checkpoint = load_checkpoint_with_fallback(
            best_path,
            last_path,
            training_started_at,
        )
        for split in ("val", "test"):
            best_model.val(
                data=runtime_data,
                split=split,
                imgsz=IMAGE_SIZE,
                batch=args.batch,
                device=args.device,
                project=str(project_path),
                name=f"{args.name}_{split}",
                exist_ok=args.exist_ok,
                plots=True,
            )

    print(f"Training output: {save_dir}")
    print(f"Evaluation checkpoint: {evaluation_checkpoint}")
    print(f"Dataset: {data_path}")
    print(f"Image size: {IMAGE_SIZE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
