"""Train a YOLO26 detector from the command line.

The script is intentionally self-contained so it can be launched from any
working directory, for example::

    python train_yolo26.py

All paths that point to files in this repository are resolved relative to this
script, not relative to the caller's current directory.  Use ``--help`` to
override the defaults for a different checkpoint, dataset, or run settings.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
# Keep Ultralytics settings inside this repository rather than AppData.
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")


def load_ultralytics():
    """Load Ultralytics only after CLI parsing so ``--help`` needs no install."""
    try:
        from ultralytics import YOLO, __version__ as ultralytics_version
    except ModuleNotFoundError as error:
        if error.name == "ultralytics":
            raise RuntimeError(
                "未找到 Ultralytics。请先安装依赖："
                "python -m pip install -r requirements.txt"
            ) from error
        raise
    return YOLO, ultralytics_version


def require_yolo26_support(ultralytics_version: str) -> None:
    """Fail early when an old Ultralytics installation is used."""
    version_parts = tuple(
        int(part) if part.isdigit() else 0
        for part in ultralytics_version.split(".")[:3]
    )
    if version_parts < (8, 4, 0):
        raise RuntimeError(
            "YOLO26 requires Ultralytics 8.4.0 or newer. "
            f"This interpreter has {ultralytics_version}. "
            "Install it with: python -m pip install -U \"ultralytics>=8.4.0\""
        )


def repo_path(value: str) -> Path:
    """Resolve a repository-relative path from the script location."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def model_source(value: str) -> str:
    """Prefer a local repository checkpoint, while preserving remote model names."""
    path = Path(value).expanduser()
    if path.is_absolute() and path.is_file():
        return str(path.resolve())
    if not path.is_absolute() and path.is_file():
        return str(path.resolve())

    repo_checkpoint = repo_path(value)
    return str(repo_checkpoint.resolve()) if repo_checkpoint.is_file() else value


def select_device(value: str) -> str:
    """Resolve ``auto`` to the first CUDA device when one is available."""
    if value.lower() != "auto":
        return value

    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="YOLO26 checkpoint/name (default: local yolo26n.pt)",
    )
    parser.add_argument(
        "--data",
        default="configs/yolo26_data.yaml",
        help="Dataset YAML, relative to this script by default",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=416, help="Training image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size; use -1 for auto batch")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device such as 0, 0,1, cpu, or auto (default: auto)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--project", default="runs/yolo26", help="Output directory")
    parser.add_argument("--name", default="yolo26n_traffic_signs", help="Run name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow reuse of an existing output directory",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="Resume from a last.pt checkpoint instead of starting fresh",
    )
    parser.add_argument(
        "--test-after-train",
        action="store_true",
        help="Evaluate best.pt on the test split after training",
    )
    parser.add_argument(
        "--end2end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use YOLO26's native NMS-free head for test evaluation (default: true).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    YOLO, ultralytics_version = load_ultralytics()
    require_yolo26_support(ultralytics_version)
    data_path = repo_path(args.data).resolve()
    project_path = repo_path(args.project).resolve()
    device = select_device(args.device)

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.workers < 0:
        raise ValueError("epochs and imgsz must be positive; workers cannot be negative.")
    if args.batch != -1 and args.batch <= 0:
        raise ValueError("batch must be -1 (auto) or a positive integer.")

    checkpoint = model_source(args.resume or args.model)
    if args.resume and not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")

    model = YOLO(checkpoint, task="detect")
    train_kwargs = dict(
        data=str(data_path),  # Absolute path avoids Windows datasets-dir ambiguity.
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        seed=args.seed,
        deterministic=True,
        workers=args.workers,
        project=str(project_path),
        name=args.name,
        exist_ok=args.exist_ok,
    )
    if args.resume:
        train_kwargs["resume"] = True
    model.train(**train_kwargs)

    save_dir = Path(model.trainer.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    print(f"Training output: {save_dir.resolve()}")
    print(f"Best checkpoint: {best_path.resolve()}")

    if args.test_after_train:
        if not best_path.is_file():
            raise FileNotFoundError(f"Training ended without best.pt: {best_path}")
        YOLO(str(best_path), task="detect").val(
            data=str(data_path),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            end2end=args.end2end,
            project=str(project_path),
            name=f"{args.name}_test_e2e_{str(args.end2end).lower()}",
            exist_ok=args.exist_ok,
            plots=True,
        )

    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
