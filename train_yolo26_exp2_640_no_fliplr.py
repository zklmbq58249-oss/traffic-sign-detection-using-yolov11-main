"""EXP-2: retrain EXP-1 with horizontal flipping disabled.

Only ``fliplr`` differs from EXP-1. This is an ablation for numeric speed
limit signs, where mirrored digits may introduce unrealistic training images.
The script trains at 640px and evaluates the resulting best.pt on the test
split without overwriting any previous run.
"""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

from train_yolo26 import (
    load_ultralytics,
    model_source,
    require_yolo26_support,
    select_device,
)


ROOT = Path(__file__).resolve().parent
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")


MODEL = "yolo26n.pt"
DATA = "configs/yolo26_data.yaml"
IMGSZ = 640
EPOCHS = 50
BATCH = 16
DEVICE = "0"
SEED = 42
WORKERS = 4
PROJECT = "runs/yolo26"
NAME = "yolo26n_traffic_signs_exp2_640_no_fliplr"
FLIPLR = 0.0


def main() -> int:
    YOLO, ultralytics_version = load_ultralytics()
    require_yolo26_support(ultralytics_version)

    data_path = (ROOT / DATA).resolve()
    project_path = (ROOT / PROJECT).resolve()
    device = select_device(DEVICE)
    checkpoint = model_source(MODEL)

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint}")

    print("Starting EXP-2 training")
    print(f"Model: {checkpoint}")
    print(f"Data: {data_path}")
    print(f"Image size: {IMGSZ}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch: {BATCH}")
    print(f"Device: {device}")
    print(f"Horizontal flip probability: {FLIPLR}")

    model = YOLO(checkpoint, task="detect")
    model.train(
        data=str(data_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        seed=SEED,
        deterministic=True,
        workers=WORKERS,
        fliplr=FLIPLR,
        project=str(project_path),
        name=NAME,
        exist_ok=False,
    )

    save_dir = Path(model.trainer.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Training ended without best.pt: {best_path}")

    print(f"Training output: {save_dir.resolve()}")
    print(f"Best checkpoint: {best_path.resolve()}")

    test_metrics = YOLO(str(best_path), task="detect").val(
        data=str(data_path),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        workers=WORKERS,
        end2end=True,
        project=str(project_path),
        name=f"{NAME}_test_e2e_true",
        exist_ok=False,
        plots=True,
    )

    print(f"Test output: {Path(test_metrics.save_dir).resolve()}")
    print(f"Test metrics: {test_metrics.results_dict}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
