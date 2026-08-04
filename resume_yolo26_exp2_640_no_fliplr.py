"""Resume EXP-2 from its latest last.pt and run the final test evaluation."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")

LAST = ROOT / "runs/yolo26/yolo26n_traffic_signs_exp2_640_no_fliplr/weights/last.pt"
DATA = ROOT / "configs/yolo26_data.yaml"
PROJECT = ROOT / "runs/yolo26"
TEST_NAME = "yolo26n_traffic_signs_exp2_640_no_fliplr_test_e2e_true"


def main() -> int:
    if not LAST.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {LAST}")
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA}")

    print(f"Resuming from: {LAST.resolve()}")
    model = YOLO(str(LAST), task="detect")
    model.train(resume=True)

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training ended without best.pt: {best}")

    print(f"Best checkpoint: {best.resolve()}")
    metrics = YOLO(str(best), task="detect").val(
        data=str(DATA),
        split="test",
        imgsz=640,
        batch=16,
        device="0",
        workers=4,
        end2end=True,
        project=str(PROJECT),
        name=TEST_NAME,
        exist_ok=False,
        plots=True,
    )
    print(f"Test output: {Path(metrics.save_dir).resolve()}")
    print(f"Test metrics: {metrics.results_dict}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
