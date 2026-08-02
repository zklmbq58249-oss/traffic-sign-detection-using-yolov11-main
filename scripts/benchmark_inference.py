#!/usr/bin/env python
"""Measure headless end-to-end video inference for a fixed YOLO model and source."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import cv2
import torch

# Override a process-level user config path so benchmark runs remain self-contained.
os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parents[1] / ".ultralytics")

from ultralytics import YOLO


def synchronize(device: str) -> None:
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def next_frame(capture: cv2.VideoCapture) -> tuple[bool, object]:
    ok, frame = capture.read()
    if ok:
        return ok, frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return capture.read()


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="JSON metrics destination")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes every frame once")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.source.is_file():
        raise FileNotFoundError(f"Video not found: {args.source}")
    if args.warmup < 0 or args.max_frames < 0:
        raise ValueError("warmup and max-frames must be non-negative")

    model = YOLO(str(args.model), task="detect")
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.source}")

    for _ in range(args.warmup):
        ok, frame = next_frame(capture)
        if not ok:
            raise RuntimeError("Video has no readable frames")
        _ = model(frame, imgsz=args.imgsz, device=args.device, verbose=False)
        synchronize(args.device)

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if args.device != "cpu" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    durations_ms: list[float] = []
    while True:
        if args.max_frames and len(durations_ms) >= args.max_frames:
            break
        started = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            break
        result = model(frame, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        _ = result.plot()
        synchronize(args.device)
        durations_ms.append((time.perf_counter() - started) * 1000)

    capture.release()
    if not durations_ms:
        raise RuntimeError("No frames were benchmarked")

    mean_ms = statistics.fmean(durations_ms)
    metrics = {
        "model": str(args.model.resolve()),
        "source": str(args.source.resolve()),
        "device": args.device,
        "imgsz": args.imgsz,
        "warmup_frames": args.warmup,
        "measured_frames": len(durations_ms),
        "timing_definition": "video decode + model preprocessing/inference/postprocessing + result annotation; excludes model load and video output encoding",
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(statistics.median(durations_ms), 3),
        "p95_ms": round(percentile(durations_ms, 0.95), 3),
        "fps": round(1000 / mean_ms, 3),
        "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 3)
        if args.device != "cpu" and torch.cuda.is_available()
        else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
