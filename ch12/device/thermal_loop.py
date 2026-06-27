# Honesty rules (Ch.12 M3 device session):
# 1. Every number traces to a real subprocess: this script runs the actual
#    Core ML / MLX forward pass in a sustained loop, measures wall time
#    in two windows, and reports the ratio. No synthetic values.
# 2. Only power is allowed to be estimated. Thermal retention is exact.
# 3. Unmeasured fields stay `null` with `# PLACEHOLDER`.
# 4. No new quantization concepts; this is the Ch.11 thermal-knee idea.
"""Sustained-throughput retention = last-30s_thr / first-30s_thr * 100.

Cold-start latency that meets the budget can still fail under sustained
load when the skin-temperature loop drops clocks — the Ch.11 point,
re-measured here on fresh runs.

    python device/thermal_loop.py --mode run \\
        --model ~/models/efficientnet_lite0.mlpackage \\
        --images ~/datasets/imagenet_val_sample \\
        --duration 600 --runtime coreml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from _common import (
    Config,
    list_images,
    update_results,
    validate_dir,
    validate_file,
)
from vision_pipeline import _build_predictor, run_once

WINDOW_S = 30                          #A first-30s vs last-30s windows


def _run_sustained(duration_s: int, predict, images: list[Path]) -> tuple[float, float, int]:
    start = time.monotonic()
    deadline = start + duration_s
    timestamps: list[float] = []
    i = 0
    while time.monotonic() < deadline:
        path = images[i % len(images)]
        run_once(predict, path)
        timestamps.append(time.monotonic())
        i += 1

    if not timestamps:
        raise SystemExit("sustained loop completed zero iterations")

    first_window_end = start + WINDOW_S
    last_window_start = (timestamps[-1]) - WINDOW_S

    first = sum(1 for t in timestamps if t <= first_window_end)
    last = sum(1 for t in timestamps if t >= last_window_start)
    first_thr = first / WINDOW_S
    last_thr = last / WINDOW_S
    return first_thr, last_thr, len(timestamps)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="sustained throughput retention")
    ap.add_argument("--mode", choices=["run"], default="run")
    ap.add_argument("--runtime", choices=["coreml", "mlx"], default="coreml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--duration", type=int, default=600,
                    help="seconds of sustained inference; needs >= 70")
    args = ap.parse_args(argv)

    if args.duration < 2 * WINDOW_S + 10:
        raise SystemExit(f"--duration must be >= {2 * WINDOW_S + 10}s "
                         f"to fit two non-overlapping {WINDOW_S}s windows")

    model_path = validate_file(args.model, "model")
    image_dir = validate_dir(args.images, "images")
    images = list_images(image_dir, 64)         #A small rotating set is fine

    predict = _build_predictor(args.runtime, model_path)

    first_thr, last_thr, n = _run_sustained(args.duration, predict, images)
    retention = (last_thr / first_thr * 100.0) if first_thr > 0 else float("nan")

    update_results({"device": {"thermal_retention_pct": round(retention, 2)}})
    print(f"[thermal_loop] iters={n}  first_thr={first_thr:.2f}/s  "
          f"last_thr={last_thr:.2f}/s  retention={retention:.2f}%")


if __name__ == "__main__":
    main()
