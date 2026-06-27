# Honesty rules (Ch.12 M3 device session):
# 1. Every number traces to a real subprocess on this M3 — no fabrication.
# 2. Only `power_w_estimated` is allowed to be estimated; this file produces
#    timing numbers, all real (time.perf_counter wall time around the actual
#    stage operation, not a synthetic figure).
# 3. Unmeasured fields stay `null` with `# PLACEHOLDER`.
# 4. No new quantization concepts; the EfficientNet-Lite0 INT8/palettize
#    variant and Core ML / MLX runtimes carry over from Chapter 11.
"""EfficientNet-Lite0 end-to-end pipeline with per-stage timers.

Stages: decode → resize → normalize → infer → post (argmax).
Records per-stage mean wall time + end-to-end p95 over N images.
The chapter's point (Ch.11 §11.5, restated): infer is sub-5 ms on this
device, so the sub-second budget is decided by pre/post, not the forward
pass. The figure in §10 makes that visible.

Run (on the M3, never inside the Claude Code container):

    python device/vision_pipeline.py --mode run \\
        --model ~/models/efficientnet_lite0.mlpackage \\
        --images ~/datasets/imagenet_val_sample \\
        --runtime coreml --n 200
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from _common import (  #B shared helpers live next to this script
    BASE,
    Config,
    list_images,
    percentile,
    update_results,
    validate_dir,
    validate_file,
)


@dataclass
class StageTimes:
    decode: float = 0.0
    resize: float = 0.0
    normalize: float = 0.0
    infer: float = 0.0
    post: float = 0.0

    def total(self) -> float:
        return self.decode + self.resize + self.normalize + self.infer + self.post


def _load_coreml(model_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable that runs the Core ML forward pass on one CHW float32 batch."""
    import coremltools as ct  #C imported lazily so the module loads on non-M3 too
    model = ct.models.MLModel(str(model_path))
    spec_inputs = model.get_spec().description.input
    if not spec_inputs:
        raise SystemExit("Core ML model has no inputs")
    input_name = spec_inputs[0].name
    output_names = [o.name for o in model.get_spec().description.output]
    if not output_names:
        raise SystemExit("Core ML model has no outputs")
    output_name = output_names[0]

    def _predict(x: np.ndarray) -> np.ndarray:
        out = model.predict({input_name: x})
        return np.asarray(out[output_name])
    return _predict


def _load_mlx(model_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """MLX path: expects a callable saved by the Ch.11 export. Kept thin on purpose."""
    import mlx.core as mx  #C
    import importlib.util
    spec_path = model_path
    if spec_path.is_dir():
        spec_path = spec_path / "model.py"
    spec = importlib.util.spec_from_file_location("ch11_mlx_model", spec_path)
    if not spec or not spec.loader:
        raise SystemExit(f"MLX model loader not found at {spec_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "forward"):
        raise SystemExit("MLX model module must expose forward(x)->array")

    def _predict(x: np.ndarray) -> np.ndarray:
        y = mod.forward(mx.array(x))
        return np.asarray(y)
    return _predict


def _build_predictor(runtime: str, model_path: Path) -> Callable[[np.ndarray], np.ndarray]:
    if runtime == "coreml":
        return _load_coreml(model_path)
    if runtime == "mlx":
        return _load_mlx(model_path)
    raise SystemExit(f"unknown runtime: {runtime}")


def _decode(path: Path) -> Image.Image:                      #B PIL decode = JPEG → RGB
    with open(path, "rb") as f:
        data = f.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def _resize(img: Image.Image, size: int = 224) -> Image.Image:
    return img.resize((size, size), Image.BILINEAR)


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _normalize(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32) / 255.0          #B HWC
    arr = arr.transpose(2, 0, 1)[None, ...]                  #B 1CHW
    return (arr - _MEAN) / _STD


def _post(logits: np.ndarray) -> int:
    return int(np.argmax(logits.reshape(-1)))


def run_once(predict: Callable[[np.ndarray], np.ndarray], path: Path) -> StageTimes:
    t = StageTimes()
    t0 = time.perf_counter()
    img = _decode(path);          t1 = time.perf_counter()
    img = _resize(img);           t2 = time.perf_counter()
    x = _normalize(img);          t3 = time.perf_counter()
    logits = predict(x);          t4 = time.perf_counter()
    _ = _post(logits);            t5 = time.perf_counter()
    t.decode    = (t1 - t0) * 1000.0
    t.resize    = (t2 - t1) * 1000.0
    t.normalize = (t3 - t2) * 1000.0
    t.infer     = (t4 - t3) * 1000.0
    t.post      = (t5 - t4) * 1000.0
    return t


def main_run(cfg: Config) -> None:
    model_path = validate_file(cfg.model_path, "model")
    image_dir = validate_dir(cfg.image_dir, "images")
    paths = list_images(image_dir, cfg.n_images + cfg.warmup)
    if len(paths) < cfg.n_images + cfg.warmup:
        raise SystemExit(f"need {cfg.n_images + cfg.warmup} images, found {len(paths)}")

    predict = _build_predictor(cfg.runtime, model_path)

    for p in paths[: cfg.warmup]:                   #A warm-up, discarded
        run_once(predict, p)

    samples: list[StageTimes] = []
    for p in paths[cfg.warmup : cfg.warmup + cfg.n_images]:
        samples.append(run_once(predict, p))

    def mean(field: str) -> float:
        return float(np.mean([getattr(s, field) for s in samples]))

    totals = [s.total() for s in samples]
    p95 = float(percentile(totals, 0.95))

    patch = {
        "meta": {
            "runtime": cfg.runtime,
            "variant": model_path.name,
            "captured_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        "device": {
            "stage_latency_ms": {
                "decode":    round(mean("decode"),    4),
                "resize":    round(mean("resize"),    4),
                "normalize": round(mean("normalize"), 4),
                "infer":     round(mean("infer"),     4),
                "post":      round(mean("post"),      4),
            },
            "e2e_p95_ms": round(p95, 4),
        },
    }
    update_results(patch)
    print(f"\n[vision_pipeline] wrote stage_latency_ms + e2e_p95_ms to results_device.json")
    print(f"  decode={patch['device']['stage_latency_ms']['decode']:.3f} ms  "
          f"resize={patch['device']['stage_latency_ms']['resize']:.3f} ms  "
          f"normalize={patch['device']['stage_latency_ms']['normalize']:.3f} ms  "
          f"infer={patch['device']['stage_latency_ms']['infer']:.3f} ms  "
          f"post={patch['device']['stage_latency_ms']['post']:.3f} ms  "
          f"e2e_p95={patch['device']['e2e_p95_ms']:.2f} ms")


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="EfficientNet-Lite0 device pipeline + timers")
    ap.add_argument("--mode", choices=["run"], default="run")
    ap.add_argument("--runtime", choices=["coreml", "mlx"], default="coreml")
    ap.add_argument("--model", required=True, help="path to .mlpackage or MLX model dir")
    ap.add_argument("--images", required=True, help="directory of input images")
    ap.add_argument("--n", type=int, default=200, help="timed sample count")
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    return Config(
        runtime=args.runtime,
        model_path=args.model,
        image_dir=args.images,
        n_images=args.n,
        warmup=args.warmup,
    )


if __name__ == "__main__":
    main_run(parse_args())
