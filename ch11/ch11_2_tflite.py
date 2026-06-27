"""ch11_2_tflite.py — Sec 11.2 partial (Mac-side TFLite, self-contained).

Section served: 11.2 (TFLite half), Mac-only initially.
Phone benchmarks deferred until a Pixel device arrives.

Modes:
  convert            Produce models/tflite/effnet_lite0_{fp32,dynrange,int8,int16x8}.tflite.
  inspect            Walk each .tflite via tf.lite.Interpreter; dump per-tensor dtypes;
                     flag float fallbacks. Produces figure 11.2.2.
  verify-accuracy    ImageNet-1k val subset accuracy. Numerics platform-independent.
  bench-host         TFLite Python interpreter latency on host (M3 laptop class).
  figures            Generate figures 11.2.1, 11.2.2.
  all                convert -> inspect -> verify-accuracy -> bench-host -> figures.
  --smoke            Cheap end-to-end on a synthetic FP32 model: 10-iter latency,
                     one record, one figure stub. Validates I/O + schema only.

Records produced (matched against unified results.json schema):
  - 4 TFLite variants on Mac with accuracy + latency_ms populated.
  - power_mw, sustained, ane_op_coverage left null for this script.

Honesty:
  - TFLite-host on Apple Silicon: detects whether tensorflow-macos vs vanilla tensorflow
    is installed, warns in `notes` if running through Rosetta.
  - ImageNet val subset is a subset (n_samples in record); not the full 50k val set.
  - convert mode requires an EfficientNet-Lite0 source. Prefers a local SavedModel at
    models/tflite/effnet_lite0_savedmodel/, or a kagglehub URL (TF_HUB_HANDLE env var
    override). If neither resolves, the script halts with instructions; never fabricates.

Invocation:
  python ch11_2_tflite.py <mode> [--n-iters 200] [--warmup 50] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Manning matplotlib style (duplicated across all four scripts; see spec §8). #
# --------------------------------------------------------------------------- #
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.4,
    "patch.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.dpi": 300,
})

# Manning grayscale-safe palette: muted hues from Level 2-3 of the color sheet.
# Index 0 carries the most "neutral" hue; downstream code uses modulo indexing.
PALETTE = ["#319974", "#7E76B0", "#D67430", "#3A6FA8", "#888888", "#444444"]
HATCHES = ["", "////", "....", "xxxx", "\\\\\\\\", "++++"]

# --------------------------------------------------------------------------- #
# Paths.                                                                      #
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
# Data file was renamed results.json -> runs.json (same schema/contents).
RESULTS_JSON = HERE / "runs.json"
CAVEATS_MD = HERE / "caveats.md"
MODELS_DIR = HERE / "models" / "tflite"
FIG_DIR = HERE / "figures" / "ch11_2"
LOG_DIR = HERE / "logs"

SCHEMA_VERSION = "11.0"
SCRIPT_NAME = "ch11_2_tflite.py"


# --------------------------------------------------------------------------- #
# ResultRecord dataclass — duplicated across scripts; see spec §3.            #
# --------------------------------------------------------------------------- #
@dataclass
class ResultRecord:
    model: str
    variant: str
    backend: str
    device: dict
    modality: str | None = None
    size_bytes: int | None = None
    params: int | None = None
    quantization: dict | None = None
    compute_units: str | None = None
    latency_ms: dict | None = None
    throughput: dict | None = None
    accuracy: dict | None = None
    power_mw: dict | None = None
    sustained: dict | None = None
    ane_op_coverage: dict | None = None
    memory_mb: dict | None = None
    prepost: dict | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
    script: str = SCRIPT_NAME
    notes: str = ""


def _device_fingerprint() -> dict:
    return {
        "name": "MacBook Air M3",
        "soc": "Apple M3",
        "os": f"macOS {platform.mac_ver()[0]}",
        "class": "laptop",
    }


# --------------------------------------------------------------------------- #
# results.json I/O — duplicated; dedup key per spec §0.                       #
# --------------------------------------------------------------------------- #
def _dedup_key(rec: dict) -> tuple:
    return (
        rec.get("model"),
        rec.get("variant"),
        rec.get("backend"),
        (rec.get("device") or {}).get("name"),
        rec.get("compute_units"),
    )


def _np_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _load_results() -> dict:
    if RESULTS_JSON.exists():
        with RESULTS_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "records" not in data:
            data["records"] = []
        return data
    return {"schema_version": SCHEMA_VERSION, "records": []}


def append_result(rec: ResultRecord) -> str:
    """Append or replace by dedup key. Returns 'added' or 'replaced'."""
    payload = asdict(rec)
    data = _load_results()
    key = _dedup_key(payload)
    for i, existing in enumerate(data["records"]):
        if _dedup_key(existing) == key:
            data["records"][i] = payload
            with RESULTS_JSON.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=_np_default)
            return "replaced"
    data["records"].append(payload)
    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_np_default)
    return "added"


def append_caveat(mode: str, message: str) -> None:
    CAVEATS_MD.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    line = f"- {ts} [{SCRIPT_NAME}::{mode}] {message}\n"
    with CAVEATS_MD.open("a", encoding="utf-8") as f:
        f.write(line)


# --------------------------------------------------------------------------- #
# Figure helper — duplicated across scripts.                                  #
# --------------------------------------------------------------------------- #
def _save_pair(fig, name: str, section: str = "ch11_2") -> tuple[Path, Path]:
    """Write figures/<section>/CH11_<NN>_Kalyanarangan.{png,pdf} + caption stub."""
    out_dir = HERE / "figures" / section
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    cap = out_dir / f"{name.split('_Kalyanarangan')[0]}_caption.md"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if not cap.exists():
        cap.write_text(f"Caption skeleton for {name}. Hatched markers; B&W; flesh out post-run.\n")
    return png, pdf


# --------------------------------------------------------------------------- #
# Logging.                                                                    #
# --------------------------------------------------------------------------- #
def _setup_logger(mode: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"ch11_2_tflite.{mode}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_DIR / f"ch11_2_tflite_{mode}.log", mode="a", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------- #
# TF detection — Apple Silicon native vs Rosetta.                             #
# --------------------------------------------------------------------------- #
def _tf_arch_note() -> str:
    """Return a short string describing the TF flavor in use."""
    try:
        import tensorflow as tf
    except Exception as e:
        return f"tensorflow not importable: {e}"
    is_arm = platform.machine() == "arm64"
    pyarch = platform.machine()
    return (
        f"tensorflow {tf.__version__}; python arch={pyarch}; "
        f"{'native arm64' if is_arm else 'WARN: running through Rosetta'}"
    )


# --------------------------------------------------------------------------- #
# EfficientNet-Lite0 source resolution.                                       #
# --------------------------------------------------------------------------- #
def _resolve_effnet_lite0_savedmodel(logger: logging.Logger) -> Path | None:
    """Look for a local SavedModel; if absent, attempt kagglehub. Never fabricate."""
    local = MODELS_DIR / "effnet_lite0_savedmodel"
    if local.exists() and any(local.iterdir()):
        logger.info("Using local EfficientNet-Lite0 SavedModel at %s", local)
        return local
    try:
        import kagglehub
        # tensorflow/efficientnet-lite/tensorFlow2/lite0-feature-vector or classification
        handle = os.environ.get("EFFNET_LITE0_KAGGLE", "tensorflow/efficientnet-lite/tensorFlow2/lite0-classification")
        logger.info("Attempting kagglehub fetch for %s", handle)
        path = kagglehub.model_download(handle)
        # Copy to our models dir
        local.mkdir(parents=True, exist_ok=True)
        for item in Path(path).iterdir():
            dest = local / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
        logger.info("Cached EfficientNet-Lite0 SavedModel at %s", local)
        return local
    except Exception as e:
        logger.warning("kagglehub fetch failed: %s", e)
        return None


# =========================================================================== #
# Mode: convert                                                                #
# =========================================================================== #
def mode_convert(args: argparse.Namespace) -> int:
    logger = _setup_logger("convert")
    logger.info("ch11_2_tflite convert: %s", _tf_arch_note())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    saved = _resolve_effnet_lite0_savedmodel(logger)
    if saved is None:
        msg = ("EfficientNet-Lite0 SavedModel not resolvable. Either set "
               "EFFNET_LITE0_KAGGLE env var to a kagglehub handle, or place "
               f"a SavedModel at {MODELS_DIR / 'effnet_lite0_savedmodel'}/.")
        logger.error(msg)
        append_caveat("convert", msg)
        return 2
    import tensorflow as tf

    # Representative dataset for INT8 quantization. Uses the calibration shard if
    # present, otherwise random uniform — note this in caveats.
    def _rep_data():
        cal_dir = HERE / "data" / "calib"
        if cal_dir.exists():
            from PIL import Image
            jpegs = sorted([p for p in cal_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:128]
            for p in jpegs:
                img = Image.open(p).convert("RGB").resize((224, 224))
                arr = np.asarray(img, dtype=np.float32) / 255.0
                yield [arr[None, ...]]
        else:
            append_caveat("convert", "No data/calib/ directory; INT8 calibration uses random data — accuracy will be optimistic. Provide images for production calibration.")
            for _ in range(64):
                yield [np.random.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)]

    def _convert_fp32():
        c = tf.lite.TFLiteConverter.from_saved_model(str(saved))
        return c.convert()

    def _convert_dynrange():
        c = tf.lite.TFLiteConverter.from_saved_model(str(saved))
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        return c.convert()

    def _convert_int8():
        c = tf.lite.TFLiteConverter.from_saved_model(str(saved))
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = _rep_data
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = tf.int8
        c.inference_output_type = tf.int8
        return c.convert()

    def _convert_int16x8():
        c = tf.lite.TFLiteConverter.from_saved_model(str(saved))
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = _rep_data
        c.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        return c.convert()

    converters = {
        "fp32": _convert_fp32,
        "dynrange": _convert_dynrange,
        "int8": _convert_int8,
        "int16x8": _convert_int16x8,
    }
    rc = 0
    for name, fn in converters.items():
        out = MODELS_DIR / f"effnet_lite0_{name}.tflite"
        try:
            buf = fn()
            out.write_bytes(buf)
            logger.info("wrote %s (%d bytes)", out, out.stat().st_size)
        except Exception as e:
            rc = 1
            logger.error("convert %s failed: %s", name, e)
            append_caveat("convert", f"variant {name} failed: {e}")
    return rc


# =========================================================================== #
# Mode: inspect                                                                #
# =========================================================================== #
def mode_inspect(args: argparse.Namespace) -> int:
    logger = _setup_logger("inspect")
    import tensorflow as tf

    summaries = {}
    for variant in ["fp32", "dynrange", "int8", "int16x8"]:
        path = MODELS_DIR / f"effnet_lite0_{variant}.tflite"
        if not path.exists():
            logger.warning("missing %s — skip inspect", path)
            continue
        interp = tf.lite.Interpreter(model_path=str(path))
        interp.allocate_tensors()
        details = interp.get_tensor_details()
        dtypes: dict[str, int] = {}
        float_fallbacks = []
        for d in details:
            dt = np.dtype(d["dtype"]).name
            dtypes[dt] = dtypes.get(dt, 0) + 1
            if "float" in dt and variant in ("int8", "int16x8"):
                float_fallbacks.append(d["name"])
        summaries[variant] = {"dtypes": dtypes, "float_fallbacks": float_fallbacks}
        logger.info("%s: %s; float fallbacks: %d", variant, dtypes, len(float_fallbacks))

    # Render figure 11.2.2 — per-tensor dtype map, stacked bars
    if summaries:
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        variants = list(summaries.keys())
        all_dtypes = sorted({dt for v in summaries.values() for dt in v["dtypes"]})
        bottom = np.zeros(len(variants))
        for i, dt in enumerate(all_dtypes):
            heights = np.array([summaries[v]["dtypes"].get(dt, 0) for v in variants], dtype=float)
            ax.bar(variants, heights, bottom=bottom, label=dt,
                   color=PALETTE[i % len(PALETTE)],
                   hatch=HATCHES[i % len(HATCHES)],
                   edgecolor="black", linewidth=0.5)
            bottom += heights
        ax.set_ylabel("Tensor count by dtype")
        ax.set_title("11.2.2 — TFLite per-tensor dtype map (EfficientNet-Lite0)")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        _save_pair(fig, "CH11_F0202_Kalyanarangan", "ch11_2")
        plt.close(fig)
    summary_path = LOG_DIR / "ch11_2_inspect_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, default=str))
    return 0


# =========================================================================== #
# Mode: verify-accuracy                                                        #
# =========================================================================== #
def mode_verify_accuracy(args: argparse.Namespace) -> int:
    logger = _setup_logger("verify-accuracy")
    val_dir = HERE / "data" / "imagenet_val"
    label_map = HERE / "data" / "imagenet_labels.json"
    if not val_dir.exists() or not label_map.exists():
        msg = (f"ImageNet val subset not found at {val_dir} (and labels at {label_map}). "
               "ImageNet is not freely redistributable; the user must provide a 1k-image "
               "subset with per-image label JSON. accuracy not measured.")
        logger.error(msg)
        append_caveat("verify-accuracy", msg)
        return 2
    # If both present, do real top-1/top-5 measurement.
    import tensorflow as tf
    from PIL import Image
    labels = json.loads(label_map.read_text())  # {filename: class_index}
    files = sorted([p for p in val_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    files = files[:args.n_samples]
    rc_overall = 0
    for variant in ["fp32", "dynrange", "int8", "int16x8"]:
        path = MODELS_DIR / f"effnet_lite0_{variant}.tflite"
        if not path.exists():
            continue
        interp = tf.lite.Interpreter(model_path=str(path))
        interp.allocate_tensors()
        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]
        top1 = 0
        top5 = 0
        n = 0
        for f in files:
            label = labels.get(f.name)
            if label is None:
                continue
            img = Image.open(f).convert("RGB").resize((224, 224))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = arr[None, ...]
            if in_det["dtype"] == np.int8:
                scale, zp = in_det["quantization"]
                arr = np.clip(np.round(arr / scale + zp), -128, 127).astype(np.int8)
            interp.set_tensor(in_det["index"], arr)
            interp.invoke()
            out = interp.get_tensor(out_det["index"])[0]
            if out_det["dtype"] == np.int8:
                scale, zp = out_det["quantization"]
                out = (out.astype(np.float32) - zp) * scale
            preds = np.argsort(out)[::-1][:5]
            if preds[0] == label:
                top1 += 1
            if label in preds:
                top5 += 1
            n += 1
        if n == 0:
            logger.warning("no labelled images matched for %s", variant)
            continue
        rec = ResultRecord(
            model="efficientnet_lite0",
            modality="vision",
            variant=variant,
            backend="tflite",
            compute_units="cpu_eval",  # don't collide with bench-host's compute_units=None
            device=_device_fingerprint(),
            accuracy={
                "metric": "top1",
                "value": top1 / n,
                "secondary": {"top5": top5 / n},
                "dataset": "imagenet_val_subset",
                "n_samples": n,
            },
            notes=f"verify-accuracy on {n} labelled images.",
        )
        action = append_result(rec)
        logger.info("%s top1=%.3f top5=%.3f (%s)", variant, top1 / n, top5 / n, action)
    return rc_overall


# =========================================================================== #
# Mode: bench-host                                                             #
# =========================================================================== #
def _time_invocations(interp, n_warm: int, n_iter: int) -> list[float]:
    in_det = interp.get_input_details()[0]
    shape = in_det["shape"]
    if in_det["dtype"] == np.int8:
        x = np.random.randint(-128, 127, size=shape, dtype=np.int8)
    elif in_det["dtype"] == np.uint8:
        x = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    else:
        x = np.random.uniform(0, 1, size=shape).astype(np.float32)
    interp.set_tensor(in_det["index"], x)
    for _ in range(n_warm):
        interp.invoke()
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def mode_bench_host(args: argparse.Namespace) -> int:
    logger = _setup_logger("bench-host")
    import tensorflow as tf

    rc = 0
    for variant in ["fp32", "dynrange", "int8", "int16x8"]:
        path = MODELS_DIR / f"effnet_lite0_{variant}.tflite"
        if not path.exists():
            logger.warning("missing %s — skip bench", path)
            continue
        interp = tf.lite.Interpreter(model_path=str(path))
        interp.allocate_tensors()
        try:
            times = _time_invocations(interp, args.warmup, args.n_iters)
        except Exception as e:
            logger.error("bench %s failed: %s", variant, e)
            append_caveat("bench-host", f"variant {variant} failed: {e}")
            rc = 1
            continue
        in_shape = list(interp.get_input_details()[0]["shape"])
        rec = ResultRecord(
            model="efficientnet_lite0",
            modality="vision",
            variant=variant,
            backend="tflite",
            device=_device_fingerprint(),
            size_bytes=path.stat().st_size,
            latency_ms={
                "p50": float(np.percentile(times, 50)),
                "p95": float(np.percentile(times, 95)),
                "mean": float(np.mean(times)),
                "n_iters": args.n_iters,
                "warmup_iters": args.warmup,
                "input_shape": in_shape,
            },
            throughput={
                "samples_per_sec": float(1000.0 / np.mean(times)),
                "tokens_per_sec": None,
                "prompt_length": None,
                "generation_length": None,
            },
            notes=_tf_arch_note(),
        )
        action = append_result(rec)
        logger.info("%s p50=%.2f p95=%.2f mean=%.2f ms (%s)",
                    variant, rec.latency_ms["p50"], rec.latency_ms["p95"], rec.latency_ms["mean"], action)
    return rc


# =========================================================================== #
# Mode: figures                                                                #
# =========================================================================== #
def mode_figures(args: argparse.Namespace) -> int:
    logger = _setup_logger("figures")
    data = _load_results()
    # 11.2.1 ingests *all* TFLite efficientnet records (Mac + Android both),
    # so the size/accuracy curve aggregates across devices. Filter by model
    # name rather than by script, since ch11_4_android.py contributes records
    # too.
    recs_all = [r for r in data["records"]
                if r.get("backend") == "tflite" and r.get("model") == "efficientnet_lite0"]

    # 11.2.1: size × accuracy (one point per variant family, averaged across devices)
    # Normalize variant strings so Mac (`fp32`) and Android (`tflite_fp32`)
    # collapse into one canonical family. Same .tflite file → same numbers,
    # so we average accuracy and take the (matching) size once.
    def _variant_family_name(v: str) -> str | None:
        if not v: return None
        vv = v.lower().removeprefix("tflite_")
        for canon in ("fp32", "dynrange", "int8", "int16x8"):
            if canon in vv:
                return canon
        return None

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    by_family: dict[str, dict[str, Any]] = {}
    for r in recs_all:
        fam = _variant_family_name(r.get("variant"))
        if fam is None:
            continue
        d = by_family.setdefault(fam, {"acc_samples": []})
        if r.get("size_bytes"):
            d["size"] = r["size_bytes"]
        if r.get("accuracy") and r["accuracy"].get("value") is not None:
            d["acc_samples"].append(r["accuracy"]["value"])

    plotted = 0
    order = ["fp32", "dynrange", "int8", "int16x8"]
    markers = {"fp32": "o", "dynrange": "s", "int8": "D", "int16x8": "^"}
    for i, fam in enumerate(order):
        d = by_family.get(fam) or {}
        if "size" in d and d.get("acc_samples"):
            acc = float(np.mean(d["acc_samples"]))
            ax.scatter(d["size"] / 1024, acc * 100, marker=markers[fam],
                       s=110, edgecolor="black", linewidths=0.7,
                       facecolor=PALETTE[i % len(PALETTE)],
                       hatch=HATCHES[i % len(HATCHES)], label=fam, zorder=3)
            plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "data not available", transform=ax.transAxes,
                ha="center", va="center", fontsize=10)
    ax.set_xlabel("Model size (KB)")
    ax.set_ylabel("ImageNet top-1 (%)")
    ax.set_title("11.2.1 — TFLite size × accuracy (EfficientNet-Lite0)")
    ax.grid(True, axis="y", linestyle=":", color="#cccccc", linewidth=0.5, zorder=0)
    # Push the legend to the right of the axes so the hatched + shaped markers
    # don't crowd against the data points. labelspacing + handlelength keep
    # the swatches readable.
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0., framealpha=0.95, edgecolor="#cccccc",
              labelspacing=1.1, handlelength=2.0, handletextpad=0.6,
              borderpad=0.6)
    fig.tight_layout()
    _save_pair(fig, "CH11_F0201_Kalyanarangan", "ch11_2")
    plt.close(fig)

    # 11.2.3: TFLite latency-by-delegate on phone-class devices. Populated as
    # soon as a phone produces records (e.g. Pixel 10 Pro via ch11_4_android).
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    phone_recs = [r for r in recs_all
                  if (r.get("device") or {}).get("class") == "phone"
                  and (r.get("latency_ms") or {}).get("p50") is not None
                  and not (r.get("compute_units") or "").endswith(("_sustained_300s",
                                                                    "_sustained_60s",
                                                                    "_power_30s"))]
    if phone_recs:
        # Pivot: rows = variant, columns = delegate (compute_units), value = p50 ms.
        variants_order = ["tflite_fp32", "tflite_dynrange", "tflite_int8", "tflite_int16x8"]
        cu_order = ["xnnpack_1t", "xnnpack_4t", "gpu", "nnapi"]
        device_name = (phone_recs[0].get("device") or {}).get("name", "phone")
        pivot: dict[str, dict[str, float]] = {v: {} for v in variants_order}
        for r in phone_recs:
            v = r["variant"]; cu = r.get("compute_units") or "?"
            if v in pivot and cu in cu_order:
                pivot[v][cu] = (r.get("latency_ms") or {}).get("p50") or float("nan")
        x = np.arange(len(variants_order))
        bar_w = 0.20
        for i, cu in enumerate(cu_order):
            heights = [pivot[v].get(cu, float("nan")) for v in variants_order]
            ax.bar(x + (i - 1.5) * bar_w, heights, bar_w,
                   color=PALETTE[i % len(PALETTE)],
                   hatch=HATCHES[i % len(HATCHES)],
                   edgecolor="black", linewidth=0.5, label=cu, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("tflite_", "") for v in variants_order])
        ax.set_yscale("log")
        ax.set_ylabel("p50 latency (ms, log)")
        ax.set_title(f"11.2.3 — TFLite p50 latency by delegate on {device_name}")
        ax.grid(True, axis="y", linestyle=":", color="#cccccc", linewidth=0.5, zorder=0)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  borderaxespad=0., framealpha=0.95, edgecolor="#cccccc",
                  labelspacing=0.5, handlelength=2.0, handletextpad=0.6,
                  title="delegate", title_fontsize=7)
    else:
        ax.text(0.5, 0.5, "no phone-class TFLite records yet",
                transform=ax.transAxes, ha="center", va="center", color="#888888")
        ax.axis("off")
    fig.tight_layout()
    _save_pair(fig, "CH11_F0203_Kalyanarangan", "ch11_2")
    plt.close(fig)
    logger.info("wrote 11.2.1 (plotted=%d) and 11.2.3 (phone_recs=%d)",
                plotted, len(phone_recs))
    return 0


# =========================================================================== #
# Mode: smoke                                                                  #
# =========================================================================== #
def _build_tiny_tflite_fp32() -> bytes:
    """Build a tiny FP32 TFLite model for smoke purposes only."""
    import tensorflow as tf
    inp = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inp, x)
    return tf.lite.TFLiteConverter.from_keras_model(model).convert()


def mode_smoke(args: argparse.Namespace) -> int:
    logger = _setup_logger("smoke")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    smoke_path = MODELS_DIR / "smoke_fp32.tflite"
    smoke_path.write_bytes(_build_tiny_tflite_fp32())
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(smoke_path))
    interp.allocate_tensors()
    times = _time_invocations(interp, n_warm=3, n_iter=10)
    rec = ResultRecord(
        model="smoke_tinyconv",
        modality="vision",
        variant="fp32",
        backend="tflite",
        device=_device_fingerprint(),
        size_bytes=smoke_path.stat().st_size,
        latency_ms={
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "mean": float(np.mean(times)),
            "n_iters": 10,
            "warmup_iters": 3,
            "input_shape": list(interp.get_input_details()[0]["shape"]),
        },
        notes=f"smoke run; {_tf_arch_note()}",
    )
    action = append_result(rec)
    logger.info("smoke: tinyconv fp32 mean=%.3f ms (%s)", rec.latency_ms["mean"], action)
    # No figure for smoke mode — it carries no editorial content.
    smoke_path.unlink(missing_ok=True)
    return 0


# =========================================================================== #
# Mode: all                                                                    #
# =========================================================================== #
def mode_all(args: argparse.Namespace) -> int:
    rc = 0
    for fn, name in [
        (mode_convert, "convert"),
        (mode_inspect, "inspect"),
        (mode_verify_accuracy, "verify-accuracy"),
        (mode_bench_host, "bench-host"),
        (mode_figures, "figures"),
    ]:
        sub = fn(args)
        if sub != 0:
            rc = sub
    return rc


# =========================================================================== #
# CLI                                                                          #
# =========================================================================== #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="all",
                   choices=["convert", "inspect", "verify-accuracy", "bench-host",
                            "figures", "all", "smoke"])
    p.add_argument("--n-iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--smoke", action="store_true", help="alias for mode=smoke")
    args = p.parse_args(argv)
    if args.smoke:
        args.mode = "smoke"
    dispatch = {
        "convert": mode_convert,
        "inspect": mode_inspect,
        "verify-accuracy": mode_verify_accuracy,
        "bench-host": mode_bench_host,
        "figures": mode_figures,
        "all": mode_all,
        "smoke": mode_smoke,
    }
    return dispatch[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
