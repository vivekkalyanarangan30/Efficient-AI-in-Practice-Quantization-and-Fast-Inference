"""
Chapter 9, Section 9.2 — Deploy through Optimum and ONNX Runtime
Companion script for "Efficient AI in Practice: Quantization and Fast Inference"

Delta from §6.4:
  §6.4 built the INT8 ONNX artifact using the RAW `onnxruntime.quantization`
  API. §9.2 runs that same artifact through Optimum's task-aware deployment
  API across three execution providers, applies the transformer-aware
  optimizer §6.4 explicitly deferred, and tunes calibration method,
  io_binding, and session options to the target hardware.

Modes:
  --mode optimum                Optimum's HF-task-aware deployment API (Listing 9.1)
  --mode transformer-optimizer  onnxruntime.transformers.optimizer attention fusion
  --mode ep-comparison          CPU EP vs CUDA EP vs TensorRT EP (Figure 9.4)
  --mode calibration            MinMax vs Entropy vs Percentile on ResNet-18 (Table 9.3)
  --mode iobinding              io_binding + session options on CUDA EP (Listing 9.2)
  --mode all                    Run all modes (order matters: optimum first)

Usage:
  # Recommended: Colab with GPU runtime (T4 / L4 / A100).
  # Runs every mode end-to-end, including CUDA EP and TRT EP if available.
  python ch9_ort_optimum_deployment.py --mode all --save-plots

  # CPU-only (MacBook). Skips CUDA EP, TRT EP, iobinding; everything else runs.
  python ch9_ort_optimum_deployment.py --mode all --device cpu --save-plots

  # Single mode
  python ch9_ort_optimum_deployment.py --mode calibration --save-plots

  # Force a fresh ONNX export (otherwise cached artifacts are reused)
  python ch9_ort_optimum_deployment.py --mode optimum --force-export

Requires:
  Core:          onnxruntime OR onnxruntime-gpu (pick ONE), numpy, matplotlib
  optimum mode:  optimum[onnxruntime]>=1.21, transformers>=4.40, datasets
  calibration:   torch, torchvision, Pillow
  iobinding:     onnxruntime-gpu + CUDA GPU
  TRT EP:        TensorRT 8.6+ bundled with onnxruntime-gpu

Recommended install on Colab (GPU runtime):
  pip install -q "optimum[onnxruntime-gpu]>=1.21" "transformers>=4.40" \\
                 "datasets>=2.20" torch torchvision pillow matplotlib \\
                 onnx onnxscript

Optional — enable the TensorRT EP row in Table 9.2:
  pip install --extra-index-url https://pypi.nvidia.com tensorrt-cu12==10.3.0
  # The script detects the `tensorrt` package and preloads its .so files
  # at mode=ep-comparison startup. No LD_LIBRARY_PATH / runtime restart
  # needed. If this package isn't installed, the TRT row is skipped with
  # a single clean message — no multi-line ORT error spew.

  NOTE: do NOT install both `onnxruntime` and `onnxruntime-gpu` in the same
  env. The -gpu package includes the CPU EP too; installing both causes
  import conflicts. `optimum[onnxruntime-gpu]` pulls the right one.
  `onnxscript` is only used by torch's modern dynamo exporter; the script
  forces `dynamo=False`, but pip may still resolve torch's extra import.

Reference models (same as §6.4; no new models introduced):
  BERT-base SST-2:  textattack/bert-base-uncased-SST-2
  ResNet-18:        torchvision.models.resnet18(weights='IMAGENET1K_V1')
"""

import argparse
import logging
import os
import platform
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ─── Log hygiene ─────────────────────────────────────────────────────────────
# ORT's histogram-based calibration routinely produces numpy overflow
# warnings on transformer attention-mask constants (see comment in
# _build_bert_int8_artifacts). They're informational, not errors.
# Matplotlib's font_manager warns when Arial isn't installed; we handle
# that via fallback in apply_manning_style, so the warning is just noise.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=r".*overflow encountered.*")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message=r".*invalid value encountered.*")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# ─── Configuration ───────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

BERT_MODEL_ID = "textattack/bert-base-uncased-SST-2"
BERT_NUM_HEADS = 12
BERT_HIDDEN_SIZE = 768

# BERT ONNX artifact directories (produced by --mode optimum, consumed by
# the other modes). Cached between runs; delete onnx_cache/ or pass
# --force-export to rebuild.
ARTIFACT_DIRS = {
    "bert_fp32":      "bert-sst2-fp32",
    "bert_fp32_opt":  "bert-sst2-fp32-optimized",    # transformer-optimizer
    "bert_int8_dyn":  "bert-sst2-int8-dynamic",       # Optimum dynamic (QOperator)
    "bert_int8_stat": "bert-sst2-int8-static",        # Optimum static (QOperator)
    "bert_int8_qdq":  "bert-sst2-int8-qdq",           # raw ORT static (QDQ)
    "resnet18_fp32":  "resnet18-fp32.onnx",
    "resnet18_mm":    "resnet18-int8-minmax.onnx",
    "resnet18_ent":   "resnet18-int8-entropy.onnx",
    "resnet18_pct":   "resnet18-int8-percentile.onnx",
}


@dataclass
class Config:
    mode: str = "all"
    device: str = "auto"              # auto | cuda | cpu
    save_plots: bool = False
    force_export: bool = False
    output_dir: Path = SCRIPT_DIR / "figures"
    cache_dir: Path = SCRIPT_DIR / "onnx_cache"

    # BERT benchmarking
    bert_batch_size: int = 8
    bert_seq_length: int = 128
    bert_num_eval: int = 200          # SST-2 validation samples
    bert_num_calib: int = 64          # calibration samples (static INT8)
    bert_num_warmup: int = 5
    bert_num_iters: int = 30

    # ResNet-18 calibration comparison
    resnet_num_calib: int = 100       # calibration samples
    resnet_num_eval: int = 500        # agreement-rate samples
    resnet_input_size: int = 224

    seed: int = 42


# ─── Manning figure style (matches ch8_ternary.py) ───────────────────────────

COLORS = {
    "fp32":     "#7570b3",   # Purple — FP32 baseline
    "int8_dyn": "#66a61e",   # Olive — dynamic INT8
    "int8_sta": "#1b9e77",   # Teal — static INT8
    "int8_opt": "#0a7060",   # Dark teal — optimized INT8
    "cpu_ep":   "#7570b3",   # Purple — CPU EP
    "cuda_ep":  "#76b900",   # NVIDIA green — CUDA EP
    "trt_ep":   "#005f02",   # Dark green — TRT EP
    "minmax":   "#d95f02",   # Orange — MinMax
    "entropy":  "#1b9e77",   # Teal — Entropy (KL)
    "percentile":"#e6ab02",  # Gold — Percentile
    "before":   "#b2182b",   # Red — before
    "after":    "#1a9850",   # Green — after
    "no_bind":  "#d7301f",   # Red — no io_binding
    "iobind":   "#1a9850",   # Green — io_binding
}

HATCHES = {
    "fp32":     "..",
    "int8_dyn": "//",
    "int8_sta": "xx",
    "int8_opt": "++",
    "cpu_ep":   "..",
    "cuda_ep":  "//",
    "trt_ep":   "xx",
    "minmax":   "",
    "entropy":  "//",
    "percentile":"xx",
    "before":   "xx",
    "after":    "",
    "no_bind":  "xx",
    "iobind":   "",
}


def _pick_font() -> str:
    """Choose Arial if installed (Manning's house font), else DejaVu Sans."""
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Arial", "Helvetica", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "sans-serif"


def apply_manning_style():
    """Apply Manning Publications figure style guidelines."""
    plt.rcParams.update({
        "font.family": _pick_font(),
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "figure.figsize": (5.6, 3.5),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_or_show(fig, name: str, config: Config):
    """Save figure in both PNG (300 DPI) and PDF (fonttype 42), or show."""
    if config.save_plots:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        png_path = config.output_dir / f"{name}.png"
        pdf_path = config.output_dir / f"{name}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        fig.savefig(pdf_path, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")
    else:
        plt.show()
    plt.close(fig)


# ─── Environment / hardware helpers ──────────────────────────────────────────

def get_ort():
    """Import onnxruntime (CPU or GPU build). Returns module or None."""
    try:
        import onnxruntime as ort
        return ort
    except ImportError:
        return None


def get_torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def resolve_device(config: Config) -> str:
    if config.device != "auto":
        return config.device
    return "cuda" if get_torch_cuda_available() else "cpu"


def available_execution_providers(ort) -> List[str]:
    """List of EPs that onnxruntime actually has compiled in."""
    if ort is None:
        return []
    return list(ort.get_available_providers())


def _pkg_version(name: str) -> str:
    """Return installed package version, or 'NOT INSTALLED'."""
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(name)
        except PackageNotFoundError:
            return "NOT INSTALLED"
    except ImportError:
        return "NOT INSTALLED"


def print_environment(ort, config: Config):
    """Human-readable environment summary for the header."""
    print(f"  Python:         {sys.version.split()[0]}")
    print(f"  Platform:       {platform.system()} {platform.machine()}")

    if ort is None:
        print(f"  onnxruntime:    NOT INSTALLED")
    else:
        print(f"  onnxruntime:    {ort.__version__}")
        print(f"  EPs available:  {', '.join(available_execution_providers(ort))}")

    print(f"  torch:          {_pkg_version('torch')}")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name()
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            cap = torch.cuda.get_device_capability()
            print(f"  GPU:            {name} (SM {cap[0]}.{cap[1]}, {mem:.0f} GB)")
        else:
            print(f"  GPU:            None (CPU mode)")
    except ImportError:
        pass

    print(f"  optimum:        {_pkg_version('optimum')}")
    print(f"  Cache dir:      {config.cache_dir}")


# ─── Inference latency measurement (used by several modes) ───────────────────

def time_session(
    sess,
    feed: Dict[str, np.ndarray],
    num_warmup: int,
    num_iters: int,
) -> Tuple[float, float]:
    """Returns (mean_ms, std_ms) over num_iters after num_warmup warmup passes.

    Uses wall-clock via perf_counter. For CUDA EP this already includes the
    H2D/D2H copy overhead — the iobinding mode isolates that.                 #A
    """
    for _ in range(num_warmup):
        sess.run(None, feed)
    timings = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        sess.run(None, feed)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)
    return float(np.mean(timings)), float(np.std(timings))

#A Warmup matters disproportionately for the TensorRT EP, which builds or
#  loads an engine on first run (often 10-60s on BERT-base). Five warmup
#  iterations is enough for CUDA/CPU but not for TRT's cold build — the
#  ep-comparison mode handles TRT warmup separately.


# ─── Shared helpers used by multiple modes ──────────────────────────────────

def _bert_random_feed(B: int, L: int, seed: int = 42) -> Dict[str, np.ndarray]:
    """Synthetic BERT inputs (int64). Deterministic via seed."""
    rng = np.random.default_rng(seed)
    return {
        "input_ids":      rng.integers(0, 30000, (B, L), dtype=np.int64),
        "attention_mask": np.ones((B, L), dtype=np.int64),
        "token_type_ids": np.zeros((B, L), dtype=np.int64),
    }


def _find_onnx(dir_path: Path, prefer_quantized: bool = True) -> Path:
    """Resolve the primary .onnx file in a directory.

    Optimum emits `model.onnx` for FP32 and `model_quantized.onnx` for
    quantized variants; prefer the latter when asked and present.
    """
    cands = list(dir_path.glob("*.onnx"))
    if not cands:
        raise FileNotFoundError(f"No .onnx file in {dir_path}")
    if prefer_quantized:
        qs = [c for c in cands if "quantized" in c.name]
        if qs:
            return qs[0]
    return cands[0]


def _make_session(ort, model_path: Path, providers, enable_all: bool = True):
    """Standard InferenceSession with ORT_ENABLE_ALL (the sensible default)."""
    so = ort.SessionOptions()
    if enable_all:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=so,
                                providers=providers)


# ─── TensorRT EP availability probe ──────────────────────────────────────────

def _trt_libs_loadable() -> bool:
    """True if libnvinfer is findable via the process's dlopen search path.

    ORT's `get_available_providers()` lists TensorrtExecutionProvider whenever
    the ORT build has TRT support compiled in — independent of whether the
    TRT shared libraries are installed. On Colab (and most vanilla CUDA
    environments), the provider is *advertised* but libnvinfer.so.10 is not
    present. Without this probe, session creation emits a multi-line
    CDLL-failed error to stderr on every run.
    """
    import ctypes
    for so_name in ("libnvinfer.so.10", "libnvinfer.so.8", "libnvinfer.so"):
        try:
            ctypes.CDLL(so_name)
            return True
        except OSError:
            continue
    return False


def _try_preload_trt_libs() -> bool:
    """If `tensorrt` is pip-installed, preload its .so files into the process.

    Installing `tensorrt-cu12` puts libnvinfer.so.10 inside site-packages but
    not on LD_LIBRARY_PATH — and LD_LIBRARY_PATH can't be changed for a
    running Python process anyway. The fix is to dlopen the libs with
    RTLD_GLOBAL from Python; ORT's TRT provider then finds them via the
    already-resolved symbol table when the session is created.

    Returns True if at least one TRT .so was loaded successfully.
    """
    try:
        import ctypes, os
        import tensorrt  # noqa: F401  — just need its install path
        trt_lib_dir = os.path.join(os.path.dirname(tensorrt.__file__), "lib")
        if not os.path.isdir(trt_lib_dir):
            return False
        loaded = 0
        # Load the core TRT libs in dependency order. Plugin + parser are
        # required for ORT's TRT provider; others are nice-to-have.
        wanted = ("libnvinfer.so", "libnvinfer_plugin.so", "libnvonnxparser.so")
        for fname in sorted(os.listdir(trt_lib_dir)):
            if any(fname.startswith(w) for w in wanted) and ".so." in fname:
                try:
                    ctypes.CDLL(os.path.join(trt_lib_dir, fname),
                                mode=ctypes.RTLD_GLOBAL)
                    loaded += 1
                except OSError:
                    continue
        return loaded > 0
    except ImportError:
        return False
    except Exception:
        return False


# ─── QDQ-format INT8 BERT (raw ORT, the CUDA/TRT-preferred format) ───────────

class _BertCalibReader:
    """Calibration reader for BERT QDQ static quantization.

    Yields pre-tokenized {input_ids, attention_mask, token_type_ids} dicts
    matching the ONNX input signature. Same shape as ResNet18's reader; the
    only difference is the input tensors are int64 token IDs.
    """

    def __init__(self, batches: List[Dict[str, np.ndarray]]):
        self.batches = batches
        self._i = 0

    def get_next(self):
        if self._i >= len(self.batches):
            return None
        item = self.batches[self._i]
        self._i += 1
        return item

    def rewind(self):
        self._i = 0


def _produce_qdq_bert_int8(
    config: Config,
    fp32_dir: Path,
    tokenizer,
    out_dir: Path,
) -> Optional[Path]:
    """Produce a QDQ-format INT8 BERT artifact using raw quantize_static.

    The Optimum `avx512_vnni` config produces QOperator format (preferred
    by CPU EP's VNNI/AMX kernels). This QDQ variant emits fake-quant ops
    wrapping float compute — the format CUDA EP and TensorRT EP actually
    consume as INT8. Same model, same calibration samples, different
    format — that's the point of Table 9.2.                                    #M
    """
    try:
        from onnxruntime.quantization import (
            quantize_static, QuantType, QuantFormat, CalibrationMethod,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
        from datasets import load_dataset
    except ImportError as e:
        print(f"  Skipping QDQ BERT production: {e}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_model = fp32_dir / "model.onnx"
    prep_model = out_dir / "model_prep.onnx"
    out_model = out_dir / "model_quantized.onnx"

    if out_model.exists() and not config.force_export:
        print(f"  Reusing cached QDQ INT8 artifact: {out_model}")
        return out_model

    # Shape inference pre-pass. BERT's attention-mask pattern defeats
    # SymbolicShapeInference at default settings — we try progressively
    # more permissive options, falling back to skipping the pre-pass
    # entirely if symbolic shape inference cannot resolve the graph.      #M2
    model_to_quantize = fp32_model
    for strategy_label, kwargs in [
        ("auto_merge=True",        {"auto_merge": True}),
        ("skip_symbolic_shape",    {"skip_symbolic_shape": True}),
    ]:
        try:
            print(f"  Running quant_pre_process ({strategy_label})...")
            quant_pre_process(str(fp32_model), str(prep_model), **kwargs)
            model_to_quantize = prep_model
            print(f"  Pre-process succeeded → using {prep_model.name}")
            break
        except Exception as e:
            print(f"    Failed ({type(e).__name__}): {e}")
    else:
        print(f"  All pre-process strategies failed; quantizing raw FP32 graph.")

    #M2 auto_merge=True reconciles divergent shape branches (BERT's
    #  attention-mask broadcast is the usual culprit); skip_symbolic_shape
    #  falls back to ONNX shape inference alone; skipping pre-process is
    #  the nuclear option — quantize_static does its own shape-handling.

    # Tokenize calibration samples (same subset Optimum used).
    calib_ds = load_dataset("glue", "sst2",
                            split=f"train[:{config.bert_num_calib}]")
    calib_batches = []
    for ex in calib_ds:
        enc = tokenizer(
            ex["sentence"],
            padding="max_length",
            truncation=True,
            max_length=config.bert_seq_length,
            return_tensors="np",
        )
        item = {k: v.astype(np.int64) for k, v in enc.items()}
        calib_batches.append(item)

    print(f"  Quantizing to QDQ format with MinMax calibration "
          f"({len(calib_batches)} samples)...")
    reader = _BertCalibReader(calib_batches)
    # Restrict to MatMul for the same reason as the QOperator path:
    # quantizing Gather destroys the embedding table. QUInt8 activations
    # because CPU EP's VNNI fast path expects unsigned × signed.
    quantize_static(
        model_input=str(model_to_quantize),
        model_output=str(out_model),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["MatMul"],
    )
    print(f"  Saved QDQ INT8 BERT: {out_model} "
          f"({out_model.stat().st_size / (1024*1024):.1f} MB)")
    return out_model

#M Why QOperator ≠ QDQ matters beyond academic interest: the QOperator
#  format declares its quantization to the runtime via explicit INT8 ops
#  (QLinearMatMul). If the EP has no QLinearMatMul kernel, it must
#  dequantize-compute-requantize via reference paths. The QDQ format
#  declares quantization as a *hint* (QuantizeLinear → float op →
#  DequantizeLinear). The EP can choose to honor the hint with a real
#  INT8 kernel or ignore it and run float — which is why QDQ degrades
#  gracefully across EPs while QOperator does not.


# ─── Artifact path helpers ───────────────────────────────────────────────────

def artifact_path(config: Config, key: str) -> Path:
    """Resolve a cached ONNX artifact path."""
    return config.cache_dir / ARTIFACT_DIRS[key]


def require_artifact(config: Config, key: str, mode_name: str) -> Path:
    """Raise a helpful error if a prerequisite artifact is missing."""
    p = artifact_path(config, key)
    missing = (not p.exists()) or (p.is_dir() and not any(p.iterdir()))
    if missing:
        raise FileNotFoundError(
            f"\n  Missing artifact for mode={mode_name!r}: {p}\n"
            f"  Run first: python {Path(__file__).name} --mode optimum\n"
            f"  (or --mode calibration to produce ResNet-18 artifacts)"
        )
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: optimum — Optimum's HF-task-aware deployment API (Listing 9.1)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_bert_int8_artifacts(config: Config):
    """Export FP32 BERT via Optimum and produce three INT8 variants.

    Returns (paths_dict, tokenizer). Reuses cached artifacts unless
    config.force_export is set. Raises ImportError if optimum/transformers
    are unavailable — the caller reports that to the user.
    """
    from functools import partial
    from transformers import AutoTokenizer
    from optimum.onnxruntime import (
        ORTModelForSequenceClassification, ORTQuantizer,
    )
    from optimum.onnxruntime.configuration import (
        AutoQuantizationConfig, AutoCalibrationConfig,
    )

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    fp32_dir     = artifact_path(config, "bert_fp32")
    int8_dyn_dir = artifact_path(config, "bert_int8_dyn")
    int8_sta_dir = artifact_path(config, "bert_int8_stat")
    int8_qdq_dir = artifact_path(config, "bert_int8_qdq")

    # ── Step 1: Export BERT-base SST-2 to ONNX via Optimum ──                  #B
    if config.force_export or not fp32_dir.exists():
        if fp32_dir.exists():
            shutil.rmtree(fp32_dir)
        print(f"\n  Exporting {BERT_MODEL_ID} to ONNX...")
        model = ORTModelForSequenceClassification.from_pretrained(
            BERT_MODEL_ID, export=True)
        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_ID)
        model.save_pretrained(fp32_dir)
        tokenizer.save_pretrained(fp32_dir)
        del model
        print(f"  Saved FP32 ONNX artifact: {fp32_dir}")
    else:
        print(f"\n  Reusing cached FP32 ONNX artifact: {fp32_dir}")
        tokenizer = AutoTokenizer.from_pretrained(fp32_dir)

    #B `ORTModelForSequenceClassification.from_pretrained(..., export=True)`
    #  is the HF-task-aware abstraction the raw API of §6.4 lacks. It
    #  auto-exports with the correct input signature for sequence
    #  classification (input_ids, attention_mask, token_type_ids). With
    #  the raw API you had to write that signature yourself.

    # ── Step 2: Dynamic INT8 (no calibration data needed) ──
    if config.force_export or not int8_dyn_dir.exists():
        if int8_dyn_dir.exists():
            shutil.rmtree(int8_dyn_dir)
        print(f"\n  Applying DYNAMIC INT8 quantization...")
        quantizer = ORTQuantizer.from_pretrained(fp32_dir)
        qconfig_dyn = AutoQuantizationConfig.avx512_vnni(
            is_static=False, per_channel=False)
        quantizer.quantize(save_dir=int8_dyn_dir,
                           quantization_config=qconfig_dyn)
        print(f"  Saved dynamic INT8 artifact: {int8_dyn_dir}")
    else:
        print(f"\n  Reusing cached dynamic INT8 artifact: {int8_dyn_dir}")

    # ── Step 3: Static INT8 (MinMax, MatMul-only, per-tensor) ──               #C
    if config.force_export or not int8_sta_dir.exists():
        if int8_sta_dir.exists():
            shutil.rmtree(int8_sta_dir)
        print(f"\n  Applying STATIC INT8 quantization "
              f"(MinMax, MatMul-only, per-tensor)...")
        # Three non-obvious choices, each fixing a real failure mode:
        #
        # (a) operators_to_quantize = ["MatMul"] instead of Optimum's
        #     default ["MatMul","Attention","LSTM","Gather","EmbedLayerNorm"].
        #     Gather is how BERT looks up word embeddings (vocab × hidden);
        #     quantizing the embedding table collapses token representations
        #     and destroys downstream accuracy (54-58% on SST-2 = random).
        #     MatMul-only keeps the FFN speedup (where it matters) without
        #     touching embeddings.
        #
        # (b) per_channel=False avoids the LayerNorm rank-1 edge case
        #     ("Axis 1 is out-of-range" warnings → silent fallback).
        #
        # (c) MinMax instead of Entropy/Percentile: histogram-based methods
        #     overflow on BERT because the ONNX export bakes attention-mask
        #     constants of magnitude ~1e4. `np.linspace` produces NaN bin
        #     edges; ORT's `assert scale >= 0` fires. MinMax is histogram-free.
        quantizer = ORTQuantizer.from_pretrained(fp32_dir)
        qconfig_sta = AutoQuantizationConfig.avx512_vnni(
            is_static=True, per_channel=False)
        qconfig_sta.operators_to_quantize = ["MatMul"]

        def preprocess_fn(ex, tokenizer):
            return tokenizer(
                ex["sentence"], padding="max_length",
                truncation=True, max_length=config.bert_seq_length)

        calib_dataset = quantizer.get_calibration_dataset(
            "glue", dataset_config_name="sst2",
            preprocess_function=partial(preprocess_fn, tokenizer=tokenizer),
            num_samples=config.bert_num_calib,
            dataset_split="train",
        )
        calib_config = AutoCalibrationConfig.minmax(calib_dataset)
        ranges = quantizer.fit(
            dataset=calib_dataset, calibration_config=calib_config,
            operators_to_quantize=qconfig_sta.operators_to_quantize,
        )
        quantizer.quantize(
            save_dir=int8_sta_dir, quantization_config=qconfig_sta,
            calibration_tensors_range=ranges,
        )
        print(f"  Saved static INT8 artifact: {int8_sta_dir}")
    else:
        print(f"\n  Reusing cached static INT8 artifact: {int8_sta_dir}")

    #C Dynamic and static produce the same QLinear* operator set but
    #  differ in scale/zero-point provenance: dynamic computes ranges
    #  per-batch at inference; static bakes them in from a calibration
    #  pass. Static is faster but sensitive to calibration data quality
    #  — which the `calibration` mode quantifies on ResNet-18.

    # ── Step 3b: QDQ-format INT8 (raw ORT, for GPU targets) ──                #C2
    print(f"\n  Applying STATIC INT8 quantization (QDQ format, raw ORT)...")
    try:
        _produce_qdq_bert_int8(config, fp32_dir, tokenizer, int8_qdq_dir)
    except Exception as e:
        print(f"  QDQ BERT production FAILED: {type(e).__name__}: {e}")
        print(f"  ep-comparison will show the QOperator row only.")

    #C2 The QDQ artifact is built outside Optimum because Optimum's
    #  AutoQuantizationConfig targets CPU ISAs (avx512_vnni, arm64) and
    #  emits QOperator format. For a GPU-bound graph we go straight to
    #  quantize_static with quant_format=QDQ. Same model, same calibration
    #  samples, different format — the ep-comparison mode shows the delta.

    return {
        "fp32":     fp32_dir,
        "int8_dyn": int8_dyn_dir,
        "int8_sta": int8_sta_dir,
        "int8_qdq": int8_qdq_dir,
    }, tokenizer


def run_optimum(config: Config):
    """
    Listing 9.1: same ONNX artifact §6.4 produced, but built through the
    HF-task-aware abstraction instead of the raw onnxruntime.quantization API.
    """
    print("\n" + "=" * 70)
    print("MODE: optimum — Optimum's HF-task-aware deployment API (Listing 9.1)")
    print("=" * 70)

    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"\n  ERROR: Missing dependency: {e}")
        print(f"  Install: pip install 'optimum[onnxruntime-gpu]' transformers datasets torch")
        return

    ort = get_ort()
    if ort is None:
        print("\n  ERROR: onnxruntime is not installed.")
        return

    try:
        paths, tokenizer = _build_bert_int8_artifacts(config)
    except ImportError as e:
        print(f"\n  ERROR: Missing dependency: {e}")
        print(f"  Install: pip install 'optimum[onnxruntime-gpu]' transformers datasets torch")
        return

    fp32_dir     = paths["fp32"]
    int8_dyn_dir = paths["int8_dyn"]
    int8_sta_dir = paths["int8_sta"]

    # ── Step 4: SST-2 validation accuracy + end-to-end latency ──
    print(f"\n  Loading SST-2 validation set...")
    val_ds = load_dataset("glue", "sst2", split="validation")
    val_ds = val_ds.select(range(min(config.bert_num_eval, len(val_ds))))

    # Cast to plain Python list — newer `datasets` versions return a
    # lazy Column object that the tokenizer rejects.
    texts = list(val_ds["sentence"])
    labels = np.array(list(val_ds["label"]))

    encodings = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=config.bert_seq_length, return_tensors="np",
    )

    # Prefer CPUExecutionProvider for a fair cross-artifact comparison —
    # the three INT8 graphs below all use CPU-targeted ops (avx512_vnni).
    # The ep-comparison mode revisits CUDA/TRT on the optimized INT8 graph.
    provider = "CPUExecutionProvider"

    variants = [
        ("FP32 (Optimum export)",  fp32_dir,     "fp32",     "model.onnx"),
        ("INT8 dynamic (Optimum)", int8_dyn_dir, "int8_dyn", None),
        ("INT8 static (Optimum)",  int8_sta_dir, "int8_sta", None),
    ]

    results = []
    print(f"\n  Running {len(variants)} variants on {provider}")
    print(f"  Batch: {config.bert_batch_size}, Seq: {config.bert_seq_length}, "
          f"Eval samples: {len(texts)}")

    for label, dir_path, tag, explicit_file in variants:
        # Find the ONNX file inside the directory
        if explicit_file:
            onnx_path = dir_path / explicit_file
        else:
            onnx_path = _find_onnx(dir_path, prefer_quantized=True)

        # Create raw ORT session (we want to measure the graph, not Optimum's
        # wrapper overhead, to make numbers comparable across modes).
        sess = _make_session(ort, onnx_path, [provider])

        # Adapt feed to the session's declared inputs (some exports drop
        # token_type_ids; some Optimum builds also drop position_ids).
        input_names = {i.name for i in sess.get_inputs()}
        feed_all = {
            "input_ids":     encodings["input_ids"].astype(np.int64),
            "attention_mask": encodings["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in input_names and "token_type_ids" in encodings:
            feed_all["token_type_ids"] = encodings["token_type_ids"].astype(np.int64)
        feed_all = {k: v for k, v in feed_all.items() if k in input_names}

        # Accuracy over all eval samples
        preds = []
        B = config.bert_batch_size
        for i in range(0, len(texts), B):
            batch_feed = {k: v[i:i+B] for k, v in feed_all.items()}
            logits = sess.run(None, batch_feed)[0]
            preds.append(logits.argmax(axis=-1))
        preds = np.concatenate(preds)[: len(labels)]
        acc = float((preds == labels).mean())

        # Latency on a single representative batch
        bench_feed = {k: v[:config.bert_batch_size] for k, v in feed_all.items()}
        # Pad the last batch if needed
        if bench_feed["input_ids"].shape[0] < config.bert_batch_size:
            reps = config.bert_batch_size // bench_feed["input_ids"].shape[0] + 1
            bench_feed = {k: np.tile(v, (reps, 1))[:config.bert_batch_size]
                          for k, v in bench_feed.items()}

        mean_ms, std_ms = time_session(
            sess, bench_feed,
            num_warmup=config.bert_num_warmup,
            num_iters=config.bert_num_iters)

        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        results.append({
            "label": label, "tag": tag, "acc": acc,
            "mean_ms": mean_ms, "std_ms": std_ms, "size_mb": size_mb,
        })
        print(f"    {label:<30} acc={acc:.4f}  "
              f"latency={mean_ms:6.1f} ± {std_ms:4.1f} ms  "
              f"size={size_mb:6.1f} MB")

    # ── Summary table (source for Table 9.1 in prose) ──
    baseline = results[0]
    print(f"\n  ─── Listing 9.1 summary: same artifact, three build paths ───")
    header = (f"  {'Variant':<32} {'Accuracy':<10} {'Latency (ms)':<15} "
              f"{'Size (MB)':<10} {'Speedup':<8}")
    print(header)
    print(f"  {'─' * len(header)}")
    for r in results:
        speedup = baseline["mean_ms"] / r["mean_ms"]
        print(f"  {r['label']:<32} {r['acc']:<10.4f} "
              f"{r['mean_ms']:>6.1f} ± {r['std_ms']:<5.1f} "
              f"{r['size_mb']:<10.1f} {speedup:<8.2f}×")

    # ── Figure: FP32 vs dynamic vs static latency + accuracy ──
    apply_manning_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 3.0))

    tags = [r["tag"] for r in results]
    lats = [r["mean_ms"] for r in results]
    errs = [r["std_ms"] for r in results]
    accs = [r["acc"] for r in results]
    x_labels = ["FP32", "INT8\ndynamic", "INT8\nstatic"]

    for i, (tag, lat, err) in enumerate(zip(tags, lats, errs)):
        ax1.bar(i, lat, yerr=err, color=COLORS[tag], hatch=HATCHES[tag],
                edgecolor="black", linewidth=0.5, width=0.7,
                error_kw={"linewidth": 0.8, "capsize": 3})
        ax1.text(i, lat + err + max(lats) * 0.02, f"{lat:.1f}",
                 ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title(f"BERT-base SST-2, batch={config.bert_batch_size}, "
                  f"seq={config.bert_seq_length}\n({provider})")

    for i, (tag, acc) in enumerate(zip(tags, accs)):
        ax2.bar(i, acc, color=COLORS[tag], hatch=HATCHES[tag],
                edgecolor="black", linewidth=0.5, width=0.7)
        ax2.text(i, acc + 0.005, f"{acc:.4f}",
                 ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(range(len(x_labels)))
    ax2.set_xticklabels(x_labels)
    ax2.set_ylabel("SST-2 validation accuracy")
    ax2.set_ylim(0.85, 1.0)
    ax2.set_title("Accuracy (SST-2 dev)")

    fig.tight_layout()
    save_or_show(fig, "CH09_F02_Kalyanarangan_optimum_deployment", config)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: transformer-optimizer — attention fusion / LayerNorm folding
# ═══════════════════════════════════════════════════════════════════════════════

def run_transformer_optimizer(config: Config):
    """
    Subsection 3 of §9.2: the transformer-aware optimizer that §6.4 deferred.
    `onnxruntime.transformers.optimizer.optimize_model(model_type='bert')`
    fuses Q/K/V projections and attention math into a single `Attention`
    op, folds LayerNorm patterns into `SkipLayerNormalization`, and rewrites
    GELU into `FastGelu`. Before/after latency on BERT-base.
    """
    print("\n" + "=" * 70)
    print("MODE: transformer-optimizer — attention fusion + LayerNorm folding")
    print("=" * 70)

    ort = get_ort()
    if ort is None:
        print("\n  ERROR: onnxruntime is not installed.")
        return

    try:
        from onnxruntime.transformers import optimizer as trt_optimizer
        from transformers import AutoTokenizer
        import onnx
    except ImportError as e:
        print(f"\n  ERROR: {e}. Install transformers, onnx, onnxruntime.")
        return

    # Prerequisite artifact
    fp32_dir = require_artifact(config, "bert_fp32", "transformer-optimizer")
    fp32_model = fp32_dir / "model.onnx"

    opt_dir = artifact_path(config, "bert_fp32_opt")
    opt_dir.mkdir(parents=True, exist_ok=True)
    opt_model = opt_dir / "model.onnx"

    # ── Apply the transformer-aware optimizer ──                                #D
    if config.force_export or not opt_model.exists():
        print(f"\n  Running optimize_model(model_type='bert', "
              f"num_heads={BERT_NUM_HEADS}, hidden_size={BERT_HIDDEN_SIZE})")
        optimized = trt_optimizer.optimize_model(
            str(fp32_model),
            model_type="bert",
            num_heads=BERT_NUM_HEADS,
            hidden_size=BERT_HIDDEN_SIZE,
            opt_level=99,
            use_gpu=(resolve_device(config) == "cuda"),
        )
        optimized.save_model_to_file(str(opt_model))
        print(f"  Saved optimized model: {opt_model}")
    else:
        print(f"\n  Reusing cached optimized model: {opt_model}")

    #D `optimize_model` looks for specific BERT subgraph patterns and
    #  replaces them with fused ops. `num_heads` and `hidden_size` are
    #  NOT auto-detected — you must pass them correctly or the pattern
    #  matcher silently misses fusions. For BERT-base the answers are
    #  always 12 and 768; for distilled / larger variants they differ.

    # ── Node-type diff ──                                                      #E
    before = onnx.load(str(fp32_model))
    after = onnx.load(str(opt_model))
    before_ops = sorted({n.op_type for n in before.graph.node})
    after_ops = sorted({n.op_type for n in after.graph.node})

    from collections import Counter
    before_ct = Counter(n.op_type for n in before.graph.node)
    after_ct = Counter(n.op_type for n in after.graph.node)

    print(f"\n  Graph node counts:")
    print(f"    Before optimize_model: {sum(before_ct.values())} nodes")
    print(f"    After  optimize_model: {sum(after_ct.values())} nodes  "
          f"(Δ = {sum(after_ct.values()) - sum(before_ct.values())})")

    fused_ops = sorted(set(after_ops) - set(before_ops))
    removed_ops = sorted(set(before_ops) - set(after_ops))
    print(f"\n  New fused ops:        {fused_ops or '(none — already fused?)'}")
    print(f"  Removed primitive ops: {removed_ops or '(none)'}")

    #E The canonical BERT fusion signatures are:
    #    Attention           ← replaces MatMul×3 + Transpose + Softmax + MatMul
    #    SkipLayerNormalization  ← replaces Add + ReduceMean + Sub + Pow + …
    #    FastGelu / BiasGelu ← replaces the 7-op GELU approximation
    #    EmbedLayerNormalization ← token + position + segment embed + LN
    #  Their presence in `after_ops` is the sanity-check that optimize_model
    #  actually did work. If `fused_ops` comes back empty, the pattern
    #  matcher failed — usually a num_heads / hidden_size mismatch.

    # ── Benchmark: FP32 before vs FP32 after ──
    device = resolve_device(config)
    if device == "cuda" and "CUDAExecutionProvider" in available_execution_providers(ort):
        provider = "CUDAExecutionProvider"
    else:
        provider = "CPUExecutionProvider"

    print(f"\n  Benchmarking on {provider}, "
          f"batch={config.bert_batch_size}, seq={config.bert_seq_length}")

    B, L = config.bert_batch_size, config.bert_seq_length
    feed_all = _bert_random_feed(B, L, seed=config.seed)

    results = []
    for label, path, tag in [
        ("Before (raw export)",      fp32_model, "before"),
        ("After optimize_model()",   opt_model,  "after"),
    ]:
        # Turn OFF ORT's own basic fusions so we isolate what the TRANSFORMER
        # optimizer did. Otherwise the generic graph optimizer fuses some of
        # the same patterns and hides the delta.
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(str(path), sess_options=so,
                                    providers=[provider])
        input_names = {i.name for i in sess.get_inputs()}
        feed = {k: v for k, v in feed_all.items() if k in input_names}

        mean_ms, std_ms = time_session(
            sess, feed, config.bert_num_warmup, config.bert_num_iters)
        results.append({"label": label, "tag": tag,
                        "mean_ms": mean_ms, "std_ms": std_ms})
        print(f"    {label:<30} {mean_ms:6.2f} ± {std_ms:4.2f} ms")

    speedup = results[0]["mean_ms"] / results[1]["mean_ms"]
    print(f"\n  Transformer-aware optimizer speedup: {speedup:.2f}×")

    # ── Figure ──
    apply_manning_style()
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    for i, r in enumerate(results):
        ax.bar(i, r["mean_ms"], yerr=r["std_ms"],
               color=COLORS[r["tag"]], hatch=HATCHES[r["tag"]],
               edgecolor="black", linewidth=0.5, width=0.55,
               error_kw={"linewidth": 0.8, "capsize": 3})
        ax.text(i, r["mean_ms"] + r["std_ms"] + max(x["mean_ms"] for x in results) * 0.02,
                f"{r['mean_ms']:.1f} ms", ha="center", va="bottom", fontsize=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before\n(raw export)", "After\noptimize_model()"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Transformer-aware optimizer: {speedup:.2f}× on BERT-base\n"
                 f"({provider}, batch={B}, seq={L})")
    fig.tight_layout()
    save_or_show(fig, "CH09_F05_Kalyanarangan_transformer_optimizer", config)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: ep-comparison — one graph, N runtimes (Figure 9.4, Table 9.2)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ep_comparison(config: Config):
    """
    Subsection 4 of §9.2: the format × EP matrix that Table 9.2 captures.

    INT8 QOperator (Optimum avx512_vnni, from §6.4 lineage) and INT8 QDQ
    (raw quantize_static) are the same model, same calibration samples,
    same weight values — they differ only in how the quantization is
    expressed in the ONNX graph. That expression determines which EPs
    can dispatch to real INT8 kernels vs. fall back to reference paths.
    Running both through CPU EP, CUDA EP, and (when available) TRT EP
    gives the 2×N matrix that makes format–EP matching concrete.

    DirectML and CoreML are named but not benchmarked (platform-dependent).
    """
    print("\n" + "=" * 70)
    print("MODE: ep-comparison — format × EP matrix (Figure 9.4, Table 9.2)")
    print("=" * 70)

    ort = get_ort()
    if ort is None:
        print("\n  ERROR: onnxruntime is not installed.")
        return

    # ── Resolve the two INT8 artifacts ──
    qop_dir = require_artifact(config, "bert_int8_stat", "ep-comparison")
    qop_model = _find_onnx(qop_dir, prefer_quantized=True)

    # QDQ artifact may not exist if `optimum` mode was run with an older
    # version of this script; handle gracefully.
    qdq_dir = artifact_path(config, "bert_int8_qdq")
    qdq_candidate = qdq_dir / "model_quantized.onnx"
    if not qdq_candidate.exists():
        print(f"\n  WARN: QDQ INT8 artifact not found at {qdq_candidate}")
        print(f"  Re-run: python {Path(__file__).name} --mode optimum")
        print(f"  Continuing with QOperator artifact only.")
        formats = [("QOperator (avx512_vnni)", qop_model, "qop")]
    else:
        formats = [
            ("QOperator (avx512_vnni)", qop_model,      "qop"),
            ("QDQ (raw quantize_static)", qdq_candidate, "qdq"),
        ]

    print(f"\n  Formats to test:")
    for lbl, path, _ in formats:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"    - {lbl:<26} {path.name}  ({size_mb:.1f} MB)")

    # ── Build feed ──
    try:
        from transformers import AutoTokenizer
        fp32_dir = artifact_path(config, "bert_fp32")
        tokenizer = AutoTokenizer.from_pretrained(
            fp32_dir if fp32_dir.exists() else BERT_MODEL_ID)
    except Exception as e:
        print(f"  WARN: tokenizer load failed ({e}); using random ids")
        tokenizer = None

    B, L = config.bert_batch_size, config.bert_seq_length
    if tokenizer is not None:
        sample_texts = [
            "the cinematography is stunning and the acting unforgettable ."
        ] * B
        enc = tokenizer(sample_texts, padding="max_length",
                        truncation=True, max_length=L, return_tensors="np")
        feed_all = {k: enc[k].astype(np.int64) for k in enc}
    else:
        feed_all = _bert_random_feed(B, L, seed=config.seed)

    # ── EP roster ──                                                           #F
    ep_available = available_execution_providers(ort)
    system = platform.system()

    ep_plan = [("CPU EP", ["CPUExecutionProvider"], "cpu_ep")]
    if ("CUDAExecutionProvider" in ep_available
            and get_torch_cuda_available()
            and resolve_device(config) == "cuda"):
        ep_plan.append(
            ("CUDA EP", ["CUDAExecutionProvider", "CPUExecutionProvider"],
             "cuda_ep"))

    # TRT EP is advertised whenever ORT was compiled with TRT support, but
    # the actual libnvinfer.so.10 must be loadable. Probe first; if missing,
    # try preloading from the `tensorrt` pip package (if installed). Only
    # add TRT to the roster if both conditions are satisfied — this avoids
    # the ugly "Falling back to CUDA EP" error block on every run.
    trt_enabled = False
    if ("TensorrtExecutionProvider" in ep_available
            and resolve_device(config) == "cuda"):
        if _trt_libs_loadable():
            trt_enabled = True
        elif _try_preload_trt_libs() and _trt_libs_loadable():
            print(f"\n  TensorRT libs preloaded from `tensorrt` pip package.")
            trt_enabled = True
        else:
            print(f"\n  TensorRT EP: libs not loadable on this runtime → skipped.")
            print(f"  To enable TRT benchmarks, install the TRT wheels once:")
            print(f"    pip install --extra-index-url "
                  f"https://pypi.nvidia.com tensorrt-cu12==10.3.0")
            print(f"  Then re-run this mode — the script will auto-preload them.")

    if trt_enabled:
        trt_opts = {
            "trt_fp16_enable": False,
            "trt_int8_enable": True,
            "trt_max_workspace_size": 2 * (1 << 30),   # 2 GB
        }
        ep_plan.append(
            ("TensorRT EP",
             [("TensorrtExecutionProvider", trt_opts),
              "CUDAExecutionProvider",
              "CPUExecutionProvider"],
             "trt_ep"))

    mentioned_only = []
    if system == "Darwin":
        mentioned_only.append(("CoreML EP",
            "CoreMLExecutionProvider" in ep_available))
    if system == "Windows":
        mentioned_only.append(("DirectML EP",
            "DmlExecutionProvider" in ep_available))

    print(f"\n  EPs to benchmark:")
    for name, _, _ in ep_plan:
        print(f"    - {name}")
    if mentioned_only:
        print(f"  EPs mentioned (not benchmarked on this platform):")
        for name, present in mentioned_only:
            print(f"    - {name}  ({'available' if present else 'not compiled in'})")

    #F QOperator and QDQ are the two static-quant formats ONNX Runtime
    #  emits. QOperator uses explicit integer ops (QLinearMatMul,
    #  QLinearConv); QDQ uses QuantizeLinear/DequantizeLinear nodes
    #  wrapping the original float op and leaves it to the EP to decide
    #  whether to execute in INT8 or float. Consequence:
    #    • CPU EP's AVX512-VNNI kernels are INT8-native and match
    #      QOperator's explicit QLinearMatMul. Great fit.
    #    • CUDA EP has limited QOperator kernel coverage; it's much
    #      happier with QDQ, which it can either honor (rare) or fall
    #      through as float (graceful degradation).
    #    • TensorRT EP *only* understands QDQ — it folds the QDQ nodes
    #      into real INT8 cubins on engine build.
    #  Table 9.2 makes this concrete. §9.5 goes deeper on format
    #  correctness and dtype verification.

    # ── Run the format × EP grid ──
    matrix: Dict[Tuple[str, str], Dict] = {}
    for fmt_label, fmt_path, fmt_tag in formats:
        for ep_name, ep_providers, ep_tag in ep_plan:
            cell_key = (fmt_tag, ep_tag)
            print(f"\n  Running: {fmt_label}  on  {ep_name}")
            try:
                sess = _make_session(ort, fmt_path, ep_providers)

                actual_providers = sess.get_providers()
                print(f"    Actual providers: {actual_providers}")

                # TRT EP silent fallback detection.
                if ep_tag == "trt_ep" and "TensorrtExecutionProvider" not in actual_providers:
                    print(f"    TRT EP did not load — fell back to CUDA EP.")
                    print(f"    Marking cell as unavailable.")
                    matrix[cell_key] = {
                        "fmt": fmt_label, "ep": ep_name,
                        "mean_ms": float("nan"), "std_ms": 0.0,
                        "error": "TRT not loaded",
                    }
                    del sess
                    continue

                input_names = {i.name for i in sess.get_inputs()}
                feed = {k: v for k, v in feed_all.items() if k in input_names}

                warmup = 20 if ep_tag == "trt_ep" else config.bert_num_warmup
                mean_ms, std_ms = time_session(
                    sess, feed, warmup, config.bert_num_iters)

                matrix[cell_key] = {
                    "fmt": fmt_label, "ep": ep_name,
                    "mean_ms": mean_ms, "std_ms": std_ms,
                }
                print(f"    Latency: {mean_ms:.2f} ± {std_ms:.2f} ms")

            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                matrix[cell_key] = {
                    "fmt": fmt_label, "ep": ep_name,
                    "mean_ms": float("nan"), "std_ms": 0.0,
                    "error": str(e),
                }

    # ── Summary table (Table 9.2 source) ──                                    #G
    print(f"\n  ─── Table 9.2 data (measured): format × EP matrix ───")
    ep_tags = [t for _, _, t in ep_plan]
    ep_names = [n for n, _, _ in ep_plan]

    col_width = 17
    header = f"  {'Format':<28}" + "".join(f"{n:<{col_width}}" for n in ep_names)
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for fmt_label, _, fmt_tag in formats:
        row = f"  {fmt_label:<28}"
        for ep_tag in ep_tags:
            cell = matrix.get((fmt_tag, ep_tag))
            if cell is None or cell["mean_ms"] != cell["mean_ms"]:
                row += f"{'N/A':<{col_width}}"
            else:
                cell_str = f"{cell['mean_ms']:.2f} ± {cell['std_ms']:.2f}"
                row += f"{cell_str:<{col_width}}"
        print(row)

    # Best-cell-per-EP reader hint
    print(f"\n  Best format per EP (lower is better):")
    for ep_tag, ep_name in zip(ep_tags, ep_names):
        valid = [(fmt_tag, matrix[(fmt_tag, ep_tag)]["mean_ms"])
                 for _, _, fmt_tag in formats
                 if matrix.get((fmt_tag, ep_tag))
                 and matrix[(fmt_tag, ep_tag)]["mean_ms"] ==
                     matrix[(fmt_tag, ep_tag)]["mean_ms"]]
        if not valid:
            continue
        best_fmt, best_ms = min(valid, key=lambda x: x[1])
        best_label = next(f for f, _, t in formats if t == best_fmt)
        print(f"    {ep_name:<12} → {best_label}  ({best_ms:.2f} ms)")

    #G Table 9.2 reading guide for the prose:
    #    • On-diagonal cells (QOperator-CPU, QDQ-CUDA) should be the
    #      fastest in their column — format matches EP.
    #    • Off-diagonal cells (QOperator-CUDA) are the "mismatch tax."
    #      On L4 the CUDA EP with QOperator is not just slow — it's
    #      slower than FP32 on CUDA (see transformer-optimizer mode).
    #      That's the teaching moment: INT8 is not free on GPUs, and
    #      format selection is not an afterthought.
    #    • TRT EP column, when available, collapses QDQ to near-optimal
    #      INT8. Without TRT on the target, QDQ on CUDA EP is the best
    #      you can do with INT8 — and often FP16 is a better choice.

    # ── Figure 9.4: grouped bar chart ──
    apply_manning_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    bar_width = 0.35
    x = np.arange(len(ep_names))
    fmt_colors = {"qop": COLORS["int8_sta"], "qdq": COLORS["int8_opt"]}
    fmt_hatches = {"qop": HATCHES["int8_sta"], "qdq": HATCHES["int8_opt"]}
    fmt_legend = {"qop": "QOperator", "qdq": "QDQ"}

    for fi, (fmt_label, _, fmt_tag) in enumerate(formats):
        means, stds = [], []
        for ep_tag in ep_tags:
            cell = matrix.get((fmt_tag, ep_tag))
            if cell and cell["mean_ms"] == cell["mean_ms"]:
                means.append(cell["mean_ms"])
                stds.append(cell["std_ms"])
            else:
                means.append(0)
                stds.append(0)
        offset = (fi - (len(formats) - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width,
                      yerr=stds, color=fmt_colors[fmt_tag],
                      hatch=fmt_hatches[fmt_tag],
                      edgecolor="black", linewidth=0.5,
                      label=fmt_legend[fmt_tag],
                      error_kw={"linewidth": 0.8, "capsize": 2})
        for i, (m, s) in enumerate(zip(means, stds)):
            if m > 0:
                ax.text(x[i] + offset, m + s + max(means) * 0.02,
                        f"{m:.1f}", ha="center", va="bottom", fontsize=6)
            else:
                ax.text(x[i] + offset, max(means) * 0.05,
                        "N/A", ha="center", va="bottom", fontsize=6,
                        color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(ep_names)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Format × EP on BERT-base INT8  "
                 f"(batch={B}, seq={L})")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    save_or_show(fig, "CH09_F04_Kalyanarangan_ep_comparison", config)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: calibration — MinMax vs Entropy vs Percentile (Table 9.3)
# ═══════════════════════════════════════════════════════════════════════════════

def _export_resnet18(config: Config, onnx_path: Path):
    """Export torchvision ResNet-18 to ONNX. Matches the §6.4 baseline."""
    import torch
    import torchvision.models as tv_models

    print(f"  Exporting torchvision ResNet-18 (ImageNet) to ONNX...")
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    dummy = torch.randn(1, 3, config.resnet_input_size, config.resnet_input_size)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # `dynamo=False` forces the legacy TorchScript-based exporter, which
    # doesn't require the `onnxscript` package. Torch 2.6+ defaults to
    # the new dynamo exporter; for reproducibility in the book we pin
    # to the older, more thoroughly-tested path.
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Saved FP32 ResNet-18: {onnx_path} "
          f"({onnx_path.stat().st_size / (1024*1024):.1f} MB)")


class _ResNetCalibReader:
    """Minimal CalibrationDataReader for ORT static quantization.

    Feeds a fixed number of preprocessed ImageNet-normalized tensors.
    ORT iterates this via `get_next()` until it returns None.                  #G
    """

    def __init__(self, samples: List[np.ndarray], input_name: str = "input"):
        self.samples = samples
        self.input_name = input_name
        self._i = 0

    def get_next(self):
        if self._i >= len(self.samples):
            return None
        item = {self.input_name: self.samples[self._i]}
        self._i += 1
        return item

    def rewind(self):
        self._i = 0

#G The calibration pass runs the FP32 graph with ReduceMin/ReduceMax
#  instrumentation on every activation tensor, then picks a quantization
#  range from those statistics. MinMax uses the observed [min, max] as-is;
#  Entropy (KL-divergence) picks the range that minimizes the KL distance
#  between float and quantized activation histograms; Percentile clips at
#  the 99.9th percentile, ignoring outliers. All three see the same samples
#  — the only difference is the statistic used to pick the range.


def _build_calib_samples(config: Config, num_samples: int) -> List[np.ndarray]:
    """Build calibration/eval tensors.

    Tries three sources in order:
      1. torchvision.datasets.Imagenette (parquet-native, most reliable)
      2. HF frgfm/imagenette with trust_remote_code=True (script loader)
      3. ImageNet-normalized Gaussian noise (last resort)

    The fallback ladder matters: newer `datasets` versions (>=3.0) refuse
    to execute script-based loaders by default, breaking path 2 without
    trust_remote_code. torchvision.datasets.Imagenette (added in 0.17)
    downloads directly from fastai's AWS bucket, no script execution.      #H
    """
    size = config.resnet_input_size
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def _preprocess(pil_img):
        import PIL.Image
        img = pil_img.convert("RGB").resize((size, size), PIL.Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = (arr - mean) / std
        return arr[np.newaxis].astype(np.float32)

    # ── Path 1: torchvision's built-in Imagenette loader ──
    try:
        from torchvision.datasets import Imagenette
        print(f"  Loading Imagenette via torchvision ({num_samples} samples)...")
        ds_root = config.cache_dir / "imagenette_tv"
        ds_root.mkdir(parents=True, exist_ok=True)
        try:
            ds = Imagenette(root=str(ds_root), size="160px",
                            split="val", download=True)
        except RuntimeError:
            # "Dataset already downloaded" or similar — retry without download
            ds = Imagenette(root=str(ds_root), size="160px",
                            split="val", download=False)
        samples = []
        for i in range(min(num_samples, len(ds))):
            img, _label = ds[i]
            samples.append(_preprocess(img))
        if len(samples) >= num_samples:
            print(f"  Loaded {len(samples)} real Imagenette samples.")
            return samples
        else:
            print(f"  Only {len(samples)} samples available — trying next path.")
    except ImportError:
        print(f"  torchvision.datasets.Imagenette unavailable "
              f"(needs torchvision >= 0.17)")
    except Exception as e:
        print(f"  torchvision Imagenette failed: {type(e).__name__}: {e}")

    # ── Path 2: HF frgfm/imagenette with trust_remote_code=True ──
    try:
        from datasets import load_dataset
        print(f"  Falling back to HF frgfm/imagenette "
              f"(trust_remote_code=True)...")
        ds = load_dataset("frgfm/imagenette", "160px",
                          split=f"validation[:{num_samples}]",
                          trust_remote_code=True)
        samples = [_preprocess(row["image"]) for row in ds]
        print(f"  Loaded {len(samples)} real Imagenette samples (HF).")
        return samples
    except Exception as e:
        print(f"  HF imagenette failed: {type(e).__name__}: {e}")

    # ── Path 3: Gaussian noise fallback ──
    print(f"  ⚠ FALLBACK: ImageNet-normalized Gaussian noise.")
    print(f"  ⚠ Calibration-method deltas WILL COLLAPSE — noise has no")
    print(f"    outliers, which is exactly what separates MinMax/Entropy/")
    print(f"    Percentile. Install torchvision >= 0.17 for real results.")
    rng = np.random.default_rng(config.seed)
    return [rng.standard_normal((1, 3, size, size)).astype(np.float32)
            for _ in range(num_samples)]

#H Why three paths and not just one: imagenette is a public, tiny
#  (10-class, ~150 MB), ImageNet-lineage dataset that works everywhere
#  in principle — but `datasets` changed its security posture in the
#  3.x line to refuse script-based loaders without trust_remote_code=True,
#  and torchvision's Imagenette requires >= 0.17. Colab regularly has
#  one or the other out of sync. The fallback ladder is defensive,
#  not pretty.


def run_calibration(config: Config):
    """
    Subsection 5 of §9.2: calibration method as a lever. MinMax vs Entropy
    vs Percentile on a ResNet-18 static INT8 artifact, tying back to Ch4's
    finding that MSE-optimal ranges outperform naive MinMax.
    """
    print("\n" + "=" * 70)
    print("MODE: calibration — MinMax vs Entropy vs Percentile (Table 9.3)")
    print("=" * 70)

    ort = get_ort()
    if ort is None:
        print("\n  ERROR: onnxruntime is not installed.")
        return

    try:
        from onnxruntime.quantization import (
            quantize_static,
            QuantType,
            QuantFormat,
            CalibrationMethod,
        )
        import torch
        import torchvision
    except ImportError as e:
        print(f"\n  ERROR: {e}")
        print(f"  Install: pip install torch torchvision onnxruntime pillow datasets")
        return

    config.cache_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = artifact_path(config, "resnet18_fp32")

    # ── Step 1: Export FP32 ResNet-18 if needed ──
    if config.force_export or not fp32_path.exists():
        _export_resnet18(config, fp32_path)
    else:
        print(f"\n  Reusing cached FP32 ResNet-18: {fp32_path}")

    # ── Step 2: Build calibration + eval samples ──
    total_needed = config.resnet_num_calib + config.resnet_num_eval
    all_samples = _build_calib_samples(config, total_needed)
    calib_samples = all_samples[:config.resnet_num_calib]
    eval_samples = all_samples[config.resnet_num_calib:]
    print(f"  Calibration samples: {len(calib_samples)}")
    print(f"  Evaluation samples:  {len(eval_samples)}")

    # ── Step 3: Quantize with three calibration methods ──                     #I
    methods = [
        ("MinMax",     CalibrationMethod.MinMax,     "resnet18_mm",  "minmax"),
        ("Entropy",    CalibrationMethod.Entropy,    "resnet18_ent", "entropy"),
        ("Percentile", CalibrationMethod.Percentile, "resnet18_pct", "percentile"),
    ]

    # `pre_process` folds BatchNorm into Conv, removes identity nodes, and
    # runs shape inference — required for static quantization to succeed on
    # most torchvision exports. Without it, quantize_static raises on
    # unresolved shapes at the first Conv.
    from onnxruntime.quantization.shape_inference import quant_pre_process
    preprocessed_path = fp32_path.parent / (fp32_path.stem + "_prep.onnx")
    if config.force_export or not preprocessed_path.exists():
        print(f"\n  Running quant_pre_process (BN-fold + shape inference)...")
        quant_pre_process(str(fp32_path), str(preprocessed_path))
    else:
        print(f"\n  Reusing preprocessed graph: {preprocessed_path}")

    for name, method, art_key, tag in methods:
        out_path = artifact_path(config, art_key)
        if config.force_export or not out_path.exists():
            print(f"\n  Quantizing with CalibrationMethod.{name}...")
            reader = _ResNetCalibReader(calib_samples, input_name="input")
            extra_options = {}
            if method == CalibrationMethod.Percentile:
                extra_options = {"CalibPercentile": 99.999}
            quantize_static(
                model_input=str(preprocessed_path),
                model_output=str(out_path),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                calibrate_method=method,
                extra_options=extra_options,
            )
            print(f"    Saved: {out_path} "
                  f"({out_path.stat().st_size / (1024*1024):.2f} MB)")
        else:
            print(f"\n  Reusing cached: {out_path}")

    #I QDQ format is chosen over QOperator because this ResNet-18 artifact
    #  is the same one we route through CUDA / TensorRT EPs in other modes
    #  — and TRT strongly prefers QDQ. If you only targeted CPU, QOperator
    #  would be marginally faster. §9.5 gets into the format trade-offs.

    # ── Step 4: Compute agreement rate on eval set ──                          #J
    provider = "CPUExecutionProvider"
    sessions = {}
    for name, _, art_key, tag in methods:
        sessions[tag] = ort.InferenceSession(
            str(artifact_path(config, art_key)), providers=[provider])
    fp32_sess = ort.InferenceSession(str(preprocessed_path), providers=[provider])

    print(f"\n  Computing FP32-vs-INT8 top-1 agreement on {len(eval_samples)} samples...")

    fp32_preds, fp32_top5 = [], []
    for s in eval_samples:
        logits = fp32_sess.run(None, {"input": s})[0][0]
        fp32_preds.append(int(logits.argmax()))
        fp32_top5.append(set(np.argsort(-logits)[:5].tolist()))

    method_results = {}
    for name, _, art_key, tag in methods:
        sess = sessions[tag]
        top1_match = 0
        top5_contain = 0
        for i, s in enumerate(eval_samples):
            logits = sess.run(None, {"input": s})[0][0]
            pred = int(logits.argmax())
            if pred == fp32_preds[i]:
                top1_match += 1
            if pred in fp32_top5[i]:
                top5_contain += 1
        top1_rate = top1_match / len(eval_samples)
        top5_rate = top5_contain / len(eval_samples)

        # Latency on the eval set (first 50 samples, averaged)
        t0 = time.perf_counter()
        for s in eval_samples[:50]:
            sess.run(None, {"input": s})
        ms_per_image = (time.perf_counter() - t0) * 1000 / 50

        # Size
        size_mb = artifact_path(config, art_key).stat().st_size / (1024 * 1024)

        method_results[tag] = {
            "name": name,
            "top1_agreement": top1_rate,
            "top5_containment": top5_rate,
            "ms_per_image": ms_per_image,
            "size_mb": size_mb,
        }
        print(f"    {name:<12} top-1 agreement: {top1_rate:.4f}   "
              f"top-5 containment: {top5_rate:.4f}   "
              f"{ms_per_image:.2f} ms/img")

    #J Agreement rate avoids needing ImageNet labels (which require HF auth
    #  or the 150GB raw ILSVRC tarball). The question "what fraction of
    #  FP32 decisions does this INT8 artifact preserve?" is the right one
    #  for calibration quality anyway — accuracy drop vs ground truth
    #  conflates quantization error with the model's own errors.

    # ── Summary (Table 9.3 source) ──
    print(f"\n  ─── Table 9.3 data (measured) ───")
    hdr = (f"  {'Calibration':<14} {'Top-1 Agr.':<12} {'Top-5 Cont.':<13} "
           f"{'ms/img':<10} {'Size MB':<10}")
    print(hdr)
    print(f"  {'─' * len(hdr)}")
    for tag in ["minmax", "entropy", "percentile"]:
        r = method_results[tag]
        print(f"  {r['name']:<14} {r['top1_agreement']:<12.4f} "
              f"{r['top5_containment']:<13.4f} "
              f"{r['ms_per_image']:<10.2f} {r['size_mb']:<10.2f}")

    # ── Figure ──
    apply_manning_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 3.0))

    tags = ["minmax", "entropy", "percentile"]
    labels = ["MinMax", "Entropy\n(KL)", "Percentile\n(99.999)"]
    agr = [method_results[t]["top1_agreement"] for t in tags]
    lat = [method_results[t]["ms_per_image"] for t in tags]

    for i, (tag, v) in enumerate(zip(tags, agr)):
        ax1.bar(i, v, color=COLORS[tag], hatch=HATCHES[tag],
                edgecolor="black", linewidth=0.5, width=0.65)
        ax1.text(i, v + 0.005, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Top-1 FP32 agreement rate")
    ax1.set_ylim(0.5, 1.02)
    ax1.set_title("Calibration method → accuracy")

    for i, (tag, v) in enumerate(zip(tags, lat)):
        ax2.bar(i, v, color=COLORS[tag], hatch=HATCHES[tag],
                edgecolor="black", linewidth=0.5, width=0.65)
        ax2.text(i, v + max(lat) * 0.01, f"{v:.2f}",
                 ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("ms / image")
    ax2.set_title("Inference latency (CPU EP)")

    fig.suptitle(f"ResNet-18 INT8: same graph, three calibration methods",
                 fontsize=9, y=1.02)
    fig.tight_layout()
    save_or_show(fig, "CH09_F07_Kalyanarangan_calibration_methods", config)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: iobinding — H2D/D2H elimination + session options (Listing 9.2)
# ═══════════════════════════════════════════════════════════════════════════════

def run_iobinding(config: Config):
    """
    Subsection 6 of §9.2: io_binding and session options.

    Compares three configurations on the same INT8 BERT graph, CUDA EP:
      (1) Baseline:   host-side numpy input, host-side output (implicit H2D/D2H)
      (2) io_binding: device-resident input + pre-allocated device output
      (3) io_binding + session options tuned (ORT_ENABLE_ALL, thread counts)

    CPU-only runs fall back to a degraded demo that just shows the session
    options effect (no H2D/D2H to eliminate).
    """
    print("\n" + "=" * 70)
    print("MODE: iobinding — H2D/D2H elimination + session options (Listing 9.2)")
    print("=" * 70)

    ort = get_ort()
    if ort is None:
        print("\n  ERROR: onnxruntime is not installed.")
        return

    # Graph selection priority: FP32-optimized is the cleanest CUDA EP
    # dispatch (tensor-core compute, ~26 internal memcpys). INT8 formats
    # on CUDA EP without TRT are known production anti-patterns — they
    # add internal memcpys that mask iobinding's boundary-copy benefit.
    int8_model = None
    chosen_label = None
    fp32_opt_candidate = artifact_path(config, "bert_fp32_opt") / "model.onnx"
    qdq_candidate = artifact_path(config, "bert_int8_qdq") / "model_quantized.onnx"

    if fp32_opt_candidate.exists():
        int8_model = fp32_opt_candidate
        chosen_label = "FP32 optimized — tensor-core compute, clean dispatch"
    elif qdq_candidate.exists():
        int8_model = qdq_candidate
        chosen_label = "INT8 QDQ — internal memcpys will mask some benefit"
    else:
        int8_dir = require_artifact(config, "bert_int8_stat", "iobinding")
        int8_model = _find_onnx(int8_dir, prefer_quantized=True)
        chosen_label = ("INT8 QOperator — NOT recommended for iobinding; "
                        "CUDA EP adds 39+ internal memcpys")

    print(f"\n  Graph: {int8_model}")
    print(f"         ({chosen_label})")

    use_cuda = (
        resolve_device(config) == "cuda"
        and "CUDAExecutionProvider" in available_execution_providers(ort)
    )
    if not use_cuda:
        print("\n  CUDA not available → running a degraded CPU demo.")
        print("  The io_binding advantage is PCIe copy elimination, which is")
        print("  only meaningful on GPU. On CPU we can only show the session-")
        print("  options effect (graph optimization level + thread counts).")

    B, L = config.bert_batch_size, config.bert_seq_length
    feed_np = _bert_random_feed(B, L, seed=config.seed)

    results = []

    # ── Config 1: Baseline (no io_binding, default session options) ──
    # Deliberately NOT using _make_session / ORT_ENABLE_ALL here — Config 3
    # below uses ORT_ENABLE_ALL so we can measure its incremental effect.
    print(f"\n  [1/3] Baseline: default SessionOptions, implicit H2D/D2H")
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if use_cuda else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(str(int8_model),
                                sess_options=ort.SessionOptions(),
                                providers=providers)
    input_names = {i.name for i in sess.get_inputs()}
    feed = {k: v for k, v in feed_np.items() if k in input_names}
    mean_ms, std_ms = time_session(
        sess, feed, config.bert_num_warmup, config.bert_num_iters)
    results.append(("Baseline (no io_binding)", "no_bind", mean_ms, std_ms))
    print(f"    Latency: {mean_ms:.2f} ± {std_ms:.2f} ms")

    # ── Config 2: io_binding (CUDA only — skip cleanly if CPU) ──              #K
    if use_cuda:
        print(f"\n  [2/3] io_binding: inputs/output bound on CUDA device")
        try:
            import torch
            so = ort.SessionOptions()
            sess2 = ort.InferenceSession(
                str(int8_model), sess_options=so,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

            # Stage inputs on the GPU once; io_binding reads them in-place.
            dev_inputs = {
                k: torch.from_numpy(v).cuda() for k, v in feed.items()
            }

            # Pre-allocate output on device for the logits tensor.
            output_info = sess2.get_outputs()
            # BERT-base SST-2 classifier → logits shape (B, num_labels=2)
            num_labels = output_info[0].shape[-1]
            if isinstance(num_labels, str):  # dynamic dim
                num_labels = 2
            out_dev = torch.empty(
                (B, num_labels), dtype=torch.float32, device="cuda")

            def run_iobind():
                binding = sess2.io_binding()
                for k, t in dev_inputs.items():
                    binding.bind_input(
                        name=k, device_type="cuda", device_id=0,
                        element_type=np.int64, shape=tuple(t.shape),
                        buffer_ptr=t.data_ptr())
                binding.bind_output(
                    name=output_info[0].name,
                    device_type="cuda", device_id=0,
                    element_type=np.float32, shape=tuple(out_dev.shape),
                    buffer_ptr=out_dev.data_ptr())
                sess2.run_with_iobinding(binding)

            # Warmup + time
            for _ in range(config.bert_num_warmup):
                run_iobind()
            torch.cuda.synchronize()
            tlist = []
            for _ in range(config.bert_num_iters):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                run_iobind()
                torch.cuda.synchronize()
                tlist.append((time.perf_counter() - t0) * 1000)
            mean2, std2 = float(np.mean(tlist)), float(np.std(tlist))
            results.append(("io_binding (device-resident I/O)", "iobind",
                            mean2, std2))
            print(f"    Latency: {mean2:.2f} ± {std2:.2f} ms  "
                  f"(speedup vs baseline: {mean_ms/mean2:.2f}×)")
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            results.append(("io_binding", "iobind", float("nan"), 0.0))

    #K The key insight: for a (8, 128) BERT input, H2D copy is ~8 KB per
    #  tensor × 3 tensors = ~24 KB, and D2H output is ~64 bytes. Tiny in
    #  absolute terms, but a single PCIe round trip has a fixed ~5-10 µs
    #  latency floor. On INT8 BERT-base where the actual compute takes
    #  ~1-3 ms, eliminating even 20 µs of copy overhead is 1-2% — not
    #  nothing, and it compounds across 100k requests/sec serving.
    #  The real win is per-token autoregressive generation, where the
    #  input changes by one token per step; there the copy floor starts
    #  to dominate if you don't pin memory. BERT classification is the
    #  modest-win case; that's honest to report.

    # ── Config 3: Tuned SessionOptions (graph opt + threads) ──                #L
    print(f"\n  [3/3] Tuned SessionOptions: ORT_ENABLE_ALL + thread tuning")
    so3 = ort.SessionOptions()
    so3.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if not use_cuda:
        # On CPU, the only knobs that matter.
        so3.intra_op_num_threads = os.cpu_count() or 4
        so3.inter_op_num_threads = 1
        # ORT_SEQUENTIAL avoids intra-graph parallelism conflicts with the
        # intra-op pool on CPU-dense workloads like BERT attention.
        so3.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        print(f"    intra_op_num_threads = {so3.intra_op_num_threads}")
        print(f"    inter_op_num_threads = {so3.inter_op_num_threads}")
    print(f"    graph_optimization_level = ORT_ENABLE_ALL")
    sess3 = ort.InferenceSession(
        str(int8_model), sess_options=so3, providers=providers)
    mean3, std3 = time_session(
        sess3, feed, config.bert_num_warmup, config.bert_num_iters)
    results.append(("Tuned SessionOptions", "after", mean3, std3))
    print(f"    Latency: {mean3:.2f} ± {std3:.2f} ms")

    #L ORT_ENABLE_ALL enables layout-sensitive kernel selection (e.g.
    #  NCHW vs NHWC pick for convolutions) and the generic graph-level
    #  fusions (constant folding, redundant-node elimination). It does
    #  NOT trigger the transformer-aware fusions — those only come from
    #  onnxruntime.transformers.optimizer (see transformer-optimizer mode).
    #  On a graph that already went through optimize_model(), ORT_ENABLE_ALL
    #  is close to a no-op; on a raw export it's a free 10-20%.

    # ── Summary ──
    base_ms = results[0][2]
    print(f"\n  ─── Listing 9.2 summary ───")
    hdr = f"  {'Configuration':<35} {'Latency (ms)':<18} {'vs Baseline':<12}"
    print(hdr)
    print(f"  {'─' * len(hdr)}")
    for label, _, m, s in results:
        if m != m:
            print(f"  {label:<35} {'FAILED':<18} {'—':<12}")
        else:
            sp = base_ms / m
            print(f"  {label:<35} {m:>6.2f} ± {s:<5.2f}     {sp:.2f}×")

    # ── Figure ──
    apply_manning_style()
    valid = [r for r in results if r[2] == r[2]]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for i, (label, tag, m, s) in enumerate(valid):
        ax.bar(i, m, yerr=s,
               color=COLORS[tag], hatch=HATCHES[tag],
               edgecolor="black", linewidth=0.5, width=0.65,
               error_kw={"linewidth": 0.8, "capsize": 3})
        ax.text(i, m + s + max(r[2] for r in valid) * 0.02,
                f"{m:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(
        ["Baseline", "io_binding", "Tuned\nSessionOpts"][:len(valid)],
        fontsize=7)
    ax.set_ylabel("Latency (ms)")
    ep_label = "CUDA EP" if use_cuda else "CPU EP"
    ax.set_title(f"Session options & io_binding ({ep_label}, "
                 f"batch={B}, seq={L})")
    fig.tight_layout()
    save_or_show(fig, "CH09_F08_Kalyanarangan_iobinding_sessionopts", config)


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing and main entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Ch9 §9.2 — Deploy through Optimum and ONNX Runtime"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["optimum", "transformer-optimizer", "ep-comparison",
                 "calibration", "iobinding", "all"],
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device (default: auto-detect)",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save figures to disk (PNG + PDF)",
    )
    parser.add_argument(
        "--force-export", action="store_true",
        help="Ignore cached ONNX artifacts and rebuild from scratch",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="BERT inference batch size (default: 8)",
    )
    parser.add_argument(
        "--seq-length", type=int, default=128,
        help="BERT sequence length (default: 128)",
    )
    parser.add_argument(
        "--num-eval", type=int, default=200,
        help="SST-2 eval samples for optimum mode (default: 200)",
    )
    parser.add_argument(
        "--num-calib", type=int, default=64,
        help="BERT calibration samples for static INT8 (default: 64)",
    )
    parser.add_argument(
        "--resnet-num-calib", type=int, default=100,
        help="ResNet-18 calibration samples (default: 100)",
    )
    parser.add_argument(
        "--resnet-num-eval", type=int, default=500,
        help="ResNet-18 eval samples for agreement rate (default: 500)",
    )
    parser.add_argument(
        "--num-iters", type=int, default=30,
        help="Timing iterations per benchmark (default: 30)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for figures (default: ./figures)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Cache directory for ONNX artifacts (default: ./onnx_cache)",
    )
    args = parser.parse_args()

    config = Config(
        mode=args.mode,
        device=args.device,
        save_plots=args.save_plots,
        force_export=args.force_export,
        bert_batch_size=args.batch_size,
        bert_seq_length=args.seq_length,
        bert_num_eval=args.num_eval,
        bert_num_calib=args.num_calib,
        resnet_num_calib=args.resnet_num_calib,
        resnet_num_eval=args.resnet_num_eval,
        bert_num_iters=args.num_iters,
    )
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.cache_dir:
        config.cache_dir = Path(args.cache_dir)
    return config


def main():
    config = parse_args()

    print("=" * 70)
    print("Chapter 9, Section 9.2 — Deploy through Optimum and ONNX Runtime")
    print("=" * 70)
    print(f"  Mode:           {config.mode}")
    print(f"  Device:         {config.device} (resolved: {resolve_device(config)})")
    print(f"  Batch/Seq:      {config.bert_batch_size} / {config.bert_seq_length}")
    print(f"  Save plots:     {config.save_plots}")
    print(f"  Force export:   {config.force_export}")

    ort = get_ort()
    print_environment(ort, config)

    # Running order matters when mode=all: optimum must come first because
    # its BERT artifacts feed transformer-optimizer, ep-comparison, and
    # iobinding. calibration is independent (builds its own ResNet-18).
    modes_to_run = (
        ["optimum", "transformer-optimizer", "ep-comparison",
         "iobinding", "calibration"]
        if config.mode == "all" else [config.mode]
    )

    for mode in modes_to_run:
        try:
            if mode == "optimum":
                run_optimum(config)
            elif mode == "transformer-optimizer":
                run_transformer_optimizer(config)
            elif mode == "ep-comparison":
                run_ep_comparison(config)
            elif mode == "calibration":
                run_calibration(config)
            elif mode == "iobinding":
                run_iobinding(config)
        except FileNotFoundError as e:
            print(f"\n  SKIPPED mode={mode}: {e}")
        except Exception as e:
            print(f"\n  FAILED mode={mode}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()