"""
Chapter 9, Section 9.5 — Package models for serving and distribution
Companion script for "Efficient AI in Practice: Quantization and Fast Inference"

What this script demonstrates:
  Sections 9.2 through 9.4 produced quantized artifacts and proved that
  the runtime + execution provider combination decides which kernels
  actually dispatch. Section 9.5 packages those artifacts for handoff
  to a serving stack, with three concrete deliverables per model:

    1. A `manifest.json` next to each artifact that captures the
       deployment contract — input/output signatures, dtype
       declarations, the quantization recipe (scheme, calibration
       method, operators quantized and excluded, calibration-set
       fingerprint), build provenance (tool versions, host, timestamp),
       and the runtime + hardware requirements the artifact assumes.
    2. A Triton-style `config.pbtxt` per backend that the artifact can
       be served through (onnxruntime, openvino, tensorrt) so the
       reader sees a concrete model-repository layout, not just an
       idea.
    3. A compatibility-matrix figure that records, on the hardware in
       front of the script, which (artifact x runtime) cells actually
       execute at the precision their manifest claims, which fall back
       silently to a higher precision, and which require an artifact
       rebuild before they can serve at all.

  *** SCOPE ***
  This script does NOT spin up Triton, push to a model registry, or
  rebuild TensorRT engines. It is a packaging-and-verification pass
  layered on top of the artifacts the previous three sections already
  produced. The figure is honest about hardware: GPU rows stay grey on
  CPU-only boxes; AMX-INT8 rows stay grey on Cascade Lake. Run on each
  target tier you plan to deploy to, archive the figure pair, and the
  ops team has a defensible "what executed where" record.

Modes:
  --mode resnet   Package + verify ResNet-18 artifacts
  --mode bert     Package + verify BERT-base SST-2 artifacts
  --mode all      Both (default)

Usage:
  # Run inside the Ch9 directory after sections 9.2 / 9.3 / 9.4 have
  # populated the cache (script reuses their ONNX, IR, and TRT plans).
  python ch9_packaging_serving.py --mode all --save-plots

  # Build any missing artifacts on the fly (no TRT plan; that requires
  # section 9.3's pipeline).
  python ch9_packaging_serving.py --mode all --save-plots --build-missing

  # Point at a non-default cache directory:
  python ch9_packaging_serving.py --mode all --save-plots \
      --cache-dir ./ov_cache --trt-cache ./trt_cache

Install (one Python environment, CPU-only sufficient for the
manifest-and-verify pass; CUDA / TensorRT only needed to populate the
GPU rows of the matrix):
  pip install -U openvino nncf onnx onnxruntime-openvino \
                 torch torchvision \
                 transformers "optimum[onnxruntime]" datasets \
                 matplotlib pillow

  # Optional: TRT plan verification (the engine itself is built by
  # the section 9.3 script).
  pip install tensorrt

Hardware target:
  Any host. The compatibility matrix self-adapts:
    - Intel CPU only:                CPU rows light up; GPU rows grey
    - NVIDIA GPU (e.g. L4, T4, ...): GPU rows light up; CPU rows green
    - Mixed Intel CPU + NVIDIA GPU:  All rows populate

  The OPTIMIZATION_CAPABILITIES line printed at startup tells you
  which Intel ISA tier the script saw; nvidia-smi gating decides
  whether the CUDA / TRT verifiers attempt to dispatch.

Note on cache reuse:
  This script is cooperative with the section 9.4 cache layout
  (`ov_cache/resnet18_*.onnx`, `bert-sst2-fp32/model.onnx`,
   `*_int8.xml`, etc). When run from the Ch9 directory after
  section 9.4, every needed artifact is already on disk and the
  script does no rebuilding. If artifacts are missing and
  `--build-missing` is passed, the script reproduces the minimal set
  needed (FP32 ONNX, INT8 QDQ ONNX, OV IR FP32, OV IR INT8) using
  the same recipes section 9.4 used, so the manifests stay
  consistent across the chapter. TRT plans require GPU + the
  section 9.3 build pipeline and are never rebuilt here.
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("nncf").setLevel(logging.WARNING)

SCRIPT_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = "1.0"


# --- Configuration ---------------------------------------------------------

@dataclass
class Config:
    mode: str = "all"
    save_plots: bool = False
    build_missing: bool = False
    output_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "figures")
    cache_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "ov_cache")
    trt_cache: Path = field(default_factory=lambda: SCRIPT_DIR / "trt_cache")
    package_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "packages")

    # Inference shapes (mirror sections 9.2 / 9.3 / 9.4)
    resnet_batch: int = 32
    resnet_image_size: int = 224
    bert_batch: int = 8
    bert_seq_len: int = 128

    num_calib_batches: int = 8
    seed: int = 42


# --- Verification status enum ---------------------------------------------

class Status(str, Enum):
    """Five honest status categories for a (artifact, runtime) cell.

    OK              Loaded, ran a smoke pass, runtime inspector
                    confirmed the executed precision matches the
                    manifest's claimed precision (within tolerance).
    DEMOTED         Loaded and ran, but the runtime inspector reports
                    a higher-precision execution than the manifest
                    claimed. This is the silent up-convert failure
                    mode the chapter has been chasing.
    NA              Not applicable by design — this artifact format
                    is not a load operation for this runtime (e.g.
                    handing an OV IR to TensorRT). The matrix shows
                    these cells empty rather than as a failure.
    BLOCKED         A precondition is missing — runtime not installed,
                    hardware absent (no CUDA, no AMX), or the artifact
                    file itself was not produced by an earlier section.
                    The cell labels which precondition failed.
    ERROR           Load or smoke-pass threw. The cell carries a
                    short exception summary.
    """
    OK = "OK"
    DEMOTED = "DEMOTED"
    NA = "NA"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


@dataclass
class CellResult:
    status: Status
    observed_precision: str = ""    # "i8/u8 mix", "f32", "bf16", ""
    note: str = ""                   # short text for figure cell
    detail: str = ""                 # longer text for verify_report.json


# --- Manning figure style --------------------------------------------------

def _pick_font() -> str:
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for c in ("Arial", "Helvetica", "DejaVu Sans"):
        if c in available:
            return c
    return "sans-serif"


def apply_manning_style():
    plt.rcParams.update({
        "font.family": _pick_font(),
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300,
        "figure.figsize": (5.6, 3.5),
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


# Status -> (face colour, hatch, glyph) for figure cells. Grayscale-safe:
# the hatch carries the OK/DEMOTED/ERROR distinction even if colour is
# stripped. BLOCKED uses fill colour alone so cell text stays readable
# (a hatch behind text obscures it at 5pt-6pt fonts).
STATUS_STYLE: Dict[Status, Tuple[str, str, str]] = {
    Status.OK:      ("#cde7d2", "",     "✓"),   # light green, solid
    Status.DEMOTED: ("#fce4a8", "//",   "↑"),   # light amber, diag hatch
    Status.NA:      ("#ffffff", "",     ""),    # blank
    Status.BLOCKED: ("#e0e0e0", "",     "·"),   # grey, no hatch
    Status.ERROR:   ("#f4c4c4", "xx",   "✗"),   # light red, cross hatch
}


def save_or_show(fig, name: str, config: Config):
    if config.save_plots:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        png = config.output_dir / f"{name}.png"
        pdf = config.output_dir / f"{name}.pdf"
        fig.savefig(png, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        fig.savefig(pdf, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {png}")
        print(f"  Saved: {pdf}")
    else:
        plt.show()
    plt.close(fig)


# --- Environment probe -----------------------------------------------------

def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(name)
        except PackageNotFoundError:
            return "NOT INSTALLED"
    except ImportError:
        return "NOT INSTALLED"


def get_ov():
    try:
        import openvino as ov
        return ov
    except ImportError:
        return None


def get_nncf():
    try:
        import nncf
        return nncf
    except ImportError:
        return None


def get_trt():
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        return None


def has_cuda() -> bool:
    """True iff a CUDA-capable GPU is reachable. We check torch first
    (the most common install) and fall back to nvidia-smi presence."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    # Fall back to nvidia-smi -- handles the case where torch is
    # CPU-only but a GPU is present.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def cuda_compute_capability() -> Optional[Tuple[int, int]]:
    """Return (major, minor) for the first CUDA device, or None."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except Exception:
        pass
    return None


def cuda_device_name() -> Optional[str]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def has_ort() -> bool:
    try:
        import onnxruntime
        return True
    except ImportError:
        return False


def ort_available_providers() -> List[str]:
    try:
        import onnxruntime as ort
        get = getattr(ort, "get_available_providers", None)
        return list(get()) if callable(get) else []
    except Exception:
        return []


def has_ort_provider(name: str) -> bool:
    return name in ort_available_providers()


def ov_optimization_capabilities() -> List[str]:
    ov = get_ov()
    if ov is None:
        return []
    try:
        core = ov.Core()
        caps = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
        return list(caps)
    except Exception:
        return []


def print_environment(config: Config):
    print(f"  Python:              {sys.version.split()[0]}")
    print(f"  Host:                {platform.node()} ({platform.system()} "
          f"{platform.machine()})")
    print(f"  openvino:            {_pkg_version('openvino')}")
    print(f"  nncf:                {_pkg_version('nncf')}")
    print(f"  onnxruntime-openvino:{_pkg_version('onnxruntime-openvino')}")
    print(f"  onnxruntime:         {_pkg_version('onnxruntime')}")
    print(f"  onnxruntime-gpu:     {_pkg_version('onnxruntime-gpu')}")
    print(f"  tensorrt:            {_pkg_version('tensorrt')}")
    print(f"  optimum:             {_pkg_version('optimum')}")
    print(f"  transformers:        {_pkg_version('transformers')}")
    print(f"  torch:               {_pkg_version('torch')}")

    ov = get_ov()
    if ov is not None:
        try:
            core = ov.Core()
            cpu_name = core.get_property("CPU", "FULL_DEVICE_NAME")
            opt_caps = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
            print(f"  CPU:                 {cpu_name}")
            print(f"  OV CPU caps:         {', '.join(opt_caps)}")
        except Exception as e:
            print(f"  OV CPU probe:        failed ({type(e).__name__})")

    if has_cuda():
        cap = cuda_compute_capability()
        cap_str = f"sm_{cap[0]}{cap[1]}" if cap else "unknown"
        print(f"  GPU:                 {cuda_device_name()} ({cap_str})")
    else:
        print(f"  GPU:                 (none detected)")

    print(f"  ORT providers:       {', '.join(ort_available_providers()) or '(none)'}")
    print(f"  Cache dir:           {config.cache_dir}")
    print(f"  TRT cache:           {config.trt_cache}")
    print(f"  Package out:         {config.package_dir}")


# --- Artifact discovery and minimal builders ------------------------------
# Section 9.4 wrote artifacts into ov_cache/. We reuse them when present;
# otherwise we rebuild with the same recipes. We never rebuild TRT plans
# here -- those belong to section 9.3's pipeline.

@dataclass
class ArtifactSet:
    """Per-model collection of artifact paths. None entries are
    missing on disk; the verifier matrix marks their cells BLOCKED."""
    fp32_onnx: Optional[Path] = None
    qdq_int8_onnx: Optional[Path] = None
    ov_fp32_ir: Optional[Path] = None
    ov_int8_ir: Optional[Path] = None
    trt_int8_plan: Optional[Path] = None


def hash_file(path: Path) -> str:
    """SHA-256 of the file. Used for manifest integrity and for the
    calibration-set fingerprint."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_feeds(feeds: List[Dict[str, np.ndarray]]) -> str:
    """Deterministic SHA-256 over a calibration feed list. Order +
    array bytes + dtype + shape are all in scope. Two different
    runs that produced the same feeds will hash identically."""
    h = hashlib.sha256()
    for feed in feeds:
        for k in sorted(feed.keys()):
            v = feed[k]
            h.update(k.encode())
            h.update(str(v.dtype).encode())
            h.update(str(v.shape).encode())
            h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


# ---- ResNet-18 builders (mirrors section 9.4) --------------------------

def _resnet_fp32_onnx(out_path: Path, batch: int, img_size: int):
    """Export torchvision ResNet-18 to FP32 ONNX."""
    import torch
    import torchvision.models as tvm

    print(f"  Building FP32 ONNX: {out_path.name}")
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT).eval()
    dummy = torch.randn(batch, 3, img_size, img_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False forces the legacy TorchScript exporter -- the new
    # dynamo path (default since torch 2.6) takes onnxscript as a soft
    # dependency and crashes if it isn't installed. The TorchScript path
    # is bundled with torch and matches what section 9.4 used.
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14, do_constant_folding=True,
        dynamo=False,
    )


def _resnet_calib_feeds(num_batches: int, batch: int, img_size: int,
                        seed: int) -> List[Dict[str, np.ndarray]]:
    """Synthetic ImageNet-distribution calibration. Random tensors with
    pixel mean/std matching ImageNet are a faithful approximation for
    static-INT8 range estimation on Conv layers."""
    rng = np.random.default_rng(seed)
    feeds = []
    for _ in range(num_batches):
        x = rng.standard_normal((batch, 3, img_size, img_size),
                                dtype=np.float32)
        feeds.append({"input": x})
    return feeds


def _find_resnet_qdq_excludes(fp32_onnx: Path) -> List[str]:
    """Exclude input-adjacent stem Conv and the output classifier from
    quantization. Matches section 9.3 / 9.4 sensitivity exclusions."""
    import onnx
    model = onnx.load(str(fp32_onnx))
    excludes = []
    # Walk graph, find input-consuming conv and output-producing matmul
    input_names = {i.name for i in model.graph.input}
    output_names = {o.name for o in model.graph.output}
    for node in model.graph.node:
        if node.op_type == "Conv" and any(i in input_names for i in node.input):
            excludes.append(node.name)
        if node.op_type in ("MatMul", "Gemm") and \
                any(o in output_names for o in node.output):
            excludes.append(node.name)
    return excludes


def _build_qdq_int8_onnx(
    fp32_onnx: Path, qdq_onnx: Path,
    feeds: List[Dict[str, np.ndarray]],
    op_types: List[str], excludes: List[str],
):
    """ONNX QDQ INT8 with the four-knob recipe section 9.3 established
    for TRT compatibility: QDQ format, symmetric activations, op-type
    scoped, sensitive-layer excludes."""
    from onnxruntime.quantization import (
        quantize_static, CalibrationDataReader, CalibrationMethod,
        QuantFormat, QuantType,
    )

    class FeedsReader(CalibrationDataReader):
        def __init__(self, feeds):
            self.feeds = list(feeds)
            self._i = 0
        def get_next(self):
            if self._i >= len(self.feeds):
                return None
            f = self.feeds[self._i]
            self._i += 1
            return f

    print(f"  Building INT8 QDQ ONNX: {qdq_onnx.name}")
    qdq_onnx.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(fp32_onnx),
        model_output=str(qdq_onnx),
        calibration_data_reader=FeedsReader(feeds),
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types,
        nodes_to_exclude=excludes,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": True},
    )


# ---- BERT builders (mirrors section 9.4) -------------------------------

def _bert_fp32_onnx(bert_dir: Path, batch: int, seq_len: int):
    """Export textattack/bert-base-uncased-SST-2 via Optimum."""
    from optimum.onnxruntime import ORTModelForSequenceClassification

    print(f"  Building BERT FP32 ONNX: {bert_dir}")
    bert_dir.mkdir(parents=True, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-SST-2", export=True,
    )
    model.save_pretrained(str(bert_dir))


def _bert_calib_feeds(num_batches: int, batch: int, seq_len: int,
                      seed: int) -> List[Dict[str, np.ndarray]]:
    """Feeds matching the BERT input signature. Random integer ids are
    fine for activation-range probing on attention; tokenizer outputs
    on real text would calibrate marginally tighter scales but cost
    chapter dependencies (HF dataset download)."""
    from transformers import AutoTokenizer
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
    vocab_size = tok.vocab_size

    feeds = []
    for _ in range(num_batches):
        input_ids = rng.integers(0, vocab_size, (batch, seq_len),
                                 dtype=np.int64)
        # Sentence-A-only token type ids; matches SST-2 single-input task
        token_type_ids = np.zeros((batch, seq_len), dtype=np.int64)
        # Full attention (no padding) -- matches the production case
        # where inputs are pre-padded by the tokenizer to seq_len
        attention_mask = np.ones((batch, seq_len), dtype=np.int64)
        feeds.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        })
    return feeds


def _find_bert_qdq_excludes(fp32_onnx: Path) -> List[str]:
    """Exclude input-side embedding Gather and final classifier MatMul.
    Same sensitivity rationale as section 9.3."""
    import onnx
    model = onnx.load(str(fp32_onnx))
    excludes = []
    input_names = {i.name for i in model.graph.input}
    output_names = {o.name for o in model.graph.output}
    for node in model.graph.node:
        if node.op_type == "Gather" and \
                any(i in input_names for i in node.input):
            excludes.append(node.name)
        if node.op_type in ("MatMul", "Gemm") and \
                any(o in output_names for o in node.output):
            excludes.append(node.name)
    return excludes


# ---- OpenVINO IR builders ----------------------------------------------

def _convert_onnx_to_ir(onnx_path: Path, ir_xml: Path):
    """Run ov.convert_model on an ONNX file and serialize as IR."""
    ov = get_ov()
    if ov is None:
        raise RuntimeError("openvino not installed")
    print(f"  Building OV IR: {ir_xml.name}")
    model = ov.convert_model(str(onnx_path))
    ir_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(ir_xml), compress_to_fp16=False)


def _quantize_ir_with_nncf(
    fp32_ir: Path, int8_ir: Path,
    feeds: List[Dict[str, np.ndarray]],
    model_type: str, ignored_op_names: List[str],
):
    """NNCF post-training INT8. Mirrors section 9.4's quantize_with_nncf."""
    ov = get_ov()
    nncf = get_nncf()
    if ov is None or nncf is None:
        raise RuntimeError("openvino + nncf required")
    print(f"  Building OV INT8 IR: {int8_ir.name}")
    core = ov.Core()
    fp32_model = core.read_model(str(fp32_ir))

    def transform_fn(item):
        return item

    dataset = nncf.Dataset(feeds, transform_fn)
    kwargs = dict(
        calibration_dataset=dataset,
        preset=nncf.QuantizationPreset.MIXED,
        target_device=nncf.TargetDevice.CPU,
        subset_size=len(feeds),
        fast_bias_correction=True,
    )
    if model_type == "transformer":
        kwargs["model_type"] = nncf.ModelType.TRANSFORMER
    if ignored_op_names:
        kwargs["ignored_scope"] = nncf.IgnoredScope(
            names=ignored_op_names, validate=False)
    quantized = nncf.quantize(fp32_model, **kwargs)
    int8_ir.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(int8_ir), compress_to_fp16=False)


# ---- Per-model artifact discovery / build ------------------------------

def discover_or_build_resnet(config: Config) -> ArtifactSet:
    """Locate (and optionally rebuild) ResNet-18 artifacts."""
    cache = config.cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    fp32_onnx = cache / "resnet18_fp32.onnx"
    qdq_onnx = cache / "resnet18_qdq_int8.onnx"
    fp32_ir = cache / "resnet18_fp32.xml"
    int8_ir = cache / "resnet18_int8.xml"

    if not fp32_onnx.exists() and config.build_missing:
        _resnet_fp32_onnx(fp32_onnx, config.resnet_batch,
                          config.resnet_image_size)

    feeds = None  # built lazily only when needed for QDQ ONNX

    if not qdq_onnx.exists() and config.build_missing and fp32_onnx.exists():
        feeds = _resnet_calib_feeds(config.num_calib_batches,
                                    config.resnet_batch,
                                    config.resnet_image_size, config.seed)
        excludes = _find_resnet_qdq_excludes(fp32_onnx)
        _build_qdq_int8_onnx(fp32_onnx, qdq_onnx, feeds,
                             op_types=["Conv"], excludes=excludes)

    if not fp32_ir.exists() and config.build_missing and fp32_onnx.exists():
        try:
            _convert_onnx_to_ir(fp32_onnx, fp32_ir)
        except Exception as e:
            print(f"    OV IR build failed: {type(e).__name__}: {e}")

    if not int8_ir.exists() and config.build_missing and fp32_ir.exists():
        try:
            if feeds is None:
                feeds = _resnet_calib_feeds(config.num_calib_batches,
                                            config.resnet_batch,
                                            config.resnet_image_size,
                                            config.seed)
            # Walk the OV IR for op names matching the ONNX excludes.
            # NNCF and ONNX use slightly different node-naming
            # conventions; safest to derive from the IR itself.
            ignored = _find_ir_ignored_ops_resnet(fp32_ir)
            _quantize_ir_with_nncf(fp32_ir, int8_ir, feeds,
                                   model_type="cnn",
                                   ignored_op_names=ignored)
        except Exception as e:
            print(f"    OV IR INT8 build failed: {type(e).__name__}: {e}")

    # TRT plan: only ever discovered, never built here.
    trt_plan_candidates = [
        config.trt_cache / "resnet18_int8_explicit.plan",
        config.trt_cache / "resnet18_int8_implicit.plan",
        config.trt_cache / "resnet18_int8.plan",
    ]
    trt_plan = next((p for p in trt_plan_candidates if p.exists()), None)

    return ArtifactSet(
        fp32_onnx=fp32_onnx if fp32_onnx.exists() else None,
        qdq_int8_onnx=qdq_onnx if qdq_onnx.exists() else None,
        ov_fp32_ir=fp32_ir if fp32_ir.exists() else None,
        ov_int8_ir=int8_ir if int8_ir.exists() else None,
        trt_int8_plan=trt_plan,
    )


def discover_or_build_bert(config: Config) -> ArtifactSet:
    cache = config.cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    bert_dir = cache / "bert-sst2-fp32"
    fp32_onnx = bert_dir / "model.onnx"
    qdq_onnx = cache / "bert-sst2-qdq.onnx"
    fp32_ir = cache / "bert-sst2-fp32.xml"
    int8_ir = cache / "bert-sst2-int8.xml"

    if not fp32_onnx.exists() and config.build_missing:
        _bert_fp32_onnx(bert_dir, config.bert_batch, config.bert_seq_len)

    feeds = None

    if not qdq_onnx.exists() and config.build_missing and fp32_onnx.exists():
        feeds = _bert_calib_feeds(config.num_calib_batches,
                                  config.bert_batch, config.bert_seq_len,
                                  config.seed)
        excludes = _find_bert_qdq_excludes(fp32_onnx)
        _build_qdq_int8_onnx(fp32_onnx, qdq_onnx, feeds,
                             op_types=["MatMul"], excludes=excludes)

    if not fp32_ir.exists() and config.build_missing and fp32_onnx.exists():
        try:
            _convert_onnx_to_ir(fp32_onnx, fp32_ir)
        except Exception as e:
            print(f"    OV IR build failed: {type(e).__name__}: {e}")

    if not int8_ir.exists() and config.build_missing and fp32_ir.exists():
        try:
            if feeds is None:
                feeds = _bert_calib_feeds(config.num_calib_batches,
                                          config.bert_batch,
                                          config.bert_seq_len, config.seed)
            ignored = _find_ir_ignored_ops_bert(fp32_ir)
            _quantize_ir_with_nncf(fp32_ir, int8_ir, feeds,
                                   model_type="transformer",
                                   ignored_op_names=ignored)
        except Exception as e:
            print(f"    OV IR INT8 build failed: {type(e).__name__}: {e}")

    trt_plan_candidates = [
        config.trt_cache / "bert-sst2_int8_explicit.plan",
        config.trt_cache / "bert-base_int8_explicit.plan",
        config.trt_cache / "bert_sst2_int8.plan",
    ]
    trt_plan = next((p for p in trt_plan_candidates if p.exists()), None)

    return ArtifactSet(
        fp32_onnx=fp32_onnx if fp32_onnx.exists() else None,
        qdq_int8_onnx=qdq_onnx if qdq_onnx.exists() else None,
        ov_fp32_ir=fp32_ir if fp32_ir.exists() else None,
        ov_int8_ir=int8_ir if int8_ir.exists() else None,
        trt_int8_plan=trt_plan,
    )


def _find_ir_ignored_ops_resnet(fp32_ir: Path) -> List[str]:
    """Walk the IR's ov.Model and return the friendly names of the
    stem Conv (input-adjacent) and the classifier-head MatMul/Gemm
    (output-adjacent). Matches section 9.4's _find_resnet_ignored_ops."""
    ov = get_ov()
    if ov is None:
        return []
    core = ov.Core()
    model = core.read_model(str(fp32_ir))
    ignored = []
    inputs = {p.get_node().get_friendly_name() for p in model.inputs}
    outputs = {r.get_node().get_friendly_name() for r in model.outputs}
    # Scan -- naive but cheap on resnet-sized graphs
    for op in model.get_ops():
        try:
            fn = op.get_friendly_name()
            tn = op.get_type_name()
        except Exception:
            continue
        if tn in ("Convolution", "ConvolutionBackpropData") and \
                _input_is_model_input(op, model):
            ignored.append(fn)
        if tn in ("MatMul", "FullyConnected") and \
                _output_is_model_output(op, model):
            ignored.append(fn)
    return ignored


def _find_ir_ignored_ops_bert(fp32_ir: Path) -> List[str]:
    """Same idea as ResNet but targeting the input-side embedding Gather
    and the final classifier MatMul."""
    ov = get_ov()
    if ov is None:
        return []
    core = ov.Core()
    model = core.read_model(str(fp32_ir))
    ignored = []
    for op in model.get_ops():
        try:
            fn = op.get_friendly_name()
            tn = op.get_type_name()
        except Exception:
            continue
        if tn == "Gather" and _input_is_model_input(op, model):
            ignored.append(fn)
        if tn in ("MatMul", "FullyConnected") and \
                _output_is_model_output(op, model):
            ignored.append(fn)
    return ignored


def _input_is_model_input(op, model) -> bool:
    """Crude predicate: any of this op's input ports trace to a Parameter."""
    try:
        for i in range(op.get_input_size()):
            src = op.input(i).get_source_output().get_node()
            if src.get_type_name() == "Parameter":
                return True
    except Exception:
        return False
    return False


def _output_is_model_output(op, model) -> bool:
    """Crude predicate: any of this op's outputs feed a Result."""
    try:
        for i in range(op.get_output_size()):
            for tgt_input in op.output(i).get_target_inputs():
                if tgt_input.get_node().get_type_name() == "Result":
                    return True
    except Exception:
        return False
    return False


# --- Manifest construction -----------------------------------------------
# A manifest is the deployment contract: every fact the ops team needs to
# decide whether to load this artifact on this hardware. We probe the
# artifact for input/output signatures (so the manifest cannot drift from
# the file), then add the recipe and provenance from build-time context.

def _probe_onnx(path: Path) -> Dict[str, Any]:
    """Pull the input/output signatures, opset, IR version, and quant
    operator count from an ONNX file. The signatures are derived from
    the file -- they cannot drift from what the runtime will see."""
    import onnx
    model = onnx.load(str(path))

    def tensor_info(t):
        elem_type = t.type.tensor_type.elem_type
        # Map onnx elem_type enum -> dtype string. Spec lives in
        # onnx/onnx.proto; the values used here are stable across versions.
        ELEM_TYPE_NAMES = {
            1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
            6: "int32", 7: "int64", 9: "bool", 10: "float16",
            11: "float64", 16: "bfloat16",
        }
        dtype = ELEM_TYPE_NAMES.get(elem_type, f"elem_type_{elem_type}")
        shape = []
        for d in t.type.tensor_type.shape.dim:
            if d.dim_param:
                shape.append(d.dim_param)
            elif d.dim_value:
                shape.append(d.dim_value)
            else:
                shape.append("?")
        return {"name": t.name, "dtype": dtype, "shape": shape}

    inputs = [tensor_info(t) for t in model.graph.input]
    outputs = [tensor_info(t) for t in model.graph.output]

    op_types = [n.op_type for n in model.graph.node]
    qdq_ops = sum(1 for op in op_types
                  if op in ("QuantizeLinear", "DequantizeLinear"))
    quant_compute_ops = sum(1 for op in op_types
                            if op in ("QLinearConv", "QLinearMatMul",
                                      "MatMulInteger", "ConvInteger"))
    opset = model.opset_import[0].version if model.opset_import else None

    return {
        "format": "onnx",
        "ir_version": model.ir_version,
        "opset": opset,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "inputs": inputs,
        "outputs": outputs,
        "node_count": len(model.graph.node),
        "qdq_op_count": qdq_ops,
        "quant_compute_op_count": quant_compute_ops,
    }


def _probe_ov_ir(xml_path: Path) -> Dict[str, Any]:
    """Pull input/output signatures and a count of FakeQuantize ops
    from an OpenVINO IR. FakeQuantize count > 0 indicates the IR was
    NNCF-quantized; the runtime will fold these into INT8 kernels at
    compile time on the target device."""
    ov = get_ov()
    if ov is None:
        return {"format": "openvino-ir", "error": "openvino not installed"}
    core = ov.Core()
    model = core.read_model(str(xml_path))

    def port_info(port, name_fallback):
        node = port.get_node()
        try:
            name = node.get_friendly_name()
        except Exception:
            name = name_fallback
        try:
            dtype = port.get_element_type().get_type_name()
        except Exception:
            dtype = "?"
        try:
            shape = list(port.get_partial_shape().to_string())
            shape = port.get_partial_shape().to_string()
        except Exception:
            shape = "?"
        return {"name": name, "dtype": dtype, "shape": shape}

    inputs = [port_info(p, f"input_{i}") for i, p in enumerate(model.inputs)]
    outputs = [port_info(p, f"output_{i}") for i, p in enumerate(model.outputs)]

    fakequantize_count = sum(
        1 for op in model.get_ops()
        if op.get_type_name() == "FakeQuantize"
    )
    total_ops = sum(1 for _ in model.get_ops())

    return {
        "format": "openvino-ir",
        "inputs": inputs,
        "outputs": outputs,
        "total_ops": total_ops,
        "fakequantize_count": fakequantize_count,
    }


def _probe_trt_plan(plan_path: Path) -> Dict[str, Any]:
    """Probe a serialized TensorRT engine. We can read the header for
    builder version + compute capability without standing up a CUDA
    runtime; full deserialization needs an active TRT runtime + GPU."""
    trt = get_trt()
    info: Dict[str, Any] = {
        "format": "tensorrt-plan",
        "size_bytes": plan_path.stat().st_size,
    }
    if trt is None:
        info["error"] = "tensorrt not installed"
        return info
    if not has_cuda():
        info["error"] = "CUDA not available; full deserialization skipped"
        return info

    try:
        with open(plan_path, "rb") as f:
            plan_bytes = f.read()
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan_bytes)
        if engine is None:
            info["error"] = "deserialize returned None"
            return info
        info["num_io_tensors"] = engine.num_io_tensors
        info["device_memory_size"] = int(engine.device_memory_size)
        # Engine input/output names + dtypes
        io = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            dtype = str(engine.get_tensor_dtype(name))
            shape = list(engine.get_tensor_shape(name))
            mode = str(engine.get_tensor_mode(name))
            io.append({"name": name, "dtype": dtype,
                       "shape": shape, "mode": mode})
        info["io_tensors"] = io
        info["trt_version"] = trt.__version__
        info["cuda_compute_cap"] = cuda_compute_capability()
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def build_manifest(
    model_id: str,
    task: str,
    artifact_path: Path,
    artifact_kind: str,            # "onnx-fp32" / "onnx-qdq-int8" /
                                    # "openvino-ir-fp32" / "openvino-ir-int8" /
                                    # "tensorrt-plan-int8"
    quantization: Optional[Dict[str, Any]] = None,
    calibration_fingerprint: Optional[str] = None,
    target_runtime: Optional[List[str]] = None,
    target_hardware: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the deployment-contract manifest for one artifact.

    The manifest is the load-time check the ops team runs before
    serving: SHA-256 the artifact and compare to the recorded hash;
    confirm the runtime in front of you matches the version-locked
    range; confirm the hardware exposes the required ISA / compute
    capability. If any check fails, refuse to serve."""
    probe: Dict[str, Any] = {}
    if artifact_kind.startswith("onnx"):
        probe = _probe_onnx(artifact_path)
    elif artifact_kind.startswith("openvino-ir"):
        probe = _probe_ov_ir(artifact_path)
    elif artifact_kind.startswith("tensorrt"):
        probe = _probe_trt_plan(artifact_path)

    file_hash = hash_file(artifact_path)
    size_mb = artifact_path.stat().st_size / 1e6
    # OV IR is a pair (.xml + .bin); record both
    sidecar_paths: List[str] = []
    if artifact_kind.startswith("openvino-ir"):
        bin_path = artifact_path.with_suffix(".bin")
        if bin_path.exists():
            sidecar_paths.append(bin_path.name)
            size_mb += bin_path.stat().st_size / 1e6

    build_env = {
        "tool_versions": {
            "python": sys.version.split()[0],
            "onnxruntime": _pkg_version("onnxruntime"),
            "openvino": _pkg_version("openvino"),
            "nncf": _pkg_version("nncf"),
            "tensorrt": _pkg_version("tensorrt"),
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "optimum": _pkg_version("optimum"),
        },
        "host": platform.node(),
        "platform": platform.system() + " " + platform.machine(),
        "build_timestamp": datetime.datetime.now(datetime.timezone.utc)
                                  .isoformat(),
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": {
            "id": model_id,
            "task": task,
        },
        "artifact": {
            "kind": artifact_kind,
            "filename": artifact_path.name,
            "sidecar_files": sidecar_paths,
            "size_mb": round(size_mb, 3),
            "sha256": file_hash,
        },
        "signatures": {
            "inputs": probe.get("inputs", []),
            "outputs": probe.get("outputs", []),
        },
        "graph_summary": {
            k: v for k, v in probe.items()
            if k not in ("inputs", "outputs")
        },
        "quantization": quantization or {"scheme": "none"},
        "calibration_fingerprint": calibration_fingerprint,
        "target": {
            "runtime_compatible": target_runtime or [],
            "hardware_required": target_hardware or {},
        },
        "build": build_env,
    }
    return manifest


def write_manifest(manifest: Dict[str, Any], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"    Wrote: {out_path.relative_to(SCRIPT_DIR)}")


# --- Per-model recipe declarations ---------------------------------------
# These dictionaries are the "static" half of each manifest -- the
# half that comes from how we built the artifact, not from probing it.

def resnet_recipes(calib_hash: Optional[str]) -> Dict[str, Dict[str, Any]]:
    return {
        "fp32_onnx": {
            "kind": "onnx-fp32",
            "quantization": {"scheme": "none"},
            "calibration_fingerprint": None,
            "target_runtime": [
                "onnxruntime>=1.18", "openvino>=2025.0", "tensorrt>=10.0",
            ],
            "target_hardware": {
                "cpu_isa_min": ["AVX2"],
                "cpu_isa_recommended": ["AVX-512"],
                "gpu_compute_capability_min": None,
            },
        },
        "qdq_int8_onnx": {
            "kind": "onnx-qdq-int8",
            "quantization": {
                "scheme": "static",
                "format": "QDQ",
                "activation_dtype": "int8",
                "weight_dtype": "int8",
                "activation_symmetric": True,
                "weight_per_channel": False,
                "calibration_method": "MinMax",
                "operators_quantized": ["Conv"],
                "operators_excluded": "stem Conv (input-adjacent), "
                                     "classifier MatMul/Gemm (output-adjacent)",
                "calibration_num_samples": 256,  # 8 batches x 32
            },
            "calibration_fingerprint": calib_hash,
            "target_runtime": [
                "onnxruntime>=1.18 (CPU EP, OV EP, TRT EP)",
                "tensorrt>=10.0 (consumes via QDQ -> INT8 supernodes)",
            ],
            "target_hardware": {
                "cpu_isa_min": ["AVX-512 VNNI"],
                "cpu_isa_recommended": ["AMX-INT8"],
                "gpu_compute_capability_min": (7, 5),  # T4
                "note": "On CUDA EP the section 9.2 result shows partial "
                        "INT8 kernel coverage on Conv+ReLU+Pool fusion -- "
                        "use TensorRT EP or a TRT plan for production "
                        "INT8 dispatch on NVIDIA silicon.",
            },
        },
        "ov_fp32_ir": {
            "kind": "openvino-ir-fp32",
            "quantization": {"scheme": "none"},
            "calibration_fingerprint": None,
            "target_runtime": ["openvino>=2025.0"],
            "target_hardware": {
                "cpu_isa_min": ["AVX2"],
                "cpu_isa_recommended": ["AVX-512", "AMX-BF16"],
                "note": "BF16 inference precision falls back silently to "
                        "FP32 on CPUs without AVX512_BF16 / AMX-BF16. "
                        "Inspect the OPTIMIZATION_CAPABILITIES line on the "
                        "target host before relying on the BF16 row.",
            },
        },
        "ov_int8_ir": {
            "kind": "openvino-ir-int8",
            "quantization": {
                "scheme": "static",
                "tool": "nncf.quantize",
                "preset": "MIXED",                  # asym act + sym weight
                "model_type_hint": "cnn",
                "fast_bias_correction": True,
                "operators_excluded": "stem Conv (input-adjacent), "
                                     "classifier MatMul (output-adjacent)",
                "calibration_num_samples": 256,
            },
            "calibration_fingerprint": calib_hash,
            "target_runtime": ["openvino>=2025.0"],
            "target_hardware": {
                "cpu_isa_min": ["AVX-512 VNNI"],
                "cpu_isa_recommended": ["AMX-INT8"],
                "note": "INT8 fraction at runtime depends on plugin "
                        "fusion choices; verify with "
                        "compiled.get_runtime_model() -- see section 9.4.",
            },
        },
        "trt_int8_plan": {
            "kind": "tensorrt-plan-int8",
            "quantization": {
                "scheme": "static",
                "tool": "tensorrt.Builder (explicit QDQ)",
                "format": "Explicit (QDQ-in-ONNX)",
                "activation_dtype": "int8",
                "weight_dtype": "int8",
                "fallback_precision": "fp16",
            },
            "calibration_fingerprint": calib_hash,
            "target_runtime": ["tensorrt (matching builder major version)"],
            "target_hardware": {
                "gpu_compute_capability_min": (7, 5),
                "note": "TensorRT engines are SM- and TRT-major-version "
                        "specific. A plan built on sm_89 will not load on "
                        "sm_86; rebuild per target SKU. The compute "
                        "capability captured here is the build host's; "
                        "the load host must match.",
            },
        },
    }


def bert_recipes(calib_hash: Optional[str]) -> Dict[str, Dict[str, Any]]:
    return {
        "fp32_onnx": {
            "kind": "onnx-fp32",
            "quantization": {"scheme": "none"},
            "calibration_fingerprint": None,
            "target_runtime": [
                "onnxruntime>=1.18", "openvino>=2025.0", "tensorrt>=10.0",
            ],
            "target_hardware": {
                "cpu_isa_min": ["AVX2"],
                "cpu_isa_recommended": ["AVX-512", "AMX-BF16"],
            },
        },
        "qdq_int8_onnx": {
            "kind": "onnx-qdq-int8",
            "quantization": {
                "scheme": "static",
                "format": "QDQ",
                "activation_dtype": "int8",
                "weight_dtype": "int8",
                "activation_symmetric": True,
                "weight_per_channel": False,
                "calibration_method": "MinMax",
                "operators_quantized": ["MatMul"],
                "operators_excluded": "embedding Gather (input-adjacent), "
                                     "classifier MatMul (output-adjacent)",
                "calibration_num_samples": 64,  # 8 batches x 8
            },
            "calibration_fingerprint": calib_hash,
            "target_runtime": [
                "onnxruntime>=1.18 (CPU EP recommended; CUDA EP falls back "
                "on transformer subgraphs -- use TRT EP or TRT plan)",
            ],
            "target_hardware": {
                "cpu_isa_min": ["AVX-512 VNNI"],
                "cpu_isa_recommended": ["AMX-INT8"],
                "note": "Section 9.2 reported 14.4 ms on CUDA EP for this "
                        "artifact -- worse than the FP32-optimized graph -- "
                        "due to incomplete INT8 transformer kernel coverage.",
            },
        },
        "ov_fp32_ir": {
            "kind": "openvino-ir-fp32",
            "quantization": {"scheme": "none"},
            "calibration_fingerprint": None,
            "target_runtime": ["openvino>=2025.0"],
            "target_hardware": {
                "cpu_isa_min": ["AVX2"],
                "cpu_isa_recommended": ["AMX-BF16"],
                "note": "On Sapphire Rapids+ with AMX-BF16, BF16 inference "
                        "precision delivers ~5.8x over FP32 (section 9.4 "
                        "Figure 9.9) with no calibration data.",
            },
        },
        "ov_int8_ir": {
            "kind": "openvino-ir-int8",
            "quantization": {
                "scheme": "static",
                "tool": "nncf.quantize",
                "preset": "MIXED",
                "model_type_hint": "transformer",
                "fast_bias_correction": True,
                "operators_excluded": "embedding Gather (input-adjacent), "
                                     "classifier MatMul (output-adjacent)",
                "calibration_num_samples": 64,
            },
            "calibration_fingerprint": calib_hash,
            "target_runtime": ["openvino>=2025.0"],
            "target_hardware": {
                "cpu_isa_min": ["AVX-512 VNNI"],
                "cpu_isa_recommended": ["AMX-INT8"],
                "note": "Section 9.4 measured 110 ms on Sapphire Rapids "
                        "(6.4x over FP32) for this artifact at batch 8, "
                        "seq 128.",
            },
        },
        "trt_int8_plan": {
            "kind": "tensorrt-plan-int8",
            "quantization": {
                "scheme": "static",
                "tool": "tensorrt.Builder (explicit QDQ)",
                "format": "Explicit (QDQ-in-ONNX)",
                "activation_dtype": "int8",
                "weight_dtype": "int8",
                "fallback_precision": "fp16",
            },
            "calibration_fingerprint": calib_hash,
            "target_runtime": ["tensorrt (matching builder major version)"],
            "target_hardware": {
                "gpu_compute_capability_min": (7, 5),
                "note": "Section 9.3 measured 2.73 ms on L4 (sm_89) for "
                        "this artifact at batch 8, seq 128.",
            },
        },
    }


# --- Triton config.pbtxt emission -----------------------------------------
# Triton Inference Server's model-repository layout is a defensible
# convention even for teams that don't run Triton: it forces the artifact
# to ship with an explicit input/output schema and a backend declaration,
# both of which would otherwise live as tribal knowledge in the serving
# code. Writing config.pbtxt as part of packaging means the artifact and
# the schema travel together.

# ONNX dtype name -> Triton TYPE_* enum
TRITON_DTYPE_MAP = {
    "float32": "TYPE_FP32", "float": "TYPE_FP32", "f32": "TYPE_FP32",
    "float16": "TYPE_FP16", "fp16": "TYPE_FP16", "f16": "TYPE_FP16",
    "bfloat16": "TYPE_BF16", "bf16": "TYPE_BF16",
    "int64": "TYPE_INT64", "i64": "TYPE_INT64",
    "int32": "TYPE_INT32", "i32": "TYPE_INT32",
    "int8": "TYPE_INT8",   "i8": "TYPE_INT8",
    "uint8": "TYPE_UINT8", "u8": "TYPE_UINT8",
    "bool": "TYPE_BOOL",
    "float64": "TYPE_FP64",
}


def _triton_dtype(dtype_str: str) -> str:
    """Map a manifest dtype string to Triton's TYPE_* enum. Defaults to
    TYPE_FP32 with a warning prefix if the dtype is unknown."""
    s = (dtype_str or "").lower().strip()
    return TRITON_DTYPE_MAP.get(s, f"TYPE_FP32  /* unknown dtype: {dtype_str} */")


def _shape_minus_batch(shape: List[Any]) -> List[Any]:
    """Strip the leading batch dim. Triton's max_batch_size handles it."""
    if not shape:
        return []
    rest = shape[1:]
    # Replace any remaining symbolic dims with -1 (Triton's "any size")
    return [(d if isinstance(d, int) else -1) for d in rest]


def _format_dims(dims: List[Any]) -> str:
    return "[ " + ", ".join(str(d) for d in dims) + " ]"


def _format_io_block(io_list: List[Dict[str, Any]],
                     keyword: str) -> str:
    """Produce the input [ ... ] / output [ ... ] block."""
    entries = []
    for io in io_list:
        dt = _triton_dtype(io.get("dtype", ""))
        dims = _shape_minus_batch(io.get("shape", []))
        if not dims:
            dims = [1]   # Triton requires at least one dim
        entries.append(
            f"  {{\n"
            f'    name: "{io["name"]}"\n'
            f"    data_type: {dt}\n"
            f"    dims: {_format_dims(dims)}\n"
            f"  }}"
        )
    return f"{keyword} [\n" + ",\n".join(entries) + "\n]\n"


def emit_triton_config_onnx(
    manifest: Dict[str, Any],
    model_name: str,
    max_batch_size: int,
    use_openvino_ep: bool = False,
    use_cuda_ep: bool = False,
) -> str:
    """Generate config.pbtxt for ORT-served ONNX. The use_openvino_ep
    and use_cuda_ep flags activate Triton's execution-accelerator
    blocks so the same ONNX file can be routed to different EPs by
    config -- the by-config-file analog of the providers=[] list in
    section 9.2."""
    sigs = manifest["signatures"]
    inputs = _format_io_block(sigs["inputs"], "input")
    outputs = _format_io_block(sigs["outputs"], "output")

    instance_group_kind = "KIND_GPU" if use_cuda_ep else "KIND_CPU"
    accelerators = ""
    if use_openvino_ep:
        accelerators = (
            "optimization {\n"
            "  execution_accelerators {\n"
            "    cpu_execution_accelerator: [\n"
            "      { name: \"openvino\" }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        )
    elif use_cuda_ep:
        accelerators = (
            "optimization {\n"
            "  execution_accelerators {\n"
            "    gpu_execution_accelerator: [\n"
            "      { name: \"tensorrt\"\n"
            "        parameters { key: \"precision_mode\" value: \"INT8\" }\n"
            "        parameters { key: \"int8_calibration_table_name\" "
            "value: \"\" }\n"
            "      }\n"
            "    ]\n"
            "  }\n"
            "}\n"
        )

    return (
        f'name: "{model_name}"\n'
        f'platform: "onnxruntime_onnx"\n'
        f"max_batch_size: {max_batch_size}\n"
        f"\n"
        f"{inputs}"
        f"{outputs}"
        f"\n"
        f"{accelerators}"
        f"instance_group [\n"
        f"  {{\n"
        f"    count: 1\n"
        f"    kind: {instance_group_kind}\n"
        f"  }}\n"
        f"]\n"
    )


def emit_triton_config_openvino(
    manifest: Dict[str, Any],
    model_name: str,
    max_batch_size: int,
) -> str:
    """Generate config.pbtxt for OpenVINO-served IR."""
    sigs = manifest["signatures"]
    inputs = _format_io_block(sigs["inputs"], "input")
    outputs = _format_io_block(sigs["outputs"], "output")

    return (
        f'name: "{model_name}"\n'
        f'platform: "openvino"\n'
        f"max_batch_size: {max_batch_size}\n"
        f"\n"
        f"{inputs}"
        f"{outputs}"
        f"\n"
        f"parameters {{\n"
        f'  key: "PERFORMANCE_HINT"\n'
        f'  value: {{ string_value: "LATENCY" }}\n'
        f"}}\n"
        f"\n"
        f"instance_group [\n"
        f"  {{\n"
        f"    count: 1\n"
        f"    kind: KIND_CPU\n"
        f"  }}\n"
        f"]\n"
    )


def emit_triton_config_tensorrt(
    manifest: Dict[str, Any],
    model_name: str,
    max_batch_size: int,
) -> str:
    """Generate config.pbtxt for TensorRT-served plan files."""
    sigs = manifest["signatures"]

    # TRT plan signatures come from engine probe and may be empty if
    # the probe couldn't deserialize. Fall back to the ONNX-derived
    # signatures in that case (caller passes the manifest of the
    # source ONNX if needed).
    if not sigs.get("inputs") or not sigs.get("outputs"):
        return (
            f'name: "{model_name}"\n'
            f'platform: "tensorrt_plan"\n'
            f"max_batch_size: {max_batch_size}\n"
            f"\n"
            f"# NOTE: input/output signature could not be probed from\n"
            f"# the .plan file (CUDA/TRT runtime may have been absent\n"
            f"# at packaging time). Populate input[] and output[] from\n"
            f"# the source ONNX manifest before deploying.\n"
            f"\n"
            f"instance_group [\n"
            f"  {{ count: 1\n    kind: KIND_GPU\n  }}\n"
            f"]\n"
        )

    inputs = _format_io_block(sigs["inputs"], "input")
    outputs = _format_io_block(sigs["outputs"], "output")
    return (
        f'name: "{model_name}"\n'
        f'platform: "tensorrt_plan"\n'
        f"max_batch_size: {max_batch_size}\n"
        f"\n"
        f"{inputs}"
        f"{outputs}"
        f"\n"
        f"instance_group [\n"
        f"  {{ count: 1\n    kind: KIND_GPU\n  }}\n"
        f"]\n"
    )


def emit_triton_repo_for_model(
    model_name: str,
    artifacts: ArtifactSet,
    manifests: Dict[str, Dict[str, Any]],
    max_batch_size: int,
    repo_root: Path,
):
    """Write a complete Triton model-repository layout for this model.

    Layout (one directory per backend, version "1" inside each):

        repo_root/
            <model>_ort/
                config.pbtxt
                manifest.json
                1/
                    model.onnx
            <model>_ort_int8/
                config.pbtxt
                manifest.json
                1/
                    model.onnx
            <model>_openvino_fp32/
                config.pbtxt
                manifest.json
                1/
                    model.xml
                    model.bin
            <model>_openvino_int8/
                ...
            <model>_tensorrt_int8/
                config.pbtxt
                manifest.json
                1/
                    model.plan

    The manifest is duplicated next to config.pbtxt so the deployment
    contract travels alongside the artifact. Triton itself ignores the
    manifest; it's there for the load-time verifier and for ops audit.
    """
    repo_root.mkdir(parents=True, exist_ok=True)
    written: List[Tuple[str, Path]] = []

    def _stage(subdir: str, src: Path, dst_filename: str,
               config: str, manifest: Dict[str, Any]):
        target = repo_root / subdir
        version_dir = target / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        # Stage artifact (and any sidecar)
        import shutil
        shutil.copyfile(src, version_dir / dst_filename)
        if src.suffix == ".xml":
            bin_src = src.with_suffix(".bin")
            if bin_src.exists():
                shutil.copyfile(bin_src,
                                version_dir / dst_filename.replace(".xml", ".bin"))
        # config.pbtxt
        with open(target / "config.pbtxt", "w") as f:
            f.write(config)
        # manifest.json
        with open(target / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        written.append((subdir, target))

    # ONNX FP32 served via ORT/CPU EP
    if artifacts.fp32_onnx and "fp32_onnx" in manifests:
        _stage(
            f"{model_name}_ort_fp32",
            artifacts.fp32_onnx, "model.onnx",
            emit_triton_config_onnx(manifests["fp32_onnx"],
                                    f"{model_name}_ort_fp32",
                                    max_batch_size),
            manifests["fp32_onnx"],
        )

    # ONNX INT8 QDQ served via ORT/CPU EP
    if artifacts.qdq_int8_onnx and "qdq_int8_onnx" in manifests:
        _stage(
            f"{model_name}_ort_int8",
            artifacts.qdq_int8_onnx, "model.onnx",
            emit_triton_config_onnx(manifests["qdq_int8_onnx"],
                                    f"{model_name}_ort_int8",
                                    max_batch_size),
            manifests["qdq_int8_onnx"],
        )

    # ONNX INT8 QDQ via ORT + OpenVINO EP -- same artifact, different routing
    if artifacts.qdq_int8_onnx and "qdq_int8_onnx" in manifests:
        _stage(
            f"{model_name}_ort_ovep_int8",
            artifacts.qdq_int8_onnx, "model.onnx",
            emit_triton_config_onnx(manifests["qdq_int8_onnx"],
                                    f"{model_name}_ort_ovep_int8",
                                    max_batch_size,
                                    use_openvino_ep=True),
            manifests["qdq_int8_onnx"],
        )

    # OpenVINO native IR FP32
    if artifacts.ov_fp32_ir and "ov_fp32_ir" in manifests:
        _stage(
            f"{model_name}_openvino_fp32",
            artifacts.ov_fp32_ir, "model.xml",
            emit_triton_config_openvino(manifests["ov_fp32_ir"],
                                        f"{model_name}_openvino_fp32",
                                        max_batch_size),
            manifests["ov_fp32_ir"],
        )

    # OpenVINO native IR INT8
    if artifacts.ov_int8_ir and "ov_int8_ir" in manifests:
        _stage(
            f"{model_name}_openvino_int8",
            artifacts.ov_int8_ir, "model.xml",
            emit_triton_config_openvino(manifests["ov_int8_ir"],
                                        f"{model_name}_openvino_int8",
                                        max_batch_size),
            manifests["ov_int8_ir"],
        )

    # TensorRT INT8 plan
    if artifacts.trt_int8_plan and "trt_int8_plan" in manifests:
        # Source ONNX manifest's signatures fill in the plan config when
        # the plan probe couldn't deserialize.
        plan_manifest = manifests["trt_int8_plan"]
        if not plan_manifest["signatures"].get("inputs") and \
                "qdq_int8_onnx" in manifests:
            plan_manifest = dict(plan_manifest)
            plan_manifest["signatures"] = manifests["qdq_int8_onnx"]["signatures"]
        _stage(
            f"{model_name}_tensorrt_int8",
            artifacts.trt_int8_plan, "model.plan",
            emit_triton_config_tensorrt(plan_manifest,
                                        f"{model_name}_tensorrt_int8",
                                        max_batch_size),
            plan_manifest,
        )

    print(f"  Triton model-repo: {repo_root.relative_to(SCRIPT_DIR)}")
    for name, path in written:
        print(f"    {name}/  ->  {path.relative_to(SCRIPT_DIR)}")


# --- Verifier matrix ------------------------------------------------------
# Six runtime/EP columns. The verifier dispatches each (artifact, runtime)
# cell to the right loader, runs a smoke-test pass, and where possible
# inspects the runtime graph to confirm the executed precision matches the
# manifest's claim.

RUNTIMES = [
    ("ort_cpu",   "ORT\nCPU EP"),
    ("ort_cuda",  "ORT\nCUDA EP"),
    ("ort_ovep",  "ORT\nOV EP"),
    ("ort_trtep", "ORT\nTRT EP"),
    ("ov_cpu",    "OpenVINO\nCore CPU"),
    ("trt_gpu",   "TensorRT\nruntime"),
]

ARTIFACT_KINDS = [
    ("fp32_onnx",     "FP32 ONNX"),
    ("qdq_int8_onnx", "INT8 QDQ ONNX"),
    ("ov_fp32_ir",    "OV IR FP32"),
    ("ov_int8_ir",    "OV IR INT8\n(NNCF)"),
    ("trt_int8_plan", "TRT INT8 plan"),
]


@contextlib.contextmanager
def _capture_fd2():
    """Redirect file descriptor 2 (the C/C++ stderr ORT logs to) to
    a temp file for the duration of the with-block. Yields a list
    that, on exit, contains a single string with everything written
    to fd 2.

    Why this exists: ORT's TRT EP logs build errors and per-subgraph
    fallbacks ("EP Error... Falling back to...") to the C-level
    stderr stream, not Python's sys.stderr. contextlib.redirect_stderr
    won't catch them. And get_providers() isn't a substitute either:
    when TRT EP fails to build engines for individual subgraphs, ORT
    routes those subgraphs through CUDA EP at runtime but still keeps
    TensorrtExecutionProvider in the registered list, so the cell
    reports OK when the actual workload ran on CUDA EP.

    Implementation note: this is best-effort. If anything in the
    with-block writes to Python's sys.stderr buffer, that buffer
    needs to be flushed before exit; we do so. Lines that bypass the
    buffer go straight to fd 2 and are captured directly."""
    captured = [""]
    old_fd = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), 2)
        try:
            yield captured
        finally:
            try:
                sys.stderr.flush()
            except Exception:
                pass
            try:
                os.fsync(2)
            except Exception:
                pass
            os.dup2(old_fd, 2)
            os.close(old_fd)
            tmp.seek(0)
            try:
                captured[0] = tmp.read().decode("utf-8", errors="replace")
            finally:
                tmp.close()
    except Exception:
        try:
            os.dup2(old_fd, 2)
            os.close(old_fd)
        except Exception:
            pass
        try:
            tmp.close()
        except Exception:
            pass
        raise


def _detect_ort_fallbacks(stderr_log: str) -> List[str]:
    """Scan a captured ORT stderr block for indicators that an EP
    failed to serve some or all of the workload. Returns a short list
    of human-readable indicators (empty list = clean run)."""
    indicators = []
    if "EP Error" in stderr_log and "Falling back" in stderr_log:
        indicators.append("session-level fallback")
    # Per-subgraph parser failures (TRT EP couldn't build engines for
    # specific Conv/MatMul subgraphs and routed them to CUDA EP)
    if "[6] Invalid Node" in stderr_log:
        n = stderr_log.count("[6] Invalid Node")
        indicators.append(f"{n} TRT-rejected nodes")
    if "TensorRT EP failed to create engine" in stderr_log:
        indicators.append("TRT engine build failed")
    if "Calibration failure" in stderr_log:
        indicators.append("TRT INT8 calibration failure")
    return indicators


def verify_ort_cell(
    artifact_path: Path,
    runtime_name: str,
    artifact_kind: str,
    sample_feed: Dict[str, np.ndarray],
    model_name: str = "",
) -> CellResult:
    """Run a smoke-test session on ONNX through one of ORT's EPs.

    The ORT EPs are opaque about per-op execution precision, so we
    can verify "loaded and ran" but not "executed at claimed
    precision" directly. We do report which provider actually
    resolved -- if the requested EP isn't in get_providers() output,
    we mark MISSING_RUNTIME with the resolved fallback noted.

    The §9.2 anti-pattern flag is conditioned on model_name: §9.2
    measured BERT INT8 on CUDA EP at 14.4ms (worse than FP32 at
    8.96ms on the same L4) due to incomplete transformer-kernel
    coverage. ResNet was not measured; CNNs typically don't hit
    that gap. We only fire DEMOTED on transformer-class models
    where §9.2 has actual evidence to point at.
    """
    if not has_ort():
        return CellResult(Status.BLOCKED, note="ORT not installed",
                          detail="onnxruntime package missing")

    import onnxruntime as ort
    available = ort_available_providers()

    # Map runtime_name to (requested provider list, hardware predicate)
    if runtime_name == "ort_cpu":
        requested = "CPUExecutionProvider"
        prov_list = ["CPUExecutionProvider"]
    elif runtime_name == "ort_cuda":
        requested = "CUDAExecutionProvider"
        if not has_cuda():
            return CellResult(Status.BLOCKED, note="no CUDA",
                              detail="No CUDA-capable GPU detected")
        prov_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif runtime_name == "ort_ovep":
        requested = "OpenVINOExecutionProvider"
        prov_list = [
            ("OpenVINOExecutionProvider",
             {"device_type": "CPU", "precision": "ACCURACY"}),
            "CPUExecutionProvider",
        ]
    elif runtime_name == "ort_trtep":
        requested = "TensorrtExecutionProvider"
        if not has_cuda():
            return CellResult(Status.BLOCKED, note="no CUDA",
                              detail="No CUDA-capable GPU detected")
        # trt_int8_enable=true on a non-quantized ONNX makes TRT EP try
        # implicit-style INT8 calibration without a calibrator, fail,
        # and fall back to CUDA. Only enable INT8 mode for explicitly
        # quantized artifacts (QDQ ONNX), where the scales are baked
        # into the graph and TRT honours them rather than re-running
        # calibration.
        trt_opts = {"trt_fp16_enable": "true"}
        if artifact_kind == "qdq_int8_onnx":
            trt_opts["trt_int8_enable"] = "true"
        prov_list = [
            ("TensorrtExecutionProvider", trt_opts),
            "CUDAExecutionProvider", "CPUExecutionProvider",
        ]
    else:
        return CellResult(Status.NA)

    if requested not in available:
        short_name = requested.replace("ExecutionProvider", " EP")
        return CellResult(Status.BLOCKED,
                          note=f"{short_name}\nnot\ninstalled",
                          detail=f"ORT providers available: {available}")

    try:
        so = ort.SessionOptions()
        so.log_severity_level = 3   # quiet (errors still emit)
        with _capture_fd2() as cap:
            sess = ort.InferenceSession(str(artifact_path), so,
                                        providers=prov_list)
            resolved = sess.get_providers()
            out_names = [o.name for o in sess.get_outputs()]
            outputs = sess.run(out_names, sample_feed)
        ort_log = cap[0]
    except Exception as e:
        return CellResult(Status.ERROR,
                          note=f"{type(e).__name__}",
                          detail=str(e)[:300])

    # Silent EP fallback detection. Two patterns to catch:
    #
    #   (1) Session-level fallback: ORT couldn't even register the
    #       requested EP (shared library missing). The requested
    #       provider name disappears from sess.get_providers().
    #   (2) Per-subgraph fallback: ORT registered the EP successfully,
    #       but the EP's parser/builder failed on individual subgraphs
    #       and ORT routed them to the next provider in the list at
    #       runtime. sess.get_providers() still includes the requested
    #       EP -- it would only be missing if the entire session
    #       failed. The signal lives in the stderr log instead.
    #
    # The chapter's whole pedagogy is that the artifact + runtime
    # combination decides what executes. Both forms of fallback look
    # like "OK" through the public Python API. We catch both here.
    if requested not in resolved:
        return CellResult(
            Status.BLOCKED,
            note=f"{requested.replace('ExecutionProvider', '')} EP\n"
                 f"silently fell\nback to "
                 f"{(resolved[0] if resolved else '?').replace('ExecutionProvider', '')}",
            detail=f"requested {requested}; ORT silently fell back to "
                   f"{resolved}. Most common cause: shared library "
                   f"missing on LD_LIBRARY_PATH. For TRT EP, ensure "
                   f"libnvinfer.so.10 (TRT 10.x) or libnvinfer.so.8 "
                   f"(TRT 8.x) is reachable.",
        )

    fallback_indicators = _detect_ort_fallbacks(ort_log)
    if fallback_indicators and requested == "TensorrtExecutionProvider":
        # The TRT EP was registered but ORT routed kernels to CUDA EP
        # at runtime. Mark DEMOTED -- the artifact loaded and ran, but
        # not on the EP we asked for.
        return CellResult(
            Status.DEMOTED,
            observed_precision="TRT partial; CUDA fallback",
            note=f"TRT EP routed\nto CUDA EP\n"
                 f"({fallback_indicators[0]})",
            detail=f"resolved providers: {resolved}; TRT EP failures "
                   f"detected in stderr: {fallback_indicators}; first "
                   f"500 chars of log: {ort_log[:500]!r}",
        )

    primary = resolved[0] if resolved else "?"
    short = primary.replace("ExecutionProvider", "")
    detail = f"resolved providers: {resolved}; output shapes: " + \
             ", ".join(str(o.shape) for o in outputs)

    # Section-9.2 anti-pattern flag for INT8 ONNX on CUDA EP. §9.2
    # measured this on BERT and observed 14.4 ms vs the 8.96 ms FP32-
    # optimized graph on the same L4 -- attributed to incomplete
    # transformer-kernel coverage in CUDA EP, with Memcpy fallbacks
    # to CPU for unsupported INT8 ops. The same finding does NOT
    # generalize to CNNs: ResNet-class INT8 Conv kernels are well
    # covered by CUDA EP, and §9.2 has no measurement to demote them
    # against. We fire the DEMOTED+pointer only on models whose name
    # signals transformer architecture, leaving CNNs as plain OK so
    # the manuscript doesn't fabricate evidence §9.2 never collected.
    is_transformer = ("bert" in model_name.lower()
                      or "gpt" in model_name.lower()
                      or "transformer" in model_name.lower()
                      or "llama" in model_name.lower())
    if (artifact_kind == "qdq_int8_onnx"
            and runtime_name == "ort_cuda"
            and is_transformer):
        return CellResult(
            Status.DEMOTED,
            observed_precision="partial i8 + f32",
            note="partial INT8\n→ §9.2 anti-pattern",
            detail=detail + "; §9.2 measured this artifact-runtime pair "
                            "at 14.4 ms vs 8.96 ms FP32-optimized on the "
                            "same L4 (incomplete INT8 transformer-kernel "
                            "coverage in CUDA EP)",
        )

    return CellResult(Status.OK, observed_precision=short,
                      note=f"{short}", detail=detail)


def verify_ov_cell(
    artifact_path: Path,
    artifact_kind: str,
    sample_feed: Dict[str, np.ndarray],
) -> CellResult:
    """Load through ov.Core().compile_model and walk the runtime graph
    to bucket compute ops by execution precision.

    This verifier reuses the section 9.4 inspector pattern. For an
    INT8-claimed artifact, OK requires INT8 ops in the runtime graph;
    if every compute op runs at f32/bf16, we mark DEMOTED."""
    ov = get_ov()
    if ov is None:
        return CellResult(Status.BLOCKED, note="OV not installed")

    try:
        core = ov.Core()
        # For ONNX files, ov.Core can compile directly. For IR pairs,
        # the .xml path is sufficient (the .bin sidecar is loaded
        # implicitly).
        compiled = core.compile_model(
            str(artifact_path), "CPU",
            config={"PERFORMANCE_HINT": "LATENCY"},
        )
        # Smoke test: build inputs from sample_feed (works for either
        # ONNX-derived or IR-derived port names since OV preserves
        # them)
        request = compiled.create_infer_request()
        try:
            outputs = request.infer(sample_feed)
        except Exception:
            # Fall back to positional input -- some OV builds key
            # inputs by index when names disagree
            ordered = [sample_feed[k]
                       for k in sorted(sample_feed.keys())]
            outputs = request.infer(ordered)
    except Exception as e:
        return CellResult(Status.ERROR,
                          note=f"{type(e).__name__}",
                          detail=str(e)[:300])

    # Inspect runtime model
    try:
        counts = _count_ov_runtime_precisions(compiled)
    except Exception as e:
        return CellResult(Status.OK, observed_precision="?",
                          note="loaded OK\n(inspect failed)",
                          detail=f"smoke pass succeeded; runtime inspect "
                                 f"raised {type(e).__name__}: {e}")

    int8 = counts.get("i8", 0) + counts.get("u8", 0)
    bf16 = counts.get("bf16", 0)
    f32 = counts.get("f32", 0)
    f16 = counts.get("f16", 0)
    total = max(1, sum(v for v in counts.values()))
    int8_frac = int8 / total

    # OK / DEMOTED logic by claim
    if artifact_kind in ("ov_int8_ir", "qdq_int8_onnx"):
        if int8 == 0:
            return CellResult(
                Status.DEMOTED,
                observed_precision="bf16/f32",
                note=f"INT8 claimed\nbut none executed",
                detail=f"runtime ops by precision: {counts}",
            )
        elif int8_frac < 0.05:
            return CellResult(
                Status.DEMOTED,
                observed_precision=f"i8 {int8_frac:.0%}",
                note=f"INT8 fraction\n{int8_frac:.0%}",
                detail=f"runtime ops by precision: {counts}",
            )
        else:
            return CellResult(
                Status.OK,
                observed_precision=f"i8/u8 {int8_frac:.0%}",
                note=f"i8: {int8_frac:.0%}\nbf16: {bf16/total:.0%}",
                detail=f"runtime ops by precision: {counts}",
            )

    # FP32-claimed artifact: OK if compute is f32 (or bf16 -- the OV
    # CPU plugin auto-promotes elementwise ops to bf16 on AMX silicon
    # when it's faster, even from an FP32 IR; this is documented and
    # not a contract violation as long as INT8 wasn't requested)
    if f32 > 0 or bf16 > 0:
        return CellResult(
            Status.OK,
            observed_precision=f"f32/bf16",
            note=f"f32: {f32/total:.0%}\nbf16: {bf16/total:.0%}",
            detail=f"runtime ops by precision: {counts}",
        )
    return CellResult(Status.OK, observed_precision="?",
                      note="loaded OK",
                      detail=f"runtime ops by precision: {counts}")


def _count_ov_runtime_precisions(compiled) -> Dict[str, int]:
    """Walk the OV runtime graph and bucket compute ops by executed
    precision. Mirrors the section 9.4 inspector but trims it to a
    compact return value (counts dict)."""
    BOOKKEEPING = {
        "Constant", "Parameter", "Result",
        "ShapeOf", "Reshape", "Transpose",
        "Squeeze", "Unsqueeze", "Concat", "Gather",
        "Convert", "Broadcast", "Reorder", "MemoryReorder",
    }
    PREC_MAP = {
        "f32": "f32", "fp32": "f32", "float": "f32", "float32": "f32",
        "f16": "f16", "fp16": "f16", "float16": "f16",
        "bf16": "bf16", "bfloat16": "bf16",
        "i8": "i8",  "int8": "i8",
        "u8": "u8",  "uint8": "u8",
        "i32": "i32", "int32": "i32",
        "i64": "i64", "int64": "i64",
    }
    SUBSTR_KEYS = sorted(PREC_MAP.keys(), key=len, reverse=True)

    rt_model = compiled.get_runtime_model()
    counts: Dict[str, int] = {}

    for op in rt_model.get_ordered_ops():
        rt_dict = {}
        try:
            for key, value in op.get_rt_info().items():
                # Pull a string out of OVAny across versions
                s = str(value).strip()
                if "OVAny" in s:
                    for accessor in ("get", "value"):
                        try:
                            attr = getattr(value, accessor, None)
                            if attr is None:
                                continue
                            result = attr() if callable(attr) else attr
                            s = str(result).strip()
                            if "OVAny" not in s:
                                break
                        except Exception:
                            continue
                rt_dict[str(key)] = s
        except Exception:
            pass

        original_op_type = rt_dict.get("layerType", "")
        if not original_op_type:
            try:
                original_op_type = op.get_type_name()
            except Exception:
                original_op_type = ""
        if original_op_type in BOOKKEEPING:
            continue

        prec = ""
        for k in ("runtimePrecision", "runtime_precision",
                  "execPrecision", "exec_precision",
                  "outputPrecisions", "primitivePrecision"):
            v = rt_dict.get(k, "")
            if v:
                prec = v
                break
        prec_lower = prec.lower()
        bucket = None
        if prec_lower in PREC_MAP:
            bucket = PREC_MAP[prec_lower]
        else:
            for k in SUBSTR_KEYS:
                if k in prec_lower:
                    bucket = PREC_MAP[k]
                    break
        if bucket is None:
            bucket = "other"
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def verify_trt_cell(
    artifact_path: Path,
    artifact_kind: str,
    sample_feed: Dict[str, np.ndarray],
) -> CellResult:
    """Deserialize a TRT plan and run a smoke pass to confirm it loads
    and the I/O dtypes match the recorded manifest. Per-layer execution
    precision needs the engine inspector for full coverage; we surface
    a coarse summary here (input/output dtype + total layer count)
    that's enough to detect catastrophic mismatches (plan loaded, but
    is FP32-only despite an INT8 claim)."""
    trt = get_trt()
    if trt is None:
        return CellResult(Status.BLOCKED, note="tensorrt\nnot installed")
    if not has_cuda():
        return CellResult(Status.BLOCKED, note="no CUDA")

    try:
        with open(artifact_path, "rb") as f:
            plan = f.read()
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(plan)
        if engine is None:
            return CellResult(Status.ERROR,
                              note="deserialize\nreturned None")
    except Exception as e:
        # Common cause: plan was built on a different SM than the
        # current GPU. Surface the SM mismatch when we can detect it.
        cap = cuda_compute_capability()
        cap_str = f"sm_{cap[0]}{cap[1]}" if cap else "?"
        return CellResult(
            Status.ERROR,
            note=f"deserialize\nfailed",
            detail=f"{type(e).__name__}: {e} (current device: {cap_str})",
        )

    try:
        # Engine inspector: per-layer execution dtypes
        inspector = engine.create_engine_inspector()
        try:
            layer_info = inspector.get_engine_information(
                trt.LayerInformationFormat.JSON)
            li = json.loads(layer_info)
            layer_count = len(li.get("Layers", []))
            int8_count = sum(
                1 for l in li.get("Layers", [])
                if "Int8" in str(l.get("Inputs", ""))
                or "INT8" in str(l.get("Inputs", ""))
                or "int8" in str(l.get("Inputs", "")).lower()
            )
        except Exception:
            layer_count, int8_count = 0, 0

        if layer_count == 0:
            return CellResult(Status.OK, observed_precision="?",
                              note="loaded OK\n(inspect skipped)",
                              detail="engine deserialized; "
                                     "inspector not populated "
                                     "(profiling_verbosity at build was "
                                     "likely default rather than DETAILED)")
        int8_frac = int8_count / max(1, layer_count)
        if artifact_kind == "trt_int8_plan" and int8_count == 0:
            return CellResult(
                Status.DEMOTED,
                observed_precision="no int8 layers",
                note="INT8 claimed\nbut none in plan",
                detail=f"layers={layer_count}, int8_inputs={int8_count}",
            )
        return CellResult(
            Status.OK,
            observed_precision=f"int8 {int8_frac:.0%}",
            note=f"int8: {int8_frac:.0%}\nlayers: {layer_count}",
            detail=f"layers={layer_count}, int8_input_layers={int8_count}",
        )
    except Exception as e:
        return CellResult(Status.OK, observed_precision="?",
                          note="loaded OK\n(inspect failed)",
                          detail=f"{type(e).__name__}: {e}")


def verify_cell(
    artifact_kind_key: str,        # key in ArtifactSet
    runtime_name: str,
    artifact_path: Optional[Path],
    sample_feed: Dict[str, np.ndarray],
    model_name: str = "",          # "resnet18" or "bert_sst2"
) -> CellResult:
    """Top-level dispatcher. Empty cells (artifact format not
    consumable by this runtime) return Status.NA.

    model_name lets the dispatcher route the §9.2-anti-pattern flag
    only to models §9.2 actually measured (BERT). Fabricating a
    DEMOTED status on a CNN where the source chapter has no data
    would be exactly the kind of made-up evidence the manuscript
    rules forbid.
    """
    # Map artifact_kind_key -> manifest "kind" string for verifier
    KIND_TO_MANIFEST = {
        "fp32_onnx":      "onnx-fp32",
        "qdq_int8_onnx":  "onnx-qdq-int8",
        "ov_fp32_ir":     "openvino-ir-fp32",
        "ov_int8_ir":     "openvino-ir-int8",
        "trt_int8_plan":  "tensorrt-plan-int8",
    }
    manifest_kind = KIND_TO_MANIFEST.get(artifact_kind_key, "unknown")

    # NA cells -- artifact format not a load operation for this runtime
    if artifact_kind_key in ("ov_fp32_ir", "ov_int8_ir") \
            and runtime_name != "ov_cpu":
        return CellResult(Status.NA)
    if artifact_kind_key == "trt_int8_plan" and runtime_name != "trt_gpu":
        return CellResult(Status.NA)
    if artifact_kind_key in ("fp32_onnx", "qdq_int8_onnx") \
            and runtime_name == "trt_gpu":
        # Loading ONNX into TRT runtime requires building an engine
        # first -- a build operation, not a load. Mark NA.
        return CellResult(Status.NA)

    # Artifact missing
    if artifact_path is None or not artifact_path.exists():
        return CellResult(Status.BLOCKED, note="artifact\nmissing",
                          detail="re-run section 9.3 / 9.4 build first")

    # Dispatch
    if runtime_name.startswith("ort_"):
        return verify_ort_cell(artifact_path, runtime_name,
                               artifact_kind_key, sample_feed,
                               model_name=model_name)
    if runtime_name == "ov_cpu":
        return verify_ov_cell(artifact_path, artifact_kind_key, sample_feed)
    if runtime_name == "trt_gpu":
        return verify_trt_cell(artifact_path, artifact_kind_key, sample_feed)
    return CellResult(Status.NA)


# --- Compatibility-matrix plotting ---------------------------------------

def _draw_matrix_panel(ax, model_label: str,
                       rows: List[Tuple[str, str]],
                       cols: List[Tuple[str, str]],
                       cells: Dict[Tuple[str, str], CellResult]):
    """Draw one matrix panel inside ax. rows is [(key, label), ...] for
    artifact kinds; cols is [(key, label), ...] for runtime columns.

    Layout notes:
      - aspect="auto" (not "equal"): cells are wider than tall, which
        gives multi-line cell notes the horizontal room they need.
        With set_aspect("equal") and figure width 7.6", cells were
        ~1.0" square which couldn't fit "silently fell back to CUDA"
        without overflow.
      - clip_box on text patches: long notes get clipped to their
        cell's bounding rectangle rather than spilling into neighbours.
    """
    n_rows, n_cols = len(rows), len(cols)

    # Cell geometry: y inverted so first row is at the top
    for r_i, (r_key, r_label) in enumerate(rows):
        for c_i, (c_key, c_label) in enumerate(cols):
            cell = cells.get((r_key, c_key), CellResult(Status.NA))
            face, hatch, glyph = STATUS_STYLE[cell.status]
            edge = "black" if cell.status != Status.NA else "#cccccc"
            lw = 0.5 if cell.status != Status.NA else 0.3

            rect = plt.Rectangle(
                (c_i, n_rows - 1 - r_i), 1, 1,
                facecolor=face, edgecolor=edge,
                linewidth=lw, hatch=hatch,
            )
            ax.add_patch(rect)
            # Glyph + short text inside cell. On hatched cells (DEMOTED,
            # ERROR), the hatch lines cross through the text and hurt
            # readability. We give the glyph and the note a bbox in the
            # cell's own face colour so they sit in a hatch-free pocket
            # of the cell. Edge of the cell still shows the hatch, so
            # the grayscale-print distinction is preserved -- the
            # signal is now the hatch around the text, not under it.
            cx, cy = c_i + 0.5, n_rows - 1 - r_i + 0.5
            has_hatch = bool(hatch)
            note_bbox = (
                dict(facecolor=face, alpha=1.0, edgecolor="none",
                     boxstyle="round,pad=0.18")
                if has_hatch else None
            )
            glyph_bbox = (
                dict(facecolor=face, alpha=1.0, edgecolor="none",
                     boxstyle="circle,pad=0.05")
                if has_hatch else None
            )
            if glyph:
                t = ax.text(cx, cy + 0.28, glyph,
                            ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="#222", bbox=glyph_bbox)
                t.set_clip_path(rect)
            if cell.note:
                t = ax.text(cx, cy - 0.18, cell.note,
                            ha="center", va="center", fontsize=5.5,
                            color="#222", linespacing=1.05,
                            bbox=note_bbox)
                t.set_clip_path(rect)

    # Row labels (artifact kinds), tighter to the panel
    for r_i, (_, r_label) in enumerate(rows):
        ax.text(-0.08, n_rows - 1 - r_i + 0.5, r_label,
                ha="right", va="center", fontsize=7,
                linespacing=1.05)

    # Column headers
    for c_i, (_, c_label) in enumerate(cols):
        ax.text(c_i + 0.5, n_rows + 0.08, c_label,
                ha="center", va="bottom", fontsize=6.5,
                linespacing=1.05)

    ax.set_xlim(-0.05, n_cols + 0.05)
    ax.set_ylim(-0.05, n_rows + 1.0)
    ax.set_aspect("auto")    # cells can be wider than tall
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(model_label, fontsize=9, loc="left", pad=12,
                 fontweight="bold")


def _legend_handles() -> List[mpatches.Patch]:
    """Five-status legend, ordered for left-to-right reading."""
    order = [
        (Status.OK,      "executed at claimed precision"),
        (Status.DEMOTED, "loaded but precision fell back"),
        (Status.BLOCKED, "runtime / hardware / artifact absent"),
        (Status.ERROR,   "load or smoke-test threw"),
        (Status.NA,      "not applicable (build required)"),
    ]
    handles = []
    for status, label in order:
        face, hatch, _ = STATUS_STYLE[status]
        handles.append(mpatches.Patch(
            facecolor=face, edgecolor="black",
            hatch=hatch, label=label, linewidth=0.5,
        ))
    return handles


def plot_compatibility_matrix(
    resnet_cells: Dict[Tuple[str, str], CellResult],
    bert_cells: Dict[Tuple[str, str], CellResult],
    config: Config,
    figure_name: str = "CH09_F10_Kalyanarangan_packaging_matrix",
):
    """Two-panel compatibility matrix. ResNet-18 on top, BERT below.
    Both panels share the same row/column structure -- the chapter's
    point is that the matrix shape is universal even though specific
    cell outcomes differ by model and by hardware."""
    apply_manning_style()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.6, 5.2))

    _draw_matrix_panel(ax_top, "ResNet-18 (CNN)",
                       ARTIFACT_KINDS, RUNTIMES, resnet_cells)
    _draw_matrix_panel(ax_bot, "BERT-base SST-2 (transformer)",
                       ARTIFACT_KINDS, RUNTIMES, bert_cells)

    fig.legend(handles=_legend_handles(),
               loc="lower center", ncol=3, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(left=0.16, right=0.99, top=0.93, bottom=0.10,
                        hspace=0.30)
    save_or_show(fig, figure_name, config)


def print_matrix_text(model_label: str,
                      cells: Dict[Tuple[str, str], CellResult]):
    """Compact text rendering of one model's matrix for terminal output.
    Useful when the user runs without --save-plots."""
    print(f"\n  Compatibility matrix for {model_label}:")
    col_keys = [c[0] for c in RUNTIMES]
    col_labels_short = [c[1].replace("\n", " ") for c in RUNTIMES]

    # Header
    print("    " + " " * 22 + " ".join(f"{l[:11]:<11}"
                                       for l in col_labels_short))
    for r_key, r_label in ARTIFACT_KINDS:
        line = f"    {r_label[:20]:<22}"
        for c_key in col_keys:
            cell = cells.get((r_key, c_key), CellResult(Status.NA))
            tag = {
                Status.OK:      "✓ OK",
                Status.DEMOTED: "↑ FALLBACK",
                Status.BLOCKED: "·  blocked",
                Status.ERROR:   "✗ error",
                Status.NA:      "-  -",
            }[cell.status]
            line += f"{tag[:11]:<11} "
        print(line)
    print()


# --- Per-model orchestrator ----------------------------------------------

def package_and_verify_model(
    model_name: str,                # "resnet18" or "bert_sst2"
    model_id: str,                  # "torchvision/resnet18"
    task: str,                      # "image-classification"
    artifacts: ArtifactSet,
    recipes: Dict[str, Dict[str, Any]],
    sample_feed: Dict[str, np.ndarray],
    calib_fingerprint: Optional[str],
    max_batch_size: int,
    config: Config,
) -> Tuple[Dict[str, Dict[str, Any]],
           Dict[Tuple[str, str], CellResult]]:
    """Per-model: emit manifests + Triton repo + verify matrix.
    Returns (manifests_by_kind_key, cell_results)."""
    pkg_root = config.package_dir / model_name
    pkg_root.mkdir(parents=True, exist_ok=True)

    # 1. Emit manifests for every artifact present
    print(f"\n  Manifests for {model_name}:")
    manifests: Dict[str, Dict[str, Any]] = {}
    for kind_key, recipe in recipes.items():
        artifact_path = getattr(artifacts, kind_key)
        if artifact_path is None or not artifact_path.exists():
            continue
        manifest = build_manifest(
            model_id=model_id, task=task,
            artifact_path=artifact_path,
            artifact_kind=recipe["kind"],
            quantization=recipe.get("quantization"),
            calibration_fingerprint=recipe.get("calibration_fingerprint"),
            target_runtime=recipe.get("target_runtime"),
            target_hardware=recipe.get("target_hardware"),
        )
        manifests[kind_key] = manifest
        write_manifest(manifest,
                       pkg_root / "manifests" / f"{kind_key}.manifest.json")

    # 2. Emit Triton model-repository scaffolding
    triton_root = pkg_root / "triton_model_repo"
    print(f"\n  Triton model-repo for {model_name}:")
    emit_triton_repo_for_model(model_name, artifacts, manifests,
                               max_batch_size, triton_root)

    # 3. Run the verification matrix
    print(f"\n  Verification matrix for {model_name}:")
    cells: Dict[Tuple[str, str], CellResult] = {}
    for art_key, art_label in ARTIFACT_KINDS:
        artifact_path = getattr(artifacts, art_key)
        for rt_key, rt_label in RUNTIMES:
            result = verify_cell(art_key, rt_key, artifact_path,
                                 sample_feed, model_name=model_name)
            cells[(art_key, rt_key)] = result
            tag = {
                Status.OK: "OK",      Status.DEMOTED: "FALLBACK",
                Status.BLOCKED: "BLK", Status.ERROR: "ERR",
                Status.NA: "-",
            }[result.status]
            label = rt_label.replace("\n", " ")
            print(f"    {art_label:<22} | {label:<22}: "
                  f"{tag:<10} {result.note.replace(chr(10), ' ')[:40]}")

    # 4. Persist verify_report.json
    report = {
        "schema_version": SCHEMA_VERSION,
        "model": {"id": model_id, "name": model_name, "task": task},
        "host": platform.node(),
        "platform": platform.system() + " " + platform.machine(),
        "ov_optimization_capabilities": ov_optimization_capabilities(),
        "cuda_device": cuda_device_name(),
        "cuda_compute_capability": cuda_compute_capability(),
        "ort_providers": ort_available_providers(),
        "verification_timestamp": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "matrix": [
            {
                "artifact": art_key,
                "runtime":  rt_key,
                "status":   cells[(art_key, rt_key)].status.value,
                "observed_precision":
                    cells[(art_key, rt_key)].observed_precision,
                "note":     cells[(art_key, rt_key)].note,
                "detail":   cells[(art_key, rt_key)].detail,
            }
            for art_key, _ in ARTIFACT_KINDS
            for rt_key, _ in RUNTIMES
        ],
    }
    report_path = pkg_root / "verify_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Wrote: {report_path.relative_to(SCRIPT_DIR)}")

    print_matrix_text(model_name, cells)
    return manifests, cells


# --- Main entry points per model ----------------------------------------

def run_resnet(config: Config) -> Dict[Tuple[str, str], CellResult]:
    print("\n" + "-" * 72)
    print("ResNet-18 — package + verify")
    print("-" * 72)
    artifacts = discover_or_build_resnet(config)
    print(f"  FP32 ONNX:    {_status_string(artifacts.fp32_onnx)}")
    print(f"  QDQ INT8:     {_status_string(artifacts.qdq_int8_onnx)}")
    print(f"  OV FP32 IR:   {_status_string(artifacts.ov_fp32_ir)}")
    print(f"  OV INT8 IR:   {_status_string(artifacts.ov_int8_ir)}")
    print(f"  TRT plan:     {_status_string(artifacts.trt_int8_plan)}")

    sample_feed = _resnet_calib_feeds(1, config.resnet_batch,
                                      config.resnet_image_size,
                                      config.seed)[0]

    # Calibration fingerprint over the canonical 8x32 batch set
    calib_feeds = _resnet_calib_feeds(config.num_calib_batches,
                                      config.resnet_batch,
                                      config.resnet_image_size, config.seed)
    calib_hash = "sha256:" + hash_feeds(calib_feeds)

    recipes = resnet_recipes(calib_hash)
    _, cells = package_and_verify_model(
        model_name="resnet18",
        model_id="torchvision/resnet18",
        task="image-classification",
        artifacts=artifacts,
        recipes=recipes,
        sample_feed=sample_feed,
        calib_fingerprint=calib_hash,
        max_batch_size=config.resnet_batch,
        config=config,
    )
    return cells


def run_bert(config: Config) -> Dict[Tuple[str, str], CellResult]:
    print("\n" + "-" * 72)
    print("BERT-base SST-2 — package + verify")
    print("-" * 72)
    artifacts = discover_or_build_bert(config)
    print(f"  FP32 ONNX:    {_status_string(artifacts.fp32_onnx)}")
    print(f"  QDQ INT8:     {_status_string(artifacts.qdq_int8_onnx)}")
    print(f"  OV FP32 IR:   {_status_string(artifacts.ov_fp32_ir)}")
    print(f"  OV INT8 IR:   {_status_string(artifacts.ov_int8_ir)}")
    print(f"  TRT plan:     {_status_string(artifacts.trt_int8_plan)}")

    sample_feed = _bert_calib_feeds(1, config.bert_batch,
                                    config.bert_seq_len, config.seed)[0]
    calib_feeds = _bert_calib_feeds(config.num_calib_batches,
                                    config.bert_batch,
                                    config.bert_seq_len, config.seed)
    calib_hash = "sha256:" + hash_feeds(calib_feeds)

    recipes = bert_recipes(calib_hash)
    _, cells = package_and_verify_model(
        model_name="bert_sst2",
        model_id="textattack/bert-base-uncased-SST-2",
        task="sequence-classification",
        artifacts=artifacts,
        recipes=recipes,
        sample_feed=sample_feed,
        calib_fingerprint=calib_hash,
        max_batch_size=config.bert_batch,
        config=config,
    )
    return cells


def _status_string(p: Optional[Path]) -> str:
    if p is None:
        return "absent"
    try:
        size_mb = p.stat().st_size / 1e6
        return f"{p.relative_to(SCRIPT_DIR)}  ({size_mb:.1f} MB)"
    except Exception:
        return str(p)


# --- Argument parsing and main -------------------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Ch9 sec 9.5 — package and verify quantized artifacts")
    p.add_argument("--mode", default="all",
                   choices=["resnet", "bert", "all"])
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--build-missing", action="store_true",
                   help="Rebuild ONNX / IR artifacts that are absent. "
                        "TRT plans are never rebuilt here -- run section "
                        "9.3's pipeline for those.")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="Where section 9.4 stored ONNX and IR artifacts. "
                        "Defaults to ./ov_cache (matches section 9.4).")
    p.add_argument("--trt-cache", type=str, default=None,
                   help="Where section 9.3 stored TensorRT .plan files. "
                        "Defaults to ./trt_cache.")
    p.add_argument("--package-dir", type=str, default=None,
                   help="Output directory for manifests, Triton repo, "
                        "verify report. Defaults to ./packages.")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for figures. Defaults to "
                        "./figures (matches section 9.4).")
    p.add_argument("--resnet-batch", type=int, default=32)
    p.add_argument("--bert-batch", type=int, default=8)
    p.add_argument("--bert-seq-len", type=int, default=128)
    args = p.parse_args()

    cfg = Config(
        mode=args.mode, save_plots=args.save_plots,
        build_missing=args.build_missing,
        resnet_batch=args.resnet_batch,
        bert_batch=args.bert_batch, bert_seq_len=args.bert_seq_len,
    )
    if args.cache_dir:
        cfg.cache_dir = Path(args.cache_dir)
    if args.trt_cache:
        cfg.trt_cache = Path(args.trt_cache)
    if args.package_dir:
        cfg.package_dir = Path(args.package_dir)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    return cfg


def main():
    config = parse_args()

    print("=" * 72)
    print("Chapter 9 sec 9.5 — Package models for serving and distribution")
    print("=" * 72)
    print(f"  Mode:           {config.mode}")
    print(f"  Save plots:     {config.save_plots}")
    print(f"  Build missing:  {config.build_missing}")
    print()
    print_environment(config)

    if get_ov() is None:
        print("\n  ERROR: openvino is required (manifest probing of OV IR "
              "and the OV Core verifier column both depend on it).")
        sys.exit(1)

    resnet_cells: Dict[Tuple[str, str], CellResult] = {}
    bert_cells: Dict[Tuple[str, str], CellResult] = {}

    modes = ["resnet", "bert"] if config.mode == "all" else [config.mode]

    for m in modes:
        try:
            if m == "resnet":
                resnet_cells = run_resnet(config)
            elif m == "bert":
                bert_cells = run_bert(config)
        except Exception as e:
            print(f"\n  FAILED mode={m}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n  Continuing to next mode (if any)...")

    # Compatibility-matrix figure -- need cells from BOTH models for
    # the two-panel layout. If only one was run, use empty cells for
    # the missing one (panel still draws -- helps for incremental runs).
    if not resnet_cells:
        resnet_cells = {(a, r): CellResult(Status.BLOCKED, note="not run")
                        for a, _ in ARTIFACT_KINDS for r, _ in RUNTIMES}
    if not bert_cells:
        bert_cells = {(a, r): CellResult(Status.BLOCKED, note="not run")
                      for a, _ in ARTIFACT_KINDS for r, _ in RUNTIMES}

    plot_compatibility_matrix(resnet_cells, bert_cells, config)

    print("\n" + "=" * 72)
    print("Done.")
    print("  Artifacts + manifests + Triton repos: "
          f"{config.package_dir.relative_to(SCRIPT_DIR)}")
    print("  Figure (if --save-plots): "
          f"{config.output_dir.relative_to(SCRIPT_DIR)}/"
          "CH09_F10_Kalyanarangan_packaging_matrix.{png,pdf}")
    print("=" * 72)


if __name__ == "__main__":
    main()