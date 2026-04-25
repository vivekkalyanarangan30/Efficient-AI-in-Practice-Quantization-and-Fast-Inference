"""
Chapter 9, Section 9.4 — Use OpenVINO pipelines for edge and desktop CPUs
Companion script for "Efficient AI in Practice: Quantization and Fast Inference"

What this script demonstrates:
  Two routes deploy quantized models on Intel silicon, and section 9.1's
  decision tree treats them as a deliberate trade-off:

    (a) Standalone OpenVINO. ONNX -> IR (.xml + .bin) via ov.convert_model;
        INT8 via NNCF on the IR; served through ov.Core().compile_model.
        Peak Intel CPU latency, vendor-native artifact format.
    (b) Via ORT. The same FP32/INT8 ONNX served through ORT's
        OpenVINOExecutionProvider. Single serving surface across a fleet
        with mixed silicon, at some latency cost.

  The five-way benchmark (FP32 / BF16 / INT8 native + FP32 / INT8
  via-ORT) puts a number on the dispatch path on the reader's
  hardware, on the two reference models the chapter has been carrying:

    ResNet-18    (CNN, continuity with sections 9.2 and 9.3)
    BERT-base SST-2 (transformer, completes the OpenVINO column for
                     section 9.2's portability comparison)

  *** HARDWARE REQUIREMENT: this script targets a CPU with AVX512_VNNI
  AND AVX512_BF16 (or AMX equivalents) at minimum. On a CPU without
  these ISAs (e.g. consumer Skylake-X, original Xeon Scalable Gen 1
  and 2), the BF16 row silently falls back to FP32 and the INT8 row
  on transformers can be slower than FP32 because the Q/DQ overhead
  exceeds the AVX-512F INT8 throughput gain. The chapter prose
  discusses this directly; the script targets working silicon.

  Recommended cloud instances:
    GCP c3-* (Sapphire Rapids: AVX512_BF16 + AMX-BF16 + AMX-INT8)
    GCP c4-* (Emerald/Granite Rapids: same + AVX512_FP16)
    AWS c7i.* / c7i-flex.* (Sapphire Rapids: AMX-BF16 + AMX-INT8)
    AWS m7i.* (Sapphire Rapids: same)
    Any 4th gen Xeon Scalable or later

  The OPTIMIZATION_CAPABILITIES line printed at startup tells you
  which ISA tier is in front of the script. BF16 in that list = BF16
  row will execute BF16 kernels; INT8 + VNNI = INT8 row will execute
  via VPDPBUSD; AMX-INT8 = INT8 row routes through AMX tile registers
  (4-8x over AVX512_VNNI on matmul-heavy graphs).

  Section 9.4 closes by inspecting the OpenVINO runtime model
  (compiled_model.get_runtime_model()) and counting ops by execution
  precision — the OpenVINO equivalent of section 9.3's TRT engine
  inspector. The "INT8 fraction" column reveals whether the quantized
  IR actually executes INT8 kernels or silently falls back to FP32 on
  the CPU in front of it.

Modes:
  --mode resnet   Build + bench + inspect ResNet-18 on Intel CPU
  --mode bert     Build + bench + inspect BERT-base SST-2 on Intel CPU
  --mode all      Both (default)

Usage:
  python ch9_openvino_deployment.py --mode all --save-plots
  python ch9_openvino_deployment.py --mode resnet --save-plots
  python ch9_openvino_deployment.py --mode bert   --save-plots
  python ch9_openvino_deployment.py --mode all --force-rebuild

Install (one Python environment, CPU-only):
  pip install -U openvino nncf onnx \
                 onnxruntime-openvino \
                 torch torchvision \
                 transformers "optimum[onnxruntime]" datasets \
                 matplotlib pillow

  Note on ORT packaging: onnxruntime-openvino bundles both the OpenVINO
  EP and the CPU EP in one wheel. It conflicts with the plain
  onnxruntime wheel — pick one. This script targets onnxruntime-openvino
  so the ORT-OV-EP comparison columns can populate; if only plain
  onnxruntime is installed those rows are skipped with a notice.

Hardware target:
  Any modern Intel CPU. INT8 acceleration on x86 scales with ISA tier:

    AVX2 (no VNNI):    Haswell -- consumer Skylake.    ~1.5x INT8 / FP32
    AVX-512 VNNI:      Cascade Lake / Ice Lake / Tiger Lake / Rocket Lake.
                       Adds VPDPBUSD (one-cycle INT8 dot product).
                       ~2.5-3.5x INT8 / FP32
    AMX-INT8:          Sapphire Rapids / Emerald Rapids / Granite Rapids
                       (4th gen Xeon Scalable +). Tile-based INT8 matmul.
                       ~6-10x INT8 / FP32 on transformer matmuls

  The script reports OPTIMIZATION_CAPABILITIES from openvino.Core so the
  run log records the ISA tier the numbers were collected on. Numbers
  cross-check against this column, not against any single absolute target.

Note on quantization sensitivity:
  Section 9.3 established the per-model exclusion patterns: for ResNet-18,
  the input-adjacent stem Conv and the classifier fc; for BERT, the
  input-side embedding Gather and the final classifier MatMul. This
  section reuses the same exclusion lists in NNCF's IgnoredScope and in
  the ORT QDQ pass that feeds the via-ORT INT8 column.
"""

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("nncf").setLevel(logging.WARNING)

SCRIPT_DIR = Path(__file__).resolve().parent


# --- Configuration ---------------------------------------------------------

@dataclass
class Config:
    mode: str = "all"
    save_plots: bool = False
    force_rebuild: bool = False
    output_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "figures")
    cache_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "ov_cache")

    # ResNet-18 inference shape
    resnet_batch: int = 32
    resnet_image_size: int = 224

    # BERT-base inference shape (matches sections 9.2 and 9.3)
    bert_batch: int = 8
    bert_seq_len: int = 128

    # Calibration: 8 batches per model (matches section 9.3)
    num_calib_batches: int = 8

    # Benchmark settings (CPU iterations are more expensive than GPU
    # CUDA-event timings; lower iteration count keeps run time bounded
    # while still tightening the standard deviation under 1ms.)
    num_warmup: int = 10
    num_iters: int = 50

    seed: int = 42


# --- Manning figure style --------------------------------------------------

# Five execution variants. The native vs via-ORT split is encoded in
# the hatch density; the FP32 / BF16 / INT8 split is encoded in colour.
# This keeps the figure legible after black-and-white print conversion.
COLORS = {
    "ov_fp32":      "#7570b3",
    "ov_bf16":      "#e7298a",
    "ov_int8":      "#1b9e77",
    "ort_ov_fp32":  "#a8a4d4",   # lighter shade of ov_fp32
    "ort_ov_int8":  "#7fc7b1",   # lighter shade of ov_int8
}
HATCHES = {
    "ov_fp32":      "..",
    "ov_bf16":      "||",
    "ov_int8":      "xx",
    "ort_ov_fp32":  "//",
    "ort_ov_int8":  "\\\\",
}
DISPLAY_LABELS = {
    "ov_fp32":      "OV native\nFP32",
    "ov_bf16":      "OV native\nBF16",
    "ov_int8":      "OV native\nINT8 (NNCF)",
    "ort_ov_fp32":  "ORT + OV-EP\nFP32",
    "ort_ov_int8":  "ORT + OV-EP\nINT8 (QDQ)",
}
# Single-line variants for terminal table prints. The two-line forms
# above are for figure x-axis labels where the wrap helps; reusing
# them in printf-style column output corrupts column alignment.
PLAIN_LABELS = {
    "ov_fp32":      "OV native FP32",
    "ov_bf16":      "OV native BF16",
    "ov_int8":      "OV native INT8",
    "ort_ov_fp32":  "ORT+OV-EP FP32",
    "ort_ov_int8":  "ORT+OV-EP INT8",
}


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


def has_ort_ov_ep() -> bool:
    """Return True iff onnxruntime-openvino (or compatible) is
    installed AND the OpenVINOExecutionProvider is registered.

    Two common install pathologies that this guards against:

      1. Plain `onnxruntime` is installed (no OV EP). Import succeeds,
         get_available_providers() returns ['CPUExecutionProvider', ...],
         OV EP is not in the list.
      2. Mixed install: both `onnxruntime` and `onnxruntime-openvino`
         were installed at different times into the same site-packages.
         pip uninstalls leave fragments behind; the import resolves to
         a partial module where get_available_providers is missing
         entirely (AttributeError) or falsely returns ['CPUExecutionProvider'].
         Fix: `pip uninstall -y onnxruntime onnxruntime-openvino && \\
                pip cache purge && pip install onnxruntime-openvino`.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return False

    get_providers = getattr(ort, "get_available_providers", None)
    if not callable(get_providers):
        # Mixed/broken install. Surface a clear hint instead of crashing
        # downstream with AttributeError on every ORT call.
        print("  ⚠ Warning: `onnxruntime` is importable but "
              "get_available_providers is missing.")
        print("    This usually means a mixed install of "
              "`onnxruntime` and `onnxruntime-openvino`.")
        print("    Fix: `pip uninstall -y onnxruntime "
              "onnxruntime-openvino && pip cache purge && "
              "pip install onnxruntime-openvino`")
        return False

    try:
        providers = get_providers()
    except Exception as e:
        print(f"  ⚠ Warning: ort.get_available_providers() raised "
              f"{type(e).__name__}: {e}")
        return False

    return "OpenVINOExecutionProvider" in providers


def print_environment(config: Config):
    ov = get_ov()
    nncf = get_nncf()

    print(f"  Python:           {sys.version.split()[0]}")
    print(f"  openvino:         {_pkg_version('openvino')}")
    print(f"  nncf:             {_pkg_version('nncf')}")
    print(f"  onnxruntime-ov:   {_pkg_version('onnxruntime-openvino')}")
    print(f"  onnxruntime:      {_pkg_version('onnxruntime')}")
    print(f"  optimum:          {_pkg_version('optimum')}")
    print(f"  transformers:     {_pkg_version('transformers')}")
    print(f"  torch:            {_pkg_version('torch')}")

    if ov is not None:
        core = ov.Core()
        try:
            cpu_name = core.get_property("CPU", "FULL_DEVICE_NAME")
        except Exception:
            cpu_name = "?"
        try:
            opt_caps = core.get_property("CPU", "OPTIMIZATION_CAPABILITIES")
        except Exception:
            opt_caps = []
        try:
            num_threads = core.get_property("CPU", "INFERENCE_NUM_THREADS")
        except Exception:
            num_threads = "?"

        print(f"  CPU:              {cpu_name}")
        print(f"  Capabilities:     {', '.join(opt_caps) if opt_caps else 'unknown'}")
        print(f"  Inference threads:{num_threads}")
        print(f"  Available devices:{', '.join(core.available_devices)}")

    print(f"  ORT-OV-EP avail:  {has_ort_ov_ep()}")
    print(f"  Cache dir:        {config.cache_dir}")


# --- ORT calibration data reader (model-agnostic) --------------------------
# Mirrored from section 9.3. Used here only for the ORT QDQ ONNX pass that
# feeds the ORT + OV-EP INT8 column. NNCF on the IR side has its own
# nncf.Dataset adapter (defined later).

class _CalibReader:
    """ORT-compatible calibration reader. Owns a list of feed dicts."""

    def __init__(self, feeds: List[Dict[str, np.ndarray]]):
        self.feeds = feeds
        self._i = 0

    def get_next(self):
        if self._i >= len(self.feeds):
            return None
        f = self.feeds[self._i]
        self._i += 1
        return f

    def rewind(self):
        self._i = 0


# --- ONNX -> IR conversion -------------------------------------------------

def convert_onnx_to_ir(onnx_path: Path, ir_xml_path: Path, force: bool):
    """Run ov.convert_model() on an ONNX file and serialize the resulting
    ov.Model to IR (.xml + .bin)."""
    ov = get_ov()
    if ir_xml_path.exists() and not force:
        size_xml = ir_xml_path.stat().st_size / 1e3
        bin_path = ir_xml_path.with_suffix(".bin")
        size_bin = bin_path.stat().st_size / 1e6 if bin_path.exists() else 0.0
        print(f"  Reusing IR: {ir_xml_path.name} "
              f"({size_xml:.1f} KB xml + {size_bin:.1f} MB bin)")
        return

    print(f"  Converting ONNX -> IR: {ir_xml_path.name}")
    t0 = time.perf_counter()
    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, str(ir_xml_path), compress_to_fp16=False)         #A
    elapsed = time.perf_counter() - t0
    bin_path = ir_xml_path.with_suffix(".bin")
    size_bin = bin_path.stat().st_size / 1e6 if bin_path.exists() else 0.0
    print(f"    Built IR in {elapsed:.1f} s  |  {size_bin:.1f} MB on disk")


#A `compress_to_fp16=False` keeps weights in FP32 in the .bin. The default
#  is True, which silently halves weight precision before NNCF ever sees
#  the model — you would then "INT8 quantize an FP16 model" and lose the
#  small remaining FP32-vs-FP16 accuracy headroom. For an apples-to-apples
#  FP32 baseline, force this flag off.


# --- NNCF INT8 quantization ------------------------------------------------

def _make_nncf_dataset(calib_feeds: List[Dict[str, np.ndarray]]):
    """Wrap a list of model-input dicts in an nncf.Dataset.

    NNCF iterates the dataset and feeds each item through transform_fn to
    produce a model input. For ResNet the input tuple has length 1; for
    BERT it has length 3 (input_ids, attention_mask, token_type_ids).
    The transform here passes the dict through unchanged — NNCF accepts
    both positional tuples and named-input dicts."""
    nncf = get_nncf()

    def transform_fn(item):
        return item

    return nncf.Dataset(calib_feeds, transform_fn)


def quantize_with_nncf(
    fp32_ir_xml: Path,
    int8_ir_xml: Path,
    calib_feeds: List[Dict[str, np.ndarray]],
    model_type: str,                                                       #B
    ignored_op_names: List[str],
    force: bool,
):
    """Apply NNCF post-training INT8 quantization to an OpenVINO IR.

    NNCF's .quantize() API differs from ORT's quantize_static: it operates
    on an ov.Model in memory rather than on disk paths, and it returns a
    new ov.Model with FakeQuantize ops inserted. The compile step (later)
    is what folds those FakeQuantize ops into INT8 kernels on the target
    device. NNCF does not produce 'INT8 ops' in the IR per se -- that
    transformation is deferred to the OpenVINO runtime's plugin compiler.
    """
    ov = get_ov()
    nncf = get_nncf()

    if int8_ir_xml.exists() and not force:
        bin_path = int8_ir_xml.with_suffix(".bin")
        size_bin = bin_path.stat().st_size / 1e6 if bin_path.exists() else 0.0
        print(f"  Reusing INT8 IR: {int8_ir_xml.name} ({size_bin:.1f} MB)")
        return

    print(f"  Quantizing IR with NNCF: {int8_ir_xml.name}")
    t0 = time.perf_counter()

    core = ov.Core()
    fp32_model = core.read_model(str(fp32_ir_xml))

    dataset = _make_nncf_dataset(calib_feeds)

    nncf_kwargs = dict(
        calibration_dataset=dataset,
        preset=nncf.QuantizationPreset.MIXED,                              #C
        target_device=nncf.TargetDevice.CPU,
        subset_size=len(calib_feeds),
        fast_bias_correction=True,
    )
    if model_type == "transformer":
        nncf_kwargs["model_type"] = nncf.ModelType.TRANSFORMER             #D

    if ignored_op_names:
        nncf_kwargs["ignored_scope"] = nncf.IgnoredScope(
            names=ignored_op_names, validate=False,
        )
        print(f"    Ignored from quantization: {ignored_op_names}")

    quantized = nncf.quantize(fp32_model, **nncf_kwargs)

    int8_ir_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(quantized, str(int8_ir_xml), compress_to_fp16=False)

    elapsed = time.perf_counter() - t0
    bin_path = int8_ir_xml.with_suffix(".bin")
    size_bin = bin_path.stat().st_size / 1e6 if bin_path.exists() else 0.0
    print(f"    Quantized in {elapsed:.1f} s  |  {size_bin:.1f} MB on disk")


#B `model_type` is NNCF's coarse hint, separate from `target_device`.
#  Setting it to TRANSFORMER enables SmoothQuant-style activation-weight
#  rebalancing on attention matmuls and routes around the
#  GatherND-shaped subgraphs that show up in BERT-style attention. The
#  default (None) covers most CNNs; setting it to TRANSFORMER for
#  transformers gives noticeably better INT8 accuracy.

#C `MIXED` preset = asymmetric activations + symmetric weights. The
#  PERFORMANCE preset uses symmetric on both sides, which can be faster
#  on some kernels but is more aggressive on accuracy. MIXED is the
#  recommended default for general PTQ on CPU.

#D `IgnoredScope(validate=False)`: NNCF validates that named ops exist
#  in the model and errors out otherwise. With per-model name-discovery
#  walks (next sections), our names are derived from the actual graph,
#  so validation is redundant; turning it off keeps NNCF quiet on
#  graphs where graph-import has renamed a node ("/Gather_2" vs
#  "/embeddings/Gather"-style differences across opset versions).


# --- Native OpenVINO benchmark + runtime inspection ------------------------

def _ov_compile(ir_xml: Path, device: str = "CPU",
                inference_precision: Optional[str] = None):
    ov = get_ov()
    core = ov.Core()
    model = core.read_model(str(ir_xml))
    config = {"PERFORMANCE_HINT": "LATENCY"}                               #E
    if inference_precision is not None:
        config["INFERENCE_PRECISION_HINT"] = inference_precision           #Ep
    return core.compile_model(model, device, config=config)


#E PERFORMANCE_HINT=LATENCY tells the OpenVINO plugin to optimize for
#  single-batch latency: fewer parallel inference streams, smaller
#  thread pools per stream, hint-driven kernel selection. The
#  alternative (THROUGHPUT) optimizes for many concurrent batches and
#  reports very different numbers — for the per-batch latency the
#  chapter is measuring, LATENCY is the right hint.

#Ep INFERENCE_PRECISION_HINT picks the dtype the CPU plugin executes the
#  network in, independent of the IR's native dtype. Three values matter
#  on Intel CPU:
#    "f32" — full FP32 compute. Default on CPUs without 16-bit ISA
#            (consumer Skylake-X, Haswell, etc).
#    "bf16" — BF16 compute via AVX512_BF16 (Cooper Lake / Ice Lake-SP)
#            or AMX-BF16 (Sapphire Rapids+). On CPUs without either,
#            this hint is silently honored by falling back to FP32 — no
#            error, no warning. The runtime inspector reveals the
#            fallback by reporting the executed precision per op.
#    "f16" — FP16 compute via AVX512-FP16 (Sapphire Rapids+ only). BF16
#            has wider hardware coverage on server CPUs, so this script
#            uses BF16 for the 16-bit-float row.
#  For INT8 IRs (NNCF-quantized), passing None lets the plugin pick the
#  surrounding-op precision while honouring the FakeQuantize ops in the
#  graph. This is the recommended default.


def _np_to_ov_inputs(feed: Dict[str, np.ndarray], compiled) -> Dict:
    """Map a name->ndarray dict to OpenVINO's expected input keying.
    OV accepts inputs by name or by Tensor; passing the dict by name is
    the most resilient form across ONNX export quirks (renamed inputs,
    opset shifts)."""
    return {k: v for k, v in feed.items()}


def bench_ov_native(
    ir_xml: Path,
    sample_feed: Dict[str, np.ndarray],
    num_warmup: int, num_iters: int,
    inference_precision: Optional[str] = None,
) -> Tuple[float, float, "compiled"]:
    """Wall-clock latency for native OV inference. Returns
    (mean_ms, std_ms, compiled_model) so the caller can hand the
    compiled model to the runtime inspector without re-compiling."""
    compiled = _ov_compile(ir_xml, device="CPU",
                           inference_precision=inference_precision)
    request = compiled.create_infer_request()

    inputs = _np_to_ov_inputs(sample_feed, compiled)

    for _ in range(num_warmup):
        request.infer(inputs)

    timings = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        request.infer(inputs)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(timings)), float(np.std(timings)), compiled


def _extract_ov_any_value(value) -> str:
    """Pull a string out of an OVAny wrapper across OV minor versions.

    On OV 2026.x, op.get_rt_info()[key] returns an OVAny whose __str__
    returns '<OVAny class>' instead of the wrapped value. The actual
    payload lives behind .get(), .value, or .astype(str) depending on
    the build. Fall through these in order so the inspector survives
    OV 2024.x, 2025.x, and 2026.x without per-version branching.

    Returns "" when nothing usable can be extracted -- callers should
    treat that as missing rather than as a precision string."""
    # Common case: native Python str/bytes already. str() returns the
    # value, no '<OVAny class>' marker.
    s = str(value).strip()
    if s and "OVAny" not in s:
        return s

    # OV 2026.x: OVAny.get() unwraps to the underlying Python value.
    for accessor in ("get", "value"):
        try:
            attr = getattr(value, accessor, None)
            if attr is None:
                continue
            result = attr() if callable(attr) else attr
            s = str(result).strip()
            if s and "OVAny" not in s:
                return s
        except Exception:
            continue

    # Last resort: astype(str) if the wrapper exposes it.
    try:
        astype = getattr(value, "astype", None)
        if callable(astype):
            s = str(astype(str)).strip()
            if s and "OVAny" not in s:
                return s
    except Exception:
        pass

    return ""


def inspect_runtime_model(compiled, dump_first_op: bool = False) -> Dict:
    """Walk the compiled runtime graph and bucket ops by execution
    precision. The runtime model exposes the post-fusion graph that
    the plugin actually executes — FakeQuantize ops have been
    replaced by the kernel selections the plugin made, and the
    rt_info field reports the executed precision per op.

    This is the OpenVINO equivalent of section 9.3's TRT engine
    inspector. Three implementation gotchas to know about:

      1. Op type. In the runtime graph, every op's get_type_name()
         returns 'ExecutionNode' regardless of what it actually is.
         The original op type lives in rt_info['layerType'] -- we
         have to read that key, not call get_type_name().
      2. Value extraction. rt_info values are OVAny wrappers; str()
         on them returns '<OVAny class>' on OV 2026.x. Use
         _extract_ov_any_value() to pull the actual string.
      3. Bucket matching. Naive substring matching ('f16' in prec)
         puts bf16 ops in the f16 bucket because 'f16' is a suffix
         of 'bf16'. Exact-match first, longest-substring fallback.

    Set dump_first_op=True to print the rt_info contents of the
    first compute op -- useful when an OV upgrade renames keys."""
    BOOKKEEPING = {
        "Constant", "Parameter", "Result",
        "ShapeOf", "Reshape", "Transpose",
        "Squeeze", "Unsqueeze", "Concat", "Gather",
        "Convert", "Broadcast",
        # Names that appear specifically in the runtime graph after
        # fusion. The CPU plugin fuses memory-format conversions
        # under these labels; counting them as compute ops would
        # inflate the totals without measuring actual matmul/conv
        # arithmetic.
        "Reorder", "MemoryReorder",
    }

    # Canonical precision strings -> bucket key. OV 2026 emits "f32",
    # "bf16", "i8", "u8" in lowercase via runtimePrecision; older OV
    # may emit "FP32", "FP16" (uppercase). Both forms map here.
    PREC_MAP = {
        "f32": "f32", "fp32": "f32", "float": "f32", "float32": "f32",
        "f16": "f16", "fp16": "f16", "float16": "f16",
        "bf16": "bf16", "bfloat16": "bf16",
        "i8": "i8",  "int8": "i8",
        "u8": "u8",  "uint8": "u8",
        "i32": "i32", "int32": "i32",
        "i64": "i64", "int64": "i64",
    }
    # Longest-first substring fallback. bf16 before f16, i64 before i8.
    SUBSTR_KEYS = sorted(PREC_MAP.keys(), key=len, reverse=True)

    rt_model = compiled.get_runtime_model()
    counts = {"f32": 0, "f16": 0, "bf16": 0, "i8": 0, "u8": 0,
              "i32": 0, "i64": 0, "other": 0}
    detail = []
    total_ops = 0
    dumped = False

    for op in rt_model.get_ordered_ops():
        total_ops += 1
        op_name = op.get_friendly_name()[:60]

        # Pull all rt_info fields into a plain dict, unwrapping OVAny
        # wrappers as we go. RTMap doesn't behave like a Python dict
        # under `in` on OV 2026.x; iteration via items() does work.
        rt_dict = {}
        try:
            for key, value in op.get_rt_info().items():
                rt_dict[str(key)] = _extract_ov_any_value(value)
        except Exception:
            pass

        # In the runtime graph, get_type_name() == "ExecutionNode" for
        # every op -- the actual original op type is in layerType.
        original_op_type = rt_dict.get("layerType", "")
        if not original_op_type:
            original_op_type = op.get_type_name()    # fallback

        if dump_first_op and not dumped and \
                original_op_type not in BOOKKEEPING:
            print(f"    [debug] rt_info contents for first compute op "
                  f"({op_name}, layerType={original_op_type}):")
            for k, v in sorted(rt_dict.items()):
                print(f"      {k} = {v[:60]}")
            dumped = True

        # Pull the executed precision. Try the well-known key first,
        # then a few spellings used in older OV builds and downstream
        # forks.
        prec = ""
        for k in ("runtimePrecision", "runtime_precision",
                  "execPrecision", "exec_precision",
                  "outputPrecisions", "primitivePrecision"):
            v = rt_dict.get(k, "")
            if v:
                prec = v
                break

        if original_op_type in BOOKKEEPING:
            detail.append((op_name, original_op_type, prec, "bookkeeping"))
            continue

        # Match prec to a bucket. Exact lowercase first; substring fallback.
        bucket = "other"
        prec_lower = prec.lower()
        if prec_lower in PREC_MAP:
            bucket = PREC_MAP[prec_lower]
        else:
            for k in SUBSTR_KEYS:
                if k in prec_lower:
                    bucket = PREC_MAP[k]
                    break
        counts[bucket] += 1
        detail.append((op_name, original_op_type, prec, "compute"))

    compute_total = sum(v for k, v in counts.items())
    return {
        "total_ops": total_ops,
        "compute_total": compute_total,
        "counts": counts,
        "detail": detail,
    }


def print_inspect_summary(name: str, summary: Dict):
    counts = summary["counts"]
    compute_total = max(1, summary["compute_total"])
    int8_compute = counts["i8"] + counts["u8"]
    print(f"\n  Runtime model: {name}")
    print(f"    Total ops in runtime graph: {summary['total_ops']}")
    print(f"    Compute ops by execution precision:")
    for k in ("i8", "u8", "f32", "f16", "bf16", "i32", "i64", "other"):
        v = counts[k]
        if v == 0:
            continue
        print(f"      {k:<6}: {v:>4}  ({v / compute_total * 100:5.1f}%)")
    print(f"    INT8 compute fraction: "
          f"{int8_compute / compute_total * 100:5.1f}% "
          f"({int8_compute}/{compute_total})")


# --- ORT + OpenVINO EP benchmark ------------------------------------------

def bench_ort_ov_ep(
    onnx_path: Path,
    sample_feed: Dict[str, np.ndarray],
    num_warmup: int, num_iters: int,
    cache_dir: Path,
) -> Optional[Tuple[float, float]]:
    """Wall-clock latency for ORT + OV-EP. Returns None if the OV EP is
    not registered on this runtime (caller should skip the row)."""
    if not has_ort_ov_ep():
        return None

    import onnxruntime as ort

    cache_dir.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    providers = [
        ("OpenVINOExecutionProvider", {                                    #F
            "device_type": "CPU",
            "precision": "ACCURACY",
            "cache_dir": str(cache_dir),
        }),
        "CPUExecutionProvider",
    ]
    # Wrap the session creation and the inference loop together: a bad
    # ONNX -> OV-EP compile typically fails inside InferenceSession(),
    # but very rare graph-pattern issues can fail later inside run().
    # Either way, we want this row to skip cleanly so the rest of the
    # benchmark proceeds.
    try:
        sess = ort.InferenceSession(str(onnx_path), so, providers=providers)
        out_name = sess.get_outputs()[0].name
        feed = {k: v for k, v in sample_feed.items()}

        for _ in range(num_warmup):
            sess.run([out_name], feed)

        timings = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            sess.run([out_name], feed)
            timings.append((time.perf_counter() - t0) * 1000.0)
        return float(np.mean(timings)), float(np.std(timings))
    except Exception as e:
        print(f"    ORT + OV-EP inference failed: "
              f"{type(e).__name__}: {str(e)[:200]}")
        return None


#F precision="ACCURACY" tells OV EP to honour the dtypes baked into the
#  ONNX graph rather than auto-promoting INT8 to FP32 or auto-demoting
#  FP32 to FP16 to chase peak throughput. This matches the "production
#  artifact dictates execution dtype" mental model section 9.1
#  established. cache_dir lets OV EP reuse compiled blobs across runs:
#  the first call pays the compile cost, subsequent calls don't.


# --- Plotting helper -------------------------------------------------------

def plot_engine_latency(
    results: List[Tuple[str, float, float, float, float]],
    title: str, fig_name: str, config: Config,
    ref_lines: Optional[List[Tuple[str, float, str]]] = None,
):
    """Two-panel bar chart: latency + speedup vs the FP32 native
    baseline.

    Layout notes worth flagging:
      - Figure width is 7.6" so the five rotated labels fit without
        overlap. Manning's max printable column width is ~10" so this
        leaves margin.
      - Labels are rotated 30 degrees with right-anchor alignment.
        Larger angles (45, 90) waste vertical space; smaller angles
        (10, 15) don't fully separate the labels at this width.
      - Use single-line PLAIN_LABELS, not the wrapped DISPLAY_LABELS:
        the rotation already separates them, and a wrapped two-line
        rotated label looks awful.
      - subplots_adjust(bottom=0.22) reserves space for the rotated
        labels; tight_layout() alone underestimates the descent of
        rotated text.
    """
    apply_manning_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.6))

    labels = [r[0] for r in results]
    latencies = [r[1] for r in results]
    stds = [r[2] for r in results]
    fp32_ms = next((r[1] for r in results if r[0] == "ov_fp32"),
                   latencies[0])
    speedups = [fp32_ms / r[1] for r in results]
    tick_labels = [PLAIN_LABELS[l] for l in labels]

    # --- Panel 1: latency ---
    for i, (lbl, lat, sd) in enumerate(zip(labels, latencies, stds)):
        ax1.bar(i, lat, yerr=sd,
                color=COLORS[lbl], hatch=HATCHES[lbl],
                edgecolor="black", linewidth=0.5, width=0.65,
                error_kw={"linewidth": 0.8, "capsize": 3})
        ax1.text(i, lat + sd + max(latencies) * 0.02, f"{lat:.2f}",
                 ha="center", va="bottom", fontsize=7)

    if ref_lines:
        for ref_label, ref_ms, ref_color in ref_lines:
            ax1.axhline(ref_ms, color=ref_color, linestyle=":",
                        linewidth=0.9, alpha=0.7)
            ax1.text(len(labels) - 0.4, ref_ms,
                     f"{ref_label}: {ref_ms:.1f}ms",
                     fontsize=6, color=ref_color, va="bottom",
                     ha="right")

    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(tick_labels, fontsize=7,
                        rotation=30, ha="right",
                        rotation_mode="anchor")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title(title)
    # Add a top margin so the value labels above tall bars don't get
    # clipped against the title.
    ax1.set_ylim(top=max(latencies) * 1.12)

    # --- Panel 2: speedup ---
    for i, (lbl, sp) in enumerate(zip(labels, speedups)):
        ax2.bar(i, sp,
                color=COLORS[lbl], hatch=HATCHES[lbl],
                edgecolor="black", linewidth=0.5, width=0.65)
        ax2.text(i, sp + max(speedups) * 0.02, f"{sp:.2f}x",
                 ha="center", va="bottom", fontsize=7)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(tick_labels, fontsize=7,
                        rotation=30, ha="right",
                        rotation_mode="anchor")
    ax2.set_ylabel("Speedup vs OV native FP32")
    ax2.set_title("Speedup")
    ax2.set_ylim(top=max(speedups) * 1.12)

    # tight_layout undersizes the bottom margin for rotated labels;
    # follow with subplots_adjust to reserve the actual descent.
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    save_or_show(fig, fig_name, config)


# --- Generic INT32-bias DQ stripper (mirrored from section 9.3) -----------
# ORT emits INT32-typed DequantizeLinear nodes around biases. OV's ONNX
# importer is more forgiving than TRT's parser here, but stripping these
# keeps the QDQ ONNX clean and identical to the §9.3 artifact, so the
# ORT + OV-EP column compares like-for-like with the §9.3 TRT column on
# the same input file.

def _strip_int32_dq(onnx_path: Path) -> int:
    import onnx
    from onnx import numpy_helper, TensorProto

    model = onnx.load(str(onnx_path))
    graph = model.graph
    inits = {init.name: init for init in graph.initializer}

    to_remove, new_inits = [], []
    for node in graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        if node.input[0] not in inits:
            continue
        data_init = inits[node.input[0]]
        if data_init.data_type != TensorProto.INT32:
            continue

        int32_data = numpy_helper.to_array(data_init).astype(np.int64)
        scale = numpy_helper.to_array(inits[node.input[1]]).astype(np.float32)
        if len(node.input) > 2 and node.input[2] in inits:
            zp = numpy_helper.to_array(inits[node.input[2]]).astype(np.int64)
        else:
            zp = np.zeros_like(scale, dtype=np.int64)

        fp32_data = ((int32_data - zp).astype(np.float32) * scale)
        new_inits.append(numpy_helper.from_array(
            fp32_data.astype(np.float32), name=node.output[0]))
        to_remove.append(node)

    for n in to_remove:
        graph.node.remove(n)
    graph.initializer.extend(new_inits)
    onnx.save(model, str(onnx_path))
    return len(to_remove)


# --- ResNet-18 pipeline ----------------------------------------------------

def _resnet_fp32_onnx(out_path: Path, batch: int, image_size: int):
    import torch
    import torchvision.models as tvm

    print(f"  Exporting ResNet-18 FP32 ONNX: {out_path.name}")
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1).eval()
    dummy = torch.randn(batch, 3, image_size, image_size)
    torch.onnx.export(                                                     #D
        model, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        opset_version=18, dynamic_axes=None, do_constant_folding=True,
        dynamo=False,
    )
    print(f"    Wrote: {out_path.name} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")


#D `dynamo=False` forces the legacy TorchScript-based exporter. The
#  default in torch 2.6+ is the TorchDynamo path, which (a) writes
#  weights as external data in sibling files (a ResNet-18 ONNX comes out
#  at 0.1 MB instead of 45 MB), (b) renames ops to opaque tokens like
#  `node_Conv_291` instead of `/conv1/Conv`, and (c) tries to back-
#  convert from opset 18 to 17 via the C API and fails noisily. opset 18
#  is the floor torch.onnx ships implementations for in 2.10, so we ask
#  for 18 directly and skip the back-conversion attempt.


def _resnet_calib_feeds(num_batches: int, batch: int, image_size: int,
                        seed: int) -> List[Dict]:
    """ImageNet-normalized random tensors. Section 9.3 used these same
    feeds; reproducing them keeps the INT8 IR's NNCF calibration aligned
    with the QDQ ONNX's ORT calibration in section 9.3."""
    rng = np.random.default_rng(seed)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    feeds = []
    for _ in range(num_batches):
        raw = rng.random((batch, 3, image_size, image_size), dtype=np.float32)
        feeds.append({"input": ((raw - mean) / std).astype(np.float32)})
    return feeds


def _find_resnet_ignored_ops(fp32_ir_xml: Path) -> List[str]:
    """Return OV-side op names to exclude from NNCF quantization.

    Walks the FP32 IR, picks Convolution ops consuming the model input
    (the stem) and the final MatMul/FullyConnected feeding the output
    (the classifier). Section 9.3 established these as the two
    sensitivity layers for ResNet-18; this is the OV-namespace
    equivalent of TRT's nodes_to_exclude list."""
    ov = get_ov()
    core = ov.Core()
    model = core.read_model(str(fp32_ir_xml))

    ignored = []
    inputs = {p.get_friendly_name() for p in model.get_parameters()}
    for op in model.get_ordered_ops():
        op_type = op.get_type_name()
        op_name = op.get_friendly_name()
        # Stem Conv = Convolution whose input ultimately traces back to a
        # graph Parameter without crossing another Conv. The simple rule:
        # any Convolution at depth 1 from a Parameter consumer.
        if op_type == "Convolution":
            for inp in op.inputs():
                src = inp.get_source_output().get_node()
                if src.get_type_name() == "Parameter" or \
                   src.get_friendly_name() in inputs:
                    ignored.append(op_name)
                    break

    # Classifier MatMul: walk back from each Result through up to 4 ops.
    # In OV's runtime graph the FC layer surfaces as MatMul or FullyConnected.
    for result in model.get_results():
        cur = result
        for _ in range(4):
            for inp in cur.inputs():
                src = inp.get_source_output().get_node()
                if src.get_type_name() in ("MatMul", "FullyConnected"):
                    ignored.append(src.get_friendly_name())
                    cur = None
                    break
                cur = src
                break
            if cur is None:
                break

    return list(dict.fromkeys(ignored))    # dedup, preserve order


def _resnet_qdq_onnx(fp32: Path, qdq: Path,
                     calib_feeds: List[Dict[str, np.ndarray]]):
    """ORT static QDQ ONNX with the same four bridge settings section 9.3
    used. We reuse this artifact for the ORT + OV-EP INT8 column."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    print(f"  Producing ResNet QDQ INT8 ONNX")
    prep = qdq.with_name(qdq.stem + "_prep.onnx")
    src = fp32
    try:
        quant_pre_process(str(fp32), str(prep), auto_merge=True)
        src = prep
    except Exception as e:
        print(f"    quant_pre_process failed ({type(e).__name__}); "
              f"using raw FP32")

    # Mirror section 9.3's stem-Conv exclusion — ORT-side name discovery.
    import onnx
    m = onnx.load(str(fp32))
    inputs = {i.name for i in m.graph.input}
    stem = [n.name for n in m.graph.node
            if n.op_type == "Conv"
            and any(x in inputs for x in n.input)]
    if stem:
        print(f"    Excluding stem Conv from QDQ ONNX: {stem}")

    quantize_static(
        str(src), str(qdq),
        calibration_data_reader=_CalibReader(calib_feeds),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv"],
        nodes_to_exclude=stem,
        extra_options={"ActivationSymmetric": True},
    )
    if prep.exists():
        prep.unlink()

    stripped = _strip_int32_dq(qdq)
    if stripped:
        print(f"    Post-stripped {stripped} INT32-bias DQ node(s)")
    print(f"    Wrote: {qdq.name} ({qdq.stat().st_size / 1e6:.1f} MB)")


# --- BERT pipeline ---------------------------------------------------------

BERT_MODEL_ID = "textattack/bert-base-uncased-SST-2"


def _bert_fp32_onnx(out_dir: Path, batch: int, seq_len: int):
    """Same as section 9.3: Optimum's task-aware export."""
    from optimum.onnxruntime import ORTModelForSequenceClassification

    print(f"  Exporting BERT-base SST-2 FP32 ONNX: {out_dir.name}")
    model = ORTModelForSequenceClassification.from_pretrained(
        BERT_MODEL_ID, export=True)
    model.save_pretrained(str(out_dir))
    onnx_path = out_dir / "model.onnx"
    print(f"    Wrote: {onnx_path} "
          f"({onnx_path.stat().st_size / 1e6:.1f} MB)")
    return onnx_path


def _bert_calib_feeds(num_batches: int, batch: int, seq_len: int,
                      seed: int) -> List[Dict]:
    """SST-2-style calibration feeds with random in-vocab token IDs.
    Mirrors section 9.3's pattern; see that section for the rationale."""
    rng = np.random.default_rng(seed)
    feeds = []
    vocab_size = 30522
    for _ in range(num_batches):
        ids = rng.integers(100, vocab_size,
                           size=(batch, seq_len), dtype=np.int64)
        ids[:, 0] = 101
        ids[:, -1] = 102
        feeds.append({
            "input_ids": ids,
            "attention_mask": np.ones((batch, seq_len), dtype=np.int64),
            "token_type_ids": np.zeros((batch, seq_len), dtype=np.int64),
        })
    return feeds


def _find_bert_ignored_ops(fp32_ir_xml: Path) -> List[str]:
    """Return OV-side op names to exclude from NNCF quantization on BERT.

    Two exclusions, both established in section 9.2 and reused in 9.3:
      1. The embedding Gather. Quantizing this collapses distinct token
         representations into shared INT8 scales — a well-known PTQ
         failure mode on transformers. NVIDIA's docs flag the same issue.
      2. The classifier MatMul (Linear(768, 2) feeding the output).
         Section 9.3's chapter-4 sensitivity rule.
    """
    ov = get_ov()
    core = ov.Core()
    model = core.read_model(str(fp32_ir_xml))

    ignored = []
    # Embedding Gather: a Gather op whose data input is a Constant of
    # shape [vocab_size, hidden]. For BERT-base that's [30522, 768].
    for op in model.get_ordered_ops():
        if op.get_type_name() != "Gather":
            continue
        data_inp = op.input(0).get_source_output().get_node()
        if data_inp.get_type_name() != "Constant":
            continue
        shape = data_inp.get_output_shape(0)
        if len(shape) == 2 and shape[0] >= 1000 and shape[1] >= 64:        #G
            ignored.append(op.get_friendly_name())

    # Classifier MatMul: reverse-walk from the Result up to 4 ops.
    for result in model.get_results():
        cur = result
        for _ in range(4):
            for inp in cur.inputs():
                src = inp.get_source_output().get_node()
                if src.get_type_name() in ("MatMul", "FullyConnected"):
                    ignored.append(src.get_friendly_name())
                    cur = None
                    break
                cur = src
                break
            if cur is None:
                break

    return list(dict.fromkeys(ignored))


#G Heuristic for "this Gather is the embedding lookup, not a routing
#  gather inside attention": data tensor is a 2D Constant with first
#  dim large (vocab-like) and second dim a typical hidden size. For
#  BERT-base it's [30522, 768]; the bound (1000, 64) is loose enough to
#  catch BERT-tiny and DistilBERT variants without triggering on the
#  small Gather constants attention masks produce.


def _bert_qdq_onnx(fp32_onnx: Path, qdq_onnx: Path,
                   calib_feeds: List[Dict[str, np.ndarray]]):
    """ORT static QDQ ONNX for BERT. Mirrors section 9.3."""
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat, CalibrationMethod,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    print(f"  Producing BERT QDQ INT8 ONNX")
    prep = qdq_onnx.with_name(qdq_onnx.stem + "_prep.onnx")
    src = fp32_onnx
    try:
        quant_pre_process(str(fp32_onnx), str(prep), auto_merge=True)
        src = prep
    except Exception as e:
        print(f"    quant_pre_process failed ({type(e).__name__}); "
              f"using raw FP32")

    # Classifier MatMul discovery (ORT-namespace, mirrored from §9.3)
    import onnx
    m = onnx.load(str(fp32_onnx))
    outputs = {o.name for o in m.graph.output}
    output_to_producer = {}
    for n in m.graph.node:
        for out in n.output:
            output_to_producer[out] = n
    classifier = []
    for out_name in outputs:
        node = output_to_producer.get(out_name)
        cur = node
        for _ in range(4):
            if cur is None:
                break
            if cur.op_type == "MatMul":
                classifier.append(cur.name)
                break
            if cur.input:
                cur = output_to_producer.get(cur.input[0])
            else:
                break
    classifier = list(set(classifier))
    if classifier:
        print(f"    Excluding classifier MatMul(s): {classifier}")

    quantize_static(
        str(src), str(qdq_onnx),
        calibration_data_reader=_CalibReader(calib_feeds),
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=classifier,
        extra_options={"ActivationSymmetric": True},
    )
    if prep.exists():
        prep.unlink()

    stripped = _strip_int32_dq(qdq_onnx)
    if stripped:
        print(f"    Post-stripped {stripped} INT32-bias DQ node(s)")
    print(f"    Wrote: {qdq_onnx.name} "
          f"({qdq_onnx.stat().st_size / 1e6:.1f} MB)")


# --- Mode runner (shared) --------------------------------------------------

def _run_model_pipeline(
    model_name: str,
    fp32_onnx: Path,
    qdq_onnx: Path,
    fp32_ir: Path,
    int8_ir: Path,
    sample_feed: Dict[str, np.ndarray],
    config: Config,
    figure_name: str,
    figure_title: str,
    ref_lines: Optional[List[Tuple[str, float, str]]] = None,
):
    """Benchmark all five variants and inspect the three native-OV
    runtime graphs. Variants that are unavailable on this host
    (e.g. ORT-OV-EP when only plain onnxruntime is installed) get
    skipped with a notice rather than failing the whole run.

    Hardware caveat for the BF16 row: this script assumes a CPU with
    AVX512_BF16 or AMX-BF16 (4th-gen Xeon Scalable / Sapphire Rapids
    or later -- e.g. GCP c3-*, AWS c7i.*). On a CPU without either
    ISA, OV honours INFERENCE_PRECISION_HINT='bf16' by silently
    executing in FP32, and the BF16 row lands on top of the FP32
    bar. The startup OPTIMIZATION_CAPABILITIES line tells you which
    tier you're on; the runtime inspector confirms by reporting
    runtimePrecision per op."""
    print(f"\n  --- Benchmarking {model_name} variants ---")

    # Surface the ISA tier in the run log so the figure is
    # self-contextualizing. The compile call never errors on a
    # precision hint -- it just quietly falls back if the ISA is
    # missing -- so this check is informational.
    ov = get_ov()
    caps = ov.Core().get_property("CPU", "OPTIMIZATION_CAPABILITIES")
    bf16_supported = "BF16" in caps
    if not bf16_supported:
        print(f"  ⚠ Warning: CPU does not advertise BF16 in "
              f"OPTIMIZATION_CAPABILITIES ({caps}).")
        print(f"  The BF16 row will fall back to FP32. The runtime "
              f"inspector will reveal it (look for runtimePrecision="
              f"f32 on the BF16 compile).")

    results = []

    # 1. OV native FP32 -- explicit precision hint anchors the FP32
    # reference unambiguously across hosts. On a Sapphire Rapids host
    # without an explicit hint, OV's default may pick BF16 and your
    # "FP32 baseline" would secretly be BF16.
    print(f"\n  [1/5] OV native FP32: {fp32_ir.name}")
    mean, std, compiled_fp32 = bench_ov_native(
        fp32_ir, sample_feed,
        config.num_warmup, config.num_iters,
        inference_precision="f32")
    size_mb = fp32_ir.with_suffix(".bin").stat().st_size / 1e6
    results.append(("ov_fp32", mean, std, size_mb))
    print(f"    {mean:.2f} +/- {std:.2f} ms  |  {size_mb:.1f} MB")

    # 2. OV native BF16 -- FP32 IR + BF16 compile hint, no NNCF.
    # On AVX512_BF16 / AMX-BF16 hosts this gives ~1.5-3x over FP32 on
    # CNNs and ~3-5x on transformer matmuls (matmul-shaped workloads
    # benefit more from AMX tile registers).
    print(f"\n  [2/5] OV native BF16: {fp32_ir.name} "
          f"(INFERENCE_PRECISION_HINT=bf16)")
    mean, std, compiled_bf16 = bench_ov_native(
        fp32_ir, sample_feed,
        config.num_warmup, config.num_iters,
        inference_precision="bf16")
    results.append(("ov_bf16", mean, std, size_mb))    # same IR as FP32
    print(f"    {mean:.2f} +/- {std:.2f} ms  |  IR shared with FP32")

    # 3. OV native INT8 -- NNCF-quantized IR. No precision hint: the
    # plugin honours the FakeQuantize ops in the graph and picks the
    # surrounding-op precision itself.
    print(f"\n  [3/5] OV native INT8 (NNCF): {int8_ir.name}")
    mean, std, compiled_int8 = bench_ov_native(
        int8_ir, sample_feed,
        config.num_warmup, config.num_iters,
        inference_precision=None)
    size_mb_int8 = int8_ir.with_suffix(".bin").stat().st_size / 1e6
    results.append(("ov_int8", mean, std, size_mb_int8))
    print(f"    {mean:.2f} +/- {std:.2f} ms  |  {size_mb_int8:.1f} MB")

    # 4. ORT + OV-EP FP32
    print(f"\n  [4/5] ORT + OV-EP FP32: {fp32_onnx.name}")
    out = bench_ort_ov_ep(
        fp32_onnx, sample_feed,
        config.num_warmup, config.num_iters,
        cache_dir=config.cache_dir / "ort_ov_cache")
    if out is None:
        print(f"    SKIPPED -- onnxruntime-openvino not installed")
    else:
        mean, std = out
        size_mb = fp32_onnx.stat().st_size / 1e6
        results.append(("ort_ov_fp32", mean, std, size_mb))
        print(f"    {mean:.2f} +/- {std:.2f} ms  |  {size_mb:.1f} MB")

    # 5. ORT + OV-EP INT8
    print(f"\n  [5/5] ORT + OV-EP INT8 (QDQ): {qdq_onnx.name}")
    out = bench_ort_ov_ep(
        qdq_onnx, sample_feed,
        config.num_warmup, config.num_iters,
        cache_dir=config.cache_dir / "ort_ov_cache")
    if out is None:
        print(f"    SKIPPED -- onnxruntime-openvino not installed")
    else:
        mean, std = out
        size_mb = qdq_onnx.stat().st_size / 1e6
        results.append(("ort_ov_int8", mean, std, size_mb))
        print(f"    {mean:.2f} +/- {std:.2f} ms  |  {size_mb:.1f} MB")

    # Latency summary -- use PLAIN_LABELS so the column widths line up.
    fp32_ms = next((r[1] for r in results if r[0] == "ov_fp32"),
                   results[0][1])
    print(f"\n  --- {model_name} latency summary ---")
    print(f"  {'Variant':<18} {'ms':>10} {'+/-':>8} "
          f"{'MB':>7} {'Speedup':>9}")
    print(f"  {'-' * 56}")
    for variant, mean, std, size_mb in results:
        print(f"  {PLAIN_LABELS[variant]:<18} "
              f"{mean:>10.2f} {std:>8.2f} {size_mb:>7.1f} "
              f"{fp32_ms / mean:>8.2f}x")

    # Runtime model inspection (native OV path only -- ORT + OV-EP
    # wraps the OV runtime in a way that does not surface
    # get_runtime_model() to the ORT side). dump_first_op=True on
    # the FP32 inspect makes the rt_info contents visible in the
    # log, so a future OV release that renames keys is easy to spot.
    try:
        s_fp32 = inspect_runtime_model(compiled_fp32, dump_first_op=True)
        print_inspect_summary(f"{model_name}_ov_fp32", s_fp32)
    except Exception as e:
        print(f"  Inspect failed for ov_fp32: {type(e).__name__}: {e}")

    try:
        s_bf16 = inspect_runtime_model(compiled_bf16)
        print_inspect_summary(f"{model_name}_ov_bf16", s_bf16)
    except Exception as e:
        print(f"  Inspect failed for ov_bf16: {type(e).__name__}: {e}")

    try:
        s_int8 = inspect_runtime_model(compiled_int8)
        print_inspect_summary(f"{model_name}_ov_int8", s_int8)
    except Exception as e:
        print(f"  Inspect failed for ov_int8: {type(e).__name__}: {e}")

    plot_engine_latency(results, figure_title, figure_name, config,
                        ref_lines=ref_lines)
    return results


def run_resnet(config: Config):
    print("\n" + "-" * 72)
    print("ResNet-18 — OpenVINO deployment on Intel CPU")
    print("-" * 72)

    cache = config.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    fp32_onnx = cache / "resnet18_fp32.onnx"
    qdq_onnx = cache / "resnet18_qdq_int8.onnx"
    fp32_ir = cache / "resnet18_fp32.xml"
    int8_ir = cache / "resnet18_int8.xml"

    if not fp32_onnx.exists() or config.force_rebuild:
        _resnet_fp32_onnx(fp32_onnx, config.resnet_batch,
                          config.resnet_image_size)
    else:
        print(f"  Reusing: {fp32_onnx.name}")

    feeds = _resnet_calib_feeds(config.num_calib_batches, config.resnet_batch,
                                config.resnet_image_size, config.seed)
    print(f"  Calibration: {len(feeds)} batches "
          f"x {config.resnet_batch} images")

    convert_onnx_to_ir(fp32_onnx, fp32_ir, config.force_rebuild)

    if not qdq_onnx.exists() or config.force_rebuild:
        _resnet_qdq_onnx(fp32_onnx, qdq_onnx, feeds)
    else:
        print(f"  Reusing: {qdq_onnx.name}")

    if not int8_ir.exists() or config.force_rebuild:
        ignored = _find_resnet_ignored_ops(fp32_ir)
        quantize_with_nncf(
            fp32_ir, int8_ir, feeds,
            model_type="cnn",
            ignored_op_names=ignored,
            force=config.force_rebuild)
    else:
        print(f"  Reusing INT8 IR: {int8_ir.name}")

    sample_feed = feeds[0]

    return _run_model_pipeline(
        model_name="resnet18",
        fp32_onnx=fp32_onnx, qdq_onnx=qdq_onnx,
        fp32_ir=fp32_ir, int8_ir=int8_ir,
        sample_feed=sample_feed,
        config=config,
        figure_name="CH09_F08_Kalyanarangan_ov_resnet_latency",
        figure_title=f"ResNet-18 OpenVINO latency on Intel CPU "
                     f"(batch {config.resnet_batch})",
    )


def run_bert(config: Config):
    print("\n" + "-" * 72)
    print("BERT-base SST-2 — OpenVINO deployment on Intel CPU")
    print("-" * 72)

    cache = config.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    bert_dir = cache / "bert-sst2-fp32"
    fp32_onnx = bert_dir / "model.onnx"
    qdq_onnx = cache / "bert-sst2-qdq.onnx"
    fp32_ir = cache / "bert-sst2-fp32.xml"
    int8_ir = cache / "bert-sst2-int8.xml"

    if not fp32_onnx.exists() or config.force_rebuild:
        _bert_fp32_onnx(bert_dir, config.bert_batch, config.bert_seq_len)
    else:
        print(f"  Reusing: {fp32_onnx}")

    feeds = _bert_calib_feeds(config.num_calib_batches, config.bert_batch,
                              config.bert_seq_len, config.seed)
    print(f"  Calibration: {len(feeds)} batches "
          f"x {config.bert_batch} samples (seq {config.bert_seq_len})")

    convert_onnx_to_ir(fp32_onnx, fp32_ir, config.force_rebuild)

    if not qdq_onnx.exists() or config.force_rebuild:
        _bert_qdq_onnx(fp32_onnx, qdq_onnx, feeds)
    else:
        print(f"  Reusing: {qdq_onnx.name}")

    if not int8_ir.exists() or config.force_rebuild:
        ignored = _find_bert_ignored_ops(fp32_ir)
        quantize_with_nncf(
            fp32_ir, int8_ir, feeds,
            model_type="transformer",
            ignored_op_names=ignored,
            force=config.force_rebuild)
    else:
        print(f"  Reusing INT8 IR: {int8_ir.name}")

    sample_feed = feeds[0]

    # Reference lines: section 9.2's CPU EP numbers from Table 9.2
    # (BERT-base INT8 batch 8 seq 128 on CPU EP). Setting these as
    # dotted reference lines lets the figure show how OV / OV-EP move
    # against the section-9.2 baseline on the same hardware budget.
    ref_lines = [
        ("Sec 9.2 CPU EP QDQ (Cascade Lake)", 223.4, "#666666"),
    ]

    return _run_model_pipeline(
        model_name="bert_sst2",
        fp32_onnx=fp32_onnx, qdq_onnx=qdq_onnx,
        fp32_ir=fp32_ir, int8_ir=int8_ir,
        sample_feed=sample_feed,
        config=config,
        figure_name="CH09_F09_Kalyanarangan_ov_bert_latency",
        figure_title=f"BERT-base SST-2 OpenVINO latency on Intel CPU "
                     f"(batch {config.bert_batch}, seq {config.bert_seq_len})",
        ref_lines=ref_lines,
    )


# --- Argument parsing and main ---------------------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Ch9 sec 9.4 — OpenVINO deployment for ResNet-18 + BERT")
    p.add_argument("--mode", default="all",
                   choices=["resnet", "bert", "all"])
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--resnet-batch", type=int, default=32)
    p.add_argument("--bert-batch", type=int, default=8)
    p.add_argument("--bert-seq-len", type=int, default=128)
    p.add_argument("--num-warmup", type=int, default=10)
    p.add_argument("--num-iters", type=int, default=50)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    cfg = Config(
        mode=args.mode, save_plots=args.save_plots,
        force_rebuild=args.force_rebuild,
        resnet_batch=args.resnet_batch,
        bert_batch=args.bert_batch, bert_seq_len=args.bert_seq_len,
        num_warmup=args.num_warmup, num_iters=args.num_iters,
    )
    if args.cache_dir:
        cfg.cache_dir = Path(args.cache_dir)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    return cfg


def main():
    config = parse_args()

    print("=" * 72)
    print("Chapter 9 sec 9.4 — OpenVINO pipelines for edge and desktop CPUs")
    print("=" * 72)
    print(f"  Mode:          {config.mode}")
    print(f"  Save plots:    {config.save_plots}")
    print(f"  Force rebuild: {config.force_rebuild}")
    print()
    print_environment(config)

    ov = get_ov()
    nncf = get_nncf()
    if ov is None:
        print("\n  ERROR: openvino is required.")
        sys.exit(1)
    if nncf is None:
        print("\n  ERROR: nncf is required.")
        sys.exit(1)

    modes = ["resnet", "bert"] if config.mode == "all" else [config.mode]

    for m in modes:
        try:
            if m == "resnet":
                run_resnet(config)
            elif m == "bert":
                run_bert(config)
        except Exception as e:
            print(f"\n  FAILED mode={m}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            print(f"\n  Continuing to next mode (if any)...")

    print("\n" + "=" * 72)
    print("Done.")
    print("=" * 72)


if __name__ == "__main__":
    main()