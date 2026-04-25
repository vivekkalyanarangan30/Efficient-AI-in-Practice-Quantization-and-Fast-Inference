"""
Chapter 9, Section 9.3 — Build TensorRT engines with real integer execution
Companion script for "Efficient AI in Practice: Quantization and Fast Inference"

What this script demonstrates:
  TensorRT has two INT8 paths. *Implicit* (calibrator-based, deprecated in
  TRT 10.x) lets the builder opportunistically pick INT8 kernels wherever
  it finds a speedup. *Explicit* (QDQ-in-ONNX, NVIDIA's recommended path)
  pins quantization placements at specific layers, and TRT must honor those
  placements even when doing so breaks profitable fusions. NVIDIA's own
  documentation: "for some networks, initial explicit quantization might
  exhibit higher latency compared to implicit quantization."

  Two models exercise both paths: ResNet-18 (CNN, continuity with section
  9.2's CPU benchmarks) and BERT-base SST-2 (transformer, completes the
  TRT EP column missing in Table 9.2).

Modes:
  --mode resnet   Build + bench + inspect 4 ResNet engines  (Figure 9.6)
  --mode bert     Build + bench + inspect 4 BERT engines    (Figure 9.7)
  --mode all      Both (default)

Usage:
  # Recommended: Google Colab L4 (SM 8.9), or any GPU with INT8 Tensor Cores.
  python ch9_tensorrt_engines.py --mode all --save-plots

  # Just one model:
  python ch9_tensorrt_engines.py --mode resnet --save-plots
  python ch9_tensorrt_engines.py --mode bert   --save-plots

  # Force rebuild of cached engines:
  python ch9_tensorrt_engines.py --mode all --force-rebuild

Install (Colab cell):
  !pip install -q --extra-index-url https://pypi.nvidia.com \
      tensorrt-cu12 onnxruntime-gpu onnx onnxscript \
      torch torchvision matplotlib pillow \
      transformers optimum[onnxruntime-gpu] datasets

Hardware target:
  L4 (SM 8.9, Ada):       INT8 Tensor Cores                              ✓
  A100 (SM 8.0, Ampere):  INT8 Tensor Cores                              ✓
  T4 (SM 7.5, Turing):    INT8 Tensor Cores                              ✓
  H100 (SM 9.0, Hopper):  INT8 + FP8 Tensor Cores                        ✓

  FP8 is intentionally not demonstrated here. Section 8.2 covered the FP8
  format. The TRT FP8 build path requires nvidia-modelopt-emitted FP8 Q/DQ
  ONNX nodes plus Hopper-or-later silicon — out of scope for this script.

Note on quantization sensitivity:
  ResNet-18 and BERT each have layers that don't survive INT8 quantization
  cleanly. Chapter 4 covered the principle (input-adjacent and classifier-
  adjacent layers are disproportionately sensitive); this script applies it.

  ResNet exclusions: stem Conv (input-adjacent) and fc classifier
                     (output-adjacent + ORT bias-quantization quirk).
  BERT exclusions:   embedding Gather (input-adjacent — section 9.2's
                     established pattern) and the final classifier.
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
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

SCRIPT_DIR = Path(__file__).resolve().parent


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class Config:
    mode: str = "all"
    save_plots: bool = False
    force_rebuild: bool = False
    output_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "figures")
    cache_dir: Path = field(default_factory=lambda: SCRIPT_DIR / "trt_cache")

    # ResNet-18 inference shape
    resnet_batch: int = 32
    resnet_image_size: int = 224

    # BERT-base inference shape (matches section 9.2's defaults)
    bert_batch: int = 8
    bert_seq_len: int = 128

    # Calibration: 8 batches per model
    num_calib_batches: int = 8

    # Benchmark settings
    num_warmup: int = 20
    num_iters: int = 100

    workspace_gb: int = 1
    seed: int = 42


# ─── Manning figure style ────────────────────────────────────────────────────

COLORS = {
    "fp32":          "#7570b3",
    "fp16":          "#e7298a",
    "int8_implicit": "#d95f02",
    "int8_explicit": "#1b9e77",
}
HATCHES = {
    "fp32": "..", "fp16": "\\\\",
    "int8_implicit": "//", "int8_explicit": "xx",
}
DISPLAY_LABELS = {
    "fp32": "FP32", "fp16": "FP16",
    "int8_implicit": "INT8\nimplicit", "int8_explicit": "INT8\nexplicit",
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


# ─── Environment probe ───────────────────────────────────────────────────────

def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(name)
        except PackageNotFoundError:
            return "NOT INSTALLED"
    except ImportError:
        return "NOT INSTALLED"


def _trt_pkg_version() -> str:
    """TRT can be installed as 'tensorrt' or 'tensorrt-cu12'."""
    for pkg in ("tensorrt", "tensorrt-cu12", "tensorrt-cu11"):
        v = _pkg_version(pkg)
        if v != "NOT INSTALLED":
            return f"{v} ({pkg})"
    return "NOT INSTALLED"


def get_trt():
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        return None


def get_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def print_environment(config: Config):
    trt = get_trt()
    torch = get_torch()

    print(f"  Python:         {sys.version.split()[0]}")
    print(f"  torch:          {_pkg_version('torch')}")
    print(f"  tensorrt:       "
          f"{_trt_pkg_version() if trt else 'NOT INSTALLED'}")
    print(f"  onnxruntime:    {_pkg_version('onnxruntime-gpu') or _pkg_version('onnxruntime')}")
    print(f"  optimum:        {_pkg_version('optimum')}")
    print(f"  transformers:   {_pkg_version('transformers')}")

    if torch and torch.cuda.is_available():
        name = torch.cuda.get_device_name()
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cap = torch.cuda.get_device_capability()
        print(f"  GPU:            {name} (SM {cap[0]}.{cap[1]}, "
              f"{mem_gb:.0f} GB)")
    else:
        print(f"  GPU:            None — TensorRT requires a CUDA GPU.")
    print(f"  Cache dir:      {config.cache_dir}")


# ─── ORT calibration data reader (model-agnostic) ────────────────────────────

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


# ─── Generic INT32-bias DQ stripper ──────────────────────────────────────────
# ORT emits INT32-typed DequantizeLinear nodes around biases (Conv bias and
# Gemm bias are stored as INT32 for accumulator-width compatibility with ORT's
# own INT8 kernels). TRT's ONNX parser rejects INT32-typed DQ outright. The
# fix: dequantize biases statically and replace the DQ node with an FP32
# initializer. Belt-and-suspenders alongside `op_types_to_quantize`.

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


# ─── Generic TRT engine builder ──────────────────────────────────────────────

def _make_trt_builder(trt):
    logger = trt.Logger(trt.Logger.WARNING)
    return trt.Builder(logger), logger


def _parse_onnx(trt, builder, logger, onnx_path: Path):
    network = builder.create_network(0)                                    #A
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(
                f"ONNX parse failed for {onnx_path.name}:\n" +
                "\n".join(str(e) for e in errors))
    return network


#A `create_network(0)` is TRT 10's idiom — explicit batch is the default and
#  no flag is needed. TRT 8.x required `1 << EXPLICIT_BATCH`; passing flags
#  in TRT 10 still works but logs a deprecation warning.


class _Int8EntropyCalib:
    """TRT calibrator factory used only for the implicit INT8 path.

    Instantiated as a closure-based subclass to keep the calibration
    batches and device buffer alive across get_batch() calls. TRT is
    very particular: returning a numpy array, or letting the device
    tensor go out of scope, segfaults the builder.
    """

    @staticmethod
    def make(trt, calib_feeds: List[Dict[str, np.ndarray]],
             input_names: List[str], cache_path: Path):
        import torch

        class _Cal(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                super().__init__()
                self.feeds = calib_feeds
                self.idx = 0
                self.cache_path = cache_path
                self.device_tensors: Dict[str, "torch.Tensor"] = {}

            def get_batch_size(self):
                return self.feeds[0][input_names[0]].shape[0]

            def get_batch(self, names):
                if self.idx >= len(self.feeds):
                    return None
                feed = self.feeds[self.idx]
                self.idx += 1
                ptrs = []
                for n in names:
                    arr = feed[n]
                    t = torch.from_numpy(arr).contiguous().cuda()
                    self.device_tensors[n] = t        # keep alive
                    ptrs.append(int(t.data_ptr()))
                return ptrs

            def read_calibration_cache(self):
                if self.cache_path.exists():
                    return self.cache_path.read_bytes()
                return None

            def write_calibration_cache(self, cache):
                self.cache_path.write_bytes(bytes(cache))

        return _Cal()


def build_trt_engine(
    onnx_path: Path, engine_path: Path, precision: str, config: Config,
    calib_feeds: Optional[List[Dict[str, np.ndarray]]] = None,
    input_names: Optional[List[str]] = None,
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Path:
    """Build a TRT engine at the requested precision and serialize to disk."""
    trt = get_trt()
    print(f"\n  Building {precision} engine: {engine_path.name}")
    t0 = time.perf_counter()

    builder, logger = _make_trt_builder(trt)
    network = _parse_onnx(trt, builder, logger, onnx_path)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                              config.workspace_gb * (1 << 30))
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED              #B

    # If the parsed network has any dynamic dimensions, TRT requires an
    # optimization profile that binds concrete shapes for the builder.    #Bp
    # Optimum's BERT export uses dynamic_axes={'input_ids': {0:'batch',
    # 1:'sequence'}, ...}; the torchvision ResNet export keeps shapes
    # static. We detect dynamic dims and define a single-shape profile
    # (min == opt == max) using input_shapes when needed.
    if input_shapes is not None:
        has_dynamic = False
        for i in range(network.num_inputs):
            t = network.get_input(i)
            if any(dim < 0 for dim in t.shape):
                has_dynamic = True
                break
        if has_dynamic:
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                t = network.get_input(i)
                if t.name in input_shapes:
                    s = tuple(input_shapes[t.name])
                    profile.set_shape(t.name, min=s, opt=s, max=s)
            cfg.add_optimization_profile(profile)

    if precision == "fp32":
        pass
    elif precision == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8_implicit":
        if calib_feeds is None or not input_names:
            raise ValueError("int8_implicit needs calib_feeds + input_names")
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.set_flag(trt.BuilderFlag.FP16)                                 #C
        cfg.int8_calibrator = _Int8EntropyCalib.make(
            trt, calib_feeds, input_names,
            engine_path.with_suffix(".calib"))
    elif precision == "int8_explicit":
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.set_flag(trt.BuilderFlag.FP16)
    else:
        raise ValueError(f"Unknown precision: {precision}")

    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError(
            f"build_serialized_network returned None for {precision} — "
            f"check the TRT log output above for the failing layer. "
            f"Common cause: missing INT8 kernel for a fused QDQ supernode "
            f"(add the offending layer to nodes_to_exclude in the QDQ pass).")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    elapsed = time.perf_counter() - t0
    size_mb = engine_path.stat().st_size / 1e6
    print(f"    Built in {elapsed:.1f} s  |  size = {size_mb:.1f} MB")
    return engine_path


#B DETAILED verbosity is required for engine inspector to populate Format
#  fields per layer. Without it the inspector returns layer names only.
#Bp Optimization profile is required when the ONNX has dynamic dimensions —
#  TRT can't validate or pick tactics without knowing concrete shapes.
#  Optimum's transformer exports always have dynamic batch and sequence
#  axes; torchvision exports keep them static. We use min=opt=max for
#  single-shape inference. A production server handling multiple batch
#  sizes would set min=1 and max=max_batch with opt=expected_batch.
#C FP16 alongside INT8 isn't a contradiction. INT8 mode gives TRT permission
#  to pick INT8 kernels; FP16 gives it permission to fall back to FP16
#  rather than FP32 for layers where INT8 isn't profitable. Without FP16
#  enabled, the fallback is FP32 — inflating the "silent upcast" count
#  on implicit builds.


# ─── Engine load + benchmark + inspect ───────────────────────────────────────

def _load_engine(engine_path: Path):
    trt = get_trt()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def _io_names(engine) -> Tuple[List[str], List[str]]:
    trt = get_trt()
    inp, out = [], []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            inp.append(n)
        else:
            out.append(n)
    return inp, out


def bench_engine(
    engine_path: Path,
    input_shapes: Dict[str, Tuple[int, ...]],
    input_dtypes: Dict[str, "np.dtype"],
    num_warmup: int, num_iters: int,
) -> Tuple[float, float]:
    """CUDA-event-timed mean and stdev (ms) for engine inference."""
    import torch

    engine = _load_engine(engine_path)
    context = engine.create_execution_context()
    in_names, out_names = _io_names(engine)

    in_tensors = {}
    for n in in_names:
        shape = input_shapes[n]
        np_dtype = input_dtypes[n]
        # For dynamic-shape engines (BERT) we must set the runtime shape
        # before binding the address; for static engines this is a no-op.
        # context.set_input_shape exists in TRT 10.x; older APIs use
        # set_binding_shape with an index, but we target TRT 10+.
        if any(d < 0 for d in engine.get_tensor_shape(n)):
            context.set_input_shape(n, shape)
        if np_dtype == np.int64:
            t = torch.zeros(shape, dtype=torch.int64, device="cuda")
        elif np_dtype == np.int32:
            t = torch.zeros(shape, dtype=torch.int32, device="cuda")
        else:
            t = torch.randn(shape, dtype=torch.float32, device="cuda")
        in_tensors[n] = t
        context.set_tensor_address(n, int(t.data_ptr()))

    out_tensors = {}
    for n in out_names:
        shape = tuple(context.get_tensor_shape(n))
        t = torch.empty(shape, dtype=torch.float32, device="cuda")
        out_tensors[n] = t
        context.set_tensor_address(n, int(t.data_ptr()))

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(num_warmup):
            context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    with torch.cuda.stream(stream):
        for i in range(num_iters):
            starts[i].record(stream)
            context.execute_async_v3(stream.cuda_stream)
            ends[i].record(stream)
    stream.synchronize()

    timings = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return float(np.mean(timings)), float(np.std(timings))


def inspect_engine_layers(engine_path: Path) -> Dict:
    """Return a structured summary of layer execution precision.

    Reads the I/O Format strings ('Int8 NCHW', 'Half NCHW', 'Float NCHW')
    from each layer's first input tensor — that's where TRT 10 reports the
    *executed* precision. The top-level layer["Precision"] field reports
    the *requested* precision (which defaults to FLOAT) and is NOT what
    we want.
    """
    trt = get_trt()
    engine = _load_engine(engine_path)
    inspector = engine.create_engine_inspector()
    info = inspector.get_engine_information(
        trt.LayerInformationFormat.JSON)
    data = json.loads(info)
    layers = data.get("Layers", [])

    counts = {"Int8": 0, "Half": 0, "Float": 0, "Other": 0}
    detail = []
    for layer in layers:
        inputs = layer.get("Inputs", [])
        fmt = ""
        if inputs and isinstance(inputs[0], dict):
            fmt = inputs[0].get("Format/Datatype",
                                inputs[0].get("Format", ""))
        elif "Inputs" in layer and isinstance(layer["Inputs"], str):
            fmt = layer["Inputs"]

        kind = layer.get("LayerType", "?")
        name = layer.get("Name", "?")[:60]

        # Reformat/Constant/Shuffle layers are bookkeeping; bin into Other
        if kind in ("Reformat", "Constant", "Shuffle", "Identity"):
            counts["Other"] += 1
        elif "Int8" in fmt:
            counts["Int8"] += 1
        elif "Half" in fmt:
            counts["Half"] += 1
        elif "Float" in fmt:
            counts["Float"] += 1
        else:
            counts["Other"] += 1
        detail.append((name, kind, fmt))

    return {"counts": counts, "detail": detail, "total": len(layers)}


# ─── Plotting helper (shared between models) ─────────────────────────────────

def plot_engine_latency(
    results: List[Tuple[str, float, float, float, float]],
    title: str, fig_name: str, config: Config,
    ref_lines: Optional[List[Tuple[str, float, str]]] = None,
):
    """Two-panel bar chart: latency + speedup vs FP32."""
    apply_manning_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.2))

    labels = [r[0] for r in results]
    latencies = [r[1] for r in results]
    stds = [r[2] for r in results]
    fp32_ms = next((r[1] for r in results if r[0] == "fp32"), latencies[0])
    speedups = [fp32_ms / r[1] for r in results]

    for i, (lbl, lat, sd) in enumerate(zip(labels, latencies, stds)):
        ax1.bar(i, lat, yerr=sd, color=COLORS[lbl], hatch=HATCHES[lbl],
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
                     fontsize=6, color=ref_color, va="bottom", ha="right")

    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels([DISPLAY_LABELS[l] for l in labels], fontsize=7)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title(title)

    for i, (lbl, sp) in enumerate(zip(labels, speedups)):
        ax2.bar(i, sp, color=COLORS[lbl], hatch=HATCHES[lbl],
                edgecolor="black", linewidth=0.5, width=0.65)
        ax2.text(i, sp + max(speedups) * 0.02, f"{sp:.2f}x",
                 ha="center", va="bottom", fontsize=7)
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels([DISPLAY_LABELS[l] for l in labels], fontsize=7)
    ax2.set_ylabel("Speedup vs FP32")
    ax2.set_title("Speedup")

    fig.tight_layout()
    save_or_show(fig, fig_name, config)


def print_inspect_summary(name: str, summary: Dict):
    counts = summary["counts"]
    total_compute = counts["Int8"] + counts["Half"] + counts["Float"]
    print(f"\n  Inspecting {name}:")
    print(f"    Total layers: {summary['total']}")
    print(f"    Compute layers by execution precision:")
    for dt in ("Int8", "Half", "Float"):
        frac = counts[dt] / max(1, total_compute)
        print(f"      {dt:<6}: {counts[dt]:>4}  ({frac*100:5.1f}%)")
    print(f"    Other (Reformat/Constant/Shuffle): {counts['Other']}")
    print(f"    First 3 compute layers:")
    n = 0
    for name_, kind, fmt in summary["detail"]:
        if kind in ("Reformat", "Constant", "Shuffle", "Identity"):
            continue
        print(f"      [{(fmt or 'unknown')[:18]:<18}] {kind:<14} {name_}")
        n += 1
        if n >= 3:
            break


# ─── ResNet-18 pipeline ──────────────────────────────────────────────────────

def _resnet_fp32_onnx(out_path: Path, batch: int, image_size: int):
    import torch
    import torchvision.models as tvm

    print(f"  Exporting ResNet-18 FP32 ONNX: {out_path.name}")
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1).eval()
    dummy = torch.randn(batch, 3, image_size, image_size)
    torch.onnx.export(                                                     #D
        model, dummy, str(out_path),
        input_names=["input"], output_names=["logits"],
        opset_version=17, dynamic_axes=None, do_constant_folding=True,
        dynamo=False,
    )
    print(f"    Wrote: {out_path.name} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")


#D Static-shape export keeps the TRT build simple — no IOptimizationProfile
#  needed. Opset 17 keeps consistency across the chapter.


def _resnet_calib_feeds(num_batches: int, batch: int,
                        image_size: int, seed: int) -> List[Dict]:
    """ImageNet-normalized random tensors. Calibration data quality affects
    final accuracy but not the engine mechanics this section is teaching."""
    rng = np.random.default_rng(seed)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    feeds = []
    for _ in range(num_batches):
        raw = rng.random((batch, 3, image_size, image_size), dtype=np.float32)
        feeds.append({"input": ((raw - mean) / std).astype(np.float32)})
    return feeds


def _find_input_consuming_convs(onnx_path: Path) -> List[str]:
    """Return Conv node names that consume the model input directly.

    For ResNet-18 this is /conv1/Conv. TRT can't fuse the QDQ-wrapped
    `stem Conv + ReLU + MaxPool` supernode into INT8 — 3-channel input plus
    trailing MaxPool falls outside TRT's INT8 Conv tactic patterns.
    Excluding the stem keeps it FP16 while subsequent Convs run INT8.
    """
    import onnx
    m = onnx.load(str(onnx_path))
    inputs = {i.name for i in m.graph.input}
    return [n.name for n in m.graph.node
            if n.op_type == "Conv"
            and any(x in inputs for x in n.input)]


def _resnet_qdq_onnx(fp32: Path, qdq: Path,
                     calib_feeds: List[Dict[str, np.ndarray]]):
    """ORT static QDQ ONNX with the four ORT/TRT bridge settings dialed in:
      1. quant_format=QDQ: TRT rejects QOperator (QLinearConv).
      2. ActivationSymmetric=True: TRT requires zero_point=0 activations.
      3. op_types_to_quantize=['Conv']: skips Gemm to avoid INT32-bias DQ.
      4. nodes_to_exclude=[stem_conv]: the fused stem has no INT8 kernel.
    Plus a post-pass to strip any INT32-typed DQ that slipped through.
    """
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

    stem = _find_input_consuming_convs(fp32)
    if stem:
        print(f"    Excluding stem Conv from quantization: {stem}")

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


# ─── BERT pipeline ───────────────────────────────────────────────────────────

BERT_MODEL_ID = "textattack/bert-base-uncased-SST-2"


def _bert_fp32_onnx(out_dir: Path, batch: int, seq_len: int):
    """Export BERT-base SST-2 to FP32 ONNX via Optimum.

    Uses the same model and pattern as section 9.2. Optimum produces
    a directory containing model.onnx + config; we return the .onnx path.
    """
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
    """SST-2-style calibration feeds with random in-vocab token IDs."""
    rng = np.random.default_rng(seed)
    feeds = []
    vocab_size = 30522  # bert-base-uncased
    for _ in range(num_batches):
        ids = rng.integers(100, vocab_size,
                           size=(batch, seq_len), dtype=np.int64)
        ids[:, 0] = 101    # [CLS]
        ids[:, -1] = 102   # [SEP]
        feeds.append({
            "input_ids": ids,
            "attention_mask": np.ones((batch, seq_len), dtype=np.int64),
            "token_type_ids": np.zeros((batch, seq_len), dtype=np.int64),
        })
    return feeds


def _find_classifier_matmul(onnx_path: Path) -> List[str]:
    """Return MatMul node names in the classifier head.

    For BERT-SST2 the classifier is Linear(768, 2) feeding the model output.
    Walks back from graph outputs through up to 4 hops looking for a MatMul.
    """
    import onnx
    m = onnx.load(str(onnx_path))
    outputs = {o.name for o in m.graph.output}
    output_to_producer = {}
    for n in m.graph.node:
        for out in n.output:
            output_to_producer[out] = n

    classifier_matmuls = []
    for out_name in outputs:
        node = output_to_producer.get(out_name)
        if node is None:
            continue
        cur = node
        for _ in range(4):
            if cur is None:
                break
            if cur.op_type == "MatMul":
                classifier_matmuls.append(cur.name)
                break
            if cur.input:
                cur = output_to_producer.get(cur.input[0])
            else:
                break
    return list(set(classifier_matmuls))


def _bert_qdq_onnx(fp32_onnx: Path, qdq_onnx: Path,
                   calib_feeds: List[Dict[str, np.ndarray]]):
    """ORT static QDQ ONNX for BERT, with TRT-aware settings.

    Mirrors section 9.2's MatMul-only override (skips the embedding Gather
    that would collapse token reps), adds classifier-head exclusion for
    chapter-4 sensitivity, and applies the same activation-symmetric and
    INT32-DQ-strip steps as ResNet."""
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

    classifier = _find_classifier_matmul(fp32_onnx)
    if classifier:
        print(f"    Excluding classifier MatMul(s): {classifier}")

    quantize_static(
        str(src), str(qdq_onnx),
        calibration_data_reader=_CalibReader(calib_feeds),
        quant_format=QuantFormat.QDQ,
        per_channel=False,                                                 #E
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["MatMul"],                                   #F
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


#E Per-tensor weight quantization for BERT MatMul. Section 9.2 found that
#  per-channel buys <0.3pp on transformer MatMuls vs the activation error.
#F MatMul-only quantization. ORT's defaults also quantize the embedding
#  Gather, which collapses distinct token reps into shared INT8 scales —
#  a known PTQ failure mode on BERT (section 9.2). NVIDIA routes
#  embeddings through QAT for exactly this reason.


# ─── Mode runner (shared) ────────────────────────────────────────────────────

def _run_model_pipeline(
    model_name: str,
    fp32_onnx_path: Path,
    qdq_onnx_path: Path,
    calib_feeds: List[Dict[str, np.ndarray]],
    input_shapes: Dict[str, Tuple[int, ...]],
    input_dtypes: Dict[str, "np.dtype"],
    config: Config,
    figure_name: str,
    figure_title: str,
    ref_lines: Optional[List[Tuple[str, float, str]]] = None,
) -> List[Tuple[str, float, float, float, float]]:
    """Build 4 engines, benchmark, inspect, plot. Shared between models."""
    cache = config.cache_dir
    engines = {p: cache / f"{model_name}_{p}.engine"
               for p in ("fp32", "fp16", "int8_implicit", "int8_explicit")}

    for prec, eng in engines.items():
        if eng.exists() and not config.force_rebuild:
            print(f"\n  Reusing {prec} engine: {eng.name} "
                  f"({eng.stat().st_size / 1e6:.1f} MB)")
            continue

        if prec == "int8_implicit":
            cal_path = eng.with_suffix(".calib")
            if config.force_rebuild and cal_path.exists():
                cal_path.unlink()
            build_trt_engine(
                fp32_onnx_path, eng, prec, config,
                calib_feeds=calib_feeds,
                input_names=list(input_shapes.keys()),
                input_shapes=input_shapes)
        elif prec == "int8_explicit":
            build_trt_engine(qdq_onnx_path, eng, prec, config,
                             input_shapes=input_shapes)
        else:
            build_trt_engine(fp32_onnx_path, eng, prec, config,
                             input_shapes=input_shapes)

    print(f"\n  ─── Benchmarking {model_name} engines ───")
    results = []
    for prec, eng in engines.items():
        mean_ms, std_ms = bench_engine(
            eng, input_shapes, input_dtypes,
            config.num_warmup, config.num_iters)
        first_shape = next(iter(input_shapes.values()))
        batch = first_shape[0]
        throughput = (batch * 1000.0) / mean_ms
        size_mb = eng.stat().st_size / 1e6
        results.append((prec, mean_ms, std_ms, throughput, size_mb))
        print(f"    {prec:<14}: {mean_ms:6.2f} +/- {std_ms:.2f} ms  "
              f"|  {throughput:>6.0f} samples/s  |  {size_mb:5.1f} MB")

    fp32_ms = results[0][1]
    print(f"\n  ─── {model_name} latency summary ───")
    print(f"  {'Engine':<14} {'ms':>8} {'+/-':>6} "
          f"{'samp/s':>8} {'MB':>7} {'Speedup':>8}")
    print(f"  {'-' * 56}")
    for prec, mean_ms, std_ms, tput, size_mb in results:
        print(f"  {prec:<14} {mean_ms:>8.2f} {std_ms:>6.2f} "
              f"{tput:>8.0f} {size_mb:>7.1f} {fp32_ms / mean_ms:>7.2f}x")

    for prec, eng in engines.items():
        try:
            summary = inspect_engine_layers(eng)
            print_inspect_summary(f"{model_name}_{prec}", summary)
        except Exception as e:
            print(f"\n  Inspect failed for {prec}: "
                  f"{type(e).__name__}: {e}")

    plot_engine_latency(results, figure_title, figure_name, config,
                        ref_lines=ref_lines)
    return results


def run_resnet(config: Config):
    print("\n" + "─" * 72)
    print("ResNet-18 — 4 TRT engines on ImageNet-class CNN")
    print("─" * 72)

    cache = config.cache_dir
    cache.mkdir(parents=True, exist_ok=True)
    fp32 = cache / "resnet18_fp32.onnx"
    qdq = cache / "resnet18_qdq_int8.onnx"

    if not fp32.exists() or config.force_rebuild:
        _resnet_fp32_onnx(fp32, config.resnet_batch, config.resnet_image_size)
    else:
        print(f"  Reusing: {fp32.name}")

    feeds = _resnet_calib_feeds(config.num_calib_batches, config.resnet_batch,
                                config.resnet_image_size, config.seed)
    print(f"  Calibration: {len(feeds)} batches "
          f"x {config.resnet_batch} images")

    if not qdq.exists() or config.force_rebuild:
        _resnet_qdq_onnx(fp32, qdq, feeds)
    else:
        print(f"  Reusing: {qdq.name}")

    return _run_model_pipeline(
        model_name="resnet18",
        fp32_onnx_path=fp32, qdq_onnx_path=qdq, calib_feeds=feeds,
        input_shapes={"input": (config.resnet_batch, 3,
                                config.resnet_image_size,
                                config.resnet_image_size)},
        input_dtypes={"input": np.float32},
        config=config,
        figure_name="CH09_F06_Kalyanarangan_trt_resnet_latency",
        figure_title=f"ResNet-18 inference latency "
                     f"(batch {config.resnet_batch}, L4)",
    )


def run_bert(config: Config):
    print("\n" + "─" * 72)
    print("BERT-base SST-2 — 4 TRT engines on transformer classifier")
    print("─" * 72)

    cache = config.cache_dir
    bert_dir = cache / "bert-sst2-fp32"
    fp32_onnx = bert_dir / "model.onnx"
    qdq_onnx = cache / "bert-sst2-qdq.onnx"

    if not fp32_onnx.exists() or config.force_rebuild:
        _bert_fp32_onnx(bert_dir, config.bert_batch, config.bert_seq_len)
    else:
        print(f"  Reusing: {fp32_onnx}")

    feeds = _bert_calib_feeds(config.num_calib_batches, config.bert_batch,
                              config.bert_seq_len, config.seed)
    print(f"  Calibration: {len(feeds)} batches "
          f"x {config.bert_batch} samples (seq {config.bert_seq_len})")

    if not qdq_onnx.exists() or config.force_rebuild:
        _bert_qdq_onnx(fp32_onnx, qdq_onnx, feeds)
    else:
        print(f"  Reusing: {qdq_onnx.name}")

    input_shapes = {
        "input_ids": (config.bert_batch, config.bert_seq_len),
        "attention_mask": (config.bert_batch, config.bert_seq_len),
        "token_type_ids": (config.bert_batch, config.bert_seq_len),
    }
    input_dtypes = {n: np.int64 for n in input_shapes}

    # Reference lines from section 9.2's Table 9.2 (BERT batch 8 seq 128 on L4)
    ref_lines = [
        ("Sec 9.2 CUDA EP INT8", 14.4, "#666666"),
        ("Sec 9.2 FP32 optimized", 8.96, "#aaaaaa"),
    ]

    return _run_model_pipeline(
        model_name="bert_sst2",
        fp32_onnx_path=fp32_onnx, qdq_onnx_path=qdq_onnx, calib_feeds=feeds,
        input_shapes=input_shapes,
        input_dtypes=input_dtypes,
        config=config,
        figure_name="CH09_F07_Kalyanarangan_trt_bert_latency",
        figure_title=f"BERT-base SST-2 latency "
                     f"(batch {config.bert_batch}, "
                     f"seq {config.bert_seq_len}, L4)",
        ref_lines=ref_lines,
    )


# ─── Argument parsing and main ───────────────────────────────────────────────

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Ch9 sec 9.3 — TRT INT8 engines for ResNet-18 and BERT")
    p.add_argument("--mode", default="all",
                   choices=["resnet", "bert", "all"])
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--resnet-batch", type=int, default=32)
    p.add_argument("--bert-batch", type=int, default=8)
    p.add_argument("--bert-seq-len", type=int, default=128)
    p.add_argument("--num-warmup", type=int, default=20)
    p.add_argument("--num-iters", type=int, default=100)
    p.add_argument("--workspace-gb", type=int, default=1)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    cfg = Config(
        mode=args.mode, save_plots=args.save_plots,
        force_rebuild=args.force_rebuild,
        resnet_batch=args.resnet_batch,
        bert_batch=args.bert_batch, bert_seq_len=args.bert_seq_len,
        num_warmup=args.num_warmup, num_iters=args.num_iters,
        workspace_gb=args.workspace_gb,
    )
    if args.cache_dir:
        cfg.cache_dir = Path(args.cache_dir)
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    return cfg


def main():
    config = parse_args()

    print("=" * 72)
    print("Chapter 9 sec 9.3 — TRT engines with real INT8 (ResNet + BERT)")
    print("=" * 72)
    print(f"  Mode:          {config.mode}")
    print(f"  Save plots:    {config.save_plots}")
    print(f"  Force rebuild: {config.force_rebuild}")
    print()
    print_environment(config)

    trt = get_trt()
    torch = get_torch()
    if trt is None or torch is None or not torch.cuda.is_available():
        print("\n  ERROR: tensorrt + CUDA-enabled torch required.")
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