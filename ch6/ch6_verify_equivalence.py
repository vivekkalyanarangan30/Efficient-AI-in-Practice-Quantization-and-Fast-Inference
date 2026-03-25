#!/usr/bin/env python3
"""
ch6_verify_equivalence.py — Cross-framework numerical equivalence
and operator coverage diagnostics.

Companion script for Section 6.5 of
"Efficient AI in Practice: Quantization and Fast Inference"

Feeds identical inputs through PyTorch FP32, TorchAO INT8, ONNX FP32,
ORT Dynamic INT8, and ORT Static INT8 — then compares output tensors
to verify that ported models agree at the numerical level.

Also inspects ONNX graphs to report which operators received
quantization and which fell back to FP32.

Usage:
    python ch6_verify_equivalence.py --mode equivalence-vision
    python ch6_verify_equivalence.py --mode equivalence-nlp
    python ch6_verify_equivalence.py --mode operator-coverage
    python ch6_verify_equivalence.py --all --save-plots

Requirements:
    pip install torch torchvision torchao transformers datasets
    pip install onnx onnxruntime onnxruntime-extensions
    pip install matplotlib numpy

Optional (for TFLite comparison):
    pip install tensorflow~=2.18.0
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Conditional imports — degrade gracefully
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize_static,
        quant_pre_process,
        CalibrationDataReader,
        QuantFormat,
        QuantType,
    )
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths — all outputs are script-relative
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output_verify"
ARTIFACT_DIR = SCRIPT_DIR / "output_onnx"  # from ch6_onnx_export_path.py
TFLITE_DIR = SCRIPT_DIR / "output_tflite"  # from ch6_tf_mot_path.py
NUM_EVAL_SAMPLES = 100  # samples for tensor comparison
NUM_CALIB_SAMPLES = 200  # samples for ORT static calibration
SEED = 42

# ImageNette WordNet IDs in sorted order (alphabetical = class 0–9)
IMAGENETTE_WNIDS = [
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
]

# ImageNette class index → ImageNet-1K class index
IMAGENETTE_TO_INET = {
    0: 0,    # tench
    1: 217,  # English springer
    2: 482,  # cassette player
    3: 491,  # chain saw
    4: 497,  # church
    5: 566,  # French horn
    6: 569,  # garbage truck
    7: 571,  # gas pump
    8: 574,  # golf ball
    9: 701,  # parachute
}

# ---------------------------------------------------------------------------
# Manning figure style
# ---------------------------------------------------------------------------
# Colors chosen for grayscale separation (verified against Manning palette)
COLORS = {
    "blue_l2": "#6BA5D7",   # Blue Level 2 — medium gray in grayscale
    "orange_l2": "#FFB458",  # Orange Level 2 — light-medium gray
    "green_l3": "#80C21D",   # Green Level 3 — medium-dark gray (custom)
    "red_l3": "#D31518",     # Red Level 3 — dark gray
    "purple_l2": "#D4ABF3",  # Purple Level 2 — light gray
    "gray_l2": "#B0B0B0",   # Neutral medium gray
}
HATCHES = ["//", "\\\\", "xx", "..", "||", "--"]

FIG_MAX_WIDTH = 5.6  # inches
FIG_MAX_HEIGHT = 7.0


def apply_manning_style():
    """Configure matplotlib for Manning Publications compliance."""
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "pdf.fonttype": 42,  # TrueType — required by Manning
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig, name, output_dir=None):
    """Save figure as both PNG (300 DPI) and PDF (vector, fonttype 42)."""
    d = output_dir or OUTPUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=300)
    fig.savefig(d / f"{name}.pdf")
    print(f"  Saved: {d / name}.png")
    print(f"  Saved: {d / name}.pdf")
    plt.close(fig)


# ===================================================================
# DATA LOADING
# ===================================================================

def load_imagenette(data_dir, split="val", max_samples=None):
    """Load ImageNette images, return (images, labels) as numpy arrays.

    Images are preprocessed for ResNet-18: [N, 3, 224, 224], ImageNet
    normalization. Labels are ImageNet class indices (0–999).
    Downloads ImageNette2 if not present.
    """
    from PIL import Image

    dataset_path = Path(data_dir) / "imagenette2"
    if not dataset_path.exists():
        import tarfile
        import urllib.request
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
        tgz_path = Path(data_dir) / "imagenette2.tgz"
        tgz_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading ImageNette to {tgz_path}...")
        urllib.request.urlretrieve(url, tgz_path)
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("  Done.")

    split_dir = dataset_path / split
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    images, labels = [], []
    for class_idx, wnid in enumerate(IMAGENETTE_WNIDS):
        class_dir = split_dir / wnid
        if not class_dir.exists():
            continue
        inet_label = IMAGENETTE_TO_INET[class_idx]
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpeg", ".jpg", ".png"):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                # Resize/crop to 224×224 (ResNet-18 standard)         #A
                w, h = img.size
                scale = 256 / min(w, h)
                img = img.resize(
                    (int(w * scale), int(h * scale)), Image.BILINEAR
                )
                w, h = img.size
                left = (w - 224) // 2
                top = (h - 224) // 2
                img = img.crop((left, top, left + 224, top + 224))
                # To float32 [0, 1], then ImageNet normalization      #B
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = (arr - mean) / std
                arr = arr.transpose(2, 0, 1)  # [H, W, C] → [C, H, W]
                images.append(arr)
                labels.append(inet_label)
            except Exception:
                continue
            if max_samples and len(images) >= max_samples:
                print(f"  Loaded {len(images)} ImageNette {split} images")
                return np.array(images, dtype=np.float32), \
                    np.array(labels, dtype=np.int64)
    # A Resize short edge to 256, center-crop 224×224 — standard
    #   torchvision preprocessing for ImageNet-pretrained models.
    # B ImageNet channel means and standard deviations, matching
    #   torchvision.models pretrained weight expectations.

    print(f"  Loaded {len(images)} ImageNette {split} images")
    return np.array(images, dtype=np.float32), \
        np.array(labels, dtype=np.int64)


def load_sst2_samples(n_samples, max_length=128, seed=SEED):
    """Load SST-2 validation samples, tokenized and padded to max_length.

    Returns:
        input_ids:      np.int64 [N, max_length]
        attention_mask:  np.int64 [N, max_length]
        labels:          np.int64 [N]
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset

    ds = load_dataset("glue", "sst2", split="validation")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)

    sentences = [ds[int(i)]["sentence"] for i in indices]
    labels = np.array([ds[int(i)]["label"] for i in indices], dtype=np.int64)

    encoded = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    print(f"  Loaded {len(labels)} SST-2 validation samples "
          f"(padded to {max_length} tokens)")
    return (
        encoded["input_ids"].astype(np.int64),
        encoded["attention_mask"].astype(np.int64),
        labels,
    )


# ===================================================================
# EQUIVALENCE METRICS
# ===================================================================

def compute_equivalence_metrics(baseline, variant, labels=None):
    """Compare two arrays of logits element-wise.

    Args:
        baseline: np.float32 [N, C] — FP32 reference logits
        variant:  np.float32 [N, C] — logits from quantized / exported model
        labels:   np.int64 [N] — ground-truth labels (optional)

    Returns:
        dict with max_abs_diff, mean_abs_diff, mse, cosine_sim,
        top1_agreement, and optionally baseline_acc / variant_acc
    """
    diff = baseline - variant
    abs_diff = np.abs(diff)

    # Per-sample cosine similarity, then average
    dot = np.sum(baseline * variant, axis=1)
    norm_b = np.linalg.norm(baseline, axis=1)
    norm_v = np.linalg.norm(variant, axis=1)
    cos_sim = dot / (norm_b * norm_v + 1e-12)

    preds_base = np.argmax(baseline, axis=1)
    preds_var = np.argmax(variant, axis=1)
    agreement = np.mean(preds_base == preds_var)

    result = {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "mse": float(np.mean(diff ** 2)),
        "cosine_sim": float(np.mean(cos_sim)),
        "top1_agreement": float(agreement),
    }

    if labels is not None:
        result["baseline_acc"] = float(np.mean(preds_base == labels))
        result["variant_acc"] = float(np.mean(preds_var == labels))

    return result


def print_equivalence_table(results, model_name):
    """Print a formatted comparison table to console."""
    header = (
        f"\n{'=' * 90}\n"
        f"  Numerical Equivalence: {model_name} "
        f"({NUM_EVAL_SAMPLES} samples, vs PyTorch FP32 baseline)\n"
        f"{'=' * 90}"
    )
    print(header)
    print(f"  {'Variant':<25s} {'Max|Δ|':>10s} {'Mean|Δ|':>10s} "
          f"{'MSE':>12s} {'Cosine':>8s} {'Top-1 Agr':>10s}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 12} "
          f"{'-' * 8} {'-' * 10}")
    for name, m in results.items():
        cos_str = f"{m['cosine_sim']:.6f}"
        agr_str = f"{m['top1_agreement'] * 100:.1f}%"
        print(f"  {name:<25s} {m['max_abs_diff']:>10.6f} "
              f"{m['mean_abs_diff']:>10.6f} {m['mse']:>12.8f} "
              f"{cos_str:>8s} {agr_str:>10s}")

    # Also print accuracy if labels were provided
    first = next(iter(results.values()))
    if "baseline_acc" in first:
        print(f"\n  Baseline accuracy: {first['baseline_acc'] * 100:.2f}%")
        for name, m in results.items():
            if "variant_acc" in m:
                delta = (m["variant_acc"] - first["baseline_acc"]) * 100
                print(f"  {name:<25s} accuracy: "
                      f"{m['variant_acc'] * 100:.2f}% "
                      f"(Δ = {delta:+.2f} pp)")
    print()


# ===================================================================
# PYTORCH + TORCHAO INFERENCE
# ===================================================================

def pytorch_resnet_fp32(images_np):
    """Run ResNet-18 FP32 inference. Returns logits [N, 1000]."""
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    x = torch.from_numpy(images_np)
    with torch.no_grad():
        logits = model(x).numpy()  # [N, 1000]

    return logits, model


def pytorch_bert_fp32(input_ids_np, attention_mask_np):
    """Run BERT-base FP32 inference on SST-2. Returns logits [N, 2]."""
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-SST-2"
    )
    model.eval()

    input_ids = torch.from_numpy(input_ids_np)
    attention_mask = torch.from_numpy(attention_mask_np)

    all_logits = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            out = model(
                input_ids=input_ids[i:i + batch_size],
                attention_mask=attention_mask[i:i + batch_size],
            )
            all_logits.append(out.logits.numpy())

    return np.concatenate(all_logits, axis=0), model


def torchao_resnet_int8wo(images_np, model_fp32):
    """Quantize ResNet-18 with TorchAO INT8 weight-only, return logits."""
    import copy
    from torchao.quantization import quantize_, Int8WeightOnlyConfig

    model = copy.deepcopy(model_fp32)
    quantize_(model, Int8WeightOnlyConfig())
    model.eval()

    x = torch.from_numpy(images_np)
    with torch.no_grad():
        logits = model(x).numpy()
    return logits


def torchao_bert_int8wo(input_ids_np, attention_mask_np, model_fp32):
    """Quantize BERT-base with TorchAO INT8 weight-only, return logits."""
    import copy
    from torchao.quantization import quantize_, Int8WeightOnlyConfig

    model = copy.deepcopy(model_fp32)
    quantize_(model, Int8WeightOnlyConfig())
    model.eval()

    input_ids = torch.from_numpy(input_ids_np)
    attention_mask = torch.from_numpy(attention_mask_np)

    all_logits = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            out = model(
                input_ids=input_ids[i:i + batch_size],
                attention_mask=attention_mask[i:i + batch_size],
            )
            all_logits.append(out.logits.numpy())

    return np.concatenate(all_logits, axis=0)


# ===================================================================
# ONNX EXPORT + ORT INFERENCE
# ===================================================================

def ensure_resnet_onnx_artifacts(model_fp32, images_np):
    """Export ResNet-18 to ONNX and create ORT quantized variants if needed.

    Returns dict of {variant_name: onnx_path}.
    """
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    fp32_path = ARTIFACT_DIR / "resnet18_fp32.onnx"
    dynamic_path = ARTIFACT_DIR / "resnet18_dynamic_int8.onnx"
    static_path = ARTIFACT_DIR / "resnet18_static_int8.onnx"
    preprocessed_path = ARTIFACT_DIR / "resnet18_fp32_preprocessed.onnx"

    # --- FP32 export ---
    if not fp32_path.exists():
        print("  Exporting ResNet-18 to ONNX (FP32)...")
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model_fp32, dummy, str(fp32_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            dynamo=False,
        )
        print(f"    → {fp32_path} ({fp32_path.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"  Found existing: {fp32_path.name}")

    # --- Dynamic INT8 ---
    if not dynamic_path.exists():
        print("  Quantizing ResNet-18 with ORT dynamic INT8...")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(dynamic_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],          #A
        )
        # A Only MatMul/Gemm — Conv operators produce ConvInteger
        #   nodes that lack CPU kernel support on many platforms.
        print(f"    → {dynamic_path.name} "
              f"({dynamic_path.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"  Found existing: {dynamic_path.name}")

    # --- Static INT8 ---
    if not static_path.exists():
        print("  Quantizing ResNet-18 with ORT static INT8...")
        # Preprocess
        if not preprocessed_path.exists():
            quant_pre_process(str(fp32_path), str(preprocessed_path))

        # Calibration reader
        class ResNetCalibReader(CalibrationDataReader):
            def __init__(self, imgs, input_name="input"):
                self.images = imgs
                self.input_name = input_name
                self.index = 0

            def get_next(self):
                if self.index >= len(self.images):
                    return None
                data = {self.input_name:
                        self.images[self.index:self.index + 1]}
                self.index += 1
                return data

            def rewind(self):
                self.index = 0

        calib_reader = ResNetCalibReader(
            images_np[:NUM_CALIB_SAMPLES]
        )
        quantize_static(
            model_input=str(preprocessed_path),
            model_output=str(static_path),
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        print(f"    → {static_path.name} "
              f"({static_path.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"  Found existing: {static_path.name}")

    return {
        "ONNX FP32": fp32_path,
        "ORT Dynamic INT8": dynamic_path,
        "ORT Static INT8": static_path,
    }


def ensure_bert_onnx_artifacts(model_fp32, input_ids_np, attention_mask_np):
    """Export BERT-base to ONNX and create ORT quantized variants if needed."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    fp32_path = ARTIFACT_DIR / "bert_fp32.onnx"
    dynamic_path = ARTIFACT_DIR / "bert_dynamic_int8.onnx"

    # --- FP32 export ---
    if not fp32_path.exists():
        print("  Exporting BERT-base to ONNX (FP32)...")
        dummy_ids = torch.zeros(1, 128, dtype=torch.long)
        dummy_mask = torch.ones(1, 128, dtype=torch.long)
        dummy_token_type = torch.zeros(1, 128, dtype=torch.long)
        torch.onnx.export(
            model_fp32,
            (dummy_ids, dummy_mask, dummy_token_type),
            str(fp32_path),
            opset_version=17,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            dynamo=False,
        )
        print(f"    → {fp32_path} ({fp32_path.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"  Found existing: {fp32_path.name}")

    # --- Dynamic INT8 ---
    if not dynamic_path.exists():
        print("  Quantizing BERT-base with ORT dynamic INT8...")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(dynamic_path),
            weight_type=QuantType.QInt8,
        )
        print(f"    → {dynamic_path.name} "
              f"({dynamic_path.stat().st_size / 1e6:.2f} MB)")
    else:
        print(f"  Found existing: {dynamic_path.name}")

    return {
        "ONNX FP32": fp32_path,
        "ORT Dynamic INT8": dynamic_path,
    }


def ort_inference(onnx_path, feed_dict):
    """Run ONNX Runtime inference. Returns output array(s)."""
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]

    # Batch through the session
    # ORT handles variable batch natively if dynamic axes were set
    results = sess.run(output_names, feed_dict)
    return results[0]  # first output = logits


def ort_inference_batched(onnx_path, feed_dict_fn, n_samples, batch_size=32):
    """Run ORT inference in batches. feed_dict_fn(start, end) returns
    a dict of numpy arrays for that slice."""
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    output_names = [out.name for out in sess.get_outputs()]

    all_logits = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        feed = feed_dict_fn(start, end)
        results = sess.run(output_names, feed)
        all_logits.append(results[0])

    return np.concatenate(all_logits, axis=0)


# ===================================================================
# TFLITE INFERENCE (optional)
# ===================================================================

def tflite_bert_inference(tflite_path, input_ids_np, attention_mask_np):
    """Run TFLite interpreter on BERT. Returns logits [N, 2]."""
    if not HAS_TF:
        return None

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Map input names to detail indices
    input_map = {d["name"]: d for d in input_details}

    all_logits = []
    for i in range(len(input_ids_np)):
        # TFLite BERT expects [1, 128] fixed shape
        ids = input_ids_np[i:i + 1].astype(np.int32)
        mask = attention_mask_np[i:i + 1].astype(np.int32)

        # Set tensors — handle different possible input name conventions
        for d in input_details:
            name = d["name"]
            if "input_ids" in name or name == "input_ids":
                interpreter.set_tensor(d["index"], ids)
            elif "attention_mask" in name or name == "attention_mask":
                interpreter.set_tensor(d["index"], mask)

        interpreter.invoke()
        logits = interpreter.get_tensor(output_details[0]["index"])
        all_logits.append(logits.copy())

    return np.concatenate(all_logits, axis=0).astype(np.float32)


# ===================================================================
# OPERATOR COVERAGE ANALYSIS
# ===================================================================

def analyze_onnx_operator_coverage(onnx_path, model_label=""):
    """Inspect ONNX graph and report quantization coverage by op type.

    For QDQ format: identifies ops that have QuantizeLinear /
    DequantizeLinear neighbors — those ops execute in INT8.

    For operator-replacement format: identifies QLinear* ops.

    Returns:
        dict with op_counts, quantized_ops, fp32_ops, coverage_pct
    """
    if not HAS_ONNX:
        print("  [SKIP] onnx not installed")
        return None

    model = onnx.load(str(onnx_path))
    graph = model.graph

    # Count all op types
    op_counts = defaultdict(int)
    for node in graph.node:
        op_counts[node.op_type] += 1

    # Identify quantization-related nodes
    qdq_types = {"QuantizeLinear", "DequantizeLinear",
                 "DynamicQuantizeLinear"}
    qlinear_types = {t for t in op_counts if t.startswith("QLinear")}

    # Build a set of tensor names that flow through DequantizeLinear
    # (i.e., tensors that have been dequantized — their consumers
    #  operate on quantized data in the runtime)
    dequant_outputs = set()
    quant_inputs = set()
    for node in graph.node:
        if node.op_type == "DequantizeLinear":
            for out in node.output:
                dequant_outputs.add(out)
        if node.op_type == "QuantizeLinear":
            for inp in node.input:
                quant_inputs.add(inp)
        if node.op_type == "DynamicQuantizeLinear":
            for out in node.output:
                dequant_outputs.add(out)

    # An op is "quantized" if at least one input comes from
    # DequantizeLinear / DynamicQuantizeLinear
    compute_ops = {k: v for k, v in op_counts.items()
                   if k not in qdq_types and k not in {"Constant", "Shape",
                   "Gather", "Unsqueeze", "Squeeze", "Reshape", "Flatten",
                   "Concat", "Transpose", "Cast", "Identity", "Slice",
                   "ConstantOfShape", "Expand", "Where", "Equal",
                   "ScatterND"}}

    quantized_compute = defaultdict(int)
    fp32_compute = defaultdict(int)

    for node in graph.node:
        if node.op_type in qdq_types:
            continue
        if node.op_type not in compute_ops:
            continue

        # Check if any input comes from a DequantizeLinear
        has_quant_input = any(inp in dequant_outputs for inp in node.input)
        # Check if it's a QLinear replacement op
        is_qlinear = node.op_type.startswith("QLinear")

        if has_quant_input or is_qlinear:
            quantized_compute[node.op_type] += 1
        else:
            fp32_compute[node.op_type] += 1

    total_compute = sum(compute_ops.values())
    total_quantized = sum(quantized_compute.values())
    coverage_pct = (total_quantized / total_compute * 100
                    if total_compute > 0 else 0)

    # Print report
    print(f"\n  Operator Coverage: {model_label}")
    print(f"  {'─' * 60}")
    print(f"  Total compute ops: {total_compute}")
    print(f"  Quantized:         {total_quantized} ({coverage_pct:.1f}%)")
    print(f"  FP32 fallback:     {total_compute - total_quantized}")
    print(f"\n  {'Op Type':<25s} {'Total':>6s} {'INT8':>6s} {'FP32':>6s}")
    print(f"  {'─' * 25} {'─' * 6} {'─' * 6} {'─' * 6}")

    all_op_types = sorted(
        set(list(quantized_compute.keys()) + list(fp32_compute.keys()))
    )
    for op in all_op_types:
        q = quantized_compute.get(op, 0)
        f = fp32_compute.get(op, 0)
        total = q + f
        marker = "✓" if q > 0 and f == 0 else ("◐" if q > 0 else "✗")
        print(f"  {op:<25s} {total:>6d} {q:>6d} {f:>6d}  {marker}")

    # QDQ node count
    n_qdq = sum(op_counts.get(t, 0) for t in qdq_types)
    print(f"\n  QDQ / DynamicQuantize nodes: {n_qdq}")
    print()

    return {
        "model_label": model_label,
        "total_compute_ops": total_compute,
        "total_quantized": total_quantized,
        "coverage_pct": coverage_pct,
        "quantized_by_type": dict(quantized_compute),
        "fp32_by_type": dict(fp32_compute),
        "all_op_counts": dict(op_counts),
    }


# ===================================================================
# MAIN EXPERIMENT MODES
# ===================================================================

def run_equivalence_vision(save_plots=False):
    """ResNet-18 cross-framework equivalence test."""
    print("\n" + "=" * 70)
    print("  VISION: ResNet-18 on ImageNette")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading evaluation data...")
    data_dir = SCRIPT_DIR / "data"
    images_np, labels = load_imagenette(
        data_dir, split="val",
        max_samples=max(NUM_EVAL_SAMPLES, NUM_CALIB_SAMPLES),
    )
    eval_images = images_np[:NUM_EVAL_SAMPLES]
    eval_labels = labels[:NUM_EVAL_SAMPLES]

    # --- PyTorch FP32 baseline ---
    print("\n[2/4] Running PyTorch FP32 baseline...")
    baseline_logits, model_fp32 = pytorch_resnet_fp32(eval_images)
    print(f"  Logits shape: {baseline_logits.shape}")

    results = OrderedDict()

    # --- TorchAO INT8 weight-only ---
    print("\n[3/4] Running TorchAO INT8 weight-only...")
    torchao_logits = torchao_resnet_int8wo(eval_images, model_fp32)

    results["TorchAO INT8WO"] = compute_equivalence_metrics(
        baseline_logits, torchao_logits, eval_labels
    )

    # --- ONNX / ORT variants ---
    print("\n[4/4] Running ONNX Runtime variants...")
    if not HAS_ORT:
        print("  [SKIP] onnxruntime not installed")
    else:
        onnx_paths = ensure_resnet_onnx_artifacts(
            model_fp32, images_np
        )

        for variant_name, onnx_path in onnx_paths.items():
            print(f"  Running {variant_name}...")
            feed = {"input": eval_images}
            ort_logits = ort_inference(onnx_path, feed)

            results[variant_name] = compute_equivalence_metrics(
                baseline_logits, ort_logits, eval_labels
            )

    print_equivalence_table(results, "ResNet-18 on ImageNette")
    return results, baseline_logits


def run_equivalence_nlp(save_plots=False):
    """BERT-base cross-framework equivalence test."""
    print("\n" + "=" * 70)
    print("  NLP: BERT-base on SST-2")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading evaluation data...")
    input_ids, attention_mask, labels = load_sst2_samples(NUM_EVAL_SAMPLES)

    # --- PyTorch FP32 baseline ---
    print("\n[2/4] Running PyTorch FP32 baseline...")
    baseline_logits, model_fp32 = pytorch_bert_fp32(input_ids, attention_mask)
    print(f"  Logits shape: {baseline_logits.shape}")

    results = OrderedDict()

    # --- TorchAO INT8 weight-only ---
    print("\n[3/4] Running TorchAO INT8 weight-only...")
    torchao_logits = torchao_bert_int8wo(input_ids, attention_mask, model_fp32)

    results["TorchAO INT8WO"] = compute_equivalence_metrics(
        baseline_logits, torchao_logits, labels
    )

    # --- ONNX / ORT variants ---
    print("\n[4/4] Running ONNX Runtime variants...")
    if not HAS_ORT:
        print("  [SKIP] onnxruntime not installed")
    else:
        onnx_paths = ensure_bert_onnx_artifacts(
            model_fp32, input_ids, attention_mask
        )

        for variant_name, onnx_path in onnx_paths.items():
            print(f"  Running {variant_name}...")

            sess_tmp = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            expected_inputs = {inp.name for inp in sess_tmp.get_inputs()}
            del sess_tmp

            token_type_ids = np.zeros_like(input_ids)

            def make_feed(s, e, _expected=expected_inputs):
                feed = {
                    "input_ids": input_ids[s:e],
                    "attention_mask": attention_mask[s:e],
                }
                if "token_type_ids" in _expected:
                    feed["token_type_ids"] = token_type_ids[s:e]
                return feed

            ort_logits = ort_inference_batched(
                onnx_path, make_feed, len(input_ids), batch_size=32
            )

            results[variant_name] = compute_equivalence_metrics(
                baseline_logits, ort_logits, labels
            )

    # --- TFLite variant (optional) ---
    tflite_path = TFLITE_DIR / "bert_dynamic.tflite"
    if tflite_path.exists() and HAS_TF:
        print("  Running TFLite Dynamic INT8...")
        tflite_logits = tflite_bert_inference(
            tflite_path, input_ids, attention_mask
        )
        if tflite_logits is not None:
            results["TFLite Dynamic"] = compute_equivalence_metrics(
                baseline_logits, tflite_logits, labels
            )
    elif not tflite_path.exists():
        print(f"  [SKIP] TFLite: {tflite_path} not found "
              f"(run ch6_tf_mot_path.py first)")
    elif not HAS_TF:
        print("  [SKIP] TFLite: tensorflow not installed")

    print_equivalence_table(results, "BERT-base on SST-2")
    return results, baseline_logits


def run_operator_coverage():
    """Inspect ONNX graphs and report quantization coverage."""
    print("\n" + "=" * 70)
    print("  OPERATOR COVERAGE ANALYSIS")
    print("=" * 70)

    if not HAS_ONNX:
        print("  [SKIP] onnx not installed")
        return {}

    all_results = {}

    # ResNet-18 variants
    for label, fname in [
        ("ResNet-18 FP32", "resnet18_fp32.onnx"),
        ("ResNet-18 ORT Dynamic", "resnet18_dynamic_int8.onnx"),
        ("ResNet-18 ORT Static", "resnet18_static_int8.onnx"),
    ]:
        path = ARTIFACT_DIR / fname
        if path.exists():
            result = analyze_onnx_operator_coverage(path, label)
            if result:
                all_results[label] = result
        else:
            print(f"\n  [SKIP] {fname} not found in {ARTIFACT_DIR}")

    # BERT-base variants
    for label, fname in [
        ("BERT-base FP32", "bert_fp32.onnx"),
        ("BERT-base ORT Dynamic", "bert_dynamic_int8.onnx"),
    ]:
        path = ARTIFACT_DIR / fname
        if path.exists():
            result = analyze_onnx_operator_coverage(path, label)
            if result:
                all_results[label] = result
        else:
            print(f"\n  [SKIP] {fname} not found in {ARTIFACT_DIR}")

    return all_results


# ===================================================================
# PLOTTING
# ===================================================================

def plot_equivalence(vision_results, nlp_results, save_plots=False):
    """Figure 6.6: Cross-framework numerical equivalence.

    Two-panel figure:
      Left — ResNet-18 max-abs-diff (log scale) with cosine sim annotated
      Right — BERT-base max-abs-diff (log scale) with cosine sim annotated
    Top-1 agreement annotated on each bar.
    """
    if not HAS_MPL:
        print("  [SKIP] matplotlib not installed — cannot generate figures")
        return

    apply_manning_style()

    fig, axes = plt.subplots(1, 2, figsize=(FIG_MAX_WIDTH, 2.8))

    color_list = [
        COLORS["blue_l2"], COLORS["orange_l2"], COLORS["green_l3"],
        COLORS["red_l3"], COLORS["purple_l2"],
    ]
    hatch_list = HATCHES[:5]

    for ax, results, title in [
        (axes[0], vision_results, "ResNet-18 (ImageNette)"),
        (axes[1], nlp_results, "BERT-base (SST-2)"),
    ]:
        if not results:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(title, fontsize=9, fontweight="bold")
            continue

        names = list(results.keys())
        max_diffs = [results[n]["max_abs_diff"] for n in names]
        cosine_sims = [results[n]["cosine_sim"] for n in names]
        agreements = [results[n]["top1_agreement"] for n in names]

        y_pos = np.arange(len(names))

        for i, (name, md) in enumerate(zip(names, max_diffs)):
            c = color_list[i % len(color_list)]
            h = hatch_list[i % len(hatch_list)]
            bar = ax.barh(
                y_pos[i], md, height=0.6,
                color=c, hatch=h, edgecolor="black", linewidth=0.5,
            )
            # Annotate with cosine sim and agreement
            agr_str = f"{agreements[i] * 100:.1f}%"
            cos_str = f"cos={cosine_sims[i]:.4f}"
            ax.text(
                md * 1.3 if md > 0 else 1e-8, y_pos[i],
                f"  {cos_str}, agr={agr_str}",
                va="center", fontsize=6.5,
            )

        ax.set_xscale("log")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Max |Δ logit| vs FP32 baseline (log scale)", fontsize=7)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.invert_yaxis()
        # Set sensible x-axis limits
        if max_diffs:
            ax.set_xlim(
                left=min(max_diffs) * 0.3,
                right=max(max_diffs) * 50,
            )

    fig.suptitle(
        "Cross-framework numerical equivalence",
        fontsize=10, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_plots:
        save_figure(fig, "CH06_F06_Kalyanarangan")
    else:
        plt.show()


def plot_operator_coverage(coverage_results, save_plots=False):
    """Figure 6.7: Operator quantization coverage.

    Grouped horizontal stacked bar chart showing quantized vs FP32 ops
    for ResNet-18 dynamic, ResNet-18 static, and BERT-base dynamic.
    """
    if not HAS_MPL:
        print("  [SKIP] matplotlib not installed — cannot generate figures")
        return

    apply_manning_style()

    # Select the quantized variants (skip FP32 baselines)
    quant_variants = {k: v for k, v in coverage_results.items()
                      if "FP32" not in k}

    if not quant_variants:
        print("  [SKIP] No quantized ONNX variants found for coverage plot")
        return

    n_variants = len(quant_variants)
    fig, axes = plt.subplots(
        1, n_variants,
        figsize=(FIG_MAX_WIDTH, max(2.5, 0.8 * n_variants + 1.5)),
        squeeze=False,
    )
    axes = axes[0]

    for idx, (label, result) in enumerate(quant_variants.items()):
        ax = axes[idx]

        quantized = result["quantized_by_type"]
        fp32 = result["fp32_by_type"]
        all_ops = sorted(
            set(list(quantized.keys()) + list(fp32.keys())),
            key=lambda x: quantized.get(x, 0) + fp32.get(x, 0),
            reverse=True,
        )
        # Show top 12 op types to avoid clutter
        all_ops = all_ops[:12]

        y_pos = np.arange(len(all_ops))
        q_counts = [quantized.get(op, 0) for op in all_ops]
        f_counts = [fp32.get(op, 0) for op in all_ops]

        ax.barh(y_pos, q_counts, height=0.6,
                color=COLORS["blue_l2"], hatch="//",
                edgecolor="black", linewidth=0.5, label="INT8")
        ax.barh(y_pos, f_counts, height=0.6, left=q_counts,
                color=COLORS["orange_l2"], hatch="\\\\",
                edgecolor="black", linewidth=0.5, label="FP32")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_ops, fontsize=6.5)
        ax.set_xlabel("Op count", fontsize=7)
        ax.set_title(
            f"{label}\n({result['coverage_pct']:.0f}% coverage)",
            fontsize=8, fontweight="bold",
        )
        ax.invert_yaxis()
        if idx == 0:
            ax.legend(fontsize=6.5, loc="lower right")

    fig.suptitle(
        "Operator quantization coverage by ONNX variant",
        fontsize=10, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if save_plots:
        save_figure(fig, "CH06_F07_Kalyanarangan")
    else:
        plt.show()


def plot_logit_scatter(
    vision_results_raw, nlp_results_raw, save_plots=False
):
    """Figure 6.8: Logit-level scatter — FP32 baseline vs selected variant.

    Two panels:
      Left  — ResNet-18: PyTorch FP32 logits vs ORT Static INT8 logits
      Right — BERT-base: PyTorch FP32 logits vs ORT Dynamic INT8 logits
    Each point is one logit element from one sample.
    """
    if not HAS_MPL:
        return

    apply_manning_style()

    fig, axes = plt.subplots(1, 2, figsize=(FIG_MAX_WIDTH, 2.6))

    for ax, data, title in [
        (axes[0], vision_results_raw, "ResNet-18"),
        (axes[1], nlp_results_raw, "BERT-base"),
    ]:
        if data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(title, fontsize=9, fontweight="bold")
            continue

        baseline, variant, variant_name = data

        # Flatten for scatter
        b_flat = baseline.flatten()
        v_flat = variant.flatten()

        # Subsample if too many points
        if len(b_flat) > 5000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(b_flat), 5000, replace=False)
            b_flat = b_flat[idx]
            v_flat = v_flat[idx]

        ax.scatter(
            b_flat, v_flat, s=3, alpha=0.3,
            color=COLORS["blue_l2"], edgecolors="none",
        )

        # Identity line
        lo = min(b_flat.min(), v_flat.min())
        hi = max(b_flat.max(), v_flat.max())
        margin = (hi - lo) * 0.05
        ax.plot(
            [lo - margin, hi + margin], [lo - margin, hi + margin],
            color=COLORS["red_l3"], linewidth=1, linestyle="--",
            label="y = x (perfect agreement)",
        )
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlabel("PyTorch FP32 logit", fontsize=7)
        ax.set_ylabel(f"{variant_name} logit", fontsize=7)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, loc="upper left")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        "Logit-level agreement: FP32 baseline vs quantized variant",
        fontsize=10, fontweight="bold", y=1.03,
    )
    plt.tight_layout()

    if save_plots:
        save_figure(fig, "CH06_F08_Kalyanarangan")
    else:
        plt.show()


# ===================================================================
# FULL-RUN ORCHESTRATION (for --all and scatter plot data collection)
# ===================================================================

def run_equivalence_vision_full(save_plots=False):
    """Vision equivalence with raw logits stored for scatter plot."""
    print("\n" + "=" * 70)
    print("  VISION: ResNet-18 on ImageNette")
    print("=" * 70)

    print("\n[1/5] Loading evaluation data...")
    data_dir = SCRIPT_DIR / "data"
    images_np, labels = load_imagenette(
        data_dir, split="val",
        max_samples=max(NUM_EVAL_SAMPLES, NUM_CALIB_SAMPLES),
    )
    eval_images = images_np[:NUM_EVAL_SAMPLES]
    eval_labels = labels[:NUM_EVAL_SAMPLES]

    print("\n[2/5] Running PyTorch FP32 baseline...")
    baseline_logits, model_fp32 = pytorch_resnet_fp32(eval_images)
    print(f"  Logits shape: {baseline_logits.shape}")

    results = OrderedDict()
    scatter_variant = None
    scatter_name = None

    print("\n[3/5] Running TorchAO INT8 weight-only...")
    torchao_logits = torchao_resnet_int8wo(eval_images, model_fp32)
    results["TorchAO INT8WO"] = compute_equivalence_metrics(
        baseline_logits, torchao_logits, eval_labels
    )

    print("\n[4/5] Running ONNX Runtime variants...")
    if HAS_ORT:
        onnx_paths = ensure_resnet_onnx_artifacts(model_fp32, images_np)

        for variant_name, onnx_path in onnx_paths.items():
            print(f"  Running {variant_name}...")
            feed = {"input": eval_images}
            try:
                ort_logits = ort_inference(onnx_path, feed)
            except Exception as e:
                print(f"    [SKIP] {variant_name} failed: {e}")
                continue
            results[variant_name] = compute_equivalence_metrics(
                baseline_logits, ort_logits, eval_labels
            )
            # Keep ORT Static for scatter plot (most interesting pair)
            if "Static" in variant_name:
                scatter_variant = ort_logits
                scatter_name = variant_name
    else:
        print("  [SKIP] onnxruntime not installed")

    print("\n[5/5] Results:")
    print_equivalence_table(results, "ResNet-18 on ImageNette")

    # Scatter data: use ORT Static if available, else TorchAO
    if scatter_variant is None:
        scatter_variant = torchao_logits
        scatter_name = "TorchAO INT8WO"

    scatter_data = (baseline_logits, scatter_variant, scatter_name)
    return results, scatter_data


def run_equivalence_nlp_full(save_plots=False):
    """NLP equivalence with raw logits stored for scatter plot."""
    print("\n" + "=" * 70)
    print("  NLP: BERT-base on SST-2")
    print("=" * 70)

    print("\n[1/5] Loading evaluation data...")
    input_ids, attention_mask, labels = load_sst2_samples(NUM_EVAL_SAMPLES)

    print("\n[2/5] Running PyTorch FP32 baseline...")
    baseline_logits, model_fp32 = pytorch_bert_fp32(input_ids, attention_mask)
    print(f"  Logits shape: {baseline_logits.shape}")

    results = OrderedDict()
    scatter_variant = None
    scatter_name = None

    print("\n[3/5] Running TorchAO INT8 weight-only...")
    torchao_logits = torchao_bert_int8wo(
        input_ids, attention_mask, model_fp32
    )
    results["TorchAO INT8WO"] = compute_equivalence_metrics(
        baseline_logits, torchao_logits, labels
    )

    print("\n[4/5] Running ONNX Runtime variants...")
    if HAS_ORT:
        onnx_paths = ensure_bert_onnx_artifacts(
            model_fp32, input_ids, attention_mask
        )

        for variant_name, onnx_path in onnx_paths.items():
            print(f"  Running {variant_name}...")

            # Build feed with token_type_ids if the ONNX model expects it
            sess_tmp = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            expected_inputs = {inp.name for inp in sess_tmp.get_inputs()}
            del sess_tmp

            token_type_ids = np.zeros_like(input_ids)

            def make_feed(s, e, _expected=expected_inputs):
                feed = {
                    "input_ids": input_ids[s:e],
                    "attention_mask": attention_mask[s:e],
                }
                if "token_type_ids" in _expected:
                    feed["token_type_ids"] = token_type_ids[s:e]
                return feed

            try:
                ort_logits = ort_inference_batched(
                    onnx_path, make_feed, len(input_ids), batch_size=32
                )
            except Exception as e:
                print(f"    [SKIP] {variant_name} failed: {e}")
                continue
            results[variant_name] = compute_equivalence_metrics(
                baseline_logits, ort_logits, labels
            )
            # Keep ORT Dynamic for scatter (most interesting for BERT)
            if "Dynamic" in variant_name:
                scatter_variant = ort_logits
                scatter_name = variant_name
    else:
        print("  [SKIP] onnxruntime not installed")

    # TFLite (optional)
    tflite_path = TFLITE_DIR / "bert_dynamic.tflite"
    if tflite_path.exists() and HAS_TF:
        print("  Running TFLite Dynamic INT8...")
        tflite_logits = tflite_bert_inference(
            tflite_path, input_ids, attention_mask
        )
        if tflite_logits is not None:
            results["TFLite Dynamic"] = compute_equivalence_metrics(
                baseline_logits, tflite_logits, labels
            )
    elif not tflite_path.exists():
        print(f"  [SKIP] TFLite: {tflite_path.name} not found")
    elif not HAS_TF:
        print("  [SKIP] TFLite: tensorflow not installed")

    print("\n[5/5] Results:")
    print_equivalence_table(results, "BERT-base on SST-2")

    if scatter_variant is None:
        scatter_variant = torchao_logits
        scatter_name = "TorchAO INT8WO"

    scatter_data = (baseline_logits, scatter_variant, scatter_name)
    return results, scatter_data


# ===================================================================
# MAIN
# ===================================================================

def main():
    global ARTIFACT_DIR, TFLITE_DIR, NUM_EVAL_SAMPLES, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Cross-framework numerical equivalence "
                    "and operator coverage diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=[
            "equivalence-vision",
            "equivalence-nlp",
            "operator-coverage",
        ],
        help="Run a specific analysis mode",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all modes and generate all figures",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save figures as PNG (300 DPI) + PDF (vector)",
    )
    parser.add_argument(
        "--artifact-dir", type=str, default=None,
        help="Directory containing ONNX artifacts from ch6_onnx_export_path.py "
             "(default: ./output_onnx)",
    )
    parser.add_argument(
        "--tflite-dir", type=str, default=None,
        help="Directory containing TFLite artifacts from ch6_tf_mot_path.py "
             "(default: ./output_tflite)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=NUM_EVAL_SAMPLES,
        help=f"Number of evaluation samples (default: {NUM_EVAL_SAMPLES})",
    )

    args = parser.parse_args()

    if not args.mode and not args.all:
        parser.print_help()
        sys.exit(1)

    # Override globals if flags provided
    if args.artifact_dir:
        ARTIFACT_DIR = Path(args.artifact_dir)
    if args.tflite_dir:
        TFLITE_DIR = Path(args.tflite_dir)
    NUM_EVAL_SAMPLES = args.num_samples
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Eval samples:   {NUM_EVAL_SAMPLES}")
    print(f"  Artifact dir:   {ARTIFACT_DIR}")
    print(f"  TFLite dir:     {TFLITE_DIR}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  PyTorch:        {'✓' if HAS_TORCH else '✗'}")
    print(f"  ONNX:           {'✓' if HAS_ONNX else '✗'}")
    print(f"  ONNX Runtime:   {'✓' if HAS_ORT else '✗'}")
    print(f"  TensorFlow:     {'✓' if HAS_TF else '✗'}")
    print(f"  Matplotlib:     {'✓' if HAS_MPL else '✗'}")

    all_results = {}

    if args.all:
        # Run everything — collect raw logits for scatter plots
        vision_results, vision_scatter = run_equivalence_vision_full(
            args.save_plots
        )
        nlp_results, nlp_scatter = run_equivalence_nlp_full(
            args.save_plots
        )
        coverage_results = run_operator_coverage()

        all_results["vision"] = vision_results
        all_results["nlp"] = nlp_results
        all_results["coverage"] = coverage_results

        if args.save_plots:
            print("\n" + "=" * 70)
            print("  GENERATING FIGURES")
            print("=" * 70)
            plot_equivalence(vision_results, nlp_results, save_plots=True)
            plot_operator_coverage(coverage_results, save_plots=True)
            plot_logit_scatter(
                vision_scatter, nlp_scatter, save_plots=True
            )

    elif args.mode == "equivalence-vision":
        results, _ = run_equivalence_vision()
        all_results["vision"] = results

    elif args.mode == "equivalence-nlp":
        results, _ = run_equivalence_nlp()
        all_results["nlp"] = results

    elif args.mode == "operator-coverage":
        results = run_operator_coverage()
        all_results["coverage"] = results

    # Save results JSON
    json_path = OUTPUT_DIR / "equivalence_results.json"

    def make_serializable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, OrderedDict):
            return {k: make_serializable(v) for k, v in obj.items()}
        return obj

    with open(json_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nResults saved: {json_path}")
    print("Done.")


if __name__ == "__main__":
    main()