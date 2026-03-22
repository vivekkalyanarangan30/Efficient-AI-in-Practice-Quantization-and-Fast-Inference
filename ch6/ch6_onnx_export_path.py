#!/usr/bin/env python3
"""
Chapter 6, Section 6.4 — Export to ONNX and drive ONNX Runtime quantization

Models : ResNet-18 (ImageNette) and BERT-base (SST-2)
Target : CPU via ONNX Runtime CPUExecutionProvider
Focus  : Export from PyTorch FP32 → ONNX, then apply ORT's own quantizers
         (dynamic + static) on the exported ONNX graph.

This script intentionally stops at the quantized-artifact boundary.
Runtime optimization, execution-provider tuning, and Optimum/ORTQuantizer
workflows belong to Chapter 9 (section 9.2).

Usage:
    python ch6_onnx_export_path.py --model resnet --mode all
    python ch6_onnx_export_path.py --model bert --mode all
    python ch6_onnx_export_path.py --model resnet --mode mixed  # layer sensitivity analysis
    python ch6_onnx_export_path.py --all --save-plots           # both models + figures

Modes:
    export   — Export FP32 PyTorch model to ONNX and validate
    dynamic  — ORT dynamic quantization (INT8 weights, on-the-fly activations)
    static   — ORT static quantization (INT8 weights + activations, with calibration)
    mixed    — Mixed-precision analysis: per-layer sensitivity + selective exclusion
    evaluate — Run all ONNX variants through ORT and report accuracy/size
    all      — Run everything in sequence

Requirements:
    pip install torch torchvision transformers datasets onnx onnxruntime matplotlib tqdm numpy
"""

import argparse
import gc
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Output directory — script-relative so it works from project root or ch6/
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = SCRIPT_DIR / "ch6_onnx_artifacts"

# ---------------------------------------------------------------------------
# Lazy imports — keeps startup fast and error messages clear
# ---------------------------------------------------------------------------

def _import_torch():
    import torch
    return torch

def _import_onnx():
    import onnx
    return onnx

def _import_ort():
    import onnxruntime as ort
    return ort

def _import_ort_quantization():
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )
    return CalibrationDataReader, QuantFormat, QuantType, quantize_dynamic, quantize_static


# ============================================================================
# Shared utilities
# ============================================================================

def file_size_mb(path: str) -> float:
    """Return file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def count_onnx_ops(model_path: str) -> dict:
    """Count operator types in an ONNX graph. Useful for verifying
    that quantization inserted QuantizeLinear/DequantizeLinear nodes."""
    onnx = _import_onnx()
    model = onnx.load(model_path)
    op_counts = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    return op_counts


def print_quant_ops(op_counts: dict):
    """Print quantization-relevant operators from an op-count dict."""
    quant_ops = ["QuantizeLinear", "DequantizeLinear", "MatMulInteger",
                 "ConvInteger", "QLinearConv", "QLinearMatMul",
                 "DynamicQuantizeLinear", "MatMulIntegerToFloat",
                 "QGemm"]
    found = {k: v for k, v in op_counts.items() if k in quant_ops}
    if found:
        print("  Quantization operators:")
        for op, count in sorted(found.items()):
            print(f"    {op}: {count}")
    else:
        print("  No quantization operators found (FP32 graph)")


# ============================================================================
# ImageNette ↔ ImageNet class mapping (shared with ch6_pytorch_torchao_path.py)
# ============================================================================

IMAGENETTE_TO_IMAGENET = {
    0: 0,      # tench
    1: 217,    # English springer
    2: 482,    # cassette player
    3: 491,    # chain saw
    4: 497,    # church
    5: 566,    # French horn
    6: 569,    # garbage truck
    7: 571,    # gas pump
    8: 574,    # golf ball
    9: 701,    # parachute
}
IMAGENET_TO_IMAGENETTE = {v: k for k, v in IMAGENETTE_TO_IMAGENET.items()}

# Column indices for selecting ImageNette classes from 1000-class output
# Used during evaluation: logits_10 = logits_1000[:, IMAGENETTE_COLUMNS]
IMAGENETTE_COLUMNS = [IMAGENETTE_TO_IMAGENET[i] for i in range(10)]




# ============================================================================
# ResNet-18 on ImageNette
# ============================================================================

def _ensure_imagenette(data_dir: Path):
    """Download ImageNette to data_dir if not already present."""
    imagenette_dir = data_dir / "imagenette2"
    val_dir = imagenette_dir / "val"

    if val_dir.exists():
        print(f"  Using existing ImageNette at {imagenette_dir}")
        return imagenette_dir

    import tarfile
    import urllib.request

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    tgz_path = data_dir / "imagenette2.tgz"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading ImageNette from {url}...")
    urllib.request.urlretrieve(url, tgz_path)
    print("  Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(data_dir)
    tgz_path.unlink()
    print("  Download complete.")
    return imagenette_dir


def load_imagenette_val():
    """Load ImageNette validation set with standard ImageNet preprocessing.

    Uses the local data/ folder (same as ch6_pytorch_torchao_path.py).
    Downloads from fastai S3 only if the folder is missing.
    Returns numpy arrays ready for ORT inference.
    """
    torch = _import_torch()
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    print("  Loading ImageNette validation split...")

    data_dir = SCRIPT_DIR / "data"
    imagenette_dir = _ensure_imagenette(data_dir)
    val_dir = imagenette_dir / "val"

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder(str(val_dir), transform=preprocess)

    images, labels = [], []
    for img_tensor, label in dataset:
        images.append(img_tensor.numpy())
        labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    print(f"  Loaded {len(labels)} validation images")
    return images, labels


def load_resnet18():
    """Load pretrained ResNet-18 in eval mode."""
    torch = _import_torch()
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    return model


def export_resnet_onnx(save_dir: Path) -> str:
    """
    Export ResNet-18 to ONNX with a fixed batch size of 1.

    ResNet-18 is a straightforward export — all operators have direct ONNX
    equivalents. No dynamic axes: the model is exported with fixed shape
    [1, 3, 224, 224], matching the standard ImageNet input. We add dynamic
    batch later only if needed; for this section, batch=1 is sufficient.
    """
    torch = _import_torch()
    onnx = _import_onnx()

    print(f"\n{'='*70}")
    print("EXPORT: ResNet-18 → ONNX")
    print(f"{'='*70}")

    model = load_resnet18()
    dummy_input = torch.randn(1, 3, 224, 224)

    onnx_path = str(save_dir / "resnet18_fp32.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=17,                                          #A
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,                                              #B
    )
    # A Opset 17 covers all ResNet ops and is well-supported by ORT.
    # B PyTorch 2.x defaults to the dynamo-based ONNX exporter, which
    #   can strip weights from simple models and produce broken graphs.
    #   dynamo=False forces the stable TorchScript exporter that embeds
    #   all weights into the .onnx file. This is the production-safe path
    #   until the dynamo exporter matures.

    # Validate the exported model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    size = file_size_mb(onnx_path)

    # Sanity check: ResNet-18 FP32 should be ~44 MB. If < 1 MB, the
    # exporter stripped weights (common dynamo exporter failure mode).
    if size < 1.0:
        raise RuntimeError(
            f"Exported model is only {size:.2f} MB — weights are missing. "
            f"Ensure dynamo=False is set in torch.onnx.export()."
        )

    op_counts = count_onnx_ops(onnx_path)
    n_nodes = sum(op_counts.values())

    print(f"  Saved: {onnx_path}")
    print(f"  Size:  {size:.2f} MB")
    print(f"  Graph: {n_nodes} nodes, {len(op_counts)} unique op types")
    print_quant_ops(op_counts)

    del model
    gc.collect()
    return onnx_path


def quantize_resnet_dynamic(fp32_path: str, save_dir: Path) -> str:
    """
    ORT dynamic quantization on ResNet-18.

    Dynamic quantization replaces weight tensors with INT8 and computes
    activation scales at inference time. By default, it targets MatMul,
    Gemm, AND Conv operators. The Conv path produces ConvInteger nodes,
    which are not supported on all CPUs (notably ARM64 / Apple Silicon).

    We restrict to MatMul only, which is the intended target for dynamic
    quantization anyway — ORT's own docs recommend static quantization
    for CNNs and dynamic for transformers. For ResNet-18, MatMul only
    covers the final classification layer (1 of 21 weight-bearing layers),
    so compression will be minimal. This is the honest result: dynamic
    quantization is the wrong tool for CNNs.
    """
    _, _, QuantType, quantize_dynamic, _ = _import_ort_quantization()

    print(f"\n{'='*70}")
    print("QUANTIZE: ResNet-18 — ORT dynamic (INT8 weights, MatMul only)")
    print(f"{'='*70}")

    out_path = str(save_dir / "resnet18_dynamic_int8.onnx")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],                           #B
    )
    # B Restrict to MatMul only. The default also quantizes Conv to
    #   ConvInteger, which fails on ARM64/Apple Silicon CPUs. For CNNs,
    #   static quantization (with calibration) is the correct path —
    #   it produces QDQ nodes that all execution providers support.

    size = file_size_mb(out_path)
    op_counts = count_onnx_ops(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Size:  {size:.2f} MB")
    print_quant_ops(op_counts)

    return out_path


def quantize_resnet_static(fp32_path: str, save_dir: Path) -> str:
    """
    ORT static quantization on ResNet-18 with calibration.

    Static quantization bakes both weight AND activation scales into the
    graph. This requires a calibration dataset — same role as Chapter 4's
    calibration set, but delivered through ORT's CalibrationDataReader
    interface instead of PyTorch's observer pattern.

    The three-step ORT pipeline:
      1. Preprocess the FP32 ONNX (shape inference + graph optimization)
      2. Calibrate with representative data (collect activation ranges)
      3. Quantize (insert QDQ nodes with computed scales)

    We use QDQ format (QuantizeLinear/DequantizeLinear pairs) because
    it's the recommended format since ORT 1.11 and works with all
    execution providers.
    """
    CalibrationDataReader, QuantFormat, QuantType, _, quantize_static = (
        _import_ort_quantization()
    )
    import subprocess

    print(f"\n{'='*70}")
    print("QUANTIZE: ResNet-18 — ORT static (INT8 W+A, calibrated)")
    print(f"{'='*70}")

    # Step 1: Preprocess — shape inference + optimization
    preprocessed_path = str(save_dir / "resnet18_preprocessed.onnx")
    print("  Step 1/3: Preprocessing (shape inference + optimization)...")
    subprocess.run(
        [sys.executable, "-m", "onnxruntime.quantization.preprocess",
         "--input", fp32_path, "--output", preprocessed_path],
        check=True, capture_output=True, text=True,
    )
    print(f"    Preprocessed model: {file_size_mb(preprocessed_path):.2f} MB")

    # Step 2+3: Calibrate and quantize
    # Load a subset of ImageNette for calibration
    images, _ = load_imagenette_val()
    calib_images = images[:200]  # 200 samples, same as Ch4 recommendation

    class ResNetCalibrationReader(CalibrationDataReader):
        """Feeds calibration images to ORT's static quantization pipeline."""
        def __init__(self, images, input_name="input"):
            self.images = images
            self.input_name = input_name
            self.index = 0

        def get_next(self):                                        #C
            if self.index >= len(self.images):
                return None
            data = {self.input_name: self.images[self.index:self.index+1]}
            self.index += 1
            return data
        # C get_next() returns a dict mapping input names to numpy arrays.
        #   Return None to signal end of calibration data. ORT calls this
        #   repeatedly until None, collecting activation ranges at each node.

        def rewind(self):
            self.index = 0

    print(f"  Step 2/3: Calibrating with {len(calib_images)} samples...")
    calib_reader = ResNetCalibrationReader(calib_images)

    out_path = str(save_dir / "resnet18_static_int8.onnx")
    print("  Step 3/3: Quantizing (QDQ format, per-channel weights)...")
    quantize_static(
        model_input=preprocessed_path,
        model_output=out_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,                              #D
        activation_type=QuantType.QUInt8,                          #E
        weight_type=QuantType.QInt8,                               #F
        per_channel=True,                                          #G
    )
    # D QDQ format: inserts QuantizeLinear/DequantizeLinear node pairs.
    #   The graph stays readable, and any runtime can fuse or lower them.
    # E Activations: unsigned INT8 (0–255). Asymmetric, matching the
    #   non-negative output of ReLU in ResNet.
    # F Weights: signed INT8 (–128 to 127). Symmetric by default.
    # G Per-channel: one scale per output channel, matching Chapter 3's
    #   recommendation for convolutional weights.

    size = file_size_mb(out_path)
    op_counts = count_onnx_ops(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Size:  {size:.2f} MB")
    print_quant_ops(op_counts)

    # Clean up preprocessed intermediate
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    return out_path


def inspect_quantizable_nodes(fp32_path: str):
    """
    List all nodes in the ONNX graph that ORT would quantize.
    This is the starting point for selective precision decisions:
    you can't exclude a node from quantization if you don't know its name.
    """
    onnx = _import_onnx()
    model = onnx.load(fp32_path)

    quantizable_types = {"Conv", "MatMul", "Gemm"}
    nodes = []
    for node in model.graph.node:
        if node.op_type in quantizable_types:
            nodes.append({
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            })
    return nodes


def run_mixed_precision_analysis(fp32_path: str, save_dir: Path):
    """
    Selective layer quantization: keep chosen layers in FP32,
    quantize everything else to INT8.

    ORT's quantize_static accepts a nodes_to_exclude parameter —
    a list of node names that stay in full precision. This is the
    same principle as Chapter 4's mixed-precision strategy, but
    applied at the ONNX export level rather than inside PyTorch.

    We run five configurations on ResNet-18:
      1. All INT8 (baseline — same as quantize_resnet_static)
      2. First conv in FP32 (protect the input boundary)
      3. Last block + FC in FP32 (protect the output boundary)
      4. First conv + last block + FC in FP32 (protect both)
      5. Sensitivity sweep: exclude each conv one at a time

    The teaching point: you don't have to quantize everything. ORT
    gives you a node-level knob to trade size for accuracy on exactly
    the layers that matter.
    """
    CalibrationDataReader, QuantFormat, QuantType, _, quantize_static = (
        _import_ort_quantization()
    )
    import subprocess

    print(f"\n{'='*70}")
    print("MIXED PRECISION: ResNet-18 — selective layer quantization")
    print(f"{'='*70}")

    # Step 1: Identify quantizable nodes
    nodes = inspect_quantizable_nodes(fp32_path)
    conv_nodes = [n for n in nodes if n["op_type"] == "Conv"]
    matmul_nodes = [n for n in nodes if n["op_type"] in ("MatMul", "Gemm")]

    print(f"\n  Quantizable nodes in graph:")
    print(f"    Conv:   {len(conv_nodes)} layers")
    print(f"    MatMul: {len(matmul_nodes)} layers")
    print(f"    Total:  {len(nodes)} layers")
    print()

    # Show the node names (reader needs these for nodes_to_exclude)
    print("  Conv layer node names (in graph order):")
    for i, n in enumerate(conv_nodes):
        tag = ""
        if i == 0:
            tag = "  ← first conv (input boundary)"
        elif i == len(conv_nodes) - 1:
            tag = "  ← last conv"
        print(f"    [{i:2d}] {n['name']}{tag}")
    for n in matmul_nodes:
        print(f"    [FC] {n['name']}  ← classifier head")

    # Step 2: Preprocess (reuse if exists)
    preprocessed_path = str(save_dir / "resnet18_preprocessed.onnx")
    if not os.path.exists(preprocessed_path):
        print("\n  Preprocessing FP32 graph...")
        subprocess.run(
            [sys.executable, "-m", "onnxruntime.quantization.preprocess",
             "--input", fp32_path, "--output", preprocessed_path],
            check=True, capture_output=True, text=True,
        )

    # Step 3: Load calibration data
    images, labels = load_imagenette_val()
    calib_images = images[:200]

    class _CalibReader(CalibrationDataReader):
        def __init__(self, imgs, input_name="input"):
            self.imgs = imgs
            self.input_name = input_name
            self.idx = 0
        def get_next(self):
            if self.idx >= len(self.imgs):
                return None
            data = {self.input_name: self.imgs[self.idx:self.idx+1]}
            self.idx += 1
            return data
        def rewind(self):
            self.idx = 0

    # Identify strategic node groups for exclusion
    first_conv = [conv_nodes[0]["name"]]
    last_block_names = [n["name"] for n in conv_nodes[-2:]]  # last residual block
    fc_names = [n["name"] for n in matmul_nodes]
    all_node_names = [n["name"] for n in nodes]

    # Define configurations: (label, nodes_to_exclude)
    configs = [
        ("All INT8",                         []),
        ("First conv → FP32",                first_conv),
        ("Last block + FC → FP32",           last_block_names + fc_names),
        ("First + last block + FC → FP32",   first_conv + last_block_names + fc_names),
    ]

    # Step 4: Run each configuration
    mixed_results = []

    for config_label, exclude_nodes in configs:
        tag = config_label.replace(" ", "_").replace("+", "").replace("→", "to")
        tag = tag.replace("__", "_").lower()
        out_path = str(save_dir / f"resnet18_mixed_{tag}.onnx")

        n_excluded = len(exclude_nodes)
        n_quantized = len(all_node_names) - n_excluded
        print(f"\n  Config: {config_label}")
        print(f"    Quantized: {n_quantized}/{len(all_node_names)} layers | "
              f"FP32: {n_excluded}/{len(all_node_names)} layers")

        calib_reader = _CalibReader(calib_images)
        quantize_static(
            model_input=preprocessed_path,
            model_output=out_path,
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            nodes_to_exclude=exclude_nodes if exclude_nodes else None,  #L
        )
        # L nodes_to_exclude: list of node name strings. These nodes keep
        #   FP32 weights and activations. Everything else gets QDQ pairs.
        #   This is the ONNX-level equivalent of Chapter 4's mixed-precision
        #   strategy — but applied after export, not during training.

        size = file_size_mb(out_path)
        result = evaluate_resnet_onnx(out_path, images, labels, config_label)
        result["n_excluded"] = n_excluded
        result["n_quantized"] = n_quantized
        result["n_total"] = len(all_node_names)
        result["exclude_nodes"] = exclude_nodes
        mixed_results.append(result)

    # Step 5: Per-layer sensitivity sweep
    #   For each conv layer, quantize everything EXCEPT that one layer.
    #   The accuracy difference vs full-INT8 shows that layer's sensitivity.
    print(f"\n  {'─'*60}")
    print(f"  Per-layer sensitivity sweep (excluding one layer at a time)")
    print(f"  {'─'*60}")

    full_int8_acc = mixed_results[0]["accuracy"]  # "All INT8" config
    layer_sensitivities = []

    for i, node in enumerate(conv_nodes):
        tag = f"excl_conv{i:02d}"
        out_path = str(save_dir / f"resnet18_sens_{tag}.onnx")

        calib_reader = _CalibReader(calib_images)
        quantize_static(
            model_input=preprocessed_path,
            model_output=out_path,
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            nodes_to_exclude=[node["name"]],
        )

        result = evaluate_resnet_onnx(out_path, images, labels,
                                       f"Excl [{i:2d}] {node['name']}")
        acc_gain = result["accuracy"] - full_int8_acc
        layer_sensitivities.append({
            "index": i,
            "name": node["name"],
            "accuracy": result["accuracy"],
            "acc_gain_pp": acc_gain,
            "size_mb": result["size_mb"],
        })

        # Clean up per-layer artifacts to save disk
        if os.path.exists(out_path):
            os.remove(out_path)

    # Print sensitivity ranking
    ranked = sorted(layer_sensitivities, key=lambda x: -x["acc_gain_pp"])
    print(f"\n  Layer sensitivity ranking (accuracy gain from keeping in FP32):")
    print(f"  {'Layer':<8s} {'Node name':<35s} {'Acc gain':>10s}")
    print(f"  {'─'*55}")
    for s in ranked[:10]:  # Top 10 most sensitive
        print(f"  [{s['index']:2d}]    {s['name']:<35s} {s['acc_gain_pp']:+.3f} pp")

    # Clean up preprocessed model
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    return mixed_results, layer_sensitivities, conv_nodes


def evaluate_resnet_onnx(model_path: str, images: np.ndarray,
                         labels: np.ndarray, label: str) -> dict:
    """Run ResNet-18 ONNX model through ORT and measure accuracy."""
    ort = _import_ort()

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],                        #H
    )
    # H Explicit CPUExecutionProvider — no CUDA, no auto-detection.
    #   This section targets CPU; GPU execution providers are Chapter 9.

    input_name = session.get_inputs()[0].name
    correct = 0
    total = len(labels)

    for i in range(total):
        outputs = session.run(None, {input_name: images[i:i+1]})
        # ResNet outputs 1000 ImageNet classes; select the 10 ImageNette
        # columns and argmax over those to match ImageFolder labels (0–9).
        logits_10 = outputs[0][:, IMAGENETTE_COLUMNS]
        pred = np.argmax(logits_10, axis=1)[0]
        if pred == labels[i]:
            correct += 1

    accuracy = 100.0 * correct / total
    size = file_size_mb(model_path)

    print(f"  {label:30s} — Size: {size:8.2f} MB | "
          f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

    return {
        "label": label,
        "size_mb": size,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


# ============================================================================
# BERT-base on SST-2
# ============================================================================

def load_sst2_val():
    """Load SST-2 validation set and tokenize with BERT tokenizer.

    Uses nyu-mll/glue (parquet format, no loading script) which works
    with datasets >= 3.x. Falls back to the legacy 'glue' path for
    older library versions.
    """
    from transformers import AutoTokenizer
    from datasets import load_dataset

    print("  Loading SST-2 validation split...")

    ds = None
    # Attempt 1: nyu-mll/glue (parquet, no loading script — works on datasets 3.x+)
    try:
        ds = load_dataset("nyu-mll/glue", "sst2", split="validation")
        print("    (loaded from nyu-mll/glue)")
    except Exception:
        pass

    # Attempt 2: legacy 'glue' path (works on older datasets versions)
    if ds is None:
        try:
            ds = load_dataset("glue", "sst2", split="validation")
            print("    (loaded from glue)")
        except Exception as e:
            raise RuntimeError(
                f"Cannot load SST-2 dataset. Try: pip install -U datasets\n"
                f"Error: {e}"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        "textattack/bert-base-uncased-SST-2"
    )

    # Extract sentences as a plain list of strings — required by the
    # tokenizer. ds["sentence"] may return a list or an Arrow column;
    # wrapping in list() guarantees the correct type.
    sentences = list(ds["sentence"])

    # Tokenize all samples with fixed padding to 128
    encoded = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",
    )

    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)
    # token_type_ids: BERT requires them; all zeros for single-sentence
    token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
    labels = np.array(ds["label"], dtype=np.int64)

    print(f"  Loaded {len(labels)} validation samples "
          f"(shape: [{input_ids.shape[0]}, {input_ids.shape[1]}])")
    return input_ids, attention_mask, token_type_ids, labels


def load_bert_model():
    """Load BERT-base fine-tuned on SST-2."""
    torch = _import_torch()
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-SST-2"
    ).eval()
    return model


def export_bert_onnx(save_dir: Path) -> str:
    """
    Export BERT-base to ONNX with dynamic sequence length.

    BERT export is trickier than ResNet for two reasons:
    1. Multiple inputs (input_ids, attention_mask, token_type_ids)
    2. Dynamic axes — sequence length varies across inputs

    We export with dynamic axes on both batch and sequence dimensions.
    This produces an ONNX graph that accepts any [batch, seq_len] input,
    matching production serving where batch size and sequence length vary.
    """
    torch = _import_torch()
    onnx = _import_onnx()

    print(f"\n{'='*70}")
    print("EXPORT: BERT-base (SST-2) → ONNX")
    print(f"{'='*70}")

    model = load_bert_model()

    # Dummy inputs for tracing — shape doesn't matter because of dynamic_axes
    dummy_ids = torch.ones(1, 128, dtype=torch.long)
    dummy_mask = torch.ones(1, 128, dtype=torch.long)
    dummy_types = torch.zeros(1, 128, dtype=torch.long)

    onnx_path = str(save_dir / "bert_sst2_fp32.onnx")

    # BERT forward() accepts keyword arguments; torch.onnx.export needs
    # positional args, so we pass a tuple matching the model's signature.
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask, dummy_types),
        onnx_path,
        opset_version=17,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={                                             #I
            "input_ids":      {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "token_type_ids": {0: "batch", 1: "seq_len"},
            "logits":         {0: "batch"},
        },
        dynamo=False,                                              #J
    )
    # I Dynamic axes let the same ONNX file accept variable batch sizes
    #   and sequence lengths. Without this, the graph is specialized to
    #   [1, 128] and will reject any other shape at runtime.
    # J Force the TorchScript exporter. The dynamo exporter can mishandle
    #   HuggingFace model wrappers that return dataclass outputs.

    # Validate
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    size = file_size_mb(onnx_path)

    # Sanity check: BERT-base FP32 should be ~420 MB.
    if size < 10.0:
        raise RuntimeError(
            f"Exported model is only {size:.2f} MB — weights are missing. "
            f"Ensure dynamo=False is set in torch.onnx.export()."
        )

    op_counts = count_onnx_ops(onnx_path)
    n_nodes = sum(op_counts.values())

    print(f"  Saved: {onnx_path}")
    print(f"  Size:  {size:.2f} MB")
    print(f"  Graph: {n_nodes} nodes, {len(op_counts)} unique op types")
    print_quant_ops(op_counts)

    del model
    gc.collect()
    return onnx_path


def quantize_bert_dynamic(fp32_path: str, save_dir: Path) -> str:
    """
    ORT dynamic quantization on BERT-base.

    Dynamic quantization is the recommended ORT path for transformers —
    same recommendation as TFLite dynamic range from section 6.3, and for
    the same reason: transformer activations vary too much across inputs
    for static scales to work well.

    ORT's quantize_dynamic targets MatMul nodes by default, which covers
    all 74 linear projections in BERT (Q, K, V, O, FFN up, FFN down per
    encoder block × 12 blocks + classifier head).
    """
    _, _, QuantType, quantize_dynamic, _ = _import_ort_quantization()

    print(f"\n{'='*70}")
    print("QUANTIZE: BERT-base — ORT dynamic (INT8 weights)")
    print(f"{'='*70}")

    out_path = str(save_dir / "bert_sst2_dynamic_int8.onnx")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=out_path,
        weight_type=QuantType.QInt8,
        per_channel=False,                                         #J
    )
    # J per_channel=False (default for dynamic). Dynamic quantization
    #   uses per-tensor symmetric for weights. Per-channel is available
    #   but adds overhead in the dynamic path without clear accuracy gain
    #   for transformers. Static quantization is where per-channel shines.

    size = file_size_mb(out_path)
    op_counts = count_onnx_ops(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Size:  {size:.2f} MB")
    print_quant_ops(op_counts)

    return out_path


def quantize_bert_static(fp32_path: str, save_dir: Path) -> str:
    """
    ORT static quantization on BERT-base with calibration.

    We include this to demonstrate the contrast with dynamic: static
    quantization bakes activation ranges into the graph, which can
    degrade transformer accuracy when those ranges don't represent
    all inputs. This mirrors the TFLite finding from section 6.3
    (full integer collapsed BERT to ~50% accuracy).

    ORT's static quantization is more robust than TFLite's because it
    uses QDQ format (QuantizeLinear/DequantizeLinear nodes) rather than
    replacing operators with integer-only variants. The runtime decides
    whether to fuse QDQ pairs into true integer ops. If it can't, it
    falls back to float — you don't get garbage outputs, just less
    speedup. This is a design philosophy difference worth noting.
    """
    CalibrationDataReader, QuantFormat, QuantType, _, quantize_static = (
        _import_ort_quantization()
    )
    import subprocess

    print(f"\n{'='*70}")
    print("QUANTIZE: BERT-base — ORT static (INT8 W+A, calibrated)")
    print(f"{'='*70}")

    # Step 1: Preprocess (symbolic shape inference is critical for transformers)
    preprocessed_path = str(save_dir / "bert_sst2_preprocessed.onnx")
    print("  Step 1/3: Preprocessing (symbolic shape inference + optimization)...")
    try:
        subprocess.run(
            [sys.executable, "-m", "onnxruntime.quantization.preprocess",
             "--input", fp32_path, "--output", preprocessed_path],
            check=True, capture_output=True, text=True,
        )
        source_path = preprocessed_path
        print(f"    Preprocessed model: {file_size_mb(preprocessed_path):.2f} MB")
    except subprocess.CalledProcessError as e:
        # Preprocessing can fail on complex transformer graphs due to
        # symbolic shape inference limitations. Fall back to the raw FP32
        # model — quantization still works, just with less optimization.
        print(f"    Preprocessing failed (common for transformers): {e.stderr[:200]}")
        print("    Falling back to raw FP32 model for quantization...")
        source_path = fp32_path

    # Step 2+3: Calibrate and quantize
    input_ids, attention_mask, token_type_ids, _ = load_sst2_val()
    n_calib = 200

    class BertCalibrationReader(CalibrationDataReader):
        """Feeds tokenized SST-2 samples to ORT's calibration pipeline."""
        def __init__(self, input_ids, attention_mask, token_type_ids,
                     n_samples=200):
            self.input_ids = input_ids[:n_samples]
            self.attention_mask = attention_mask[:n_samples]
            self.token_type_ids = token_type_ids[:n_samples]
            self.index = 0

        def get_next(self):
            if self.index >= len(self.input_ids):
                return None
            data = {
                "input_ids":      self.input_ids[self.index:self.index+1],
                "attention_mask": self.attention_mask[self.index:self.index+1],
                "token_type_ids": self.token_type_ids[self.index:self.index+1],
            }
            self.index += 1
            return data

        def rewind(self):
            self.index = 0

    print(f"  Step 2/3: Calibrating with {n_calib} samples...")
    calib_reader = BertCalibrationReader(
        input_ids, attention_mask, token_type_ids, n_samples=n_calib
    )

    out_path = str(save_dir / "bert_sst2_static_int8.onnx")
    print("  Step 3/3: Quantizing (QDQ format)...")
    try:
        quantize_static(
            model_input=source_path,
            model_output=out_path,
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=False,                                     #K
        )
        # K per_channel=False for BERT static. BERT's linear layers are all
        #   MatMul, not Conv. Per-channel on MatMul quantizes along axis 0 of
        #   the weight matrix, which is valid but adds overhead. We keep
        #   per-tensor here for a fair comparison with the dynamic path.
        #   Per-channel tuning is a Chapter 9 optimization topic.
    except Exception as e:
        # Static quantization on transformers commonly fails when
        # preprocessing can't complete symbolic shape inference.
        # ORT's quantize_static calls shape_inference internally, and
        # without fully resolved shapes, the intermediate model can't
        # be reloaded (protobuf parsing error on large models).
        #
        # This is the honest result: static quantization with baked-in
        # activation scales is the wrong tool for transformers anyway.
        # Dynamic quantization is the correct ORT path — matching
        # TFLite's dynamic range (section 6.3) and TorchAO's W8A8
        # dynamic (section 6.2).
        print(f"\n  Static quantization failed: {type(e).__name__}")
        print("  This is expected for transformers without successful")
        print("  preprocessing. Dynamic quantization is the correct")
        print("  ORT path for BERT — static activation ranges cannot")
        print("  faithfully represent transformer distributions.")
        # Clean up
        if os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
        return None

    size = file_size_mb(out_path)
    op_counts = count_onnx_ops(out_path)
    print(f"  Saved: {out_path}")
    print(f"  Size:  {size:.2f} MB")
    print_quant_ops(op_counts)

    # Clean up
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)

    return out_path


def evaluate_bert_onnx(model_path: str, input_ids: np.ndarray,
                       attention_mask: np.ndarray,
                       token_type_ids: np.ndarray,
                       labels: np.ndarray, label: str) -> dict:
    """Run BERT ONNX model through ORT and measure accuracy on SST-2."""
    ort = _import_ort()

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    correct = 0
    total = len(labels)

    # Infer one sample at a time (matches the book's evaluation pattern)
    for i in range(total):
        outputs = session.run(
            None,
            {
                "input_ids":      input_ids[i:i+1],
                "attention_mask": attention_mask[i:i+1],
                "token_type_ids": token_type_ids[i:i+1],
            },
        )
        pred = np.argmax(outputs[0], axis=1)[0]
        if pred == labels[i]:
            correct += 1

    accuracy = 100.0 * correct / total
    size = file_size_mb(model_path)

    print(f"  {label:30s} — Size: {size:8.2f} MB | "
          f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

    return {
        "label": label,
        "size_mb": size,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }


# ============================================================================
# PyTorch FP32 baselines (for cross-framework comparison)
# ============================================================================

def evaluate_resnet_pytorch(images: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate ResNet-18 in PyTorch FP32 — the baseline."""
    torch = _import_torch()
    model = load_resnet18()
    device = torch.device("cpu")
    model = model.to(device)

    correct = 0
    total = len(labels)
    imagenette_idx = torch.tensor(IMAGENETTE_COLUMNS, device=device)
    with torch.no_grad():
        for i in range(total):
            x = torch.from_numpy(images[i:i+1]).to(device)
            out = model(x)
            pred = out[:, imagenette_idx].argmax(dim=1).item()
            if pred == labels[i]:
                correct += 1

    accuracy = 100.0 * correct / total
    print(f"  {'PyTorch FP32 baseline':30s} — Accuracy: {accuracy:.2f}% "
          f"({correct}/{total})")

    del model
    gc.collect()
    return {"label": "PyTorch FP32", "accuracy": accuracy,
            "correct": correct, "total": total}


def evaluate_bert_pytorch(input_ids: np.ndarray, attention_mask: np.ndarray,
                          token_type_ids: np.ndarray,
                          labels: np.ndarray) -> dict:
    """Evaluate BERT-base in PyTorch FP32 — the baseline."""
    torch = _import_torch()
    model = load_bert_model()
    device = torch.device("cpu")
    model = model.to(device)

    correct = 0
    total = len(labels)
    with torch.no_grad():
        for i in range(total):
            ids = torch.from_numpy(input_ids[i:i+1]).to(device)
            mask = torch.from_numpy(attention_mask[i:i+1]).to(device)
            types = torch.from_numpy(token_type_ids[i:i+1]).to(device)
            out = model(input_ids=ids, attention_mask=mask,
                        token_type_ids=types)
            pred = out.logits.argmax(dim=1).item()
            if pred == labels[i]:
                correct += 1

    accuracy = 100.0 * correct / total
    print(f"  {'PyTorch FP32 baseline':30s} — Accuracy: {accuracy:.2f}% "
          f"({correct}/{total})")

    del model
    gc.collect()
    return {"label": "PyTorch FP32", "accuracy": accuracy,
            "correct": correct, "total": total}


# ============================================================================
# Manning-compliant figure generation
# ============================================================================

# Official Manning color palette (RGB tuples, 0–1 range)
MANNING_COLORS = {
    "blue_L3":   (0/255, 96/255, 177/255),
    "blue_L2":   (107/255, 165/255, 215/255),
    "green_L3":  (128/255, 194/255, 29/255),
    "green_L4":  (10/255, 137/255, 2/255),
    "red_L3":    (211/255, 21/255, 24/255),
    "orange_L3": (227/255, 123/255, 69/255),
    "orange_L2": (255/255, 180/255, 88/255),
    "purple_L3": (119/255, 59/255, 154/255),
    "gray_50":   (128/255, 128/255, 128/255),
    "gray_25":   (191/255, 191/255, 191/255),
    "black":     (0, 0, 0),
}

# Hatch patterns for grayscale differentiation (never rely on color alone)
HATCHES = ["", "///", "\\\\\\", "...", "xx"]


def set_manning_style():
    """Configure matplotlib rcParams for Manning Publications compliance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,       # TrueType fonts in PDF (editable)
        "ps.fonttype": 42,
    })
    return plt


def save_manning_figure(fig, base_name: str, save_dir: Path):
    """Save figure as PNG (300 DPI) + PDF (vector, editable fonts)."""
    png_path = save_dir / f"{base_name}.png"
    pdf_path = save_dir / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"  Figure saved: {png_path}")
    print(f"  Figure saved: {pdf_path}")


def save_ort_quantization_figure(results: list, pytorch_baseline: dict,
                                 model_name: str, save_dir: Path):
    """
    Horizontal bar chart: ONNX model sizes with compression ratios.

    This replaces what would otherwise be a table. Each bar shows the
    ONNX artifact size, annotated with compression ratio and accuracy delta.
    """
    plt = set_manning_style()

    fig, ax = plt.subplots(figsize=(5.5, 2.2))

    # Colors and hatches paired for grayscale safety
    bar_styles = {
        "ONNX FP32":        (MANNING_COLORS["gray_50"],   ""),
        "ONNX dynamic INT8": (MANNING_COLORS["blue_L2"],   "///"),
        "ONNX static INT8":  (MANNING_COLORS["blue_L3"],   "\\\\\\"),
    }

    fp32_size = None
    base_acc = pytorch_baseline["accuracy"]

    # Sort: FP32 at top, then by size descending
    sorted_results = sorted(results, key=lambda r: -r["size_mb"])

    y_positions = list(range(len(sorted_results)))
    bar_height = 0.6

    for i, r in enumerate(sorted_results):
        label = r["label"]
        size = r["size_mb"]
        acc = r["accuracy"]

        if "FP32" in label:
            fp32_size = size

        color, hatch = bar_styles.get(label, (MANNING_COLORS["gray_25"], ""))

        ax.barh(i, size, height=bar_height, color=color, hatch=hatch,
                edgecolor="black", linewidth=0.5, zorder=2)

        # Size label inside or beside bar
        ax.text(size - 0.5, i, f"{size:.1f} MB",
                ha="right", va="center", fontsize=7, fontweight="bold",
                color="white" if size > 15 else "black")

    # Add compression + accuracy annotations on the right
    max_size = max(r["size_mb"] for r in sorted_results)
    for i, r in enumerate(sorted_results):
        size = r["size_mb"]
        acc = r["accuracy"]

        if fp32_size and "FP32" not in r["label"]:
            comp = fp32_size / size
            delta = acc - base_acc
            annotation = f"{comp:.2f}×  ({delta:+.2f} pp)"
        else:
            annotation = f"baseline ({acc:.2f}%)"

        ax.text(max_size + 1.5, i, annotation,
                ha="left", va="center", fontsize=7, style="italic")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in sorted_results])
    ax.set_xlabel("Model size (MB)")
    ax.set_title(f"ORT quantization: {model_name}")
    ax.set_xlim(0, max_size * 1.65)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.invert_yaxis()

    plt.tight_layout()

    tag = model_name.split()[0].lower()
    save_manning_figure(fig, f"CH06_F_ORT_{tag}", save_dir)
    plt.close(fig)


def save_sensitivity_figure(layer_sensitivities: list, conv_nodes: list,
                            mixed_results: list, save_dir: Path):
    """
    Two-panel figure for section 6.4's mixed-precision analysis.

    Panel A: Per-layer sensitivity — horizontal bar chart showing accuracy
    gain from keeping each conv layer in FP32. Highlights the layers that
    matter most.

    Panel B: Precision allocation map — shows 4 strategic configs as
    horizontal strips, each cell a layer, colored INT8 (blue) or FP32
    (green). Accuracy and size annotated on the right.
    """
    plt = set_manning_style()

    fig, (ax_sens, ax_map) = plt.subplots(
        2, 1, figsize=(5.5, 5.5),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    # --- Panel A: Per-layer sensitivity ---
    n_layers = len(layer_sensitivities)
    indices = range(n_layers)
    gains = [s["acc_gain_pp"] for s in layer_sensitivities]

    # Color bars by sensitivity: positive gain = sensitive (worth protecting)
    bar_colors = []
    for g in gains:
        if g > 0.05:
            bar_colors.append(MANNING_COLORS["green_L3"])   # sensitive
        elif g > 0.01:
            bar_colors.append(MANNING_COLORS["orange_L3"])  # moderate
        else:
            bar_colors.append(MANNING_COLORS["gray_25"])    # insensitive

    bars = ax_sens.barh(indices, gains, height=0.7, color=bar_colors,
                         edgecolor="black", linewidth=0.3, zorder=2)

    # Label the most sensitive layers
    for i, s in enumerate(layer_sensitivities):
        if s["acc_gain_pp"] > 0.01:
            ax_sens.text(s["acc_gain_pp"] + 0.002, i,
                         f"{s['acc_gain_pp']:+.3f} pp",
                         va="center", fontsize=6, style="italic")

    ax_sens.set_yticks(list(indices))
    # Short labels: "conv 0", "conv 1", ...
    ylabels = [f"conv {s['index']}" for s in layer_sensitivities]
    ylabels[0] = "conv 0 (input)"
    ylabels[-1] = f"conv {layer_sensitivities[-1]['index']} (last)"
    ax_sens.set_yticklabels(ylabels, fontsize=6)
    ax_sens.set_xlabel("Accuracy gain from keeping layer in FP32 (pp)")
    ax_sens.set_title("Per-layer quantization sensitivity (ResNet-18)")
    ax_sens.axvline(x=0, color="black", linewidth=0.5)
    ax_sens.grid(axis="x", alpha=0.3, zorder=0)
    ax_sens.invert_yaxis()

    # --- Panel B: Precision allocation map ---
    n_total = len(conv_nodes) + 1  # +1 for FC layer
    config_labels = []
    config_accs = []
    config_sizes = []

    # Build the map data: each row is a config, each column is a layer
    map_data = []

    for mr in mixed_results:
        excluded = set(mr.get("exclude_nodes", []))
        row = []
        for node in conv_nodes:
            row.append(0 if node["name"] in excluded else 1)  # 1=INT8, 0=FP32
        # FC layer
        fc_excluded = any("MatMul" in name or "Gemm" in name or "fc" in name.lower()
                          for name in excluded)
        row.append(0 if fc_excluded else 1)
        map_data.append(row)
        config_labels.append(mr["label"])
        config_accs.append(mr["accuracy"])
        config_sizes.append(mr["size_mb"])

    map_array = np.array(map_data)

    # Draw cells as rectangles with hatch for FP32 (grayscale-safe).
    # INT8 = blue_L2, solid. FP32 = orange_L2 with diagonal hatch.
    # These colors have good grayscale separation (~155 vs ~185),
    # and the hatch provides a secondary cue in B&W print.
    import matplotlib.patches as mpatches

    for row_i in range(map_array.shape[0]):
        for col_j in range(map_array.shape[1]):
            val = map_array[row_i, col_j]
            if val == 1:  # INT8
                color = MANNING_COLORS["blue_L2"]
                hatch = ""
            else:         # FP32
                color = MANNING_COLORS["orange_L2"]
                hatch = "///"
            rect = mpatches.FancyBboxPatch(
                (col_j - 0.5, row_i - 0.5), 1, 1,
                boxstyle="square,pad=0",
                facecolor=color, hatch=hatch,
                edgecolor="white", linewidth=0.5)
            ax_map.add_patch(rect)

    ax_map.set_xlim(-0.5, map_array.shape[1] - 0.5)
    ax_map.set_ylim(map_array.shape[0] - 0.5, -0.5)

    # X labels: layer indices
    x_labels = [str(i) for i in range(len(conv_nodes))] + ["FC"]
    ax_map.set_xticks(range(n_total))
    ax_map.set_xticklabels(x_labels, fontsize=5.5)
    ax_map.set_xlabel("Layer index")

    # Y labels: config names
    ax_map.set_yticks(range(len(config_labels)))
    ax_map.set_yticklabels(config_labels, fontsize=6.5)

    # Annotate accuracy and size on the right
    for i, (acc, size) in enumerate(zip(config_accs, config_sizes)):
        ax_map.text(n_total + 0.3, i, f"{acc:.2f}%  {size:.1f} MB",
                    va="center", fontsize=6.5, style="italic")

    ax_map.set_title("Precision allocation (hatched = FP32, solid = INT8)")

    # Legend as text annotation
    ax_map.text(n_total + 0.3, -0.8, "Acc      Size",
                fontsize=6, fontweight="bold", va="center")

    plt.tight_layout()
    save_manning_figure(fig, "CH06_F_MixedPrecision", save_dir)
    plt.close(fig)


def save_cross_framework_figure(resnet_results: list, bert_results: list,
                                save_dir: Path):
    """
    Cross-framework comparison: grouped bar chart showing compression
    ratios from all three section 6 paths (TorchAO, TFLite, ORT).

    Hard-codes the section 6.2 and 6.3 numbers from actual script output
    alongside the ORT numbers from this run.
    """
    plt = set_manning_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.0))

    # --- Panel A: ResNet-18 compression ---
    resnet_data = [
        ("TorchAO\nINT8-WO",     1.03,  MANNING_COLORS["green_L3"],  ""),
        ("ORT\ndynamic",          None,   MANNING_COLORS["blue_L2"],   "///"),
        ("ORT\nstatic",           None,   MANNING_COLORS["blue_L3"],   "\\\\\\"),
    ]

    # Fill in ORT numbers from actual results
    for r in resnet_results:
        fp32 = next((x["size_mb"] for x in resnet_results if "FP32" in x["label"]), None)
        if fp32 and "dynamic" in r["label"].lower():
            resnet_data[1] = ("ORT\ndynamic", fp32 / r["size_mb"],
                              MANNING_COLORS["blue_L2"], "///")
        elif fp32 and "static" in r["label"].lower():
            resnet_data[2] = ("ORT\nstatic", fp32 / r["size_mb"],
                              MANNING_COLORS["blue_L3"], "\\\\\\")

    x_pos = range(len(resnet_data))
    for i, (label, comp, color, hatch) in enumerate(resnet_data):
        if comp is None:
            continue
        bar = ax1.bar(i, comp, width=0.65, color=color, hatch=hatch,
                      edgecolor="black", linewidth=0.5, zorder=2)
        ax1.text(i, comp + 0.08, f"{comp:.2f}×", ha="center", va="bottom",
                 fontsize=7, fontweight="bold")

    ax1.set_xticks(list(x_pos))
    ax1.set_xticklabels([d[0] for d in resnet_data], fontsize=6.5)
    ax1.set_ylabel("Compression ratio")
    ax1.set_title("ResNet-18 on ImageNette")
    ax1.set_ylim(0, 5.0)
    ax1.axhline(y=1.0, color=MANNING_COLORS["gray_50"], linestyle="--",
                linewidth=0.7, zorder=1)
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # --- Panel B: BERT-base compression ---
    bert_data = [
        ("TorchAO\nINT8-WO",     2.40,  MANNING_COLORS["green_L3"],   ""),
        ("TFLite\ndynamic",       3.94,  MANNING_COLORS["orange_L3"],  "..."),
        ("ORT\ndynamic",          None,  MANNING_COLORS["blue_L2"],    "///"),
        ("ORT\nstatic",           None,  MANNING_COLORS["blue_L3"],    "\\\\\\"),
    ]

    # Fill in from actual BERT results
    for r in bert_results:
        fp32 = next((x["size_mb"] for x in bert_results if "FP32" in x["label"]), None)
        if fp32 and "dynamic" in r["label"].lower():
            bert_data[2] = ("ORT\ndynamic", fp32 / r["size_mb"],
                            MANNING_COLORS["blue_L2"], "///")
        elif fp32 and "static" in r["label"].lower():
            bert_data[3] = ("ORT\nstatic", fp32 / r["size_mb"],
                            MANNING_COLORS["blue_L3"], "\\\\\\")

    x_pos2 = range(len(bert_data))
    for i, (label, comp, color, hatch) in enumerate(bert_data):
        if comp is None:
            continue
        bar = ax2.bar(i, comp, width=0.65, color=color, hatch=hatch,
                      edgecolor="black", linewidth=0.5, zorder=2)
        ax2.text(i, comp + 0.08, f"{comp:.2f}×", ha="center", va="bottom",
                 fontsize=7, fontweight="bold")

    ax2.set_xticks(list(x_pos2))
    ax2.set_xticklabels([d[0] for d in bert_data], fontsize=6.5)
    ax2.set_ylabel("Compression ratio")
    ax2.set_title("BERT-base on SST-2")
    ax2.set_ylim(0, 5.5)
    ax2.axhline(y=1.0, color=MANNING_COLORS["gray_50"], linestyle="--",
                linewidth=0.7, zorder=1)
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    save_manning_figure(fig, "CH06_F_CrossFramework", save_dir)
    plt.close(fig)


# ============================================================================
# Summary printer
# ============================================================================

def print_summary(model_name: str, results: list, pytorch_baseline: dict):
    """Print a compact summary comparing all variants for one model."""
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*70}")

    base_acc = pytorch_baseline["accuracy"]

    # Find FP32 ONNX result for size baseline
    fp32_result = next((r for r in results if "FP32" in r["label"]), None)
    base_size = fp32_result["size_mb"] if fp32_result else None

    print(f"  PyTorch FP32 accuracy: {base_acc:.2f}%")
    if base_size:
        print(f"  ONNX FP32 size:        {base_size:.2f} MB")
    print()

    fmt = "  {:<30s}  {:>8s}  {:>6s}  {:>10s}  {:>10s}"
    print(fmt.format("Variant", "Size(MB)", "Comp.", "Accuracy", "Δ vs FP32"))
    print(f"  {'-'*68}")

    for r in results:
        size = r["size_mb"]
        acc = r["accuracy"]
        delta = acc - base_acc

        if base_size and base_size > 0:
            comp = f"{base_size / size:.2f}×"
        else:
            comp = "—"

        delta_str = f"{delta:+.2f} pp" if "FP32" not in r["label"] else "—"

        print(f"  {r['label']:<30s}  {size:8.2f}  {comp:>6s}  "
              f"{acc:9.2f}%  {delta_str:>10s}")


# ============================================================================
# Runner functions per model
# ============================================================================

def run_resnet(modes: set, save_dir: Path, save_plots: bool):
    """Run the full ResNet-18 ONNX pipeline."""
    print(f"\n{'#'*70}")
    print("# ResNet-18 on ImageNette — ONNX Runtime path")
    print(f"{'#'*70}")

    resnet_results = []

    # Always need the FP32 ONNX for quantization
    fp32_path = str(save_dir / "resnet18_fp32.onnx")
    if "export" in modes or not os.path.exists(fp32_path):
        fp32_path = export_resnet_onnx(save_dir)

    dynamic_path = str(save_dir / "resnet18_dynamic_int8.onnx")
    static_path = str(save_dir / "resnet18_static_int8.onnx")

    if "dynamic" in modes:
        dynamic_path = quantize_resnet_dynamic(fp32_path, save_dir)

    if "static" in modes:
        static_path = quantize_resnet_static(fp32_path, save_dir)

    # Mixed precision analysis (ResNet only — this is where layer
    # selection matters most, since ORT quantizes all Conv layers)
    mixed_results = None
    layer_sensitivities = None
    conv_nodes = None
    if "mixed" in modes:
        mixed_results, layer_sensitivities, conv_nodes = (
            run_mixed_precision_analysis(fp32_path, save_dir)
        )

    if "evaluate" in modes:
        print(f"\n{'='*70}")
        print("EVALUATE: ResNet-18 — all ONNX variants")
        print(f"{'='*70}")

        images, labels = load_imagenette_val()

        # PyTorch FP32 baseline
        pt_baseline = evaluate_resnet_pytorch(images, labels)

        # ONNX FP32
        if os.path.exists(fp32_path):
            resnet_results.append(
                evaluate_resnet_onnx(fp32_path, images, labels, "ONNX FP32"))

        # ONNX dynamic INT8
        if os.path.exists(dynamic_path):
            resnet_results.append(
                evaluate_resnet_onnx(dynamic_path, images, labels,
                                    "ONNX dynamic INT8"))

        # ONNX static INT8
        if os.path.exists(static_path):
            resnet_results.append(
                evaluate_resnet_onnx(static_path, images, labels,
                                    "ONNX static INT8"))

        print_summary("ResNet-18 on ImageNette", resnet_results, pt_baseline)

        if save_plots and resnet_results:
            fig_dir = save_dir / "figures"
            fig_dir.mkdir(exist_ok=True)
            save_ort_quantization_figure(
                resnet_results, pt_baseline,
                "ResNet-18 on ImageNette", fig_dir)

            if mixed_results and layer_sensitivities and conv_nodes:
                save_sensitivity_figure(
                    layer_sensitivities, conv_nodes,
                    mixed_results, fig_dir)

        return resnet_results, pt_baseline

    # Generate mixed-precision figure even without evaluate mode
    if save_plots and mixed_results and layer_sensitivities and conv_nodes:
        fig_dir = save_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        save_sensitivity_figure(
            layer_sensitivities, conv_nodes,
            mixed_results, fig_dir)

    return resnet_results, None


def run_bert(modes: set, save_dir: Path, save_plots: bool):
    """Run the full BERT-base ONNX pipeline."""
    print(f"\n{'#'*70}")
    print("# BERT-base on SST-2 — ONNX Runtime path")
    print(f"{'#'*70}")

    bert_results = []

    # Always need the FP32 ONNX
    fp32_path = str(save_dir / "bert_sst2_fp32.onnx")
    if "export" in modes or not os.path.exists(fp32_path):
        fp32_path = export_bert_onnx(save_dir)

    dynamic_path = str(save_dir / "bert_sst2_dynamic_int8.onnx")
    static_path = str(save_dir / "bert_sst2_static_int8.onnx")

    if "dynamic" in modes:
        dynamic_path = quantize_bert_dynamic(fp32_path, save_dir)

    if "static" in modes:
        result = quantize_bert_static(fp32_path, save_dir)
        if result is not None:
            static_path = result

    if "evaluate" in modes:
        print(f"\n{'='*70}")
        print("EVALUATE: BERT-base — all ONNX variants")
        print(f"{'='*70}")

        input_ids, attention_mask, token_type_ids, labels = load_sst2_val()

        # PyTorch FP32 baseline
        pt_baseline = evaluate_bert_pytorch(
            input_ids, attention_mask, token_type_ids, labels)

        # ONNX FP32
        if os.path.exists(fp32_path):
            bert_results.append(
                evaluate_bert_onnx(fp32_path, input_ids, attention_mask,
                                   token_type_ids, labels, "ONNX FP32"))

        # ONNX dynamic INT8
        if os.path.exists(dynamic_path):
            bert_results.append(
                evaluate_bert_onnx(dynamic_path, input_ids, attention_mask,
                                   token_type_ids, labels, "ONNX dynamic INT8"))

        # ONNX static INT8
        if os.path.exists(static_path):
            bert_results.append(
                evaluate_bert_onnx(static_path, input_ids, attention_mask,
                                   token_type_ids, labels, "ONNX static INT8"))

        print_summary("BERT-base on SST-2", bert_results, pt_baseline)

        if save_plots and bert_results:
            fig_dir = save_dir / "figures"
            fig_dir.mkdir(exist_ok=True)
            save_ort_quantization_figure(
                bert_results, pt_baseline,
                "BERT-base on SST-2", fig_dir)

        return bert_results, pt_baseline

    return bert_results, None


# ============================================================================
# Cross-framework summary
# ============================================================================

def print_cross_framework_summary():
    """
    Print a cross-framework comparison referencing results from
    sections 6.2 (TorchAO) and 6.3 (TFLite).

    This is a template — fill in actual numbers after running all three
    scripts. Numbers marked [PLACEHOLDER] must come from real runs.
    """
    print(f"\n{'='*70}")
    print("CROSS-FRAMEWORK COMPARISON (sections 6.2 / 6.3 / 6.4)")
    print(f"{'='*70}")
    print()
    print("  Fill in from actual runs of all three companion scripts:")
    print()

    fmt = "  {:<12s}  {:<25s}  {:>10s}  {:>10s}  {:>10s}"
    print(fmt.format("Model", "Framework / Config", "Size(MB)", "Comp.", "Accuracy"))
    print(f"  {'-'*72}")
    print()
    print("  ResNet-18 (ImageNette)")
    print("  ─────────────────────")
    print("  TorchAO  INT8-WO:     [from 6.2 — ~1.03× on Linear only]")
    print("  ORT      dynamic INT8: [from this script]")
    print("  ORT      static INT8:  [from this script]")
    print()
    print("  BERT-base (SST-2)")
    print("  ─────────────────")
    print("  TorchAO  INT8-WO:     [from 6.2 — 174.05 MB, 2.40×, 92.32%]")
    print("  TorchAO  W8A8 dyn:    [from 6.2 — 173.40 MB, 2.41×, 92.20%]")
    print("  TFLite   dynamic:     [from 6.3 — 105.85 MB, 3.94×, 92.43%]")
    print("  ORT      dynamic INT8: [from this script]")
    print("  ORT      static INT8:  [from this script]")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ch6 §6.4 — Export to ONNX and drive ORT quantization"
    )
    parser.add_argument(
        "--model", choices=["resnet", "bert"], default=None,
        help="Which model to run (default: use --all for both)"
    )
    parser.add_argument(
        "--mode", choices=["export", "dynamic", "static", "mixed",
                          "evaluate", "all"],
        default="all",
        help="Which pipeline stage to run"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run both models through all modes"
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save comparison figures to artifact directory"
    )
    args = parser.parse_args()

    # Determine what to run
    if args.all:
        models = ["resnet", "bert"]
        modes = {"export", "dynamic", "static", "mixed", "evaluate"}
    elif args.model:
        models = [args.model]
        modes = {"export", "dynamic", "static", "mixed", "evaluate"} if args.mode == "all" else {args.mode}
        # Evaluate implies we need all artifacts
        if "evaluate" in modes and len(modes) == 1:
            modes = {"export", "dynamic", "static", "evaluate"}
    else:
        parser.print_help()
        print("\nExample: python ch6_onnx_export_path.py --all")
        return

    # Create artifact directory
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts directory: {ARTIFACT_DIR}")

    all_resnet_results, all_bert_results = [], []
    resnet_baseline, bert_baseline = None, None

    if "resnet" in models:
        resnet_results, resnet_baseline = run_resnet(
            modes, ARTIFACT_DIR, args.save_plots)
        all_resnet_results = resnet_results

    if "bert" in models:
        bert_results, bert_baseline = run_bert(
            modes, ARTIFACT_DIR, args.save_plots)
        all_bert_results = bert_results

    # Cross-framework comparison figure (uses hard-coded 6.2/6.3 numbers + this run)
    if args.save_plots and (all_resnet_results or all_bert_results):
        fig_dir = ARTIFACT_DIR / "figures"
        fig_dir.mkdir(exist_ok=True)
        save_cross_framework_figure(
            all_resnet_results, all_bert_results, fig_dir)

    # Cross-framework reference (console output)
    if "evaluate" in modes:
        print_cross_framework_summary()

    print(f"\n{'='*70}")
    print("Done. All ONNX artifacts saved to:", ARTIFACT_DIR)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()