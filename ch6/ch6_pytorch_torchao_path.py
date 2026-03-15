#!/usr/bin/env python3
"""
Chapter 6.2 Companion Script: Run the PyTorch and TorchAO Path End-to-End

This script demonstrates the two PyTorch-native quantization APIs that
practitioners actually use in production:

  1. TorchAO quantize_()  — In-place weight replacement via tensor subclasses.
     One-liner quantization for transformer models where nn.Linear dominates
     parameter count. We apply INT8 weight-only and W8A8 dynamic configs to
     BERT-base on SST-2, and show *why* TorchAO gives minimal compression
     on CNNs (it only targets nn.Linear by default, and Conv2d weights are
     not supported by the stable configs).

  2. PT2E (PyTorch 2 Export Quantization) — Graph-level quantization that
     understands Conv2d, BatchNorm folding, and operator fusion patterns.
     The correct path for CNN models. We apply static INT8 quantization to
     ResNet-18 on ImageNette with calibration.

Models and datasets:
  - ResNet-18 on ImageNette (10-class ImageNet subset, auto-downloads)
  - BERT-base-uncased on SST-2 (binary sentiment, auto-downloads)

Usage:
    python ch6_pytorch_torchao_path.py --all --save-plots
    python ch6_pytorch_torchao_path.py --bert
    python ch6_pytorch_torchao_path.py --resnet
    python ch6_pytorch_torchao_path.py --all --num-samples 200   # quick dev

Requirements:
    pip install torch torchvision torchao transformers datasets matplotlib tqdm
"""

import argparse
import copy
import gc
import json
import os
import sys
import tempfile
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageFolder
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from datasets import load_dataset
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import torchao
    from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig,
        Int8WeightOnlyConfig,
        quantize_,
    )
    HAS_TORCHAO = True
    TORCHAO_VERSION = getattr(torchao, "__version__", "unknown")
except ImportError:
    HAS_TORCHAO = False
    TORCHAO_VERSION = None

# PT2E imports — these live in torch.ao or torchao depending on version
HAS_PT2E = False
try:
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torch.ao.quantization.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    HAS_PT2E = True
except ImportError:
    try:
        from torchao.quantization.pt2e.quantize_pt2e import (
            prepare_pt2e, convert_pt2e,
        )
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        HAS_PT2E = True
    except ImportError:
        pass

if not HAS_PT2E:
    print("[warn] PT2E quantization not available. CNN quantization disabled.")


# ============================================================================
# Manning color palette (grayscale-safe)
# ============================================================================
COLORS = {
    "blue_l1": (197/255, 223/255, 239/255),
    "blue_l2": (107/255, 165/255, 215/255),
    "blue_l3": (0/255, 96/255, 177/255),
    "blue_l4": (0/255, 45/255, 139/255),
    "green_l2": (194/255, 227/255, 115/255),
    "green_l3": (128/255, 194/255, 29/255),
    "red_l2": (244/255, 110/255, 96/255),
    "red_l3": (211/255, 21/255, 24/255),
    "orange_l2": (255/255, 180/255, 88/255),
    "black_l3": (75/255, 75/255, 75/255),
}
HATCH_PATTERNS = ["", "///", "xxx", "\\\\\\", "...", "ooo"]


# ============================================================================
# Data classes
# ============================================================================
@dataclass
class QuantizationResult:
    model_name: str
    config_name: str
    config_desc: str
    quant_api: str                # "torchao" or "pt2e"
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    original_accuracy: float
    quantized_accuracy: float
    accuracy_delta: float
    original_latency_ms: float = 0.0
    quantized_latency_ms: float = 0.0
    speedup: float = 0.0
    num_quantized_layers: int = 0
    num_total_layers: int = 0
    device: str = ""


# ============================================================================
# Data loading
# ============================================================================

IMAGENETTE_TO_IMAGENET = {
    0: 0, 1: 217, 2: 482, 3: 491, 4: 497,
    5: 566, 6: 569, 7: 571, 8: 574, 9: 701,
}


class ImageNetteWrapper(nn.Module):
    """Maps a 1000-class ImageNet model to 10 ImageNette classes."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.register_buffer(
            "class_indices",
            torch.tensor([IMAGENETTE_TO_IMAGENET[i] for i in range(10)]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)
        return logits[:, self.class_indices]


def get_imagenette_loader(
    split="val", batch_size=64, num_samples=None, num_workers=4, data_dir="./data",
):
    imagenette_dir = Path(data_dir) / "imagenette2"
    split_dir = imagenette_dir / split

    if not split_dir.exists():
        print(f"Downloading ImageNette to {data_dir}/ ...")
        import urllib.request, tarfile
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
        tgz_path = Path(data_dir) / "imagenette2.tgz"
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(url, tgz_path)
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(data_dir)
        tgz_path.unlink()
        print("Download complete.")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(str(split_dir), transform=transform)
    if num_samples and num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:num_samples].tolist()
        dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    return loader, 10


def get_sst2_loader(
    tokenizer, batch_size=32, num_samples=None, max_length=128, split="validation",
):
    dataset = load_dataset("glue", "sst2", split=split)
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], padding="max_length",
                         truncation=True, max_length=max_length,
                         return_tensors="pt")

    tokenized = dataset.map(tokenize_fn, batched=True,
                            remove_columns=["sentence", "idx"])
    tokenized.set_format("torch")
    return DataLoader(tokenized, batch_size=batch_size, shuffle=False), 2


# ============================================================================
# Measurement utilities
# ============================================================================

def measure_model_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / (1024 * 1024)


def evaluate_vision(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="    Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def evaluate_nlp(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="    Eval", leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            preds = model(input_ids=ids, attention_mask=mask).logits.argmax(-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def benchmark_latency(model, example_input, warmup=10, iterations=50,
                      device=torch.device("cpu")):
    model.eval()
    use_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            if isinstance(example_input, dict):
                model(**example_input)
            else:
                model(example_input)
            if use_cuda:
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if use_cuda:
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                if isinstance(example_input, dict):
                    model(**example_input)
                else:
                    model(example_input)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                if isinstance(example_input, dict):
                    model(**example_input)
                else:
                    model(example_input)
                times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return float(np.median(arr)), float(np.std(arr))


def print_arch_breakdown(model, model_name):
    """Show parameter distribution across layer types."""
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    n_lin = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    p_conv = sum(p.numel() for m in model.modules()
                 if isinstance(m, nn.Conv2d) for p in m.parameters())
    p_lin = sum(p.numel() for m in model.modules()
                if isinstance(m, nn.Linear) for p in m.parameters())
    total = sum(p.numel() for p in model.parameters())

    print(f"\n  Architecture breakdown ({model_name}):")
    print(f"    Conv2d layers:  {n_conv:>3d}  ({p_conv:,} params, "
          f"{100*p_conv/total:.1f}%)")
    print(f"    Linear layers:  {n_lin:>3d}  ({p_lin:,} params, "
          f"{100*p_lin/total:.1f}%)")
    print(f"    Total params:   {total:,}")
    return n_conv, n_lin, p_conv, p_lin, total


# ============================================================================
# TorchAO weight inspection
# ============================================================================

def inspect_torchao_weights(model, config_name):
    """
    Listing 6.1: Inspect weight tensors after TorchAO quantize_()
    """
    shown = 0
    n_quantized = 0
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight
        w_type = type(w).__name__
        is_q = (hasattr(w, "tensor_impl") or "Quantized" in w_type
                or "Activation" in w_type)

        if is_q:
            n_quantized += 1
            if shown < 3:
                extra = ""
                if hasattr(w, "block_size"):
                    extra += f", block_size={w.block_size}"
                if hasattr(w, "tensor_impl") and hasattr(w.tensor_impl, "int_data"):
                    try:
                        idata = w.tensor_impl.int_data
                        if callable(idata): idata = idata()
                        extra += f", int_dtype={idata.dtype}"
                    except Exception:
                        pass
                print(f"    {name}: {w_type}, shape={list(w.shape)}{extra}")
                shown += 1

    remaining = n_quantized - shown
    if remaining > 0:
        print(f"    ... and {remaining} more quantized layers")
    print(f"    → {n_quantized}/{n_linear} nn.Linear layers quantized")
    return n_quantized, n_linear


# ============================================================================
# PART 1: TorchAO path — Listing 6.2
# ============================================================================

def run_torchao_experiment(
    model_loader_fn, evaluate_fn, model_name, device, loader,
    example_input, do_benchmark=True, do_compile=False,
):
    """
    Listing 6.2: TorchAO quantize_() end-to-end

    Applies INT8-WO and W8A8-Dynamic to a model, measures accuracy,
    size, and latency. Optionally benchmarks with torch.compile to
    show the eager-mode overhead vs compiled speedup.
    """
    results = []
    configs = [
        ("INT8-WO", "INT8 weight-only (per-channel symmetric)",
         Int8WeightOnlyConfig()),
        ("W8A8-Dynamic", "INT8 weight + INT8 dynamic activation (per-token)",
         Int8DynamicActivationInt8WeightConfig()),
    ]

    # --- Baseline ---
    print(f"\n  BASELINE ({model_name}, original precision)")
    baseline = model_loader_fn().to(device).eval()
    base_size = measure_model_size_mb(baseline)
    base_acc = evaluate_fn(baseline, loader, device)
    base_lat, base_std = 0.0, 0.0
    if do_benchmark:
        base_lat, base_std = benchmark_latency(baseline, example_input,
                                               device=device)
    lat_str = f" | Latency: {base_lat:.2f} ms (±{base_std:.2f})" if do_benchmark else ""
    print(f"    Size: {base_size:.2f} MB | Accuracy: {base_acc:.2f}%{lat_str}")
    del baseline; gc.collect()

    # --- Quantized variants ---
    for cfg_name, cfg_desc, config in configs:
        print(f"\n  {model_name} → {cfg_name}")
        print(f"    Config: {cfg_desc}")

        q_model = model_loader_fn().to(device).eval()
        try:
            quantize_(q_model, config)                             #A
        except Exception as e:
            print(f"    [skip] quantize_() failed: {e}")
            del q_model; gc.collect()
            continue
        # A One-line in-place quantization — replaces weight tensors

        q_size = measure_model_size_mb(q_model)
        compression = base_size / q_size if q_size > 0 else 0
        n_q, n_total = inspect_torchao_weights(q_model, cfg_name)

        q_acc = evaluate_fn(q_model, loader, device)
        acc_delta = q_acc - base_acc

        q_lat, speedup = 0.0, 0.0
        compiled_lat, compiled_speedup = 0.0, 0.0
        if do_benchmark:
            q_lat, q_std = benchmark_latency(q_model, example_input,
                                             device=device)
            speedup = base_lat / q_lat if q_lat > 0 else 0
            print(f"    Size: {q_size:.2f} MB ({compression:.2f}x) | "
                  f"Acc: {q_acc:.2f}% (Δ{acc_delta:+.2f}%)")
            print(f"    Latency (eager):    {q_lat:.2f} ms "
                  f"({speedup:.2f}x vs FP32)")

            # torch.compile: the missing piece for latency gains
            if do_compile:
                try:
                    compiled = torch.compile(q_model, mode="max-autotune") #G
                    # Extra warmup for compilation overhead
                    compiled_lat, c_std = benchmark_latency(
                        compiled, example_input, warmup=20,
                        iterations=50, device=device)
                    compiled_speedup = (base_lat / compiled_lat
                                        if compiled_lat > 0 else 0)
                    print(f"    Latency (compiled): {compiled_lat:.2f} ms "
                          f"({compiled_speedup:.2f}x vs FP32)")
                    # G torch.compile fuses the quantized matmul kernels.
                    #   Without it, tensor subclass dispatch adds Python overhead
                    #   that makes eager-mode quantized inference *slower* than FP32.
                except Exception as e:
                    print(f"    [skip] torch.compile failed: {e}")
        else:
            print(f"    Size: {q_size:.2f} MB ({compression:.2f}x) | "
                  f"Acc: {q_acc:.2f}% (Δ{acc_delta:+.2f}%)")

        results.append(QuantizationResult(
            model_name=model_name, config_name=cfg_name,
            config_desc=cfg_desc, quant_api="torchao",
            original_size_mb=base_size, quantized_size_mb=q_size,
            compression_ratio=compression,
            original_accuracy=base_acc, quantized_accuracy=q_acc,
            accuracy_delta=acc_delta,
            original_latency_ms=base_lat, quantized_latency_ms=q_lat,
            speedup=speedup,
            num_quantized_layers=n_q, num_total_layers=n_total,
            device=str(device),
        ))
        del q_model; gc.collect()

    return results


# ============================================================================
# PART 2: PT2E path (CNNs) — Listing 6.3
# ============================================================================

def run_pt2e_experiment(
    model_loader_fn, evaluate_fn, model_name, device, loader,
    example_input, num_calib_batches=10, do_benchmark=True,
):
    """
    Listing 6.3: PT2E static quantization for CNNs

    Four-step pipeline:
      1. torch.export.export_for_training() — capture the graph
      2. prepare_pt2e() — insert observers, fold BatchNorm into Conv2d
      3. Calibrate — run representative data through the observed graph
      4. convert_pt2e() — replace observers with quantized integer ops
    """
    results = []

    # --- Baseline ---
    print(f"\n  BASELINE ({model_name}, original precision)")
    baseline = model_loader_fn().to(device).eval()
    base_size = measure_model_size_mb(baseline)
    base_acc = evaluate_fn(baseline, loader, device)
    base_lat, base_std = 0.0, 0.0
    if do_benchmark:
        base_lat, base_std = benchmark_latency(
            baseline, example_input.to(device), device=device)
    lat_str = f" | Latency: {base_lat:.2f} ms (±{base_std:.2f})" if do_benchmark else ""
    print(f"    Size: {base_size:.2f} MB | Accuracy: {base_acc:.2f}%{lat_str}")
    del baseline; gc.collect()

    # --- PT2E static INT8 ---
    cfg_name = "PT2E-INT8-Static"
    cfg_desc = ("Static INT8 via XNNPACK quantizer — per-channel weights, "
                "per-tensor activations, Conv-BN folded")
    print(f"\n  {model_name} → {cfg_name}")
    print(f"    Config: {cfg_desc}")

    float_model = model_loader_fn().to(device).eval()

    # Step 1: Export
    print("    Step 1/4: Exporting to ATen IR ...")
    example_inputs = (example_input.to(device),)
    try:
        exported = torch.export.export_for_training(               #B
            float_model, example_inputs
        ).module()
    except Exception:
        try:
            from torch._export import capture_pre_autograd_graph
            exported = capture_pre_autograd_graph(float_model, example_inputs)
        except Exception as e:
            print(f"    [skip] Export failed: {e}")
            del float_model; gc.collect()
            return results
    # B export_for_training preserves BN statistics for folding

    # Step 2: Prepare
    print("    Step 2/4: Inserting observers + Conv-BN folding ...")
    quantizer = XNNPACKQuantizer().set_global(                     #C
        get_symmetric_quantization_config(is_per_channel=True)
    )
    prepared = prepare_pt2e(exported, quantizer)
    # C XNNPACK quantizer: symmetric INT8, per-channel weights, per-tensor acts

    # Step 3: Calibrate
    print(f"    Step 3/4: Calibrating ({num_calib_batches} batches) ...")
    # Exported models override .eval() — use the PT2E-safe variant
    try:
        torch.ao.quantization.allow_exported_model_train_eval(prepared)
        prepared.eval()
    except (AttributeError, TypeError):
        try:
            from torch.ao.quantization import move_exported_model_to_eval
            move_exported_model_to_eval(prepared)
        except ImportError:
            pass  # Already in eval-like state from export
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_calib_batches:
                break
            prepared(images.to(device))

    # Step 4: Convert
    print("    Step 4/4: Converting to quantized model ...")
    quantized = convert_pt2e(prepared)                             #D
    # D Observers replaced with quantize/dequantize ops

    # Exported models override .eval()/.train(). Re-enable them so
    # downstream evaluate/benchmark functions work without changes.
    try:
        torch.ao.quantization.allow_exported_model_train_eval(quantized)
    except (AttributeError, TypeError):
        pass
    try:
        from torch.ao.quantization import move_exported_model_to_eval
        move_exported_model_to_eval(quantized)
    except (ImportError, AttributeError):
        try:
            from torchao.quantization.pt2e import move_exported_model_to_eval
            move_exported_model_to_eval(quantized)
        except (ImportError, AttributeError):
            pass

    # Measure size
    q_size = measure_model_size_mb(quantized)
    compression = base_size / q_size if q_size > 0 else 0

    # Count quantized ops in graph
    n_dq = 0
    if hasattr(quantized, "graph"):
        for node in quantized.graph.nodes:
            if node.op == "call_function" and "dequantize" in str(node.target):
                n_dq += 1
    print(f"    Quantized graph: {n_dq} dequantize nodes inserted")

    # Evaluate
    print("    Evaluating ...")
    q_acc = evaluate_fn(quantized, loader, device)
    acc_delta = q_acc - base_acc

    q_lat, speedup = 0.0, 0.0
    if do_benchmark:
        q_lat, q_std = benchmark_latency(
            quantized, example_input.to(device), device=device)
        speedup = base_lat / q_lat if q_lat > 0 else 0

    lat_str = (f" | Latency: {q_lat:.2f} ms ({speedup:.2f}x vs FP32)"
               if do_benchmark else "")
    print(f"    Size: {q_size:.2f} MB ({compression:.2f}x) | "
          f"Acc: {q_acc:.2f}% (Δ{acc_delta:+.2f}%){lat_str}")

    results.append(QuantizationResult(
        model_name=model_name, config_name=cfg_name,
        config_desc=cfg_desc, quant_api="pt2e",
        original_size_mb=base_size, quantized_size_mb=q_size,
        compression_ratio=compression,
        original_accuracy=base_acc, quantized_accuracy=q_acc,
        accuracy_delta=acc_delta,
        original_latency_ms=base_lat, quantized_latency_ms=q_lat,
        speedup=speedup,
        num_quantized_layers=n_dq, num_total_layers=0,
        device=str(device),
    ))
    del quantized, prepared, exported, float_model; gc.collect()
    return results


# ============================================================================
# Save/load round-trip — Listing 6.4
# ============================================================================

def verify_save_load_roundtrip(
    model_loader_fn, evaluate_fn, model_name, device, loader,
    save_dir="./ch6_artifacts",
):
    """
    Listing 6.4: Serialize and reload a TorchAO-quantized model
    """
    print(f"\n  SAVE/LOAD ROUND-TRIP ({model_name})")
    os.makedirs(save_dir, exist_ok=True)

    model = model_loader_fn().to(device).eval()
    quantize_(model, Int8WeightOnlyConfig())
    acc_before = evaluate_fn(model, loader, device)
    size_before = measure_model_size_mb(model)
    print(f"    Before: accuracy={acc_before:.2f}%, size={size_before:.2f} MB")

    save_path = os.path.join(save_dir, f"{model_name}_int8wo.pt")
    torch.save(model.state_dict(), save_path)                      #E
    file_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"    Saved state_dict → {save_path} ({file_mb:.2f} MB)")
    # E state_dict preserves AffineQuantizedTensor subclass metadata

    loaded = model_loader_fn().to(device).eval()
    quantize_(loaded, Int8WeightOnlyConfig())                      #F
    loaded.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=False))
    # F Rebuild quantized skeleton first, then fill with saved weights

    acc_after = evaluate_fn(loaded, loader, device)
    match = abs(acc_before - acc_after) < 1e-6
    print(f"    After:  accuracy={acc_after:.2f}%")
    print(f"    Round-trip: {'PASS' if match else 'FAIL'}")

    del model, loaded; gc.collect()
    return match


# ============================================================================
# Summary table and plot
# ============================================================================

def print_summary_table(results):
    print(f"\n{'='*105}")
    print("SUMMARY: PyTorch Quantization Results")
    print(f"{'='*105}")
    print(f"{'Model':<12} {'API':<8} {'Config':<20} {'Size':>8} "
          f"{'Ratio':>7} {'Acc':>7} {'Δ Acc':>8} "
          f"{'Latency':>10} {'Speedup':>8}")
    print("-" * 105)

    current = None
    for r in results:
        if r.model_name != current:
            current = r.model_name
            lat_s = (f"{r.original_latency_ms:.2f}ms"
                     if r.original_latency_ms > 0 else "—")
            print(f"{r.model_name:<12} {'—':<8} {'FP32 (baseline)':<20} "
                  f"{r.original_size_mb:>7.1f}M {'1.00x':>7} "
                  f"{r.original_accuracy:>6.2f}% {'—':>8} "
                  f"{lat_s:>10} {'1.00x':>8}")

        lat_s = (f"{r.quantized_latency_ms:.2f}ms"
                 if r.quantized_latency_ms > 0 else "—")
        spd_s = f"{r.speedup:.2f}x" if r.speedup > 0 else "—"
        print(f"{'':.<12} {r.quant_api:<8} {r.config_name:<20} "
              f"{r.quantized_size_mb:>7.1f}M {r.compression_ratio:>6.2f}x "
              f"{r.quantized_accuracy:>6.2f}% {r.accuracy_delta:>+7.2f}% "
              f"{lat_s:>10} {spd_s:>8}")

    print("-" * 105)


def plot_comparison(results, save_path=None):
    if not HAS_MATPLOTLIB or not results:
        return

    models = OrderedDict()
    for r in results:
        models.setdefault(r.model_name, []).append(r)

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 3, figsize=(14, 4.5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    bar_colors = [COLORS["blue_l2"], COLORS["green_l3"], COLORS["red_l2"],
                  COLORS["orange_l2"]]
    hatches = HATCH_PATTERNS[:4]

    for row, (mname, mresults) in enumerate(models.items()):
        configs = [r.config_name for r in mresults]
        n = len(configs)
        x = np.arange(n)

        # Size
        ax = axes[row, 0]
        base_s = mresults[0].original_size_mb
        sizes = [r.quantized_size_mb for r in mresults]
        bars = ax.bar(x, sizes, color=bar_colors[:n], edgecolor="black", lw=0.8)
        for i, b in enumerate(bars): b.set_hatch(hatches[i])
        ax.axhline(base_s, color=COLORS["black_l3"], ls="--", lw=1.2,
                    label=f"FP32 ({base_s:.1f} MB)")
        ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=8, rotation=15)
        ax.set_ylabel("Size (MB)")
        ax.set_title(f"{mname} — Size", fontweight="bold")
        ax.legend(fontsize=8)
        for i, v in enumerate(sizes):
            ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)

        # Accuracy
        ax = axes[row, 1]
        base_a = mresults[0].original_accuracy
        accs = [r.quantized_accuracy for r in mresults]
        bars = ax.bar(x, accs, color=bar_colors[:n], edgecolor="black", lw=0.8)
        for i, b in enumerate(bars): b.set_hatch(hatches[i])
        ax.axhline(base_a, color=COLORS["black_l3"], ls="--", lw=1.2,
                    label=f"FP32 ({base_a:.1f}%)")
        ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=8, rotation=15)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{mname} — Accuracy", fontweight="bold")
        ax.legend(fontsize=8)
        mn = min(accs + [base_a])
        ax.set_ylim(max(0, mn - 5), 100)
        for i, v in enumerate(accs):
            ax.text(i, v + 0.3, f"{v-base_a:+.2f}%", ha="center", fontsize=8)

        # Compression
        ax = axes[row, 2]
        ratios = [r.compression_ratio for r in mresults]
        bars = ax.bar(x, ratios, color=bar_colors[:n], edgecolor="black", lw=0.8)
        for i, b in enumerate(bars): b.set_hatch(hatches[i])
        ax.axhline(1.0, color=COLORS["black_l3"], ls="--", lw=1.2)
        ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=8, rotation=15)
        ax.set_ylabel("Compression (×)")
        ax.set_title(f"{mname} — Compression", fontweight="bold")
        for i, v in enumerate(ratios):
            ax.text(i, v + 0.05, f"{v:.2f}×", ha="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"  Plot saved: {save_path}")
    plt.close()


# ============================================================================
# Experiment runners
# ============================================================================

def run_resnet_experiments(device, num_samples=None, do_benchmark=True,
                           do_compile=False):
    """
    ResNet-18 on ImageNette:
      Part A — TorchAO (shows the nn.Linear-only limitation)
      Part B — PT2E static quantization (proper CNN path)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: ResNet-18 on ImageNette")
    print("=" * 70)

    loader, _ = get_imagenette_loader(split="val", num_samples=num_samples)
    print(f"  Dataset: ImageNette val ({len(loader.dataset)} samples)")
    print(f"  Device:  {device}")

    def wrapped_model_loader():
        base = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        return ImageNetteWrapper(base)

    def raw_model_loader():
        return torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

    probe = wrapped_model_loader()
    print_arch_breakdown(probe, "ResNet-18")
    del probe

    example_input = torch.randn(1, 3, 224, 224, device=device)
    all_results = []

    # --- Part A: TorchAO (the lesson) ---
    if HAS_TORCHAO:
        print(f"\n{'—'*60}")
        print("  PART A: TorchAO quantize_() on a CNN")
        print("  TorchAO's stable configs target nn.Linear only.")
        print("  ResNet-18 is 95.5% Conv2d → expect minimal compression.")
        print(f"{'—'*60}")
        torchao_results = run_torchao_experiment(
            wrapped_model_loader, evaluate_vision, "ResNet-18 (TorchAO)",
            device, loader, example_input, do_benchmark, do_compile,
        )
        all_results.extend(torchao_results)

    # --- Part B: PT2E (the fix) ---
    if HAS_PT2E:
        print(f"\n{'—'*60}")
        print("  PART B: PT2E export quantization on a CNN")
        print("  PT2E captures the full graph: Conv2d, BN folding, fusion.")
        print("  This is the correct PyTorch path for CNN quantization.")
        print(f"{'—'*60}")

        # PT2E evaluator: maps 1000-class output → 10 ImageNette classes
        nette_indices = torch.tensor(
            [IMAGENETTE_TO_IMAGENET[i] for i in range(10)])

        def eval_pt2e_vision(model, loader, device):
            model.eval()
            idx = nette_indices.to(device)
            correct = total = 0
            with torch.no_grad():
                for images, labels in tqdm(loader, desc="    Eval", leave=False):
                    images, labels = images.to(device), labels.to(device)
                    preds = model(images)[:, idx].argmax(1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)
            return 100.0 * correct / total

        pt2e_results = run_pt2e_experiment(
            raw_model_loader, eval_pt2e_vision, "ResNet-18 (PT2E)",
            device, loader, example_input, do_benchmark=do_benchmark,
        )
        all_results.extend(pt2e_results)
    else:
        print("\n  [skip] PT2E not available")

    return all_results


def run_bert_experiments(device, num_samples=None, do_benchmark=True,
                         do_compile=False):
    """BERT-base on SST-2: TorchAO INT8-WO and W8A8-Dynamic."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: BERT-base-uncased on SST-2")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loader, _ = get_sst2_loader(tokenizer, batch_size=32,
                                num_samples=num_samples)
    print(f"  Dataset: SST-2 val ({len(loader.dataset)} samples)")
    print(f"  Device:  {device}")

    def model_loader():
        return AutoModelForSequenceClassification.from_pretrained(
            "textattack/bert-base-uncased-SST-2")

    probe = model_loader()
    print_arch_breakdown(probe, "BERT-base")
    del probe

    example_batch = next(iter(loader))
    example_input = {
        "input_ids": example_batch["input_ids"][:1].to(device),
        "attention_mask": example_batch["attention_mask"][:1].to(device),
    }

    print(f"\n{'—'*60}")
    print("  TorchAO quantize_() on a transformer")
    print("  BERT-base: 74 nn.Linear layers (78.2% of params)")
    print("  TorchAO is built for this architecture.")
    print(f"{'—'*60}")

    results = run_torchao_experiment(
        model_loader, evaluate_nlp, "BERT-base",
        device, loader, example_input, do_benchmark, do_compile,
    )

    verify_save_load_roundtrip(
        model_loader, evaluate_nlp, "bert_base", device, loader,
    )

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ch 6.2: PyTorch/TorchAO quantization path end-to-end")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resnet", action="store_true")
    parser.add_argument("--bert", action="store_true")
    parser.add_argument("--no-benchmark", dest="benchmark",
                        action="store_false", default=True)
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Also benchmark with torch.compile "
                        "(shows compiled vs eager latency)")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./ch6_outputs")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if not (args.all or args.resnet or args.bert):
        args.all = True

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 70)
    print("Chapter 6.2: The PyTorch and TorchAO Quantization Path")
    print("=" * 70)
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  TorchAO:  {TORCHAO_VERSION or 'not installed'}")
    print(f"  PT2E:     {'available' if HAS_PT2E else 'not available'}")
    print(f"  Device:   {device}")
    if device.type == "cuda":
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")
    print(f"  Compile:  {args.compile}")

    all_results = []

    if args.all or args.resnet:
        if HAS_TORCHVISION:
            all_results.extend(
                run_resnet_experiments(device, args.num_samples,
                                      args.benchmark, args.compile))

    if args.all or args.bert:
        if HAS_TRANSFORMERS and HAS_TORCHAO:
            all_results.extend(
                run_bert_experiments(device, args.num_samples,
                                    args.benchmark, args.compile))

    if all_results:
        print_summary_table(all_results)
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, "ch6_torchao_results.json")
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults saved to: {json_path}")

        if args.save_plots and HAS_MATPLOTLIB:
            plot_path = os.path.join(args.output_dir,
                                     "CH06_F02_torchao_comparison.png")
            plot_comparison(all_results, save_path=plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()