#!/usr/bin/env python3
"""
Chapter 6.2 Companion Script: Run the PyTorch and TorchAO Path End-to-End

TorchAO's quantize_() API replaces weight tensors in-place with quantized
subclasses — one line of code, no graph capture, no calibration for
weight-only configs. This script applies it to three models and measures
what quantization actually delivers: compression and accuracy preservation.

Models:
  - ResNet-18 on ImageNette  → The architectural lesson (why TorchAO gives
    minimal compression on CNNs)
  - BERT-base on SST-2       → The transformer sweet spot (2.40× compression,
    -0.11% accuracy)
  - TinyLlama-1.1B           → Compression at LLM scale (3.38×, 4.2 GB → 1.2 GB)

Latency is measured for completeness but is NOT the deliverable of this
section. TorchAO quantization reduces size unconditionally; latency
improvement requires a production runtime (ONNX Runtime, TensorRT, or
torch.compile on supported GPU hardware) — covered in Chapter 9.

Usage:
    python ch6_pytorch_torchao_path.py --all --save-plots
    python ch6_pytorch_torchao_path.py --bert
    python ch6_pytorch_torchao_path.py --resnet
    python ch6_pytorch_torchao_path.py --tinyllama
    python ch6_pytorch_torchao_path.py --all --num-samples 200   # quick dev
    python ch6_pytorch_torchao_path.py --all --no-benchmark      # skip latency

Requirements:
    pip install torch torchvision torchao transformers datasets matplotlib tqdm
"""

import argparse
import gc
import json
import os
import tempfile
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
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
    print("[error] torchao not installed: pip install torchao")


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
    "orange_l2": (255/255, 180/255, 88/255),
    "black_l3": (75/255, 75/255, 75/255),
}
HATCH_PATTERNS = ["", "///", "xxx", "\\\\\\"]


# ============================================================================
# Data classes
# ============================================================================
@dataclass
class QuantizationResult:
    model_name: str
    config_name: str
    config_desc: str
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    original_accuracy: Optional[float]
    quantized_accuracy: Optional[float]
    accuracy_delta: Optional[float]
    original_latency_ms: float = 0.0
    quantized_latency_ms: float = 0.0
    latency_ratio: float = 0.0
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
    split="val", batch_size=64, num_samples=None, num_workers=4,
    data_dir="./data",
):
    imagenette_dir = Path(data_dir) / "imagenette2"
    split_dir = imagenette_dir / split
    if not split_dir.exists():
        print(f"  Downloading ImageNette to {data_dir}/ ...")
        import urllib.request, tarfile
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
        tgz_path = Path(data_dir) / "imagenette2.tgz"
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(url, tgz_path)
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(data_dir)
        tgz_path.unlink()
        print("  Download complete.")

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(str(split_dir), transform=transform)
    if num_samples and num_samples < len(dataset):
        dataset = Subset(dataset,
                         torch.randperm(len(dataset))[:num_samples].tolist())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def get_sst2_loader(tokenizer, batch_size=32, num_samples=None,
                     max_length=128, split="validation"):
    dataset = load_dataset("glue", "sst2", split=split)
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))

    def tok_fn(ex):
        return tokenizer(ex["sentence"], padding="max_length",
                         truncation=True, max_length=max_length,
                         return_tensors="pt")

    tokenized = dataset.map(tok_fn, batched=True,
                            remove_columns=["sentence", "idx"])
    tokenized.set_format("torch")
    return DataLoader(tokenized, batch_size=batch_size, shuffle=False)


# ============================================================================
# Measurement utilities
# ============================================================================

def measure_model_size_mb(model):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / (1024 * 1024)


def evaluate_vision(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="    Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            correct += model(images).argmax(1).eq(labels).sum().item()
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
    return float(np.median(times)), float(np.std(times))


def print_arch_breakdown(model, model_name):
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    n_lin = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    p_conv = sum(p.numel() for m in model.modules()
                 if isinstance(m, nn.Conv2d) for p in m.parameters())
    p_lin = sum(p.numel() for m in model.modules()
                if isinstance(m, nn.Linear) for p in m.parameters())
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Architecture breakdown ({model_name}):")
    if n_conv > 0:
        print(f"    Conv2d layers:  {n_conv:>3d}  ({p_conv:,} params, "
              f"{100*p_conv/total:.1f}%)")
    print(f"    Linear layers:  {n_lin:>3d}  ({p_lin:,} params, "
          f"{100*p_lin/total:.1f}%)")
    print(f"    Total params:   {total:,}")
    return n_conv, n_lin, p_conv, p_lin, total


# ============================================================================
# Weight inspection — Listing 6.1
# ============================================================================

def inspect_torchao_weights(model, config_name):
    """
    Listing 6.1: Inspect weight tensors after TorchAO quantize_()

    After quantize_(), nn.Linear weights become AffineQuantizedTensor
    subclasses that store INT8 data plus per-channel scales. The module
    structure stays unchanged — still nn.Linear.
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
                if hasattr(w, "tensor_impl"):
                    impl = w.tensor_impl
                    if hasattr(impl, "int_data"):
                        try:
                            idata = impl.int_data
                            if callable(idata): idata = idata()
                            extra += f", int_dtype={idata.dtype}"
                        except Exception:
                            pass
                print(f"    {name}: {w_type}, shape={list(w.shape)}{extra}")
                shown += 1

    if n_quantized - shown > 0:
        print(f"    ... and {n_quantized - shown} more quantized layers")
    print(f"    → {n_quantized}/{n_linear} nn.Linear layers quantized")
    return n_quantized, n_linear


# ============================================================================
# Core quantization pipeline — Listing 6.2
# ============================================================================

def run_torchao_experiment(
    model_loader_fn, evaluate_fn, model_name, device, loader,
    example_input, do_benchmark=True,
):
    """
    Listing 6.2: TorchAO quantize_() end-to-end

    Applies INT8 weight-only and W8A8 dynamic configs, measures accuracy,
    size, and latency for each.
    """
    results = []
    configs = [
        ("INT8-WO",
         "INT8 weight-only (per-channel symmetric)",
         Int8WeightOnlyConfig()),                                  #A
        ("W8A8-Dynamic",
         "INT8 weight + INT8 dynamic activation (per-token)",
         Int8DynamicActivationInt8WeightConfig()),                  #B
    ]
    # A Weights quantized to INT8, activations stay in FP32 at inference
    # B Weights quantized statically, activations quantized per-token at runtime

    # --- Baseline ---
    print(f"\n  BASELINE ({model_name}, original precision)")
    baseline = model_loader_fn().to(device).eval()
    base_size = measure_model_size_mb(baseline)
    base_acc = evaluate_fn(baseline, loader, device) if evaluate_fn else None
    base_lat, base_std = 0.0, 0.0
    if do_benchmark:
        base_lat, base_std = benchmark_latency(baseline, example_input,
                                               device=device)
    acc_str = f" | Accuracy: {base_acc:.2f}%" if base_acc is not None else ""
    lat_str = f" | Latency: {base_lat:.2f} ms (±{base_std:.2f})" if do_benchmark else ""
    print(f"    Size: {base_size:.2f} MB{acc_str}{lat_str}")
    del baseline; gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Quantized variants ---
    for cfg_name, cfg_desc, config in configs:
        print(f"\n  {model_name} → {cfg_name}")
        print(f"    Config: {cfg_desc}")

        q_model = model_loader_fn().to(device).eval()
        try:
            quantize_(q_model, config)                             #C
        except Exception as e:
            print(f"    [skip] quantize_() failed: {e}")
            del q_model; gc.collect()
            continue
        # C In-place: replaces nn.Linear weight tensors with quantized subclasses

        q_size = measure_model_size_mb(q_model)
        compression = base_size / q_size if q_size > 0 else 0
        n_q, n_total = inspect_torchao_weights(q_model, cfg_name)

        q_acc = evaluate_fn(q_model, loader, device) if evaluate_fn else None
        acc_delta = (q_acc - base_acc) if (q_acc is not None
                                           and base_acc is not None) else None

        q_lat, lat_ratio = 0.0, 0.0
        if do_benchmark:
            q_lat, q_std = benchmark_latency(q_model, example_input,
                                             device=device)
            lat_ratio = base_lat / q_lat if q_lat > 0 else 0

        # Print results
        parts = [f"Size: {q_size:.2f} MB ({compression:.2f}x)"]
        if q_acc is not None:
            parts.append(f"Acc: {q_acc:.2f}% (Δ{acc_delta:+.2f}%)")
        if do_benchmark:
            parts.append(f"Latency: {q_lat:.2f} ms ({lat_ratio:.2f}x)")
        print(f"    {' | '.join(parts)}")

        results.append(QuantizationResult(
            model_name=model_name, config_name=cfg_name,
            config_desc=cfg_desc,
            original_size_mb=base_size, quantized_size_mb=q_size,
            compression_ratio=compression,
            original_accuracy=base_acc, quantized_accuracy=q_acc,
            accuracy_delta=acc_delta,
            original_latency_ms=base_lat, quantized_latency_ms=q_lat,
            latency_ratio=lat_ratio,
            num_quantized_layers=n_q, num_total_layers=n_total,
            device=str(device),
        ))
        del q_model; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ============================================================================
# Save/load round-trip — Listing 6.3
# ============================================================================

def verify_save_load_roundtrip(
    model_loader_fn, evaluate_fn, model_name, device, loader,
    save_dir="./ch6_artifacts",
):
    """
    Listing 6.3: Serialize and reload a TorchAO-quantized model

    Save the state_dict (not full model — avoids cross-version pickle
    issues). Reload into a fresh quantized skeleton. Verify accuracy
    is identical.
    """
    print(f"\n  SAVE/LOAD ROUND-TRIP ({model_name})")
    os.makedirs(save_dir, exist_ok=True)

    model = model_loader_fn().to(device).eval()
    quantize_(model, Int8WeightOnlyConfig())
    acc_before = evaluate_fn(model, loader, device)
    size_before = measure_model_size_mb(model)
    print(f"    Before: accuracy={acc_before:.2f}%, size={size_before:.2f} MB")

    save_path = os.path.join(save_dir, f"{model_name}_int8wo.pt")
    torch.save(model.state_dict(), save_path)                      #D
    file_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"    Saved state_dict → {save_path} ({file_mb:.2f} MB)")
    # D State-dict preserves AffineQuantizedTensor subclass metadata

    loaded = model_loader_fn().to(device).eval()
    quantize_(loaded, Int8WeightOnlyConfig())                      #E
    loaded.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=False))
    # E Rebuild quantized skeleton first, then fill with saved weights

    acc_after = evaluate_fn(loaded, loader, device)
    match = abs(acc_before - acc_after) < 1e-6
    print(f"    After:  accuracy={acc_after:.2f}%")
    print(f"    Round-trip: {'PASS' if match else 'FAIL'}")

    del model, loaded; gc.collect()
    return match


# ============================================================================
# Summary and visualization
# ============================================================================

def print_summary_table(results):
    print(f"\n{'='*100}")
    print("SUMMARY: TorchAO Quantization Results")
    print(f"{'='*100}")
    print(f"{'Model':<20} {'Config':<16} {'Size':>9} {'Ratio':>7} "
          f"{'Accuracy':>9} {'Δ Acc':>8} {'Latency':>10} {'Lat Ratio':>10}")
    print("-" * 100)

    current = None
    for r in results:
        if r.model_name != current:
            current = r.model_name
            acc_s = (f"{r.original_accuracy:.2f}%"
                     if r.original_accuracy is not None else "—")
            lat_s = (f"{r.original_latency_ms:.2f}ms"
                     if r.original_latency_ms > 0 else "—")
            print(f"{r.model_name:<20} {'FP32 baseline':<16} "
                  f"{r.original_size_mb:>8.1f}M {'1.00x':>7} "
                  f"{acc_s:>9} {'—':>8} {lat_s:>10} {'1.00x':>10}")

        acc_s = (f"{r.quantized_accuracy:.2f}%"
                 if r.quantized_accuracy is not None else "—")
        delta_s = (f"{r.accuracy_delta:+.2f}%"
                   if r.accuracy_delta is not None else "—")
        lat_s = (f"{r.quantized_latency_ms:.2f}ms"
                 if r.quantized_latency_ms > 0 else "—")
        rat_s = f"{r.latency_ratio:.2f}x" if r.latency_ratio > 0 else "—"
        print(f"{'':.<20} {r.config_name:<16} "
              f"{r.quantized_size_mb:>8.1f}M {r.compression_ratio:>6.2f}x "
              f"{acc_s:>9} {delta_s:>8} {lat_s:>10} {rat_s:>10}")

    print("-" * 100)


def plot_comparison(results, save_path=None):
    if not HAS_MATPLOTLIB or not results:
        return

    models = OrderedDict()
    for r in results:
        models.setdefault(r.model_name, []).append(r)

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    bar_colors = [COLORS["blue_l2"], COLORS["green_l3"]]
    hatches = HATCH_PATTERNS

    for col, (mname, mresults) in enumerate(models.items()):
        ax = axes[col]
        configs = [r.config_name for r in mresults]
        n = len(configs)
        x = np.arange(n)
        base_s = mresults[0].original_size_mb
        sizes = [r.quantized_size_mb for r in mresults]

        bars = ax.bar(x, sizes, color=bar_colors[:n], edgecolor="black",
                      lw=0.8)
        for i, b in enumerate(bars):
            b.set_hatch(hatches[i])
        ax.axhline(base_s, color=COLORS["black_l3"], ls="--", lw=1.2,
                    label=f"FP32 ({base_s:.0f} MB)")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=8, rotation=15)
        ax.set_ylabel("Model size (MB)", fontsize=10)
        ax.set_title(mname, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        for i, v in enumerate(sizes):
            ratio = mresults[i].compression_ratio
            ax.text(i, v + base_s * 0.02, f"{v:.0f} MB\n({ratio:.2f}×)",
                    ha="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
        print(f"\n  Plot saved: {save_path}")
    plt.close()


# ============================================================================
# Experiment runners
# ============================================================================

def run_resnet_experiment(device, num_samples=None, do_benchmark=True):
    """
    ResNet-18 on ImageNette: TorchAO on a CNN.

    Demonstrates that TorchAO's stable configs only target nn.Linear.
    ResNet-18 is 95.5% Conv2d → 1.03× compression. This is the expected
    result, not a bug. CNN quantization in PyTorch requires PT2E (export
    quantization with a backend-specific quantizer) or a runtime-level
    quantizer — both covered in Chapter 9.
    """
    assert HAS_TORCHVISION and HAS_TORCHAO
    print("\n" + "=" * 70)
    print("EXPERIMENT: ResNet-18 on ImageNette (TorchAO)")
    print("=" * 70)

    loader = get_imagenette_loader(split="val", num_samples=num_samples)
    print(f"  Dataset: ImageNette val ({len(loader.dataset)} samples)")
    print(f"  Device:  {device}")

    def model_loader():
        base = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        return ImageNetteWrapper(base)

    probe = model_loader()
    print_arch_breakdown(probe, "ResNet-18")
    print(f"  ⚠ TorchAO quantize_() targets nn.Linear only — "
          f"expect minimal compression.")
    del probe

    example_input = torch.randn(1, 3, 224, 224, device=device)
    return run_torchao_experiment(
        model_loader, evaluate_vision, "ResNet-18",
        device, loader, example_input, do_benchmark,
    )


def run_bert_experiment(device, num_samples=None, do_benchmark=True):
    """
    BERT-base on SST-2: TorchAO on a transformer.

    74 nn.Linear layers (78.2% of params) → TorchAO compresses effectively.
    INT8 weight-only: 2.40× compression, -0.11% accuracy.
    """
    assert HAS_TRANSFORMERS and HAS_TORCHAO
    print("\n" + "=" * 70)
    print("EXPERIMENT: BERT-base-uncased on SST-2 (TorchAO)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    loader = get_sst2_loader(tokenizer, batch_size=32,
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

    results = run_torchao_experiment(
        model_loader, evaluate_nlp, "BERT-base",
        device, loader, example_input, do_benchmark,
    )

    # Save/load round-trip
    verify_save_load_roundtrip(
        model_loader, evaluate_nlp, "bert_base", device, loader,
    )

    return results


def run_tinyllama_experiment(device, do_benchmark=True):
    """
    TinyLlama-1.1B: TorchAO at LLM scale.

    155 nn.Linear layers (94% of params). Demonstrates compression
    at a scale where the storage savings are tangible: 4.2 GB → 1.2 GB.
    No accuracy eval (perplexity evaluation belongs in Chapter 7).
    """
    assert HAS_TRANSFORMERS and HAS_TORCHAO
    print("\n" + "=" * 70)
    print("EXPERIMENT: TinyLlama-1.1B (TorchAO)")
    print("=" * 70)

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model:   {MODEL_ID}")
    print(f"  Device:  {device}")

    def model_loader():
        return AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32)

    probe = model_loader()
    print_arch_breakdown(probe, "TinyLlama-1.1B")
    del probe; gc.collect()

    # Example input for latency benchmarking (sequence length 64)
    prompt = "The key advantage of quantization in production is"
    input_ids = tokenizer(prompt, return_tensors="pt",
                          padding="max_length", max_length=64,
                          truncation=True)["input_ids"].to(device)

    # No accuracy eval — perplexity for LLMs is Chapter 7's territory.
    # Pass evaluate_fn=None; the pipeline handles it gracefully.
    return run_torchao_experiment(
        model_loader, None, "TinyLlama-1.1B",
        device, None, input_ids, do_benchmark,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ch 6.2: TorchAO quantization path end-to-end")
    parser.add_argument("--all", action="store_true",
                        help="Run all three experiments")
    parser.add_argument("--resnet", action="store_true",
                        help="ResNet-18 on ImageNette")
    parser.add_argument("--bert", action="store_true",
                        help="BERT-base on SST-2")
    parser.add_argument("--tinyllama", action="store_true",
                        help="TinyLlama-1.1B compression")
    parser.add_argument("--no-benchmark", dest="benchmark",
                        action="store_false", default=True,
                        help="Skip latency benchmarking")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit eval samples (None = full set)")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./ch6_outputs")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu/cuda)")
    args = parser.parse_args()

    if not (args.all or args.resnet or args.bert or args.tinyllama):
        args.all = True

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("=" * 70)
    print("Chapter 6.2: The PyTorch and TorchAO Quantization Path")
    print("=" * 70)
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  TorchAO:  {TORCHAO_VERSION or 'not installed'}")
    print(f"  Device:   {device}")
    if device.type == "cuda":
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")

    all_results = []

    # --- ResNet-18: the architectural lesson ---
    if args.all or args.resnet:
        if HAS_TORCHVISION and HAS_TORCHAO:
            all_results.extend(
                run_resnet_experiment(device, args.num_samples,
                                     args.benchmark))

    # --- BERT-base: the transformer sweet spot ---
    if args.all or args.bert:
        if HAS_TRANSFORMERS and HAS_TORCHAO:
            all_results.extend(
                run_bert_experiment(device, args.num_samples,
                                   args.benchmark))

    # --- TinyLlama-1.1B: compression at scale ---
    if args.all or args.tinyllama:
        if HAS_TRANSFORMERS and HAS_TORCHAO:
            all_results.extend(
                run_tinyllama_experiment(device, args.benchmark))

    # --- Summary ---
    if all_results:
        print_summary_table(all_results)

        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, "ch6_torchao_results.json")
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults saved to: {json_path}")

        if args.save_plots and HAS_MATPLOTLIB:
            plot_path = os.path.join(args.output_dir,
                                     "CH06_F02_Kalyanarangan.png")
            plot_comparison(all_results, save_path=plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()