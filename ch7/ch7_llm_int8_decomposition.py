#!/usr/bin/env python3
"""
Chapter 7 — Section 7.2 Companion Script
Apply outlier-aware weight paths: LLM.int8() mixed-precision decomposition

This script builds LLM.int8() from scratch, then validates it against the
production bitsandbytes implementation. The pedagogical arc:

  1. Manual decomposition  — split a matmul into FP16 outlier + INT8 normal paths
  2. Threshold sweep        — why 6.0 is the right cutoff (not 5.0 or 8.0)
  3. bitsandbytes INT8      — one-line production deployment
  4. Perplexity showdown    — FP16 vs naive INT8 vs LLM.int8() on WikiText-2
  5. Figures                — decomposition anatomy, threshold curve, perplexity bars

Hardware: Colab T4 (16 GB VRAM). OPT-6.7B in LLM.int8() uses ~7 GB.

Usage:
    # Full pipeline (T4 GPU required)
    python ch7_llm_int8_decomposition.py --mode all --save-plots

    # Manual decomposition only (fits on T4 with FP16 model)
    python ch7_llm_int8_decomposition.py --mode decomposition --save-plots

    # Threshold sweep (requires FP16 model in memory)
    python ch7_llm_int8_decomposition.py --mode threshold --save-plots

    # Perplexity comparison (loads models sequentially)
    python ch7_llm_int8_decomposition.py --mode perplexity --save-plots

    # CPU-only (uses OPT-125M for illustration, not publishable numbers)
    python ch7_llm_int8_decomposition.py --mode all --save-plots \
        --model opt-125m --device cpu

References:
    Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers
    at Scale," NeurIPS 2022.
"""

import argparse
import gc
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============================================================================
# Configuration
# ============================================================================

MODEL_REGISTRY = {
    "opt-125m": "facebook/opt-125m",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-6.7b": "facebook/opt-6.7b",
}

@dataclass
class Config:
    """Central configuration for all experiments."""
    model_key: str = "opt-6.7b"
    device: str = "cuda"
    seed: int = 42

    # Calibration / profiling
    calib_samples: int = 128
    calib_seq_len: int = 512

    # Perplexity evaluation
    eval_samples: int = 64            # WikiText-2 test sequences
    eval_seq_len: int = 512

    # Threshold sweep
    thresholds: List[float] = field(default_factory=lambda: [
        0.0, 2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 10.0, 20.0
    ])

    # Visualization — Manning publication quality
    save_plots: bool = False
    plot_dir: str = "ch7_plots"
    dpi: int = 300                    # Manning requires 300 DPI minimum
    figsize: Tuple[int, int] = (10, 6)

    @property
    def model_name(self) -> str:
        return MODEL_REGISTRY[self.model_key]


# ============================================================================
# Manning Publication Figure Styling
# ============================================================================

def setup_manning_style():
    """
    Configure matplotlib for Manning publication figures.

    - Grayscale-safe palette with distinct hatching patterns for B&W print
    - Serif font family for body text
    - White background, no unnecessary chartjunk
    - Sizes tuned for ~4.5-inch column width in Manning's layout
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "grid.color": "#cccccc",
        "lines.linewidth": 1.5,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    })


# Manning house style (matches CH06_F03_Kalyanarangan):
# - Gray solid fill for baseline/reference
# - Blue family with hatching for quantized variants
# - Hatching provides B&W print safety; color provides screen clarity
# Line plots: (color, marker, linestyle)
MANNING_LINE_PALETTE = [
    ("#333333", "o",  "-"),      # Dark gray, circle, solid
    ("#4A90D9", "s",  "--"),     # Blue, square, dashed
    ("#1B5299", "^",  "-."),    # Dark blue, triangle, dash-dot
]

# Bar chart semantic colors (matching CH06 horizontal bar style)
COLORS = {
    "baseline":   "#999999",     # Gray — baseline / reference
    "primary":    "#4A90D9",     # Blue — primary quantized variant
    "secondary":  "#1B5299",     # Dark blue — secondary variant
}
HATCHES = {
    "baseline":   "",            # Solid fill (gray, no hatch)
    "primary":    "//",          # Diagonal hatch
    "secondary":  "xx",          # Cross hatch
}


# ============================================================================
# Shared utilities
# ============================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_mem_mb() -> float:
    """Current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def free_gpu():
    """Flush GPU memory. Caller must `del` their own model references first."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_wikitext2_tokens(tokenizer, n_samples: int, seq_len: int, split: str = "test"):
    """Load WikiText-2 and return a list of token tensors."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[0]

    sequences = []
    for i in range(0, len(input_ids) - seq_len, seq_len):
        seq = input_ids[i : i + seq_len].unsqueeze(0)
        sequences.append(seq)
        if len(sequences) >= n_samples:
            break

    print(f"  Loaded {len(sequences)} sequences of length {seq_len} from WikiText-2 ({split})")
    return sequences


def save_or_show(fig, filename_stem: str, cfg: Config):
    """
    Save figure in three formats (PDF, SVG, PNG) for Manning publication.

    Manning requires vector formats (PDF preferred, SVG as fallback).
    PNG is generated at 300 DPI for quick preview and web/Colab display.

    Args:
        fig: matplotlib figure
        filename_stem: base name WITHOUT extension (e.g., "fig7_4_decomposition")
        cfg: Config with save_plots, plot_dir, dpi
    """
    fig.tight_layout()
    if cfg.save_plots:
        os.makedirs(cfg.plot_dir, exist_ok=True)
        for ext in ("pdf", "png"):
            path = os.path.join(cfg.plot_dir, f"{filename_stem}.{ext}")
            fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {cfg.plot_dir}/{filename_stem}.{{pdf,png}}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================================
# Section 1: Manual Mixed-Precision Decomposition
# ============================================================================

def quantize_absmax_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel absmax INT8 quantization (Listing 7.3 candidate).

    For a weight matrix W of shape [out_features, in_features], compute one
    scale per output channel (row).  This is the "normal path" quantization
    in LLM.int8().

    Returns:
        q_int8: quantized tensor (int8 dtype)
        scales: per-channel scale factors (float32)
    """
    scales = tensor.float().abs().amax(dim=1, keepdim=True) / 127.0    # [out, 1]
    scales = torch.clamp(scales, min=1e-8)
    q_int8 = torch.round(tensor.float() / scales).clamp(-128, 127).to(torch.int8)
    return q_int8, scales


def mixed_precision_matmul(
    X: torch.Tensor,
    W: torch.Tensor,
    threshold: float = 6.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Manual LLM.int8() decomposition (Listing 7.4 candidate).

    Given activation X [batch*seq, in_features] and weight W [out_features, in_features]:
      1. Find outlier dimensions where any activation value exceeds `threshold`
      2. Split X and W into outlier columns (FP16 path) and normal columns (INT8 path)
      3. Compute both matmuls, sum the results

    This is the core algorithm from Dettmers et al. (2022), implemented from
    scratch so you can see every step.

    Edge cases handled:
      - threshold=0.0 → all dims are outliers → pure FP16 (no INT8 path)
      - threshold very high → no outliers → pure INT8 (no FP16 path)
    """
    # --- Step 1: Identify outlier feature dimensions ---
    # An outlier dimension is any input feature where at least one activation
    # value (across all tokens in the batch) exceeds the threshold.
    outlier_mask = X.abs().max(dim=0).values > threshold               # [in_features]
    outlier_dims = outlier_mask.nonzero(as_tuple=True)[0]
    normal_dims = (~outlier_mask).nonzero(as_tuple=True)[0]

    n_outlier = len(outlier_dims)
    n_normal = len(normal_dims)
    n_total = X.shape[1]

    stats = {
        "n_outlier": n_outlier,
        "n_normal": n_normal,
        "n_total": n_total,
        "outlier_pct": 100.0 * n_outlier / n_total,
        "outlier_dims": outlier_dims.cpu().tolist(),
    }

    # --- Edge case: ALL dims are outliers (e.g., threshold=0.0) ---
    if n_normal == 0:
        # Pure FP16 path — no INT8 at all
        out = X.float() @ W.float().T
        return out.to(X.dtype), stats

    # --- Edge case: NO outliers (threshold very high) ---
    if n_outlier == 0:
        # Pure INT8 path
        W_q, W_s = quantize_absmax_int8(W)
        X_f = X.float()
        X_scales = X_f.abs().amax(dim=1, keepdim=True) / 127.0
        X_scales = torch.clamp(X_scales, min=1e-8)
        X_q = torch.round(X_f / X_scales).clamp(-128, 127).to(torch.int8)

        # INT8 matmul (simulated — real hardware uses integer arithmetic)
        out = (X_q.float() @ W_q.float().T) * (X_scales * W_s.float().T)
        return out.to(X.dtype), stats

    # --- Step 2: Split into two paths ---
    # Full-precision path: outlier columns only (use float32 for CPU compat;
    # on GPU, bitsandbytes uses FP16 — the key point is NO quantization error)
    X_outlier = X[:, outlier_dims].float()                             # [batch, n_outlier]
    W_outlier = W[:, outlier_dims].float()                             # [out, n_outlier]

    # INT8 path: everything else
    X_normal = X[:, normal_dims].float()                               # [batch, n_normal]
    W_normal = W[:, normal_dims]                                       # [out, n_normal]

    # --- Step 3: Compute both paths ---
    # Full-precision matmul for outlier dimensions — no quantization error
    out_outlier = X_outlier @ W_outlier.T                              # [batch, out]

    # INT8 matmul for normal dimensions — per-channel weights, per-token activations
    W_q, W_s = quantize_absmax_int8(W_normal)
    X_scales = X_normal.abs().amax(dim=1, keepdim=True) / 127.0
    X_scales = torch.clamp(X_scales, min=1e-8)
    X_q = torch.round(X_normal / X_scales).clamp(-128, 127).to(torch.int8)

    out_normal = (X_q.float() @ W_q.float().T) * (X_scales * W_s.float().T)  # [batch, out]

    # --- Step 4: Sum the two paths ---
    out = out_outlier.float() + out_normal.float()

    return out.to(X.dtype), stats


def run_decomposition_experiment(cfg: Config):
    """
    Experiment 1: Apply manual LLM.int8() decomposition to every fc1 layer
    in the model and measure reconstruction error vs FP16 ground truth.

    Generates Figure 7.4: Decomposition anatomy — per-layer outlier fraction
    and reconstruction error comparison.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Manual Mixed-Precision Decomposition")
    print("=" * 72)

    device = cfg.device
    model_name = cfg.model_name

    # Load model in FP16 for ground-truth computation
    print(f"\nLoading {model_name} in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    print(f"  GPU memory after load: {gpu_mem_mb():.0f} MB")

    # Capture activations from calibration data
    print(f"\nRunning calibration data ({cfg.calib_samples} samples)...")
    calib_seqs = load_wikitext2_tokens(
        tokenizer, cfg.calib_samples, cfg.calib_seq_len, split="train"
    )

    # Determine model architecture (OPT-specific)
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    # Hook to capture fc1 inputs — store on CPU to save GPU memory
    captured_inputs = {}
    def make_capture_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach()
            if name not in captured_inputs:
                captured_inputs[name] = []
            if len(captured_inputs[name]) < 8:
                captured_inputs[name].append(x.reshape(-1, x.shape[-1]).cpu())
        return hook_fn

    hooks = []
    for idx in range(num_layers):
        layer = model.model.decoder.layers[idx]
        h = layer.fc1.register_forward_hook(make_capture_hook(f"layer_{idx:02d}"))
        hooks.append(h)

    # Run calibration
    with torch.no_grad():
        for seq in calib_seqs[:8]:  # Use 8 sequences for activation capture
            model(seq.to(device))

    for h in hooks:
        h.remove()

    # Copy fc1 weights to CPU before freeing the model
    print(f"\n  Copying fc1 weights to CPU and freeing GPU model...")
    fc1_weights_cpu = {}
    for idx in range(num_layers):
        key = f"layer_{idx:02d}"
        if key in captured_inputs and captured_inputs[key]:
            fc1_weights_cpu[key] = model.model.decoder.layers[idx].fc1.weight.data.cpu().clone()

    del model
    free_gpu()
    print(f"  GPU memory after free: {gpu_mem_mb():.0f} MB")

    # Now decompose each layer entirely on CPU
    print(f"\nDecomposing {num_layers} fc1 layers on CPU (threshold=6.0)...")
    print(f"{'Layer':<12} {'Outlier dims':>14} {'Outlier %':>10} "
          f"{'FP16 MSE':>12} {'INT8 MSE':>12} {'Mixed MSE':>12} {'Ratio':>8}")
    print("-" * 82)

    layer_results = []

    for idx in range(num_layers):
        key = f"layer_{idx:02d}"
        if key not in captured_inputs or not captured_inputs[key]:
            continue
        if key not in fc1_weights_cpu:
            continue

        # Concatenate captured activations (already on CPU)
        X = torch.cat(captured_inputs[key], dim=0)[:2048]  # Cap at 2048 tokens
        W = fc1_weights_cpu[key]

        # Ground truth: FP32 matmul on CPU
        X_f = X.float()
        W_f = W.float()
        Y_fp16 = (X_f @ W_f.T)

        # Naive INT8 (no decomposition)
        W_q, W_s = quantize_absmax_int8(W)
        X_scales = X.abs().amax(dim=1, keepdim=True) / 127.0
        X_scales = torch.clamp(X_scales, min=1e-8)
        X_q = torch.round(X.float() / X_scales).clamp(-128, 127).to(torch.int8)
        Y_int8 = (X_q.float() @ W_q.float().T) * (X_scales * W_s.float().T)

        # LLM.int8() decomposition
        Y_mixed, stats = mixed_precision_matmul(X, W, threshold=6.0)

        # Compute MSEs
        mse_int8 = ((Y_fp16 - Y_int8) ** 2).mean().item()
        mse_mixed = ((Y_fp16 - Y_mixed.float()) ** 2).mean().item()
        ratio = mse_int8 / max(mse_mixed, 1e-12)

        layer_results.append({
            "layer": idx,
            "n_outlier": stats["n_outlier"],
            "outlier_pct": stats["outlier_pct"],
            "mse_int8": mse_int8,
            "mse_mixed": mse_mixed,
            "ratio": ratio,
            "outlier_dims": stats["outlier_dims"],
        })

        print(f"  Layer {idx:2d}    {stats['n_outlier']:>10d}     "
              f"{stats['outlier_pct']:>7.2f}%   "
              f"{'—':>10s}   {mse_int8:>10.4f}   {mse_mixed:>10.6f}   "
              f"{ratio:>6.1f}×")

        # Free per-layer intermediates
        del X, W, X_f, W_f, Y_fp16, Y_int8, Y_mixed
        gc.collect()

    # Free captured data
    del captured_inputs, fc1_weights_cpu
    gc.collect()

    # Summary statistics
    if layer_results:
        avg_outlier_pct = np.mean([r["outlier_pct"] for r in layer_results])
        avg_ratio = np.mean([r["ratio"] for r in layer_results])
        print(f"\n  Average outlier fraction: {avg_outlier_pct:.2f}%")
        print(f"  Average error reduction (INT8 → mixed): {avg_ratio:.1f}×")

        # Collect universal outlier dims (appear in >80% of layers)
        all_dims = []
        for r in layer_results:
            all_dims.extend(r["outlier_dims"])
        dim_counts = Counter(all_dims)
        threshold_count = int(0.8 * len(layer_results))
        universal = sorted([d for d, c in dim_counts.items() if c >= threshold_count])
        print(f"  Universal outlier dims (>80% layers): {universal[:10]}")

    return layer_results


def plot_decomposition_anatomy(layer_results: list, cfg: Config):
    """
    Figure 7.4: Two-panel figure showing per-layer outlier fraction and
    reconstruction error comparison.  Grayscale-safe with hatching.
    """
    if not layer_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    layers = [r["layer"] for r in layer_results]
    outlier_pcts = [r["outlier_pct"] for r in layer_results]
    mse_int8 = [r["mse_int8"] for r in layer_results]
    mse_mixed = [r["mse_mixed"] for r in layer_results]

    # Panel 1: Outlier fraction per layer
    ax1.bar(layers, outlier_pcts, color=COLORS["primary"], edgecolor="black",
            linewidth=0.5, hatch=HATCHES["primary"])
    ax1.set_xlabel("Transformer Layer")
    ax1.set_ylabel("Outlier Dimensions (%)")
    ax1.set_title("Fraction of Dimensions Routed to FP16 Path")
    avg_pct = np.mean(outlier_pcts)
    ax1.axhline(avg_pct, color="#333333", linestyle="--", linewidth=1.0,
                label=f"Mean: {avg_pct:.2f}%")
    ax1.legend()
    ax1.set_xlim(-0.5, max(layers) + 0.5)

    # Panel 2: Reconstruction error comparison
    x = np.arange(len(layers))
    width = 0.35
    ax2.bar(x - width/2, mse_int8, width, label="Naive INT8",
            color=COLORS["baseline"], edgecolor="black",
            linewidth=0.5, hatch=HATCHES["baseline"])
    ax2.bar(x + width/2, mse_mixed, width, label="LLM.int8()",
            color=COLORS["secondary"], edgecolor="black",
            linewidth=0.5, hatch=HATCHES["secondary"])
    ax2.set_xlabel("Transformer Layer")
    ax2.set_ylabel("Reconstruction MSE (vs FP16)")
    ax2.set_title("Matmul Error: Naive INT8 vs Mixed-Precision")
    ax2.set_xticks(x[::4])
    ax2.set_xticklabels([str(l) for l in layers[::4]])
    ax2.legend()
    ax2.set_yscale("log")

    save_or_show(fig, "fig7_4_decomposition_anatomy", cfg)


# ============================================================================
# Section 2: Threshold Sweep
# ============================================================================

def run_threshold_sweep(cfg: Config):
    """
    Experiment 2: Sweep the outlier threshold from 0 to 20 and measure
    outlier fraction vs reconstruction error for each setting.

    Generates Figure 7.5: Why 6.0 — the threshold sweet spot.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Threshold Sweep")
    print("=" * 72)

    device = cfg.device
    model_name = cfg.model_name

    print(f"\nLoading {model_name} in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Capture activations from target layers: early, middle, late
    target_layers = [0, model.config.num_hidden_layers // 2, model.config.num_hidden_layers - 1]
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            if layer_idx not in captured:
                captured[layer_idx] = []
            if len(captured[layer_idx]) < 4:
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                captured[layer_idx].append(x.cpu())
        return hook_fn

    hooks = []
    for idx in target_layers:
        layer = model.model.decoder.layers[idx]
        h = layer.fc1.register_forward_hook(make_hook(idx))
        hooks.append(h)

    calib_seqs = load_wikitext2_tokens(
        tokenizer, cfg.calib_samples, cfg.calib_seq_len, split="train"
    )
    with torch.no_grad():
        for seq in calib_seqs[:4]:
            model(seq.to(device))

    for h in hooks:
        h.remove()

    # Copy weights to CPU and free model
    fc1_weights_cpu = {}
    for idx in target_layers:
        if idx in captured:
            fc1_weights_cpu[idx] = model.model.decoder.layers[idx].fc1.weight.data.cpu().clone()

    del model
    free_gpu()
    print(f"  GPU memory after free: {gpu_mem_mb():.0f} MB")

    # Sweep thresholds on each target layer (all on CPU)
    print(f"\nSweeping {len(cfg.thresholds)} thresholds across layers {target_layers}...")
    print(f"{'Threshold':>10} {'Layer':>6} {'Outlier %':>10} {'Outlier dims':>13} "
          f"{'INT8 MSE':>12} {'Mixed MSE':>12} {'Ratio':>8}")
    print("-" * 73)

    sweep_results = defaultdict(list)

    for layer_idx in target_layers:
        if layer_idx not in captured or layer_idx not in fc1_weights_cpu:
            continue
        X = torch.cat(captured[layer_idx], dim=0)[:2048]
        W = fc1_weights_cpu[layer_idx]

        Y_fp16 = (X.float() @ W.float().T)

        # Naive INT8 baseline (same for all thresholds)
        W_q, W_s = quantize_absmax_int8(W)
        X_s = X.abs().amax(dim=1, keepdim=True) / 127.0
        X_s = torch.clamp(X_s, min=1e-8)
        X_q = torch.round(X.float() / X_s).clamp(-128, 127).to(torch.int8)
        Y_int8 = (X_q.float() @ W_q.float().T) * (X_s * W_s.float().T)
        mse_int8 = ((Y_fp16 - Y_int8) ** 2).mean().item()

        for thr in cfg.thresholds:
            Y_mixed, stats = mixed_precision_matmul(X, W, threshold=thr)
            mse_mixed = ((Y_fp16 - Y_mixed.float()) ** 2).mean().item()
            # Guard: if threshold=0.0, mixed is pure FP16 → mse_mixed ≈ 0
            ratio = mse_int8 / max(mse_mixed, 1e-12)

            sweep_results[layer_idx].append({
                "threshold": thr,
                "outlier_pct": stats["outlier_pct"],
                "n_outlier": stats["n_outlier"],
                "mse_mixed": mse_mixed,
                "mse_int8": mse_int8,
                "ratio": ratio,
            })

            print(f"  {thr:>8.1f}   {layer_idx:>4d}   {stats['outlier_pct']:>8.2f}%"
                  f"   {stats['n_outlier']:>10d}   {mse_int8:>10.4f}   "
                  f"{mse_mixed:>10.6f}   {ratio:>6.1f}×")

    del captured, fc1_weights_cpu
    gc.collect()

    return sweep_results


def plot_threshold_sweep(sweep_results: dict, cfg: Config):
    """
    Figure 7.5: Dual-axis plot showing outlier fraction and error ratio
    as a function of the threshold.  Grayscale-safe with distinct markers
    and line styles per layer.
    """
    if not sweep_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    layer_indices = sorted(sweep_results.keys())

    # Panel 1: Outlier fraction vs threshold
    for i, layer_idx in enumerate(layer_indices):
        p = MANNING_LINE_PALETTE[i % len(MANNING_LINE_PALETTE)]
        thresholds = [r["threshold"] for r in sweep_results[layer_idx]]
        outlier_pcts = [r["outlier_pct"] for r in sweep_results[layer_idx]]
        ax1.plot(thresholds, outlier_pcts, color=p[0], marker=p[1],
                 linestyle=p[2], markersize=5, linewidth=1.5,
                 label=f"Layer {layer_idx}")

    ax1.axvline(6.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.6,
                label="Default threshold (6.0)")
    ax1.axvspan(5.0, 7.0, alpha=0.06, color="#4A90D9")
    ax1.set_xlabel("Outlier Threshold")
    ax1.set_ylabel("Dimensions Routed to FP16 (%)")
    ax1.set_title("FP16 Overhead vs Threshold")
    ax1.legend()
    ax1.set_xlim(-0.5, max(cfg.thresholds) + 0.5)

    # Panel 2: Error reduction ratio vs threshold
    for i, layer_idx in enumerate(layer_indices):
        p = MANNING_LINE_PALETTE[i % len(MANNING_LINE_PALETTE)]
        thresholds = [r["threshold"] for r in sweep_results[layer_idx]]
        ratios = [r["ratio"] for r in sweep_results[layer_idx]]
        ax2.plot(thresholds, ratios, color=p[0], marker=p[1],
                 linestyle=p[2], markersize=5, linewidth=1.5,
                 label=f"Layer {layer_idx}")

    ax2.axvline(6.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.6,
                label="Default threshold (6.0)")
    ax2.axvspan(5.0, 7.0, alpha=0.06, color="#4A90D9")
    ax2.set_xlabel("Outlier Threshold")
    ax2.set_ylabel("Error Reduction Ratio (INT8 MSE / Mixed MSE)")
    ax2.set_title("Quality Gain vs Threshold")
    ax2.legend()
    ax2.set_yscale("log")
    ax2.set_xlim(-0.5, max(cfg.thresholds) + 0.5)

    save_or_show(fig, "fig7_5_threshold_sweep", cfg)


# ============================================================================
# Section 3: bitsandbytes LLM.int8() — Production Path
# ============================================================================

def run_bitsandbytes_int8(cfg: Config):
    """
    Experiment 3: Load the model with bitsandbytes LLM.int8() and inspect
    the resulting memory footprint and module structure.

    This is the production path — one line replaces the entire manual
    decomposition from Experiment 1.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: bitsandbytes LLM.int8() Production Path")
    print("=" * 72)

    if cfg.device == "cpu":
        print("  Skipping bitsandbytes INT8 (requires CUDA)")
        return None

    model_name = cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Listing 7.5 candidate: One-line LLM.int8() deployment ---
    print(f"\nLoading {model_name} with LLM.int8()...")
    mem_before = gpu_mem_mb()

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,          # The threshold from our sweep
    )

    model_int8 = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
    )

    mem_after = gpu_mem_mb()
    mem_int8 = mem_after - mem_before
    print(f"  GPU memory for LLM.int8() model: {mem_int8:.0f} MB")

    # Inspect module types — show which layers got replaced
    module_types = defaultdict(int)
    for name, module in model_int8.named_modules():
        class_name = type(module).__name__
        module_types[class_name] += 1

    print(f"\n  Module type distribution:")
    for cls, count in sorted(module_types.items(), key=lambda x: -x[1])[:10]:
        print(f"    {cls:<30s} {count:>4d}")

    # Quick generation test
    print(f"\n  Generation test:")
    prompt = "The key innovation of mixed-precision decomposition is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model_int8.device)
    with torch.no_grad():
        out = model_int8.generate(**inputs, max_new_tokens=50, do_sample=False)
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"    Prompt: {prompt}")
    print(f"    Output: {generated[:200]}")

    result = {
        "mem_int8_mb": mem_int8,
        "model": model_int8,
        "tokenizer": tokenizer,
    }
    return result


# ============================================================================
# Section 4: Perplexity Showdown
# ============================================================================

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, sequences, device, label=""):
    """
    Evaluate perplexity on pre-tokenized sequences.
    Standard autoregressive perplexity: exp(mean NLL per token).
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    # For models loaded with device_map="auto", send input to the device
    # of the first parameter (embedding layer).
    input_device = next(model.parameters()).device

    for i, seq in enumerate(sequences):
        seq = seq.to(input_device)
        outputs = model(seq, labels=seq)
        nll = outputs.loss.item()

        n_tokens = seq.shape[1] - 1  # Exclude first token (no prediction)
        total_nll += nll * n_tokens
        total_tokens += n_tokens

        if (i + 1) % 16 == 0:
            running_ppl = np.exp(total_nll / total_tokens)
            print(f"    [{label}] {i+1}/{len(sequences)} sequences, running PPL: {running_ppl:.2f}")

    avg_nll = total_nll / total_tokens
    ppl = np.exp(avg_nll)
    return ppl


def run_perplexity_comparison(cfg: Config):
    """
    Experiment 4: The perplexity showdown.

    Compare three configurations on WikiText-2 test:
      1. FP16 baseline
      2. Naive per-channel INT8 (simulated — quantize all weights, no decomposition)
      3. LLM.int8() via bitsandbytes

    This is the climactic result: naive INT8 degrades; LLM.int8() matches FP16.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Perplexity Showdown — FP16 vs INT8 vs LLM.int8()")
    print("=" * 72)

    if cfg.device == "cpu" and cfg.model_key == "opt-6.7b":
        print("  Skipping perplexity (OPT-6.7B requires GPU)")
        return None

    model_name = cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eval_seqs = load_wikitext2_tokens(
        tokenizer, cfg.eval_samples, cfg.eval_seq_len, split="test"
    )

    results = {}

    # --- 1. FP16 Baseline ---
    print(f"\n[1/3] FP16 Baseline...")
    mem_before = gpu_mem_mb()
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=cfg.device,
    )
    mem_fp16 = gpu_mem_mb() - mem_before
    print(f"  GPU memory: {mem_fp16:.0f} MB")

    ppl_fp16 = evaluate_perplexity(model_fp16, tokenizer, eval_seqs, cfg.device, "FP16")
    print(f"  FP16 Perplexity: {ppl_fp16:.2f}")
    results["FP16"] = {"ppl": ppl_fp16, "mem_mb": mem_fp16}

    # --- 2. Naive INT8 W+A (simulated) ---
    # Quantize weights per-channel AND activations per-tensor at every Linear.
    # This simulates what real INT8 matmul hardware does without any outlier
    # handling.  The per-tensor activation scale is set by the 60–106× outlier
    # channels from Section 7.1, crushing the 99.9% of normal channels to
    # ~6 usable INT8 levels.  This is the scenario LLM.int8() was designed to fix.
    #
    # Weight-only per-channel INT8 (our previous bug) shows NO degradation
    # because weight distributions are well-behaved — it's the ACTIVATIONS
    # that have the 106× outliers.
    print(f"\n[2/3] Naive INT8 W+A (per-tensor activations, per-channel weights)...")
    model_naive = model_fp16  # Reuse the loaded model, modify in-place

    # Step 2a: Quantize weights per-channel (same as before, on CPU to save VRAM)
    n_quantized = 0
    for name, module in model_naive.named_modules():
        if isinstance(module, torch.nn.Linear):
            dev = module.weight.data.device
            W_cpu = module.weight.data.cpu()
            q, s = quantize_absmax_int8(W_cpu)
            W_deq = (q.float() * s).to(torch.float16)
            module.weight.data = W_deq.to(dev)
            n_quantized += 1

    print(f"  Quantized {n_quantized} Linear layer weights to simulated INT8")

    # Step 2b: Register forward pre-hooks for per-tensor activation quantization.
    # This is where the damage happens — one scale for the entire activation
    # tensor, dominated by the outlier channels.
    act_hooks = []

    def naive_act_quant_hook(module, args):
        """Per-tensor absmax quantize-dequantize on Linear input activations."""
        x = args[0]
        scale = x.abs().max() / 127.0                       # ONE scale for ALL dims
        scale = torch.clamp(scale, min=1e-8)
        x_q = torch.round(x / scale).clamp(-128, 127)
        x_deq = (x_q * scale).to(x.dtype)
        return (x_deq,) + args[1:]

    for name, module in model_naive.named_modules():
        if isinstance(module, torch.nn.Linear):
            h = module.register_forward_pre_hook(naive_act_quant_hook)
            act_hooks.append(h)

    print(f"  Registered {len(act_hooks)} activation quantization hooks")
    ppl_naive = evaluate_perplexity(model_naive, tokenizer, eval_seqs, cfg.device, "Naive INT8 W+A")
    print(f"  Naive INT8 W+A Perplexity: {ppl_naive:.2f}")
    results["Naive INT8"] = {"ppl": ppl_naive, "mem_mb": mem_fp16}

    # Remove hooks before freeing model
    for h in act_hooks:
        h.remove()

    del model_naive, model_fp16
    free_gpu()

    # --- 3. LLM.int8() via bitsandbytes ---
    if cfg.device != "cpu":
        print(f"\n[3/3] LLM.int8() (bitsandbytes)...")
        mem_before = gpu_mem_mb()

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        model_int8 = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
        mem_int8 = gpu_mem_mb() - mem_before
        print(f"  GPU memory: {mem_int8:.0f} MB")

        ppl_int8 = evaluate_perplexity(model_int8, tokenizer, eval_seqs, cfg.device, "LLM.int8()")
        print(f"  LLM.int8() Perplexity: {ppl_int8:.2f}")
        results["LLM.int8()"] = {"ppl": ppl_int8, "mem_mb": mem_int8}

        del model_int8
        free_gpu()
    else:
        print("\n[3/3] Skipping LLM.int8() (requires CUDA)")

    # --- Summary table ---
    print("\n" + "=" * 60)
    print("PERPLEXITY COMPARISON — WikiText-2 Test")
    print("=" * 60)
    print(f"{'Method':<20s} {'Perplexity':>12s} {'Δ vs FP16':>12s} {'Memory (MB)':>12s}")
    print("-" * 56)
    for method, data in results.items():
        delta = data["ppl"] - results["FP16"]["ppl"]
        delta_str = f"{delta:+.2f}" if method != "FP16" else "—"
        print(f"  {method:<18s} {data['ppl']:>10.2f}   {delta_str:>10s}   {data['mem_mb']:>10.0f}")

    return results


def plot_perplexity_comparison(results: dict, cfg: Config):
    """
    Figure 7.6: Perplexity comparison bar chart with memory overlay.
    Grayscale-safe with hatching patterns for B&W print.
    """
    if not results:
        return

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    methods = list(results.keys())
    ppls = [results[m]["ppl"] for m in methods]
    mems = [results[m]["mem_mb"] / 1024 for m in methods]  # GB

    method_key_map = {"FP16": "baseline", "Naive INT8": "primary",
                      "LLM.int8()": "secondary"}
    hatch_map = {"FP16": "", "Naive INT8": "//", "LLM.int8()": "xx"}

    bars = ax1.bar(methods, ppls, width=0.5, edgecolor="black", linewidth=0.8)
    for bar, m in zip(bars, methods):
        key = method_key_map.get(m, "baseline")
        bar.set_facecolor(COLORS.get(key, "#999999"))
        bar.set_hatch(hatch_map.get(m, ""))

    # Value labels on bars
    for bar, ppl in zip(bars, ppls):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{ppl:.2f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")

    ax1.set_ylabel("Perplexity (\u2193 is better)")
    ax1.set_title("WikiText-2 Perplexity: FP16 vs Naive INT8 (W+A) vs LLM.int8()")

    # FP16 baseline reference line
    fp16_ppl = results["FP16"]["ppl"]
    ax1.axhline(fp16_ppl, color="#333333", linestyle="--", linewidth=0.8, alpha=0.4)
    ax1.text(len(methods) - 0.5, fp16_ppl + 0.15, "FP16 baseline",
             fontsize=8, color="#555555")

    # Memory annotations inside bars
    for i, (m, mem) in enumerate(zip(methods, mems)):
        ax1.text(i, ppls[i] * 0.05, f"{mem:.1f} GB", ha="center", va="bottom",
                 fontsize=9, color="white", fontweight="bold")

    ax1.set_ylim(0, max(ppls) * 1.15)

    save_or_show(fig, "fig7_6_perplexity_showdown", cfg)


# ============================================================================
# Section 5: Combined Visualization — Decomposition Flow Diagram
# ============================================================================

def plot_decomposition_flow(cfg: Config):
    """
    Figure 7.3b (conceptual): A schematic showing the LLM.int8() data flow.
    X → detect outliers → split → FP16 path + INT8 path → sum → output

    Grayscale-safe: uses fill intensity + border style to distinguish paths.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("LLM.int8() Mixed-Precision Decomposition Flow",
                 fontsize=12, pad=20)

    def box(x, y, w, h, text, facecolor, edgecolor="black",
            linestyle="-", linewidth=1.2, fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=facecolor, edgecolor=edgecolor,
            linewidth=linewidth, linestyle=linestyle,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2, text="", color="black", linestyle="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5,
                                    linestyle=linestyle))
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.15, text, fontsize=8, ha="center",
                    color=color, style="italic")

    # Input
    box(0.2, 2, 1.4, 1, "X\n[batch, d]", "#e8e8e8", fontsize=10)

    # Detect
    arrow(1.6, 2.5, 2.6, 2.5)
    box(2.6, 1.8, 1.8, 1.4, "Detect\nOutliers\n|x| > 6.0", "#ccdcec", fontsize=9)

    # Split arrows — dashed for outlier, solid for normal
    arrow(4.4, 3.0, 5.4, 3.8, "outlier cols", "#333333", linestyle="--")
    arrow(4.4, 2.0, 5.4, 1.2, "normal cols", "#333333")

    # FP16 path — white fill, dashed border
    box(5.4, 3.3, 2.0, 1.0, "FP16 Matmul\n(\u223c0.1% dims)", "#ffffff",
        edgecolor="#1B5299", linestyle="--", linewidth=1.5, fontsize=9)

    # INT8 path — blue fill, solid border
    box(5.4, 0.7, 2.0, 1.0, "INT8 Matmul\n(\u223c99.9% dims)", "#b3cde3",
        edgecolor="#1B5299", linewidth=1.5, fontsize=9)

    # Merge arrows
    arrow(7.4, 3.8, 8.6, 2.8, "", "#333333", linestyle="--")
    arrow(7.4, 1.2, 8.6, 2.2, "", "#333333")

    # Sum
    box(8.6, 1.8, 1.2, 1.4, "\u03a3\nSum", "#d9e6f2", fontsize=12)

    # Output
    arrow(9.8, 2.5, 10.8, 2.5)
    box(10.8, 2, 1.6, 1, "Output\n[batch, out]", "#e8e8e8", fontsize=10)

    # Annotations
    ax.text(6.4, 4.6, "Zero quantization error\non critical dimensions",
            fontsize=8, ha="center", color="#1B5299", style="italic")
    ax.text(6.4, 0.15,
            "Per-channel weights \u00d7 per-token activations\n"
            "3 orders of magnitude less error than per-tensor",
            fontsize=8, ha="center", color="#1B5299", style="italic")

    save_or_show(fig, "fig7_3b_decomposition_flow", cfg)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chapter 7.2 — LLM.int8() Mixed-Precision Decomposition"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "decomposition", "threshold", "bitsandbytes", "perplexity", "flow"],
        help="Which experiment(s) to run",
    )
    parser.add_argument("--model", type=str, default="opt-6.7b", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--device", type=str, default=None, help="Force device (cuda/cpu)")
    parser.add_argument("--save-plots", action="store_true", help="Save figures to disk")
    parser.add_argument("--plot-dir", type=str, default="ch7_plots")
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--calib-samples", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config(
        model_key=args.model,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        save_plots=args.save_plots,
        plot_dir=args.plot_dir,
        eval_samples=args.eval_samples,
        calib_samples=args.calib_samples,
    )

    set_seed(cfg.seed)
    setup_manning_style()

    print(f"Configuration:")
    print(f"  Model:  {cfg.model_name}")
    print(f"  Device: {cfg.device}")
    print(f"  GPU:    {torch.cuda.get_device_name() if cfg.device == 'cuda' else 'N/A'}")
    if cfg.device == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  VRAM:   {total_mem:.0f} MB")
    print()

    mode = args.mode

    # Experiment 1: Manual decomposition
    if mode in ("all", "decomposition"):
        layer_results = run_decomposition_experiment(cfg)
        plot_decomposition_anatomy(layer_results, cfg)

    # Experiment 2: Threshold sweep
    if mode in ("all", "threshold"):
        sweep_results = run_threshold_sweep(cfg)
        plot_threshold_sweep(sweep_results, cfg)

    # Flow diagram (no model needed)
    if mode in ("all", "flow"):
        plot_decomposition_flow(cfg)

    # Experiment 4: Perplexity showdown (run BEFORE bitsandbytes —
    # perplexity loads/frees 3 models sequentially and needs max headroom)
    if mode in ("all", "perplexity"):
        ppl_results = run_perplexity_comparison(cfg)
        plot_perplexity_comparison(ppl_results, cfg)

    # Experiment 3: bitsandbytes (run LAST — bnb Linear8bitLt layers
    # hold CUDA memory that is difficult to fully reclaim)
    if mode in ("all", "bitsandbytes"):
        bnb_result = run_bitsandbytes_int8(cfg)
        if bnb_result and "model" in bnb_result:
            del bnb_result["model"]
        del bnb_result
        free_gpu()

    print("\n" + "=" * 72)
    print("All experiments complete.")
    if cfg.save_plots:
        print(f"Figures saved to: {cfg.plot_dir}/")
        print(f"  Each figure saved as .pdf (vector, selectable text) and .png (raster)")
    print("=" * 72)


if __name__ == "__main__":
    main()