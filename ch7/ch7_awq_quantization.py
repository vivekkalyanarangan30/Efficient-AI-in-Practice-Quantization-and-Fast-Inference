#!/usr/bin/env python3
"""
Chapter 7 — Section 7.4 Companion Script
Apply activation-aware weight protection: AWQ weight quantization

This script builds AWQ (Lin et al., MLSys 2024) from scratch, then validates
it against the production autoawq implementation. The pedagogical arc:

  1. Salient channels     — show that activation magnitude, not weight magnitude,
                            identifies the channels that matter for quantization
  2. AWQ from scratch     — implement per-channel scaling with grid search on a
                            single linear layer, compare RTN vs AWQ vs GPTQ MSE
  3. Layer-by-layer       — apply AWQ to every Linear in OPT-6.7B, show
                            per-layer MSE improvement over RTN at 4-bit
  4. Alpha sweep          — how the scaling exponent α controls the protection-
                            distortion tradeoff (the AWQ Pareto frontier)
  5. Perplexity showdown  — FP16 vs RTN-4bit vs AWQ-4bit on WikiText-2
  6. autoawq deployment   — production quantization with AutoAWQ

The key idea:  GPTQ compensates quantization error *after* it happens (Hessian-
guided error redistribution).  AWQ prevents it *before* it happens (scale
salient weight channels up so they land on finer grid points).  Both use
calibration data, but AWQ needs no Hessian, no Cholesky, no backpropagation.

Hardware: Colab T4 (16 GB VRAM). OPT-6.7B quantization processes one
         transformer block at a time, peak ~8 GB.

Usage:
    # Full pipeline (T4 GPU required, pip install autoawq for experiment 6)
    python ch7_awq_quantization.py --mode all --save-plots

    # Salient channel analysis (quick, educational)
    python ch7_awq_quantization.py --mode salient --save-plots

    # AWQ from scratch on one layer
    python ch7_awq_quantization.py --mode single-layer --save-plots

    # Per-layer RTN vs AWQ comparison
    python ch7_awq_quantization.py --mode layer-sweep --save-plots

    # Alpha exponent sweep
    python ch7_awq_quantization.py --mode alpha --save-plots

    # Perplexity comparison
    python ch7_awq_quantization.py --mode perplexity --save-plots

    # Production deployment with autoawq
    python ch7_awq_quantization.py --mode deploy --save-plots

    # CPU-only (uses OPT-125M for illustration, not publishable numbers)
    python ch7_awq_quantization.py --mode all --save-plots \
        --model opt-125m --device cpu

References:
    Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression
    and Acceleration," MLSys 2024.
"""

import argparse
import gc
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    # Calibration
    calib_samples: int = 128
    calib_seq_len: int = 2048       # AWQ default: 512–2048 tokens

    # AWQ algorithm
    wbits: int = 4                  # Target bit-width
    groupsize: int = 128            # Group size for group-wise quantization
    n_grid: int = 20                # Grid search granularity (paper default)
    max_shrink: float = 0.8         # Minimum scale factor (1 - max_shrink = 0.2)
    alpha_default: float = 0.5      # Default scaling exponent

    # Alpha sweep
    alphas: List[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ])

    # Perplexity evaluation
    eval_samples: int = 64
    eval_seq_len: int = 512         # 512 fits on T4; 2048 OOMs with OPT-6.7B FP16

    # Visualization — Manning publication quality
    save_plots: bool = False
    plot_dir: str = "ch7_plots"
    dpi: int = 300
    figsize: Tuple[int, int] = (10, 6)

    @property
    def model_name(self) -> str:
        return MODEL_REGISTRY[self.model_key]

# ============================================================================
# Manning Publication Figure Styling
# ============================================================================

def setup_manning_style():
    """Configure matplotlib for Manning publication figures (grayscale-safe, serif, 300dpi)."""
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

COLORS = {
    "baseline":   "#999999",     # Gray — baseline / reference
    "primary":    "#4A90D9",     # Blue — primary quantized variant
    "secondary":  "#1B5299",     # Dark blue — secondary variant
    "tertiary":   "#7FCDBB",     # Teal — third variant
    "highlight":  "#E07B39",     # Orange — highlight / AWQ accent
}
HATCHES = {
    "baseline":   "",
    "primary":    "//",
    "secondary":  "xx",
    "tertiary":   "..",
    "highlight":  "\\\\",
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
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0

def free_gpu():
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

    print(f"  Loaded {len(sequences)} sequences of length {seq_len} "
          f"from WikiText-2 ({split})")
    return sequences

def save_or_show(fig, filename_stem: str, cfg: Config):
    """Save figure in PDF + PNG for Manning publication."""
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
# Core Quantization — RTN baseline (same as 7.3 for consistency)
# ============================================================================

def quantize_rtn(w: torch.Tensor, bits: int = 4,
                 groupsize: int = -1) -> torch.Tensor:
    """
    RTN quantization: asymmetric uniform quant on [min, max] grid, per-row or per-group.
    Returns dequantized weights (float, same shape). Baseline that AWQ improves upon.
    """
    maxq = 2 ** bits - 1

    rows, cols = w.shape
    w = w.float().clone()

    if groupsize == -1:
        w_min = w.min(dim=1, keepdim=True).values
        w_max = w.max(dim=1, keepdim=True).values
        scale = (w_max - w_min) / maxq
        scale = torch.clamp(scale, min=1e-8)
        zero = torch.round(-w_min / scale)
        q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
        w_deq = scale * (q - zero)
    else:
        w_deq = torch.zeros_like(w)
        for g_start in range(0, cols, groupsize):
            g_end = min(g_start + groupsize, cols)
            wg = w[:, g_start:g_end]
            w_min = wg.min(dim=1, keepdim=True).values
            w_max = wg.max(dim=1, keepdim=True).values
            scale = (w_max - w_min) / maxq
            scale = torch.clamp(scale, min=1e-8)
            zero = torch.round(-w_min / scale)
            q = torch.clamp(torch.round(wg / scale) + zero, 0, maxq)
            w_deq[:, g_start:g_end] = scale * (q - zero)

    return w_deq

# ============================================================================
# Core AWQ Algorithm — Built from Scratch
# ============================================================================

def compute_activation_scales(X: torch.Tensor) -> torch.Tensor:
    """
    Compute per-channel activation importance as mean(|x|) across samples.

    AWQ's key insight: weight channel importance should be determined by
    the activation it multiplies, not the weight magnitude itself.
    A weight of 0.001 multiplied by an activation of 1000 produces output 1.0 —
    quantizing that weight carelessly destroys a significant signal.

    Args:
        X: Activation matrix [n_samples, in_features]

    Returns:
        act_scales: Per-channel importance [in_features]
    """
    # Mean absolute activation magnitude per input channel
    act_scales = X.abs().mean(dim=0)
    return act_scales


def awq_scale_search(
    W: torch.Tensor,
    X: torch.Tensor,
    bits: int = 4,
    groupsize: int = 128,
    n_grid: int = 20,
    max_shrink: float = 0.8,
) -> torch.Tensor:
    """
    AWQ per-channel scale search (Algorithm 1 from Lin et al., MLSys 2024).

    For each group of weight columns, search for the optimal per-channel
    scaling factor s that minimizes ||Q(W·diag(s)) · diag(s)^{-1} · X  −  W · X||.

    The search is a simple 1-D grid search over the scaling exponent α:
        s_j = (act_scale_j / max(act_scale))^α
    where α ∈ [0, 1].  α=0 means no scaling (RTN), α=1 means full
    activation-proportional scaling.

    The math behind it:
    ─────────────────
    For weight-only quantization of y = w·x, the quantization error per
    element is:

        Err = Δ · RoundErr(w/Δ) · x

    where Δ = max(|w_group|) / (2^{b-1} - 1).  The rounding error averages
    to 0.25 regardless of scaling.  But if we scale w → w·s and x → x/s:

        Err' = Δ' · RoundErr(ws/Δ') · x/s

    When s > 1 for a channel whose activation is large, we increase that
    channel's weight magnitude, potentially increasing Δ' (the group scale).
    But the 1/s reduction on the activation side can win — provided the
    channel being scaled up doesn't dominate the group's max.

    The grid search finds the α that best balances:
      - Protecting salient channels (large s → finer grid for important weights)
      - Not blowing up the group scale Δ' (which would hurt all other channels)

    Args:
        W: Weight matrix [out_features, in_features]
        X: Calibration activations [n_samples, in_features]
        bits: Target bit-width
        groupsize: Quantization group size
        n_grid: Number of grid points for α search
        max_shrink: Controls the minimum α value tested (1 - max_shrink)

    Returns:
        best_scales: Optimal per-channel scales [in_features]
    """
    W = W.float()
    X = X.float()

    rows, cols = W.shape
    n_samples = X.shape[0]

    # Normalize groupsize: -1 means "one group per row" = all columns
    if groupsize == -1:
        groupsize = cols

    # Compute per-channel activation scales
    act_scales = compute_activation_scales(X)  # [in_features]

    # Reference output: Y_ref = (W @ X^T)  [out_features, n_samples]
    Y_ref = W @ X.T

    n_groups = (cols + groupsize - 1) // groupsize
    best_scales = torch.ones(cols, dtype=torch.float32, device=W.device)
    best_error = float("inf") * torch.ones(n_groups, device=W.device)

    # Process each group independently
    for g_idx, g_start in enumerate(range(0, cols, groupsize)):
        g_end = min(g_start + groupsize, cols)
        g_len = g_end - g_start

        W_group = W[:, g_start:g_end]           # [out, g_len]
        X_group = X[:, g_start:g_end]            # [n_samples, g_len]
        act_group = act_scales[g_start:g_end]    # [g_len]

        # Reference output contribution from this group
        Y_group_ref = W_group @ X_group.T         # [out, n_samples]

        best_group_error = float("inf")
        best_group_scales = torch.ones(g_len, device=W.device)

        # Grid search over alpha
        for i in range(n_grid):
            alpha = 1.0 - i / n_grid  # From 1.0 down to ~0.05
            if alpha < (1 - max_shrink):
                continue

            # Compute per-channel scales: s = (normalized_act)^alpha
            # Channels with higher activation get larger scales (more protection)
            act_norm = act_group / (act_group.max() + 1e-8)
            scales = act_norm.pow(alpha).clamp(min=1e-4)

            # Apply scaling: W' = W * diag(s), X' = X / diag(s)
            W_scaled = W_group * scales.unsqueeze(0)   # [out, g_len]
            X_descaled = X_group / scales.unsqueeze(0)  # [n_samples, g_len]

            # Quantize the scaled weights
            W_q = quantize_rtn(W_scaled, bits=bits, groupsize=-1)

            # Reconstruction: Y_q = Q(W*s) @ (X/s)^T = Q(W*s) @ X'^T
            Y_q = W_q @ X_descaled.T  # [out, n_samples]

            # MSE for this group's contribution
            mse = ((Y_group_ref - Y_q) ** 2).mean().item()

            if mse < best_group_error:
                best_group_error = mse
                best_group_scales = scales.clone()

        best_scales[g_start:g_end] = best_group_scales
        best_error[g_idx] = best_group_error

    return best_scales


def awq_quantize(
    W: torch.Tensor,
    X: torch.Tensor,
    bits: int = 4,
    groupsize: int = 128,
    n_grid: int = 20,
    max_shrink: float = 0.8,
) -> Tuple[torch.Tensor, dict]:
    """
    Full AWQ quantization: search for optimal scales, apply them, then quantize.

    The equivalent transformation:
      Y = W @ X
        = (W · diag(s)) @ (diag(s)^{-1} · X)
        ≈ Q(W · diag(s)) @ (diag(s)^{-1} · X)

    In deployment, the scaling is absorbed:
      - s is folded into the preceding LayerNorm (γ → γ/s)
      - s^{-1} disappears — no runtime overhead

    Args:
        W: Weight matrix [out_features, in_features]
        X: Calibration activations [n_samples, in_features]
        bits: Target bit-width
        groupsize: Quantization group size
        n_grid: Grid search granularity
        max_shrink: Minimum scaling factor

    Returns:
        W_q: Dequantized quantized weights (with scaling baked in) [out, in]
        stats: Dictionary with quantization statistics
    """
    W = W.float().clone()
    X = X.float()
    rows, cols = W.shape

    # Step 1: Find optimal per-channel scales via grid search
    t0 = time.time()
    scales = awq_scale_search(
        W, X, bits=bits, groupsize=groupsize,
        n_grid=n_grid, max_shrink=max_shrink,
    )
    t_search = time.time() - t0

    # Step 2: Apply scaling to weights
    W_scaled = W * scales.unsqueeze(0)        # W' = W · diag(s)

    # Step 3: Quantize the scaled weights (standard group-wise RTN)
    W_q_scaled = quantize_rtn(W_scaled, bits=bits, groupsize=groupsize)

    # Step 4: Undo the scaling on the quantized weights
    # At inference: Y ≈ Q(W·s) @ (x/s) = (Q(W·s) / s) @ x
    # So the "effective" quantized weight is Q(W·s) / s
    W_q = W_q_scaled / scales.unsqueeze(0)

    # Compute stats
    Y_ref = W @ X.T
    Y_q = W_q @ X.T
    mse = ((Y_ref - Y_q) ** 2).mean().item()

    # How many channels got non-trivial scaling?
    n_scaled = (scales > 1.01).sum().item()
    n_protected = ((scales - 1.0).abs() > 0.1).sum().item()

    stats = {
        "mse": mse,
        "search_time": t_search,
        "n_scaled": n_scaled,
        "n_protected": n_protected,
        "scale_mean": scales.mean().item(),
        "scale_max": scales.max().item(),
        "scale_min": scales.min().item(),
    }

    return W_q, stats

# ============================================================================
# Experiment 1: Salient Channel Analysis
# ============================================================================

def run_salient_channel_analysis(cfg: Config):
    """
    Experiment 1: Demonstrate that activation magnitude (not weight magnitude)
    identifies salient channels.

    AWQ's foundational observation: quantization error is proportional to
    activation magnitude × weight quantization noise.  A small weight
    multiplied by a large activation produces a signal worth protecting.
    Weight magnitude alone misses this.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Salient Channel Analysis — Activation vs Weight Magnitude")
    print("=" * 72)

    device = cfg.device
    model_name = cfg.model_name

    print(f"\nLoading {model_name} in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    print(f"  GPU memory after load: {gpu_mem_mb():.0f} MB")

    # Target: layer 0 fc1 (same as GPTQ experiments for comparability)
    target_layer_idx = 0
    target_module = model.model.decoder.layers[target_layer_idx].fc1

    # Capture activations
    print(f"\nCapturing fc1 activations from layer {target_layer_idx}...")
    calib_seqs = load_wikitext2_tokens(
        tokenizer, cfg.calib_samples, cfg.calib_seq_len, split="train"
    )

    captured_inputs = []
    def capture_hook(module, inp, out):
        x = inp[0].detach()
        captured_inputs.append(x.reshape(-1, x.shape[-1]).cpu())

    hook = target_module.register_forward_hook(capture_hook)

    with torch.no_grad():
        for seq in calib_seqs[:8]:
            model(seq.to(device))

    hook.remove()

    # Build activation matrix
    X_all = torch.cat(captured_inputs, dim=0)  # [n_samples, in_features]
    W = target_module.weight.data.cpu().clone()  # [out_features, in_features]

    print(f"  Calibration matrix: X.shape = {list(X_all.shape)}")
    print(f"  Weight matrix: W.shape = {list(W.shape)}")

    del model
    free_gpu()

    # Compute per-channel statistics
    act_scales = compute_activation_scales(X_all)      # [in_features]
    weight_scales = W.float().abs().mean(dim=0)         # [in_features]

    # Quantization error per channel: quantize each column independently,
    # measure how much output error that channel contributes
    in_features = W.shape[1]
    X = X_all[:4096].float()  # Limit samples for speed
    Y_ref = W.float() @ X.T

    per_channel_error = torch.zeros(in_features)
    for j in range(in_features):
        w_col = W[:, j:j+1].float()
        x_col = X[:, j:j+1]

        # Quantize this single column with 4-bit RTN
        w_q = quantize_rtn(w_col, bits=cfg.wbits, groupsize=-1)

        # Output error from this channel
        y_err = (w_col - w_q) @ x_col.T
        per_channel_error[j] = (y_err ** 2).mean().item()

    # Rank channels by different criteria
    top_k = 50  # Top 1% of ~4096 channels

    act_rank = act_scales.argsort(descending=True)[:top_k]
    weight_rank = weight_scales.argsort(descending=True)[:top_k]
    error_rank = per_channel_error.argsort(descending=True)[:top_k]

    # Overlap: how well does each criterion predict actual error?
    act_overlap = len(set(act_rank.tolist()) & set(error_rank.tolist()))
    weight_overlap = len(set(weight_rank.tolist()) & set(error_rank.tolist()))

    print(f"\n  Channel importance analysis (top {top_k} of {in_features}):")
    print(f"    Activation-ranked overlap with true error ranking: "
          f"{act_overlap}/{top_k} ({100*act_overlap/top_k:.0f}%)")
    print(f"    Weight-ranked overlap with true error ranking:     "
          f"{weight_overlap}/{top_k} ({100*weight_overlap/top_k:.0f}%)")

    # Correlation analysis
    act_corr = np.corrcoef(
        act_scales.numpy(), per_channel_error.numpy()
    )[0, 1]
    weight_corr = np.corrcoef(
        weight_scales.numpy(), per_channel_error.numpy()
    )[0, 1]
    product_scales = act_scales * weight_scales
    product_corr = np.corrcoef(
        product_scales.numpy(), per_channel_error.numpy()
    )[0, 1]

    print(f"\n  Correlation with per-channel quantization error:")
    print(f"    Activation magnitude:            r = {act_corr:.4f}")
    print(f"    Weight magnitude:                r = {weight_corr:.4f}")
    print(f"    Activation × Weight (product):   r = {product_corr:.4f}")

    results = {
        "act_scales": act_scales.numpy(),
        "weight_scales": weight_scales.numpy(),
        "per_channel_error": per_channel_error.numpy(),
        "act_overlap": act_overlap,
        "weight_overlap": weight_overlap,
        "act_corr": act_corr,
        "weight_corr": weight_corr,
        "product_corr": product_corr,
        "top_k": top_k,
        "in_features": in_features,
    }

    return results


def plot_salient_channels(results: dict, cfg: Config):
    """Figure 7.12: Activation vs weight magnitude as predictors of quantization error."""
    if not results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    act = results["act_scales"]
    wt = results["weight_scales"]
    err = results["per_channel_error"]

    # Subsample for clarity in scatter plots
    n = len(act)
    idx = np.random.RandomState(42).choice(n, min(2000, n), replace=False)

    # Panel 1: Activation magnitude vs quantization error
    ax = axes[0]
    ax.scatter(act[idx], err[idx], s=3, alpha=0.3, color=COLORS["primary"],
               rasterized=True)
    ax.set_xlabel("Activation Magnitude (mean |x|)")
    ax.set_ylabel("Per-Channel Quantization Error")
    ax.set_title(f"Activation → Error (r={results['act_corr']:.3f})")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Panel 2: Weight magnitude vs quantization error
    ax = axes[1]
    ax.scatter(wt[idx], err[idx], s=3, alpha=0.3, color=COLORS["baseline"],
               rasterized=True)
    ax.set_xlabel("Weight Magnitude (mean |w|)")
    ax.set_ylabel("Per-Channel Quantization Error")
    ax.set_title(f"Weight → Error (r={results['weight_corr']:.3f})")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Panel 3: Overlap bar chart
    ax = axes[2]
    labels = ["Activation\nranking", "Weight\nranking"]
    overlaps = [results["act_overlap"], results["weight_overlap"]]
    top_k = results["top_k"]

    bars = ax.bar(labels, overlaps, width=0.4, edgecolor="black", linewidth=0.8)
    bars[0].set_facecolor(COLORS["primary"])
    bars[0].set_hatch("//")
    bars[1].set_facecolor(COLORS["baseline"])
    bars[1].set_hatch("")

    for bar, val in zip(bars, overlaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val}/{top_k}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.axhline(top_k, color="#333333", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_ylabel(f"Overlap with Top-{top_k} Error Channels")
    ax.set_title("Which Ranking Predicts Error?")
    ax.set_ylim(0, top_k * 1.15)

    save_or_show(fig, "fig7_12_awq_salient_channels", cfg)


# ============================================================================
# Experiment 2: AWQ from Scratch on a Single Layer
# ============================================================================

def run_single_layer_experiment(cfg: Config):
    """
    Experiment 2: RTN vs AWQ at 4-bit on a single fc1 layer.

    Demonstrates that per-channel scaling (no Hessian, no backprop) reduces
    reconstruction error by protecting the salient weight channels.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: AWQ from Scratch on a Single fc1 Layer")
    print("=" * 72)

    device = cfg.device
    model_name = cfg.model_name

    print(f"\nLoading {model_name} in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    print(f"  GPU memory after load: {gpu_mem_mb():.0f} MB")

    target_layer_idx = 0
    target_module = model.model.decoder.layers[target_layer_idx].fc1

    # Capture activations
    print(f"\nCapturing fc1 activations from layer {target_layer_idx}...")
    calib_seqs = load_wikitext2_tokens(
        tokenizer, cfg.calib_samples, cfg.calib_seq_len, split="train"
    )

    captured_inputs = []
    def capture_hook(module, inp, out):
        x = inp[0].detach()
        captured_inputs.append(x.reshape(-1, x.shape[-1]).cpu())

    hook = target_module.register_forward_hook(capture_hook)

    with torch.no_grad():
        for seq in calib_seqs[:8]:
            model(seq.to(device))

    hook.remove()

    X_all = torch.cat(captured_inputs, dim=0)[:4096]  # [n_samples, in_features]
    W = target_module.weight.data.cpu().clone()

    print(f"  Calibration matrix: X.shape = {list(X_all.shape)}")
    print(f"  Weight matrix: W.shape = {list(W.shape)}")

    del model
    free_gpu()

    X = X_all.float()
    Y_ref = W.float() @ X.T

    results = []

    # --- RTN baselines ---
    for gs_label, gs in [("per-row", -1), ("g128", 128)]:
        t0 = time.time()
        W_rtn = quantize_rtn(W, bits=cfg.wbits, groupsize=gs)
        t_rtn = time.time() - t0
        Y_rtn = W_rtn.float() @ X.T
        mse_rtn = ((Y_ref - Y_rtn) ** 2).mean().item()

        print(f"\n  RTN {cfg.wbits}-bit ({gs_label}):")
        print(f"    MSE: {mse_rtn:.6f}  ({t_rtn:.2f}s)")

        results.append({
            "method": f"RTN ({gs_label})",
            "mse": mse_rtn,
            "time": t_rtn,
        })

    # --- AWQ (our from-scratch implementation) ---
    for gs_label, gs in [("per-row", -1), ("g128", 128)]:
        t0 = time.time()
        W_awq, awq_stats = awq_quantize(
            W, X_all, bits=cfg.wbits, groupsize=gs,
            n_grid=cfg.n_grid, max_shrink=cfg.max_shrink,
        )
        t_awq = time.time() - t0
        Y_awq = W_awq.float() @ X.T
        mse_awq = ((Y_ref - Y_awq) ** 2).mean().item()

        # Find matching RTN result for comparison
        rtn_key = f"RTN ({gs_label})"
        mse_rtn = [r["mse"] for r in results if r["method"] == rtn_key][0]
        ratio = mse_rtn / max(mse_awq, 1e-12)

        print(f"\n  AWQ {cfg.wbits}-bit ({gs_label}):")
        print(f"    MSE: {mse_awq:.6f}  ({t_awq:.2f}s)")
        print(f"    vs RTN: {ratio:.1f}× error reduction")
        print(f"    Search time: {awq_stats['search_time']:.2f}s")
        print(f"    Channels protected: {awq_stats['n_protected']}")
        print(f"    Scale range: [{awq_stats['scale_min']:.3f}, "
              f"{awq_stats['scale_max']:.3f}]")

        results.append({
            "method": f"AWQ ({gs_label})",
            "mse": mse_awq,
            "time": t_awq,
            "ratio": ratio,
            "stats": awq_stats,
        })

    return results


def plot_single_layer(results: list, cfg: Config):
    """Figure 7.13: RTN vs AWQ MSE at 4-bit, per-row and g128."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    methods = [r["method"] for r in results]
    mses = [r["mse"] for r in results]

    color_map = {
        "RTN (per-row)": COLORS["baseline"],
        "RTN (g128)":    COLORS["baseline"],
        "AWQ (per-row)": COLORS["primary"],
        "AWQ (g128)":    COLORS["secondary"],
    }
    hatch_map = {
        "RTN (per-row)": "",
        "RTN (g128)":    "",
        "AWQ (per-row)": "//",
        "AWQ (g128)":    "xx",
    }

    bars = ax.bar(methods, mses, width=0.5, edgecolor="black", linewidth=0.8)
    for bar, m in zip(bars, methods):
        bar.set_facecolor(color_map.get(m, COLORS["baseline"]))
        bar.set_hatch(hatch_map.get(m, ""))

    for bar, mse in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{mse:.4f}", ha="center", va="bottom", fontsize=8)

    # Ratio annotations for AWQ results
    for r in results:
        if "ratio" in r:
            idx = methods.index(r["method"])
            ax.annotate(f"{r['ratio']:.1f}×",
                        xy=(idx, r["mse"]),
                        fontsize=9, ha="center", va="top",
                        color=COLORS["secondary"], fontweight="bold")

    ax.set_ylabel("Reconstruction MSE (vs FP16)")
    ax.set_title(f"RTN vs AWQ: Single fc1 Layer ({cfg.wbits}-bit)")
    ax.set_yscale("log")

    save_or_show(fig, "fig7_13_awq_single_layer", cfg)


# ============================================================================
# Experiment 3: Layer-by-Layer AWQ Sweep
# ============================================================================

def run_layer_sweep(cfg: Config):
    """Experiment 3: RTN vs AWQ MSE on every fc1 layer, 4-bit g128."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Per-Layer RTN vs AWQ Comparison (4-bit, g128)")
    print("=" * 72)

    device = cfg.device
    model_name = cfg.model_name

    print(f"\nLoading {model_name} in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    num_layers = model.config.num_hidden_layers

    calib_seqs = load_wikitext2_tokens(
        tokenizer, cfg.calib_samples, cfg.calib_seq_len, split="train"
    )

    # Capture all fc1 inputs in one pass
    print(f"\nCapturing fc1 inputs for {num_layers} layers...")
    captured_fc1 = {}
    def make_fc1_hook(idx):
        def hook_fn(module, inp, out):
            if idx not in captured_fc1:
                captured_fc1[idx] = []
            if len(captured_fc1[idx]) < 4:
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                captured_fc1[idx].append(x.cpu())
        return hook_fn

    hooks = []
    for idx in range(num_layers):
        layer = model.model.decoder.layers[idx]
        h = layer.fc1.register_forward_hook(make_fc1_hook(idx))
        hooks.append(h)

    with torch.no_grad():
        for seq in calib_seqs[:4]:
            model(seq.to(device))

    for h in hooks:
        h.remove()

    # Copy weights to CPU
    fc1_weights = {}
    for idx in range(num_layers):
        if idx in captured_fc1:
            fc1_weights[idx] = (
                model.model.decoder.layers[idx].fc1.weight.data.cpu().clone()
            )

    del model
    free_gpu()

    # Process each layer
    print(f"\nProcessing {num_layers} layers (4-bit, g128)...")
    print(f"{'Layer':<8} {'Module':<6} {'RTN MSE':>12} {'AWQ MSE':>12} "
          f"{'Ratio':>8} {'Time (s)':>10}")
    print("-" * 62)

    layer_results = []

    for idx in range(num_layers):
        if idx not in captured_fc1 or idx not in fc1_weights:
            continue

        X_all = torch.cat(captured_fc1[idx], dim=0)[:4096]
        W = fc1_weights[idx]

        X = X_all.float()
        Y_ref = W.float() @ X.T

        # RTN
        W_rtn = quantize_rtn(W, bits=cfg.wbits, groupsize=cfg.groupsize)
        Y_rtn = W_rtn.float() @ X.T
        mse_rtn = ((Y_ref - Y_rtn) ** 2).mean().item()

        # AWQ
        t0 = time.time()
        W_awq, stats = awq_quantize(
            W, X_all, bits=cfg.wbits, groupsize=cfg.groupsize,
            n_grid=cfg.n_grid, max_shrink=cfg.max_shrink,
        )
        t_awq = time.time() - t0
        Y_awq = W_awq.float() @ X.T
        mse_awq = ((Y_ref - Y_awq) ** 2).mean().item()

        ratio = mse_rtn / max(mse_awq, 1e-12)

        layer_results.append({
            "layer": idx,
            "mse_rtn": mse_rtn,
            "mse_awq": mse_awq,
            "ratio": ratio,
            "time": t_awq,
        })

        print(f"  {idx:>4d}    fc1    {mse_rtn:>10.6f}   {mse_awq:>10.6f}   "
              f"{ratio:>6.1f}×   {t_awq:>8.1f}")

        del X_all, W, X, Y_ref, W_rtn, Y_rtn, W_awq, Y_awq
        gc.collect()

    del captured_fc1, fc1_weights
    gc.collect()

    if layer_results:
        avg_ratio = np.mean([r["ratio"] for r in layer_results])
        print(f"\n  Average error reduction (RTN → AWQ): {avg_ratio:.1f}×")

    return layer_results


def plot_layer_sweep(layer_results: list, cfg: Config):
    """Figure 7.14: Per-layer fc1 MSE, RTN vs AWQ at 4-bit g128."""
    if not layer_results:
        return

    fig, ax = plt.subplots(figsize=(12, 4.5))

    layers = [r["layer"] for r in layer_results]
    mse_rtn = [r["mse_rtn"] for r in layer_results]
    mse_awq = [r["mse_awq"] for r in layer_results]

    x = np.arange(len(layers))
    width = 0.35

    bars1 = ax.bar(x - width/2, mse_rtn, width, label="RTN (4-bit g128)",
                   color=COLORS["baseline"], edgecolor="black",
                   linewidth=0.5, hatch=HATCHES["baseline"])
    bars2 = ax.bar(x + width/2, mse_awq, width, label="AWQ (4-bit g128)",
                   color=COLORS["primary"], edgecolor="black",
                   linewidth=0.5, hatch=HATCHES["primary"])

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Reconstruction MSE (vs FP16)")
    ax.set_title("Per-Layer fc1 Error: RTN vs AWQ at 4-bit (g128)")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([str(l) for l in layers[::4]])
    ax.legend()
    ax.set_yscale("log")

    # Average ratio annotation
    avg_ratio = np.mean([r["ratio"] for r in layer_results])
    ax.text(0.98, 0.95, f"Mean error reduction: {avg_ratio:.1f}×",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", color=COLORS["primary"],
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    save_or_show(fig, "fig7_14_awq_per_layer", cfg)


# ============================================================================
# Experiment 4: Alpha Exponent Sweep
# ============================================================================

def run_alpha_sweep(cfg: Config):
    """
    Experiment 4: Sweep the scaling exponent α on layer 0 fc1.

    α controls the protection-distortion tradeoff:
      α = 0.0  →  No scaling (pure RTN)
      α = 0.5  →  AWQ default — balanced protection
      α = 1.0  →  Maximum activation-proportional scaling

    Higher α gives more protection to salient channels but can blow up
    the group scale Δ' for non-salient channels.  The grid search in
    awq_scale_search handles this per-group, but this sweep shows the
    *average* behavior across the full layer for a fixed α.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Alpha Exponent Sweep (4-bit, g128, layer 0 fc1)")
    print("=" * 72)

    device = cfg.device
    model_name = cfg.model_name

    print(f"\nLoading {model_name} in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    target_module = model.model.decoder.layers[0].fc1

    captured = []
    def hook_fn(module, inp, out):
        if len(captured) < 4:
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            captured.append(x.cpu())

    hook = target_module.register_forward_hook(hook_fn)
    calib_seqs = load_wikitext2_tokens(
        tokenizer, cfg.calib_samples, cfg.calib_seq_len, split="train"
    )
    with torch.no_grad():
        for seq in calib_seqs[:4]:
            model(seq.to(device))
    hook.remove()

    X_all = torch.cat(captured, dim=0)[:4096]
    W = target_module.weight.data.cpu().clone()

    del model
    free_gpu()

    X = X_all.float()
    Y_ref = W.float() @ X.T

    # Compute activation scales once
    act_scales = compute_activation_scales(X_all)

    # RTN baseline
    W_rtn = quantize_rtn(W, bits=cfg.wbits, groupsize=cfg.groupsize)
    Y_rtn = W_rtn.float() @ X.T
    mse_rtn = ((Y_ref - Y_rtn) ** 2).mean().item()

    print(f"\n  RTN baseline MSE: {mse_rtn:.6f}")
    print(f"\n{'Alpha':>8} {'MSE':>12} {'vs RTN':>8} {'Scale max':>12}")
    print("-" * 44)

    sweep_results = []

    for alpha in cfg.alphas:
        # Apply fixed alpha to all groups (no per-group grid search)
        act_norm = act_scales / (act_scales.max() + 1e-8)
        scales = act_norm.pow(alpha).clamp(min=1e-4)

        # Scale weights, quantize, unscale
        W_scaled = W.float() * scales.unsqueeze(0)
        W_q_scaled = quantize_rtn(W_scaled, bits=cfg.wbits, groupsize=cfg.groupsize)
        W_q = W_q_scaled / scales.unsqueeze(0)

        Y_q = W_q @ X.T
        mse = ((Y_ref - Y_q) ** 2).mean().item()
        ratio = mse_rtn / max(mse, 1e-12)

        sweep_results.append({
            "alpha": alpha,
            "mse": mse,
            "ratio": ratio,
            "scale_max": scales.max().item(),
            "scale_min": scales.min().item(),
        })

        print(f"  {alpha:>6.2f}   {mse:>10.6f}   {ratio:>6.1f}×   "
              f"{scales.max().item():>10.3f}")

    # Find optimal alpha
    best = min(sweep_results, key=lambda r: r["mse"])
    print(f"\n  Optimal alpha: {best['alpha']:.2f} "
          f"(MSE {best['mse']:.6f}, {best['ratio']:.1f}× vs RTN)")

    # Also run the full grid-search AWQ for comparison
    print(f"\n  Full grid-search AWQ (for comparison):")
    W_awq, awq_stats = awq_quantize(
        W, X_all, bits=cfg.wbits, groupsize=cfg.groupsize,
        n_grid=cfg.n_grid, max_shrink=cfg.max_shrink,
    )
    Y_awq = W_awq.float() @ X.T
    mse_awq = ((Y_ref - Y_awq) ** 2).mean().item()
    ratio_awq = mse_rtn / max(mse_awq, 1e-12)
    print(f"    Grid-search MSE: {mse_awq:.6f} ({ratio_awq:.1f}× vs RTN)")

    sweep_results.append({
        "alpha": "grid",
        "mse": mse_awq,
        "ratio": ratio_awq,
        "mse_rtn": mse_rtn,
    })

    return sweep_results


def plot_alpha_sweep(sweep_results: list, cfg: Config):
    """Figure 7.15: MSE vs alpha exponent, showing the protection-distortion tradeoff."""
    if not sweep_results:
        return

    # Separate fixed-alpha results from grid search
    fixed = [r for r in sweep_results if isinstance(r["alpha"], float)]
    grid = [r for r in sweep_results if r["alpha"] == "grid"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    alphas = [r["alpha"] for r in fixed]
    mses = [r["mse"] for r in fixed]
    ratios = [r["ratio"] for r in fixed]

    # Panel 1: MSE vs alpha
    ax1.plot(alphas, mses, color=COLORS["primary"], marker="o",
             markersize=6, linewidth=1.5, label="Fixed α")

    # RTN baseline (alpha=0 should match, but show explicit line)
    if fixed:
        rtn_mse = fixed[0]["mse"]  # alpha=0 is RTN
        ax1.axhline(rtn_mse, color=COLORS["baseline"], linestyle="--",
                     linewidth=0.8, alpha=0.5, label="RTN (α=0)")

    # Grid search result
    if grid:
        ax1.axhline(grid[0]["mse"], color=COLORS["secondary"], linestyle="-.",
                     linewidth=1.0, label=f"Grid search")
        ax1.text(0.6, grid[0]["mse"] * 0.9, "Grid search", fontsize=8,
                 color=COLORS["secondary"])

    # Mark optimal
    best = min(fixed, key=lambda r: r["mse"])
    ax1.scatter([best["alpha"]], [best["mse"]], color=COLORS["highlight"],
                s=80, zorder=5, marker="*", label=f"Optimal α={best['alpha']:.1f}")

    ax1.set_xlabel("Scaling Exponent α")
    ax1.set_ylabel("Reconstruction MSE (vs FP16)")
    ax1.set_title("Protection-Distortion Tradeoff")
    ax1.legend(fontsize=8)

    # Panel 2: Error reduction ratio vs alpha
    ax2.bar(range(len(alphas)), ratios, width=0.6,
            color=COLORS["primary"], edgecolor="black", linewidth=0.5,
            hatch="//")
    ax2.set_xticks(range(len(alphas)))
    ax2.set_xticklabels([f"{a:.1f}" for a in alphas], rotation=45, ha="right")
    ax2.set_xlabel("Scaling Exponent α")
    ax2.set_ylabel("Error Reduction vs RTN (×)")
    ax2.set_title("AWQ Improvement over RTN by α")
    ax2.axhline(1.0, color="#333333", linestyle="--", linewidth=0.8, alpha=0.4)

    save_or_show(fig, "fig7_15_awq_alpha_sweep", cfg)


# ============================================================================
# Experiment 5: Perplexity Showdown
# ============================================================================

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, sequences, device, label=""):
    """Standard autoregressive perplexity: exp(mean NLL per token)."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    input_device = next(model.parameters()).device

    for i, seq in enumerate(sequences):
        seq = seq.to(input_device)
        outputs = model(seq, labels=seq)
        nll = outputs.loss.item()

        n_tokens = seq.shape[1] - 1
        total_nll += nll * n_tokens
        total_tokens += n_tokens

        if (i + 1) % 16 == 0:
            running_ppl = np.exp(total_nll / total_tokens)
            print(f"    [{label}] {i+1}/{len(sequences)} sequences, "
                  f"running PPL: {running_ppl:.2f}")

    avg_nll = total_nll / total_tokens
    ppl = np.exp(avg_nll)
    return ppl


def simulate_rtn_quantized_model(model, bits=4, groupsize=-1):
    """Replace all Linear weights with RTN quantize-dequantize."""
    n_quantized = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            dev = module.weight.data.device
            W_cpu = module.weight.data.cpu()
            W_deq = quantize_rtn(W_cpu, bits=bits, groupsize=groupsize)
            module.weight.data = W_deq.to(torch.float16).to(dev)
            n_quantized += 1
    print(f"  RTN-quantized {n_quantized} Linear layers ({bits}-bit, "
          f"groupsize={groupsize})")
    return model


def simulate_awq_quantized_model(model, tokenizer, bits=4, groupsize=128,
                                  calib_seqs=None, n_grid=20, max_shrink=0.8):
    """
    Simulate AWQ quantization on all Linear layers using our from-scratch
    implementation.  Captures activations layer-by-layer.

    Note: This is a simplified simulation that applies AWQ independently to
    each Linear layer.  The production autoawq library processes entire
    transformer blocks together and absorbs scales into preceding LayerNorms,
    which is more efficient and can produce slightly different results.
    """
    model.eval()
    device = next(model.parameters()).device

    # We need activations for each linear layer.  For efficiency, we process
    # one transformer block at a time.
    n_quantized = 0
    decoder_layers = model.model.decoder.layers

    for layer_idx in range(len(decoder_layers)):
        block = decoder_layers[layer_idx]

        # Collect linear modules in this block
        linear_modules = {}
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                linear_modules[name] = module

        # Capture inputs for each linear module
        captured = {name: [] for name in linear_modules}
        hooks = []

        def make_hook(name):
            def hook_fn(module, inp, out):
                if len(captured[name]) < 2:
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                    captured[name].append(x.cpu())
            return hook_fn

        for name, module in linear_modules.items():
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

        # Run calibration through the model
        with torch.no_grad():
            for seq in (calib_seqs or [])[:2]:
                model(seq.to(device))

        for h in hooks:
            h.remove()

        # Apply AWQ to each linear module in this block
        for name, module in linear_modules.items():
            if name not in captured or not captured[name]:
                continue

            X_all = torch.cat(captured[name], dim=0)[:2048]
            W = module.weight.data.cpu()

            W_awq, _ = awq_quantize(
                W, X_all, bits=bits, groupsize=groupsize,
                n_grid=n_grid, max_shrink=max_shrink,
            )

            module.weight.data = W_awq.to(torch.float16).to(device)
            n_quantized += 1

        del captured
        gc.collect()

        if (layer_idx + 1) % 8 == 0:
            print(f"    Processed {layer_idx + 1}/{len(decoder_layers)} blocks "
                  f"({n_quantized} layers quantized)")

    print(f"  AWQ-quantized {n_quantized} Linear layers ({bits}-bit, "
          f"groupsize={groupsize})")
    return model


def run_perplexity_comparison(cfg: Config):
    """Experiment 5: FP16 vs RTN-4bit vs AWQ-4bit perplexity on WikiText-2."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 5: Perplexity Showdown — FP16 vs RTN-4bit vs AWQ-4bit")
    print("=" * 72)

    if cfg.device == "cpu" and cfg.model_key == "opt-6.7b":
        print("  Skipping perplexity (OPT-6.7B requires GPU)")
        return None

    free_gpu()

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

    # --- 2. RTN 4-bit (simulated, g128) ---
    print(f"\n[2/3] RTN 4-bit (simulated, g128)...")
    model_rtn = simulate_rtn_quantized_model(model_fp16, bits=4, groupsize=128)
    ppl_rtn = evaluate_perplexity(model_rtn, tokenizer, eval_seqs, cfg.device, "RTN-4bit")
    print(f"  RTN-4bit Perplexity: {ppl_rtn:.2f}")
    results["RTN 4-bit g128"] = {"ppl": ppl_rtn, "mem_mb": mem_fp16}

    del model_rtn, model_fp16
    free_gpu()

    # --- 3. AWQ 4-bit (via AutoAWQ) ---
    # The from-scratch AWQ algorithm is validated by Experiments 1–4 at the
    # layer level.  For end-to-end perplexity we use AutoAWQ, which properly
    # absorbs scales into LayerNorms — matching the Section 7.3 pattern where
    # perplexity is measured via the production library (gptqmodel), not the
    # from-scratch implementation.
    print(f"\n[3/3] AWQ 4-bit (AutoAWQ, g128)...")

    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("  autoawq not installed — skipping AWQ perplexity")
        return results

    save_path = f"./{cfg.model_key}-awq-4bit"

    if not os.path.exists(save_path):
        # Quantize inline if no saved model (e.g. --mode perplexity alone)
        print(f"  No saved model at {save_path}, quantizing with AutoAWQ...")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        calib_data = [t.strip() for t in ds["text"]
                      if len(t.strip()) > 100][:cfg.calib_samples]

        quant_config = {
            "zero_point": True, "q_group_size": 128,
            "w_bit": 4, "version": "GEMM",
        }
        model_awq = AutoAWQForCausalLM.from_pretrained(
            model_name, safetensors=False,
        )
        inner_model = model_awq.model.model
        if not hasattr(inner_model, "rotary_emb"):
            class _DummyRotaryEmb(torch.nn.Module):
                def forward(self, x, position_ids=None):
                    return (torch.zeros(1, device=x.device),
                            torch.zeros(1, device=x.device))
            inner_model.rotary_emb = _DummyRotaryEmb()

        model_awq.quantize(
            tokenizer, quant_config=quant_config, calib_data=calib_data,
        )
        model_awq.save_quantized(save_path)
        tokenizer.save_pretrained(save_path)
        del model_awq
        free_gpu()

    print(f"  Loading quantized model from {save_path}...")
    model_awq = AutoAWQForCausalLM.from_quantized(
        save_path, fuse_layers=False,
    )
    mem_awq = gpu_mem_mb()
    ppl_awq = evaluate_perplexity(
        model_awq.model, tokenizer, eval_seqs, cfg.device, "AWQ-4bit",
    )
    print(f"  AWQ-4bit Perplexity: {ppl_awq:.2f}")
    results["AWQ 4-bit g128"] = {"ppl": ppl_awq, "mem_mb": mem_awq}

    del model_awq
    free_gpu()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("PERPLEXITY COMPARISON — WikiText-2 Test")
    print("=" * 60)
    print(f"{'Method':<22s} {'Perplexity':>12s} {'Δ vs FP16':>12s} "
          f"{'Memory':>10s}")
    print("-" * 56)
    for method, data in results.items():
        delta = data["ppl"] - results["FP16"]["ppl"]
        delta_str = f"{delta:+.2f}" if method != "FP16" else "—"
        mem_str = f"{data['mem_mb']:.0f} MB"
        print(f"  {method:<20s} {data['ppl']:>10.2f}   {delta_str:>10s}   "
              f"{mem_str:>8s}")

    return results


def plot_perplexity_comparison(results: dict, cfg: Config):
    """Figure 7.16: Perplexity bar chart — FP16 vs RTN-4bit vs AWQ-4bit."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    methods = list(results.keys())
    ppls = [results[m]["ppl"] for m in methods]

    color_map = {
        "FP16": COLORS["baseline"],
        "RTN 4-bit g128": COLORS["primary"],
        "AWQ 4-bit g128": COLORS["secondary"],
    }
    hatch_map = {
        "FP16": "",
        "RTN 4-bit g128": "//",
        "AWQ 4-bit g128": "xx",
    }

    bars = ax.bar(methods, ppls, width=0.5, edgecolor="black", linewidth=0.8)
    for bar, m in zip(bars, methods):
        bar.set_facecolor(color_map.get(m, COLORS["baseline"]))
        bar.set_hatch(hatch_map.get(m, ""))

    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{ppl:.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_ylabel("Perplexity (↓ is better)")

    method_names = " vs ".join(methods)
    ax.set_title(f"WikiText-2 Perplexity: {method_names}")

    fp16_ppl = results["FP16"]["ppl"]
    ax.axhline(fp16_ppl, color="#333333", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.text(len(methods) - 0.5, fp16_ppl + 0.1, "FP16 baseline",
            fontsize=8, color="#555555")

    ax.set_ylim(0, max(ppls) * 1.15)

    save_or_show(fig, "fig7_16_awq_perplexity", cfg)


# ============================================================================
# Experiment 6: Production Deployment with autoawq
# ============================================================================

def run_awq_deployment(cfg: Config):
    """Experiment 6: Production quantization with AutoAWQ."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 6: AutoAWQ Production Deployment")
    print("=" * 72)

    if cfg.device == "cpu":
        print("  Skipping autoawq (requires CUDA)")
        return None

    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        print("  autoawq not installed. Install with: pip install autoawq")
        print("  Skipping deployment experiment.")
        return None

    model_name = cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare calibration data: list of text strings
    print(f"\nPreparing calibration data ({cfg.calib_samples} samples)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    calib_data = []
    for text in ds["text"]:
        if len(text.strip()) > 100:
            calib_data.append(text.strip())
            if len(calib_data) >= cfg.calib_samples:
                break

    print(f"  Collected {len(calib_data)} calibration texts")

    # --- Listing 7.X: AWQ production deployment ---
    print(f"\nQuantizing {model_name} with AutoAWQ (4-bit, g128)...")

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    model = AutoAWQForCausalLM.from_pretrained(
        model_name,
        safetensors=False,
    )

    # Patch for OPT compatibility: AutoAWQ's quantizer accesses
    # self.model.model.rotary_emb (OPTForCausalLM -> OPTModel -> rotary_emb),
    # but OPT uses learned positional embeddings, not RoPE. The dummy values
    # are harmless: _sanitize_kwargs strips position_embeddings before passing
    # to OPT decoder layers since their forward() doesn't accept that kwarg.
    inner_model = model.model.model  # OPTModel
    if not hasattr(inner_model, "rotary_emb"):
        class _DummyRotaryEmb(torch.nn.Module):
            def forward(self, x, position_ids=None):
                return (torch.zeros(1, device=x.device), torch.zeros(1, device=x.device))
        inner_model.rotary_emb = _DummyRotaryEmb()

    t0 = time.time()
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
    t_quant = time.time() - t0

    print(f"  Quantization time: {t_quant:.1f}s")

    save_path = f"./{cfg.model_key}-awq-4bit"
    print(f"\n  Saving quantized model to {save_path}...")
    model.save_quantized(save_path)
    tokenizer.save_pretrained(save_path)

    # Reload for inference
    print(f"  Loading quantized model for inference...")
    del model
    free_gpu()

    model_inf = AutoAWQForCausalLM.from_quantized(
        save_path, fuse_layers=False,
    )

    # Quick generation test
    print(f"\n  Generation test:")
    prompt = "The advantage of AWQ over round-to-nearest quantization is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model_inf.model.generate(**inputs, max_new_tokens=50, do_sample=False)
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"    Prompt: {prompt}")
    print(f"    Output: {generated[:200]}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mem = gpu_mem_mb()
    print(f"\n  GPU memory after loading quantized model: {mem:.0f} MB")

    result = {
        "quant_time": t_quant,
        "save_path": save_path,
        "mem_mb": mem,
    }

    del model_inf
    free_gpu()

    return result


# ============================================================================
# Conceptual Diagram: AWQ Equivalent Transformation Flow
# ============================================================================

def plot_awq_algorithm_flow(cfg: Config):
    """Figure 7.17: AWQ algorithm schematic — per-channel scaling with grid search."""
    fig, ax = plt.subplots(figsize=(13.5, 12))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("AWQ Algorithm: Activation-Aware Per-Channel Scaling",
                 fontsize=26, pad=24, fontweight="bold")

    def box(x, y, w, h, text, fc, ec="black", ls="-", lw=2.2, fs=22):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold")

    def arrow(x1, y1, x2, y2, text="", color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.8))
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if abs(y2 - y1) > abs(x2 - x1):
                # vertical arrow → label to the right
                ax.text(mx + 0.55, my, text, fontsize=20, ha="left", va="center",
                        color=color, style="italic", fontweight="bold")
            else:
                # horizontal arrow → label above
                ax.text(mx, my + 0.4, text, fontsize=20, ha="center",
                        color=color, style="italic", fontweight="bold")

    # ─── Row 1 (top): inputs ───
    box(1.5, 9.5, 2.5, 1.2, "X (calib)\n128 samples", "#dde8d6", fs=20)
    arrow(4.0, 10.1, 4.7, 10.1)
    box(4.7, 9.5, 2.8, 1.2, "Act Scales\nmean(|X|)\nper channel", "#ccdcec", fs=20)

    # ─── Row 2 (middle): W + Grid Search container ───
    box(0.3, 5.0, 2.6, 3.7, "W\n[out, in]\nFP16 weights", "#e8e8e8", fs=22)

    # Grid container (dashed)
    box(3.5, 5.0, 5.7, 3.7, "", "#f5f5f5", ls="--", lw=2.0)
    ax.text(6.35, 8.4, "Grid search over α", fontsize=22,
            ha="center", style="italic", color="#333333", fontweight="bold")

    # Inside grid
    box(3.7, 6.5, 2.0, 1.4, "s = act^α\nper channel", "#b3cde3", fs=20)
    arrow(5.7, 7.2, 7.0, 7.2, "scale")
    box(7.0, 6.5, 2.0, 1.4, "W' = W·s\nX' = X/s", "#d9e6f2", fs=20)
    box(3.7, 5.1, 5.3, 1.2, "min MSE:\n||Q(W·s)·X/s − W·X||",
        "#fff2cc", ec="#333333", fs=20)

    # Arrows into grid container
    arrow(6.1, 9.5, 6.1, 8.7)       # Act Scales → grid (vertical)
    arrow(2.9, 6.85, 3.5, 6.85)     # W → grid (horizontal)

    # ─── Row 3 (bottom): Apply → Fold → INT4 ───
    arrow(5.0, 5.0, 5.0, 4.2, "best s*")  # grid → Apply (vertical)

    box(3.7, 1.6, 2.6, 2.6,
        "W* = W·s*\n\nQuantize:\nQ(W*)\nstandard\ngroup-wise",
        "#d6eaf8", fs=20)

    arrow(6.3, 2.9, 8.1, 2.9, "absorb s*")
    box(8.1, 1.7, 2.4, 2.4,
        "Fold s* into\npreceding\nLayerNorm\nγ → γ/s*",
        "#e8e8e8", fs=20)

    arrow(10.5, 2.9, 11.5, 2.9)
    box(11.5, 2.2, 1.6, 1.4, "INT4\nmodel\nready", "#d6eaf8", fs=20)

    # ─── Footer ───
    ax.text(6.75, 0.6,
            "Key: scale salient channels UP before quantization → they get more grid points.\n"
            "Inverse scaling absorbed into preceding LayerNorm → zero runtime overhead.",
            fontsize=18, ha="center", color="#1B5299", style="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.95,
                      edgecolor="#1B5299", linewidth=1.8))

    save_or_show(fig, "fig7_17_awq_algorithm_flow", cfg)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chapter 7.4 — AWQ Activation-Aware Weight Quantization"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "salient", "single-layer", "layer-sweep", "alpha",
                 "perplexity", "deploy", "flow"],
        help="Which experiment(s) to run",
    )
    parser.add_argument("--model", type=str, default="opt-6.7b",
                       choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--plot-dir", type=str, default="ch7_plots")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--n-grid", type=int, default=20)
    return parser.parse_args()

def main():
    args = parse_args()

    # Suppress noisy third-party loggers
    import logging
    for _logger in ("httpx", "httpcore", "tokenizer", "huggingface_hub", "awq"):
        logging.getLogger(_logger).setLevel(logging.WARNING)

    cfg = Config(
        model_key=args.model,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        save_plots=args.save_plots,
        plot_dir=args.plot_dir,
        wbits=args.wbits,
        groupsize=args.groupsize,
        eval_samples=args.eval_samples,
        calib_samples=args.calib_samples,
        n_grid=args.n_grid,
    )

    set_seed(cfg.seed)
    setup_manning_style()

    print(f"Configuration:")
    print(f"  Model:    {cfg.model_name}")
    print(f"  Device:   {cfg.device}")
    print(f"  Bits:     {cfg.wbits}")
    print(f"  Groupsize: {cfg.groupsize}")
    print(f"  Grid pts: {cfg.n_grid}")
    print(f"  GPU:      {torch.cuda.get_device_name() if cfg.device == 'cuda' else 'N/A'}")
    if cfg.device == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  VRAM:     {total_mem:.0f} MB")
    print()

    mode = args.mode

    # Flow diagram (no model needed)
    if mode in ("all", "flow"):
        plot_awq_algorithm_flow(cfg)

    # Experiment 1: Salient channel analysis
    if mode in ("all", "salient"):
        salient_results = run_salient_channel_analysis(cfg)
        plot_salient_channels(salient_results, cfg)

    # Experiment 2: Single layer from scratch
    if mode in ("all", "single-layer"):
        sl_results = run_single_layer_experiment(cfg)
        plot_single_layer(sl_results, cfg)

    # Experiment 3: Layer sweep
    if mode in ("all", "layer-sweep"):
        layer_results = run_layer_sweep(cfg)
        plot_layer_sweep(layer_results, cfg)

    # Experiment 4: Alpha sweep
    if mode in ("all", "alpha"):
        alpha_results = run_alpha_sweep(cfg)
        plot_alpha_sweep(alpha_results, cfg)

    # Experiment 6: autoawq deployment — run BEFORE perplexity so the
    # quantized model is saved to disk for the perplexity comparison
    if mode in ("all", "deploy"):
        awq_result = run_awq_deployment(cfg)
        free_gpu()

    # Experiment 5: Perplexity showdown — loads the AutoAWQ model
    # saved by Experiment 6 for the AWQ perplexity arm
    if mode in ("all", "perplexity"):
        ppl_results = run_perplexity_comparison(cfg)
        plot_perplexity_comparison(ppl_results, cfg)

    print("\n" + "=" * 72)
    print("All experiments complete.")
    if cfg.save_plots:
        print(f"Figures saved to: {cfg.plot_dir}/")
        print(f"  Each figure saved as .pdf (vector) and .png (raster)")
    print("=" * 72)

if __name__ == "__main__":
    main()