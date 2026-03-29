#!/usr/bin/env python3
"""
Chapter 7 — Section 7.3 Companion Script
Apply Hessian-aware groupwise methods: GPTQ weight quantization

This script builds GPTQ (Frantar et al., ICLR 2023) from scratch, then validates
it against the production gptqmodel implementation. The pedagogical arc:

  1. GPTQ from scratch     — implement Algorithm 1 on a single linear layer,
                              compare RTN vs GPTQ reconstruction error
  2. Layer-by-layer         — apply GPTQ to every Linear in OPT-6.7B, show
                              per-layer MSE improvement over RTN at 4-bit
  3. Group size sweep       — why g128 recovers most of the accuracy gap
  4. Perplexity showdown    — FP16 vs RTN-4bit vs GPTQ-4bit on WikiText-2
  5. gptqmodel deployment   — one-line production quantization

Hardware: Colab T4 (16 GB VRAM). OPT-6.7B quantization processes one
         transformer block at a time, peak ~8 GB.

Usage:
    # Full pipeline (T4 GPU required, pip install gptqmodel for experiments 4-5)
    python ch7_gptq_quantization.py --mode all --save-plots

    # GPTQ from scratch on one layer (quick, fits on T4)
    python ch7_gptq_quantization.py --mode single-layer --save-plots

    # Per-layer RTN vs GPTQ comparison
    python ch7_gptq_quantization.py --mode layer-sweep --save-plots

    # Group size sweep
    python ch7_gptq_quantization.py --mode groupsize --save-plots

    # Perplexity comparison (loads models sequentially)
    python ch7_gptq_quantization.py --mode perplexity --save-plots

    # Production deployment with gptqmodel
    python ch7_gptq_quantization.py --mode deploy --save-plots

    # CPU-only (uses OPT-125M for illustration, not publishable numbers)
    python ch7_gptq_quantization.py --mode all --save-plots \
        --model opt-125m --device cpu

References:
    Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
    Pre-trained Transformers," ICLR 2023.
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
    calib_seq_len: int = 2048       # GPTQ uses 2048-token segments (paper default)

    # GPTQ algorithm
    wbits: int = 4                  # Target bit-width
    blocksize: int = 128            # Lazy batch update block size (Algorithm 1)
    groupsize: int = -1             # -1 = per-row, 128 = standard grouping
    damp_percent: float = 0.01     # Dampening for Hessian diagonal (paper: 1%)

    # Perplexity evaluation
    eval_samples: int = 64
    eval_seq_len: int = 512         # 512 fits on T4; 2048 OOMs with OPT-6.7B FP16

    # Group size sweep
    group_sizes: List[int] = field(default_factory=lambda: [
        -1, 1024, 512, 256, 128, 64, 32
    ])

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
}
HATCHES = {
    "baseline":   "",
    "primary":    "//",
    "secondary":  "xx",
    "tertiary":   "..",
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
# Core GPTQ Algorithm — Built from Scratch
# ============================================================================

def quantize_to_nearest(w: torch.Tensor, bits: int = 4,
                         groupsize: int = -1) -> torch.Tensor:
    """
    RTN quantization: asymmetric uniform quant on [min, max] grid, per-row or per-group.
    Returns dequantized weights (float, same shape). Baseline that GPTQ improves upon.
    """
    maxq = 2 ** bits - 1  # e.g., 15 for 4-bit

    rows, cols = w.shape
    w = w.float().clone()

    if groupsize == -1:
        # Per-row: one scale+zero per output channel
        w_min = w.min(dim=1, keepdim=True).values
        w_max = w.max(dim=1, keepdim=True).values
        scale = (w_max - w_min) / maxq
        scale = torch.clamp(scale, min=1e-8)
        zero = torch.round(-w_min / scale)
        q = torch.clamp(torch.round(w / scale) + zero, 0, maxq)
        w_deq = scale * (q - zero)
    else:
        # Per-group: partition in_features into groups
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

def gptq_quantize(
    W: torch.Tensor,
    H: torch.Tensor,
    bits: int = 4,
    blocksize: int = 128,
    groupsize: int = -1,
    damp_percent: float = 0.01,
) -> Tuple[torch.Tensor, dict]:
    """
    GPTQ: Algorithm 1 from Frantar et al. (ICLR 2023).

    Given a weight matrix W [out_features, in_features] and the Hessian
    H = 2 * X @ X^T (where X [in_features, n_samples] are calibration
    activations), quantize W column-by-column with Hessian-guided error
    compensation.

    Three key improvements over naive OBQ:
      1. Fixed column order (not greedy) — allows shared H^{-1} across rows
      2. Lazy batch updates — block of B columns at a time for GPU efficiency
      3. Cholesky reformulation — numerically stable precomputation of H^{-1}

    Args:
        W: Weight matrix [out_features, in_features]
        H: Hessian matrix [in_features, in_features]
        bits: Target quantization bit-width
        blocksize: Lazy batch size B (default 128, per paper)
        groupsize: -1 for per-row, else group size
        damp_percent: Dampening for Hessian diagonal stability

    Returns:
        Q: Dequantized quantized weights [out_features, in_features]
        stats: Dictionary with quantization statistics
    """
    W = W.float().clone()
    rows, cols = W.shape
    maxq = 2 ** bits - 1

    # --- Step 0: Compute Hessian inverse via Cholesky (Section 4, Step 3) ---
    H = H.float().clone()
    dead_mask = torch.diag(H) == 0           # Dead columns (never activated)
    H[dead_mask, dead_mask] = 1.0            # Avoid singularity

    # Dampening: add λ = 1% of mean diagonal to stabilize inversion
    damp = damp_percent * torch.diag(H).mean()
    H.diagonal().add_(damp)

    # Cholesky decomposition of H^{-1}: precompute upper triangular factor.
    # The paper overwrites H^{-1} with Cholesky(H^{-1})^T (upper triangular).
    # This avoids the numerically unstable repeated Gaussian elimination
    # in the original OBQ formulation.
    try:
        L = torch.linalg.cholesky(H)         # Lower triangular L: H = L @ L^T
        Hinv = torch.cholesky_inverse(L)     # H^{-1} from Cholesky factor
        Hinv = torch.linalg.cholesky(Hinv, upper=True)  # Upper Cholesky of H^{-1}
    except torch.linalg.LinAlgError:
        # Fallback: add stronger dampening and retry
        H.diagonal().add_(damp * 10)
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    Losses = torch.zeros(rows, device=W.device)

    # Precompute per-row scale/zero (for groupsize == -1).
    # For grouping: recompute at each group boundary using current (error-
    # compensated) weights — this is why "grouping interacts very well with
    # GPTQ" (Frantar et al., Table 5).
    cur_scale = None
    cur_zero = None

    if groupsize == -1:
        w_min = W.min(dim=1, keepdim=True).values
        w_max = W.max(dim=1, keepdim=True).values
        cur_scale = ((w_max - w_min) / maxq).clamp(min=1e-8)
        cur_zero = torch.round(-w_min / cur_scale)

    # --- Main loop: Algorithm 1 ---
    for block_start in range(0, cols, blocksize):
        block_end = min(block_start + blocksize, cols)
        block_len = block_end - block_start

        # Error accumulator for this block
        Err = torch.zeros(rows, block_len, device=W.device, dtype=W.dtype)

        for j in range(block_start, block_end):
            # Recompute scale/zero at group boundaries using current weights
            if groupsize != -1 and j % groupsize == 0:
                g_end = min(j + groupsize, cols)
                wg = W[:, j:g_end]
                w_min = wg.min(dim=1, keepdim=True).values
                w_max = wg.max(dim=1, keepdim=True).values
                cur_scale = ((w_max - w_min) / maxq).clamp(min=1e-8)
                cur_zero = torch.round(-w_min / cur_scale)

            w_col = W[:, j]                   # Current column to quantize
            d = Hinv[j, j]                    # Diagonal of Cholesky factor

            # Quantize this column to the nearest grid point
            q_col = torch.clamp(
                torch.round(w_col.unsqueeze(1) / cur_scale) + cur_zero,
                0, maxq
            )
            q_col = (cur_scale * (q_col - cur_zero)).squeeze(1)
            Q[:, j] = q_col

            # Quantization error, scaled by inverse Hessian diagonal
            err = (w_col - q_col) / d
            Err[:, j - block_start] = err

            # Track loss
            Losses += (w_col - q_col).float().pow(2) / (2.0 * d ** 2)

            # Update remaining columns in this block (lazy batch update)
            W[:, j:block_end] -= err.unsqueeze(1) * Hinv[j, j:block_end].unsqueeze(0)

        # Global update: propagate block errors to all remaining columns
        W[:, block_end:] -= Err @ Hinv[block_start:block_end, block_end:]

    # Zero out dead columns
    Q[:, dead_mask] = 0.0

    stats = {
        "mean_loss": Losses.mean().item(),
        "max_loss": Losses.max().item(),
        "n_dead": dead_mask.sum().item(),
    }

    return Q, stats

def compute_hessian(X: torch.Tensor) -> torch.Tensor:
    """Compute H = X @ X^T / n_samples (Hessian for layer reconstruction objective)."""
    n_samples = X.shape[1]
    H = (X @ X.T) / n_samples  # Normalize by number of samples for stability
    return H

# ============================================================================
# Experiment 1: GPTQ from Scratch on a Single Layer
# ============================================================================

def run_single_layer_experiment(cfg: Config):
    """Experiment 1: RTN vs GPTQ at 4-bit and 3-bit on a single fc1 layer, measuring reconstruction MSE."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: GPTQ from Scratch on a Single fc1 Layer")
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

    # Pick a target layer — layer 0's fc1 (FFN up-projection)
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
        for seq in calib_seqs[:8]:   # 8 sequences × 2048 tokens = 16384 samples
            model(seq.to(device))

    hook.remove()

    # Build activation matrix X [in_features, n_samples]
    X_all = torch.cat(captured_inputs, dim=0)  # [n_samples, in_features]
    X = X_all.T.float()                         # [in_features, n_samples]
    print(f"  Calibration matrix: X.shape = {list(X.shape)} "
          f"({X.shape[1]} token positions)")

    # Copy weights and free GPU model
    W = target_module.weight.data.cpu().clone()  # [out_features, in_features]
    print(f"  Weight matrix: W.shape = {list(W.shape)}")

    del model
    free_gpu()

    # Ground truth output: Y = W @ X  (on CPU)
    Y_ref = (W.float() @ X).float()

    # --- Compute the Hessian ---
    print(f"\n  Computing Hessian H = X @ X^T...")
    t0 = time.time()
    H = compute_hessian(X)
    t_hessian = time.time() - t0
    print(f"  Hessian shape: {list(H.shape)}, computed in {t_hessian:.1f}s")

    # Visualize Hessian diagonal (tells us which input features matter most)
    H_diag = torch.diag(H)
    print(f"  Hessian diagonal: min={H_diag.min():.4f}, "
          f"max={H_diag.max():.4f}, mean={H_diag.mean():.4f}")
    print(f"  Top-10 most sensitive dimensions: "
          f"{H_diag.topk(10).indices.tolist()}")

    # --- Compare RTN vs GPTQ at 4-bit and 3-bit ---
    results = []
    for bits in [4, 3]:
        print(f"\n  --- {bits}-bit quantization ---")

        # RTN (round-to-nearest)
        t0 = time.time()
        W_rtn = quantize_to_nearest(W, bits=bits, groupsize=-1)
        t_rtn = time.time() - t0
        Y_rtn = W_rtn.float() @ X
        mse_rtn = ((Y_ref - Y_rtn) ** 2).mean().item()

        # GPTQ (Hessian-aware)
        t0 = time.time()
        W_gptq, gptq_stats = gptq_quantize(
            W.clone(), H.clone(), bits=bits,
            blocksize=cfg.blocksize, groupsize=-1,
            damp_percent=cfg.damp_percent,
        )
        t_gptq = time.time() - t0
        Y_gptq = W_gptq.float() @ X
        mse_gptq = ((Y_ref - Y_gptq) ** 2).mean().item()

        ratio = mse_rtn / max(mse_gptq, 1e-12)

        print(f"    RTN    MSE: {mse_rtn:.6f}  ({t_rtn:.2f}s)")
        print(f"    GPTQ   MSE: {mse_gptq:.6f}  ({t_gptq:.2f}s)")
        print(f"    GPTQ / RTN error reduction: {ratio:.1f}×")
        print(f"    GPTQ mean loss: {gptq_stats['mean_loss']:.6f}")

        results.append({
            "bits": bits,
            "mse_rtn": mse_rtn,
            "mse_gptq": mse_gptq,
            "ratio": ratio,
            "t_rtn": t_rtn,
            "t_gptq": t_gptq,
        })

    # --- GPTQ with grouping ---
    print(f"\n  --- 4-bit GPTQ with groupsize=128 ---")
    t0 = time.time()
    W_gptq_g128, stats_g128 = gptq_quantize(
        W.clone(), H.clone(), bits=4,
        blocksize=cfg.blocksize, groupsize=128,
        damp_percent=cfg.damp_percent,
    )
    t_gptq_g128 = time.time() - t0
    Y_gptq_g128 = W_gptq_g128.float() @ X
    mse_gptq_g128 = ((Y_ref - Y_gptq_g128) ** 2).mean().item()

    mse_rtn_4bit = results[0]["mse_rtn"]
    mse_gptq_4bit = results[0]["mse_gptq"]
    print(f"    GPTQ-g128 MSE: {mse_gptq_g128:.6f}  ({t_gptq_g128:.2f}s)")
    print(f"    vs RTN:  {mse_rtn_4bit / max(mse_gptq_g128, 1e-12):.1f}× reduction")
    print(f"    vs GPTQ: {mse_gptq_4bit / max(mse_gptq_g128, 1e-12):.1f}× "
          f"further reduction from grouping")

    results.append({
        "bits": 4,
        "groupsize": 128,
        "mse_gptq_g128": mse_gptq_g128,
    })

    return results, H_diag.numpy()

def plot_hessian_and_single_layer(results: list, H_diag: np.ndarray, cfg: Config):
    """Figure 7.8: Hessian diagonal (left) and RTN vs GPTQ MSE at 4/3-bit (right)."""
    if not results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Hessian diagonal
    dims = np.arange(len(H_diag))
    ax1.plot(dims, H_diag, color="#333333", linewidth=0.3, alpha=0.6)

    # Highlight top-10 most sensitive
    top_k = 10
    top_indices = np.argsort(H_diag)[-top_k:]
    ax1.scatter(top_indices, H_diag[top_indices], color=COLORS["primary"],
                s=30, zorder=5, label=f"Top-{top_k} most sensitive")

    ax1.set_xlabel("Input Feature Dimension")
    ax1.set_ylabel("Hessian Diagonal Value")
    ax1.set_title("Per-Dimension Sensitivity (Hessian Diagonal)")
    ax1.legend()
    ax1.set_yscale("log")

    # Panel 2: RTN vs GPTQ at 4-bit and 3-bit
    r4 = [r for r in results if r["bits"] == 4 and "groupsize" not in r][0]
    r3 = [r for r in results if r["bits"] == 3][0]

    labels = ["4-bit RTN", "4-bit GPTQ", "3-bit RTN", "3-bit GPTQ"]
    mses = [r4["mse_rtn"], r4["mse_gptq"], r3["mse_rtn"], r3["mse_gptq"]]
    colors_list = [COLORS["baseline"], COLORS["primary"],
                   COLORS["baseline"], COLORS["secondary"]]
    hatches_list = ["", "//", "", "xx"]

    bars = ax2.bar(labels, mses, width=0.5, edgecolor="black", linewidth=0.8)
    for bar, color, hatch in zip(bars, colors_list, hatches_list):
        bar.set_facecolor(color)
        bar.set_hatch(hatch)

    # Value labels
    for bar, mse in zip(bars, mses):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{mse:.4f}", ha="center", va="bottom", fontsize=8)

    ax2.set_ylabel("Reconstruction MSE (vs FP16)")
    ax2.set_title("RTN vs GPTQ: Single fc1 Layer")
    ax2.set_yscale("log")

    # Ratio annotations
    ax2.annotate(f"{r4['ratio']:.0f}×", xy=(1, r4['mse_gptq']),
                 fontsize=9, ha="center", va="top", color=COLORS["primary"],
                 fontweight="bold")
    ax2.annotate(f"{r3['ratio']:.0f}×", xy=(3, r3['mse_gptq']),
                 fontsize=9, ha="center", va="top", color=COLORS["secondary"],
                 fontweight="bold")

    save_or_show(fig, "fig7_8_gptq_single_layer", cfg)

# ============================================================================
# Experiment 2: Layer-by-Layer RTN vs GPTQ
# ============================================================================

def run_layer_sweep(cfg: Config):
    """Experiment 2: RTN vs GPTQ MSE on every fc1 layer. One block at a time to fit T4 VRAM."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Per-Layer RTN vs GPTQ Comparison (4-bit)")
    print("=" * 72)

    device = cfg.device
    model_name = cfg.model_name

    print(f"\nLoading {model_name} in FP16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    print(f"  GPU memory: {gpu_mem_mb():.0f} MB")

    num_layers = model.config.num_hidden_layers

    # Capture activations for fc1 and fc2 of each layer
    calib_seqs = load_wikitext2_tokens(
        tokenizer, cfg.calib_samples, cfg.calib_seq_len, split="train"
    )

    # We process each layer by hooking its fc1 and capturing inputs
    print(f"\nProcessing {num_layers} layers (4-bit, per-row)...")
    print(f"{'Layer':<8} {'Module':<6} {'RTN MSE':>12} {'GPTQ MSE':>12} "
          f"{'Ratio':>8} {'Time (s)':>10}")
    print("-" * 62)

    layer_results = []

    # Capture all fc1 inputs in one pass
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

    # Process each layer on CPU
    for idx in range(num_layers):
        if idx not in captured_fc1 or idx not in fc1_weights:
            continue

        X_all = torch.cat(captured_fc1[idx], dim=0)[:4096]
        W = fc1_weights[idx]

        X = X_all.T.float()           # [in_features, n_samples]
        Y_ref = W.float() @ X

        # Hessian
        H = compute_hessian(X)

        # RTN
        W_rtn = quantize_to_nearest(W, bits=cfg.wbits, groupsize=-1)
        Y_rtn = W_rtn.float() @ X
        mse_rtn = ((Y_ref - Y_rtn) ** 2).mean().item()

        # GPTQ
        t0 = time.time()
        W_gptq, stats = gptq_quantize(
            W.clone(), H.clone(), bits=cfg.wbits,
            blocksize=cfg.blocksize, groupsize=-1,
            damp_percent=cfg.damp_percent,
        )
        t_gptq = time.time() - t0
        Y_gptq = W_gptq.float() @ X
        mse_gptq = ((Y_ref - Y_gptq) ** 2).mean().item()

        ratio = mse_rtn / max(mse_gptq, 1e-12)

        layer_results.append({
            "layer": idx,
            "mse_rtn": mse_rtn,
            "mse_gptq": mse_gptq,
            "ratio": ratio,
            "time": t_gptq,
        })

        print(f"  {idx:>4d}    fc1    {mse_rtn:>10.6f}   {mse_gptq:>10.6f}   "
              f"{ratio:>6.1f}×   {t_gptq:>8.1f}")

        del X_all, W, X, H, Y_ref, W_rtn, Y_rtn, W_gptq, Y_gptq
        gc.collect()

    del captured_fc1, fc1_weights
    gc.collect()

    if layer_results:
        avg_ratio = np.mean([r["ratio"] for r in layer_results])
        print(f"\n  Average error reduction (RTN → GPTQ): {avg_ratio:.1f}×")

    return layer_results

def plot_layer_sweep(layer_results: list, cfg: Config):
    """Figure 7.9: Per-layer fc1 MSE, RTN vs GPTQ at 4-bit."""
    if not layer_results:
        return

    fig, ax = plt.subplots(figsize=(12, 4.5))

    layers = [r["layer"] for r in layer_results]
    mse_rtn = [r["mse_rtn"] for r in layer_results]
    mse_gptq = [r["mse_gptq"] for r in layer_results]

    x = np.arange(len(layers))
    width = 0.35

    bars1 = ax.bar(x - width/2, mse_rtn, width, label="RTN (4-bit)",
                   color=COLORS["baseline"], edgecolor="black",
                   linewidth=0.5, hatch=HATCHES["baseline"])
    bars2 = ax.bar(x + width/2, mse_gptq, width, label="GPTQ (4-bit)",
                   color=COLORS["secondary"], edgecolor="black",
                   linewidth=0.5, hatch=HATCHES["secondary"])

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Reconstruction MSE (vs FP16)")
    ax.set_title("Per-Layer fc1 Error: RTN vs GPTQ at 4-bit (per-row)")
    ax.set_xticks(x[::4])
    ax.set_xticklabels([str(l) for l in layers[::4]])
    ax.legend()
    ax.set_yscale("log")

    # Average ratio annotation
    avg_ratio = np.mean([r["ratio"] for r in layer_results])
    ax.text(0.98, 0.95, f"Mean error reduction: {avg_ratio:.0f}×",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", color=COLORS["secondary"],
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    save_or_show(fig, "fig7_9_gptq_per_layer", cfg)

# ============================================================================
# Experiment 3: Group Size Sweep
# ============================================================================

def run_groupsize_sweep(cfg: Config):
    """Experiment 3: Sweep group sizes on layer 0 fc1 — show how finer grouping reduces MSE."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Group Size Sweep (4-bit, layer 0 fc1)")
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

    # Capture activations
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
    X = X_all.T.float()
    W = target_module.weight.data.cpu().clone()

    del model
    free_gpu()

    Y_ref = W.float() @ X
    H = compute_hessian(X)

    print(f"\nSweeping group sizes on layer 0 fc1 (4-bit)...")
    print(f"{'Group Size':>12} {'RTN MSE':>12} {'GPTQ MSE':>12} "
          f"{'Ratio':>8} {'Extra bits':>12}")
    print("-" * 62)

    sweep_results = []
    in_features = W.shape[1]

    for gs in cfg.group_sizes:
        gs_label = "per-row" if gs == -1 else str(gs)

        # RTN
        W_rtn = quantize_to_nearest(W, bits=cfg.wbits, groupsize=gs)
        Y_rtn = W_rtn.float() @ X
        mse_rtn = ((Y_ref - Y_rtn) ** 2).mean().item()

        # GPTQ
        W_gptq, stats = gptq_quantize(
            W.clone(), H.clone(), bits=cfg.wbits,
            blocksize=cfg.blocksize, groupsize=gs,
            damp_percent=cfg.damp_percent,
        )
        Y_gptq = W_gptq.float() @ X
        mse_gptq = ((Y_ref - Y_gptq) ** 2).mean().item()

        ratio = mse_rtn / max(mse_gptq, 1e-12)

        # Extra bits from scale/zero overhead
        # Each group needs one FP16 scale + one FP16 zero = 32 bits
        # Amortized over group_size weights × wbits bits each
        if gs == -1:
            extra_bits = 0.0
        else:
            n_groups = math.ceil(in_features / gs)
            scale_bits = 32 * n_groups      # FP16 scale + FP16 zero per group
            weight_bits = in_features * cfg.wbits
            extra_bits = scale_bits / in_features  # Per-weight overhead

        sweep_results.append({
            "groupsize": gs,
            "gs_label": gs_label,
            "mse_rtn": mse_rtn,
            "mse_gptq": mse_gptq,
            "ratio": ratio,
            "extra_bits": extra_bits,
        })

        print(f"  {gs_label:>10s}   {mse_rtn:>10.6f}   {mse_gptq:>10.6f}   "
              f"{ratio:>6.1f}×   {extra_bits:>10.3f}")

    del X_all, X, W, H
    gc.collect()

    return sweep_results

def plot_groupsize_sweep(sweep_results: list, cfg: Config):
    """Figure 7.10: MSE vs group size (left) and Pareto frontier of effective bits (right)."""
    if not sweep_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    gs_labels = [r["gs_label"] for r in sweep_results]
    mse_rtn = [r["mse_rtn"] for r in sweep_results]
    mse_gptq = [r["mse_gptq"] for r in sweep_results]

    x = np.arange(len(gs_labels))
    width = 0.35

    # Panel 1: MSE vs group size (RTN and GPTQ side by side)
    bars1 = ax1.bar(x - width/2, mse_rtn, width, label="RTN",
                    color=COLORS["baseline"], edgecolor="black",
                    linewidth=0.5, hatch=HATCHES["baseline"])
    bars2 = ax1.bar(x + width/2, mse_gptq, width, label="GPTQ",
                    color=COLORS["secondary"], edgecolor="black",
                    linewidth=0.5, hatch=HATCHES["secondary"])

    ax1.set_xlabel("Group Size")
    ax1.set_ylabel("Reconstruction MSE (vs FP16)")
    ax1.set_title("4-bit Quantization Error vs Group Size")
    ax1.set_xticks(x)
    ax1.set_xticklabels(gs_labels, rotation=45, ha="right")
    ax1.legend()
    ax1.set_yscale("log")

    # Panel 2: GPTQ MSE vs effective bits (Pareto curve)
    eff_bits = [cfg.wbits + r["extra_bits"] for r in sweep_results]
    ax2.plot(eff_bits, mse_gptq, color=COLORS["secondary"], marker="s",
             linestyle="-", markersize=7, linewidth=1.5)

    # Annotate each point with group size label
    for eb, mse, label in zip(eff_bits, mse_gptq, gs_labels):
        ax2.annotate(f"g{label}", (eb, mse), textcoords="offset points",
                     xytext=(5, 5), fontsize=8, color="#333333")

    ax2.set_xlabel("Effective Bits per Weight (including scale overhead)")
    ax2.set_ylabel("Reconstruction MSE (GPTQ)")
    ax2.set_title("Grouping Pareto Frontier")
    ax2.set_yscale("log")

    save_or_show(fig, "fig7_10_gptq_groupsize", cfg)

# ============================================================================
# Experiment 4: Perplexity Showdown
# ============================================================================

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, sequences, device, label=""):
    """Standard autoregressive perplexity: exp(mean NLL per token) over pre-tokenized sequences."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0

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
            print(f"    [{label}] {i+1}/{len(sequences)} sequences, "
                  f"running PPL: {running_ppl:.2f}")

    avg_nll = total_nll / total_tokens
    ppl = np.exp(avg_nll)
    return ppl

def simulate_rtn_quantized_model(model, bits=4, groupsize=-1):
    """Replace all Linear weights with RTN quantize-dequantize (FP16 storage, measures quality not memory)."""
    n_quantized = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            dev = module.weight.data.device
            W_cpu = module.weight.data.cpu()
            W_deq = quantize_to_nearest(W_cpu, bits=bits, groupsize=groupsize)
            module.weight.data = W_deq.to(torch.float16).to(dev)
            n_quantized += 1
    print(f"  RTN-quantized {n_quantized} Linear layers ({bits}-bit, "
          f"groupsize={groupsize})")
    return model

def get_gptq_basename(save_path: str) -> str:
    """Detect the quantized model filename in a saved gptqmodel directory."""
    import glob
    candidates = [
        "gptq_model-4bit-128g",
        "model",
        "gptq_model",
    ]
    for name in candidates:
        if (os.path.exists(os.path.join(save_path, f"{name}.safetensors")) or
                os.path.exists(os.path.join(save_path, f"{name}.safetensors.index.json"))):
            return name
    # Fallback: scan for any safetensors file
    hits = glob.glob(os.path.join(save_path, "*.safetensors*"))
    if hits:
        stem = os.path.basename(hits[0]).split(".safetensors")[0]
        return stem
    raise FileNotFoundError(
        f"No safetensors file found in {save_path}. "
        f"Delete the directory and re-run to re-quantize."
    )


def run_perplexity_comparison(cfg: Config):
    """Experiment 4: FP16 vs RTN-4bit vs GPTQ-4bit perplexity on WikiText-2."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Perplexity Showdown — FP16 vs RTN-4bit vs GPTQ-4bit")
    print("=" * 72)

    if cfg.device == "cpu" and cfg.model_key == "opt-6.7b":
        print("  Skipping perplexity (OPT-6.7B requires GPU)")
        return None

    # Ensure clean GPU state — previous experiments may have left fragmentation
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

    # --- 2. RTN 4-bit (simulated) ---
    print(f"\n[2/3] RTN 4-bit (simulated, per-row)...")
    model_rtn = simulate_rtn_quantized_model(model_fp16, bits=4, groupsize=-1)
    ppl_rtn = evaluate_perplexity(model_rtn, tokenizer, eval_seqs, cfg.device, "RTN-4bit")
    print(f"  RTN-4bit Perplexity: {ppl_rtn:.2f}")
    results["RTN 4-bit"] = {"ppl": ppl_rtn, "mem_mb": mem_fp16}

    del model_rtn, model_fp16
    free_gpu()

    # --- 3. GPTQ 4-bit via gptqmodel ---
    # Fail loudly if not on GPU — GPTQ requires CUDA
    if cfg.device == "cpu":
        raise RuntimeError(
            "GPTQ perplexity experiment requires CUDA. "
            "Re-run with --device cuda or on a GPU runtime."
        )

    from gptqmodel import GPTQModel, QuantizeConfig

    print(f"\n[3/3] GPTQ 4-bit (gptqmodel, g128)...")

    # Download PyTorch weights only to explicit local dir.
    # Skip if already present — avoids re-downloading 13.5GB every run.
    # local_dir_use_symlinks=False prevents double-writing to HF cache.
    local_model_path = "./opt-6.7b-pt"
    if not os.path.isdir(local_model_path):
        from huggingface_hub import snapshot_download
        snapshot_download(
            model_name,
            local_dir=local_model_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["flax_*", "tf_*", "*.h5", "*.msgpack"],
        )
        print(f"  Downloaded PyTorch weights to {local_model_path}")
    else:
        print(f"  Using cached weights at {local_model_path}")

    # Prepare calibration data: plain Python lists (not tensors/BatchEncoding)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    calib_data = []
    for text in ds["text"]:
        if len(text.strip()) > 100:
            enc = tokenizer(text, return_tensors=None,
                          max_length=cfg.calib_seq_len, truncation=True)
            if len(enc["input_ids"]) >= 128:
                calib_data.append({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                })
                if len(calib_data) >= cfg.calib_samples:
                    break
    print(f"  Prepared {len(calib_data)} calibration sequences")

    quantize_config = QuantizeConfig(
        bits=4, group_size=128, damp_percent=0.01, desc_act=False,
    )

    gptq_save_path = "./opt-6.7b-gptq-4bit"

    if os.path.isdir(gptq_save_path):
        print(f"  Found saved model at {gptq_save_path}, loading directly...")
        _basename = get_gptq_basename(gptq_save_path)
        model_gptq = GPTQModel.from_quantized(gptq_save_path, device=cfg.device, model_basename=_basename, backend="torch")
    else:
        model_gptq = GPTQModel.load(local_model_path, quantize_config)
        model_gptq.quantize(calib_data)

        print(f"  Saving quantized model to {gptq_save_path}...")
        model_gptq.save_quantized(gptq_save_path)

        # Clean up offload dir AFTER save — gptqmodel reads it during save_quantized
        import shutil
        if os.path.isdir("./gptqmodel_offload"):
            shutil.rmtree("./gptqmodel_offload")
            print("  Cleaned up gptqmodel_offload/")
        tokenizer.save_pretrained(gptq_save_path)
        print(f"  Saved — future runs will load directly.")

        del model_gptq
        free_gpu()
        _basename = get_gptq_basename(gptq_save_path)
        model_gptq = GPTQModel.from_quantized(gptq_save_path, device=cfg.device, model_basename=_basename, backend="torch")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mem_gptq = gpu_mem_mb()
    print(f"  GPU memory: {mem_gptq:.0f} MB")

    ppl_gptq = evaluate_perplexity(
        model_gptq.model, tokenizer, eval_seqs, cfg.device, "GPTQ-4bit-g128"
    )
    print(f"  GPTQ-4bit-g128 Perplexity: {ppl_gptq:.2f}")
    results["GPTQ 4-bit g128"] = {"ppl": ppl_gptq, "mem_mb": mem_gptq}

    del model_gptq
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
    """Figure 7.11: Perplexity bar chart (dynamic — plots whatever methods completed)."""
    if not results:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    methods = list(results.keys())
    ppls = [results[m]["ppl"] for m in methods]

    color_map = {
        "FP16": COLORS["baseline"],
        "RTN 4-bit": COLORS["primary"],
        "GPTQ 4-bit g128": COLORS["secondary"],
    }
    hatch_map = {
        "FP16": "",
        "RTN 4-bit": "//",
        "GPTQ 4-bit g128": "xx",
    }

    bars = ax.bar(methods, ppls, width=0.5, edgecolor="black", linewidth=0.8)
    for bar, m in zip(bars, methods):
        bar.set_facecolor(color_map.get(m, COLORS["baseline"]))
        bar.set_hatch(hatch_map.get(m, ""))

    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{ppl:.2f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_ylabel("Perplexity (\u2193 is better)")

    # Dynamic title based on what methods are actually present
    method_names = " vs ".join(methods)
    ax.set_title(f"WikiText-2 Perplexity: {method_names}")

    fp16_ppl = results["FP16"]["ppl"]
    ax.axhline(fp16_ppl, color="#333333", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.text(len(methods) - 0.5, fp16_ppl + 0.1, "FP16 baseline",
            fontsize=8, color="#555555")

    ax.set_ylim(0, max(ppls) * 1.15)

    save_or_show(fig, "fig7_11_gptq_perplexity", cfg)

# ============================================================================
# Experiment 5: Production Deployment with gptqmodel
# ============================================================================

def run_gptq_deployment(cfg: Config):
    """Experiment 5: Production quantization with gptqmodel (successor to archived auto-gptq)."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 5: gptqmodel Production Deployment")
    print("=" * 72)

    if cfg.device == "cpu":
        print("  Skipping gptqmodel (requires CUDA)")
        return None

    from gptqmodel import GPTQModel, QuantizeConfig

    model_name = cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare calibration data: list of tokenized dicts
    print(f"\nPreparing calibration data ({cfg.calib_samples} samples)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    calib_data = []
    for text in ds["text"]:
        if len(text.strip()) > 100:
            enc = tokenizer(text, return_tensors=None,
                          max_length=cfg.calib_seq_len, truncation=True)
            if len(enc["input_ids"]) >= 128:
                calib_data.append({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                })
                if len(calib_data) >= cfg.calib_samples:
                    break

    print(f"  Collected {len(calib_data)} calibration sequences")

    # --- Listing 7.8 candidate: GPTQ production deployment ---
    print(f"\nQuantizing {model_name} with gptqmodel (4-bit, g128)...")
    mem_before = gpu_mem_mb()

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=128,
        damp_percent=0.01,         # The 1% dampening from our algorithm
        desc_act=False,            # Column order: sequential (not activation-ordered)
    )

    # Download PyTorch weights only. Skip if already present.
    # local_dir_use_symlinks=False prevents double-writing to HF cache.
    local_model_path = "./opt-6.7b-pt"
    if not os.path.isdir(local_model_path):
        from huggingface_hub import snapshot_download
        snapshot_download(
            model_name,
            local_dir=local_model_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["flax_*", "tf_*", "*.h5", "*.msgpack"],
        )
        print(f"  Downloaded PyTorch weights to {local_model_path}")
    else:
        print(f"  Using cached weights at {local_model_path}")

    model = GPTQModel.load(
        local_model_path,
        quantize_config,
    )

    t0 = time.time()
    model.quantize(calib_data)
    t_quant = time.time() - t0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    mem_after = gpu_mem_mb()
    print(f"  Quantization time: {t_quant:.1f}s")
    print(f"  GPU memory after quantization: {mem_after:.0f} MB")

    # Save quantized model — required before inference (weights are on meta
    # device after quantize(); save + reload materializes them onto GPU)
    save_path = "./opt-6.7b-gptq-4bit"
    print(f"\n  Saving quantized model to {save_path}...")
    model.save_quantized(save_path)
    tokenizer.save_pretrained(save_path)

    # Clean up offload dir AFTER save — gptqmodel reads it during save_quantized
    import shutil
    if os.path.isdir("./gptqmodel_offload"):
        shutil.rmtree("./gptqmodel_offload")
        print("  Cleaned up gptqmodel_offload/")

    del model
    free_gpu()

    # Reload for inference
    print(f"  Loading quantized model for inference...")
    _basename = get_gptq_basename(save_path)
    model_inf = GPTQModel.from_quantized(save_path, device="cuda:0", model_basename=_basename, backend="torch")

    # Quick generation test
    print(f"\n  Generation test:")
    prompt = "The advantage of GPTQ over round-to-nearest quantization is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model_inf.generate(**inputs, max_new_tokens=50, do_sample=False)
    generated = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"    Prompt: {prompt}")
    print(f"    Output: {generated[:200]}")

    result = {
        "model": model_inf,
        "tokenizer": tokenizer,
        "quant_time": t_quant,
        "save_path": save_path,
    }
    return result

# ============================================================================
# Conceptual Diagram: GPTQ Column-wise Quantization Flow
# ============================================================================

def plot_gptq_algorithm_flow(cfg: Config):
    """Figure 7.7: GPTQ algorithm schematic — column-wise quantization with error compensation."""
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("GPTQ Algorithm: Column-wise Quantization with Error Compensation",
                 fontsize=12, pad=20)

    def box(x, y, w, h, text, fc, ec="black", ls="-", lw=1.2, fs=9):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold")

    def arrow(x1, y1, x2, y2, text="", color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        if text:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my + 0.2, text, fontsize=8, ha="center",
                    color=color, style="italic")

    # Weight matrix W
    box(0.2, 1.5, 1.6, 2.5, "W\n[out, in]\nFP16 weights", "#e8e8e8", fs=9)

    # Calibration data → Hessian
    box(0.2, 4.5, 1.6, 0.8, "X (calib)\n128 samples", "#dde8d6", fs=8)
    arrow(1.0, 4.5, 1.0, 4.1)
    box(2.4, 4.2, 2.0, 0.8, "H = X·Xᵀ\nHessian", "#ccdcec", fs=9)
    arrow(3.4, 4.2, 3.4, 3.8)
    box(2.4, 3.0, 2.0, 0.8, "Cholesky(H⁻¹)\nnumerical\nstability", "#ccdcec", fs=8)

    # Arrow to main loop
    arrow(1.8, 2.75, 5.0, 2.75)
    arrow(3.4, 3.0, 5.0, 2.75)

    # Main loop: Block processing
    box(5.0, 1.5, 3.5, 2.5, "", "#f5f5f5", ls="--", lw=1.0)
    ax.text(6.75, 3.8, "Process block of B=128 columns", fontsize=9,
            ha="center", style="italic", color="#333333")

    # Inside block: column quantization
    box(5.2, 2.8, 1.4, 0.8, "Quantize\ncolumn j", "#b3cde3", fs=8)
    arrow(6.6, 3.2, 7.2, 3.2, "err")
    box(7.2, 2.8, 1.1, 0.8, "Error\n÷ H⁻¹ⱼⱼ", "#d9e6f2", fs=8)

    # Error compensation
    box(5.2, 1.7, 3.1, 0.8, "Update remaining cols:\nW[:,j:] -= err × H⁻¹[j,j:]",
        "#fff2cc", ec="#333333", fs=8)

    # Arrow to next block
    arrow(8.5, 2.75, 9.5, 2.75, "next block")

    # Output
    box(9.5, 1.5, 2.0, 2.5, "Q\n[out, in]\nINT4 weights\n+ scales", "#d6eaf8", fs=9)

    # Arrow to output
    arrow(11.5, 2.75, 12.5, 2.75)
    box(12.5, 2.0, 1.2, 1.5, "W_q\nready for\ninference", "#e8e8e8", fs=9)

    # Key insight annotation
    ax.text(6.75, 0.5,
            "Key: error from quantizing column j is compensated in columns j+1..d\n"
            "using Hessian curvature — directions the model is sensitive to get larger corrections",
            fontsize=8, ha="center", color="#1B5299", style="italic",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="#1B5299"))

    save_or_show(fig, "fig7_7_gptq_algorithm_flow", cfg)

# ============================================================================
# Main
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Chapter 7.3 — GPTQ Hessian-Aware Groupwise Quantization"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "single-layer", "layer-sweep", "groupsize",
                 "deploy", "perplexity", "flow"],
        help="Which experiment(s) to run",
    )
    parser.add_argument("--model", type=str, default="opt-6.7b",
                       choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--plot-dir", type=str, default="ch7_plots")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--groupsize", type=int, default=-1)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--calib-samples", type=int, default=128)
    return parser.parse_args()

def main():
    args = parse_args()

    # Suppress noisy third-party loggers
    import logging
    for _logger in ("httpx", "httpcore", "tokenicer", "huggingface_hub", "gptqmodel"):
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
    )

    set_seed(cfg.seed)
    setup_manning_style()

    print(f"Configuration:")
    print(f"  Model:    {cfg.model_name}")
    print(f"  Device:   {cfg.device}")
    print(f"  Bits:     {cfg.wbits}")
    print(f"  Blocksize: {cfg.blocksize}")
    print(f"  Groupsize: {cfg.groupsize}")
    print(f"  GPU:      {torch.cuda.get_device_name() if cfg.device == 'cuda' else 'N/A'}")
    if cfg.device == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  VRAM:     {total_mem:.0f} MB")
    print()

    mode = args.mode

    # Flow diagram (no model needed)
    if mode in ("all", "flow"):
        plot_gptq_algorithm_flow(cfg)

    # Experiment 1: Single-layer from scratch
    if mode in ("all", "single-layer"):
        results, H_diag = run_single_layer_experiment(cfg)
        plot_hessian_and_single_layer(results, H_diag, cfg)

    # Experiment 2: Per-layer sweep
    if mode in ("all", "layer-sweep"):
        layer_results = run_layer_sweep(cfg)
        plot_layer_sweep(layer_results, cfg)

    # Experiment 3: Group size sweep
    if mode in ("all", "groupsize"):
        sweep_results = run_groupsize_sweep(cfg)
        plot_groupsize_sweep(sweep_results, cfg)

    # Experiment 4: Perplexity showdown
    if mode in ("all", "perplexity"):
        ppl_results = run_perplexity_comparison(cfg)
        plot_perplexity_comparison(ppl_results, cfg)

    # Experiment 5: gptqmodel deployment (run LAST — quantized
    # modules hold CUDA memory that is difficult to fully reclaim)
    if mode in ("all", "deploy"):
        gptq_result = run_gptq_deployment(cfg)
        if gptq_result and "model" in gptq_result:
            del gptq_result["model"]
        free_gpu()

    print("\n" + "=" * 72)
    print("All experiments complete.")
    if cfg.save_plots:
        print(f"Figures saved to: {cfg.plot_dir}/")
        print(f"  Each figure saved as .pdf (vector) and .png (raster)")
    print("=" * 72)

if __name__ == "__main__":
    main()