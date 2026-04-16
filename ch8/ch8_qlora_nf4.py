"""
Chapter 8, Section 8.4 — QLoRA with NF4
Companion script for "Efficient AI in Practice: Quantization and Fast Inference"

Modes:
  --mode codebook   Build the NF4 codebook from normal quantiles, compare to INT4/FP4
  --mode mse        Weight reconstruction MSE: NF4 vs INT4 vs FP4 on OPT-6.7B
  --mode qlora      Full QLoRA fine-tuning on Llama-2-7B (requires bitsandbytes, peft)
  --mode memory     Memory breakdown: NF4 vs FP16 loading of Llama-2-7B
  --mode all        Run all modes

Usage:
  # Format exploration only (CPU-safe, no model needed)
  python ch8_qlora_nf4.py --mode codebook --save-plots

  # Weight reconstruction on H100 (uses OPT-6.7B)
  python ch8_qlora_nf4.py --mode mse --save-plots

  # Full QLoRA pipeline on H100 (uses Llama-2-7B, requires bitsandbytes + peft)
  python ch8_qlora_nf4.py --mode qlora --save-plots

  # Memory analysis only (loads model, no training)
  python ch8_qlora_nf4.py --mode memory --save-plots

  # CPU-only fallback for MSE mode (illustrative only)
  python ch8_qlora_nf4.py --mode mse --save-plots --mse-model opt-125m --device cpu

Requires:
  All modes:     torch >= 2.4, matplotlib, numpy, scipy
  mse mode:      transformers, datasets
  qlora mode:    transformers, datasets, bitsandbytes >= 0.43, peft >= 0.7
  memory mode:   transformers, bitsandbytes >= 0.43

Note: Llama-2-7B requires accepting Meta's license at
      https://huggingface.co/meta-llama/Llama-2-7b-hf
"""

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ─── Configuration ───────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

MSE_MODEL_MAP = {
    "opt-6.7b": "facebook/opt-6.7b",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-350m": "facebook/opt-350m",
    "opt-125m": "facebook/opt-125m",
}

QLORA_MODEL = "meta-llama/Llama-2-7b-hf"


@dataclass
class Config:
    mode: str = "all"
    mse_model: str = "opt-6.7b"
    device: str = "auto"
    save_plots: bool = False
    output_dir: Path = SCRIPT_DIR / "figures"
    num_calibration_seqs: int = 64
    seq_length: int = 512
    seed: int = 42
    # QLoRA training settings
    qlora_max_steps: int = 200
    qlora_train_samples: int = 1000
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-4


# ─── Manning figure style ────────────────────────────────────────────────────

COLORS = {
    "nf4":      "#d95f02",   # Orange
    "nf4_b64":  "#d95f02",   # Orange (block-64, bnb default)
    "int4":     "#1b9e77",   # Teal
    "int4_g128":"#66a61e",   # Olive
    "fp4":      "#e6ab02",   # Gold
    "fp4_b16":  "#e6ab02",   # Gold (block-16)
    "fp8":      "#2166ac",   # Blue
    "bf16":     "#7570b3",   # Purple
    "lora":     "#e7298a",   # Magenta
    "optim":    "#a6761d",   # Brown
    "activ":    "#666666",   # Gray
}

HATCHES = {
    "nf4":      "",
    "nf4_b64":  "",
    "int4":     "++",
    "int4_g128":"//",
    "fp4":      "\\\\",
    "fp4_b16":  "\\\\",
    "fp8":      "||",
    "bf16":     "..",
    "lora":     "xx",
    "optim":    "//",
    "activ":    "\\\\",
}


def apply_manning_style():
    """Apply Manning Publications figure style guidelines."""
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
        "figure.figsize": (5.6, 3.5),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,      # TrueType fonts in PDF
        "ps.fonttype": 42,
    })


def save_or_show(fig, name: str, config: Config):
    """Save figure in both PNG (300 DPI) and PDF (fonttype 42), or show."""
    if config.save_plots:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        png_path = config.output_dir / f"{name}.png"
        pdf_path = config.output_dir / f"{name}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        fig.savefig(pdf_path, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")
    else:
        plt.show()
    plt.close(fig)


# ─── NF4 codebook specification ─────────────────────────────────────────────

def build_nf4_codebook(offset: float = 0.9677083) -> np.ndarray:
    """Build the NF4 codebook from normal distribution quantiles.

    The construction follows Dettmers et al. (2023), QLoRA:
    1. Weights are normalized to [-1, 1] per block (absmax scaling).
    2. The normalized distribution is approximately N(0, σ) truncated to
       [-1, 1]. NF4 places 16 quantization levels at quantile positions
       that minimize expected quantization error for this distribution.
    3. Zero must be exactly representable (for weight sparsity).
    4. The remaining 15 slots split asymmetrically: 7 negative, 8 positive.
       The extra positive slot gives finer resolution where most of the
       probability mass lies after absmax normalization.                      #A
    """
    from scipy.stats import norm

    # ── Step 1: Compute quantile positions ──
    # The offset (≈0.968) determines the outermost quantile. The positive
    # and negative sides use different numbers of bins: 8 positive (more
    # resolution near the peak) and 7 negative.

    # Positive side: 8 levels from quantile positions in (0.5, offset]       #B
    pos_quantiles = np.linspace(0.5, offset, 9)[1:]   # 8 values
    pos_levels = norm.ppf(pos_quantiles)

    # Negative side: 7 levels from quantile positions in [1-offset, 0.5)
    neg_quantiles = np.linspace(1 - offset, 0.5, 8)[:-1]  # 7 values
    neg_levels = norm.ppf(neg_quantiles)

    # ── Step 2: Combine and normalize to [-1, 1] ──                         #C
    all_levels = np.sort(np.concatenate([neg_levels, [0.0], pos_levels]))
    all_levels = all_levels / np.abs(all_levels).max()

    return all_levels

#A The asymmetric split (7 neg + 0 + 8 pos = 16 levels) is deliberate.
#  Neural network weights after absmax normalization cluster near zero
#  with slight positive skew. The extra positive level gives finer
#  resolution where a majority of weight values land.
#B norm.ppf is the inverse CDF (percent-point function). ppf(0.5) = 0.
#  linspace(0.5, 0.968, 9)[1:] spaces 8 quantile positions uniformly
#  across the right half of the CDF, each covering equal probability mass.
#C Rescaling to [-1, 1] aligns the extreme levels with ±1, matching the
#  absmax normalization used during blockwise quantization.


# The reference NF4 codebook from bitsandbytes (for validation)
BNB_NF4_CODEBOOK = np.array([
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844,
    -0.1848, -0.0911, 0.0,
     0.0796,  0.1609,  0.2461,  0.3379,
     0.4407,  0.5626,  0.7230,  1.0,
])


# ─── NF4 quantization (from scratch) ────────────────────────────────────────

def quantize_nf4_block(tensor: torch.Tensor,
                        codebook: torch.Tensor,
                        block_size: int = 64) -> torch.Tensor:
    """Quantize a weight tensor to NF4 with blockwise absmax scaling.

    This reproduces the bitsandbytes NF4 quantization path:
    1. Partition tensor into blocks of `block_size` elements
    2. Compute absmax per block → scale factor
    3. Normalize to [-1, 1]
    4. Snap to nearest NF4 codebook entry
    5. Dequantize back to float                                              #D
    """
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, orig_shape[-1]).float()
    n_cols = flat.shape[-1]

    # Pad if not divisible by block_size
    pad_size = (block_size - n_cols % block_size) % block_size
    if pad_size > 0:
        flat = F.pad(flat, (0, pad_size))

    blocked = flat.reshape(flat.shape[0], -1, block_size)

    # Per-block absmax scaling
    amax = blocked.abs().amax(dim=-1)
    scales = amax.clamp(min=1e-12)

    # Normalize to [-1, 1]
    normalized = blocked / scales.unsqueeze(-1)
    normalized = normalized.clamp(-1.0, 1.0)

    # Snap to nearest NF4 level (vectorized lookup)                          #E
    cb = codebook.to(tensor.device)
    norm_flat = normalized.reshape(-1, 1)
    distances = (norm_flat - cb.unsqueeze(0)).abs()
    nearest_idx = distances.argmin(dim=1)
    quantized_flat = cb[nearest_idx]
    quantized = quantized_flat.reshape(blocked.shape)

    # Dequantize: restore original scale
    dequantized = quantized * scales.unsqueeze(-1)

    # Reshape back, strip padding
    flat_deq = dequantized.reshape(blocked.shape[0], -1)
    if pad_size > 0:
        flat_deq = flat_deq[:, :n_cols]
    return flat_deq.reshape(orig_shape)

#D bitsandbytes stores the 4-bit indices packed into bytes and performs
#  dequantization in custom CUDA kernels before the BF16 matmul. We
#  simulate the same rounding error by snapping to the codebook in float.
#  The quantization error is identical; only the storage format differs.
#E The codebook has 16 entries. The lookup is O(n × 16) — trivial
#  compared to the O(n × 8) lookup used for FP4 E2M1 in Section 8.3.


# ─── INT4 and FP4 quantization (for comparison, from §8.3) ──────────────────

def quantize_int4_per_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Symmetric per-tensor INT4 quantization."""
    amax = tensor.abs().max()
    scale = amax / 7.0 if amax > 0 else torch.tensor(1.0)
    quantized = torch.round(tensor / scale).clamp(-8, 7)
    return quantized * scale


def quantize_int4_group(tensor: torch.Tensor,
                         group_size: int = 128) -> torch.Tensor:
    """Symmetric groupwise INT4 quantization (GPTQ/AWQ style)."""
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, orig_shape[-1]).float()
    n_cols = flat.shape[-1]

    pad_size = (group_size - n_cols % group_size) % group_size
    if pad_size > 0:
        flat = F.pad(flat, (0, pad_size))

    grouped = flat.reshape(flat.shape[0], -1, group_size)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 7.0).clamp(min=1e-12)
    quantized = torch.round(grouped / scale).clamp(-8, 7)
    dequantized = quantized * scale

    flat_deq = dequantized.reshape(flat.shape[0], -1)
    if pad_size > 0:
        flat_deq = flat_deq[:, :n_cols]
    return flat_deq.reshape(orig_shape)


def quantize_fp4_block(tensor: torch.Tensor,
                        block_size: int = 16) -> torch.Tensor:
    """FP4 E2M1 with blockwise scaling (from Section 8.3)."""
    fp4_max = 6.0
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, orig_shape[-1]).float()
    n_cols = flat.shape[-1]

    pad_size = (block_size - n_cols % block_size) % block_size
    if pad_size > 0:
        flat = F.pad(flat, (0, pad_size))

    blocked = flat.reshape(flat.shape[0], -1, block_size)

    amax = blocked.abs().amax(dim=-1)
    scales = (amax / fp4_max).clamp(min=1e-12)

    scaled = blocked / scales.unsqueeze(-1)
    scaled = scaled.clamp(-fp4_max, fp4_max)

    signs = scaled.sign()
    abs_vals = scaled.abs()

    pos_levels = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32, device=tensor.device,
    )
    abs_flat = abs_vals.reshape(-1, 1)
    distances = (abs_flat - pos_levels.unsqueeze(0)).abs()
    nearest_idx = distances.argmin(dim=1)
    quantized_flat = pos_levels[nearest_idx]
    quantized = quantized_flat.reshape(scaled.shape) * signs

    dequantized = quantized * scales.unsqueeze(-1)
    flat_deq = dequantized.reshape(blocked.shape[0], -1)
    if pad_size > 0:
        flat_deq = flat_deq[:, :n_cols]
    return flat_deq.reshape(orig_shape)


def quantize_fp8_e4m3_per_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """FP8 E4M3 per-tensor quantization (from Section 8.2)."""
    fp8_max = 448.0
    amax = tensor.abs().max().item()
    if amax == 0:
        return tensor.clone()
    scale = amax / fp8_max
    scaled = (tensor.float() / scale).clamp(-fp8_max, fp8_max)
    try:
        quantized = scaled.to(torch.float8_e4m3fn).float()
    except RuntimeError:
        quantized = scaled.cpu().to(torch.float8_e4m3fn).float().to(
            tensor.device)
    return quantized * scale


# ─── FP4 E2M1 value enumeration (for number line) ───────────────────────────

def enumerate_fp4_e2m1() -> np.ndarray:
    """Enumerate all FP4 E2M1 representable values."""
    E, M, bias = 2, 1, 1
    values = []
    for bits in range(16):
        sign = (bits >> 3) & 1
        exp_field = (bits >> M) & ((1 << E) - 1)
        mant_field = bits & ((1 << M) - 1)
        if exp_field == 0:
            value = (mant_field / (1 << M)) * (2.0 ** (1 - bias))
        else:
            value = (1.0 + mant_field / (1 << M)) * (2.0 ** (exp_field - bias))
        if sign:
            value = -value
        values.append(value)
    return np.array(sorted(set(values)))


# ─── Model and data loading ──────────────────────────────────────────────────

def resolve_device(config: Config) -> str:
    if config.device != "auto":
        return config.device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_gpu_capability() -> tuple:
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return (0, 0)


def load_opt_model_and_tokenizer(config: Config, device: str):
    """Load OPT model and tokenizer for MSE experiments."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MSE_MODEL_MAP.get(config.mse_model, config.mse_model)
    print(f"\n  Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    cap = get_gpu_capability()
    if cap >= (8, 0):
        model_dtype = torch.bfloat16
    elif cap >= (7, 0):
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.float()

    model.eval()
    dtype_name = {torch.bfloat16: "BF16", torch.float16: "FP16",
                  torch.float32: "FP32"}[model_dtype]
    print(f"  Model dtype: {dtype_name}")
    print(f"  Model device: {next(model.parameters()).device}")

    return model, tokenizer, model_dtype


def gpu_memory_mb() -> float:
    """Current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


def gpu_memory_gb() -> float:
    """Current GPU memory allocated in GB."""
    return gpu_memory_mb() / 1024


def compute_real_param_counts(model_id: str) -> dict:
    """Compute parameter counts from model config, not from loaded tensors.

    When a model is loaded in 4-bit via bitsandbytes, p.numel() returns
    the packed count (~half the real count). This function derives the
    real parameter count from the architecture config, which is correct
    regardless of how weights are stored in memory.                          #P
    """
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_id)

    h = config.hidden_size
    i = config.intermediate_size
    n = config.num_hidden_layers
    v = config.vocab_size

    # Per transformer layer
    attn = 4 * h * h           # q, k, v, o projections
    mlp = 3 * h * i            # gate, up, down projections
    ln = 2 * h                 # input + post-attention layer norms
    linear_per_layer = attn + mlp
    all_per_layer = linear_per_layer + ln

    # Model-level
    embed = v * h              # token embedding (nn.Embedding, not linear)
    final_ln = h               # final RMSNorm
    lm_head = v * h            # output projection (nn.Linear)

    total_linear = n * linear_per_layer + lm_head
    total_other = n * ln + embed + final_ln
    total_all = total_linear + total_other

    return {
        "total": total_all,
        "linear": total_linear,
        "other": total_other,
        "hidden_size": h,
        "intermediate_size": i,
        "num_layers": n,
        "vocab_size": v,
    }

#P The packed-parameter trap: bitsandbytes stores 4-bit weights as
#  uint8 tensors with two values per byte. A (4096, 4096) weight matrix
#  becomes a (4096, 2048) uint8 tensor, so p.numel() returns 8M instead
#  of 16M. Every calculation that feeds on numel() — FP16 baselines,
#  compression ratios, LoRA percentages — inherits the error. Computing
#  from the config avoids this entirely.


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: codebook — NF4 codebook construction and number line comparison
# ═══════════════════════════════════════════════════════════════════════════════

def run_codebook(config: Config):
    print("\n" + "=" * 70)
    print("MODE: codebook — NF4 codebook from normal quantiles")
    print("=" * 70)

    # ── Step 1: Build the codebook ──
    nf4_codebook = build_nf4_codebook()

    print(f"\n  NF4 codebook (16 levels, from normal quantiles):")
    print(f"  {'Index':<8} {'Level':<12} {'Type':<12}")
    print(f"  {'─' * 32}")
    for i, v in enumerate(nf4_codebook):
        if v == 0.0:
            vtype = "zero"
        elif v < 0:
            vtype = "negative"
        else:
            vtype = "positive"
        print(f"  {i:<8} {v:<12.4f} {vtype:<12}")

    n_neg = np.sum(nf4_codebook < 0)
    n_pos = np.sum(nf4_codebook > 0)
    print(f"\n  Layout: {n_neg} negative + zero + {n_pos} positive = "
          f"{len(nf4_codebook)} levels")

    # ── Step 2: Validate against bitsandbytes reference ──                  #F
    max_dev = np.abs(nf4_codebook - BNB_NF4_CODEBOOK).max()
    print(f"\n  Validation vs bitsandbytes reference codebook:")
    print(f"  Max absolute deviation: {max_dev:.4f}")
    print(f"  {'Index':<8} {'Ours':<12} {'bnb ref':<12} {'Δ':<12}")
    print(f"  {'─' * 44}")
    for i, (ours, ref) in enumerate(zip(nf4_codebook, BNB_NF4_CODEBOOK)):
        print(f"  {i:<8} {ours:<12.4f} {ref:<12.4f} {ours - ref:<12.4f}")

    #F The slight deviations from bitsandbytes arise because bnb uses a
    #  precomputed table with truncated precision, while we compute from
    #  scipy.stats.norm.ppf in float64. Both are valid NF4 implementations.

    # ── Step 3: Spacing analysis ──
    # NF4 spacing is non-uniform AND non-exponential — it follows the
    # normal density. Denser near zero, sparser in the tails.
    gaps = np.diff(nf4_codebook)
    print(f"\n  Spacing between consecutive NF4 levels:")
    for i, (v, gap) in enumerate(zip(nf4_codebook[:-1], gaps)):
        print(f"    {v:+.4f} → {nf4_codebook[i+1]:+.4f}:  gap = {gap:.4f}")

    # Compare to INT4 uniform
    int4_levels = np.linspace(-1, 1, 16)
    int4_gap = int4_levels[1] - int4_levels[0]
    print(f"\n  INT4 uniform spacing: {int4_gap:.4f} (constant)")
    print(f"  NF4 smallest gap (near zero): {gaps.min():.4f}")
    print(f"  NF4 largest gap (tails): {gaps.max():.4f}")
    print(f"  NF4 is {int4_gap / gaps.min():.1f}× denser near zero than INT4")

    # ── Step 4: MSE comparison on synthetic Gaussian data ──                #G
    # Draw 100,000 samples from N(0, 1), normalize to [-1, 1] via absmax,
    # and quantize with NF4 vs INT4 uniform vs FP4 E2M1
    np.random.seed(config.seed)
    samples = np.random.randn(100_000).astype(np.float32)
    samples_norm = samples / np.abs(samples).max()

    def quantize_to_codebook(data, codebook):
        """Snap each value to the nearest codebook entry."""
        data_2d = data.reshape(-1, 1)
        cb_2d = codebook.reshape(1, -1)
        idx = np.abs(data_2d - cb_2d).argmin(axis=1)
        return codebook[idx]

    # NF4
    recon_nf4 = quantize_to_codebook(samples_norm, nf4_codebook)
    mse_nf4 = np.mean((samples_norm - recon_nf4) ** 2)

    # INT4 uniform 16 levels in [-1, 1]
    recon_int4 = quantize_to_codebook(samples_norm, int4_levels)
    mse_int4 = np.mean((samples_norm - recon_int4) ** 2)

    # FP4 E2M1 (normalized to [-1, 1])
    fp4_vals = enumerate_fp4_e2m1()
    fp4_norm = fp4_vals / np.abs(fp4_vals).max()  # scale to [-1, 1]
    recon_fp4 = quantize_to_codebook(samples_norm, fp4_norm)
    mse_fp4 = np.mean((samples_norm - recon_fp4) ** 2)

    print(f"\n  Quantization MSE on N(0,1) data (100K samples, normalized):")
    print(f"  {'Format':<20} {'MSE':<14} {'vs NF4':<12}")
    print(f"  {'─' * 46}")
    print(f"  {'NF4 (quantile)':<20} {mse_nf4:<14.6f} {'1.00×':<12}")
    print(f"  {'INT4 (uniform)':<20} {mse_int4:<14.6f} "
          f"{mse_int4/mse_nf4:.2f}×")
    print(f"  {'FP4 E2M1':<20} {mse_fp4:<14.6f} "
          f"{mse_fp4/mse_nf4:.2f}×")

    #G This test isolates the codebook design from the scaling strategy.
    #  All three formats are compared on the same normalized data with no
    #  block or group scaling. NF4 wins because its levels are placed at
    #  the information-theoretically optimal positions for Gaussian data.

    # ── Figure 8.12: NF4 codebook number line ──
    apply_manning_style()

    # Panel 1: Full range [-1, 1] — all three formats
    fig, axes = plt.subplots(3, 1, figsize=(5.6, 3.0), sharex=True)

    for ax, vals, name, color in [
        (axes[0], nf4_codebook,
         f"NF4 ({len(nf4_codebook)} levels, normal quantiles)",
         COLORS["nf4"]),
        (axes[1], int4_levels,
         f"INT4 uniform ({len(int4_levels)} levels)",
         COLORS["int4"]),
        (axes[2], fp4_norm,
         f"FP4 E2M1 ({len(fp4_norm)} levels, normalized)",
         COLORS["fp4"]),
    ]:
        ax.eventplot([vals], lineoffsets=0, linelengths=0.6,
                     colors=[color], linewidths=0.8)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=6, rotation=0, ha="right", va="center",
                      labelpad=5)
        ax.set_xlim(-1.15, 1.15)
        ax.spines["left"].set_visible(False)

    axes[2].set_xlabel("Normalized value")
    fig.suptitle(
        "NF4, INT4, and FP4 quantization levels in [-1, 1]",
        fontsize=9, y=0.98)
    fig.tight_layout(rect=[0.30, 0.0, 1.0, 0.95])

    save_or_show(fig, "CH08_F08_Kalyanarangan", config)

    # Panel 2: Zoomed [-0.3, 0.3] — where most weights live
    fig2, axes2 = plt.subplots(3, 1, figsize=(5.6, 3.0), sharex=True)

    for ax, vals, name, color in [
        (axes2[0], nf4_codebook, "NF4", COLORS["nf4"]),
        (axes2[1], int4_levels, "INT4 uniform", COLORS["int4"]),
        (axes2[2], fp4_norm, "FP4 E2M1", COLORS["fp4"]),
    ]:
        mask = (vals >= -0.3) & (vals <= 0.3)
        v = vals[mask]
        ax.eventplot([v], lineoffsets=0, linelengths=0.6,
                     colors=[color], linewidths=0.8)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=7, rotation=0, ha="right",
                      va="center", labelpad=5)
        ax.set_xlim(-0.35, 0.35)
        ax.spines["left"].set_visible(False)

        ax.text(0.98, 0.85, f"{len(v)} levels in [-0.3, 0.3]",
                transform=ax.transAxes, fontsize=6, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

    axes2[2].set_xlabel("Normalized value")
    fig2.suptitle("Near-zero density: levels in [-0.3, 0.3]",
                  fontsize=9, y=0.98)
    fig2.tight_layout(rect=[0.15, 0.0, 1.0, 0.95])

    save_or_show(fig2, "CH08_F08b_Kalyanarangan_zoom", config)

    # Density statistics
    print("\n  Density statistics:")
    print(f"  {'Range':<18} {'NF4':<10} {'INT4':<10} {'FP4 E2M1':<10}")
    print(f"  {'─' * 48}")
    for lo, hi in [(-0.1, 0.1), (-0.2, 0.2), (-0.3, 0.3), (-0.5, 0.5)]:
        c_nf4 = np.sum((nf4_codebook >= lo) & (nf4_codebook <= hi))
        c_int4 = np.sum((int4_levels >= lo) & (int4_levels <= hi))
        c_fp4 = np.sum((fp4_norm >= lo) & (fp4_norm <= hi))
        rng = f"[{lo}, {hi}]"
        print(f"  {rng:<18} {c_nf4:<10} {c_int4:<10} {c_fp4:<10}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: mse — Weight reconstruction MSE: NF4 vs INT4 vs FP4 on OPT-6.7B
# ═══════════════════════════════════════════════════════════════════════════════

def run_mse(config: Config):
    print("\n" + "=" * 70)
    print("MODE: mse — Weight reconstruction: NF4 vs INT4 vs FP4 ({})".format(
        config.mse_model))
    print("=" * 70)

    device = resolve_device(config)
    model, tokenizer, model_dtype = load_opt_model_and_tokenizer(
        config, device)

    nf4_codebook = build_nf4_codebook()
    nf4_cb_tensor = torch.tensor(nf4_codebook, dtype=torch.float32)

    num_layers = len(model.model.decoder.layers)
    layer_indices = list(range(num_layers))

    results = []
    for idx in layer_indices:
        layer = model.model.decoder.layers[idx]
        w = layer.fc1.weight.detach().float().cpu()

        # NF4 block-64 (bitsandbytes default block size)                     #H
        deq_nf4 = quantize_nf4_block(w, nf4_cb_tensor, block_size=64)
        mse_nf4 = F.mse_loss(deq_nf4, w).item()

        # INT4 symmetric group-128 (GPTQ/AWQ standard)
        deq_int4_g128 = quantize_int4_group(w, group_size=128)
        mse_int4_g128 = F.mse_loss(deq_int4_g128, w).item()

        # INT4 symmetric group-64 (matched block size for fair comparison)
        deq_int4_g64 = quantize_int4_group(w, group_size=64)
        mse_int4_g64 = F.mse_loss(deq_int4_g64, w).item()

        # FP4 E2M1 block-16 (NVFP4 standard)
        deq_fp4_b16 = quantize_fp4_block(w, block_size=16)
        mse_fp4_b16 = F.mse_loss(deq_fp4_b16, w).item()

        # FP8 E4M3 per-tensor (8-bit reference)
        deq_fp8 = quantize_fp8_e4m3_per_tensor(w)
        mse_fp8 = F.mse_loss(deq_fp8, w).item()

        results.append({
            "layer": idx,
            "mse_nf4_b64": mse_nf4,
            "mse_int4_g128": mse_int4_g128,
            "mse_int4_g64": mse_int4_g64,
            "mse_fp4_b16": mse_fp4_b16,
            "mse_fp8": mse_fp8,
            "shape": tuple(w.shape),
        })

    #H Block size 64 is the bitsandbytes default for NF4. Comparing NF4
    #  block-64 against INT4 group-128 reflects the actual library defaults.
    #  INT4 group-64 isolates the effect of codebook design from block size.

    # Print results
    print(f"\n  Weight reconstruction MSE — fc1 layers ({config.mse_model})")
    print(f"  {'Layer':<7} {'NF4 b64':<12} {'INT4 g64':<12} "
          f"{'INT4 g128':<12} {'FP4 b16':<12} {'FP8':<12}")
    print(f"  {'─' * 67}")
    for r in results:
        print(f"  {r['layer']:<7} {r['mse_nf4_b64']:<12.2e} "
              f"{r['mse_int4_g64']:<12.2e} {r['mse_int4_g128']:<12.2e} "
              f"{r['mse_fp4_b16']:<12.2e} {r['mse_fp8']:<12.2e}")

    # Averages
    avg = {k: np.mean([r[k] for r in results])
           for k in ["mse_nf4_b64", "mse_int4_g64", "mse_int4_g128",
                      "mse_fp4_b16", "mse_fp8"]}
    print(f"  {'Avg':<7} {avg['mse_nf4_b64']:<12.2e} "
          f"{avg['mse_int4_g64']:<12.2e} {avg['mse_int4_g128']:<12.2e} "
          f"{avg['mse_fp4_b16']:<12.2e} {avg['mse_fp8']:<12.2e}")

    # Key comparisons
    print(f"\n  Key comparisons (average across layers):")
    nf4_v = avg["mse_nf4_b64"]
    for label, key in [("INT4 group-64 (same block size)", "mse_int4_g64"),
                       ("INT4 group-128 (GPTQ default)", "mse_int4_g128"),
                       ("FP4 block-16 (NVFP4)", "mse_fp4_b16"),
                       ("FP8 E4M3 per-tensor", "mse_fp8")]:
        ratio = avg[key] / nf4_v
        direction = "worse" if ratio > 1 else "better"
        print(f"    {label:<35} {ratio:>6.2f}× {direction} than NF4")

    # ── Figure 8.13: Weight MSE across formats ──
    apply_manning_style()

    # Subsample layers for readability if > 16
    if len(results) > 16:
        step = max(1, len(results) // 8)
        plot_results = results[::step]
    else:
        plot_results = results

    fig, ax = plt.subplots(figsize=(5.6, 3.5))

    layers = [r["layer"] for r in plot_results]
    x = np.arange(len(layers))
    width = 0.15

    configs_plot = [
        ("NF4 block-64", "mse_nf4_b64",
         COLORS["nf4_b64"], HATCHES["nf4_b64"]),
        ("INT4 group-64", "mse_int4_g64",
         COLORS["int4"], "//"),
        ("INT4 group-128", "mse_int4_g128",
         COLORS["int4_g128"], HATCHES["int4_g128"]),
        ("FP4 block-16", "mse_fp4_b16",
         COLORS["fp4_b16"], HATCHES["fp4_b16"]),
        ("FP8 E4M3", "mse_fp8",
         COLORS["fp8"], HATCHES["fp8"]),
    ]

    for i, (label, key, color, hatch) in enumerate(configs_plot):
        offset = x + (i - len(configs_plot) / 2 + 0.5) * width
        ax.bar(offset, [r[key] for r in plot_results], width,
               label=label, color=color, hatch=hatch,
               edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Weight MSE (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(loc="upper right", framealpha=0.9, fontsize=6)
    ax.set_title(
        f"NF4 vs INT4 vs FP4: weight quantization error "
        f"({config.mse_model} fc1)")

    fig.tight_layout()
    save_or_show(fig, "CH08_F09_Kalyanarangan", config)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: qlora — Full QLoRA fine-tuning on Llama-2-7B
# ═══════════════════════════════════════════════════════════════════════════════

def run_qlora(config: Config):
    print("\n" + "=" * 70)
    print("MODE: qlora — QLoRA fine-tuning (Llama-2-7B, NF4 + LoRA)")
    print("=" * 70)

    # ── Dependency check ──
    try:
        import bitsandbytes as bnb
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer,
            BitsAndBytesConfig, TrainingArguments, Trainer,
            DataCollatorForLanguageModeling,
        )
        from datasets import load_dataset
    except ImportError as e:
        print(f"\n  ERROR: Missing dependency: {e}")
        print(f"  QLoRA mode requires: bitsandbytes >= 0.43, peft >= 0.7")
        print(f"  Install: pip install bitsandbytes peft trl")
        return

    device = resolve_device(config)
    if device != "cuda":
        print("\n  ERROR: QLoRA training requires a CUDA GPU.")
        print("  Use --mode codebook or --mode mse for CPU-safe experiments.")
        return

    cap = get_gpu_capability()
    torch.manual_seed(config.seed)

    # ── Step 1: Load model in NF4 ──                                        #I
    print(f"\n  Step 1: Loading {QLORA_MODEL} in NF4...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    mem_before = gpu_memory_gb()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cap >= (8, 0)
                                else torch.float16,
        bnb_4bit_use_double_quant=True,                                      #J
    )

    tokenizer = AutoTokenizer.from_pretrained(QLORA_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        QLORA_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    mem_after_load = gpu_memory_gb()
    nf4_mem = mem_after_load - mem_before

    print(f"  NF4 base model loaded: {nf4_mem:.2f} GB GPU memory")

    #I BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    #  quantizes all nn.Linear weights to NF4 at load time. The weights are
    #  stored as packed 4-bit indices with per-block (64-element) FP32 scales.
    #  During forward pass, bitsandbytes CUDA kernels dequantize to
    #  BF16/FP16 before the matmul — no FP4 tensor cores are used.
    #J Double quantization: the FP32 block scales themselves are quantized
    #  to FP8. For a 7B model with ~6.7B linear params, this saves:
    #    Scales without DQ: (6.7e9 / 64) × 4 bytes ≈ 419 MB
    #    Scales with DQ:    (6.7e9 / 64) × 1 byte  ≈ 105 MB
    #  Saving ~314 MB — small in absolute terms, but it can make the
    #  difference between fitting and not fitting on a 16 GB GPU.

    # ── Step 2: Attach LoRA adapters ──                                     #K
    print(f"\n  Step 2: Attaching LoRA adapters (r={config.lora_r})...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    mem_after_lora = gpu_memory_gb()
    lora_mem = mem_after_lora - mem_after_load

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    real_counts = compute_real_param_counts(QLORA_MODEL)
    total_real = real_counts["total"]
    trainable_pct = trainable / total_real * 100

    print(f"  Trainable parameters: {trainable:,} / {total_real:,} "
          f"({trainable_pct:.2f}%)")
    print(f"  LoRA adapter memory: {lora_mem:.2f} GB")

    #K LoRA decomposes each target weight matrix W (d × d) into W + A×B
    #  where A is (d × r) and B is (r × d). Only A and B are trainable.
    #  For Llama-2-7B with r=16 targeting all attention and MLP projections:
    #    7 target modules × 32 layers × 2 × d × r parameters
    #  The frozen NF4 base stays in 4-bit; the adapters train in BF16.

    model.print_trainable_parameters()

    # ── Step 3: Prepare dataset ──
    print(f"\n  Step 3: Loading training data "
          f"({config.qlora_train_samples} samples)...")

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.select(range(min(config.qlora_train_samples,
                                       len(dataset))))

    def format_alpaca(sample):
        """Format an Alpaca sample into an instruction-following prompt."""
        if sample.get("input", "").strip():
            text = (f"### Instruction:\n{sample['instruction']}\n\n"
                    f"### Input:\n{sample['input']}\n\n"
                    f"### Response:\n{sample['output']}")
        else:
            text = (f"### Instruction:\n{sample['instruction']}\n\n"
                    f"### Response:\n{sample['output']}")
        return text

    def tokenize_fn(sample):
        text = format_alpaca(sample)
        tokenized = tokenizer(
            text, truncation=True, max_length=512, padding="max_length",
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = dataset.map(tokenize_fn, remove_columns=dataset.column_names)

    # ── Step 4: Train ──                                                    #L
    print(f"\n  Step 4: Training ({config.qlora_max_steps} steps)...")

    output_dir = SCRIPT_DIR / "qlora_output"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=config.qlora_max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        logging_steps=10,
        save_strategy="no",
        bf16=cap >= (8, 0),
        fp16=cap < (8, 0) and cap >= (7, 0),
        optim="paged_adamw_8bit",
        report_to="none",
        seed=config.seed,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    #L paged_adamw_8bit keeps optimizer states in 8-bit with CPU paging,
    #  reducing optimizer memory from ~2× model parameters (standard Adam)
    #  to roughly 0.5× while maintaining convergence quality.

    # Record training loss
    train_result = trainer.train()
    loss_history = [
        entry["loss"] for entry in trainer.state.log_history
        if "loss" in entry
    ]
    step_history = [
        entry["step"] for entry in trainer.state.log_history
        if "loss" in entry
    ]

    mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    mem_after_train = gpu_memory_gb()

    print(f"\n  Training complete.")
    print(f"  Final loss: {loss_history[-1]:.4f}" if loss_history
          else "  No loss recorded")
    print(f"  Peak GPU memory during training: {mem_peak:.2f} GB")

    # ── Step 5: Memory summary ──                                           #M
    print(f"\n  ─── Memory summary (measured on "
          f"{torch.cuda.get_device_name()}) ───")
    print(f"  {'Stage':<35} {'GPU Memory':<15}")
    print(f"  {'─' * 50}")
    print(f"  {'Before model load':<35} {mem_before:.2f} GB")
    print(f"  {'NF4 base model loaded':<35} {mem_after_load:.2f} GB "
          f"(+{nf4_mem:.2f} GB)")
    print(f"  {'+ LoRA adapters attached':<35} {mem_after_lora:.2f} GB "
          f"(+{lora_mem:.2f} GB)")
    print(f"  {'Peak during training':<35} {mem_peak:.2f} GB")

    #M The memory breakdown reflects three components: (1) NF4 base weights
    #  (~3.5 GB for 7B params at 4 bits + double-quantized scales),
    #  (2) LoRA adapters in BF16, (3) optimizer states + gradients +
    #  activations during training. The peak is what determines whether
    #  a given GPU can run QLoRA.

    # ── Consumer GPU projection ──                                          #N
    gpu_configs = [
        ("T4 (16 GB)", 16.0),
        ("RTX 4090 (24 GB)", 24.0),
        ("A100 (40 GB)", 40.0),
        ("A100 (80 GB)", 80.0),
        ("H100 (80 GB)", 80.0),
    ]
    print(f"\n  Consumer GPU fit analysis (projected from peak {mem_peak:.1f} GB):")
    print(f"  {'GPU':<20} {'VRAM':<10} {'Fits?':<8} {'Headroom':<12}")
    print(f"  {'─' * 50}")
    for name, vram in gpu_configs:
        fits = "✓" if mem_peak < vram * 0.95 else "✗"
        headroom = vram - mem_peak
        print(f"  {name:<20} {vram:<10.0f} {fits:<8} {headroom:+.1f} GB")

    #N The peak memory determines the minimum GPU. QLoRA's value proposition
    #  is that NF4 + LoRA + 8-bit optimizer makes 7B fine-tuning fit on 16 GB
    #  consumer GPUs, where FP16 full fine-tuning would need >60 GB.

    # ── Figure 8.14: Training loss curve ──
    if loss_history:
        apply_manning_style()
        fig, ax = plt.subplots(figsize=(5.6, 2.8))

        ax.plot(step_history, loss_history,
                color=COLORS["nf4"], linewidth=1.2)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.set_title(f"QLoRA training loss (Llama-2-7B, NF4, "
                     f"r={config.lora_r})")

        # Annotate start and end loss
        ax.annotate(f"{loss_history[0]:.2f}",
                    (step_history[0], loss_history[0]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=7)
        ax.annotate(f"{loss_history[-1]:.2f}",
                    (step_history[-1], loss_history[-1]),
                    textcoords="offset points", xytext=(-30, 5),
                    fontsize=7)

        fig.tight_layout()
        save_or_show(fig, "CH08_F11_Kalyanarangan", config)

    # ── Figure 8.15: Memory breakdown ──
    apply_manning_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))

    # Estimate component breakdown
    base_weight_mem = nf4_mem
    adapter_mem = lora_mem
    training_overhead = mem_peak - mem_after_lora  # optimizer + grad + activ

    components = ["NF4 base\nweights", "LoRA\nadapters",
                  "Optimizer +\ngrad + activ"]
    values = [base_weight_mem, adapter_mem, training_overhead]
    comp_colors = [COLORS["nf4"], COLORS["lora"], COLORS["optim"]]
    comp_hatches = [HATCHES["nf4"], HATCHES["lora"], HATCHES["optim"]]

    bars = ax.bar(components, values, color=comp_colors,
                  edgecolor="black", linewidth=0.5, hatch=comp_hatches)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.1f} GB", ha="center", va="bottom", fontsize=7)

    # GPU memory lines
    for name, vram, ls in [("T4 (16 GB)", 16, "--"),
                           ("RTX 4090 (24 GB)", 24, "-.")]:
        ax.axhline(y=vram, color="gray", linestyle=ls, linewidth=0.5,
                   alpha=0.7)
        ax.text(len(components) - 0.5, vram + 0.2, name,
                fontsize=6, color="gray", ha="right")

    ax.set_ylabel("GPU memory (GB)")
    ax.set_title(f"QLoRA memory breakdown (Llama-2-7B, peak {mem_peak:.1f} GB)")

    # Show stacked total
    ax.axhline(y=mem_peak, color="red", linestyle=":", linewidth=0.8)
    ax.text(0, mem_peak + 0.3, f"Peak: {mem_peak:.1f} GB",
            fontsize=7, color="red")

    fig.tight_layout()
    save_or_show(fig, "CH08_F10_Kalyanarangan", config)

    # Cleanup
    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: memory — Memory breakdown without training
# ═══════════════════════════════════════════════════════════════════════════════

def run_memory(config: Config):
    print("\n" + "=" * 70)
    print("MODE: memory — NF4 vs FP16 memory comparison (Llama-2-7B)")
    print("=" * 70)

    try:
        import bitsandbytes as bnb
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        print(f"\n  ERROR: Missing dependency: {e}")
        print(f"  Memory mode requires: bitsandbytes >= 0.43, peft >= 0.7")
        return

    device = resolve_device(config)
    if device != "cuda":
        print("\n  ERROR: Memory mode requires a CUDA GPU.")
        return

    cap = get_gpu_capability()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ── Load in NF4 (with double quantization) ──
    print(f"\n  Loading {QLORA_MODEL} in NF4 (double quant)...")

    mem_start = gpu_memory_gb()

    bnb_config_nf4_dq = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cap >= (8, 0)
                                else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(QLORA_MODEL)

    model_nf4_dq = AutoModelForCausalLM.from_pretrained(
        QLORA_MODEL,
        quantization_config=bnb_config_nf4_dq,
        device_map="auto",
    )
    mem_nf4_dq = gpu_memory_gb() - mem_start

    # Count parameters from config (not from packed tensors)
    real_counts = compute_real_param_counts(QLORA_MODEL)
    total_params = real_counts["total"]
    linear_params = real_counts["linear"]
    other_params = real_counts["other"]

    print(f"  NF4 (double quant): {mem_nf4_dq:.2f} GB")
    print(f"  Total parameters: {total_params:,} (from model config)")

    del model_nf4_dq
    torch.cuda.empty_cache()

    # ── Load in NF4 (without double quantization) ──                        #O
    print(f"\n  Loading {QLORA_MODEL} in NF4 (no double quant)...")

    mem_start2 = gpu_memory_gb()

    bnb_config_nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cap >= (8, 0)
                                else torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model_nf4 = AutoModelForCausalLM.from_pretrained(
        QLORA_MODEL,
        quantization_config=bnb_config_nf4,
        device_map="auto",
    )
    mem_nf4 = gpu_memory_gb() - mem_start2

    print(f"  NF4 (no double quant): {mem_nf4:.2f} GB")

    del model_nf4
    torch.cuda.empty_cache()

    #O Comparing with and without double quantization isolates the scale
    #  quantization savings. For a 7B model with ~6.7B linear params and
    #  block size 64, the expected savings are approximately:
    #    Without DQ: (6.7e9 / 64) × 4 bytes ≈ 419 MB for scales
    #    With DQ:    (6.7e9 / 64) × 1 byte  ≈ 105 MB for scales
    #  The ~314 MB difference is modest but measurable.

    # ── Theoretical memory calculation ──
    # Compare NF4 with FP16 (the format QLoRA replaces)
    bytes_fp16 = total_params * 2
    bytes_nf4_weights = linear_params * 0.5  # 4-bit = 0.5 bytes/param
    other_params = total_params - linear_params
    bytes_other = other_params * 2  # non-linear params stay in FP16
    bytes_scales_no_dq = (linear_params / 64) * 4  # FP32 scales
    bytes_scales_dq = (linear_params / 64) * 1  # FP8 scales (double quant)
    bytes_scales_dq += (linear_params / 64 / 256) * 4  # second-level FP32

    theoretical_nf4_dq = bytes_nf4_weights + bytes_other + bytes_scales_dq
    theoretical_nf4 = bytes_nf4_weights + bytes_other + bytes_scales_no_dq
    theoretical_fp16 = bytes_fp16

    print(f"\n  ─── Theoretical memory calculation ───")
    print(f"  Linear parameters: {linear_params:,} "
          f"({linear_params/total_params*100:.1f}% of total)")
    print(f"  Non-linear parameters: {other_params:,}")
    print(f"")
    print(f"  {'Config':<25} {'Weights':<12} {'Scales':<12} {'Other':<12} "
          f"{'Total':<12} {'Compression':<12}")
    print(f"  {'─' * 85}")
    print(f"  {'FP16 baseline':<25} "
          f"{bytes_fp16/(1024**3):.2f} GB{'':<4} {'—':<12} {'—':<12} "
          f"{bytes_fp16/(1024**3):.2f} GB{'':<4} {'1.00×'}")
    print(f"  {'NF4 (no double quant)':<25} "
          f"{bytes_nf4_weights/(1024**3):.2f} GB{'':<4} "
          f"{bytes_scales_no_dq/(1024**3):.2f} GB{'':<4} "
          f"{bytes_other/(1024**3):.2f} GB{'':<4} "
          f"{theoretical_nf4/(1024**3):.2f} GB{'':<4} "
          f"{theoretical_fp16/theoretical_nf4:.2f}×")
    print(f"  {'NF4 (double quant)':<25} "
          f"{bytes_nf4_weights/(1024**3):.2f} GB{'':<4} "
          f"{bytes_scales_dq/(1024**3):.2f} GB{'':<4} "
          f"{bytes_other/(1024**3):.2f} GB{'':<4} "
          f"{theoretical_nf4_dq/(1024**3):.2f} GB{'':<4} "
          f"{theoretical_fp16/theoretical_nf4_dq:.2f}×")

    # ── QLoRA training memory projection ──
    print(f"\n  ─── QLoRA training memory projection (Llama-2-7B) ───")

    # LoRA adapter size from architecture dimensions
    h = real_counts["hidden_size"]       # 4096
    i = real_counts["intermediate_size"] # 11008
    n_layers = real_counts["num_layers"] # 32
    r = config.lora_r

    # LoRA adds A (d_in × r) and B (r × d_out) per target per layer:
    #   Attention (q,k,v,o): each (h×h), LoRA = 2×h×r per target, 4 targets
    #   MLP (gate,up,down): gate/up are (h×i), down is (i×h),
    #                       LoRA = (h+i)×r per target, 3 targets
    lora_per_layer = 4 * (2 * h * r) + 3 * ((h + i) * r)
    lora_params = n_layers * lora_per_layer
    lora_bytes = lora_params * 2  # FP32 master weights (BF16 compute via autocast)

    # Optimizer: paged_adamw_8bit uses ~1 byte per optimizer param
    optim_bytes = lora_params * 1  # 8-bit Adam ≈ 1 byte/param

    # Gradients: one gradient per trainable param in BF16
    grad_bytes = lora_params * 2

    # Activation memory depends on batch size and sequence length
    # Rough estimate for batch_size=4, seq_len=512, Llama-2-7B
    activ_bytes_est = 4 * 512 * 4096 * 32 * 2 * 2  # very rough

    print(f"  LoRA adapter parameters: {lora_params:,} "
          f"({lora_bytes/(1024**2):.0f} MB)")
    print(f"  Optimizer states (8-bit Adam): "
          f"{optim_bytes/(1024**2):.0f} MB")
    print(f"  Gradients (BF16): "
          f"{grad_bytes/(1024**2):.0f} MB")

    proj_total_no_activ = (theoretical_nf4_dq + lora_bytes +
                           optim_bytes + grad_bytes)
    print(f"\n  Projected total (excl. activations): "
          f"{proj_total_no_activ/(1024**3):.2f} GB")

    # GPU fit analysis
    print(f"\n  {'GPU':<25} {'VRAM':<10} {'Base model':<12} {'+ LoRA/opt':<12}"
          f" {'Fits (est.)?':<12}")
    print(f"  {'─' * 71}")
    for name, vram in [("T4", 16), ("RTX 3090", 24),
                       ("RTX 4090", 24), ("A100-40", 40),
                       ("A100-80 / H100", 80)]:
        base = theoretical_nf4_dq / (1024**3)
        with_lora = proj_total_no_activ / (1024**3)
        fits = "✓ (tight)" if with_lora < vram * 0.85 else (
            "✓" if with_lora < vram * 0.95 else "✗")
        print(f"  {name:<25} {vram:<10} {base:<12.1f} {with_lora:<12.1f} "
              f"{fits}")

    print(f"\n  Measured values (from actual loading):")
    print(f"    NF4 with double quant: {mem_nf4_dq:.2f} GB")
    print(f"    NF4 without double quant: {mem_nf4:.2f} GB")
    print(f"    Double quant savings: {mem_nf4 - mem_nf4_dq:.2f} GB")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Ch8 §8.4 — QLoRA with NF4"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["codebook", "mse", "qlora", "memory", "all"],
        help="Which experiment to run"
    )
    parser.add_argument(
        "--mse-model", default="opt-6.7b",
        choices=list(MSE_MODEL_MAP.keys()),
        help="Model for MSE experiments (default: opt-6.7b)"
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device (default: auto-detect)"
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save figures to disk (PNG + PDF)"
    )
    parser.add_argument(
        "--num-seqs", type=int, default=64,
        help="Number of evaluation sequences (default: 64)"
    )
    parser.add_argument(
        "--seq-length", type=int, default=512,
        help="Sequence length (default: 512)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for figures"
    )
    parser.add_argument(
        "--qlora-steps", type=int, default=200,
        help="Number of QLoRA training steps (default: 200)"
    )
    parser.add_argument(
        "--qlora-samples", type=int, default=1000,
        help="Number of training samples (default: 1000)"
    )
    parser.add_argument(
        "--lora-r", type=int, default=16,
        help="LoRA rank (default: 16)"
    )
    args = parser.parse_args()

    config = Config(
        mode=args.mode,
        mse_model=args.mse_model,
        device=args.device,
        save_plots=args.save_plots,
        num_calibration_seqs=args.num_seqs,
        seq_length=args.seq_length,
        qlora_max_steps=args.qlora_steps,
        qlora_train_samples=args.qlora_samples,
        lora_r=args.lora_r,
    )
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    return config


def main():
    config = parse_args()

    print("=" * 70)
    print("Chapter 8, Section 8.4 — QLoRA with NF4")
    print("=" * 70)
    print(f"  Mode:      {config.mode}")
    if config.mode in ("mse", "all"):
        print(f"  MSE model: {config.mse_model}")
    print(f"  Device:    {config.device}")

    # Report GPU info
    if torch.cuda.is_available():
        cap = get_gpu_capability()
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU:       {gpu_name} (SM {cap[0]}.{cap[1]}, "
              f"{gpu_mem:.0f} GB)")
    else:
        print(f"  GPU:       None (CPU mode)")

    modes_to_run = (
        ["codebook", "mse", "qlora", "memory"]
        if config.mode == "all" else [config.mode]
    )

    for mode in modes_to_run:
        if mode == "codebook":
            run_codebook(config)
        elif mode == "mse":
            run_mse(config)
        elif mode == "qlora":
            run_qlora(config)
        elif mode == "memory":
            run_memory(config)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()