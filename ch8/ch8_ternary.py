"""
Chapter 8, Section 8.5 — Ternary (1.58-bit) models
Companion script for "Efficient AI in Practice: Quantization and Fast Inference"

Modes:
  --mode numberline    {-1, 0, +1} alongside NF4/INT4/FP4 on a shared number line
  --mode ptq           PTQ catastrophe on OPT-6.7B (naive / TWN / per-row TWN)
  --mode bitlinear     BitLinear forward/backward + toy MLP: QAT vs PTQ
  --mode realmodel     BitNet b1.58 2B4T inference vs OPT-2.7B BF16 baseline
  --mode all           Run all modes

Usage:
  # Format exploration (CPU-safe, no model needed)
  python ch8_ternary.py --mode numberline --save-plots

  # PTQ on OPT-6.7B (needs ~13 GB VRAM for BF16 on H100/A100)
  python ch8_ternary.py --mode ptq --save-plots

  # BitLinear toy MLP (CPU-safe, ~30 s)
  python ch8_ternary.py --mode bitlinear --save-plots

  # Real ternary model inference
  python ch8_ternary.py --mode realmodel --save-plots

  # CPU-only PTQ fallback (illustrative only; uses opt-125m)
  python ch8_ternary.py --mode ptq --ptq-model opt-125m --device cpu --save-plots

Requires:
  All modes:         torch >= 2.4, matplotlib, numpy, scipy
  ptq, realmodel:    transformers >= 4.51 (BitNet support merged 2025-04-28),
                     datasets

Note: microsoft/bitnet-b1.58-2B-4T is supported directly by transformers
      mainline since 2025-04-28. No forks or special installs needed.
"""

import argparse
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Configuration ───────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

PTQ_MODEL_MAP = {
    "opt-6.7b": "facebook/opt-6.7b",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-350m": "facebook/opt-350m",
    "opt-125m": "facebook/opt-125m",
}

# Real ternary model and its BF16 comparison baseline.
# BitNet b1.58 2B4T is 2B params trained from scratch on 4T tokens.
# OPT-2.7B is the closest OPT size (continuity with Ch7 and §8.2-§8.4).
#
# IMPORTANT: microsoft/bitnet-b1.58-2B-4T stores PACKED ternary weights
# (uint8 tensors with multiple trits per byte). The standard transformers
# forward pass cannot use packed weights without torch.compile unpacking.
# Use the -bf16 variant instead — same model, same perplexity, BF16 master
# weights that work with standard HF transformers inference.
TERNARY_MODEL_MAP = {
    "bitnet-b1.58-2B": "microsoft/bitnet-b1.58-2B-4T-bf16",
    "falcon-e-1b":     "tiiuae/Falcon-E-1B-Base",
    "falcon-e-3b":     "tiiuae/Falcon-E-3B-Base",
}

BASELINE_MODEL_MAP = {
    "opt-2.7b":    "facebook/opt-2.7b",
    "opt-1.3b":    "facebook/opt-1.3b",
    "opt-6.7b":    "facebook/opt-6.7b",
}


@dataclass
class Config:
    mode: str = "all"
    ptq_model: str = "opt-6.7b"
    ternary_model: str = "bitnet-b1.58-2B"
    baseline_model: str = "opt-2.7b"
    device: str = "auto"
    save_plots: bool = False
    output_dir: Path = SCRIPT_DIR / "figures"
    num_calibration_seqs: int = 64
    seq_length: int = 512
    seed: int = 42
    # BitLinear toy training
    bitlinear_steps: int = 500
    bitlinear_batch: int = 128
    bitlinear_lr: float = 1e-3


# ─── Manning figure style ────────────────────────────────────────────────────

COLORS = {
    "bf16":        "#7570b3",   # Purple — BF16 baseline
    "int4":        "#1b9e77",   # Teal — INT4
    "fp4":         "#e6ab02",   # Gold — FP4
    "nf4":         "#d95f02",   # Orange — NF4
    "fp8":         "#2166ac",   # Blue — FP8
    "ternary":     "#b2182b",   # Deep red — ternary
    "ternary_twn": "#ef8a62",   # Light red — TWN
    "ternary_row": "#fddbc7",   # Pale red — per-row TWN
    "qat":         "#1a9850",   # Green — QAT success
    "ptq":         "#d7301f",   # Red — PTQ failure
    "fp32":        "#7570b3",   # Purple — FP32 reference
}

HATCHES = {
    "bf16":        "..",
    "int4":        "//",
    "fp4":         "\\\\",
    "nf4":         "",
    "fp8":         "||",
    "ternary":     "xx",
    "ternary_twn": "++",
    "ternary_row": "--",
    "qat":         "",
    "ptq":         "xx",
    "fp32":        "..",
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
        "pdf.fonttype": 42,
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


# ─── Format reference codebooks (for number-line comparison) ─────────────────

def build_nf4_codebook(offset: float = 0.9677083) -> np.ndarray:
    """NF4 codebook from normal distribution quantiles (Dettmers 2023, §8.4).

    Replicated here so the ternary script is self-contained.
    """
    from scipy.stats import norm
    pos_quantiles = np.linspace(0.5, offset, 9)[1:]
    pos_levels = norm.ppf(pos_quantiles)
    neg_quantiles = np.linspace(1 - offset, 0.5, 8)[:-1]
    neg_levels = norm.ppf(neg_quantiles)
    all_levels = np.sort(np.concatenate([neg_levels, [0.0], pos_levels]))
    return all_levels / np.abs(all_levels).max()


def enumerate_fp4_e2m1() -> np.ndarray:
    """Enumerate all 16 FP4 E2M1 representable values (from §8.3)."""
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


# Ternary has three representable values, period.
TERNARY_LEVELS = np.array([-1.0, 0.0, 1.0])


# ─── Ternary quantization methods ────────────────────────────────────────────

def quantize_ternary_naive(w: torch.Tensor) -> torch.Tensor:
    """Naive per-tensor ternary: w_q = sign(w) * mean(|w|).

    The "naive" approach simply takes sign(w) and scales by the mean of |w|.
    Because sign(0) = 0, only exactly-zero inputs map to the zero level.
    In practice real weights are rarely exactly zero, so this collapses to
    a BINARY quantization where every weight becomes ±scale.                  #A
    """
    scale = w.abs().mean()
    if scale.item() == 0.0:
        return torch.zeros_like(w)
    return torch.sign(w) * scale

#A This is deliberately a weak baseline. A quantization scheme that cannot
#  produce the zero value is missing the key expressive advantage of ternary
#  over binary — the ability to sparsify unimportant weights. The TWN
#  variants below introduce a threshold that lets small-magnitude weights
#  round down to zero.


def quantize_ternary_twn_per_tensor(
    w: torch.Tensor,
    threshold_factor: float = 0.7,
) -> torch.Tensor:
    """Ternary Weight Networks (Li et al. 2016): per-tensor threshold ternary.

    Steps:
      1. τ = threshold_factor * mean(|W|)     (threshold_factor=0.7 per paper)
      2. Mask: |w| > τ → active, else → zero
      3. Scale α = mean(|w|) over active weights
      4. w_q = sign(w) * α * (|w| > τ)                                         #B
    """
    abs_w = w.abs()
    threshold = threshold_factor * abs_w.mean()
    mask = abs_w > threshold
    if mask.sum().item() == 0:
        return torch.zeros_like(w)
    alpha = abs_w[mask].mean()
    return torch.sign(w) * alpha * mask.to(w.dtype)

#B Li et al. (2016) show that τ = 0.7 * E[|w|] is optimal for Gaussian
#  weights under an MSE objective. The scale α is a single scalar for the
#  entire weight tensor — the same "per-tensor scaling" pathology that §8.3
#  demonstrated is catastrophic for FP4. Expect it to be worse for ternary.


def quantize_ternary_twn_per_row(
    w: torch.Tensor,
    threshold_factor: float = 0.7,
) -> torch.Tensor:
    """Per-output-row TWN: each row (output channel) gets its own τ and α.

    This is the finest granularity that keeps the storage overhead bounded
    at one scale factor per row (rather than per block). It mirrors INT4
    per-channel scaling from Chapter 7.                                       #C
    """
    if w.dim() != 2:
        return quantize_ternary_twn_per_tensor(w, threshold_factor)
    abs_w = w.abs()
    # Per-row threshold: τ_i = factor × mean(|w_i|)
    row_abs_mean = abs_w.mean(dim=1, keepdim=True)
    threshold = threshold_factor * row_abs_mean
    mask = abs_w > threshold
    # Per-row scale: α_i = mean of active weights in row i
    active_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
    alpha = (abs_w * mask.to(w.dtype)).sum(dim=1, keepdim=True) / active_count
    return torch.sign(w) * alpha * mask.to(w.dtype)

#C Per-row scaling is what INT8 and INT4 use in Chapter 7 (GPTQ, AWQ) and
#  what FP4 block-16 effectively approximates (§8.3). The open question
#  here is whether the same scaling trick that rescues INT4 and FP4 is
#  enough to rescue ternary. The mode results will answer it.


# ─── BitLinear module (BitNet b1.58) ─────────────────────────────────────────

def weight_quant_bitlinear(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """BitNet b1.58 weight quantization: absmean → round → clamp(-1, 1).

    Matches the forward-pass quantization used by BitNet b1.58 2B4T
    (microsoft/bitnet-b1.58-2B-4T) and Falcon-E (tiiuae/Falcon-E-*-Base).
    """
    scale = 1.0 / w.abs().mean().clamp(min=eps)
    return (w * scale).round().clamp(-1, 1) / scale


def activation_quant_bitlinear(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """BitNet b1.58 activation quantization: per-token absmax INT8."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    return (x * scale).round().clamp(-128, 127) / scale


class BitLinear(nn.Linear):
    """Quantization-aware Linear layer — weights ternary, activations INT8.

    The straight-through estimator (STE) is implemented with the standard
    subtract-detach-add trick. Forward pass uses the quantized values;
    backward pass treats the quantization as identity so gradients flow
    to the underlying full-precision master weights.                          #D
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Activation quantization with STE
        x_q = x + (activation_quant_bitlinear(x) - x).detach()                #E
        # Weight quantization with STE
        w = self.weight
        w_q = w + (weight_quant_bitlinear(w) - w).detach()
        return F.linear(x_q, w_q, self.bias)

#D Concretely, `x + (x_q - x).detach()` evaluates to `x_q` in the forward
#  pass (because the detached correction pulls x down to x_q), but the
#  autograd graph sees only `x` on the non-detached path — so the gradient
#  is d(x)/dx = 1, not d(x_q)/dx = 0. This is the straight-through estimator.
#E Activation quantization happens FIRST, on the unquantized input. The
#  BitNet forward order matters: if you quantized weights first and then
#  computed x_q from the quantized intermediate, the training dynamics
#  would be different.


# ─── WikiText-2 perplexity (same protocol as §8.2-§8.4) ──────────────────────

def load_wikitext2_test_ids(tokenizer, config: Config) -> torch.Tensor:
    """Load WikiText-2 test set and tokenize to a flat ID tensor."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    return encodings.input_ids[0]                                             #F

#F We concatenate the entire test split into a single stream before tokenizing.
#  This matches the standard WikiText-2 perplexity protocol used by GPTQ,
#  AWQ, and the rest of this book — not per-document perplexity.


@torch.no_grad()
def compute_perplexity(
    model,
    input_ids: torch.Tensor,
    num_seqs: int,
    seq_length: int,
    device: str,
) -> float:
    """Compute perplexity on num_seqs non-overlapping windows of seq_length.

    Matches the 64 × 512-token evaluation window from §8.2-§8.4.
    """
    model.eval()
    total_len = input_ids.size(0)
    max_possible = total_len // seq_length
    n = min(num_seqs, max_possible)

    nlls = []
    for i in range(n):
        start = i * seq_length
        end = start + seq_length
        ids = input_ids[start:end].unsqueeze(0).to(device)
        out = model(ids, labels=ids)
        nlls.append(out.loss.detach().float().cpu().item())                   #G
    return float(np.exp(np.mean(nlls)))

#G outputs.loss is already the average NLL per token for this batch. Taking
#  the mean across batches and then exp() gives the geometric-mean perplexity
#  per token — the standard reporting metric.


# ─── Model / device helpers ──────────────────────────────────────────────────

def resolve_device(config: Config) -> str:
    if config.device != "auto":
        return config.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_gpu_capability() -> tuple:
    if torch.cuda.is_available():
        return torch.cuda.get_device_capability()
    return (0, 0)


def gpu_memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return 0.0


def select_dtype(device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    cap = get_gpu_capability()
    if cap >= (8, 0):
        return torch.bfloat16
    elif cap >= (7, 0):
        return torch.float16
    return torch.float32


def load_causal_lm(
    model_id: str,
    device: str,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
):
    """Load a causal LM + tokenizer in the appropriate dtype for the device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    if dtype is None:
        dtype = select_dtype(device)

    # Handle torch_dtype → dtype rename across transformers versions.
    # Newer transformers (>= 4.51) deprecates torch_dtype in favor of dtype.
    import transformers
    _hf_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    _dtype_kwarg = "dtype" if _hf_version >= (4, 51) else "torch_dtype"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **{_dtype_kwarg: dtype},
        device_map=device if device != "cpu" else None,
        trust_remote_code=trust_remote_code,
    )
    if device == "cpu":
        model = model.float()
    model.eval()
    dtype_name = {
        torch.bfloat16: "BF16", torch.float16: "FP16", torch.float32: "FP32"
    }.get(dtype, str(dtype))
    print(f"  Model dtype: {dtype_name}")
    print(f"  Model device: {next(model.parameters()).device}")
    return model, tokenizer, dtype


def model_linear_param_count(model) -> int:
    """Count parameters in all nn.Linear modules (ignores embeddings / LN)."""
    return sum(
        m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
        for m in model.modules() if isinstance(m, nn.Linear)
    )


def model_total_param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def model_storage_bytes(model) -> int:
    """On-GPU storage in bytes (sum of each tensor's numel × element_size)."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


# ─── State management: temporarily quantize Linear weights ──────────────────

@contextmanager
def linear_weights_quantized(
    model,
    quantize_fn: Callable[[torch.Tensor], torch.Tensor],
    skip_substrings=("lm_head",),
):
    """Apply quantize_fn to every nn.Linear weight in-place; restore on exit.

    Originals are parked on CPU to keep GPU memory flat during the swap.     #H
    """
    saved: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in skip_substrings):
            continue
        saved[name] = module.weight.data.detach().cpu().clone()
        q = quantize_fn(module.weight.data)
        with torch.no_grad():
            module.weight.data.copy_(q)
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if name in saved:
                with torch.no_grad():
                    module.weight.data.copy_(
                        saved[name].to(module.weight.device,
                                       dtype=module.weight.dtype))
        saved.clear()

#H Saving to CPU means we don't double GPU memory for the swap. On
#  OPT-6.7B (13 GB BF16) the saved copy on CPU is ~13 GB of host RAM,
#  which is fine. If host RAM is constrained, you can stream per-layer
#  instead of caching all at once.


def measure_linear_reconstruction_mse(
    model,
    quantize_fn: Callable[[torch.Tensor], torch.Tensor],
    skip_substrings=("lm_head",),
) -> float:
    """Mean-squared reconstruction error over all Linear weights."""
    total_sq = 0.0
    total_n = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in skip_substrings):
            continue
        w = module.weight.data.detach().float()
        q = quantize_fn(module.weight.data).float()
        total_sq += ((w - q) ** 2).sum().item()
        total_n += w.numel()
    return total_sq / max(total_n, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: numberline — {-1, 0, +1} alongside NF4/INT4/FP4 on a shared axis
# ═══════════════════════════════════════════════════════════════════════════════

def run_numberline(config: Config):
    print("\n" + "=" * 70)
    print("MODE: numberline — Ternary {-1, 0, +1} vs 4-bit codebooks")
    print("=" * 70)

    # Build reference codebooks in [-1, 1]
    nf4 = build_nf4_codebook()
    int4 = np.linspace(-1, 1, 16)                                            #I
    fp4 = enumerate_fp4_e2m1()
    fp4_norm = fp4 / np.abs(fp4).max()
    ternary = TERNARY_LEVELS.copy()

    #I INT4 symmetric has 16 levels (values -8..+7 mapped via scale, or
    #  equivalently 16 evenly spaced points in [-1, 1]). This matches the
    #  convention in §8.4's codebook comparison (np.linspace(-1, 1, 16)).

    # ── Density summary ──
    print(f"\n  Representable values by format (normalized to [-1, 1]):")
    print(f"  {'Format':<20} {'Total':<8} {'Non-neg':<10} {'In [-0.3,0.3]':<15}")
    print(f"  {'─' * 53}")
    for name, vals in [
        ("Ternary {-1,0,+1}", ternary),
        ("INT4 (uniform)",    int4),
        ("FP4 E2M1",          fp4_norm),
        ("NF4 (quantile)",    nf4),
    ]:
        total = len(vals)
        nonneg = int(np.sum(vals >= 0))
        nearzero = int(np.sum((vals >= -0.3) & (vals <= 0.3)))
        print(f"  {name:<20} {total:<8} {nonneg:<10} {nearzero:<15}")

    # ── Storage cost (theoretical bits per weight) ──
    print(f"\n  Storage cost (theoretical bits per weight):")
    print(f"  {'Format':<20} {'log2(levels)':<16} {'Typical pack':<18}")
    print(f"  {'─' * 54}")
    print(f"  {'Ternary':<20} {f'{math.log2(3):.3f}':<16} "
          f"{'1.60 (5 trits/byte)':<18}")                                     #J
    print(f"  {'INT4':<20} {f'{math.log2(16):.3f}':<16} "
          f"{'4.00 (2/byte)':<18}")
    print(f"  {'FP4 E2M1':<20} {f'{math.log2(16):.3f}':<16} "
          f"{'4.00 (2/byte)':<18}")
    print(f"  {'NF4':<20} {f'{math.log2(16):.3f}':<16} "
          f"{'4.00 (2/byte)':<18}")

    #J log2(3) ≈ 1.585. Practical ternary storage uses base-3 packing:
    #  3^5 = 243 < 256, so 5 ternary digits fit in one byte (1.60 bits each).
    #  bitnet.cpp's TL1 and TL2 kernels use this pack; the i2_s format uses
    #  a simpler 4 trits per byte (2.00 bits each) for lookup efficiency.

    # ── Figure: full range [-1, 1] ──
    apply_manning_style()

    fig, axes = plt.subplots(4, 1, figsize=(5.6, 3.2), sharex=True)
    panels = [
        (axes[0], ternary,   f"Ternary ({len(ternary)} levels)",
         COLORS["ternary"]),
        (axes[1], nf4,       f"NF4 ({len(nf4)} levels)",
         COLORS["nf4"]),
        (axes[2], int4,      f"INT4 ({len(int4)} levels)",
         COLORS["int4"]),
        (axes[3], fp4_norm,  f"FP4 E2M1 ({len(fp4_norm)} levels)",
         COLORS["fp4"]),
    ]
    for ax, vals, name, color in panels:
        ax.eventplot([vals], lineoffsets=0, linelengths=0.6,
                     colors=[color], linewidths=1.0)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center",
                      labelpad=5)
        ax.set_xlim(-1.15, 1.15)
        ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel("Normalized value")
    fig.suptitle(
        "Ternary vs 4-bit quantization levels in [-1, 1]",
        fontsize=9, y=0.99)
    fig.tight_layout(rect=[0.25, 0.0, 1.0, 0.95])
    save_or_show(fig, "CH08_F17_Kalyanarangan_ternary_numberline", config)

    # ── Figure: near-zero density ──
    fig2, axes2 = plt.subplots(4, 1, figsize=(5.6, 3.2), sharex=True)
    panels2 = [
        (axes2[0], ternary,   "Ternary",         COLORS["ternary"]),
        (axes2[1], nf4,       "NF4",             COLORS["nf4"]),
        (axes2[2], int4,      "INT4 uniform",    COLORS["int4"]),
        (axes2[3], fp4_norm,  "FP4 E2M1",        COLORS["fp4"]),
    ]
    for ax, vals, name, color in panels2:
        mask = (vals >= -0.3) & (vals <= 0.3)
        v = vals[mask]
        ax.eventplot([v], lineoffsets=0, linelengths=0.6,
                     colors=[color], linewidths=1.0)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center",
                      labelpad=5)
        ax.set_xlim(-0.35, 0.35)
        ax.spines["left"].set_visible(False)
        ax.text(0.98, 0.85, f"{len(v)} levels",
                transform=ax.transAxes, fontsize=6, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))
    axes2[-1].set_xlabel("Normalized value")
    fig2.suptitle("Near-zero density: levels in [-0.3, 0.3]",
                  fontsize=9, y=0.99)
    fig2.tight_layout(rect=[0.15, 0.0, 1.0, 0.95])
    save_or_show(fig2, "CH08_F17b_Kalyanarangan_ternary_nearzero", config)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: ptq — Ternary PTQ catastrophe on OPT-6.7B
# ═══════════════════════════════════════════════════════════════════════════════

def run_ptq(config: Config):
    print("\n" + "=" * 70)
    print(f"MODE: ptq — Ternary PTQ catastrophe ({config.ptq_model})")
    print("=" * 70)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset
    except ImportError as e:
        print(f"\n  ERROR: Missing dependency: {e}")
        print(f"  PTQ mode requires: transformers, datasets")
        print(f"  Install: pip install transformers datasets")
        return

    device = resolve_device(config)
    model_id = PTQ_MODEL_MAP.get(config.ptq_model, config.ptq_model)
    model, tokenizer, dtype = load_causal_lm(model_id, device)

    # ── Tokenize WikiText-2 test set ──
    print(f"\n  Loading WikiText-2 test set...")
    input_ids = load_wikitext2_test_ids(tokenizer, config)
    print(f"  Total tokens: {input_ids.size(0):,}")
    print(f"  Evaluation window: {config.num_calibration_seqs} × "
          f"{config.seq_length} = "
          f"{config.num_calibration_seqs * config.seq_length:,} tokens")

    # ── Baseline (unquantized) perplexity ──
    print(f"\n  Measuring baseline perplexity...")
    ppl_baseline = compute_perplexity(
        model, input_ids, config.num_calibration_seqs,
        config.seq_length, device)
    print(f"  {dtype} perplexity (WikiText-2): {ppl_baseline:.3f}")

    # ── Reconstruction MSE for each ternary method ──                        #K
    methods = [
        ("naive",       quantize_ternary_naive,
         "Naive per-tensor (sign×absmean, no zeros)"),
        ("twn_tensor",  quantize_ternary_twn_per_tensor,
         "TWN per-tensor (τ = 0.7·mean(|W|))"),
        ("twn_row",     quantize_ternary_twn_per_row,
         "TWN per-row (one τ and α per output channel)"),
    ]

    print(f"\n  Measuring reconstruction MSE across Linear layers...")
    mse_results = {}
    for name, qfn, descr in methods:
        mse = measure_linear_reconstruction_mse(model, qfn)
        mse_results[name] = mse
        print(f"    {descr}")
        print(f"      weight MSE: {mse:.6e}")

    #K MSE is measured against the original BF16/FP16 weights, layer-wise.
    #  This mirrors the Figures 8.5, 8.9, 8.10, 8.14 protocol from §8.2-§8.4.
    #  The key expectation: even per-row TWN should have MSE 10-100× worse
    #  than INT4 group-128 — at 3 levels there is simply not enough
    #  expressive capacity to approximate a continuous distribution well.

    # ── Perplexity for each ternary PTQ method ──
    print(f"\n  Measuring perplexity for each PTQ method...")
    ppl_results = {}
    for name, qfn, descr in methods:
        print(f"\n    Applying: {descr}")
        with linear_weights_quantized(model, qfn):
            ppl = compute_perplexity(
                model, input_ids, config.num_calibration_seqs,
                config.seq_length, device)
        ppl_results[name] = ppl
        print(f"      perplexity: {ppl:.3f}  (Δ vs baseline: "
              f"{ppl - ppl_baseline:+.1f})")

    # ── Summary ──
    print(f"\n  ─── PTQ catastrophe summary ({config.ptq_model}) ───")
    print(f"  {'Method':<40} {'MSE':<14} {'Perplexity':<12} {'Δ':<10}")
    print(f"  {'─' * 76}")
    print(f"  {'BF16 baseline (no quantization)':<40} "
          f"{'—':<14} {ppl_baseline:<12.3f} {'—':<10}")
    for name, _, descr in methods:
        delta = ppl_results[name] - ppl_baseline
        print(f"  {descr:<40} {mse_results[name]:<14.6e} "
              f"{ppl_results[name]:<12.3f} {delta:+.1f}")

    # ── Figure 8.18: perplexity bar chart ──
    apply_manning_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))

    bars_data = [
        ("BF16\nbaseline",   ppl_baseline,          COLORS["bf16"],
         HATCHES["bf16"]),
        ("Ternary\nnaive",   ppl_results["naive"],  COLORS["ternary"],
         HATCHES["ternary"]),
        ("TWN\nper-tensor",  ppl_results["twn_tensor"],
         COLORS["ternary_twn"], HATCHES["ternary_twn"]),
        ("TWN\nper-row",     ppl_results["twn_row"],
         COLORS["ternary_row"], HATCHES["ternary_row"]),
    ]

    x = np.arange(len(bars_data))
    for i, (label, val, color, hatch) in enumerate(bars_data):
        ax.bar(i, val, color=color, hatch=hatch,
               edgecolor="black", linewidth=0.5, width=0.7)
        # Annotate value on top of bar
        ax.text(i, val * 1.05, f"{val:.1f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in bars_data])
    ax.set_ylabel("WikiText-2 perplexity (log scale)")
    ax.set_title(
        f"Ternary PTQ perplexity on {config.ptq_model.upper()}  "
        f"(BF16 = {ppl_baseline:.2f})")

    # Reference line at baseline
    ax.axhline(y=ppl_baseline, color="gray", linestyle=":",
               linewidth=0.8, alpha=0.7)

    fig.tight_layout()
    save_or_show(fig, "CH08_F18_Kalyanarangan_ternary_ptq_ppl", config)

    # Free up GPU memory before later modes run in --mode all
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: bitlinear — BitLinear forward/backward + toy MLP: QAT vs PTQ
# ═══════════════════════════════════════════════════════════════════════════════

def run_bitlinear(config: Config):
    print("\n" + "=" * 70)
    print("MODE: bitlinear — BitLinear forward pass + QAT vs PTQ on toy MLP")
    print("=" * 70)

    device = resolve_device(config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # ── Part 1: Verify BitLinear forward pass numerics ──
    print(f"\n  Part 1: BitLinear forward pass verification")
    print(f"  ─────────────────────────────────────────")

    # Create a BitLinear layer with known weights
    layer = BitLinear(8, 4, bias=False).to(device)
    x = torch.randn(3, 8, device=device, requires_grad=True)

    # Run forward pass
    y = layer(x)
    print(f"  Input shape:   {tuple(x.shape)}")
    print(f"  Output shape:  {tuple(y.shape)}")

    # Inspect the quantized weights
    with torch.no_grad():
        w_raw = layer.weight
        w_q = weight_quant_bitlinear(w_raw)
        unique_levels = torch.unique(
            (w_q * (1.0 / w_raw.abs().mean().clamp(min=1e-5))).round()
        ).tolist()
    print(f"  Raw weight sample:      {w_raw.flatten()[:5].cpu().tolist()}")
    print(f"  Quantized weight/scale: {unique_levels}  "
          f"(should be subset of {{-1, 0, +1}})")

    # Verify gradient flow through STE                                        #L
    loss = y.sum()
    loss.backward()
    print(f"  Gradient w.r.t. input  (sample): "
          f"{x.grad.flatten()[:5].cpu().tolist()}")
    print(f"  Gradient w.r.t. weight (sample): "
          f"{layer.weight.grad.flatten()[:5].cpu().tolist()}")
    w_grad_nonzero = (layer.weight.grad != 0).sum().item()
    w_grad_total = layer.weight.grad.numel()
    print(f"  Weight gradients nonzero: {w_grad_nonzero}/{w_grad_total}  "
          f"(STE routes gradients to master weights despite quant)")

    #L If the STE weren't in place, `.round()` would zero out all gradients
    #  and the weight gradient would be identically zero. The subtract-
    #  detach-add trick ensures the backward pass treats quantization as
    #  identity, preserving gradient magnitude. Some input gradients may
    #  be zero if the corresponding weight was quantized to zero — that
    #  is correct behavior, not an STE failure.

    # ── Part 2: Toy MLP — QAT vs PTQ vs FP32 ──                               #M
    print(f"\n  Part 2: Toy regression — QAT (BitLinear) vs PTQ vs FP32")
    print(f"  ─────────────────────────────────────────")

    input_dim, hidden_dim, output_dim = 64, 256, 32
    n_train, n_test = 10_000, 2_000

    # Teacher network: a random FP32 MLP that we want the student to imitate
    teacher = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    ).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    x_train = torch.randn(n_train, input_dim, device=device)
    x_test = torch.randn(n_test, input_dim, device=device)
    with torch.no_grad():
        y_train = teacher(x_train)
        y_test = teacher(x_test)

    def make_mlp(linear_cls: type) -> nn.Module:
        return nn.Sequential(
            linear_cls(input_dim, hidden_dim),
            nn.GELU(),
            linear_cls(hidden_dim, output_dim),
        ).to(device)

    # Student A: BitLinear (QAT from scratch)
    torch.manual_seed(config.seed)
    student_qat = make_mlp(BitLinear)

    # Student B: FP32 baseline (trained in full precision, never quantized)
    torch.manual_seed(config.seed)
    student_fp32 = make_mlp(nn.Linear)

    def train_student(student, label: str) -> List[float]:
        opt = torch.optim.Adam(student.parameters(), lr=config.bitlinear_lr)
        losses = []
        student.train()
        for step in range(config.bitlinear_steps):
            idx = torch.randint(0, n_train, (config.bitlinear_batch,))
            xb, yb = x_train[idx], y_train[idx]
            pred = student(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if step % 100 == 0:
                print(f"    [{label}] step {step:4d}  loss {loss.item():.6f}")
        return losses

    print(f"\n  Training FP32 student...")
    losses_fp32 = train_student(student_fp32, "FP32")

    print(f"\n  Training BitLinear (QAT) student...")
    losses_qat = train_student(student_qat, "QAT ")

    # Student C: PTQ — take trained FP32 student, ternarize weights post-hoc
    print(f"\n  Applying ternary PTQ to trained FP32 student...")
    student_ptq = make_mlp(nn.Linear)
    student_ptq.load_state_dict(student_fp32.state_dict())
    with torch.no_grad():
        for mod in student_ptq.modules():
            if isinstance(mod, nn.Linear):
                mod.weight.data.copy_(weight_quant_bitlinear(mod.weight.data))

    # Evaluate on held-out test set
    def eval_loss(student) -> float:
        student.eval()
        with torch.no_grad():
            return F.mse_loss(student(x_test), y_test).item()

    final_fp32 = eval_loss(student_fp32)
    final_qat = eval_loss(student_qat)
    final_ptq = eval_loss(student_ptq)

    print(f"\n  ─── Final test MSE ───")
    print(f"  {'Model':<30} {'Test MSE':<14} {'vs FP32':<10}")
    print(f"  {'─' * 54}")
    print(f"  {'FP32 (no quantization)':<30} {final_fp32:<14.6f} {'1.00×':<10}")
    print(f"  {'FP32 → ternary PTQ':<30} {final_ptq:<14.6f} "
          f"{final_ptq/final_fp32:.1f}×")
    print(f"  {'BitLinear (QAT)':<30} {final_qat:<14.6f} "
          f"{final_qat/final_fp32:.1f}×")

    #M The contrast between PTQ and QAT is the key empirical argument for
    #  §8.5.3. The FP32 model learns a task; we then either (a) post-hoc
    #  ternarize its weights (PTQ) or (b) train a fresh model whose weights
    #  are ternarized on every forward pass (QAT). The two approaches face
    #  the same weight constraint but produce very different end quality.

    # ── Figure 8.19: loss curves ──
    apply_manning_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.2))

    steps = np.arange(len(losses_fp32))
    # Running average for readability
    window = 10
    def smooth(vals):
        vals = np.array(vals)
        kernel = np.ones(window) / window
        return np.convolve(vals, kernel, mode="valid")

    s_fp32 = smooth(losses_fp32)
    s_qat = smooth(losses_qat)
    x_smooth = steps[window - 1:]

    ax.plot(x_smooth, s_fp32, color=COLORS["fp32"], linewidth=1.3,
            label=f"FP32 (final test MSE {final_fp32:.4f})")
    ax.plot(x_smooth, s_qat, color=COLORS["qat"], linewidth=1.3,
            label=f"BitLinear QAT (final {final_qat:.4f})")
    # PTQ: dashed horizontal line showing post-hoc quantization disaster
    ax.axhline(y=final_ptq, color=COLORS["ptq"], linestyle="--",
               linewidth=1.3,
               label=f"FP32 → ternary PTQ (test MSE {final_ptq:.4f})")

    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE loss (log scale)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("QAT vs PTQ on toy MLP  "
                 f"({input_dim}→{hidden_dim}→{output_dim}, "
                 f"{config.bitlinear_steps} steps)")
    fig.tight_layout()
    save_or_show(fig, "CH08_F19_Kalyanarangan_bitlinear_qat_vs_ptq", config)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: realmodel — BitNet b1.58 2B4T inference vs OPT-2.7B baseline
# ═══════════════════════════════════════════════════════════════════════════════

def run_realmodel(config: Config):
    print("\n" + "=" * 70)
    print(f"MODE: realmodel — {config.ternary_model} vs {config.baseline_model}")
    print("=" * 70)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset
    except ImportError as e:
        print(f"\n  ERROR: Missing dependency: {e}")
        print(f"  realmodel mode requires: transformers >= 4.51, datasets")
        return

    device = resolve_device(config)

    ternary_id = TERNARY_MODEL_MAP.get(
        config.ternary_model, config.ternary_model)
    baseline_id = BASELINE_MODEL_MAP.get(
        config.baseline_model, config.baseline_model)

    # ── Load ternary model ──
    print(f"\n  Loading ternary model: {ternary_id}")
    print(f"  (Supported in HF transformers mainline since 2025-04-28.)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # BitNet b1.58 is natively supported in transformers >= 4.51
    # (merged 2025-04-28). Do NOT use trust_remote_code=True — it
    # causes transformers to look for a configuration_bitnet.py file
    # in the HF repo, which doesn't exist because support is built in.
    # Falcon-E models also use the native BitNet architecture class.
    try:
        tern_model, tern_tok, tern_dtype = load_causal_lm(
            ternary_id, device, trust_remote_code=False)
    except Exception as e:
        print(f"\n  ERROR loading ternary model: {e}")
        print(f"  Your transformers version may be too old. Try:")
        print(f"    pip install --upgrade 'transformers>=4.51'")
        return

    tern_params = model_total_param_count(tern_model)
    tern_linear_params = model_linear_param_count(tern_model)
    tern_bytes = model_storage_bytes(tern_model)
    tern_gpu_mem = gpu_memory_gb()

    print(f"\n  Ternary model loaded (BF16 master weights):")
    print(f"    NOTE: This is the -bf16 variant with unpacked master weights.")
    print(f"    The model was TRAINED with ternary quantization (BitLinear),")
    print(f"    but we load the BF16 master weights for correct HF inference.")
    print(f"    Total params (incl. embeddings): {tern_params:,}")
    print(f"    Linear params only:              {tern_linear_params:,}")
    print(f"    BF16 tensor storage (as loaded):  "
          f"{tern_bytes / (1024**3):.3f} GB")
    print(f"    GPU memory allocated:            {tern_gpu_mem:.3f} GB")

    # Theoretical ternary-packed size (what bitnet.cpp actually uses)
    # 1.58 bits/weight for linear params + BF16 for embeddings/LN
    other_params = tern_params - tern_linear_params
    theoretical_bytes = (
        tern_linear_params * (1.6 / 8) +  # 1.6 bits / 8 = 0.2 bytes per weight
        other_params * 2                  # BF16 everything else
    )
    print(f"    Theoretical packed size (1.6 bits/linear param + BF16 other):")
    print(f"      {theoretical_bytes / (1024**3):.3f} GB")

    # ── Perplexity on WikiText-2 ──
    print(f"\n  Measuring WikiText-2 perplexity (ternary model)...")
    tern_input_ids = load_wikitext2_test_ids(tern_tok, config)
    ppl_ternary = compute_perplexity(
        tern_model, tern_input_ids, config.num_calibration_seqs,
        config.seq_length, device)
    print(f"  Perplexity (ternary): {ppl_ternary:.3f}")

    # Free ternary before loading baseline
    del tern_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Load BF16 baseline ──
    print(f"\n  Loading baseline model: {baseline_id}")
    base_model, base_tok, base_dtype = load_causal_lm(baseline_id, device)

    base_params = model_total_param_count(base_model)
    base_linear_params = model_linear_param_count(base_model)
    base_bytes = model_storage_bytes(base_model)
    base_gpu_mem = gpu_memory_gb()

    print(f"\n  Baseline model loaded:")
    print(f"    Total params:          {base_params:,}")
    print(f"    Linear params:         {base_linear_params:,}")
    print(f"    Tensor storage:        {base_bytes / (1024**3):.3f} GB")
    print(f"    GPU memory allocated:  {base_gpu_mem:.3f} GB")

    print(f"\n  Measuring WikiText-2 perplexity (baseline)...")
    base_input_ids = load_wikitext2_test_ids(base_tok, config)
    ppl_baseline = compute_perplexity(
        base_model, base_input_ids, config.num_calibration_seqs,
        config.seq_length, device)
    print(f"  Perplexity (baseline): {ppl_baseline:.3f}")

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Summary table (Table 8.6 source) ──                                   #N
    print(f"\n  ─── Table 8.6 data (measured) ───")
    header = f"  {'Model':<35} {'Params':<12} {'Storage':<14} {'WikiText-2 PPL':<15}"
    print(header)
    print(f"  {'─' * len(header)}")
    print(f"  {ternary_id:<35} {tern_params/1e9:<12.2f}B "
          f"{tern_bytes/(1024**3):<14.2f} {ppl_ternary:<15.3f}")
    print(f"  {'  (theoretical packed)':<35} {'':<12} "
          f"{theoretical_bytes/(1024**3):<14.2f} {'':<15}")
    print(f"  {baseline_id:<35} {base_params/1e9:<12.2f}B "
          f"{base_bytes/(1024**3):<14.2f} {ppl_baseline:<15.3f}")

    #N Two caveats for reader:
    #  (1) Cross-tokenizer perplexities are not directly comparable at the
    #      decimal level — OPT's GPT-2 BPE and BitNet's Llama tokenizer
    #      produce different token distributions for the same text. They
    #      are comparable at the order-of-magnitude level, which is what
    #      matters for "does it produce coherent text?"
    #  (2) We load the BF16 master weights (-bf16 variant) because the
    #      packed variant requires torch.compile for weight unpacking.
    #      The perplexity is identical — same model, different storage.
    #      The "theoretical packed" row shows what bitnet.cpp would store.

    # ── Figure 8.20: storage comparison ──
    apply_manning_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 3.0))

    # Left panel: storage (GB)
    storage_data = [
        (f"{config.baseline_model}\n(BF16)",
         base_bytes / (1024**3), COLORS["bf16"], HATCHES["bf16"]),
        (f"{config.ternary_model}\n(BF16 master wts)",
         tern_bytes / (1024**3), COLORS["ternary_twn"], HATCHES["ternary"]),
        (f"{config.ternary_model}\n(packed, bitnet.cpp)",
         theoretical_bytes / (1024**3), COLORS["ternary"], ""),
    ]
    x = np.arange(len(storage_data))
    for i, (label, val, color, hatch) in enumerate(storage_data):
        ax1.bar(i, val, color=color, hatch=hatch,
                edgecolor="black", linewidth=0.5, width=0.7)
        ax1.text(i, val + 0.05, f"{val:.2f}",
                 ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s[0] for s in storage_data], fontsize=6)
    ax1.set_ylabel("Storage (GB)")
    ax1.set_title("Model storage")

    # Right panel: perplexity
    ppl_data = [
        (f"{config.baseline_model}\n(BF16)",   ppl_baseline,
         COLORS["bf16"], HATCHES["bf16"]),
        (f"{config.ternary_model}",            ppl_ternary,
         COLORS["ternary_twn"], HATCHES["ternary"]),
    ]
    x2 = np.arange(len(ppl_data))
    for i, (label, val, color, hatch) in enumerate(ppl_data):
        ax2.bar(i, val, color=color, hatch=hatch,
                edgecolor="black", linewidth=0.5, width=0.55)
        ax2.text(i, val + max(p[1] for p in ppl_data) * 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([p[0] for p in ppl_data], fontsize=6)
    ax2.set_ylabel("WikiText-2 perplexity")
    ax2.set_title("Perplexity (different tokenizers;\n"
                  "compare at order-of-magnitude)")

    fig.tight_layout()
    save_or_show(
        fig, "CH08_F20_Kalyanarangan_ternary_realmodel", config)


# ═══════════════════════════════════════════════════════════════════════════════
# Argument parsing and main entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Ch8 §8.5 — Ternary (1.58-bit) models"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["numberline", "ptq", "bitlinear", "realmodel", "all"],
        help="Which experiment to run",
    )
    parser.add_argument(
        "--ptq-model", default="opt-6.7b",
        choices=list(PTQ_MODEL_MAP.keys()),
        help="Model for PTQ mode (default: opt-6.7b, same as §8.2-§8.4)",
    )
    parser.add_argument(
        "--ternary-model", default="bitnet-b1.58-2B",
        choices=list(TERNARY_MODEL_MAP.keys()),
        help="Ternary model for realmodel mode",
    )
    parser.add_argument(
        "--baseline-model", default="opt-2.7b",
        choices=list(BASELINE_MODEL_MAP.keys()),
        help="BF16 baseline for realmodel mode",
    )
    parser.add_argument(
        "--device", default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device (default: auto-detect)",
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save figures to disk (PNG + PDF)",
    )
    parser.add_argument(
        "--num-seqs", type=int, default=64,
        help="Number of evaluation sequences (default: 64, as in §8.2-§8.4)",
    )
    parser.add_argument(
        "--seq-length", type=int, default=512,
        help="Sequence length (default: 512)",
    )
    parser.add_argument(
        "--bitlinear-steps", type=int, default=500,
        help="Training steps for BitLinear toy MLP (default: 500)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for figures",
    )
    args = parser.parse_args()

    config = Config(
        mode=args.mode,
        ptq_model=args.ptq_model,
        ternary_model=args.ternary_model,
        baseline_model=args.baseline_model,
        device=args.device,
        save_plots=args.save_plots,
        num_calibration_seqs=args.num_seqs,
        seq_length=args.seq_length,
        bitlinear_steps=args.bitlinear_steps,
    )
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    return config


def main():
    config = parse_args()

    print("=" * 70)
    print("Chapter 8, Section 8.5 — Ternary (1.58-bit) models")
    print("=" * 70)
    print(f"  Mode:           {config.mode}")
    if config.mode in ("ptq", "all"):
        print(f"  PTQ model:      {config.ptq_model}")
    if config.mode in ("realmodel", "all"):
        print(f"  Ternary model:  {config.ternary_model}")
        print(f"  Baseline model: {config.baseline_model}")
    print(f"  Device:         {config.device}")

    if torch.cuda.is_available():
        cap = get_gpu_capability()
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU:            {gpu_name} (SM {cap[0]}.{cap[1]}, "
              f"{gpu_mem:.0f} GB)")
    else:
        print(f"  GPU:            None (CPU mode)")

    modes_to_run = (
        ["numberline", "ptq", "bitlinear", "realmodel"]
        if config.mode == "all" else [config.mode]
    )

    for mode in modes_to_run:
        if mode == "numberline":
            run_numberline(config)
        elif mode == "ptq":
            run_ptq(config)
        elif mode == "bitlinear":
            run_bitlinear(config)
        elif mode == "realmodel":
            run_realmodel(config)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()