"""
Chapter 8, Section 8.3 — FP4 with blockwise scaling
Companion script for "Efficient AI in Practice: Quantization and Fast Inference"

Modes:
  --mode encode      FP4 E2M1 encoding/decoding, enumerate all 16 representable values
  --mode numberline  Generate Figure 8.7 (FP4 E2M1 vs INT4 vs FP8 E4M3 number line)
  --mode blockscale  Block size sweep on OPT-6.7B weights — why blockwise is mandatory
  --mode compare     Weight reconstruction MSE: FP4 vs INT4 vs FP8 across block sizes
  --mode deploy      FP4 weight quantization quality + perplexity on OPT-6.7B
  --mode all         Run all modes

Usage:
  # Full pipeline on H100
  python ch8_fp4_blockscale.py --mode all --save-plots

  # Format exploration only (CPU-safe, no model needed)
  python ch8_fp4_blockscale.py --mode encode --save-plots
  python ch8_fp4_blockscale.py --mode numberline --save-plots

  # CPU-only fallback (uses OPT-125M, illustrative only)
  python ch8_fp4_blockscale.py --mode all --save-plots --model opt-125m --device cpu

Requires: torch >= 2.4, transformers, datasets, matplotlib
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ─── Configuration ───────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_MAP = {
    "opt-6.7b": "facebook/opt-6.7b",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-350m": "facebook/opt-350m",
    "opt-125m": "facebook/opt-125m",
}


@dataclass
class Config:
    mode: str = "all"
    model: str = "opt-6.7b"
    device: str = "auto"
    save_plots: bool = False
    output_dir: Path = SCRIPT_DIR / "figures"
    num_calibration_seqs: int = 64
    seq_length: int = 512
    seed: int = 42


# ─── Manning figure style ────────────────────────────────────────────────────

# Colors: grayscale-safe palette with distinct hatching for B&W print
COLORS = {
    "fp4":      "#d95f02",   # Orange
    "fp4_b16":  "#d95f02",   # Orange (block-16)
    "fp4_b32":  "#e6ab02",   # Gold (block-32)
    "fp4_b64":  "#66a61e",   # Olive (block-64)
    "fp4_b128": "#7570b3",   # Purple (block-128)
    "fp4_pt":   "#e7298a",   # Magenta (per-tensor)
    "int4":     "#1b9e77",   # Teal
    "fp8":      "#2166ac",   # Blue
    "bf16":     "#7570b3",   # Purple
}

HATCHES = {
    "fp4":      "",
    "fp4_b16":  "",
    "fp4_b32":  "//",
    "fp4_b64":  "\\\\",
    "fp4_b128": "..",
    "fp4_pt":   "xx",
    "int4":     "++",
    "fp8":      "||",
    "bf16":     "..",
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


# ─── FP4 E2M1 format specification ──────────────────────────────────────────

@dataclass
class FP4Spec:
    """Specification for the FP4 E2M1 format (OCP MX / NVFP4).

    E2M1 uses 1 sign bit, 2 exponent bits, 1 mantissa bit, bias = 1.
    No infinity, no NaN — all 16 bit patterns encode finite values.
    This yields exactly 7 unique positive values, 7 negative, zero,
    and negative zero (treated as zero).
    """
    name: str
    exponent_bits: int
    mantissa_bits: int
    bias: int
    has_infinity: bool
    max_value: float
    min_normal: float
    min_subnormal: float

    @classmethod
    def e2m1(cls) -> "FP4Spec":
        # E2M1: 2 exponent bits, 1 mantissa bit, bias = 1               #A
        # Max exponent field = 3, so max power = 2^(3-1) = 4
        # Max mantissa = 1.1₂ = 1.5, so max value = 4 × 1.5 = 6.0
        # No infinity, no NaN — every pattern is a valid number
        return cls(
            name="E2M1",
            exponent_bits=2,
            mantissa_bits=1,
            bias=1,
            has_infinity=False,
            max_value=6.0,              # 1.5 × 2^2 = 6.0
            min_normal=1.0,             # 1.0 × 2^(1-1) = 1.0
            min_subnormal=0.5,          # 0.5 × 2^(1-1) = 0.5
        )

#A E2M1 is the only standardized 4-bit float. NVIDIA's NVFP4 format on
#  Blackwell uses E2M1 with blockwise FP8 scale factors (one E4M3 scale
#  per 16-element block). The OCP Microscaling (MX) specification
#  standardizes this as MXFP4 with shared block exponents.


def enumerate_fp4_values(spec: FP4Spec) -> np.ndarray:
    """Enumerate all finite representable values for the FP4 E2M1 format.

    Walks every 4-bit pattern and decodes using the standard formula:
      Normal:    (-1)^s × 2^(e - bias) × (1 + m/2^M)
      Subnormal: (-1)^s × 2^(1 - bias) × (m/2^M)
    """
    E = spec.exponent_bits
    M = spec.mantissa_bits
    bias = spec.bias

    values = []
    for bits in range(16):  # all 4-bit patterns                        #B
        sign = (bits >> 3) & 1
        exp_field = (bits >> M) & ((1 << E) - 1)
        mant_field = bits & ((1 << M) - 1)

        # No reserved patterns — all 16 are valid finite values
        if exp_field == 0:
            # Subnormal: value = (-1)^s × 2^(1-bias) × (m / 2^M)
            value = (mant_field / (1 << M)) * (2.0 ** (1 - bias))
        else:
            # Normal: value = (-1)^s × 2^(e-bias) × (1 + m / 2^M)
            value = (1.0 + mant_field / (1 << M)) * (2.0 ** (exp_field - bias))

        if sign:
            value = -value

        values.append(value)

    return np.array(sorted(set(values)))

#B Only 16 patterns exist. Compare with FP8's 256 patterns — FP4 has
#  16× fewer bit patterns, which translates to dramatically fewer
#  representable values: 7 unique positive values vs FP8 E4M3's 127.


def fp4_encode_decode_demo(spec: FP4Spec) -> dict:
    """Demonstrate FP4 encode/decode from scratch for every representable value."""
    E = spec.exponent_bits
    M = spec.mantissa_bits
    bias = spec.bias

    # Show all positive values — there are only 8 (including zero)
    all_values = enumerate_fp4_values(spec)
    positive = all_values[all_values >= 0]

    results = {}
    for value in positive:
        if value == 0.0:
            bits = 0b0000
        else:
            sign = 0
            abs_val = abs(value)

            if abs_val < 2.0 ** (1 - bias):
                # Subnormal
                exp_field = 0
                mant_val = abs_val / (2.0 ** (1 - bias))
                mant_field = int(round(mant_val * (1 << M)))
                mant_field = min(mant_field, (1 << M) - 1)
            else:
                exp_unbiased = int(math.floor(math.log2(abs_val)))
                exp_field = exp_unbiased + bias
                significand = abs_val / (2.0 ** exp_unbiased) - 1.0
                mant_field = int(round(significand * (1 << M)))
                if mant_field >= (1 << M):
                    mant_field = 0
                    exp_field += 1

            bits = (sign << 3) | (exp_field << M) | mant_field

        # Decode back
        sign_d = (bits >> 3) & 1
        exp_d = (bits >> M) & ((1 << E) - 1)
        mant_d = bits & ((1 << M) - 1)

        if exp_d == 0:
            decoded = (mant_d / (1 << M)) * (2.0 ** (1 - bias))
        else:
            decoded = (1.0 + mant_d / (1 << M)) * (2.0 ** (exp_d - bias))
        if sign_d:
            decoded = -decoded

        bit_str = f"{bits:04b}"
        layout = f"{bit_str[0]}_{bit_str[1:1+E]}_{bit_str[1+E:]}"

        label = f"{value:.1f}"
        results[label] = {
            "original": value,
            "bits": bits,
            "layout": layout,
            "decoded": decoded,
            "error": abs(decoded - value),
        }

    return results


# ─── FP4 quantization ───────────────────────────────────────────────────────

def quantize_to_fp4_simulated(tensor: torch.Tensor,
                               spec: FP4Spec,
                               scale: float = 1.0) -> torch.Tensor:
    """Simulate FP4 E2M1 quantization: scale, round to nearest E2M1 value,
    descale.

    Since PyTorch has no native FP4 dtype, we build a lookup table of all
    positive E2M1 values and snap each element to the nearest entry.        #C
    """
    scaled = tensor.float() / scale

    # Build the E2M1 representable value table (positive only, then mirror)
    pos_values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    fp4_levels = torch.tensor(pos_values, dtype=torch.float32,
                               device=tensor.device)

    # Clamp to representable range
    clamped = scaled.clamp(-spec.max_value, spec.max_value)

    # Round to nearest: compute absolute, find closest FP4 level, restore sign
    signs = clamped.sign()
    abs_vals = clamped.abs()

    # Vectorized nearest-level lookup                                       #D
    # Expand dims: abs_vals [..., 1] vs fp4_levels [1, ..., K]
    abs_flat = abs_vals.reshape(-1, 1)
    distances = (abs_flat - fp4_levels.unsqueeze(0)).abs()
    nearest_indices = distances.argmin(dim=1)
    quantized_flat = fp4_levels[nearest_indices]

    quantized = quantized_flat.reshape(abs_vals.shape) * signs
    return quantized * scale

#C No PyTorch FP4 dtype exists (as of PyTorch 2.6). FP4 support lives
#  in specialized libraries (bitsandbytes, TensorRT-LLM) and in hardware
#  (Blackwell tensor cores). We simulate by snapping to the known E2M1
#  value grid — the quantization error is identical to hardware FP4.
#D The lookup-table approach is O(n × 8) — 8 positive levels. For FP8 we
#  used PyTorch's native dtype cast; for FP4 we must do it ourselves.


def compute_per_tensor_scale_fp4(tensor: torch.Tensor,
                                  spec: FP4Spec) -> float:
    """Compute per-tensor scale: maps tensor amax to FP4 max representable."""
    amax = tensor.abs().max().item()
    if amax == 0:
        return 1.0
    return amax / spec.max_value


def quantize_block_fp4(tensor: torch.Tensor,
                        spec: FP4Spec,
                        block_size: int = 16) -> torch.Tensor:
    """Quantize with per-block scaling and dequantize back to float.

    NVFP4 uses block size 16 with an E4M3 scale factor per block.
    The OCP MX spec allows block sizes of 16 or 32.
    """
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, orig_shape[-1]).float()
    n_cols = flat.shape[-1]

    # Pad if not divisible
    pad_size = (block_size - n_cols % block_size) % block_size
    if pad_size > 0:
        flat = F.pad(flat, (0, pad_size))

    blocked = flat.reshape(flat.shape[0], -1, block_size)

    # Per-block scale from local amax                                      #E
    amax = blocked.abs().amax(dim=-1)
    scales = amax / spec.max_value
    scales = scales.clamp(min=1e-12)

    # Scale → quantize via lookup → dequantize
    scaled = blocked / scales.unsqueeze(-1)
    scaled = scaled.clamp(-spec.max_value, spec.max_value)

    # Round each element to nearest E2M1 value
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

    # Dequantize
    dequantized = quantized * scales.unsqueeze(-1)

    # Reshape back, strip padding
    flat_deq = dequantized.reshape(blocked.shape[0], -1)
    if pad_size > 0:
        flat_deq = flat_deq[:, :n_cols]
    return flat_deq.reshape(orig_shape)

#E Each block of 16 values gets its own FP32 scale. NVFP4 stores these
#  scales as FP8 E4M3 values (1 byte per scale) to minimize overhead:
#  1 byte scale / 16 elements = 0.0625 bytes overhead per element.
#  We use FP32 scales here for accuracy; the scale quantization error
#  is negligible compared to the E2M1 rounding error.


# ─── INT4 and FP8 quantization (for comparison) ─────────────────────────────

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


def load_model_and_tokenizer(config: Config, device: str):
    """Load OPT model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODEL_MAP.get(config.model, config.model)
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


def load_wikitext2_tokens(tokenizer, config: Config) -> torch.Tensor:
    """Load WikiText-2 test set, tokenize, return as [num_seqs, seq_length]."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    tokens = tokenizer.encode(text, return_tensors="pt",
                               add_special_tokens=False)[0]

    total_needed = config.num_calibration_seqs * config.seq_length
    tokens = tokens[:total_needed]
    num_seqs = tokens.shape[0] // config.seq_length
    tokens = tokens[:num_seqs * config.seq_length]
    tokens = tokens.reshape(num_seqs, config.seq_length)

    print(f"  WikiText-2: {num_seqs} sequences × {config.seq_length} tokens")
    return tokens


# ─── Perplexity evaluation ───────────────────────────────────────────────────

def evaluate_perplexity(model, tokenizer, config: Config,
                        device: str) -> float:
    """Evaluate WikiText-2 perplexity with the standard protocol."""
    tokens = load_wikitext2_tokens(tokenizer, config)
    model_device = next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, seq in enumerate(tokens):
            input_ids = seq.unsqueeze(0).to(model_device)
            outputs = model(input_ids, labels=input_ids)
            nll = outputs.loss.item() * (seq.shape[0] - 1)
            total_nll += nll
            total_tokens += seq.shape[0] - 1

            if (i + 1) % 16 == 0:
                running_ppl = np.exp(total_nll / total_tokens)
                print(f"    Sequences {i+1}/{len(tokens)}, "
                      f"running PPL: {running_ppl:.2f}")

    ppl = np.exp(total_nll / total_tokens)
    return ppl


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: encode — FP4 E2M1 bit layout and all representable values
# ═══════════════════════════════════════════════════════════════════════════════

def run_encode(config: Config):
    print("\n" + "=" * 70)
    print("MODE: encode — FP4 E2M1 encoding and representable values")
    print("=" * 70)

    spec = FP4Spec.e2m1()

    print(f"\n{'─' * 60}")
    print(f"  Format: {spec.name} ({spec.exponent_bits}E{spec.mantissa_bits}M)")
    print(f"  Bias: {spec.bias}")
    print(f"  Has infinity: {spec.has_infinity}")
    print(f"  Max representable: {spec.max_value}")
    print(f"  Min normal: {spec.min_normal}")
    print(f"  Min subnormal: {spec.min_subnormal}")
    print(f"  Mantissa levels per binade: {2**spec.mantissa_bits}")

    # Enumerate all values
    values = enumerate_fp4_values(spec)
    positive = values[values > 0]
    print(f"\n  Total finite values (including ±0): {len(values)}")
    print(f"  Unique positive values: {len(positive)}")

    # Show every single value — there are only 15 (or 16 with -0)
    print(f"\n  Complete FP4 E2M1 value table:")
    print(f"  {'Value':<10} {'Type':<12}")
    print(f"  {'─' * 22}")
    for v in values:
        if v == 0.0:
            vtype = "zero"
        elif abs(v) < spec.min_normal:
            vtype = "subnormal"
        else:
            vtype = "normal"
        print(f"  {v:<10.1f} {vtype:<12}")

    # Encode/decode for all positive values
    results = fp4_encode_decode_demo(spec)
    print(f"\n  Encode/decode — all positive E2M1 values:")
    print(f"  {'Value':<10} {'Bits':<8} {'Layout':<10} {'Decoded':<10} "
          f"{'Error':<10}")
    print(f"  {'─' * 48}")
    for label, r in results.items():
        print(f"  {r['original']:<10.1f} {r['bits']:04b}     "
              f"{r['layout']:<10} {r['decoded']:<10.1f} {r['error']:<10.2e}")

    # Compare with FP8 E4M3 and INT4
    print(f"\n{'─' * 60}")
    print("  Comparison: FP4 E2M1 vs FP8 E4M3 vs INT4")
    print(f"  {'Property':<30} {'FP4 E2M1':<15} {'FP8 E4M3':<15} "
          f"{'INT4 sym':<15}")
    print(f"  {'─' * 75}")
    print(f"  {'Total bits':<30} {'4':<15} {'8':<15} {'4':<15}")
    print(f"  {'Exponent bits':<30} {'2':<15} {'4':<15} {'—':<15}")
    print(f"  {'Mantissa bits':<30} {'1':<15} {'3':<15} {'—':<15}")
    print(f"  {'Unique positive values':<30} {'7':<15} {'127':<15} {'7':<15}")
    print(f"  {'Max value':<30} {'6.0':<15} {'448':<15} {'7':<15}")
    print(f"  {'Levels per binade':<30} {'2':<15} {'8':<15} {'—':<15}")
    print(f"  {'Spacing near zero':<30} {'non-uniform':<15} {'non-uniform':<15}"
          f" {'uniform':<15}")
    print(f"  {'Hardware accel.':<30} {'Blackwell':<15} {'H100+':<15} "
          f"{'Ampere+':<15}")

    # The key insight: spacing comparison
    print(f"\n  Spacing between consecutive positive values:")
    for v1, v2 in zip(positive[:-1], positive[1:]):
        gap = v2 - v1
        print(f"    {v1:.1f} → {v2:.1f}:  gap = {gap:.1f}")
    print(f"\n  Note: Gaps double with each binade (1→2 gap=0.5, 2→4 gap=1.0,"
          f" 4→8 gap=2.0).")
    print(f"  With only 2 levels per binade, the spacing is very coarse.")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: numberline — Figure 8.7: FP4 vs INT4 vs FP8 number line
# ═══════════════════════════════════════════════════════════════════════════════

def run_numberline(config: Config):
    print("\n" + "=" * 70)
    print("MODE: numberline — FP4 E2M1 vs INT4 vs FP8 E4M3 number line")
    print("=" * 70)

    apply_manning_style()

    spec = FP4Spec.e2m1()
    fp4_vals = enumerate_fp4_values(spec)

    # INT4 symmetric: 15 levels from -7 to +7, with per-tensor scale
    # Use scale that maps 7 → 6.0 (same max as FP4 for fair visual comparison)
    int4_scale = 6.0 / 7.0
    int4_vals = np.arange(-8, 8) * int4_scale

    # FP8 E4M3 for comparison — enumerate the full set, then filter
    # We import the enumeration inline to keep this script self-contained
    fp8_vals = _enumerate_fp8_e4m3()

    # ── Panel 1: Full positive range [0, 6] ──
    fig, axes = plt.subplots(3, 1, figsize=(5.6, 3.0), sharex=True)

    x_max = 6.5

    for ax, vals, name, color in [
        (axes[0], fp4_vals,
         "FP4 E2M1 (7 positive values, ±6 range)", COLORS["fp4"]),
        (axes[1], int4_vals,
         f"INT4 symmetric (scale={int4_scale:.3f})", COLORS["int4"]),
        (axes[2], fp8_vals,
         "FP8 E4M3 (127 positive values, ±448 range)", COLORS["fp8"]),
    ]:
        mask = (vals >= 0) & (vals <= x_max)
        v = vals[mask]
        ax.eventplot([v], lineoffsets=0, linelengths=0.6,
                     colors=[color], linewidths=0.8)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=6, rotation=0, ha="right", va="center",
                      labelpad=5)
        ax.set_xlim(-0.1, x_max)
        ax.spines["left"].set_visible(False)

        # Annotate count
        ax.text(0.98, 0.85, f"{len(v)} values in [0, {x_max:.0f}]",
                transform=ax.transAxes, fontsize=6, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

    axes[2].set_xlabel("Value")
    fig.suptitle("FP4, INT4, and FP8 representable values [0, 6]",
                 fontsize=9, y=0.98)
    fig.tight_layout(rect=[0.28, 0.0, 1.0, 0.95])

    save_or_show(fig, "CH08_F04_Kalyanarangan", config)

    # ── Panel 2: Zoomed [0, 2] — where most weights live ──
    fig2, axes2 = plt.subplots(3, 1, figsize=(5.6, 3.0), sharex=True)

    x_max_zoom = 2.1

    for ax, vals, name, color in [
        (axes2[0], fp4_vals, "FP4 E2M1", COLORS["fp4"]),
        (axes2[1], int4_vals, "INT4 symmetric", COLORS["int4"]),
        (axes2[2], fp8_vals, "FP8 E4M3", COLORS["fp8"]),
    ]:
        mask = (vals >= 0) & (vals <= x_max_zoom)
        v = vals[mask]
        ax.eventplot([v], lineoffsets=0, linelengths=0.6,
                     colors=[color], linewidths=0.8)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=7, rotation=0, ha="right",
                      va="center", labelpad=5)
        ax.set_xlim(-0.05, x_max_zoom)
        ax.spines["left"].set_visible(False)

        ax.text(0.98, 0.85, f"{len(v)} values in [0, 2]",
                transform=ax.transAxes, fontsize=6, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

    axes2[2].set_xlabel("Value")
    fig2.suptitle("Near-zero density: representable values in [0, 2]",
                  fontsize=9, y=0.98)
    fig2.tight_layout(rect=[0.15, 0.0, 1.0, 0.95])

    save_or_show(fig2, "CH08_F04b_Kalyanarangan_zoom", config)

    # Print density statistics
    print("\n  Density statistics (positive values):")
    print(f"  {'Range':<15} {'FP4 E2M1':<12} {'INT4':<12} {'FP8 E4M3':<12}")
    print(f"  {'─' * 51}")
    for lo, hi in [(0, 0.5), (0, 1), (0, 2), (0, 6)]:
        c_fp4 = np.sum((fp4_vals >= lo) & (fp4_vals <= hi))
        c_int4 = np.sum((int4_vals >= lo) & (int4_vals <= hi))
        c_fp8 = np.sum((fp8_vals >= lo) & (fp8_vals <= hi))
        print(f"  [{lo}, {hi}]{'':<{9-len(f'[{lo}, {hi}]')}} "
              f"{c_fp4:<12} {c_int4:<12} {c_fp8:<12}")


def _enumerate_fp8_e4m3() -> np.ndarray:
    """Enumerate all finite E4M3 values (self-contained helper)."""
    E, M, bias = 4, 3, 7
    e_max = 15
    values = []
    for bits in range(256):
        sign = (bits >> 7) & 1
        exp_field = (bits >> M) & ((1 << E) - 1)
        mant_field = bits & ((1 << M) - 1)
        if exp_field == e_max and mant_field == (1 << M) - 1:
            continue  # NaN
        if exp_field == 0:
            value = (mant_field / (1 << M)) * (2.0 ** (1 - bias))
        else:
            value = (1.0 + mant_field / (1 << M)) * (2.0 ** (exp_field - bias))
        if sign:
            value = -value
        values.append(value)
    return np.array(sorted(set(values)))


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: blockscale — Block size sweep, why blockwise is mandatory
# ═══════════════════════════════════════════════════════════════════════════════

def run_blockscale(config: Config):
    print("\n" + "=" * 70)
    print("MODE: blockscale — Block size sweep on model weights (FP4 E2M1)")
    print("=" * 70)

    device = resolve_device(config)
    model, tokenizer, model_dtype = load_model_and_tokenizer(config, device)
    spec = FP4Spec.e2m1()

    block_sizes = [16, 32, 64, 128]
    block_labels = {16: "b16", 32: "b32", 64: "b64", 128: "b128"}

    num_layers = len(model.model.decoder.layers)
    # Sample representative layers
    if num_layers > 16:
        sample_indices = [0, 1, num_layers // 4, num_layers // 2,
                          3 * num_layers // 4, num_layers - 2, num_layers - 1]
    else:
        sample_indices = list(range(num_layers))
    sample_indices = sorted(set(i for i in sample_indices
                                if 0 <= i < num_layers))

    results = []
    for idx in sample_indices:
        layer = model.model.decoder.layers[idx]
        w = layer.fc1.weight.detach().float().cpu()

        # Per-tensor FP4
        scale_pt = compute_per_tensor_scale_fp4(w, spec)
        deq_pt = quantize_to_fp4_simulated(w, spec, scale_pt)
        mse_pt = F.mse_loss(deq_pt, w).item()

        row = {"layer": idx, "mse_per_tensor": mse_pt, "shape": tuple(w.shape)}

        # Blockwise FP4 at each block size
        for bs in block_sizes:
            deq_blk = quantize_block_fp4(w, spec, block_size=bs)
            mse_blk = F.mse_loss(deq_blk, w).item()
            row[f"mse_block_{bs}"] = mse_blk

        # Compute scale overhead per element for each block size
        for bs in block_sizes:
            # 4 bytes (FP32 scale) per block of bs elements
            # FP4 = 0.5 bytes/element, scale overhead = 4/bs bytes/element
            overhead_pct = (4.0 / bs) / 0.5 * 100  # as % of FP4 storage
            row[f"overhead_pct_{bs}"] = overhead_pct

        results.append(row)

    # Print results table
    print(f"\n  Block size sweep — FP4 E2M1 on {config.model} fc1 weights")
    print(f"  {'Layer':<8} {'Per-tensor':<14}", end="")
    for bs in block_sizes:
        print(f"{'Block-' + str(bs):<14}", end="")
    print(f"{'Ratio pt/b16':<14}")
    print(f"  {'─' * (8 + 14 + 14 * len(block_sizes) + 14)}")

    for r in results:
        print(f"  {r['layer']:<8} {r['mse_per_tensor']:<14.2e}", end="")
        for bs in block_sizes:
            print(f"{r[f'mse_block_{bs}']:<14.2e}", end="")
        ratio = r['mse_per_tensor'] / r['mse_block_16'] if r['mse_block_16'] > 0 else float('inf')
        print(f"{ratio:<14.1f}×")

    # Storage overhead table
    print(f"\n  Scale overhead per block size (FP32 scales):")
    print(f"  {'Block size':<15} {'Scale bytes/elem':<20} {'Overhead vs FP4':<20}"
          f" {'Effective bits':<15}")
    print(f"  {'─' * 70}")
    for bs in block_sizes:
        scale_bytes = 4.0 / bs
        overhead_pct = scale_bytes / 0.5 * 100
        effective_bits = 4.0 + (32.0 / bs)  # 4 bits + 32 scale bits / block_size
        print(f"  {bs:<15} {scale_bytes:<20.4f} {overhead_pct:<20.1f}% "
              f"{effective_bits:<15.2f}")
    # Per-tensor for reference
    print(f"  {'per-tensor':<15} {'≈0':<20} {'≈0':<20}% {'4.00':<15}")

    # ── Figure 8.8: Block size comparison ──
    apply_manning_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.5))

    layers = [r["layer"] for r in results]
    x = np.arange(len(layers))
    n_bars = len(block_sizes) + 1  # block sizes + per-tensor
    width = 0.8 / n_bars

    # Per-tensor bars (worst)
    ax.bar(x - width * n_bars / 2 + width * 0.5,
           [r["mse_per_tensor"] for r in results],
           width, label="Per-tensor",
           color=COLORS["fp4_pt"], hatch=HATCHES["fp4_pt"],
           edgecolor="black", linewidth=0.5)

    # Block size bars
    block_colors = ["fp4_b128", "fp4_b64", "fp4_b32", "fp4_b16"]
    for i, (bs, ckey) in enumerate(zip(reversed(block_sizes),
                                        block_colors)):
        offset = x - width * n_bars / 2 + width * (i + 1.5)
        ax.bar(offset,
               [r[f"mse_block_{bs}"] for r in results],
               width, label=f"Block-{bs}",
               color=COLORS[ckey], hatch=HATCHES[ckey],
               edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Weight reconstruction MSE (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(loc="upper right", framealpha=0.9, fontsize=6)
    ax.set_title(f"FP4 E2M1: block scaling vs per-tensor ({config.model} fc1)")

    fig.tight_layout()
    save_or_show(fig, "CH08_F05_Kalyanarangan", config)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: compare — Weight reconstruction across format families
# ═══════════════════════════════════════════════════════════════════════════════

def run_compare(config: Config):
    print("\n" + "=" * 70)
    print("MODE: compare — Weight MSE: FP4 vs INT4 vs FP8 ({})".format(
        config.model))
    print("=" * 70)

    device = resolve_device(config)
    model, tokenizer, model_dtype = load_model_and_tokenizer(config, device)
    spec = FP4Spec.e2m1()

    num_layers = len(model.model.decoder.layers)
    # Use all layers for fc1
    layer_indices = list(range(num_layers))

    results = []
    for idx in layer_indices:
        layer = model.model.decoder.layers[idx]
        w = layer.fc1.weight.detach().float().cpu()

        # FP4 E2M1 per-tensor
        scale_pt = compute_per_tensor_scale_fp4(w, spec)
        deq_fp4_pt = quantize_to_fp4_simulated(w, spec, scale_pt)
        mse_fp4_pt = F.mse_loss(deq_fp4_pt, w).item()

        # FP4 E2M1 block-16 (NVFP4 standard)
        deq_fp4_b16 = quantize_block_fp4(w, spec, block_size=16)
        mse_fp4_b16 = F.mse_loss(deq_fp4_b16, w).item()

        # INT4 symmetric per-tensor
        deq_int4_pt = quantize_int4_per_tensor(w)
        mse_int4_pt = F.mse_loss(deq_int4_pt, w).item()

        # INT4 group-128 (GPTQ/AWQ style)
        deq_int4_g128 = quantize_int4_group(w, group_size=128)
        mse_int4_g128 = F.mse_loss(deq_int4_g128, w).item()

        # FP8 E4M3 per-tensor (8-bit reference)
        deq_fp8 = quantize_fp8_e4m3_per_tensor(w)
        mse_fp8 = F.mse_loss(deq_fp8, w).item()

        results.append({
            "layer": idx,
            "mse_fp4_pt": mse_fp4_pt,
            "mse_fp4_b16": mse_fp4_b16,
            "mse_int4_pt": mse_int4_pt,
            "mse_int4_g128": mse_int4_g128,
            "mse_fp8": mse_fp8,
            "shape": tuple(w.shape),
        })

    # Print results
    print(f"\n  Weight reconstruction MSE — fc1 layers ({config.model})")
    print(f"  {'Layer':<7} {'FP4 pt':<12} {'FP4 b16':<12} {'INT4 pt':<12}"
          f" {'INT4 g128':<12} {'FP8 E4M3':<12}")
    print(f"  {'─' * 67}")
    for r in results:
        print(f"  {r['layer']:<7} {r['mse_fp4_pt']:<12.2e} "
              f"{r['mse_fp4_b16']:<12.2e} {r['mse_int4_pt']:<12.2e} "
              f"{r['mse_int4_g128']:<12.2e} {r['mse_fp8']:<12.2e}")

    # Averages
    avg = {k: np.mean([r[k] for r in results])
           for k in ["mse_fp4_pt", "mse_fp4_b16", "mse_int4_pt",
                      "mse_int4_g128", "mse_fp8"]}
    print(f"  {'Avg':<7} {avg['mse_fp4_pt']:<12.2e} "
          f"{avg['mse_fp4_b16']:<12.2e} {avg['mse_int4_pt']:<12.2e} "
          f"{avg['mse_int4_g128']:<12.2e} {avg['mse_fp8']:<12.2e}")

    # Ratios
    print(f"\n  Ratios vs FP8 E4M3 per-tensor (lower = closer to FP8):")
    for label, key in [("FP4 per-tensor", "mse_fp4_pt"),
                        ("FP4 block-16", "mse_fp4_b16"),
                        ("INT4 per-tensor", "mse_int4_pt"),
                        ("INT4 group-128", "mse_int4_g128")]:
        ratio = avg[key] / avg["mse_fp8"]
        print(f"    {label:<20} {ratio:>8.1f}× worse than FP8")

    # ── Figure 8.9: Cross-format weight MSE comparison ──                  #F
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
        ("FP4 per-tensor", "mse_fp4_pt", COLORS["fp4_pt"], HATCHES["fp4_pt"]),
        ("FP4 block-16", "mse_fp4_b16", COLORS["fp4_b16"], HATCHES["fp4_b16"]),
        ("INT4 per-tensor", "mse_int4_pt", COLORS["int4"], "++"),
        ("INT4 group-128", "mse_int4_g128", COLORS["int4"], "//"),
        ("FP8 E4M3", "mse_fp8", COLORS["fp8"], HATCHES["fp8"]),
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
    ax.set_title(f"Weight quantization error across formats ({config.model} fc1)")

    fig.tight_layout()
    save_or_show(fig, "CH08_F06_Kalyanarangan", config)

    #F This figure is the Rosetta Stone of Section 8.3: it shows that FP4
    #  block-16 achieves MSE competitive with INT4 group-128, both being
    #  dramatically better than their per-tensor counterparts. The FP8
    #  reference line shows the cost of halving the bit width.

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: deploy — FP4 weight quantization quality on OPT-6.7B
# ═══════════════════════════════════════════════════════════════════════════════

def run_deploy(config: Config):
    print("\n" + "=" * 70)
    print("MODE: deploy — FP4 weight quantization quality and perplexity")
    print("=" * 70)

    device = resolve_device(config)
    cap = get_gpu_capability()

    # ── Hardware note ──
    if cap < (10, 0):                                                       #G
        print(f"\n  GPU compute capability: "
              f"{cap[0]}.{cap[1] if cap[0] > 0 else 'N/A'}")
        print(f"  FP4 tensor cores require SM ≥ 10.0 (Blackwell).")
        print(f"  Running simulation path: FP4 weights → dequantize → "
              f"{'BF16' if cap >= (8, 0) else 'FP16'} matmul.")
        print(f"  This measures quality impact but NOT throughput gains.")
        print(f"  For FP4 acceleration, Blackwell hardware is required.\n")

    #G FP4 tensor cores debuted on Blackwell (SM 10.0). H100 = SM 9.0 has
    #  FP8 but not FP4. As of early 2025, no publicly available cloud
    #  instance offers Blackwell GPUs for general use. The simulation path
    #  is therefore the only option for reproducible quality measurement.

    model, tokenizer, model_dtype = load_model_and_tokenizer(config, device)
    spec = FP4Spec.e2m1()

    # ── Step 1: Baseline perplexity ──
    print("\n  Step 1: Evaluating baseline perplexity...")
    ppl_baseline = evaluate_perplexity(model, tokenizer, config, device)
    dtype_name = {torch.bfloat16: "BF16", torch.float16: "FP16",
                  torch.float32: "FP32"}[model_dtype]
    print(f"  {dtype_name} baseline: {ppl_baseline:.2f}")

    # Memory baseline
    param_bytes_baseline = sum(p.numel() * p.element_size()
                               for p in model.parameters())

    # ── Step 2: FP4 per-tensor (expected to degrade) ──
    print("\n  Step 2: FP4 E2M1 per-tensor quantization...")
    # Deep copy weights for per-tensor experiment
    original_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            original_weights[name] = module.weight.data.clone()

    n_quantized = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data.float()
            scale = compute_per_tensor_scale_fp4(w, spec)
            w_deq = quantize_to_fp4_simulated(w, spec, scale)
            module.weight.data = w_deq.to(module.weight.dtype)
            n_quantized += 1

    print(f"  Quantized {n_quantized} linear layers (FP4 per-tensor)")
    ppl_fp4_pt = evaluate_perplexity(model, tokenizer, config, device)
    print(f"  FP4 per-tensor: {ppl_fp4_pt:.2f}")

    # ── Step 3: Restore weights and apply FP4 block-16 ──
    print("\n  Step 3: FP4 E2M1 block-16 quantization...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in original_weights:
            module.weight.data = original_weights[name]

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data.float()
            w_deq = quantize_block_fp4(w, spec, block_size=16)
            module.weight.data = w_deq.to(module.weight.dtype)

    ppl_fp4_b16 = evaluate_perplexity(model, tokenizer, config, device)
    print(f"  FP4 block-16: {ppl_fp4_b16:.2f}")

    # ── Step 4: Restore weights and apply FP4 block-32 ──
    print("\n  Step 4: FP4 E2M1 block-32 quantization...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in original_weights:
            module.weight.data = original_weights[name]

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data.float()
            w_deq = quantize_block_fp4(w, spec, block_size=32)
            module.weight.data = w_deq.to(module.weight.dtype)

    ppl_fp4_b32 = evaluate_perplexity(model, tokenizer, config, device)
    print(f"  FP4 block-32: {ppl_fp4_b32:.2f}")

    # ── Memory calculations ──                                             #H
    linear_params = sum(m.weight.numel() for m in model.modules()
                        if isinstance(m, torch.nn.Linear))
    other_params = sum(p.numel() for p in model.parameters()) - linear_params
    bytes_per_other = 2 if model_dtype in (torch.float16, torch.bfloat16) else 4

    # FP4: 0.5 bytes per weight
    fp4_weight_bytes = linear_params * 0.5
    other_bytes = other_params * bytes_per_other

    # Block-16 scale overhead: 1 FP32 scale per 16 elements
    # In NVFP4, scales are E4M3 (1 byte), but we compute for FP32 here
    scale_bytes_b16 = (linear_params / 16) * 4  # FP32 scales
    scale_bytes_b16_nvfp4 = (linear_params / 16) * 1  # E4M3 scales (NVFP4)
    scale_bytes_b32 = (linear_params / 32) * 4

    bytes_fp4_b16 = fp4_weight_bytes + scale_bytes_b16 + other_bytes
    bytes_fp4_b16_nvfp4 = fp4_weight_bytes + scale_bytes_b16_nvfp4 + other_bytes
    bytes_fp4_b32 = fp4_weight_bytes + scale_bytes_b32 + other_bytes
    bytes_fp4_pt = fp4_weight_bytes + other_bytes  # negligible scale overhead

    #H Memory accounting: FP4 packs two weights per byte (0.5 bytes/weight).
    #  Scale overhead depends on block size and scale format:
    #    Block-16 with FP32 scales: +0.25 bytes/weight (50% overhead)
    #    Block-16 with E4M3 scales (NVFP4): +0.0625 bytes/weight (12.5%)
    #    Block-32 with FP32 scales: +0.125 bytes/weight (25% overhead)
    #  NVFP4 uses E4M3 scales specifically to keep the overhead low.

    # ── Results table ──
    delta_pt = (ppl_fp4_pt - ppl_baseline) / ppl_baseline * 100
    delta_b16 = (ppl_fp4_b16 - ppl_baseline) / ppl_baseline * 100
    delta_b32 = (ppl_fp4_b32 - ppl_baseline) / ppl_baseline * 100

    gpu_name = (torch.cuda.get_device_name()
                if torch.cuda.is_available() else "CPU")
    print(f"\n  ─── Results ({config.model} on {gpu_name}) ───")
    print(f"  {'Config':<30} {'PPL':<10} {'Δ vs BF16':<12} "
          f"{'Size (est.)':<14} {'Compression':<12}")
    print(f"  {'─' * 78}")
    print(f"  {dtype_name + ' baseline':<30} {ppl_baseline:<10.2f} "
          f"{'—':<12} "
          f"{param_bytes_baseline/(1024**3):.2f} GB{'':<6} {'1.00×':<12}")
    print(f"  {'FP4 per-tensor':<30} {ppl_fp4_pt:<10.2f} "
          f"{delta_pt:+.2f}%{'':<5} "
          f"{bytes_fp4_pt/(1024**3):.2f} GB{'':<6} "
          f"{param_bytes_baseline/bytes_fp4_pt:.2f}×")
    print(f"  {'FP4 block-16 (FP32 scales)':<30} {ppl_fp4_b16:<10.2f} "
          f"{delta_b16:+.2f}%{'':<5} "
          f"{bytes_fp4_b16/(1024**3):.2f} GB{'':<6} "
          f"{param_bytes_baseline/bytes_fp4_b16:.2f}×")
    print(f"  {'FP4 block-32 (FP32 scales)':<30} {ppl_fp4_b32:<10.2f} "
          f"{delta_b32:+.2f}%{'':<5} "
          f"{bytes_fp4_b32/(1024**3):.2f} GB{'':<6} "
          f"{param_bytes_baseline/bytes_fp4_b32:.2f}×")

    print(f"\n  NVFP4 memory note:")
    print(f"    NVFP4 stores block-16 scales as E4M3 (1 byte), not FP32.")
    print(f"    Estimated NVFP4 size: "
          f"{bytes_fp4_b16_nvfp4/(1024**3):.2f} GB "
          f"({param_bytes_baseline/bytes_fp4_b16_nvfp4:.2f}× compression)")

    # ── Figure 8.10: Perplexity vs memory ──
    apply_manning_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.8))

    # Left: Memory comparison
    configs = [dtype_name, "FP4\nper-tensor", "FP4\nblock-16", "FP4\nblock-32"]
    mem_values = [
        param_bytes_baseline / (1024**3),
        bytes_fp4_pt / (1024**3),
        bytes_fp4_b16 / (1024**3),
        bytes_fp4_b32 / (1024**3),
    ]
    colors_list = [COLORS["bf16"], COLORS["fp4_pt"],
                   COLORS["fp4_b16"], COLORS["fp4_b32"]]
    hatches_list = [HATCHES["bf16"], HATCHES["fp4_pt"],
                    HATCHES["fp4_b16"], HATCHES["fp4_b32"]]

    bars = ax1.bar(configs, mem_values, color=colors_list,
                   edgecolor="black", linewidth=0.5, hatch=hatches_list)
    ax1.set_ylabel("Model size (GB)")
    ax1.set_title("Memory")
    for bar, val in zip(bars, mem_values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=6)

    # Right: Perplexity comparison
    ppl_values = [ppl_baseline, ppl_fp4_pt, ppl_fp4_b16, ppl_fp4_b32]
    bars2 = ax2.bar(configs, ppl_values, color=colors_list,
                    edgecolor="black", linewidth=0.5, hatch=hatches_list)
    ax2.set_ylabel("Perplexity (WikiText-2)")
    ax2.set_title("Quality")

    # Emphasize quality difference — use a y-axis that shows the gap
    ppl_min = min(ppl_values) * 0.95
    ppl_max = max(ppl_values) * 1.05
    ax2.set_ylim(ppl_min, ppl_max)

    for bar, val in zip(bars2, ppl_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}", ha="center", va="bottom", fontsize=6)

    fig.suptitle(f"FP4 inference: memory and quality ({config.model})",
                 fontsize=9)
    fig.tight_layout()
    save_or_show(fig, "CH08_F07_Kalyanarangan", config)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Ch8 §8.3 — FP4 with blockwise scaling"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["encode", "numberline", "blockscale", "compare",
                 "deploy", "all"],
        help="Which experiment to run"
    )
    parser.add_argument(
        "--model", default="opt-6.7b",
        choices=list(MODEL_MAP.keys()),
        help="Model to use (default: opt-6.7b)"
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
        help="Number of calibration/evaluation sequences (default: 64)"
    )
    parser.add_argument(
        "--seq-length", type=int, default=512,
        help="Sequence length (default: 512)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for figures"
    )
    args = parser.parse_args()

    config = Config(
        mode=args.mode,
        model=args.model,
        device=args.device,
        save_plots=args.save_plots,
        num_calibration_seqs=args.num_seqs,
        seq_length=args.seq_length,
    )
    if args.output_dir:
        config.output_dir = Path(args.output_dir)

    return config


def main():
    config = parse_args()

    print("=" * 70)
    print("Chapter 8, Section 8.3 — FP4 with blockwise scaling")
    print("=" * 70)
    print(f"  Mode:   {config.mode}")
    print(f"  Model:  {config.model}")
    print(f"  Device: {config.device}")

    # Report GPU info
    if torch.cuda.is_available():
        cap = get_gpu_capability()
        gpu_name = torch.cuda.get_device_name()
        print(f"  GPU:    {gpu_name} (SM {cap[0]}.{cap[1]})")
        if cap >= (10, 0):
            print(f"  FP4:    ✓ Hardware FP4 tensor cores available (Blackwell)")
        elif cap >= (9, 0):
            print(f"  FP4:    ✗ No FP4 tensor cores (FP8 available, H100)")
        else:
            print(f"  FP4:    ✗ No FP4 tensor cores (simulation only)")
    else:
        print(f"  GPU:    None (CPU mode)")

    modes_to_run = (
        ["encode", "numberline", "blockscale", "compare", "deploy"]
        if config.mode == "all" else [config.mode]
    )

    for mode in modes_to_run:
        if mode == "encode":
            run_encode(config)
        elif mode == "numberline":
            run_numberline(config)
        elif mode == "blockscale":
            run_blockscale(config)
        elif mode == "compare":
            run_compare(config)
        elif mode == "deploy":
            run_deploy(config)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()