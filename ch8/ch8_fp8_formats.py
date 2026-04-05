"""
Chapter 8, Section 8.2 — FP8 formats and kernel caveats
Companion script for "Efficient AI in Practice: Quantization and Fast Inference"

Modes:
  --mode encode      FP8 E4M3/E5M2 encoding/decoding, enumerate all representable values
  --mode numberline  Generate Figure 8.1 (E4M3 vs E5M2 vs INT8 number line)
  --mode scaling     Per-tensor vs block-32 scaling on OPT-6.7B activations
  --mode compare     Reconstruction MSE: FP8 E4M3 vs E5M2 vs INT8 on OPT-6.7B layers
  --mode deploy      FP8 weight quantization quality + storage verification (Hopper+)
  --mode all         Run all modes

Usage:
  # Full pipeline on H100
  python ch8_fp8_formats.py --mode all --save-plots

  # Format exploration only (CPU-safe, no model needed)
  python ch8_fp8_formats.py --mode encode --save-plots
  python ch8_fp8_formats.py --mode numberline --save-plots

  # CPU-only fallback (uses OPT-125M, illustrative only)
  python ch8_fp8_formats.py --mode all --save-plots --model opt-125m --device cpu

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
    "e4m3":   "#2166ac",   # Blue
    "e5m2":   "#b2182b",   # Red
    "int8":   "#4dac26",   # Green
    "bf16":   "#7570b3",   # Purple
    "block":  "#e08214",   # Orange
    "tensor": "#1b7837",   # Dark green
}

HATCHES = {
    "e4m3":   "",
    "e5m2":   "//",
    "int8":   "xx",
    "bf16":   "..",
    "block":  "\\\\",
    "tensor": "||",
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


# ─── FP8 format specification ────────────────────────────────────────────────

@dataclass
class FP8Spec:
    """Specification for an FP8 format variant."""
    name: str
    exponent_bits: int
    mantissa_bits: int
    bias: int
    has_infinity: bool
    max_value: float     # computed
    min_normal: float    # computed
    min_subnormal: float # computed

    @classmethod
    def e4m3(cls) -> "FP8Spec":
        # E4M3: 4 exponent bits, 3 mantissa bits, bias = 7                 #A
        # Special: no infinity, NaN = 0_1111_111 only (single NaN encoding)
        return cls(
            name="E4M3",
            exponent_bits=4,
            mantissa_bits=3,
            bias=7,
            has_infinity=False,
            max_value=448.0,            # 1.75 × 2^8 = 448
            min_normal=2**-6,           # 1.0 × 2^(1-7) = 0.015625
            min_subnormal=2**-9,        # 0.125 × 2^(-6) = 2^-9
        )

    @classmethod
    def e5m2(cls) -> "FP8Spec":
        # E5M2: 5 exponent bits, 2 mantissa bits, bias = 15                #B
        # Special: has infinity (11111_00), multiple NaN encodings
        return cls(
            name="E5M2",
            exponent_bits=5,
            mantissa_bits=2,
            bias=15,
            has_infinity=True,
            max_value=57344.0,          # 1.75 × 2^15 = 57344
            min_normal=2**-14,          # 1.0 × 2^(1-15)
            min_subnormal=2**-16,       # 0.25 × 2^(-14) = 2^-16
        )

#A E4M3 trades infinity for one extra mantissa bit — 8 levels per binade
#  vs E5M2's 4. This is why E4M3 is the inference format: more precision
#  near zero where activations cluster.
#B E5M2 mirrors IEEE FP16 conventions (5 exponent bits). The wider range
#  (±57,344 vs ±448) handles gradient magnitudes during training.


def enumerate_fp8_values(spec: FP8Spec) -> np.ndarray:
    """Enumerate all finite representable values for an FP8 format.

    Walks every 8-bit pattern and decodes using the standard IEEE-like
    formula: value = (-1)^s × 2^(e - bias) × (1 + m/2^M) for normals,
    and (-1)^s × 2^(1 - bias) × (m/2^M) for subnormals.
    """
    E = spec.exponent_bits
    M = spec.mantissa_bits
    bias = spec.bias
    e_max = (1 << E) - 1  # all-ones exponent

    values = []
    for bits in range(256):  # all 8-bit patterns
        sign = (bits >> 7) & 1
        exp_field = (bits >> M) & ((1 << E) - 1)
        mant_field = bits & ((1 << M) - 1)

        # Skip NaN and infinity encodings
        if exp_field == e_max:
            if spec.has_infinity:
                # E5M2: exp all-ones → inf (mant=0) or NaN (mant≠0)
                continue
            else:
                # E4M3fn: only exp all-ones AND mant all-ones → NaN
                if mant_field == (1 << M) - 1:
                    continue
                # All other exp=15 patterns are valid normal values

        # Decode
        if exp_field == 0:
            # Subnormal: value = (-1)^s × 2^(1-bias) × (mant / 2^M)
            value = (mant_field / (1 << M)) * (2.0 ** (1 - bias))
        else:
            # Normal: value = (-1)^s × 2^(exp-bias) × (1 + mant / 2^M)
            value = (1.0 + mant_field / (1 << M)) * (2.0 ** (exp_field - bias))

        if sign:
            value = -value

        values.append(value)

    return np.array(sorted(set(values)))


def fp8_encode_decode_demo(spec: FP8Spec) -> dict:
    """Demonstrate FP8 encode/decode from scratch for a few key values."""
    E = spec.exponent_bits
    M = spec.mantissa_bits
    bias = spec.bias

    examples = {
        "zero": 0.0,
        "one": 1.0,
        "min_subnormal": spec.min_subnormal,
        "min_normal": spec.min_normal,
        "max_value": spec.max_value,
        "negative_one": -1.0,
    }

    results = {}
    for label, value in examples.items():
        if value == 0.0:
            bits = 0b00000000
        else:
            sign = 1 if value < 0 else 0
            abs_val = abs(value)

            # Find exponent: floor(log2(abs_val))
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

            bits = (sign << 7) | (exp_field << M) | mant_field

        # Decode back
        sign_d = (bits >> 7) & 1
        exp_d = (bits >> M) & ((1 << E) - 1)
        mant_d = bits & ((1 << M) - 1)

        if exp_d == 0:
            decoded = (mant_d / (1 << M)) * (2.0 ** (1 - bias))
        else:
            decoded = (1.0 + mant_d / (1 << M)) * (2.0 ** (exp_d - bias))
        if sign_d:
            decoded = -decoded

        bit_str = f"{bits:08b}"
        layout = f"{bit_str[0]}_{bit_str[1:1+E]}_{bit_str[1+E:]}"

        results[label] = {
            "original": value,
            "bits": bits,
            "layout": layout,
            "decoded": decoded,
            "error": abs(decoded - value),
        }

    return results


# ─── FP8 quantization simulation ─────────────────────────────────────────────

def quantize_to_fp8_simulated(tensor: torch.Tensor,
                               spec: FP8Spec,
                               scale: float = 1.0) -> torch.Tensor:
    """Simulate FP8 quantization: scale → clamp → round-to-nearest → descale.

    This simulates FP8 quantization without requiring FP8 tensor core
    hardware. The rounding step snaps each value to the nearest FP8
    representable value by converting through the FP8 dtype on CPU.
    """
    scaled = tensor.float() / scale

    # Clamp to representable range
    scaled = scaled.clamp(-spec.max_value, spec.max_value)

    # Round to nearest FP8 representable value via PyTorch dtype cast
    if spec.mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    else:
        fp8_dtype = torch.float8_e5m2

    # Cast to FP8 and back — this performs correct rounding                 #A
    # Move to CPU for the dtype cast if needed (FP8 storage dtypes may
    # not be supported on all CUDA architectures for casting)
    orig_device = scaled.device
    try:
        quantized = scaled.to(fp8_dtype).float()
    except RuntimeError:
        quantized = scaled.cpu().to(fp8_dtype).float().to(orig_device)

    return quantized * scale

#A PyTorch's FP8 dtype cast handles rounding to nearest representable
#  value, subnormal encoding, and NaN/overflow clamping. We use the
#  dtype purely for value quantization — no FP8 tensor core arithmetic.


def compute_per_tensor_scale(tensor: torch.Tensor,
                              spec: FP8Spec) -> float:
    """Compute per-tensor scale: maps tensor amax to FP8 max representable."""
    amax = tensor.abs().max().item()
    if amax == 0:
        return 1.0
    return amax / spec.max_value                                            #B

#B Scale = amax / fp8_max. After dividing by scale, the largest tensor
#  value maps exactly to fp8_max. This is the per-tensor static scaling
#  strategy used by vLLM's --quantization fp8.


def compute_block_scales(tensor: torch.Tensor,
                          spec: FP8Spec,
                          block_size: int = 32) -> torch.Tensor:
    """Compute per-block scales for MXFP8-style block scaling.

    Partitions the last dimension into blocks of block_size, computes
    amax per block, and returns scales of shape [..., num_blocks].
    """
    orig_shape = tensor.shape
    flat = tensor.reshape(-1, orig_shape[-1])
    n_cols = flat.shape[-1]

    # Pad if not divisible
    pad_size = (block_size - n_cols % block_size) % block_size
    if pad_size > 0:
        flat = F.pad(flat, (0, pad_size))

    blocked = flat.reshape(flat.shape[0], -1, block_size)
    amax = blocked.abs().amax(dim=-1)                                       #C
    scales = amax / spec.max_value
    scales = scales.clamp(min=1e-12)
    return scales, blocked, pad_size, orig_shape

#C Each block of 32 values gets its own scale factor — 32× more scales
#  than per-tensor, but each scale covers a narrower range, reducing
#  quantization error on tensors with activation outliers.


def quantize_block_fp8(tensor: torch.Tensor,
                        spec: FP8Spec,
                        block_size: int = 32) -> torch.Tensor:
    """Quantize with per-block scaling and dequantize back to float."""
    scales, blocked, pad_size, orig_shape = compute_block_scales(
        tensor, spec, block_size
    )

    # Scale, quantize via FP8 dtype, dequantize
    fp8_dtype = (torch.float8_e4m3fn if spec.mantissa_bits == 3
                 else torch.float8_e5m2)
    scaled = blocked / scales.unsqueeze(-1)
    scaled = scaled.clamp(-spec.max_value, spec.max_value)
    try:
        quantized = scaled.to(fp8_dtype).float()
    except RuntimeError:
        quantized = scaled.cpu().to(fp8_dtype).float().to(scaled.device)
    dequantized = quantized * scales.unsqueeze(-1)

    # Reshape back, strip padding
    flat_deq = dequantized.reshape(blocked.shape[0], -1)
    n_cols = orig_shape[-1]
    if pad_size > 0:
        flat_deq = flat_deq[:, :n_cols]
    return flat_deq.reshape(orig_shape)


# ─── INT8 quantization (for comparison) ──────────────────────────────────────

def quantize_int8_per_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Symmetric per-tensor INT8 quantization (comparison baseline)."""
    amax = tensor.abs().max()
    scale = amax / 127.0 if amax > 0 else torch.tensor(1.0)
    quantized = torch.round(tensor / scale).clamp(-128, 127)
    return quantized * scale


def quantize_int8_per_channel(tensor: torch.Tensor,
                               axis: int = 0) -> torch.Tensor:
    """Symmetric per-channel INT8 quantization."""
    amax = tensor.abs().amax(dim=axis, keepdim=True)
    scale = amax / 127.0
    scale = scale.clamp(min=1e-12)
    quantized = torch.round(tensor / scale).clamp(-128, 127)
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

    # Determine dtype based on device capability
    cap = get_gpu_capability()
    if cap >= (8, 0):  # A100+: use BF16
        model_dtype = torch.bfloat16
    elif cap >= (7, 0):  # V100/T4: use FP16
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=model_dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.float()  # FP32 on CPU for accuracy

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


# ─── Activation capture ──────────────────────────────────────────────────────

def capture_activations(model, tokenizer, config: Config,
                        device: str, max_seqs: int = 8,
                        layer_indices: Optional[list] = None):
    """Capture fc1 (FFN up-projection) input activations from OPT layers.

    Returns dict mapping layer_index -> activation tensor [tokens, hidden_dim].
    """
    tokens = load_wikitext2_tokens(tokenizer, config)
    tokens = tokens[:max_seqs]

    # Determine which layers to capture
    num_layers = len(model.model.decoder.layers)
    if layer_indices is None:
        # Sample early, middle, late layers
        layer_indices = [0, 1, num_layers // 4, num_layers // 2,
                         3 * num_layers // 4, num_layers - 2, num_layers - 1]
        layer_indices = sorted(set(i for i in layer_indices
                                   if 0 <= i < num_layers))

    activations = {}
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            # fc1 input is the input to the up-projection
            if idx not in activations:
                activations[idx] = []
            # input is a tuple; take first element
            x = input[0].detach().float().cpu()
            activations[idx].append(x)
        return hook_fn

    for idx in layer_indices:
        layer = model.model.decoder.layers[idx]
        h = layer.fc1.register_forward_hook(make_hook(idx))
        hooks.append(h)

    with torch.no_grad():
        for seq_tokens in tokens:
            input_ids = seq_tokens.unsqueeze(0).to(
                next(model.parameters()).device
            )
            model(input_ids)

    for h in hooks:
        h.remove()

    # Concatenate across sequences
    for idx in activations:
        activations[idx] = torch.cat(activations[idx], dim=0)
        # Flatten batch×seq into rows: [total_tokens, hidden_dim]
        act = activations[idx]
        if act.dim() == 3:
            activations[idx] = act.reshape(-1, act.shape[-1])

    return activations


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
            nll = outputs.loss.item() * (seq.shape[0] - 1)  # per-token NLL
            total_nll += nll
            total_tokens += seq.shape[0] - 1

            if (i + 1) % 16 == 0:
                running_ppl = np.exp(total_nll / total_tokens)
                print(f"    Sequences {i+1}/{len(tokens)}, "
                      f"running PPL: {running_ppl:.2f}")

    ppl = np.exp(total_nll / total_tokens)
    return ppl


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: encode — FP8 bit layout and representable values
# ═══════════════════════════════════════════════════════════════════════════════

def run_encode(config: Config):
    print("\n" + "=" * 70)
    print("MODE: encode — FP8 E4M3/E5M2 encoding and representable values")
    print("=" * 70)

    for spec in [FP8Spec.e4m3(), FP8Spec.e5m2()]:
        print(f"\n{'─' * 60}")
        print(f"  Format: {spec.name} ({spec.exponent_bits}E{spec.mantissa_bits}M)")
        print(f"  Bias: {spec.bias}")
        print(f"  Has infinity: {spec.has_infinity}")
        print(f"  Max representable: {spec.max_value}")
        print(f"  Min normal: {spec.min_normal:.10f} (2^{int(np.log2(spec.min_normal))})")
        print(f"  Min subnormal: {spec.min_subnormal:.10f} (2^{int(np.log2(spec.min_subnormal))})")
        print(f"  Mantissa levels per binade: {2**spec.mantissa_bits}")

        # Enumerate all representable values
        values = enumerate_fp8_values(spec)
        positive = values[values > 0]
        print(f"\n  Total finite values: {len(values)}")
        print(f"  Unique positive values: {len(positive)}")
        print(f"  (including zero and negatives)")

        # Show encode/decode examples
        results = fp8_encode_decode_demo(spec)
        print(f"\n  Encode/decode examples:")
        print(f"  {'Value':<20} {'Bits':<12} {'Layout':<14} {'Decoded':<20} {'Error':<12}")
        print(f"  {'─' * 78}")
        for label, r in results.items():
            print(f"  {r['original']:<20.10f} {r['bits']:08b}     "
                  f"{r['layout']:<14} {r['decoded']:<20.10f} {r['error']:<12.2e}")

        # Verify against PyTorch's FP8 dtype
        fp8_dtype = (torch.float8_e4m3fn if spec.mantissa_bits == 3
                     else torch.float8_e5m2)
        print(f"\n  PyTorch FP8 dtype verification ({fp8_dtype}):")
        print(f"  {'Value':<20} {'Our decode':<15} {'PyTorch':<15} {'Match'}")
        print(f"  {'─' * 60}")
        for label, r in results.items():
            pt_rt = torch.tensor(r["original"]).to(fp8_dtype).float().item()
            match = abs(r["decoded"] - pt_rt) < 1e-10
            print(f"  {r['original']:<20.10f} {r['decoded']:<15.10f} "
                  f"{pt_rt:<15.10f} {'✓' if match else '✗'}")

    # Print the comparison table
    e4m3 = FP8Spec.e4m3()
    e5m2 = FP8Spec.e5m2()
    e4m3_vals = enumerate_fp8_values(e4m3)
    e5m2_vals = enumerate_fp8_values(e5m2)

    print(f"\n{'─' * 60}")
    print("  Summary: E4M3 vs E5M2 vs INT8 vs BF16")
    print(f"  {'Property':<30} {'E4M3':<15} {'E5M2':<15} {'INT8':<15} {'BF16':<15}")
    print(f"  {'─' * 90}")
    print(f"  {'Exponent bits':<30} {'4':<15} {'5':<15} {'—':<15} {'8':<15}")
    print(f"  {'Mantissa bits':<30} {'3':<15} {'2':<15} {'—':<15} {'7':<15}")
    print(f"  {'Max value':<30} {'448':<15} {'57,344':<15} {'127':<15} {'3.4e38':<15}")
    print(f"  {'Levels per binade':<30} {'8':<15} {'4':<15} {'—':<15} {'128':<15}")
    print(f"  {'Finite values (total)':<30} {len(e4m3_vals):<15} {len(e5m2_vals):<15} {'256':<15} {'~4B':<15}")
    print(f"  {'Has infinity':<30} {'No':<15} {'Yes':<15} {'No':<15} {'Yes':<15}")
    print(f"  {'Primary use':<30} {'Inference':<15} {'Gradients':<15} {'Inference':<15} {'Baseline':<15}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: numberline — Figure 8.1
# ═══════════════════════════════════════════════════════════════════════════════

def run_numberline(config: Config):
    print("\n" + "=" * 70)
    print("MODE: numberline — Figure 8.1 (E4M3 vs E5M2 vs INT8 number line)")
    print("=" * 70)

    apply_manning_style()

    e4m3 = FP8Spec.e4m3()
    e5m2 = FP8Spec.e5m2()

    e4m3_vals = enumerate_fp8_values(e4m3)
    e5m2_vals = enumerate_fp8_values(e5m2)

    # INT8 symmetric values for a reference range
    # Use scale that maps 127 → 2.0 (comparable zoom to FP8 near zero)
    int8_scale = 2.0 / 127.0
    int8_vals = np.arange(-128, 128) * int8_scale

    # ── Panel 1: Full positive range zoom [0, 8] ──
    fig, axes = plt.subplots(3, 1, figsize=(5.6, 3.0), sharex=True)

    x_max = 8.0

    for ax, vals, name, color in [
        (axes[0], e4m3_vals, "E4M3 (±448 range, 8 levels/binade)", COLORS["e4m3"]),
        (axes[1], e5m2_vals, "E5M2 (±57,344 range, 4 levels/binade)", COLORS["e5m2"]),
        (axes[2], int8_vals, f"INT8 symmetric (scale={int8_scale:.4f})", COLORS["int8"]),
    ]:
        mask = (vals >= 0) & (vals <= x_max)
        v = vals[mask]
        ax.eventplot([v], lineoffsets=0, linelengths=0.6,
                     colors=[color], linewidths=0.5)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=6.5, rotation=0, ha="right", va="center",
                      labelpad=5)
        ax.set_xlim(-0.1, x_max)
        ax.spines["left"].set_visible(False)

    axes[2].set_xlabel("Value")
    fig.suptitle("FP8 and INT8 representable values [0, 8]", fontsize=9, y=0.98)
    fig.tight_layout(rect=[0.22, 0.0, 1.0, 0.95])

    save_or_show(fig, "CH08_F01_Kalyanarangan", config)

    # ── Panel 2: Zoomed [0, 0.5] — where density matters most ──
    fig2, axes2 = plt.subplots(3, 1, figsize=(5.6, 3.0), sharex=True)

    x_max_zoom = 0.5

    for ax, vals, name, color in [
        (axes2[0], e4m3_vals, "E4M3", COLORS["e4m3"]),
        (axes2[1], e5m2_vals, "E5M2", COLORS["e5m2"]),
        (axes2[2], int8_vals, "INT8 symmetric", COLORS["int8"]),
    ]:
        mask = (vals >= 0) & (vals <= x_max_zoom)
        v = vals[mask]
        ax.eventplot([v], lineoffsets=0, linelengths=0.6,
                     colors=[color], linewidths=0.8)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=7, rotation=0, ha="right",
                      va="center", labelpad=5)
        ax.set_xlim(-0.01, x_max_zoom)
        ax.spines["left"].set_visible(False)

        # Annotate count
        ax.text(0.98, 0.85, f"{len(v)} values in [0, {x_max_zoom}]",
                transform=ax.transAxes, fontsize=6, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.8))

    axes2[2].set_xlabel("Value")
    fig2.suptitle(f"Near-zero density: representable values in [0, {x_max_zoom}]",
                  fontsize=9, y=0.98)
    fig2.tight_layout(rect=[0.15, 0.0, 1.0, 0.95])

    save_or_show(fig2, "CH08_F01b_Kalyanarangan_zoom", config)

    # Print density statistics
    print("\n  Density statistics (positive values only):")
    print(f"  {'Range':<20} {'E4M3':<10} {'E5M2':<10} {'INT8':<10}")
    print(f"  {'─' * 50}")
    for lo, hi in [(0, 0.1), (0, 0.5), (0, 1), (0, 2), (0, 8), (0, 448)]:
        c_e4 = np.sum((e4m3_vals >= lo) & (e4m3_vals <= hi))
        c_e5 = np.sum((e5m2_vals >= lo) & (e5m2_vals <= hi))
        c_i8 = np.sum((int8_vals >= lo) & (int8_vals <= hi))
        print(f"  [{lo}, {hi}]{'':<{14-len(f'[{lo}, {hi}]')}} "
              f"{c_e4:<10} {c_e5:<10} {c_i8:<10}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: scaling — Per-tensor vs block-32 scaling on model activations
# ═══════════════════════════════════════════════════════════════════════════════

def run_scaling(config: Config):
    print("\n" + "=" * 70)
    print("MODE: scaling — Per-tensor vs block-32 FP8 scaling comparison")
    print("=" * 70)

    device = resolve_device(config)
    model, tokenizer, model_dtype = load_model_and_tokenizer(config, device)
    e4m3 = FP8Spec.e4m3()

    # Capture activations
    print("\n  Capturing activations...")
    activations = capture_activations(model, tokenizer, config, device)

    results = []
    for layer_idx in sorted(activations.keys()):
        act = activations[layer_idx]
        act_f32 = act.float()

        # Per-tensor FP8 E4M3
        scale_pt = compute_per_tensor_scale(act_f32, e4m3)
        deq_pt = quantize_to_fp8_simulated(act_f32, e4m3, scale_pt)
        mse_pt = F.mse_loss(deq_pt, act_f32).item()

        # Block-32 FP8 E4M3
        deq_b32 = quantize_block_fp8(act_f32, e4m3, block_size=32)
        mse_b32 = F.mse_loss(deq_b32, act_f32).item()

        # Per-tensor INT8 (comparison)
        deq_int8 = quantize_int8_per_tensor(act_f32)
        mse_int8 = F.mse_loss(deq_int8, act_f32).item()

        # Activation statistics
        amax = act_f32.abs().max().item()
        outlier_ratio = (act_f32.abs() > 6.0).float().mean().item() * 100

        ratio = mse_pt / mse_b32 if mse_b32 > 0 else float("inf")

        results.append({
            "layer": layer_idx,
            "amax": amax,
            "outlier_pct": outlier_ratio,
            "mse_fp8_tensor": mse_pt,
            "mse_fp8_block32": mse_b32,
            "mse_int8_tensor": mse_int8,
            "block_improvement": ratio,
        })

    # Print results table
    print(f"\n  Per-tensor vs block-32 scaling — FP8 E4M3 on {config.model}")
    print(f"  Activations: fc1 input, {config.num_calibration_seqs} "
          f"sequences × {config.seq_length} tokens")
    print()
    print(f"  {'Layer':<8} {'Amax':<10} {'Out%':<8} "
          f"{'FP8 tensor':<14} {'FP8 blk-32':<14} {'INT8 tensor':<14} "
          f"{'Blk ratio':<10}")
    print(f"  {'─' * 78}")
    for r in results:
        print(f"  {r['layer']:<8} {r['amax']:<10.1f} "
              f"{r['outlier_pct']:<8.2f} "
              f"{r['mse_fp8_tensor']:<14.6f} "
              f"{r['mse_fp8_block32']:<14.6f} "
              f"{r['mse_int8_tensor']:<14.6f} "
              f"{r['block_improvement']:<10.1f}×")

    # ── Figure 8.2: Scaling comparison ──
    apply_manning_style()
    fig, ax = plt.subplots(figsize=(5.6, 3.5))

    layers = [r["layer"] for r in results]
    x = np.arange(len(layers))
    width = 0.25

    bars_tensor = ax.bar(x - width, [r["mse_fp8_tensor"] for r in results],
                          width, label="FP8 E4M3 per-tensor",
                          color=COLORS["tensor"], hatch=HATCHES["tensor"],
                          edgecolor="black", linewidth=0.5)
    bars_block = ax.bar(x, [r["mse_fp8_block32"] for r in results],
                         width, label="FP8 E4M3 block-32",
                         color=COLORS["block"], hatch=HATCHES["block"],
                         edgecolor="black", linewidth=0.5)
    bars_int8 = ax.bar(x + width, [r["mse_int8_tensor"] for r in results],
                        width, label="INT8 per-tensor",
                        color=COLORS["int8"], hatch=HATCHES["int8"],
                        edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Reconstruction MSE (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title(f"Activation quantization error: scaling strategies ({config.model})")

    fig.tight_layout()
    save_or_show(fig, "CH08_F02_Kalyanarangan", config)

    # Clean up model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: compare — Reconstruction MSE across formats
# ═══════════════════════════════════════════════════════════════════════════════

def run_compare(config: Config):
    print("\n" + "=" * 70)
    print("MODE: compare — Reconstruction MSE: FP8 vs INT8 on model weights")
    print("=" * 70)

    device = resolve_device(config)
    model, tokenizer, model_dtype = load_model_and_tokenizer(config, device)

    e4m3 = FP8Spec.e4m3()
    e5m2 = FP8Spec.e5m2()

    # Collect weight MSE for fc1 and fc2 layers
    num_layers = len(model.model.decoder.layers)
    layer_indices = list(range(num_layers))

    results = []
    for idx in layer_indices:
        layer = model.model.decoder.layers[idx]

        for proj_name, proj in [("fc1", layer.fc1), ("fc2", layer.fc2)]:
            w = proj.weight.detach().float().cpu()

            # FP8 E4M3 per-tensor
            scale_e4 = compute_per_tensor_scale(w, e4m3)
            deq_e4 = quantize_to_fp8_simulated(w, e4m3, scale_e4)
            mse_e4 = F.mse_loss(deq_e4, w).item()

            # FP8 E5M2 per-tensor
            scale_e5 = compute_per_tensor_scale(w, e5m2)
            deq_e5 = quantize_to_fp8_simulated(w, e5m2, scale_e5)
            mse_e5 = F.mse_loss(deq_e5, w).item()

            # INT8 per-channel (standard baseline)
            deq_i8 = quantize_int8_per_channel(w, axis=0)
            mse_i8 = F.mse_loss(deq_i8, w).item()

            results.append({
                "layer": idx,
                "proj": proj_name,
                "mse_e4m3": mse_e4,
                "mse_e5m2": mse_e5,
                "mse_int8": mse_i8,
                "shape": tuple(w.shape),
            })

    # Print results (fc1 only for brevity)
    fc1_results = [r for r in results if r["proj"] == "fc1"]
    print(f"\n  Weight reconstruction MSE — fc1 layers ({config.model})")
    print(f"  {'Layer':<8} {'Shape':<20} {'FP8 E4M3':<14} "
          f"{'FP8 E5M2':<14} {'INT8 per-ch':<14}")
    print(f"  {'─' * 70}")
    for r in fc1_results:
        print(f"  {r['layer']:<8} {str(r['shape']):<20} "
              f"{r['mse_e4m3']:<14.2e} {r['mse_e5m2']:<14.2e} "
              f"{r['mse_int8']:<14.2e}")

    # Average
    avg_e4 = np.mean([r["mse_e4m3"] for r in fc1_results])
    avg_e5 = np.mean([r["mse_e5m2"] for r in fc1_results])
    avg_i8 = np.mean([r["mse_int8"] for r in fc1_results])
    print(f"  {'Avg':<8} {'—':<20} {avg_e4:<14.2e} {avg_e5:<14.2e} "
          f"{avg_i8:<14.2e}")

    # ── Figure: Weight MSE comparison (fc1 layers) ──
    apply_manning_style()

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    x = np.arange(len(fc1_results))
    width = 0.25

    ax.bar(x - width, [r["mse_e4m3"] for r in fc1_results], width,
           label="FP8 E4M3", color=COLORS["e4m3"],
           hatch=HATCHES["e4m3"], edgecolor="black", linewidth=0.5)
    ax.bar(x, [r["mse_e5m2"] for r in fc1_results], width,
           label="FP8 E5M2", color=COLORS["e5m2"],
           hatch=HATCHES["e5m2"], edgecolor="black", linewidth=0.5)
    ax.bar(x + width, [r["mse_int8"] for r in fc1_results], width,
           label="INT8 per-channel", color=COLORS["int8"],
           hatch=HATCHES["int8"], edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Weight MSE (log scale)")

    # Show every Nth tick for readability
    n_layers = len(fc1_results)
    if n_layers > 16:
        tick_step = max(1, n_layers // 8)
        tick_indices = list(range(0, n_layers, tick_step))
        ax.set_xticks([x[i] for i in tick_indices])
        ax.set_xticklabels([str(fc1_results[i]["layer"]) for i in tick_indices])
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([str(r["layer"]) for r in fc1_results])

    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title(f"Weight quantization error by format ({config.model} fc1)")

    fig.tight_layout()
    save_or_show(fig, "CH08_F02b_Kalyanarangan_weights", config)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# MODE: deploy — FP8 inference on H100 with perplexity
# ═══════════════════════════════════════════════════════════════════════════════

def run_deploy(config: Config):
    print("\n" + "=" * 70)
    print("MODE: deploy — FP8 weight quantization quality and storage verification")
    print("=" * 70)

    device = resolve_device(config)
    cap = get_gpu_capability()

    # ── Hardware capability gate ──
    if cap < (9, 0):                                                        #D
        print(f"\n  GPU compute capability: {cap[0]}.{cap[1]}")
        print(f"  FP8 tensor cores require SM ≥ 9.0 (Hopper/H100+).")
        if cap >= (7, 5):
            print(f"  Your GPU (SM {cap[0]}.{cap[1]}) supports INT8/INT4 "
                  f"tensor cores but not FP8.")
        print(f"\n  Running simulation-only path (quantize weights to FP8, "
              f"dequantize to {('BF16' if cap >= (8,0) else 'FP16')} "
              f"before matmul).")
        print(f"  This measures quality impact but NOT throughput gains.")
        print(f"  For real FP8 acceleration, run on H100 or later.\n")
        _deploy_simulation(config, device)
        return

    print(f"\n  GPU: {torch.cuda.get_device_name()}")
    print(f"  Compute capability: {cap[0]}.{cap[1]} — FP8 tensor cores available")

    _deploy_h100(config, device)

#D The hardware gate. FP8 tensor cores shipped on Hopper (SM 9.0).
#  T4 = SM 7.5, A100 = SM 8.0 — neither has FP8 hardware.
#  The simulation path below quantizes and dequantizes correctly but
#  computes matmul in BF16/FP16, so throughput is unchanged.


def _deploy_simulation(config: Config, device: str):
    """Simulation path for pre-Hopper GPUs: FP8 weight quantization quality."""
    model, tokenizer, model_dtype = load_model_and_tokenizer(config, device)

    # Baseline perplexity
    print("\n  Evaluating baseline perplexity...")
    ppl_baseline = evaluate_perplexity(model, tokenizer, config, device)
    dtype_name = {torch.bfloat16: "BF16", torch.float16: "FP16",
                  torch.float32: "FP32"}[model_dtype]
    print(f"  {dtype_name} baseline: {ppl_baseline:.2f}")

    # Compute memory
    param_bytes = sum(p.numel() * p.element_size()
                      for p in model.parameters())
    mem_baseline_mb = param_bytes / (1024 * 1024)

    # Quantize all linear weights to FP8 E4M3 (simulated: dequantize in place)
    e4m3 = FP8Spec.e4m3()
    n_quantized = 0

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data
            w_f32 = w.float()
            scale = compute_per_tensor_scale(w_f32, e4m3)

            # Quantize to FP8 and dequantize back to original dtype
            w_deq = quantize_to_fp8_simulated(w_f32, e4m3, scale)
            module.weight.data = w_deq.to(w.dtype)
            n_quantized += 1

    print(f"\n  Quantized {n_quantized} linear layers to FP8 E4M3 (simulated)")
    print("  Evaluating FP8-simulated perplexity...")
    ppl_fp8 = evaluate_perplexity(model, tokenizer, config, device)
    print(f"  FP8 E4M3 (simulated): {ppl_fp8:.2f}")

    delta = (ppl_fp8 - ppl_baseline) / ppl_baseline * 100

    # Memory estimate (FP8 = 1 byte per param for linear layers)
    linear_params = sum(m.weight.numel() for m in model.modules()
                        if isinstance(m, torch.nn.Linear))
    total_params = sum(p.numel() for p in model.parameters())
    non_linear_bytes = (total_params - linear_params) * (
        2 if model_dtype in (torch.float16, torch.bfloat16) else 4
    )
    fp8_bytes = linear_params * 1 + non_linear_bytes
    mem_fp8_mb = fp8_bytes / (1024 * 1024)

    print(f"\n  ─── Results ───")
    print(f"  {'Config':<30} {'Perplexity':<15} {'Δ vs baseline':<15} "
          f"{'Memory (est.)':<15}")
    print(f"  {'─' * 75}")
    print(f"  {dtype_name + ' baseline':<30} {ppl_baseline:<15.2f} "
          f"{'—':<15} {mem_baseline_mb:<15.1f} MB")
    print(f"  {'FP8 E4M3 per-tensor (sim.)':<30} {ppl_fp8:<15.2f} "
          f"{delta:+.2f}%{'':<9} {mem_fp8_mb:<15.1f} MB")
    print(f"\n  Note: Memory shown is estimated FP8 model size.")
    print(f"  Throughput is unchanged in simulation — matmul runs in "
          f"{dtype_name}.")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _deploy_h100(config: Config, device: str):
    """FP8 quality measurement on Hopper+: quantize to FP8, dequantize to
    BF16 for perplexity, verify FP8 storage, and demonstrate _scaled_mm."""
    model, tokenizer, model_dtype = load_model_and_tokenizer(config, device)

    # ── Step 1: Baseline perplexity ──
    print("\n  Step 1: Evaluating BF16 baseline perplexity...")
    ppl_baseline = evaluate_perplexity(model, tokenizer, config, device)
    print(f"  BF16 baseline: {ppl_baseline:.2f}")

    # Memory baseline
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        mem_baseline = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        mem_baseline = 0

    # ── Step 2: FP8 weight quantization with real FP8 storage ──
    print("\n  Step 2: Quantizing weights to FP8 E4M3...")
    e4m3 = FP8Spec.e4m3()
    n_quantized = 0
    fp8_data = {}  # Store FP8 weights and scales

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = module.weight.data
            w_f32 = w.float()
            scale = compute_per_tensor_scale(w_f32, e4m3)

            # Store actual FP8 tensor (1 byte per element)                  #E
            w_scaled = (w_f32 / scale).clamp(
                -e4m3.max_value, e4m3.max_value
            )
            w_fp8 = w_scaled.to(torch.float8_e4m3fn)

            fp8_data[name + ".weight"] = {
                "fp8": w_fp8,
                "scale": torch.tensor(scale, dtype=torch.float32),
            }

            # For perplexity eval, dequantize in-place to BF16
            module.weight.data = (w_fp8.float() * scale).to(torch.bfloat16)
            n_quantized += 1

    print(f"  Quantized {n_quantized} layers — weights stored as FP8 E4M3")

    #E We store the actual torch.float8_e4m3fn tensor (1 byte/element).
    #  On Hopper+, torch._scaled_mm can consume this directly for FP8
    #  tensor core matmul. For perplexity measurement, we dequantize to
    #  BF16 because the standard model forward pass expects BF16 inputs.
    #  The perplexity captures the full effect of FP8 rounding error;
    #  real throughput gains require a serving runtime that dispatches
    #  FP8 matmul kernels instead of dequantizing first.

    # ── Step 3: Verify FP8 storage is real ──
    sample_key = list(fp8_data.keys())[0]
    sample_fp8 = fp8_data[sample_key]["fp8"]
    print(f"\n  Step 3: FP8 storage verification:")
    print(f"    Layer: {sample_key}")
    print(f"    dtype: {sample_fp8.dtype}")
    print(f"    element_size: {sample_fp8.element_size()} byte(s)")
    print(f"    shape: {sample_fp8.shape}")
    print(f"    storage bytes: {sample_fp8.numel() * sample_fp8.element_size():,}")

    # ── Step 4: FP8 perplexity ──
    print("\n  Step 4: Evaluating FP8 perplexity (dequantized matmul)...")
    ppl_fp8 = evaluate_perplexity(model, tokenizer, config, device)
    print(f"  FP8 E4M3: {ppl_fp8:.2f}")

    # ── Step 5: Demonstrate torch._scaled_mm (actual FP8 matmul) ──
    print("\n  Step 5: Demonstrating torch._scaled_mm (FP8 tensor core matmul)...")
    try:
        # Pick a representative layer
        test_layer = model.model.decoder.layers[0].fc1
        w_entry = fp8_data["model.decoder.layers.0.fc1.weight"]
        w_fp8 = w_entry["fp8"].cuda()
        w_scale = w_entry["scale"].cuda()

        # Create a dummy activation in FP8
        hidden_dim = w_fp8.shape[1]
        x_bf16 = torch.randn(1, 32, hidden_dim, device="cuda",
                              dtype=torch.bfloat16)
        x_f32 = x_bf16.float()
        x_scale_val = x_f32.abs().max().item() / e4m3.max_value
        x_fp8 = (x_f32 / x_scale_val).clamp(
            -e4m3.max_value, e4m3.max_value
        ).to(torch.float8_e4m3fn).reshape(-1, hidden_dim)

        # FP8 matmul via scaled_mm
        scale_a = torch.tensor(x_scale_val, dtype=torch.float32, device="cuda")
        scale_b = w_scale

        # torch._scaled_mm expects (M, K) @ (K, N)
        # A must be row-major (contiguous), B must be column-major
        # .t() without .contiguous() preserves column-major strides
        result_fp8 = torch._scaled_mm(
            x_fp8,
            w_fp8.t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )

        # Compare against BF16 matmul
        result_bf16 = F.linear(x_bf16.reshape(-1, hidden_dim),
                               test_layer.weight.data)

        cosine_sim = F.cosine_similarity(
            result_fp8.float().flatten(),
            result_bf16.float().flatten(),
            dim=0,
        ).item()

        print(f"    FP8 matmul output shape: {result_fp8.shape}")
        print(f"    Cosine similarity (FP8 vs BF16): {cosine_sim:.6f}")
        print(f"    ✓ FP8 tensor core matmul verified")

    except Exception as e:
        print(f"    torch._scaled_mm failed: {e}")
        print(f"    Perplexity measurement is unaffected — it uses the")
        print(f"    dequantized path. FP8 matmul dispatch requires compatible")
        print(f"    PyTorch + driver versions for this GPU.")

    # ── Step 6: Memory comparison ──
    param_bytes_bf16 = sum(p.numel() * 2 for p in model.parameters())
    linear_params = sum(m.weight.numel() for m in model.modules()
                        if isinstance(m, torch.nn.Linear))
    other_params = sum(p.numel() for p in model.parameters()) - linear_params
    param_bytes_fp8 = linear_params * 1 + other_params * 2  # FP8 + BF16 rest

    delta = (ppl_fp8 - ppl_baseline) / ppl_baseline * 100

    gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    print(f"\n  ─── Results ({config.model} on {gpu_name}) ───")
    print(f"  {'Config':<30} {'Perplexity':<12} {'Δ vs BF16':<12} "
          f"{'Model size':<15} {'Compression':<12}")
    print(f"  {'─' * 81}")
    print(f"  {'BF16 baseline':<30} {ppl_baseline:<12.2f} {'—':<12} "
          f"{param_bytes_bf16/(1024**3):.2f} GB{'':<8} {'1.00×':<12}")
    compression = param_bytes_bf16 / param_bytes_fp8
    print(f"  {'FP8 E4M3 per-tensor':<30} {ppl_fp8:<12.2f} "
          f"{delta:+.2f}%{'':<6} "
          f"{param_bytes_fp8/(1024**3):.2f} GB{'':<8} "
          f"{compression:.2f}×")

    # ── Figure 8.3: Memory + perplexity comparison ──
    apply_manning_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.8))

    # Left: Memory
    configs = ["BF16", "FP8 E4M3"]
    mem_values = [param_bytes_bf16 / (1024**3), param_bytes_fp8 / (1024**3)]
    colors = [COLORS["bf16"], COLORS["e4m3"]]
    hatches_list = [HATCHES["bf16"], HATCHES["e4m3"]]

    bars = ax1.bar(configs, mem_values, color=colors, edgecolor="black",
                   linewidth=0.5, hatch=[HATCHES["bf16"], HATCHES["e4m3"]])
    ax1.set_ylabel("Model size (GB)")
    ax1.set_title("Memory")
    for bar, val in zip(bars, mem_values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    # Right: Perplexity
    ppl_values = [ppl_baseline, ppl_fp8]
    bars2 = ax2.bar(configs, ppl_values, color=colors, edgecolor="black",
                    linewidth=0.5, hatch=[HATCHES["bf16"], HATCHES["e4m3"]])
    ax2.set_ylabel("Perplexity (WikiText-2)")
    ax2.set_title("Quality")
    # Set y-axis to show small differences
    ppl_min = min(ppl_values) * 0.98
    ppl_max = max(ppl_values) * 1.02
    ax2.set_ylim(ppl_min, ppl_max)
    for bar, val in zip(bars2, ppl_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"FP8 inference: memory and quality ({config.model})",
                 fontsize=9)
    fig.tight_layout()
    save_or_show(fig, "CH08_F03_Kalyanarangan", config)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Ch8 §8.2 — FP8 formats and kernel caveats"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["encode", "numberline", "scaling", "compare",
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
    print("Chapter 8, Section 8.2 — FP8 formats and kernel caveats")
    print("=" * 70)
    print(f"  Mode:   {config.mode}")
    print(f"  Model:  {config.model}")
    print(f"  Device: {config.device}")

    # Check PyTorch version for FP8 dtype support
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if torch_version < (2, 1):
        print(f"\n  WARNING: PyTorch {torch.__version__} may not support "
              f"float8 dtypes. Recommend PyTorch >= 2.1.")

    # Report GPU info
    if torch.cuda.is_available():
        cap = get_gpu_capability()
        gpu_name = torch.cuda.get_device_name()
        print(f"  GPU:    {gpu_name} (SM {cap[0]}.{cap[1]})")
        if cap >= (9, 0):
            print(f"  FP8:    ✓ Hardware FP8 tensor cores available")
        else:
            print(f"  FP8:    ✗ No FP8 tensor cores (simulation only)")
    else:
        print(f"  GPU:    None (CPU mode)")

    modes_to_run = (
        ["encode", "numberline", "scaling", "compare", "deploy"]
        if config.mode == "all" else [config.mode]
    )

    for mode in modes_to_run:
        if mode == "encode":
            run_encode(config)
        elif mode == "numberline":
            run_numberline(config)
        elif mode == "scaling":
            run_scaling(config)
        elif mode == "compare":
            run_compare(config)
        elif mode == "deploy":
            run_deploy(config)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()