#!/usr/bin/env python3
"""
Chapter 5, Section 5.5: Adapt Transformers Efficiently
=====================================================

Companion script for "Efficient AI in Practice: Quantization and Fast Inference"
Manning Publications

Demonstrates:
  1. Transformer layer sensitivity profiling — measure which sub-layers
     (attention Q/K/V/O projections, FFN up/down, LayerNorm) are most
     sensitive to fake quantization in BERT-base-uncased.
  2. Selective QAT — apply fake quantizers only to resilient layers,
     keep sensitive layers in FP32 (mixed-precision QAT).
  3. The softmax amplification problem — show how small quantization
     errors in Q/K projections get amplified through softmax.
  4. Quantization map visualization — which parts of a transformer block
     get INT8, which stay FP32, and why.

Usage:
  python ch5_transformer_qat.py --mode sensitivity       # Layer sensitivity profiling
  python ch5_transformer_qat.py --mode softmax-amp       # Softmax amplification demo
  python ch5_transformer_qat.py --mode selective-qat     # Selective QAT on BERT
  python ch5_transformer_qat.py --mode all --save-plots  # Run everything

Requirements:
  pip install torch transformers datasets matplotlib
"""

import argparse
import sys
import warnings
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Manning-compliant figure styling ──────────────────────────────────────────
MANNING_COLORS = {
    "blue": "#4A90B8",
    "orange": "#D4823E",
    "green": "#5EA55E",
    "red": "#C04E4E",
    "purple": "#8B6DAF",
    "gray": "#7A7A7A",
    "dark": "#2D2D2D",
    "light_gray": "#D0D0D0",
}
HATCHES = ["//", "\\\\", "xx", "..", "||", "--", "++", "oo"]

def setup_manning_style():
    """Apply Manning Publications figure styling."""
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })

setup_manning_style()


# ── Core fake quantization (reused from Section 5.2/5.4) ─────────────────────

def fake_quantize_per_channel(weight, bits=8, axis=0):
    """Per-channel symmetric fake quantization (from Section 5.4, Listing 5.22).
    
    Returns the fake-quantized weight (same shape, FP32 dtype) and
    the quantization MSE.
    """
    q_max = (1 << (bits - 1)) - 1
    # Flatten all dims except channel axis
    if axis == 0:
        flat = weight.detach().reshape(weight.shape[0], -1)
    else:
        perm = list(range(weight.dim()))
        perm[0], perm[axis] = perm[axis], perm[0]
        flat = weight.detach().permute(perm).reshape(weight.shape[axis], -1)

    abs_max = flat.abs().amax(dim=1)
    scales = abs_max / q_max
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)

    # Broadcast scales
    shape = [1] * weight.dim()
    shape[axis] = -1
    scales_bc = scales.view(shape)

    w_q = torch.clamp(torch.round(weight / scales_bc), -q_max, q_max)
    w_dq = w_q * scales_bc

    mse = (weight - w_dq).pow(2).mean().item()
    return w_dq, mse


def fake_quantize_per_tensor(tensor, bits=8):
    """Per-tensor symmetric fake quantization for activations."""
    q_max = (1 << (bits - 1)) - 1
    abs_max = tensor.detach().abs().max()
    if abs_max == 0:
        return tensor, 0.0
    scale = abs_max / q_max
    t_q = torch.clamp(torch.round(tensor / scale), -q_max, q_max)
    t_dq = t_q * scale
    mse = (tensor - t_dq).pow(2).mean().item()
    return t_dq, mse


# ── Mode 1: Layer Sensitivity Profiling ──────────────────────────────────────

def run_sensitivity_analysis(save_plots=False):
    """Profile which sub-layers of BERT are most sensitive to fake quantization.
    
    Strategy: For each linear layer, replace its weights with fake-quantized
    weights (one layer at a time), run a forward pass, and measure the output
    divergence from the FP32 baseline. This is the same perturbation analysis
    from Section 5.1 (Listing 5.2), but applied per-sublayer within a 
    transformer block.
    """
    print("=" * 70)
    print("LAYER SENSITIVITY PROFILING — BERT-base-uncased")
    print("=" * 70)
    print()

    from transformers import AutoModel, AutoTokenizer

    model_name = "bert-base-uncased"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Prepare diverse calibration inputs
    texts = [
        "The Federal Reserve raised interest rates by 25 basis points.",
        "Machine learning models require careful optimization for deployment.",
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "Quantization reduces model size while preserving accuracy.",
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    # Get FP32 baseline output
    with torch.no_grad():
        baseline = model(**inputs).last_hidden_state.clone()

    # Identify all linear layers and their roles within each transformer block
    layer_results = OrderedDict()

    for block_idx in range(model.config.num_hidden_layers):
        block = model.encoder.layer[block_idx]

        # Map of sub-layer name → module
        sublayers = OrderedDict([
            (f"L{block_idx}.attn.query",   block.attention.self.query),
            (f"L{block_idx}.attn.key",     block.attention.self.key),
            (f"L{block_idx}.attn.value",   block.attention.self.value),
            (f"L{block_idx}.attn.output",  block.attention.output.dense),
            (f"L{block_idx}.ffn.up",       block.intermediate.dense),
            (f"L{block_idx}.ffn.down",     block.output.dense),
        ])

        for name, module in sublayers.items():
            # Save original weights
            orig_weight = module.weight.data.clone()

            # Replace with fake-quantized weights (INT8, per-channel)
            fq_weight, weight_mse = fake_quantize_per_channel(
                orig_weight, bits=8, axis=0
            )
            module.weight.data.copy_(fq_weight)

            # Measure output divergence
            with torch.no_grad():
                perturbed = model(**inputs).last_hidden_state

            output_mse = (baseline - perturbed).pow(2).mean().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                baseline.reshape(-1).unsqueeze(0),
                perturbed.reshape(-1).unsqueeze(0),
            ).item()

            layer_results[name] = {
                "weight_mse": weight_mse,
                "output_mse": output_mse,
                "cos_sim": cos_sim,
                "shape": tuple(orig_weight.shape),
            }

            # Restore original weights
            module.weight.data.copy_(orig_weight)

    # Print results
    print(f"\n{'Layer':<25} {'Shape':<18} {'Weight MSE':>12} {'Output MSE':>12} {'Cos Sim':>10}")
    print("─" * 80)

    # Collect for plotting
    names = []
    output_mses = []
    categories = []  # 'attn' or 'ffn'

    for name, r in layer_results.items():
        print(f"{name:<25} {str(r['shape']):<18} {r['weight_mse']:>12.2e} "
              f"{r['output_mse']:>12.2e} {r['cos_sim']:>10.6f}")
        names.append(name)
        output_mses.append(r["output_mse"])
        categories.append("attn" if "attn" in name else "ffn")

    # Aggregate by sublayer type across all blocks
    print("\n\nAGGREGATE SENSITIVITY BY SUBLAYER TYPE")
    print("─" * 55)
    type_map = {}
    for name, r in layer_results.items():
        # Extract sublayer type: query, key, value, output, up, down
        sublayer_type = name.split(".")[-1]
        if sublayer_type not in type_map:
            type_map[sublayer_type] = []
        type_map[sublayer_type].append(r["output_mse"])

    type_names = []
    type_means = []
    type_stds = []
    for stype, mses in type_map.items():
        mean_mse = np.mean(mses)
        std_mse = np.std(mses)
        type_names.append(stype)
        type_means.append(mean_mse)
        type_stds.append(std_mse)
        print(f"  {stype:<12} mean output MSE: {mean_mse:.2e}  (std: {std_mse:.2e})")

    # Aggregate attention vs FFN
    print("\n\nATTENTION vs FFN AGGREGATE")
    print("─" * 55)
    attn_mses = [m for n, m in zip(names, output_mses) if "attn" in n]
    ffn_mses = [m for n, m in zip(names, output_mses) if "ffn" in n]
    print(f"  Attention layers:  mean output MSE = {np.mean(attn_mses):.2e}")
    print(f"  FFN layers:        mean output MSE = {np.mean(ffn_mses):.2e}")
    ratio = np.mean(attn_mses) / np.mean(ffn_mses)
    print(f"  Attention/FFN ratio: {ratio:.2f}×")

    # ── Figure: sensitivity by sublayer type ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors_list = list(MANNING_COLORS.values())

    # Left: per-block sensitivity, attention vs FFN
    block_count = model.config.num_hidden_layers
    block_indices = list(range(block_count))
    attn_per_block = []
    ffn_per_block = []
    for b in block_indices:
        attn_vals = [layer_results[n]["output_mse"]
                     for n in layer_results if n.startswith(f"L{b}.attn")]
        ffn_vals = [layer_results[n]["output_mse"]
                    for n in layer_results if n.startswith(f"L{b}.ffn")]
        attn_per_block.append(np.mean(attn_vals))
        ffn_per_block.append(np.mean(ffn_vals))

    x = np.arange(block_count)
    width = 0.35
    bars1 = axes[0].bar(x - width/2, attn_per_block, width,
                        label="Attention (Q/K/V/O)",
                        color=MANNING_COLORS["blue"], hatch=HATCHES[0],
                        edgecolor="white", linewidth=0.5)
    bars2 = axes[0].bar(x + width/2, ffn_per_block, width,
                        label="FFN (up/down)",
                        color=MANNING_COLORS["orange"], hatch=HATCHES[1],
                        edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Transformer Block")
    axes[0].set_ylabel("Output MSE (log scale)")
    axes[0].set_title("Per-block sensitivity: Attention vs FFN")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(i) for i in block_indices], fontsize=7)
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper left", framealpha=0.9)

    # Right: aggregate by sublayer type
    bar_colors = [MANNING_COLORS["blue"]] * 4 + [MANNING_COLORS["orange"]] * 2
    bar_hatches = [HATCHES[0]] * 4 + [HATCHES[1]] * 2
    bars = axes[1].bar(range(len(type_names)), type_means,
                       yerr=type_stds, capsize=3,
                       color=bar_colors, hatch=bar_hatches,
                       edgecolor="white", linewidth=0.5)
    axes[1].set_xticks(range(len(type_names)))
    axes[1].set_xticklabels(type_names, rotation=30, ha="right")
    axes[1].set_ylabel("Mean Output MSE (log scale)")
    axes[1].set_title("Sensitivity by sublayer type (across all blocks)")
    axes[1].set_yscale("log")

    plt.tight_layout()
    if save_plots:
        fig.savefig("fig_5_16_transformer_sensitivity.png")
        fig.savefig("fig_5_16_transformer_sensitivity.pdf")
        print("\nSaved: fig_5_16_transformer_sensitivity.png/pdf")
    plt.show()

    return layer_results


# ── Mode 2: Softmax Amplification ────────────────────────────────────────────

def run_softmax_amplification(save_plots=False):
    """Show how small quantization errors in Q/K projections get amplified
    by the softmax operation in attention.
    
    This is the mechanism that makes attention layers more sensitive than
    FFN layers during QAT—the same insight from Chapter 3 (Section 3.2,
    "The transformer attention sensitivity"), but now shown through the
    lens of how it affects gradient-based training.
    """
    print("=" * 70)
    print("SOFTMAX AMPLIFICATION OF QUANTIZATION ERRORS")
    print("=" * 70)
    print()

    from transformers import AutoModel, AutoTokenizer

    model_name = "bert-base-uncased"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    text = "The Federal Reserve raised interest rates by 25 basis points."
    inputs = tokenizer(text, return_tensors="pt")
    seq_len = inputs["input_ids"].shape[1]

    # Extract Q, K from layer 0
    block = model.encoder.layer[0]
    with torch.no_grad():
        # Get hidden states at layer 0 input
        embedding_output = model.embeddings(
            input_ids=inputs["input_ids"],
            token_type_ids=inputs.get("token_type_ids"),
        )

        # Compute Q, K projections
        q_proj = block.attention.self.query
        k_proj = block.attention.self.key

        Q_fp32 = q_proj(embedding_output)  # [1, seq_len, 768]
        K_fp32 = k_proj(embedding_output)

        # Reshape for attention: [batch, heads, seq_len, head_dim]
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads
        Q_heads = Q_fp32.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        K_heads = K_fp32.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

        # FP32 attention scores and weights
        scores_fp32 = torch.matmul(Q_heads, K_heads.transpose(-2, -1))
        scores_fp32 = scores_fp32 / (head_dim ** 0.5)
        weights_fp32 = torch.softmax(scores_fp32, dim=-1)

        # Now fake-quantize Q and K weights, recompute
        q_fq, q_mse = fake_quantize_per_channel(q_proj.weight.data, bits=8)
        k_fq, k_mse = fake_quantize_per_channel(k_proj.weight.data, bits=8)

        Q_int8 = torch.nn.functional.linear(embedding_output, q_fq, q_proj.bias)
        K_int8 = torch.nn.functional.linear(embedding_output, k_fq, k_proj.bias)

        Q_heads_q = Q_int8.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
        K_heads_q = K_int8.view(1, seq_len, num_heads, head_dim).transpose(1, 2)

        scores_int8 = torch.matmul(Q_heads_q, K_heads_q.transpose(-2, -1))
        scores_int8 = scores_int8 / (head_dim ** 0.5)
        weights_int8 = torch.softmax(scores_int8, dim=-1)

    # Measure amplification at each stage
    q_error = (Q_fp32 - Q_int8).pow(2).mean().item()
    score_error = (scores_fp32 - scores_int8).pow(2).mean().item()
    weight_error = (weights_fp32 - weights_int8).pow(2).mean().item()

    print(f"Q/K projection weight MSE:    Q={q_mse:.2e}, K={k_mse:.2e}")
    print(f"Q output MSE:                 {q_error:.2e}")
    print(f"Attention score MSE:          {score_error:.2e}")
    print(f"Attention weight MSE (post-softmax): {weight_error:.2e}")
    print()

    # Amplification ratios
    if q_error > 0:
        score_amp = score_error / q_error
        weight_amp = weight_error / q_error
        print(f"Score amplification over Q output error:  {score_amp:.1f}×")
        print(f"Weight amplification over Q output error: {weight_amp:.1f}×")

    # Per-head analysis
    print(f"\nPer-head attention weight MSE (head 0-{num_heads-1}):")
    head_mses = []
    for h in range(num_heads):
        h_mse = (weights_fp32[0, h] - weights_int8[0, h]).pow(2).mean().item()
        head_mses.append(h_mse)
    
    for h in range(num_heads):
        bar = "█" * int(head_mses[h] / max(head_mses) * 30)
        print(f"  Head {h:2d}: {head_mses[h]:.2e}  {bar}")

    # ── Figure: softmax amplification ──
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Left: error at each stage
    stages = ["Q/K\nProjection", "Attention\nScores", "Post-Softmax\nWeights"]
    errors = [q_error, score_error, weight_error]
    colors = [MANNING_COLORS["blue"], MANNING_COLORS["orange"], MANNING_COLORS["red"]]
    bars = axes[0].bar(stages, errors, color=colors,
                       hatch=[HATCHES[0], HATCHES[1], HATCHES[2]],
                       edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, errors):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f"{val:.1e}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Error amplification through attention")
    axes[0].set_yscale("log")

    # Center: FP32 vs INT8 attention pattern, head 0
    head_idx = 0
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # Truncate token labels for readability
    short_tokens = [t[:6] for t in tokens]

    im = axes[1].imshow(weights_fp32[0, head_idx].numpy(),
                        cmap="Blues", aspect="auto")
    axes[1].set_title(f"FP32 attention (head {head_idx})")
    axes[1].set_xticks(range(len(short_tokens)))
    axes[1].set_yticks(range(len(short_tokens)))
    axes[1].set_xticklabels(short_tokens, rotation=45, ha="right", fontsize=6)
    axes[1].set_yticklabels(short_tokens, fontsize=6)
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # Right: difference map
    diff = (weights_fp32[0, head_idx] - weights_int8[0, head_idx]).abs().numpy()
    im2 = axes[2].imshow(diff, cmap="Reds", aspect="auto")
    axes[2].set_title(f"|FP32 − INT8| attention (head {head_idx})")
    axes[2].set_xticks(range(len(short_tokens)))
    axes[2].set_yticks(range(len(short_tokens)))
    axes[2].set_xticklabels(short_tokens, rotation=45, ha="right", fontsize=6)
    axes[2].set_yticklabels(short_tokens, fontsize=6)
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    if save_plots:
        fig.savefig("fig_5_17_softmax_amplification.png")
        fig.savefig("fig_5_17_softmax_amplification.pdf")
        print("\nSaved: fig_5_17_softmax_amplification.png/pdf")
    plt.show()


# ── Mode 3: Selective QAT ────────────────────────────────────────────────────

class FakeQuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear that applies per-channel fake
    quantization to weights during forward pass (STE in backward).
    
    This is the transformer version of FQConv2d from Listing 5.23.
    """
    def __init__(self, original_linear, bits=8):
        super().__init__()
        self.linear = original_linear
        self.bits = bits
        self.q_max = (1 << (bits - 1)) - 1
        self.q_min = -self.q_max

    def forward(self, x):
        w = self.linear.weight
        # Per-channel fake quantization (axis=0 = output features)
        flat = w.detach().reshape(w.shape[0], -1)
        abs_max = flat.abs().amax(dim=1)
        scales = abs_max / self.q_max
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        scales_bc = scales.view(-1, 1)

        # Fake quantize with STE (straight-through via detach trick)
        w_q = torch.clamp(torch.round(w / scales_bc), self.q_min, self.q_max)
        w_dq = w_q * scales_bc
        # STE: use fake-quantized in forward, but pass gradients to original w
        w_fq = w + (w_dq - w).detach()

        return nn.functional.linear(x, w_fq, self.linear.bias)


def apply_selective_qat(model, strategy="ffn_only", bits=8):
    """Apply fake quantizers to selected linear layers in a BERT model.
    
    Strategies:
      - 'all':        Quantize every linear layer
      - 'ffn_only':   Quantize only FFN layers (intermediate.dense, output.dense)
      - 'attn_only':  Quantize only attention projections (Q, K, V, O)
      - 'none':       No quantization (FP32 baseline)
    
    Returns: count of quantized layers
    """
    count = 0
    for block_idx in range(model.config.num_hidden_layers):
        block = model.encoder.layer[block_idx]
        attn = block.attention
        
        if strategy in ("all", "attn_only"):
            attn.self.query = FakeQuantizedLinear(attn.self.query, bits)
            attn.self.key = FakeQuantizedLinear(attn.self.key, bits)
            attn.self.value = FakeQuantizedLinear(attn.self.value, bits)
            attn.output.dense = FakeQuantizedLinear(attn.output.dense, bits)
            count += 4

        if strategy in ("all", "ffn_only"):
            block.intermediate.dense = FakeQuantizedLinear(
                block.intermediate.dense, bits
            )
            block.output.dense = FakeQuantizedLinear(block.output.dense, bits)
            count += 2

    return count


def run_selective_qat(save_plots=False):
    """Compare selective QAT strategies on BERT-base-uncased.
    
    We measure output divergence (not fine-tuning accuracy) to show which
    strategy introduces the least distortion — the same validation approach
    from Section 5.1's perturbation analysis, but applied to QAT strategies.
    
    For a real fine-tuning experiment, you would use a downstream task like
    SST-2. Here we measure the FP32-vs-QAT output gap, which predicts
    how much fine-tuning work the optimizer must do.
    """
    print("=" * 70)
    print("SELECTIVE QAT STRATEGIES — BERT-base-uncased")
    print("=" * 70)
    print()

    from transformers import AutoModel, AutoTokenizer

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Diverse eval inputs
    texts = [
        "The Federal Reserve raised interest rates by 25 basis points.",
        "Machine learning models require careful optimization for deployment.",
        "The quick brown fox jumps over the lazy dog near the riverbank.",
        "Quantization reduces model size while preserving accuracy.",
        "Climate change poses significant challenges to global agriculture.",
        "The new smartphone features an advanced neural processing unit.",
        "Investors are closely watching the bond market for recession signals.",
        "Open source software has transformed the technology industry.",
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    strategies = ["none", "ffn_only", "attn_only", "all"]
    bits_list = [8, 4]  # Test both INT8 and INT4
    
    results = {}
    
    # Get FP32 baseline once
    print(f"Loading {model_name} for FP32 baseline...")
    model_fp32 = AutoModel.from_pretrained(model_name)
    model_fp32.eval()
    with torch.no_grad():
        baseline = model_fp32.last_hidden_state if hasattr(model_fp32, 'last_hidden_state') else None
        baseline = model_fp32(**inputs).last_hidden_state.clone()
    del model_fp32

    for bits in bits_list:
        print(f"\n{'─'*50}")
        print(f"INT{bits} Fake Quantization")
        print(f"{'─'*50}")
        
        for strategy in strategies:
            if strategy == "none" and bits != bits_list[0]:
                # FP32 baseline only needed once
                results[f"INT{bits}_{strategy}"] = results[f"INT{bits_list[0]}_none"]
                continue

            print(f"\n  Strategy: {strategy} (INT{bits})")
            model = AutoModel.from_pretrained(model_name)
            model.eval()

            if strategy != "none":
                n_quantized = apply_selective_qat(model, strategy=strategy, bits=bits)
                print(f"    Quantized layers: {n_quantized}")
            else:
                n_quantized = 0
                print(f"    Quantized layers: 0 (FP32 baseline)")

            with torch.no_grad():
                output = model(**inputs).last_hidden_state

            mse = (baseline - output).pow(2).mean().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                baseline.reshape(-1).unsqueeze(0),
                output.reshape(-1).unsqueeze(0),
            ).item()
            max_diff = (baseline - output).abs().max().item()

            results[f"INT{bits}_{strategy}"] = {
                "mse": mse,
                "cos_sim": cos_sim,
                "max_diff": max_diff,
                "n_quantized": n_quantized,
            }

            print(f"    Output MSE:      {mse:.2e}")
            print(f"    Cosine sim:      {cos_sim:.8f}")
            print(f"    Max |diff|:      {max_diff:.4f}")

            del model

    # Summary table
    print("\n\n" + "=" * 70)
    print("SUMMARY: Selective QAT Strategy Comparison")
    print("=" * 70)
    print(f"\n{'Strategy':<20} {'Bits':>4} {'Layers':>7} {'Output MSE':>12} {'Cos Sim':>12} {'Max |Δ|':>10}")
    print("─" * 68)
    for key in sorted(results.keys()):
        r = results[key]
        parts = key.split("_", 1)
        bits_str = parts[0]
        strat = parts[1]
        print(f"{strat:<20} {bits_str:>4} {r['n_quantized']:>7} "
              f"{r['mse']:>12.2e} {r['cos_sim']:>12.8f} {r['max_diff']:>10.4f}")

    # ── Figure: strategy comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax_idx, bits in enumerate(bits_list):
        ax = axes[ax_idx]
        strat_names = []
        mses = []
        colors = []
        hatches_list = []
        color_map = {
            "none": MANNING_COLORS["gray"],
            "ffn_only": MANNING_COLORS["green"],
            "attn_only": MANNING_COLORS["blue"],
            "all": MANNING_COLORS["orange"],
        }
        hatch_map = {
            "none": "",
            "ffn_only": HATCHES[0],
            "attn_only": HATCHES[1],
            "all": HATCHES[2],
        }

        for strategy in strategies:
            key = f"INT{bits}_{strategy}"
            r = results[key]
            label = strategy.replace("_", " ")
            strat_names.append(f"{label}\n({r['n_quantized']} layers)")
            mses.append(r["mse"])
            colors.append(color_map[strategy])
            hatches_list.append(hatch_map[strategy])

        bars = ax.bar(range(len(strat_names)), mses,
                      color=colors, hatch=hatches_list,
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, mses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1e}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(strat_names)))
        ax.set_xticklabels(strat_names, fontsize=7)
        ax.set_ylabel("Output MSE vs FP32")
        ax.set_title(f"INT{bits}: Selective QAT strategies")
        ax.set_yscale("log")

    plt.tight_layout()
    if save_plots:
        fig.savefig("fig_5_18_selective_qat.png")
        fig.savefig("fig_5_18_selective_qat.pdf")
        print("\nSaved: fig_5_18_selective_qat.png/pdf")
    plt.show()

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chapter 5.5: Adapt Transformers Efficiently"
    )
    parser.add_argument(
        "--mode",
        choices=["sensitivity", "softmax-amp", "selective-qat", "all"],
        default="all",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save figures as PNG and SVG",
    )
    args = parser.parse_args()

    if args.mode in ("sensitivity", "all"):
        run_sensitivity_analysis(save_plots=args.save_plots)

    if args.mode in ("softmax-amp", "all"):
        run_softmax_amplification(save_plots=args.save_plots)

    if args.mode in ("selective-qat", "all"):
        run_selective_qat(save_plots=args.save_plots)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()