"""
Chapter 7.1 – Handle outliers and key-value cache considerations
================================================================
Companion script for "Quantizing Large Language Models in Practice."

This script profiles the activation outlier phenomenon that makes LLM
quantization fundamentally different from quantizing smaller models:
  - Channel magnitude variation that grows with model scale
  - Cross-layer consistency: the SAME hidden dimensions spike everywhere
  - Why naive INT8 quantization fails even with moderate outliers
  - KV cache memory pressure and quantization trade-offs

Models: facebook/opt-125m (baseline), facebook/opt-6.7b (outlier emergence)
Optional: facebook/opt-1.3b (intermediate)

Calibration data: WikiText-2 (standard LLM evaluation corpus)

Hardware requirements:
  - opt-125m: any CPU (laptop-friendly)
  - opt-1.3b: any CPU or modest GPU
  - opt-6.7b: Colab T4 GPU (16GB VRAM) or equivalent

Usage:
  # Profile outlier channel patterns (default: opt-6.7b on GPU)
  python ch7_outlier_profiling.py --mode profile --model opt-6.7b

  # Compare emergence across scales (runs models sequentially)
  python ch7_outlier_profiling.py --mode emergence

  # Quantization error decomposition
  python ch7_outlier_profiling.py --mode quant-error --model opt-6.7b

  # KV cache memory analysis (analytical, no GPU)
  python ch7_outlier_profiling.py --mode kv-cache

  # Generate all figures and results (Colab T4)
  python ch7_outlier_profiling.py --mode all --save-plots

  # Laptop-friendly subset (CPU only)
  python ch7_outlier_profiling.py --mode all --save-plots --model opt-125m \
      --emergence-models opt-125m opt-1.3b
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_DIR = SCRIPT_DIR / "figures"

# Manning color palette (grayscale-safe)
MANNING_COLORS = {
    "blue_l2": "#6BA7D5",
    "red_l3":  "#D3151E",
    "green_l3": "#80C232",
    "orange_l2": "#FFB454",
    "brown_l2": "#D1A66C",
    "purple_l1": "#E8E6F3",
    "black_l4": "#000000",
    "gray_50":  "#808080",
    "gray_25":  "#C0C0C0",
}

HATCHES = ["", "//", "\\\\", "xx", "..", "||", "--", "++"]

# OPT model configs
MODEL_CONFIGS = {
    "opt-125m": {
        "name": "facebook/opt-125m",
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "head_dim": 64,
        "ffn_dim": 3072,
        "params_b": 0.125,
    },
    "opt-1.3b": {
        "name": "facebook/opt-1.3b",
        "hidden_size": 2048,
        "num_layers": 24,
        "num_heads": 32,
        "head_dim": 64,
        "ffn_dim": 8192,
        "params_b": 1.3,
    },
    "opt-6.7b": {
        "name": "facebook/opt-6.7b",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
        "ffn_dim": 16384,
        "params_b": 6.7,
    },
}


# ---------------------------------------------------------------------------
# Plotting utilities (Manning-compliant)
# ---------------------------------------------------------------------------
def setup_matplotlib():
    """Configure matplotlib for Manning publication standards."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def save_figure(fig, name, save_plots=True):
    """Save figure in both PNG (300 DPI) and PDF (vector) formats."""
    if not save_plots:
        return
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{name}.png"
    pdf_path = FIGURE_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


def save_results(data, name):
    """Save results as JSON for prose grounding."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Results: {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_calibration_data(tokenizer, num_samples=32, seq_length=512):
    """Load WikiText-2 calibration sequences."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    tokens = tokenizer(full_text, return_tensors="pt")["input_ids"][0]

    blocks = []
    for i in range(0, len(tokens) - seq_length, seq_length):
        blocks.append(tokens[i : i + seq_length].unsqueeze(0))
        if len(blocks) >= num_samples:
            break

    print(f"  Loaded {len(blocks)} calibration blocks of {seq_length} tokens")
    return blocks


# ---------------------------------------------------------------------------
# Model loading with memory awareness
# ---------------------------------------------------------------------------
def load_model(model_key, device=None):
    """
    Load an OPT model with device-appropriate settings.

    For opt-6.7b on T4: FP16 on GPU (~13.4GB of 15.8GB VRAM).
    For smaller models: FP16 on GPU if available, else FP32 on CPU.
    """
    config = MODEL_CONFIGS[model_key]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Loading {model_key} on {device}...")

    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            config["name"], torch_dtype=torch.float16
        ).to(device).eval()
    else:
        # CPU: use FP16 for large models to reduce memory
        if config["params_b"] >= 1.0:
            model = AutoModelForCausalLM.from_pretrained(
                config["name"], torch_dtype=torch.float16
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                config["name"]
            ).eval()

    tokenizer = AutoTokenizer.from_pretrained(config["name"])

    # Report memory
    if device.type == "cuda":
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU memory: {mem_gb:.1f} / {total_gb:.1f} GB "
              f"({mem_gb/total_gb*100:.0f}%)")
    else:
        param_gb = sum(p.numel() * p.element_size()
                       for p in model.parameters()) / 1024**3
        print(f"  Model size: {param_gb:.1f} GB")

    return model, tokenizer, device


def unload_model(model):
    """Free model memory."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc; gc.collect()


# ---------------------------------------------------------------------------
# Experiment 1: Profile activation channel patterns
# ---------------------------------------------------------------------------
def profile_outliers(model_key="opt-6.7b", num_samples=16, seq_length=256):
    """
    Hook into every fc1 (FFN up-projection) and self_attn.out_proj,
    capture per-channel activation magnitudes, and characterize the
    distribution shape -- not just binary outlier detection.

    Key metrics (adaptive, not threshold-dependent):
    - max/median ratio: severity of worst channel vs. typical
    - p99/p50 ratio: spread of the top percentile
    - coefficient of variation: overall channel non-uniformity
    - top-K channel indices and cross-layer consistency
    """
    config = MODEL_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"Experiment 1: Profiling activation channel patterns ({model_key})")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_key)
    calibration_blocks = load_calibration_data(
        tokenizer, num_samples=num_samples, seq_length=seq_length
    )

    # ---- Hook setup ----
    # We hook into the INPUT of each Linear layer.  The activation
    # ENTERING fc1 is the tensor that must be quantized for W*x, and
    # its outlier channels determine quantization error.
    layer_channel_maxes = defaultdict(list)                      #A
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()                        #B
            # Flatten all dims except the last (channel) dim,
            # then take per-channel max.  Handles both 3D
            # [batch, seq, hidden] and 2D [batch*seq, hidden].
            flat = x.reshape(-1, x.shape[-1])                    #C
            channel_max = flat.abs().amax(dim=0)                 #D
            layer_channel_maxes[name].append(channel_max.cpu())
        return hook_fn

    #A Accumulate per-channel stats across calibration samples
    #B Cast to float32 for stable statistics
    #C Flatten to [elements, channels] regardless of input ndim
    #D Per-channel max across all elements

    for layer_idx in range(config["num_layers"]):
        layer = model.model.decoder.layers[layer_idx]
        h1 = layer.fc1.register_forward_hook(
            make_hook(f"layer_{layer_idx:02d}_fc1")
        )
        h2 = layer.self_attn.out_proj.register_forward_hook(
            make_hook(f"layer_{layer_idx:02d}_attn_out")
        )
        hooks.extend([h1, h2])

    # ---- Run calibration ----
    print("  Running calibration passes...")
    with torch.no_grad():
        for i, block in enumerate(calibration_blocks):
            model(block.to(device))
            if (i + 1) % 4 == 0:
                print(f"    Processed {i+1}/{len(calibration_blocks)} blocks")

    for h in hooks:
        h.remove()

    # ---- Aggregate and analyze ----
    results = {}
    for name in sorted(layer_channel_maxes.keys()):
        maxes = layer_channel_maxes[name]
        stacked = torch.stack(maxes)           # [num_samples, hidden_dim]
        mean_max = stacked.mean(dim=0)         # avg max per channel

        # Distribution shape metrics (not threshold-dependent)
        sorted_vals, sorted_idx = mean_max.sort(descending=True)
        p50 = mean_max.median().item()
        p90 = torch.quantile(mean_max, 0.90).item()
        p99 = torch.quantile(mean_max, 0.99).item()
        p999 = torch.quantile(mean_max, 0.999).item()
        chan_max = mean_max.max().item()
        chan_min = mean_max.min().item()
        chan_mean = mean_max.mean().item()
        chan_std = mean_max.std().item() if mean_max.numel() > 1 else 0.0

        # Key ratios
        max_to_median = chan_max / p50 if p50 > 0 else 0
        p99_to_p50 = p99 / p50 if p50 > 0 else 0
        cv = chan_std / chan_mean if chan_mean > 0 else 0              #A

        # Top-K outlier channel indices (top 1% by magnitude)
        hidden_dim = mean_max.shape[0]
        top_k = max(1, int(hidden_dim * 0.01))
        top_k_indices = sorted_idx[:top_k].tolist()
        top_k_magnitudes = sorted_vals[:top_k].tolist()

        # Outlier count at various thresholds
        outlier_counts = {}
        for thresh_mult in [3.0, 6.0, 10.0, 20.0, 50.0, 100.0]:
            threshold = thresh_mult * p50
            count = int((mean_max > threshold).sum().item())
            outlier_counts[f">{thresh_mult:.0f}x_median"] = count

        #A Coefficient of variation: std/mean across channels

        results[name] = {
            "hidden_dim": hidden_dim,
            "percentiles": {
                "p50": round(p50, 2),
                "p90": round(p90, 2),
                "p99": round(p99, 2),
                "p999": round(p999, 2),
                "max": round(chan_max, 2),
                "min": round(chan_min, 2),
            },
            "ratios": {
                "max_to_median": round(max_to_median, 2),
                "p99_to_p50": round(p99_to_p50, 2),
                "p999_to_p50": round(p999 / p50, 2) if p50 > 0 else 0,
                "coeff_of_variation": round(cv, 4),
            },
            "outlier_counts": outlier_counts,
            "top_1pct_indices": top_k_indices,
            "top_1pct_magnitudes": [round(m, 2) for m in top_k_magnitudes],
            "mean_channel_max": mean_max.tolist(),
        }

    # ---- Cross-layer consistency analysis ----
    # The key insight from Dettmers et al.: outlier channels are not
    # random.  The SAME hidden dimensions spike across ALL layers.
    # We measure this by checking overlap of top-1% channels.
    fc1_keys = sorted([k for k in results if "fc1" in k])
    top_sets = []
    for k in fc1_keys:
        top_sets.append(set(results[k]["top_1pct_indices"]))

    if len(top_sets) >= 2:
        universal = set.intersection(*top_sets)
        dim_counts = Counter()
        for s in top_sets:
            dim_counts.update(s)
        majority = {d for d, c in dim_counts.items()
                    if c > len(top_sets) / 2}

        # Jaccard similarity between consecutive layers
        jaccards = []
        for i in range(len(top_sets) - 1):
            inter = len(top_sets[i] & top_sets[i+1])
            union = len(top_sets[i] | top_sets[i+1])
            jaccards.append(inter / union if union > 0 else 0)
        avg_jaccard = sum(jaccards) / len(jaccards) if jaccards else 0
    else:
        universal = set()
        majority = set()
        avg_jaccard = 0
        jaccards = []

    consistency = {
        "num_layers_compared": len(top_sets),
        "universal_top1pct_dims": sorted(universal),
        "num_universal": len(universal),
        "majority_top1pct_dims": sorted(majority),
        "num_majority": len(majority),
        "avg_jaccard_consecutive": round(avg_jaccard, 4),
        "jaccard_per_pair": [round(j, 4) for j in jaccards],
    }
    results["_consistency"] = consistency

    # ---- Print summary ----
    print(f"\n  Channel magnitude distribution ({model_key}):")
    print(f"  {'Layer':<22} {'p50':>8} {'p99':>8} {'max':>8} "
          f"{'max/p50':>9} {'p99/p50':>9} {'CV':>8} "
          f"{'#>6x':>6} {'#>10x':>6}")
    print(f"  {'-'*88}")
    for k in fc1_keys:
        r = results[k]
        p = r["percentiles"]
        rat = r["ratios"]
        oc = r["outlier_counts"]
        print(f"  {k:<22} {p['p50']:>8.1f} {p['p99']:>8.1f} "
              f"{p['max']:>8.1f} {rat['max_to_median']:>9.1f}x "
              f"{rat['p99_to_p50']:>8.1f}x {rat['coeff_of_variation']:>8.3f} "
              f"{oc.get('>6x_median', 0):>6} {oc.get('>10x_median', 0):>6}")

    print(f"\n  Cross-layer top-1% channel consistency:")
    print(f"    Universal (in ALL layers): {consistency['num_universal']} dims"
          f" -> {sorted(universal)[:20]}"
          f"{'...' if len(universal) > 20 else ''}")
    print(f"    Majority (>50% of layers): {consistency['num_majority']} dims")
    print(f"    Avg Jaccard (consecutive):  "
          f"{consistency['avg_jaccard_consecutive']:.4f}")

    save_results(results, f"outlier_profile_{model_key.replace('-', '_')}")
    unload_model(model)
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Outlier emergence across model scales
# ---------------------------------------------------------------------------
def outlier_emergence(models_to_compare=None, num_samples=16, seq_length=256):
    """
    Compare channel magnitude distribution across model scales.

    Expected pattern (Dettmers et al.):
      - OPT-125M: low max/median ratio, low CV, random top channels
      - OPT-6.7B: high max/median ratio, high CV, strong consistency

    Each model is loaded, profiled, and unloaded sequentially to fit
    in T4 VRAM.
    """
    if models_to_compare is None:
        models_to_compare = ["opt-125m", "opt-6.7b"]

    print(f"\n{'='*60}")
    print(f"Experiment 2: Outlier emergence across scales")
    print(f"  Models: {', '.join(models_to_compare)}")
    print(f"{'='*60}")

    emergence_results = {}
    for model_key in models_to_compare:
        profile = profile_outliers(
            model_key=model_key,
            num_samples=num_samples,
            seq_length=seq_length,
        )

        fc1_keys = sorted([k for k in profile if "fc1" in k])
        if not fc1_keys:
            continue

        all_max_to_median = []
        all_p99_to_p50 = []
        all_cv = []
        per_layer_detail = {}

        for k in fc1_keys:
            r = profile[k]
            rat = r["ratios"]
            oc = r["outlier_counts"]
            all_max_to_median.append(rat["max_to_median"])
            all_p99_to_p50.append(rat["p99_to_p50"])
            all_cv.append(rat["coeff_of_variation"])

            per_layer_detail[k] = {
                "max_to_median": rat["max_to_median"],
                "p99_to_p50": rat["p99_to_p50"],
                "cv": rat["coeff_of_variation"],
                "num_6x": oc.get(">6x_median", 0),
                "num_10x": oc.get(">10x_median", 0),
                "num_50x": oc.get(">50x_median", 0),
                "num_100x": oc.get(">100x_median", 0),
                "top_1pct_indices": r["top_1pct_indices"],
                "percentiles": r["percentiles"],
            }

        emergence_results[model_key] = {
            "params_b": MODEL_CONFIGS[model_key]["params_b"],
            "hidden_dim": MODEL_CONFIGS[model_key]["hidden_size"],
            "num_layers": len(fc1_keys),
            "avg_max_to_median": round(float(np.mean(all_max_to_median)), 2),
            "max_max_to_median": round(float(np.max(all_max_to_median)), 2),
            "avg_p99_to_p50": round(float(np.mean(all_p99_to_p50)), 2),
            "avg_cv": round(float(np.mean(all_cv)), 4),
            "consistency": profile.get("_consistency", {}),
            "per_layer": per_layer_detail,
        }

    # ---- Print comparison table ----
    print(f"\n{'='*60}")
    print(f"  Emergence comparison:")
    print(f"  {'Model':<12} {'Params':>8} {'Avg max/p50':>12} "
          f"{'Max max/p50':>12} {'Avg p99/p50':>12} {'Avg CV':>10} "
          f"{'Jaccard':>10}")
    print(f"  {'-'*76}")
    for model_key, stats in emergence_results.items():
        jaccard = stats["consistency"].get("avg_jaccard_consecutive", 0)
        print(f"  {model_key:<12} {stats['params_b']:>7.1f}B "
              f"{stats['avg_max_to_median']:>12.1f}x "
              f"{stats['max_max_to_median']:>12.1f}x "
              f"{stats['avg_p99_to_p50']:>11.1f}x "
              f"{stats['avg_cv']:>10.4f} "
              f"{jaccard:>10.4f}")

    save_results(emergence_results, "outlier_emergence")
    return emergence_results


# ---------------------------------------------------------------------------
# Experiment 3: Quantization error decomposition (memory-efficient)
# ---------------------------------------------------------------------------
def quantization_error_decomposition(
    model_key="opt-6.7b", num_samples=16, seq_length=256
):
    """
    Apply naive INT8 quantization and decompose MSE into contributions
    from high-magnitude vs. normal channels.

    Memory-efficient: computes quantization error INSIDE the hook,
    accumulating per-channel MSE without storing full activation tensors.
    Critical for OPT-6.7b on T4.
    """
    config = MODEL_CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"Experiment 3: Quantization error decomposition ({model_key})")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_key)
    calibration_blocks = load_calibration_data(
        tokenizer, num_samples=num_samples, seq_length=seq_length
    )

    # ---- Phase 1: Collect per-channel magnitude profiles ----
    print("  Phase 1: Collecting channel magnitude profiles...")
    channel_maxes = defaultdict(list)
    hooks = []

    def make_profile_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            flat = x.reshape(-1, x.shape[-1])
            cm = flat.abs().amax(dim=0)
            channel_maxes[name].append(cm.cpu())
        return hook_fn

    for layer_idx in range(config["num_layers"]):
        layer = model.model.decoder.layers[layer_idx]
        h = layer.fc1.register_forward_hook(
            make_profile_hook(f"layer_{layer_idx:02d}")
        )
        hooks.append(h)

    with torch.no_grad():
        for block in calibration_blocks:
            model(block.to(device))

    for h in hooks:
        h.remove()

    # Identify top-1% channels per layer
    channel_profiles = {}
    for name in sorted(channel_maxes.keys()):
        stacked = torch.stack(channel_maxes[name])
        mean_max = stacked.mean(dim=0)
        sorted_vals, sorted_idx = mean_max.sort(descending=True)
        top_k = max(1, int(mean_max.shape[0] * 0.01))
        top_indices = set(sorted_idx[:top_k].tolist())
        channel_profiles[name] = {
            "mean_max": mean_max,
            "top_indices": top_indices,
            "top_k": top_k,
        }

    del channel_maxes

    # ---- Phase 2: Streaming error decomposition ----
    # Compute quantization error INSIDE the hook, accumulating
    # per-channel MSE without storing full activation tensors.
    print("  Phase 2: Computing error decomposition (in-hook)...")

    per_tensor_sse = defaultdict(lambda: torch.zeros(
        config["hidden_size"], dtype=torch.float64))
    per_channel_sse = defaultdict(lambda: torch.zeros(
        config["hidden_size"], dtype=torch.float64))
    element_counts = defaultdict(int)
    hooks = []

    def make_error_hook(name):
        def hook_fn(module, input, output):
            x = input[0].detach().float()                        #A
            flat = x.reshape(-1, x.shape[-1])                    #B
            n_elements = flat.shape[0]

            # Per-tensor INT8 quantization
            abs_max = flat.abs().max()
            scale_pt = torch.clamp(abs_max / 127.0, min=1e-8)   #C
            q_pt = torch.round(flat / scale_pt).clamp(-128, 127)
            deq_pt = q_pt * scale_pt
            sq_err_pt = ((flat - deq_pt) ** 2).sum(dim=0)       #D

            # Per-channel INT8 quantization
            ch_abs_max = flat.abs().amax(dim=0)
            scale_pc = torch.clamp(ch_abs_max / 127.0, min=1e-8)
            q_pc = torch.round(flat / scale_pc).clamp(-128, 127)
            deq_pc = q_pc * scale_pc
            sq_err_pc = ((flat - deq_pc) ** 2).sum(dim=0)

            per_tensor_sse[name] += sq_err_pt.cpu().double()
            per_channel_sse[name] += sq_err_pc.cpu().double()
            element_counts[name] += n_elements

        return hook_fn

    #A Cast to float32 for stable quantization arithmetic
    #B Flatten batch*seq into elements dim, keep channel dim
    #C Single scale from global max -- outlier channels set this
    #D Sum squared error per channel across all elements

    for layer_idx in range(config["num_layers"]):
        layer = model.model.decoder.layers[layer_idx]
        h = layer.fc1.register_forward_hook(
            make_error_hook(f"layer_{layer_idx:02d}")
        )
        hooks.append(h)

    with torch.no_grad():
        for i, block in enumerate(calibration_blocks):
            model(block.to(device))
            if (i + 1) % 4 == 0:
                print(f"    Processed {i+1}/{len(calibration_blocks)} blocks")

    for h in hooks:
        h.remove()

    # ---- Compute final MSE and decomposition ----
    results = {}
    for name in sorted(per_tensor_sse.keys()):
        n = element_counts[name]
        if n == 0:
            continue

        mse_pt = (per_tensor_sse[name] / n).float()
        mse_pc = (per_channel_sse[name] / n).float()

        profile = channel_profiles[name]
        hidden_dim = mse_pt.shape[0]
        top_indices = profile["top_indices"]

        top_mask = torch.zeros(hidden_dim, dtype=torch.bool)
        for idx in top_indices:
            top_mask[idx] = True
        normal_mask = ~top_mask

        total_mse_pt = mse_pt.sum().item()
        top_mse_pt = mse_pt[top_mask].sum().item()
        normal_mse_pt = mse_pt[normal_mask].sum().item()

        total_mse_pc = mse_pc.sum().item()
        top_mse_pc = mse_pc[top_mask].sum().item()
        normal_mse_pc = mse_pc[normal_mask].sum().item()

        pct_top_dims = len(top_indices) / hidden_dim * 100
        pct_top_mse_pt = (top_mse_pt / total_mse_pt * 100
                          if total_mse_pt > 0 else 0)
        pct_top_mse_pc = (top_mse_pc / total_mse_pc * 100
                          if total_mse_pc > 0 else 0)

        reduction = (total_mse_pt / total_mse_pc
                     if total_mse_pc > 0 else float("inf"))

        results[name] = {
            "hidden_dim": hidden_dim,
            "num_top_channels": len(top_indices),
            "pct_top_channels": round(pct_top_dims, 2),
            "per_tensor": {
                "total_mse": total_mse_pt,
                "top_channel_mse": top_mse_pt,
                "normal_channel_mse": normal_mse_pt,
                "pct_top_channel_mse": round(pct_top_mse_pt, 1),
            },
            "per_channel": {
                "total_mse": total_mse_pc,
                "top_channel_mse": top_mse_pc,
                "normal_channel_mse": normal_mse_pc,
                "pct_top_channel_mse": round(pct_top_mse_pc, 1),
            },
            "mse_reduction_ratio": round(reduction, 1),
        }

    # ---- Print summary ----
    print(f"\n  Error decomposition (top-1% channels vs rest):")
    print(f"  {'Layer':<15} {'Top dims':>10} {'% of dims':>10} "
          f"{'% MSE (PT)':>11} {'% MSE (PC)':>11} {'PT/PC':>8}")
    print(f"  {'-'*65}")
    for name, stats in sorted(results.items()):
        print(f"  {name:<15} {stats['num_top_channels']:>10} "
              f"{stats['pct_top_channels']:>9.1f}% "
              f"{stats['per_tensor']['pct_top_channel_mse']:>10.1f}% "
              f"{stats['per_channel']['pct_top_channel_mse']:>10.1f}% "
              f"{stats['mse_reduction_ratio']:>8.1f}x")

    save_results(
        results,
        f"quant_error_decomposition_{model_key.replace('-', '_')}"
    )
    unload_model(model)
    return results


# ---------------------------------------------------------------------------
# Experiment 4: KV cache memory analysis
# ---------------------------------------------------------------------------
def kv_cache_memory_analysis():
    """
    Analytical computation of KV cache memory at various model scales,
    sequence lengths, batch sizes, and quantization levels.

    GQA-aware: uses kv_heads (not num_heads) for models that use
    grouped-query attention.
    """
    print(f"\n{'='*60}")
    print("Experiment 4: KV cache memory analysis")
    print(f"{'='*60}")

    models = {
        "Llama-3.1-8B":  {"layers": 32, "heads": 32, "kv_heads": 8,
                          "head_dim": 128, "params_gb": 16.0},
        "Llama-3.1-70B": {"layers": 80, "heads": 64, "kv_heads": 8,
                          "head_dim": 128, "params_gb": 140.0},
        "Mistral-7B":    {"layers": 32, "heads": 32, "kv_heads": 8,
                          "head_dim": 128, "params_gb": 14.0},
        "Qwen2.5-72B":   {"layers": 80, "heads": 64, "kv_heads": 8,
                          "head_dim": 128, "params_gb": 144.0},
    }

    precisions = {"FP16": 2, "INT8": 1, "INT4": 0.5}
    seq_lengths = [2048, 8192, 32768, 131072]

    def kv_cache_gb(num_layers, kv_heads, head_dim, seq_len,
                    batch_size, bytes_per_elem):
        """KV cache size in GB.  GQA-aware: uses kv_heads, not heads."""
        total_bytes = (2 * num_layers * kv_heads * head_dim
                       * seq_len * batch_size * bytes_per_elem)
        return total_bytes / (1024 ** 3)

    results = {}
    for model_name, cfg in models.items():
        model_results = {}
        for seq_len in seq_lengths:
            for batch_size in [1, 8, 32]:
                key = f"seq{seq_len}_batch{batch_size}"
                row = {}
                for prec_name, bpe in precisions.items():
                    size_gb = kv_cache_gb(
                        cfg["layers"], cfg["kv_heads"], cfg["head_dim"],
                        seq_len, batch_size, bpe)
                    row[prec_name] = round(size_gb, 2)
                row["savings_int4_vs_fp16"] = round(
                    row["FP16"] - row["INT4"], 2)
                row["pct_of_model_fp16"] = round(
                    row["FP16"] / cfg["params_gb"] * 100, 1)
                model_results[key] = row
        results[model_name] = model_results

    # ---- Print key scenarios ----
    print(f"\n  KV cache size (GB) at FP16 -- Batch=1:")
    print(f"  {'Model':<18} {'2K ctx':>8} {'8K ctx':>8} "
          f"{'32K ctx':>8} {'128K ctx':>9}")
    print(f"  {'-'*52}")
    for model_name in models:
        row = results[model_name]
        vals = [row[f"seq{s}_batch1"]["FP16"] for s in seq_lengths]
        print(f"  {model_name:<18} {vals[0]:>8.2f} {vals[1]:>8.2f} "
              f"{vals[2]:>8.2f} {vals[3]:>9.2f}")

    print(f"\n  Savings from INT4 KV cache -- Batch=8:")
    print(f"  {'Model':<18} {'FP16 KV':>9} {'INT4 KV':>9} "
          f"{'Saved':>8} {'% of model':>11}")
    print(f"  {'-'*55}")
    for model_name, cfg in models.items():
        for seq_len in [8192, 32768]:
            row = results[model_name][f"seq{seq_len}_batch8"]
            print(f"  {model_name:<18} {row['FP16']:>8.1f}G "
                  f"{row['INT4']:>8.1f}G "
                  f"{row['savings_int4_vs_fp16']:>7.1f}G "
                  f"{row['pct_of_model_fp16']:>10.1f}%")

    # ---- Concurrent users on 80GB A100 ----
    gpu_vram = 80
    print(f"\n  Concurrent users on {gpu_vram}GB A100 "
          f"(Llama-3.1-8B, INT4 weights):")
    model_cfg = models["Llama-3.1-8B"]
    model_size_int4 = model_cfg["params_gb"] / 4
    available_vram = gpu_vram - model_size_int4 - 2
    print(f"  Model (INT4): ~{model_size_int4:.0f}GB | "
          f"Available for KV: ~{available_vram:.0f}GB")

    user_table = {}
    for seq_len in seq_lengths:
        for prec_name, bpe in precisions.items():
            per_user = kv_cache_gb(
                model_cfg["layers"], model_cfg["kv_heads"],
                model_cfg["head_dim"], seq_len, 1, bpe)
            max_users = (int(available_vram / per_user)
                         if per_user > 0 else 0)
            user_table[f"seq{seq_len}_{prec_name}"] = {
                "per_user_gb": round(per_user, 3),
                "max_users": max_users,
            }

    print(f"  {'Seq len':<10} {'FP16/user':>10} {'Users':>6}  |  "
          f"{'INT4/user':>10} {'Users':>6}  |  {'Multiplier':>11}")
    print(f"  {'-'*65}")
    for seq_len in seq_lengths:
        fp16 = user_table[f"seq{seq_len}_FP16"]
        int4 = user_table[f"seq{seq_len}_INT4"]
        mult = (int4["max_users"] / fp16["max_users"]
                if fp16["max_users"] > 0 else 0)
        print(f"  {seq_len:<10} {fp16['per_user_gb']:>9.3f}G "
              f"{fp16['max_users']:>6}  |  "
              f"{int4['per_user_gb']:>9.3f}G {int4['max_users']:>6}  |  "
              f"{mult:>10.1f}x")

    results["_concurrent_users"] = user_table
    results["_gpu_config"] = {
        "gpu": "A100-80GB",
        "model": "Llama-3.1-8B",
        "model_size_int4_gb": model_size_int4,
        "available_vram_gb": available_vram,
    }

    save_results(results, "kv_cache_analysis")
    return results


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------
def plot_outlier_heatmap(profile_results, model_key, save_plots=True):
    """
    Figure 7.1: Per-channel activation magnitude heatmap.
    Rows = transformer layers (fc1 input), columns = hidden dimensions.
    Outlier channels appear as bright vertical stripes.
    """
    plt = setup_matplotlib()

    fc1_keys = sorted(
        [k for k in profile_results if "fc1" in k and not k.startswith("_")]
    )
    if not fc1_keys:
        print("  No fc1 layer data -- skipping Figure 7.1.")
        return

    rows = []
    for k in fc1_keys:
        channel_data = profile_results[k].get("mean_channel_max", [])
        if isinstance(channel_data, list) and len(channel_data) > 0:
            rows.append(channel_data)
    if len(rows) < 2:
        print(f"  Only {len(rows)} layer(s) with data -- skipping heatmap.")
        return

    matrix = np.array(rows)
    if matrix.ndim != 2:
        print(f"  Unexpected matrix shape {matrix.shape} -- skipping.")
        return

    num_layers, hidden_dim = matrix.shape
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    log_matrix = np.log10(np.clip(matrix, 1e-6, None))
    im = ax.imshow(log_matrix, aspect="auto", cmap="hot",
                   interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("log10(mean channel max)", fontsize=7)

    ax.set_xlabel("Hidden dimension index")
    ax.set_ylabel("Transformer layer")
    step = max(1, num_layers // 8)
    ax.set_yticks(range(0, num_layers, step))
    ax.set_yticklabels([fc1_keys[i].replace("_fc1", "")
                        for i in range(0, num_layers, step)])

    # Mark universal top-1% dims
    consistency = profile_results.get("_consistency", {})
    universal_dims = consistency.get("universal_top1pct_dims", [])
    if 0 < len(universal_dims) <= 15:
        for dim in universal_dims:
            ax.axvline(x=dim, color=MANNING_COLORS["blue_l2"],
                       linewidth=0.5, linestyle="--", alpha=0.7)

    fig.tight_layout()
    save_figure(fig, "CH07_F01_Kalyanarangan", save_plots)
    plt.close(fig)
    print(f"  Figure 7.1: Activation heatmap ({num_layers} layers "
          f"x {hidden_dim} dims)")


def plot_emergence_comparison(emergence_results, save_plots=True):
    """
    Figure 7.2: Outlier emergence across model scales.
    Per-layer max/median ratio for each model.
    """
    plt = setup_matplotlib()

    n_models = len(emergence_results)
    if n_models == 0:
        print("  No emergence data -- skipping Figure 7.2.")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(5.5, 2.8),
                             squeeze=False)
    axes = axes[0]

    colors = [MANNING_COLORS["blue_l2"], MANNING_COLORS["orange_l2"],
              MANNING_COLORS["red_l3"]]
    hatches_list = [HATCHES[0], HATCHES[1], HATCHES[2]]

    global_max_ratio = 0
    for data in emergence_results.values():
        for l_data in data.get("per_layer", {}).values():
            global_max_ratio = max(global_max_ratio,
                                   l_data["max_to_median"])

    for idx, (model_key, data) in enumerate(emergence_results.items()):
        ax = axes[idx]
        per_layer = data.get("per_layer", {})
        if not per_layer:
            continue

        layers = sorted(per_layer.keys())
        ratios = [per_layer[l]["max_to_median"] for l in layers]
        layer_indices = list(range(len(layers)))

        ax.bar(layer_indices, ratios,
               color=colors[idx % len(colors)],
               edgecolor="black", linewidth=0.3,
               hatch=hatches_list[idx % len(hatches_list)],
               alpha=0.85)
        ax.set_xlabel("Layer index")
        params_b = data.get("params_b", "?")
        ax.set_title(f"{model_key} ({params_b}B)", fontsize=8,
                     fontweight="bold")
        ax.set_xticks(range(0, len(layers), max(1, len(layers) // 6)))

        # Shared y-axis scale for fair comparison
        ax.set_ylim(0, global_max_ratio * 1.15)

        # Annotate max ratio
        if ratios:
            max_r = max(ratios)
            max_i = ratios.index(max_r)
            ax.text(max_i, max_r + global_max_ratio * 0.03,
                    f"{max_r:.1f}x", ha="center", va="bottom",
                    fontsize=5, fontweight="bold")

    axes[0].set_ylabel("max / median ratio (per channel)")
    fig.suptitle("Channel magnitude variation by layer and scale",
                 fontsize=8, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "CH07_F02_Kalyanarangan", save_plots)
    plt.close(fig)
    print(f"  Figure 7.2: Emergence comparison generated")


def plot_error_decomposition(error_results, model_key, save_plots=True):
    """
    Figure 7.3: Quantization error decomposition.
    Stacked bars: top-1% channel MSE vs normal channel MSE.
    """
    plt = setup_matplotlib()

    layers = sorted([k for k in error_results if not k.startswith("_")])
    if not layers:
        print("  No error data -- skipping Figure 7.3.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.5), sharex=True)
    layer_indices = list(range(len(layers)))
    width = 0.7

    for ax_idx, (scheme, title) in enumerate([
        ("per_tensor", "Per-tensor INT8"),
        ("per_channel", "Per-channel INT8"),
    ]):
        ax = axes[ax_idx]
        normal_mse = [error_results[l][scheme]["normal_channel_mse"]
                      for l in layers]
        top_mse = [error_results[l][scheme]["top_channel_mse"]
                   for l in layers]

        ax.bar(layer_indices, normal_mse, width,
               label="Normal channels (99%)",
               color=MANNING_COLORS["blue_l2"], edgecolor="black",
               linewidth=0.3)
        ax.bar(layer_indices, top_mse, width,
               bottom=normal_mse,
               label="Top-1% channels",
               color=MANNING_COLORS["red_l3"], edgecolor="black",
               linewidth=0.3, hatch="//")

        ax.set_ylabel("MSE")
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.legend(loc="upper left", fontsize=6, framealpha=0.9)

        # Annotate selected bars
        for i in range(0, len(layers), max(1, len(layers) // 6)):
            total = top_mse[i] + normal_mse[i]
            if total > 0:
                pct = top_mse[i] / total * 100
                ax.text(i, total * 1.02, f"{pct:.0f}%",
                        ha="center", va="bottom", fontsize=5,
                        color=MANNING_COLORS["red_l3"],
                        fontweight="bold")

    axes[-1].set_xlabel("Transformer layer index")
    axes[-1].set_xticks(range(0, len(layers), max(1, len(layers) // 8)))
    fig.suptitle(f"Where quantization error comes from ({model_key})",
                 fontsize=8, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "CH07_F03_Kalyanarangan", save_plots)
    plt.close(fig)
    print(f"  Figure 7.3: Error decomposition generated")


def plot_kv_cache_analysis(kv_results, save_plots=True):
    """
    Figure 7.4: KV cache memory and concurrent user capacity.
    """
    plt = setup_matplotlib()

    concurrent = kv_results.get("_concurrent_users", {})
    gpu_cfg = kv_results.get("_gpu_config", {})
    if not concurrent:
        print("  No concurrent user data -- skipping Figure 7.4.")
        return

    seq_lengths = [2048, 8192, 32768, 131072]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))

    # Left: KV cache size vs context (Llama-3.1-8B, batch=1)
    fp16_sizes = [kv_results.get("Llama-3.1-8B", {}).get(
        f"seq{s}_batch1", {}).get("FP16", 0) for s in seq_lengths]
    int8_sizes = [kv_results.get("Llama-3.1-8B", {}).get(
        f"seq{s}_batch1", {}).get("INT8", 0) for s in seq_lengths]
    int4_sizes = [kv_results.get("Llama-3.1-8B", {}).get(
        f"seq{s}_batch1", {}).get("INT4", 0) for s in seq_lengths]

    ax = axes[0]
    x = np.arange(len(seq_lengths))
    w = 0.25
    ax.bar(x - w, fp16_sizes, w, label="FP16",
           color=MANNING_COLORS["red_l3"], edgecolor="black",
           linewidth=0.3, hatch="//")
    ax.bar(x, int8_sizes, w, label="INT8",
           color=MANNING_COLORS["blue_l2"], edgecolor="black",
           linewidth=0.3, hatch="\\\\")
    ax.bar(x + w, int4_sizes, w, label="INT4",
           color=MANNING_COLORS["green_l3"], edgecolor="black",
           linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(["2K", "8K", "32K", "128K"])
    ax.set_xlabel("Context length")
    ax.set_ylabel("KV cache size (GB)")
    ax.set_title("Llama-3.1-8B, batch=1", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left")

    # Right: concurrent users on A100
    ax2 = axes[1]
    fp16_users = [concurrent.get(f"seq{s}_FP16", {}).get("max_users", 0)
                  for s in seq_lengths]
    int4_users = [concurrent.get(f"seq{s}_INT4", {}).get("max_users", 0)
                  for s in seq_lengths]

    w2 = 0.3
    ax2.bar(x - w2/2, fp16_users, w2, label="FP16 KV",
            color=MANNING_COLORS["red_l3"], edgecolor="black",
            linewidth=0.3, hatch="//")
    ax2.bar(x + w2/2, int4_users, w2, label="INT4 KV",
            color=MANNING_COLORS["green_l3"], edgecolor="black",
            linewidth=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["2K", "8K", "32K", "128K"])
    ax2.set_xlabel("Context length")
    ax2.set_ylabel("Max concurrent users")
    ax2.set_title(f"Users on {gpu_cfg.get('gpu', 'A100-80GB')}",
                  fontsize=8, fontweight="bold")
    ax2.legend(fontsize=6, loc="upper right")

    for i in range(len(seq_lengths)):
        if fp16_users[i] > 0:
            mult = int4_users[i] / fp16_users[i]
            y_pos = max(fp16_users[i], int4_users[i])
            ax2.text(i, y_pos * 1.05, f"{mult:.0f}x",
                     ha="center", va="bottom", fontsize=6,
                     fontweight="bold")

    fig.tight_layout()
    save_figure(fig, "CH07_F04_Kalyanarangan", save_plots)
    plt.close(fig)
    print(f"  Figure 7.4: KV cache analysis generated")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ch7.1 -- Outlier profiling and KV cache analysis"
    )
    parser.add_argument(
        "--mode",
        choices=["profile", "emergence", "quant-error", "kv-cache", "all"],
        default="all",
        help="Which experiment(s) to run",
    )
    parser.add_argument(
        "--model", default="opt-6.7b",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model for profile and quant-error experiments",
    )
    parser.add_argument(
        "--emergence-models", nargs="+",
        default=["opt-125m", "opt-6.7b"],
        choices=list(MODEL_CONFIGS.keys()),
        help="Models for emergence comparison (loaded sequentially)",
    )
    parser.add_argument("--num-samples", type=int, default=16,
                        help="Number of calibration sequences")
    parser.add_argument("--seq-length", type=int, default=256,
                        help="Calibration sequence length "
                             "(256 recommended for 6.7b on T4)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save Manning-compliant figures")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("profile", "all"):
        profile_results = profile_outliers(
            model_key=args.model,
            num_samples=args.num_samples,
            seq_length=args.seq_length,
        )
        if args.save_plots:
            plot_outlier_heatmap(profile_results, args.model, True)

    if args.mode in ("emergence", "all"):
        emergence_results = outlier_emergence(
            models_to_compare=args.emergence_models,
            num_samples=args.num_samples,
            seq_length=args.seq_length,
        )
        if args.save_plots:
            plot_emergence_comparison(emergence_results, True)

    if args.mode in ("quant-error", "all"):
        error_results = quantization_error_decomposition(
            model_key=args.model,
            num_samples=args.num_samples,
            seq_length=args.seq_length,
        )
        if args.save_plots:
            plot_error_decomposition(error_results, args.model, True)

    if args.mode in ("kv-cache", "all"):
        kv_results = kv_cache_memory_analysis()
        if args.save_plots:
            plot_kv_cache_analysis(kv_results, True)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
    if args.save_plots:
        print(f"All figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()