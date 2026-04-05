#!/usr/bin/env python3
"""
Chapter 7 — Section 7.5 Companion Script
Compress KV cache with vector quantization: TurboQuant

This script builds TurboQuant (Zandieh et al., ICLR 2026) from scratch and
applies it to real KV cache tensors from OPT-6.7B.  The pedagogical arc:

  1. Lloyd-Max codebooks   — compute optimal scalar quantizers for the Beta
                             distribution that arises after random rotation,
                             validate against paper's theoretical MSE bounds
  2. TurboQuant core       — implement the MSE-optimal vector quantizer
                             (Algorithm 1 from the paper)
  3. KV cache capture      — populate real KV cache through forward passes
                             over WikiText-2, not fabricated tensors
  4. Reconstruction MSE    — quantize-dequantize real cache entries,
                             compare FP16 vs TQ4 vs TQ3
  5. Per-layer sensitivity — which layers degrade most under TurboQuant
  6. Perplexity showdown   — end-to-end WikiText-2 perplexity through
                             hook-based KV cache quantization

The key idea:  Every quantization method in Sections 7.2–7.4 targets *weights*.
The KV cache is a fundamentally different target — it arrives one vector at a
time during generation, with no opportunity for offline calibration.  TurboQuant
is data-oblivious: it randomly rotates each vector onto the unit sphere, where
coordinates follow a known Beta distribution, then applies pre-computed optimal
scalar quantizers.  No calibration data, no Hessian, no training.

NOTE on QJL:  The paper also presents Algorithm 2 (TurboQuant_prod), which
allocates (b-1) bits to MSE quantization and 1 bit to a Quantized Johnson-
Lindenstrauss (QJL) correction on the residual.  This produces unbiased inner-
product estimates, targeting vector database search where ranking fidelity
matters.  For KV cache attention, softmax amplifies QJL's variance more than
the MSE bias it removes, so MSE-only consistently wins on perplexity —
confirmed by 6+ independent implementations across Python, C, Rust, and MLX.
The TurboQuantizer class includes a reference QJL implementation (quantize_prod
/ dequantize_prod), but all experiments use Algorithm 1 (MSE-only).

Hardware: Colab T4 (16 GB VRAM).  OPT-6.7B in FP16 uses ~13 GB.
Model:   OPT-6.7B (head_dim=128 matches paper, consistent with §7.1–7.4).

Usage:
    # Full pipeline (T4 GPU required)
    python ch7_turboquant_kv_cache.py --mode all --save-plots

    # Codebook validation only (CPU-safe, quick)
    python ch7_turboquant_kv_cache.py --mode codebook --save-plots

    # KV cache capture + reconstruction MSE
    python ch7_turboquant_kv_cache.py --mode reconstruction --save-plots

    # Per-layer sensitivity
    python ch7_turboquant_kv_cache.py --mode sensitivity --save-plots

    # Perplexity comparison
    python ch7_turboquant_kv_cache.py --mode perplexity --save-plots

    # CPU-only (uses OPT-125M for illustration, not publishable numbers)
    python ch7_turboquant_kv_cache.py --mode all --save-plots \
        --model opt-125m --device cpu

References:
    Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal
    Distortion Rate," ICLR 2026.  arXiv:2504.19874.
"""

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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

# Paper Table 1 — theoretical MSE for unit-norm vectors (full vector)
PAPER_MSE = {1: 0.36, 2: 0.117, 3: 0.030, 4: 0.009}


@dataclass
class Config:
    """Central configuration for all experiments."""
    model_key: str = "opt-6.7b"
    device: str = "cuda"
    seed: int = 42
    calib_samples: int = 32
    calib_seq_len: int = 512
    eval_samples: int = 64
    eval_seq_len: int = 512
    head_dim: int = 128
    bit_widths: List[int] = field(default_factory=lambda: [2, 3, 4])
    lloyd_max_iters: int = 2000
    lloyd_max_tol: float = 1e-12
    codebook_dir: str = ""
    save_plots: bool = False
    plot_dir: str = "ch7_plots"
    dpi: int = 300
    figsize: Tuple[int, int] = (10, 6)

    @property
    def model_name(self):
        return MODEL_REGISTRY[self.model_key]


# ============================================================================
# Manning Publication Figure Styling
# ============================================================================

def setup_manning_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "grid.color": "#cccccc", "lines.linewidth": 1.5,
        "savefig.facecolor": "white", "savefig.edgecolor": "white",
    })

COLORS  = {"baseline": "#999999", "tq4": "#4A90D9", "tq3": "#E07B39"}
HATCHES = {"baseline": "",        "tq4": "//",      "tq3": "\\\\"}


# ============================================================================
# Shared Utilities
# ============================================================================

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def gpu_mem_mb():
    return torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_wikitext2_tokens(tokenizer, n_samples, seq_len, split="test"):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    seqs = []
    for i in range(0, len(ids) - seq_len, seq_len):
        seqs.append(ids[i:i+seq_len].unsqueeze(0))
        if len(seqs) >= n_samples:
            break
    print(f"  Loaded {len(seqs)} sequences of length {seq_len} "
          f"from WikiText-2 ({split})")
    return seqs

def save_or_show(fig, stem, cfg):
    fig.tight_layout()
    if cfg.save_plots:
        os.makedirs(cfg.plot_dir, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(cfg.plot_dir, f"{stem}.{ext}"),
                        dpi=cfg.dpi, bbox_inches="tight", format=ext)
        print(f"  Saved: {cfg.plot_dir}/{stem}.{{pdf,png}}")
    else:
        plt.show()
    plt.close(fig)


# ============================================================================
# Lloyd-Max Codebook for Beta Distribution
# ============================================================================

def compute_lloyd_max_codebook(d, bits, max_iter=2000, tol=1e-12):
    """
    Compute Lloyd-Max optimal codebook for the coordinate distribution
    of a randomly rotated unit vector in R^d.

    After rotation, each coordinate follows Beta(d/2, d/2) on [-1, 1].
    Lloyd-Max finds centroids and boundaries that minimize MSE for this
    known distribution — solved once, reused forever.
    """
    from scipy.stats import beta as beta_dist
    from scipy.integrate import quad

    a = d / 2.0
    rv = beta_dist(a, a)
    n_levels = 2 ** bits

    qpts = np.linspace(0, 1, n_levels + 1)
    qpts[0], qpts[-1] = 1e-10, 1 - 1e-10
    boundaries = 2.0 * rv.ppf(qpts) - 1.0
    boundaries[0], boundaries[-1] = -1.0, 1.0

    def cond_mean(lo_t, hi_t):
        lo_u = max((lo_t + 1) / 2, 0.0)
        hi_u = min((hi_t + 1) / 2, 1.0)
        if hi_u <= lo_u:
            return (lo_t + hi_t) / 2
        num, _ = quad(lambda u: (2*u - 1) * rv.pdf(u), lo_u, hi_u, limit=100)
        den, _ = quad(lambda u: rv.pdf(u), lo_u, hi_u, limit=100)
        return num / den if den > 1e-15 else (lo_t + hi_t) / 2

    centroids = np.array([cond_mean(boundaries[i], boundaries[i+1])
                          for i in range(n_levels)])

    for iteration in range(max_iter):
        old = centroids.copy()
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i-1] + centroids[i]) / 2
        for i in range(n_levels):
            centroids[i] = cond_mean(boundaries[i], boundaries[i+1])
        if np.max(np.abs(centroids - old)) < tol:
            print(f"    Lloyd-Max converged in {iteration+1} iterations "
                  f"(d={d}, bits={bits})")
            break

    mse = 0.0
    for i in range(n_levels):
        lo_u = max((boundaries[i] + 1) / 2, 0.0)
        hi_u = min((boundaries[i+1] + 1) / 2, 1.0)
        if hi_u <= lo_u:
            continue
        val, _ = quad(lambda u, c=centroids[i]: ((2*u-1) - c)**2 * rv.pdf(u),
                      lo_u, hi_u, limit=100)
        mse += val

    return {"centroids": centroids.tolist(), "boundaries": boundaries.tolist(),
            "mse": float(mse), "bits": bits, "d": d}


def load_or_compute_codebooks(cfg):
    codebooks = {}
    d = cfg.head_dim
    for bits in cfg.bit_widths:
        path = os.path.join(cfg.codebook_dir,
                            f"lloyd_max_beta_d{d}_b{bits}.json")
        if os.path.exists(path):
            with open(path) as f:
                cb = json.load(f)
            print(f"  Loaded codebook from {path} "
                  f"(d={d}, bits={bits}, MSE={cb['mse']:.6f})")
        else:
            print(f"  Computing Lloyd-Max codebook (d={d}, bits={bits})...")
            cb = compute_lloyd_max_codebook(d, bits,
                                            max_iter=cfg.lloyd_max_iters,
                                            tol=cfg.lloyd_max_tol)
            print(f"    MSE = {cb['mse']:.6f}")
            os.makedirs(cfg.codebook_dir, exist_ok=True)
            with open(path, "w") as f:
                json.dump(cb, f, indent=2)
            print(f"    Saved to {path}")
        codebooks[bits] = cb
    return codebooks


# ============================================================================
# TurboQuant Core Algorithm
# ============================================================================

class TurboQuantizer:
    """
    TurboQuant vector quantizer (Zandieh et al., ICLR 2026).

    Primary path — Algorithm 1 (MSE-optimal):
      1. Normalize each vector, store L2 norm in FP16
      2. Rotate by random orthogonal matrix Pi
      3. Quantize each coordinate to the nearest Lloyd-Max centroid
      Dequantize: look up centroids, inverse-rotate, rescale by norm.

    Reference path — Algorithm 2 (inner-product-optimal, QJL):
      Uses (b-1) bits for MSE + 1 bit for QJL residual correction.
      Produces unbiased inner-product estimates for vector search.
      NOT used for KV cache experiments (softmax amplifies QJL variance).
    """

    def __init__(self, head_dim, codebooks, seed=42, device="cuda"):
        self.head_dim = head_dim
        self.device = device

        # Random rotation matrix (shared across all layers/heads).
        gen = torch.Generator()
        gen.manual_seed(seed)
        G = torch.randn(head_dim, head_dim, generator=gen)
        Pi, _ = torch.linalg.qr(G)                                    #A
        self.Pi = Pi.to(device)

        # Pre-load codebook tensors
        self.centroids = {}
        self.boundaries = {}
        for bits, cb in codebooks.items():
            self.centroids[bits] = torch.tensor(
                cb["centroids"], dtype=torch.float32, device=device)
            self.boundaries[bits] = torch.tensor(
                cb["boundaries"], dtype=torch.float32, device=device)

        # Random sign matrix for QJL reference implementation
        gen_qjl = torch.Generator()
        gen_qjl.manual_seed(seed + 1)
        self.S = (2 * torch.randint(
            0, 2, (head_dim, head_dim), generator=gen_qjl
        ).float() - 1).to(device)

    # ------------------------------------------------------------------
    # Algorithm 1: MSE-optimal (used for all KV cache experiments)
    # ------------------------------------------------------------------

    def quantize(self, vectors, bits):
        """Quantize vectors using TurboQuant MSE (Algorithm 1)."""
        shape = vectors.shape
        flat = vectors.reshape(-1, self.head_dim).float()

        norms = torch.norm(flat, dim=1, keepdim=True)              #B
        normalized = flat / torch.clamp(norms, min=1e-8)           #C
        rotated = normalized @ self.Pi.T                           #D

        interior_b = self.boundaries[bits][1:-1]                   #E
        indices = torch.searchsorted(interior_b, rotated)          #F

        return {"indices": indices, "norms": norms.half(),
                "bits": bits, "shape": shape}

    def dequantize(self, compressed):
        """Dequantize MSE-compressed vectors."""
        c = self.centroids[compressed["bits"]]
        recon_rot = c[compressed["indices"]]                       #G
        recon_norm = recon_rot @ self.Pi                           #H
        recon = recon_norm * compressed["norms"].float()           #I
        return recon.reshape(compressed["shape"])

    def quantize_dequantize(self, vectors, bits):
        """Round-trip for hook-based evaluation and MSE measurement."""
        return self.dequantize(self.quantize(vectors, bits))

    # ------------------------------------------------------------------
    # Algorithm 2: Inner-product-optimal with QJL (reference only)
    #
    # Targets vector database search, not KV cache attention.
    # Softmax amplifies QJL variance; MSE-only wins on perplexity.
    # Included for completeness — not called by any experiment.
    # ------------------------------------------------------------------

    def quantize_prod(self, vectors, bits):
        """TurboQuant Prod: (b-1)-bit MSE + 1-bit QJL residual."""
        assert bits >= 2, "QJL needs >= 2 total bits (1 MSE + 1 QJL)"
        shape = vectors.shape
        flat = vectors.reshape(-1, self.head_dim).float()

        norms = torch.norm(flat, dim=1, keepdim=True)
        normalized = flat / torch.clamp(norms, min=1e-8)
        rotated = normalized @ self.Pi.T

        mse_bits = bits - 1
        c = self.centroids[mse_bits]
        indices = torch.searchsorted(
            self.boundaries[mse_bits][1:-1], rotated)
        mse_recon = c[indices]

        residual = rotated - mse_recon
        gamma = torch.norm(residual, dim=1, keepdim=True)
        r_norm = residual / torch.clamp(gamma, min=1e-8)
        qjl_signs = torch.sign(r_norm @ self.S.T)
        qjl_signs[qjl_signs == 0] = 1.0

        return {"mse_indices": indices, "norms": norms.half(),
                "gamma": gamma.half(), "qjl_signs": qjl_signs,
                "mse_bits": mse_bits, "shape": shape}

    def dequantize_prod(self, compressed):
        """Dequantize Prod-compressed vectors (MSE + QJL correction)."""
        c = self.centroids[compressed["mse_bits"]]
        mse_recon = c[compressed["mse_indices"]]
        d = self.head_dim
        qjl_corr = (math.sqrt(math.pi / 2) / d
                     * compressed["gamma"].float()
                     * (compressed["qjl_signs"] @ self.S))
        recon_norm = (mse_recon + qjl_corr) @ self.Pi
        return (recon_norm * compressed["norms"].float()
                ).reshape(compressed["shape"])


# ============================================================================
# Memory Accounting
# ============================================================================

def compute_compressed_size(head_dim, bits):
    """Per-vector: bits * head_dim (indices) + 16 bits (FP16 norm)."""
    total_bits = bits * head_dim + 16
    fp16_bits = 16 * head_dim
    return {"total_bytes": total_bits / 8, "fp16_bytes": fp16_bits / 8,
            "compression_ratio": fp16_bits / total_bits,
            "bits_per_element": total_bits / head_dim}

def print_memory_table(cfg):
    print("\n  KV Cache Memory per Token per Layer (head_dim=128)")
    print(f"  {'Config':<8} {'Bits/elem':>10} {'Bytes/vec':>10} "
          f"{'Bytes/tok':>10} {'Compress':>10}")
    print(f"  {'-'*50}")
    for name, bits in [("FP16", None), ("TQ4", 4), ("TQ3", 3)]:
        if bits is None:
            bpe, bv, comp = 16.0, cfg.head_dim * 2, 1.0
        else:
            info = compute_compressed_size(cfg.head_dim, bits)
            bpe, bv, comp = (info["bits_per_element"],
                             int(info["total_bytes"]),
                             info["compression_ratio"])
        print(f"  {name:<8} {bpe:>10.2f} {bv:>10} {bv*2:>10} {comp:>9.2f}x")


# ============================================================================
# KV Cache Capture via Hooks
# ============================================================================

@torch.no_grad()
def capture_kv_cache(model, sequences, head_dim, max_sequences=8):
    """
    Capture K/V tensors via hooks on k_proj/v_proj — avoids the DynamicCache
    API entirely (which varies across transformers versions).

    Returns list of per-sequence caches, each a list of (key, value) per layer.
    Shapes: [1, num_heads, seq_len, head_dim].
    """
    model.eval()
    input_device = next(model.parameters()).device
    layers = model.model.decoder.layers
    all_caches = []

    n_seq = min(len(sequences), max_sequences)
    for i in range(n_seq):
        seq = sequences[i].to(input_device)
        layer_kvs = [{} for _ in range(len(layers))]
        hooks = []

        for lidx, layer in enumerate(layers):
            def make_hooks(idx):
                def k_hook(m, inp, out):
                    layer_kvs[idx]["k"] = out.detach().cpu()
                def v_hook(m, inp, out):
                    layer_kvs[idx]["v"] = out.detach().cpu()
                return k_hook, v_hook
            kh, vh = make_hooks(lidx)
            hooks.append(layer.self_attn.k_proj.register_forward_hook(kh))
            hooks.append(layer.self_attn.v_proj.register_forward_hook(vh))

        model(seq)
        for h in hooks:
            h.remove()

        pairs = []
        for lidx in range(len(layers)):
            kr, vr = layer_kvs[lidx]["k"], layer_kvs[lidx]["v"]
            b, s, td = kr.shape
            nh = td // head_dim
            pairs.append((
                kr.view(b, s, nh, head_dim).transpose(1, 2),
                vr.view(b, s, nh, head_dim).transpose(1, 2),
            ))
        all_caches.append(pairs)
        if (i + 1) % 4 == 0:
            print(f"    Captured KV cache: {i+1}/{n_seq} sequences")

    k0, v0 = all_caches[0][0]
    print(f"  KV cache captured: {n_seq} sequences, {len(all_caches[0])} layers")
    print(f"  Key shape per layer: {list(k0.shape)}")
    print(f"  Value shape per layer: {list(v0.shape)}")
    return all_caches


# ============================================================================
# Experiment 1: Codebook Validation
# ============================================================================

def run_codebook_validation(cfg):
    """
    Compute Lloyd-Max codebooks and validate against paper MSE.
    Paper reports MSE = {0.36, 0.117, 0.030, 0.009} for bits = {1,2,3,4}.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Lloyd-Max Codebook Validation")
    print("=" * 72)

    d = cfg.head_dim
    codebooks = load_or_compute_codebooks(cfg)

    if 1 not in codebooks:
        print(f"\n  Computing 1-bit codebook for validation...")
        codebooks[1] = compute_lloyd_max_codebook(
            d, 1, max_iter=cfg.lloyd_max_iters, tol=cfg.lloyd_max_tol)

    print(f"\n  Codebook MSE validation (d={d}):")
    print(f"  {'Bits':<6} {'Per-coord MSE':<14} {'Vector MSE':<12} "
          f"{'Paper MSE':<12} {'Ratio':<8}")
    print(f"  {'-'*54}")
    for bits in sorted(codebooks.keys()):
        pc = codebooks[bits]["mse"]
        vm = pc * d
        paper = PAPER_MSE.get(bits)
        if paper:
            print(f"  {bits:<6} {pc:<14.6f} {vm:<12.6f} "
                  f"{paper:<12.3f} {vm/paper:<8.3f}")

    print(f"\n  Empirical validation on 10,000 synthetic unit vectors...")
    set_seed(cfg.seed)
    vecs = torch.randn(10000, d)
    vecs = vecs / vecs.norm(dim=1, keepdim=True)
    tq = TurboQuantizer(d, codebooks, seed=cfg.seed, device="cpu")

    empirical = {}
    for bits in sorted(codebooks.keys()):
        if bits == 1:
            continue
        recon = tq.quantize_dequantize(vecs, bits)
        mse = ((vecs - recon) ** 2).sum(dim=1).mean().item()
        empirical[bits] = mse
        print(f"    TQ{bits} empirical MSE: {mse:.6f}  "
              f"(paper: {PAPER_MSE.get(bits, 'N/A')})")

    return codebooks, empirical


# ============================================================================
# Experiment 2: Reconstruction MSE on Real KV Cache
# ============================================================================

def run_reconstruction_analysis(cfg, codebooks):
    """Capture real KV cache, measure reconstruction MSE for TQ4 and TQ3."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: Reconstruction MSE on Real KV Cache")
    print("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    print(f"\n  Loading {cfg.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float16, device_map=cfg.device)

    calib_seqs = load_wikitext2_tokens(
        tokenizer, cfg.calib_samples, cfg.calib_seq_len, split="train")

    print(f"\n  Capturing KV cache through forward passes...")
    all_caches = capture_kv_cache(model, calib_seqs, cfg.head_dim)

    tq = TurboQuantizer(cfg.head_dim, codebooks, seed=cfg.seed, device="cpu")

    results = {}
    for name, bits in [("TQ4", 4), ("TQ3", 3)]:
        k_layers, v_layers = [], []
        for layer_idx in range(len(all_caches[0])):
            ka, va, cnt = 0.0, 0.0, 0
            for cache in all_caches:
                k, v = cache[layer_idx]
                kr = tq.quantize_dequantize(k, bits)
                vr = tq.quantize_dequantize(v, bits)
                ka += ((k.float() - kr.float())**2).mean().item()
                va += ((v.float() - vr.float())**2).mean().item()
                cnt += 1
            k_layers.append(ka / cnt)
            v_layers.append(va / cnt)

        ak, av = np.mean(k_layers), np.mean(v_layers)
        results[name] = {"key_mse_per_layer": k_layers,
                         "val_mse_per_layer": v_layers,
                         "avg_key_mse": ak, "avg_val_mse": av, "bits": bits}
        print(f"  {name:<6}  Key MSE: {ak:.6f}  Val MSE: {av:.6f}")

    del model
    free_gpu()
    return results


def plot_reconstruction_mse(results, cfg):
    """Figure 7.18: Reconstruction MSE — TQ4 vs TQ3, keys vs values."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    names, ckeys = list(results.keys()), ["tq4", "tq3"]
    x = np.arange(len(names))

    for ax_idx, (title, mkey) in enumerate([
        ("Key Cache", "avg_key_mse"), ("Value Cache", "avg_val_mse"),
    ]):
        ax = axes[ax_idx]
        vals = [results[n][mkey] for n in names]
        bars = ax.bar(x, vals, 0.5,
                      color=[COLORS[c] for c in ckeys],
                      edgecolor="black", linewidth=0.6)
        for bar, ck in zip(bars, ckeys):
            bar.set_hatch(HATCHES[ck])
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Mean Squared Error")
        ax.set_title(f"{title} Reconstruction MSE")
        for i, v in enumerate(vals):
            ax.text(i, v * 1.05, f"{v:.4f}", ha="center",
                    va="bottom", fontsize=9)

    fig.suptitle("TurboQuant KV Cache Reconstruction Error (OPT-6.7B)",
                 fontsize=12, y=1.02)
    save_or_show(fig, "fig7_18_tq_reconstruction_mse", cfg)


# ============================================================================
# Experiment 3: Per-Layer Sensitivity
# ============================================================================

def plot_per_layer_sensitivity(results, cfg):
    """Figure 7.19: Per-layer MSE for TQ4 and TQ3."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_layers = len(results["TQ3"]["key_mse_per_layer"])
    x = np.arange(n_layers)

    for ax_idx, (title, mkey) in enumerate([
        ("Key Cache", "key_mse_per_layer"),
        ("Value Cache", "val_mse_per_layer"),
    ]):
        ax = axes[ax_idx]
        for name, ck in [("TQ4", "tq4"), ("TQ3", "tq3")]:
            ax.plot(x, results[name][mkey], color=COLORS[ck],
                    label=name, linewidth=1.2, marker=".", markersize=3)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("MSE")
        ax.set_title(f"{title} — Per-Layer MSE")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Per-Layer Sensitivity to KV Cache Quantization (OPT-6.7B)",
                 fontsize=12, y=1.02)
    save_or_show(fig, "fig7_19_tq_per_layer_sensitivity", cfg)


# ============================================================================
# Experiment 4: Perplexity Showdown
# ============================================================================

@torch.no_grad()
def evaluate_perplexity(model, sequences, label=""):
    """Standard autoregressive perplexity: exp(mean NLL per token)."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    dev = next(model.parameters()).device
    for i, seq in enumerate(sequences):
        seq = seq.to(dev)
        nll = model(seq, labels=seq).loss.item()
        n = seq.shape[1] - 1
        total_nll += nll * n
        total_tokens += n
        if (i + 1) % 16 == 0:
            print(f"    [{label}] {i+1}/{len(sequences)}, "
                  f"PPL: {np.exp(total_nll / total_tokens):.2f}")
    return np.exp(total_nll / total_tokens)


def install_tq_hooks(model, tq, bits):
    """Hook k_proj/v_proj to quantize-dequantize during forward pass."""
    hooks = []
    for layer in model.model.decoder.layers:
        for proj_name in ["k_proj", "v_proj"]:
            proj = getattr(layer.self_attn, proj_name)
            def make_hook(b):
                def hook_fn(module, inp, output):
                    flat = output.reshape(-1, tq.head_dim)
                    recon = tq.quantize_dequantize(flat, b)
                    return recon.reshape(output.shape).to(output.dtype)
                return hook_fn
            hooks.append(proj.register_forward_hook(make_hook(bits)))
    return hooks


def run_perplexity_comparison(cfg, codebooks):
    """Perplexity showdown: FP16 vs TQ4 vs TQ3 on WikiText-2 test."""
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Perplexity Showdown — FP16 vs TQ4 vs TQ3")
    print("=" * 72)

    if cfg.device == "cpu" and cfg.model_key == "opt-6.7b":
        print("  Skipping perplexity (OPT-6.7B requires GPU)")
        return None

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    eval_seqs = load_wikitext2_tokens(
        tokenizer, cfg.eval_samples, cfg.eval_seq_len, split="test")

    print(f"\n  Loading {cfg.model_name}...")
    mem_before = gpu_mem_mb()
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=torch.float16, device_map=cfg.device)
    print(f"  GPU memory: {gpu_mem_mb() - mem_before:.0f} MB")

    tq = TurboQuantizer(cfg.head_dim, codebooks, seed=cfg.seed,
                        device=cfg.device)
    results = {}

    # 1. FP16 baseline
    print(f"\n[1/3] FP16 Baseline...")
    ppl = evaluate_perplexity(model, eval_seqs, "FP16")
    print(f"  FP16 Perplexity: {ppl:.2f}")
    results["FP16"] = {"ppl": ppl, "bits_per_elem": 16.0, "compression": 1.0}

    # 2-3. TQ4 and TQ3
    for idx, (name, bits) in enumerate([("TQ4", 4), ("TQ3", 3)], start=2):
        print(f"\n[{idx}/3] {name}...")
        hooks = install_tq_hooks(model, tq, bits)
        ppl = evaluate_perplexity(model, eval_seqs, name)
        print(f"  {name} Perplexity: {ppl:.2f}")
        info = compute_compressed_size(cfg.head_dim, bits)
        results[name] = {"ppl": ppl, "bits_per_elem": info["bits_per_element"],
                         "compression": info["compression_ratio"]}
        for h in hooks:
            h.remove()

    del model, tq
    free_gpu()

    # Summary table
    print(f"\n  {'='*56}")
    print(f"  {'Config':<8} {'PPL':>8} {'Bits/elem':>10} "
          f"{'Compression':>12} {'PPL delta':>10}")
    print(f"  {'-'*56}")
    fp16_ppl = results["FP16"]["ppl"]
    for name, r in results.items():
        delta = ((r["ppl"] / fp16_ppl) - 1) * 100
        print(f"  {name:<8} {r['ppl']:>8.2f} {r['bits_per_elem']:>10.2f} "
              f"{r['compression']:>11.2f}x {delta:>+9.1f}%")
    print(f"  {'='*56}")
    return results


def plot_perplexity_comparison(results, cfg):
    """Figure 7.20: Perplexity comparison — FP16 vs TQ4 vs TQ3."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    ppls = [results[n]["ppl"] for n in names]
    ckeys = ["baseline", "tq4", "tq3"]
    x = np.arange(len(names))

    bars = ax.bar(x, ppls, 0.5,
                  color=[COLORS[c] for c in ckeys],
                  edgecolor="black", linewidth=0.6)
    for bar, ck in zip(bars, ckeys):
        bar.set_hatch(HATCHES[ck])
    for i, v in enumerate(ppls):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    for i, name in enumerate(names):
        c = results[name]["compression"]
        if c > 1:
            ax.text(i, ax.get_ylim()[0] + 0.2, f"{c:.1f}x",
                    ha="center", fontsize=9, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title(f"WikiText-2 Perplexity: TurboQuant KV Cache "
                 f"({cfg.model_key.upper()})")
    save_or_show(fig, "fig7_20_tq_perplexity_comparison", cfg)


def plot_compression_vs_perplexity(results, cfg):
    """Figure 7.21: Bits-per-element vs perplexity scatter."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ckeys = {"FP16": "baseline", "TQ4": "tq4", "TQ3": "tq3"}
    markers = {"FP16": "s", "TQ4": "o", "TQ3": "D"}

    for name, r in results.items():
        ax.scatter(r["bits_per_elem"], r["ppl"],
                   color=COLORS[ckeys[name]], marker=markers[name],
                   s=140, edgecolors="black", linewidth=0.8,
                   label=f'{name} ({r["compression"]:.1f}x)', zorder=5)

    ax.set_xlabel("Bits per Element")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title(f"Compression-Quality Tradeoff ({cfg.model_key.upper()})")
    ax.legend(fontsize=10)
    ax.invert_xaxis()
    save_or_show(fig, "fig7_21_tq_compression_vs_perplexity", cfg)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Chapter 7.5 — TurboQuant KV Cache Quantization")
    p.add_argument("--mode", default="all",
                   choices=["all", "codebook", "reconstruction",
                            "sensitivity", "perplexity"])
    p.add_argument("--model", default="opt-6.7b",
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--device", default=None)
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--plot-dir", default="ch7_plots")
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--calib-samples", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    import logging
    for name in ("httpx", "httpcore", "tokenizers",
                 "huggingface_hub", "datasets"):
        logging.getLogger(name).setLevel(logging.WARNING)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    head_dims = {"opt-125m": 64, "opt-1.3b": 64, "opt-6.7b": 128}

    cfg = Config(
        model_key=args.model, device=device,
        save_plots=args.save_plots, plot_dir=args.plot_dir,
        eval_samples=args.eval_samples, calib_samples=args.calib_samples,
        head_dim=head_dims.get(args.model, 128),
        codebook_dir=str(Path(__file__).resolve().parent / "codebooks"),
    )

    set_seed(cfg.seed)
    setup_manning_style()

    print(f"Configuration:")
    print(f"  Model:     {cfg.model_name}")
    print(f"  Device:    {cfg.device}")
    print(f"  Head dim:  {cfg.head_dim}")
    print(f"  Bit widths: {cfg.bit_widths}")
    if device == "cuda":
        print(f"  GPU:       {torch.cuda.get_device_name()}")
        print(f"  VRAM:      "
              f"{torch.cuda.get_device_properties(0).total_memory/1024**2:.0f} MB")
    print()

    mode = args.mode
    codebooks = None
    recon_results = None

    if mode in ("all", "codebook"):
        codebooks, _ = run_codebook_validation(cfg)
    if codebooks is None:
        codebooks = load_or_compute_codebooks(cfg)
    if mode in ("all", "codebook"):
        print_memory_table(cfg)

    if mode in ("all", "reconstruction", "sensitivity"):
        recon_results = run_reconstruction_analysis(cfg, codebooks)
        plot_reconstruction_mse(recon_results, cfg)

    if mode in ("all", "sensitivity"):
        if recon_results is None:
            recon_results = run_reconstruction_analysis(cfg, codebooks)
        plot_per_layer_sensitivity(recon_results, cfg)

    if mode in ("all", "perplexity"):
        ppl_results = run_perplexity_comparison(cfg, codebooks)
        if ppl_results:
            plot_perplexity_comparison(ppl_results, cfg)
            plot_compression_vs_perplexity(ppl_results, cfg)

    print("\n" + "=" * 72)
    print("All experiments complete.")
    if cfg.save_plots:
        print(f"Figures saved to: {cfg.plot_dir}/")
    print("=" * 72)


if __name__ == "__main__":
    main()