#!/usr/bin/env python3
"""
Chapter 5, Section 5.5: Adapt Transformers Efficiently
=====================================================

Companion script for "Efficient AI in Practice: Quantization and Fast Inference"
Manning Publications

Demonstrates:
  1. Quick sublayer sensitivity diagnostic — which of the 6 linear sublayer
     types in BERT hurts most when weights are fake-quantized.  (Figure 5.15)
  2. Selective QAT fine-tuning on SST-2 — apply fake quantizers to different
     sublayer subsets and measure actual accuracy recovery.    (Figure 5.16)

Usage:
  python ch5_transformer_qat.py --mode sensitivity     # Quick diagnostic
  python ch5_transformer_qat.py --mode qat             # QAT fine-tuning
  python ch5_transformer_qat.py --mode all --save-plots # Everything + figures

Requirements:
  pip install torch transformers datasets matplotlib
"""

import argparse
import copy
import os
import sys
import time
import warnings
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Manning figure styling ────────────────────────────────────────────────────
MANNING_COLORS = {
    "L1": "#4A90B8", "L2": "#D4823E", "L3": "#5EA55E",
    "L4": "#C04E4E", "purple": "#8B6DAF", "gray": "#7A7A7A",
    "dark": "#2D2D2D",
}
HATCHES = ["//", "\\\\", "xx", "..", "||", "--"]


def setup_manning_style():
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3,
    })


def save_figure(fig, name, save_plots):
    if save_plots:
        fig.savefig(f"{name}.png", bbox_inches="tight", dpi=300)
        fig.savefig(f"{name}.pdf", bbox_inches="tight")
        print(f"  Saved: {name}.png, {name}.pdf")
    plt.show()
    plt.close(fig)


# ── Fake quantization primitives ──────────────────────────────────────────────

def fake_quantize_per_channel(weight, bits=8):
    """Per-channel symmetric fake quantization (same as Listing 5.12)."""
    q_max = (1 << (bits - 1)) - 1
    flat = weight.detach().reshape(weight.shape[0], -1)
    abs_max = flat.abs().amax(dim=1)
    scales = abs_max / q_max
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    scales_bc = scales.view(-1, 1)
    w_q = torch.clamp(torch.round(weight / scales_bc), -q_max, q_max)
    return w_q * scales_bc


class FakeQuantizedLinear(nn.Module):
    """Drop-in replacement applying per-channel fake quantization with STE.

    Same pattern as FQConv2d (Listing 5.13), adapted for nn.Linear.
    """
    def __init__(self, original_linear, bits=8):
        super().__init__()
        self.linear = original_linear
        self.bits = bits
        self.q_max = (1 << (bits - 1)) - 1
        self.q_min = -self.q_max

    def forward(self, x):
        w = self.linear.weight
        flat = w.detach().reshape(w.shape[0], -1)
        abs_max = flat.abs().amax(dim=1)
        scales = abs_max / self.q_max
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        scales_bc = scales.view(-1, 1)

        w_q = torch.clamp(torch.round(w / scales_bc),           #A
                          self.q_min, self.q_max)
        w_dq = w_q * scales_bc
        w_fq = w + (w_dq - w).detach()                          #B
        return F.linear(x, w_fq, self.linear.bias)
    #A Quantize-then-dequantize: round to grid, scale back
    #B STE trick: forward uses quantized weights, backward flows to original


# ── Selective QAT application ─────────────────────────────────────────────────

def get_sublayer_paths():
    """The 6 linear sublayer types within each BERT encoder block."""
    return OrderedDict([
        ("query",    "attention.self.query"),
        ("key",      "attention.self.key"),
        ("value",    "attention.self.value"),
        ("output",   "attention.output.dense"),
        ("ffn_up",   "intermediate.dense"),
        ("ffn_down", "output.dense"),
    ])


def get_sublayer(block, path):
    module = block
    for attr in path.split("."):
        module = getattr(module, attr)
    return module


def apply_selective_qat(model, strategy="all", bits=8):
    """Wrap selected linear layers with fake quantizers.

    Strategies:
      'none':       No quantization (FP32 baseline)
      'attn_only':  Q, K, V, O projections (4 per block = 48 total)
      'ffn_only':   FFN up + down (2 per block = 24 total)
      'all':        All 6 linear layers per block (72 total)

    Returns count of quantized layers.
    """
    if strategy == "none":
        return 0

    # Support both BertModel and BertForSequenceClassification
    encoder = model.bert.encoder if hasattr(model, "bert") else model.encoder

    count = 0
    for block in encoder.layer:
        if strategy in ("all", "attn_only"):
            attn = block.attention
            attn.self.query = FakeQuantizedLinear(attn.self.query, bits)
            attn.self.key = FakeQuantizedLinear(attn.self.key, bits)
            attn.self.value = FakeQuantizedLinear(attn.self.value, bits)
            attn.output.dense = FakeQuantizedLinear(attn.output.dense, bits)
            count += 4
        if strategy in ("all", "ffn_only"):
            block.intermediate.dense = FakeQuantizedLinear(
                block.intermediate.dense, bits)
            block.output.dense = FakeQuantizedLinear(
                block.output.dense, bits)
            count += 2
    return count


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sst2(tokenizer, max_train=2000, max_val=872, max_length=128):
    """Load SST-2 from GLUE with tokenization.

    Uses a subset for tractable training. max_val=872 is the full
    SST-2 validation set size.
    """
    from datasets import load_dataset

    print(f"Loading SST-2 (train: {max_train}, val: {max_val})...")
    dataset = load_dataset("glue", "sst2")

    def tokenize(examples):
        return tokenizer(
            examples["sentence"], padding="max_length",
            truncation=True, max_length=max_length)

    train_ds = dataset["train"].select(
        range(min(max_train, len(dataset["train"]))))
    val_ds = dataset["validation"].select(
        range(min(max_val, len(dataset["validation"]))))

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    train_ds.set_format("torch",
                        columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch",
                      columns=["input_ids", "attention_mask", "label"])

    return train_ds, val_ds


def make_dataloader(dataset, batch_size=32, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)


# ── Training and evaluation ───────────────────────────────────────────────────

def evaluate(model, val_loader, device):
    """Evaluate accuracy on validation set."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def train_one_epoch(model, train_loader, optimizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / n_batches


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1: Quick sublayer sensitivity diagnostic                   (Figure 5.15)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_sensitivity(save_plots=False, device="cpu"):
    """Quantize one sublayer type at a time, measure output MSE.

    Forward-pass diagnostic — no training. Use to decide where to
    place fake quantizers before committing to a QAT training run.
    """
    from transformers import BertModel, BertTokenizer

    print("\n" + "=" * 70)
    print("SUBLAYER SENSITIVITY DIAGNOSTIC")
    print("=" * 70)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased").eval().to(device)

    sentences = [
        "The federal reserve raised interest rates by 25 basis points.",
        "Transformers use multi-head attention to process sequences.",
        "Revenue exceeded expectations in the third quarter of 2024.",
        "Machine learning models require careful validation.",
        "The new architecture reduced inference latency by forty percent.",
        "Climate change impacts are accelerating across multiple regions.",
        "The patient was diagnosed with a rare autoimmune condition.",
        "The quick brown fox jumps over the lazy dog near the river.",
    ]
    inputs = tokenizer(sentences, padding=True, truncation=True,
                       max_length=64, return_tensors="pt").to(device)
    fp32_output = model(**inputs).last_hidden_state

    sublayer_paths = get_sublayer_paths()
    all_results = {}

    for bits in [8, 4]:
        print(f"\n  INT{bits} per-channel weight fake quantization:")
        print(f"  {'Sublayer':<12} {'Output MSE':>14} {'Cos Sim':>12}")
        print(f"  {'─' * 42}")
        results = {}
        for sub_name, sub_path in sublayer_paths.items():
            q_model = copy.deepcopy(model)
            for block in q_model.encoder.layer:
                layer = get_sublayer(block, sub_path)
                layer.weight.copy_(
                    fake_quantize_per_channel(layer.weight, bits))
            q_output = q_model(**inputs).last_hidden_state
            mse = F.mse_loss(q_output, fp32_output).item()
            cos = F.cosine_similarity(
                q_output.reshape(1, -1), fp32_output.reshape(1, -1)).item()
            results[sub_name] = {"mse": mse, "cos": cos}
            print(f"  {sub_name:<12} {mse:>14.6e} {cos:>12.8f}")
            del q_model
        all_results[bits] = results

        attn_mse = sum(r["mse"] for n, r in results.items()
                       if n in ("query", "key", "value", "output"))
        ffn_mse = sum(r["mse"] for n, r in results.items()
                      if n in ("ffn_up", "ffn_down"))
        print(f"\n  FFN/Attention MSE ratio: {ffn_mse/attn_mse:.1f}×")

    _plot_sensitivity(all_results, save_plots)
    del model
    return all_results


def _plot_sensitivity(all_results, save_plots):
    """Figure 5.15: sublayer sensitivity bar chart."""
    setup_manning_style()
    sublayer_names = list(get_sublayer_paths().keys())
    colors = [MANNING_COLORS["L1"], MANNING_COLORS["L2"],
              MANNING_COLORS["L3"], MANNING_COLORS["L4"],
              MANNING_COLORS["purple"], MANNING_COLORS["gray"]]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0), sharey=True)
    for ax_idx, bits in enumerate([8, 4]):
        ax = axes[ax_idx]
        results = all_results[bits]
        mses = [results[n]["mse"] for n in sublayer_names]
        bars = ax.barh(range(len(sublayer_names)), mses,
                       color=colors, edgecolor="white", height=0.7)
        for bar, h in zip(bars, HATCHES):
            bar.set_hatch(h)
        ax.set_xlabel("Output MSE vs FP32")
        ax.set_title(f"INT{bits}", fontweight="bold")
        ax.set_yticks(range(len(sublayer_names)))
        if ax_idx == 0:
            ax.set_yticklabels([
                f"{n}" + (" [attn]" if n in ("query","key","value","output")
                          else " [ffn]") for n in sublayer_names])
        ax.ticklabel_format(axis="x", style="scientific", scilimits=(0,0))
        for i, (bar, v) in enumerate(zip(bars, mses)):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                    f"  {v:.2e}", va="center", fontsize=7)
        ax.axhline(y=3.5, color=MANNING_COLORS["dark"],
                    linewidth=0.8, linestyle="--", alpha=0.5)
    fig.tight_layout()
    save_figure(fig, "CH05_F15_sublayer_sensitivity", save_plots)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2: Selective QAT fine-tuning on SST-2                      (Figure 5.16)
# ═══════════════════════════════════════════════════════════════════════════════

def run_qat(save_plots=False, device="cpu", bits=4, epochs=3,
            train_size=2000, batch_size=32, lr=2e-5):
    """Fine-tune BERT on SST-2 with selective QAT strategies.

    Compares:
      - FP32 fine-tuning baseline
      - QAT with all 72 linear layers fake-quantized
      - QAT with attention-only (48 layers)
      - QAT with FFN-only (24 layers)

    Section 5.1 showed INT4 PTQ collapses BERT to 50.9% on SST-2
    (coin-flip on a binary task). This experiment tests whether
    QAT can recover that loss, and which selective strategy works best.
    """
    from transformers import BertForSequenceClassification, BertTokenizer

    print("\n" + "=" * 70)
    print(f"SELECTIVE QAT FINE-TUNING ON SST-2 (INT{bits})")
    print(f"Epochs: {epochs} | Train: {train_size} | LR: {lr}")
    print("=" * 70)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_ds, val_ds = load_sst2(tokenizer, max_train=train_size)
    train_loader = make_dataloader(train_ds, batch_size, shuffle=True)
    val_loader = make_dataloader(val_ds, batch_size)

    strategies = ["none", "all", "attn_only", "ffn_only"]
    strategy_labels = {
        "none": "FP32", "all": f"INT{bits} all",
        "attn_only": f"INT{bits} attn", "ffn_only": f"INT{bits} ffn",
    }
    all_histories = {}

    for strategy in strategies:
        label = strategy_labels[strategy]
        print(f"\n{'─' * 70}")
        print(f"Strategy: {label}")
        print(f"{'─' * 70}")

        # Fresh model for each strategy (same pretrained init)
        torch.manual_seed(42)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2).to(device)

        # Apply selective fake quantization
        n_quantized = apply_selective_qat(model, strategy, bits)
        print(f"  Quantized layers: {n_quantized}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                       weight_decay=0.01)

        history = {"train_loss": [], "val_acc": []}
        best_acc = 0.0

        for epoch in range(epochs):
            t0 = time.time()
            avg_loss = train_one_epoch(
                model, train_loader, optimizer, device)
            val_acc = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            history["train_loss"].append(avg_loss)
            history["val_acc"].append(val_acc)
            best_acc = max(best_acc, val_acc)

            print(f"  Epoch {epoch+1}/{epochs}  "
                  f"loss={avg_loss:.4f}  val_acc={val_acc:.2f}%  "
                  f"best={best_acc:.2f}%  [{elapsed:.1f}s]")

        history["best_acc"] = best_acc
        history["final_acc"] = history["val_acc"][-1]
        history["n_quantized"] = n_quantized
        all_histories[strategy] = history
        del model

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"SUMMARY: Selective QAT on SST-2 (INT{bits}, {epochs} epochs)")
    print(f"{'═' * 70}")
    print(f"  {'Strategy':<18} {'Layers':>6} {'Best Acc':>10} "
          f"{'Final Acc':>10} {'Δ vs FP32':>10}")
    print(f"  {'─' * 58}")

    fp32_best = all_histories["none"]["best_acc"]
    for strategy in strategies:
        h = all_histories[strategy]
        label = strategy_labels[strategy]
        delta = h["best_acc"] - fp32_best
        print(f"  {label:<18} {h['n_quantized']:>6} {h['best_acc']:>9.2f}% "
              f"{h['final_acc']:>9.2f}% {delta:>+9.2f}%")

    # Reference from Section 5.1
    print(f"\n  Reference (Section 5.1):")
    print(f"    INT{bits} PTQ accuracy: 50.92% (coin-flip)")
    print(f"{'═' * 70}")

    _plot_qat(all_histories, strategy_labels, bits, epochs, save_plots)
    return all_histories


def _plot_qat(all_histories, strategy_labels, bits, epochs, save_plots):
    """Figure 5.16: QAT accuracy curves and final comparison."""
    setup_manning_style()

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0))

    # Left: accuracy curves
    ax = axes[0]
    colors = [MANNING_COLORS["dark"], MANNING_COLORS["L4"],
              MANNING_COLORS["L1"], MANNING_COLORS["L2"]]
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D"]
    strategies = ["none", "all", "attn_only", "ffn_only"]

    for i, strategy in enumerate(strategies):
        h = all_histories[strategy]
        ax.plot(range(1, epochs + 1), h["val_acc"],
                color=colors[i], linestyle=linestyles[i],
                marker=markers[i], markersize=4,
                label=strategy_labels[strategy])

    # PTQ reference line
    ax.axhline(y=50.92, color=MANNING_COLORS["gray"],
               linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(0.6, 51.8, f"INT{bits} PTQ (50.9%)", fontsize=6,
            color=MANNING_COLORS["gray"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Accuracy per epoch", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xticks(range(1, epochs + 1))

    # Right: best accuracy bar chart
    ax = axes[1]
    labels = [strategy_labels[s] for s in strategies]
    bests = [all_histories[s]["best_acc"] for s in strategies]
    bars = ax.bar(range(len(strategies)), bests, color=colors,
                  edgecolor="white", width=0.6)
    for bar, h in zip(bars, HATCHES[:len(strategies)]):
        bar.set_hatch(h)

    ax.axhline(y=50.92, color=MANNING_COLORS["gray"],
               linewidth=0.8, linestyle=":", alpha=0.7)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(labels, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("Best accuracy (%)")
    ax.set_title("Best accuracy by strategy", fontsize=9)

    for bar, v in zip(bars, bests):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", fontsize=7)

    fig.tight_layout()
    save_figure(fig, "CH05_F16_selective_qat_sst2", save_plots)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Ch5 §5.5: Adapt Transformers Efficiently")
    parser.add_argument("--mode",
                        choices=["sensitivity", "qat", "all"],
                        default="all")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bits", type=int, default=4,
                        help="Target bit-width for QAT (default: 4)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="QAT fine-tuning epochs (default: 3)")
    parser.add_argument("--train-size", type=int, default=2000,
                        help="Training subset size (default: 2000)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("=" * 70)
    print("Chapter 5, Section 5.5: Adapt Transformers Efficiently")
    print("=" * 70)
    print(f"Device: {device}")

    if args.mode in ("sensitivity", "all"):
        run_sensitivity(save_plots=args.save_plots, device=device)

    if args.mode in ("qat", "all"):
        run_qat(save_plots=args.save_plots, device=device,
                bits=args.bits, epochs=args.epochs,
                train_size=args.train_size,
                batch_size=args.batch_size, lr=args.lr)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()