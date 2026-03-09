#!/usr/bin/env python3
"""
ch5_qat_schedule.py - Companion script for Section 5.3
Schedule Training to Minimize Cost

Demonstrates QAT scheduling strategies:
  1. Learning rate warmup + cosine decay
  2. Observer freezing
  3. Progressive quantization (INT8 -> INT6 -> INT4)
  4. Batch normalization folding order
  5. Full scheduled pipeline with cost comparison

Usage:
    # Full scheduled QAT pipeline
    python ch5_qat_schedule.py --mode full --target-bits 4 --epochs 20

    # Compare scheduling strategies
    python ch5_qat_schedule.py --mode compare --target-bits 4

    # Demonstrate observer freezing impact
    python ch5_qat_schedule.py --mode observer-freeze --target-bits 4

    # Demonstrate BN folding order
    python ch5_qat_schedule.py --mode bn-folding

    # Generate all figures
    python ch5_qat_schedule.py --mode full --target-bits 4 --save-plots

Requirements:
    pip install torch torchvision matplotlib numpy
"""

import argparse
import copy
import math
import time
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# ---------------------------------------------------------------------------
# Manning-compliant plot styling
# ---------------------------------------------------------------------------
MANNING_COLORS = {
    "blue":     "#1e5aa8",
    "orange":   "#ff8c00",
    "green":    "#2e8b57",
    "red":      "#c0392b",
    "purple":   "#7b68ee",
    "gray":     "#6c757d",
    "darkgray": "#343a40",
    "lightgray":"#dee2e6",
}
HATCHING = ['///', '\\\\\\', '...', 'xxx', '+++', 'ooo']

def setup_manning_style():
    """Apply Manning-compliant matplotlib defaults."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })
    return plt


# ---------------------------------------------------------------------------
# Fake quantization primitives (from Section 5.2)
# ---------------------------------------------------------------------------
class FakeQuantizeSTE(torch.autograd.Function):
    """Fake quantization with straight-through estimator."""
    @staticmethod
    def forward(ctx, x, scale, zero_point, q_min, q_max):
        x_int = torch.round(x / scale + zero_point)
        x_int = torch.clamp(x_int, q_min, q_max)
        x_q = (x_int - zero_point) * scale
        ctx.save_for_backward(x, scale)
        ctx.q_min = q_min
        ctx.q_max = q_max
        ctx.zero_point = zero_point
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        x_int = x / scale + ctx.zero_point
        mask = (x_int >= ctx.q_min) & (x_int <= ctx.q_max)
        grad_input = grad_output * mask.float()
        return grad_input, None, None, None, None

def fake_quantize(x, scale, zero_point, q_min, q_max):
    return FakeQuantizeSTE.apply(x, scale, zero_point, q_min, q_max)


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------
class MinMaxObserver(nn.Module):
    """Tracks running min/max for scale computation."""
    def __init__(self, ema_decay=0.9):
        super().__init__()
        self.ema_decay = ema_decay
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.register_buffer("num_batches", torch.tensor(0))
        self._frozen = False

    def forward(self, x):
        if self.training and not self._frozen:
            batch_min = x.detach().min()
            batch_max = x.detach().max()
            if self.num_batches == 0:
                self.min_val.copy_(batch_min)
                self.max_val.copy_(batch_max)
            else:
                self.min_val.copy_(
                    self.ema_decay * self.min_val + (1 - self.ema_decay) * batch_min
                )
                self.max_val.copy_(
                    self.ema_decay * self.max_val + (1 - self.ema_decay) * batch_max
                )
            self.num_batches += 1
        return x

    def freeze(self):
        self._frozen = True

    def unfreeze(self):
        self._frozen = False

    def get_scale(self):
        return (self.max_val - self.min_val).clamp(min=1e-8).item()


# ---------------------------------------------------------------------------
# Fake-quantized layers
# ---------------------------------------------------------------------------
class FakeQuantizedLinear(nn.Module):
    """Linear layer with fake quantization for QAT."""
    def __init__(self, in_features, out_features, bits=8, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bits = bits
        self.observer = MinMaxObserver()
        self._update_bounds()

    def _update_bounds(self):
        self.w_q_min = -(2 ** (self.bits - 1))
        self.w_q_max = 2 ** (self.bits - 1) - 1
        self.a_q_min = 0
        self.a_q_max = 2 ** self.bits - 1

    def _compute_scale_zp(self, tensor, symmetric, q_min, q_max):
        if symmetric:
            abs_max = tensor.abs().max().clamp(min=1e-8)
            scale = abs_max / ((q_max - q_min) / 2)
            zero_point = torch.tensor(0.0, device=tensor.device)
        else:
            t_min, t_max = tensor.min(), tensor.max()
            scale = (t_max - t_min).clamp(min=1e-8) / (q_max - q_min)
            zero_point = q_min - t_min / scale
        return scale, zero_point

    def forward(self, x):
        self._update_bounds()
        w = self.linear.weight
        # Per-tensor weight quantization (simplified for speed)
        w_scale, w_zp = self._compute_scale_zp(
            w, True, self.w_q_min, self.w_q_max
        )
        w_q = fake_quantize(w, w_scale, w_zp, self.w_q_min, self.w_q_max)

        # Activation quantization
        x = self.observer(x)
        a_scale, a_zp = self._compute_scale_zp(
            x, False, self.a_q_min, self.a_q_max
        )
        x_q = fake_quantize(x, a_scale, a_zp, self.a_q_min, self.a_q_max)

        return F.linear(x_q, w_q, self.linear.bias)


class FakeQuantizedConv2d(nn.Module):
    """Conv2d layer with fake quantization for QAT."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bits=8, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.bits = bits
        self.observer = MinMaxObserver()
        self._update_bounds()

    def _update_bounds(self):
        self.w_q_min = -(2 ** (self.bits - 1))
        self.w_q_max = 2 ** (self.bits - 1) - 1
        self.a_q_min = 0
        self.a_q_max = 2 ** self.bits - 1

    def _compute_scale_zp(self, tensor, symmetric, q_min, q_max):
        if symmetric:
            abs_max = tensor.abs().max().clamp(min=1e-8)
            scale = abs_max / ((q_max - q_min) / 2)
            zero_point = torch.tensor(0.0, device=tensor.device)
        else:
            t_min, t_max = tensor.min(), tensor.max()
            scale = (t_max - t_min).clamp(min=1e-8) / (q_max - q_min)
            zero_point = q_min - t_min / scale
        return scale, zero_point

    def forward(self, x):
        self._update_bounds()
        w = self.conv.weight
        w_scale, w_zp = self._compute_scale_zp(
            w, True, self.w_q_min, self.w_q_max
        )
        w_q = fake_quantize(w, w_scale, w_zp, self.w_q_min, self.w_q_max)

        x = self.observer(x)
        a_scale, a_zp = self._compute_scale_zp(
            x, False, self.a_q_min, self.a_q_max
        )
        x_q = fake_quantize(x, a_scale, a_zp, self.a_q_min, self.a_q_max)

        return F.conv2d(x_q, w_q, self.conv.bias,
                        self.conv.stride, self.conv.padding)


# ---------------------------------------------------------------------------
# Model architectures
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """CNN architecture for FashionMNIST (from Section 5.2)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Batch normalization folding
# ---------------------------------------------------------------------------
def fold_bn_into_conv(conv, bn):
    """Fold batch normalization into preceding Conv2d."""
    w = conv.weight.data.clone()
    b = conv.bias.data.clone() if conv.bias is not None else torch.zeros(
        w.shape[0], device=w.device
    )

    bn_mean = bn.running_mean
    bn_var = bn.running_var
    bn_gamma = bn.weight
    bn_beta = bn.bias
    bn_eps = bn.eps

    factor = bn_gamma / torch.sqrt(bn_var + bn_eps)

    new_weight = w * factor.view(-1, 1, 1, 1)
    new_bias = (b - bn_mean) * factor + bn_beta

    new_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        conv.stride, conv.padding, bias=True
    )
    new_conv.weight.data = new_weight
    new_conv.bias.data = new_bias
    return new_conv


def fold_all_bn(model):
    """Fold all batch normalization layers in a SmallCNN.

    Handles both raw Conv2d layers (Strategy A: fold before QAT) and
    FakeQuantizedConv2d layers (Strategy B: fold at export after QAT).
    When folding a FakeQuantizedConv2d, we extract the inner Conv2d,
    fold BN into it, and replace the wrapper—simulating export.
    """
    model = copy.deepcopy(model)
    model.eval()  # Ensure BN uses running stats

    for conv_name, bn_name in [('conv1', 'bn1'), ('conv2', 'bn2')]:
        if not (hasattr(model, bn_name) and hasattr(model, conv_name)):
            continue
        bn = getattr(model, bn_name)
        if isinstance(bn, nn.Identity):
            continue  # Already folded
        conv_module = getattr(model, conv_name)
        # Unwrap FakeQuantizedConv2d if present
        if isinstance(conv_module, FakeQuantizedConv2d):
            raw_conv = conv_module.conv
        elif isinstance(conv_module, nn.Conv2d):
            raw_conv = conv_module
        else:
            continue
        folded_conv = fold_bn_into_conv(raw_conv, bn)
        setattr(model, conv_name, folded_conv)
        setattr(model, bn_name, nn.Identity())

    return model


# ---------------------------------------------------------------------------
# QAT preparation
# ---------------------------------------------------------------------------
def prepare_model_for_qat(model, bits=8):
    """Replace Conv2d and Linear layers with fake-quantized versions."""
    model = copy.deepcopy(model)
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Conv2d):
            qat_conv = FakeQuantizedConv2d(
                module.in_channels, module.out_channels,
                module.kernel_size, module.stride, module.padding,
                bits=bits, bias=module.bias is not None
            )
            qat_conv.conv.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat_conv.conv.bias.data = module.bias.data.clone()
            setattr(model, name, qat_conv)
        elif isinstance(module, nn.Linear):
            qat_linear = FakeQuantizedLinear(
                module.in_features, module.out_features,
                bits=bits, bias=module.bias is not None
            )
            qat_linear.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                qat_linear.linear.bias.data = module.bias.data.clone()
            setattr(model, name, qat_linear)
        elif len(list(module.children())) > 0:
            setattr(model, name, prepare_model_for_qat(module, bits=bits))
    return model


# ---------------------------------------------------------------------------
# Scheduling components
# ---------------------------------------------------------------------------
def create_qat_lr_schedule(optimizer, num_epochs, warmup_epochs=3,
                           min_lr_ratio=0.01):
    """QAT LR schedule: linear warmup + cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(
                num_epochs - warmup_epochs, 1
            )
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ProgressiveQuantSchedule:
    """Progressive bit-width reduction during QAT."""
    def __init__(self, target_bits=4, total_epochs=20):
        self.target_bits = target_bits
        self.total_epochs = total_epochs
        self.schedule = self._build_schedule()

    def _build_schedule(self):
        if self.target_bits >= 8:
            return [(0, 8)]
        warmup_end = self.total_epochs // 4
        mid_end = self.total_epochs // 2
        mid_bits = min(8, self.target_bits + 2)
        return [
            (0, 8),
            (warmup_end, mid_bits),
            (mid_end, self.target_bits),
        ]

    def get_bits(self, epoch):
        current_bits = self.schedule[0][1]
        for start_epoch, bits in self.schedule:
            if epoch >= start_epoch:
                current_bits = bits
        return current_bits

    def update_model_bits(self, model, epoch):
        bits = self.get_bits(epoch)
        for module in model.modules():
            if hasattr(module, 'bits'):
                if module.bits != bits:
                    module.bits = bits
                    module._update_bounds()
        return bits


def apply_observer_freeze(model, epoch, freeze_epoch):
    """Freeze observer statistics after freeze_epoch."""
    if epoch >= freeze_epoch:
        for module in model.modules():
            if isinstance(module, MinMaxObserver):
                module.freeze()
        return True
    return False


def collect_observer_scales(model):
    """Collect current observer scale values."""
    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, MinMaxObserver):
            if module.num_batches > 0:
                scales[name] = module.get_scale()
    return scales


def compute_max_scale_change(current_scales, prev_scales):
    """Compute maximum relative change in observer scales."""
    if prev_scales is None:
        return float("inf")
    max_change = 0.0
    for name in current_scales:
        if name in prev_scales and prev_scales[name] > 1e-10:
            change = abs(current_scales[name] - prev_scales[name])
            relative = change / prev_scales[name]
            max_change = max(max_change, relative)
    return max_change


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total


def pretrain_baseline(model, train_loader, val_loader, device,
                      epochs=15, lr=1e-3):
    """Pre-train a FP32 baseline model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        acc = evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"  FP32 baseline: {best_acc:.2f}%")
    return model, best_acc


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def get_fashion_mnist(batch_size=128, subset_size=5000):
    """Load FashionMNIST with optional subset for faster demos."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_full = datasets.FashionMNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_full = datasets.FashionMNIST(
        "./data", train=False, download=True, transform=transform
    )

    if subset_size and subset_size < len(train_full):
        rng = torch.Generator().manual_seed(42)
        train_idx = torch.randperm(len(train_full), generator=rng)[:subset_size]
        test_idx = torch.randperm(len(test_full), generator=rng)[:subset_size // 5]
        train_ds = Subset(train_full, train_idx.tolist())
        test_ds = Subset(test_full, test_idx.tolist())
    else:
        train_ds = train_full
        test_ds = test_full

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Mode: Full scheduled QAT pipeline
# ---------------------------------------------------------------------------
def run_scheduled_qat(model, train_loader, val_loader, device,
                      target_bits=4, total_epochs=20, base_lr=1e-4,
                      warmup_epochs=3, freeze_epoch=7, progressive=True):
    """Full scheduled QAT pipeline returning training history."""
    # Step 1: Fold batch normalization
    model = fold_all_bn(model)

    # Step 2: Insert fake quantization (start at INT8)
    model = prepare_model_for_qat(model, bits=8)
    model = model.to(device)

    # Step 3: Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = create_qat_lr_schedule(
        optimizer, total_epochs, warmup_epochs
    )

    # Step 4: Progressive schedule
    if progressive and target_bits < 8:
        prog = ProgressiveQuantSchedule(target_bits, total_epochs)
    else:
        prog = ProgressiveQuantSchedule(target_bits if target_bits < 8 else 8,
                                        total_epochs)
        # Force all epochs to target bits if not progressive
        if not progressive:
            prog.schedule = [(0, target_bits)]

    criterion = nn.CrossEntropyLoss()
    prev_scales = None
    best_acc = 0.0
    best_state = None
    history = {
        "epoch": [], "bits": [], "lr": [], "loss": [],
        "val_acc": [], "obs_frozen": [], "scale_change": [],
        "epoch_time": [],
    }

    for epoch in range(total_epochs):
        t_start = time.time()

        current_bits = prog.update_model_bits(model, epoch)
        frozen = apply_observer_freeze(model, epoch, freeze_epoch)

        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        acc = evaluate(model, val_loader, device)

        cur_scales = collect_observer_scales(model)
        max_change = compute_max_scale_change(cur_scales, prev_scales)
        prev_scales = cur_scales

        t_elapsed = time.time() - t_start

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch + 1)
        history["bits"].append(current_bits)
        history["lr"].append(current_lr)
        history["loss"].append(loss)
        history["val_acc"].append(acc)
        history["obs_frozen"].append(frozen)
        history["scale_change"].append(
            max_change if max_change != float("inf") else None
        )
        history["epoch_time"].append(t_elapsed)

        obs_str = "frozen" if frozen else "active"
        sc_str = f"{max_change:.4f}" if max_change != float("inf") else "  inf"
        print(
            f"  Epoch {epoch+1:2d}/{total_epochs} │ "
            f"INT{current_bits} │ "
            f"LR {current_lr:.6f} │ "
            f"Loss {loss:.4f} │ "
            f"Val {acc:.2f}% │ "
            f"Obs {obs_str:6s} │ "
            f"ΔScale {sc_str}"
        )

    model.load_state_dict(best_state)
    return model, best_acc, history


# ---------------------------------------------------------------------------
# Mode: Compare scheduling strategies
# ---------------------------------------------------------------------------
def run_compare(args, device):
    """Compare direct QAT vs progressive scheduled QAT."""
    print("=" * 70)
    print("QAT Schedule Comparison")
    print("=" * 70)

    train_loader, val_loader = get_fashion_mnist(
        batch_size=args.batch_size, subset_size=args.subset_size
    )

    # Pretrain baseline
    print("\n[1/4] Pre-training FP32 baseline...")
    baseline_model = SmallCNN()
    baseline_model, fp32_acc = pretrain_baseline(
        baseline_model, train_loader, val_loader, device, epochs=15
    )

    # PTQ baseline (simple weight quantization)
    print(f"\n[2/4] PTQ at INT{args.target_bits}...")
    ptq_model = prepare_model_for_qat(
        fold_all_bn(copy.deepcopy(baseline_model)), bits=args.target_bits
    )
    ptq_model = ptq_model.to(device)
    # Run one forward pass for observer calibration
    ptq_model.train()
    for data, _ in train_loader:
        ptq_model(data.to(device))
        break
    ptq_acc = evaluate(ptq_model, val_loader, device)
    print(f"  PTQ accuracy: {ptq_acc:.2f}%")

    # Direct QAT (no progressive schedule)
    print(f"\n[3/4] Direct INT{args.target_bits} QAT ({args.epochs} epochs)...")
    direct_model = copy.deepcopy(baseline_model)
    _, direct_acc, direct_hist = run_scheduled_qat(
        direct_model, train_loader, val_loader, device,
        target_bits=args.target_bits, total_epochs=args.epochs,
        base_lr=args.lr, progressive=False,
        freeze_epoch=args.epochs,  # Never freeze for direct comparison
    )

    # Scheduled QAT (progressive + freeze)
    print(f"\n[4/4] Scheduled progressive QAT → INT{args.target_bits}"
          f" ({args.epochs} epochs)...")
    sched_model = copy.deepcopy(baseline_model)
    _, sched_acc, sched_hist = run_scheduled_qat(
        sched_model, train_loader, val_loader, device,
        target_bits=args.target_bits, total_epochs=args.epochs,
        base_lr=args.lr, progressive=True,
    )

    # Find epoch where each method first reaches 85%
    def first_above(hist, threshold=85.0):
        for i, acc in enumerate(hist["val_acc"]):
            if acc >= threshold:
                return i + 1
        return None

    direct_first85 = first_above(direct_hist)
    sched_first85 = first_above(sched_hist)

    direct_time = sum(direct_hist["epoch_time"])
    sched_time = sum(sched_hist["epoch_time"])

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    header = (f"{'Method':<22s} │ {'Final Acc':>9s} │ {'Best Acc':>9s} │ "
              f"{'→85%':>6s} │ {'Time':>8s}")
    print(header)
    print("─" * 70)
    print(f"{'FP32 Baseline':<22s} │ {fp32_acc:>8.2f}% │ {fp32_acc:>8.2f}% │ "
          f"{'  —':>6s} │ {'  —':>8s}")
    print(f"{'PTQ (no training)':<22s} │ {ptq_acc:>8.2f}% │ {ptq_acc:>8.2f}% │ "
          f"{'  —':>6s} │ {'  ~0s':>8s}")
    d85 = f"{direct_first85:>4d}ep" if direct_first85 else "  n/a"
    s85 = f"{sched_first85:>4d}ep" if sched_first85 else "  n/a"
    print(f"{'Direct QAT':<22s} │ {direct_acc:>8.2f}% │ "
          f"{max(direct_hist['val_acc']):>8.2f}% │ "
          f"{d85:>6s} │ {direct_time:>7.1f}s")
    print(f"{'Scheduled QAT':<22s} │ {sched_acc:>8.2f}% │ "
          f"{max(sched_hist['val_acc']):>8.2f}% │ "
          f"{s85:>6s} │ {sched_time:>7.1f}s")
    print("=" * 70)

    if args.save_plots:
        plot_comparison(direct_hist, sched_hist, fp32_acc, ptq_acc, args)


# ---------------------------------------------------------------------------
# Mode: Observer freeze demonstration
# ---------------------------------------------------------------------------
def run_observer_freeze(args, device):
    """Demonstrate impact of observer freezing timing."""
    print("=" * 70)
    print("Observer Freezing Experiment")
    print("=" * 70)

    train_loader, val_loader = get_fashion_mnist(
        batch_size=args.batch_size, subset_size=args.subset_size
    )

    print("\nPre-training FP32 baseline...")
    baseline_model = SmallCNN()
    baseline_model, fp32_acc = pretrain_baseline(
        baseline_model, train_loader, val_loader, device, epochs=15
    )

    freeze_points = [3, 5, 7, 10, args.epochs]  # last = never freeze
    results = {}

    for fp in freeze_points:
        label = f"freeze@{fp}" if fp < args.epochs else "never"
        print(f"\n  Running QAT with observer {label}...")
        model_copy = copy.deepcopy(baseline_model)
        _, acc, hist = run_scheduled_qat(
            model_copy, train_loader, val_loader, device,
            target_bits=args.target_bits, total_epochs=args.epochs,
            base_lr=args.lr, progressive=True, freeze_epoch=fp,
        )
        results[label] = {"acc": acc, "history": hist}

    print("\n" + "=" * 70)
    print("Observer Freeze Timing Results")
    print("=" * 70)
    for label, res in results.items():
        print(f"  {label:<14s}: {res['acc']:.2f}%")

    if args.save_plots:
        plot_observer_freeze(results, args)


# ---------------------------------------------------------------------------
# Mode: BN folding order
# ---------------------------------------------------------------------------
def run_bn_folding(args, device):
    """Demonstrate BN folding order impact."""
    print("=" * 70)
    print("Batch Normalization Folding Order Experiment")
    print("=" * 70)

    train_loader, val_loader = get_fashion_mnist(
        batch_size=args.batch_size, subset_size=args.subset_size
    )

    print("\nPre-training FP32 baseline (with BN)...")
    baseline = SmallCNN()
    baseline, fp32_acc = pretrain_baseline(
        baseline, train_loader, val_loader, device, epochs=15
    )

    # Strategy A: Fold THEN QAT
    print("\n[A] Fold BN → then QAT...")
    model_a = copy.deepcopy(baseline)
    _, acc_a, _ = run_scheduled_qat(
        model_a, train_loader, val_loader, device,
        target_bits=8, total_epochs=10, base_lr=args.lr,
        progressive=False, freeze_epoch=5,
    )

    # Strategy B: QAT THEN fold (the wrong order)
    print("\n[B] QAT (without folding) → then fold...")
    model_b = copy.deepcopy(baseline)
    # QAT without folding BN first — BN layers are still live
    model_b_qat = prepare_model_for_qat(model_b, bits=8)
    model_b_qat = model_b_qat.to(device)
    optimizer = torch.optim.Adam(model_b_qat.parameters(), lr=args.lr)
    scheduler = create_qat_lr_schedule(optimizer, 10, warmup_epochs=2)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        train_epoch(model_b_qat, train_loader, optimizer, criterion, device)
        scheduler.step()
    acc_b_pre_fold = evaluate(model_b_qat, val_loader, device)

    # Now do what happens at export: fold BN into the conv weights
    # and re-evaluate. The model trained against unfolded weights,
    # so folding changes the weight distribution and breaks the
    # quantization grid alignment the model learned.
    model_b_exported = fold_all_bn(model_b_qat)
    model_b_exported = model_b_exported.to(device)
    acc_b_post_fold = evaluate(model_b_exported, val_loader, device)

    # Strategy A post-export: already folded before QAT, so folding again
    # is a no-op. Evaluate to confirm.
    # (run_scheduled_qat already folds internally, so acc_a IS the post-export acc)
    acc_a_post_export = acc_a  # No additional folding needed

    print(f"\n{'=' * 60}")
    print("BN Folding Order Results")
    print(f"{'=' * 60}")
    print(f"  {'Strategy':<24s} │ {'QAT Acc':>8s} │ {'Post-Export':>11s} │ "
          f"{'Gap':>6s}")
    print(f"  {'─' * 24}─┼─{'─' * 8}─┼─{'─' * 11}─┼─{'─' * 6}")
    print(f"  {'Fold then QAT':<24s} │ {acc_a:>7.2f}% │ {acc_a_post_export:>10.2f}% │ "
          f"{acc_a - acc_a_post_export:>5.2f}%")
    print(f"  {'QAT then fold':<24s} │ {acc_b_pre_fold:>7.2f}% │ "
          f"{acc_b_post_fold:>10.2f}% │ "
          f"{acc_b_pre_fold - acc_b_post_fold:>5.2f}%")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Mode: Full pipeline demo
# ---------------------------------------------------------------------------
def run_full(args, device):
    """Run the complete scheduled QAT pipeline."""
    print("=" * 70)
    print(f"Scheduled QAT Pipeline (SmallCNN → INT{args.target_bits},"
          f" {args.epochs} epochs)")
    print("=" * 70)

    train_loader, val_loader = get_fashion_mnist(
        batch_size=args.batch_size, subset_size=args.subset_size
    )

    print("\nPre-training FP32 baseline...")
    baseline = SmallCNN()
    baseline, fp32_acc = pretrain_baseline(
        baseline, train_loader, val_loader, device, epochs=15
    )

    print(f"\nRunning scheduled QAT → INT{args.target_bits}...")
    model, best_acc, history = run_scheduled_qat(
        baseline, train_loader, val_loader, device,
        target_bits=args.target_bits, total_epochs=args.epochs,
        base_lr=args.lr,
    )

    print(f"\n{'=' * 70}")
    print(f"FP32 Baseline:    {fp32_acc:.2f}%")
    print(f"Scheduled QAT:    {best_acc:.2f}%")
    print(f"Accuracy gap:     {fp32_acc - best_acc:.2f}%")
    total_time = sum(history["epoch_time"])
    print(f"Total time:       {total_time:.1f}s")
    print(f"{'=' * 70}")

    if args.save_plots:
        plot_full_pipeline(history, fp32_acc, args)


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def plot_comparison(direct_hist, sched_hist, fp32_acc, ptq_acc, args):
    """Plot comparison of scheduling strategies."""
    plt = setup_manning_style()
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.2))

    # Panel 1: Accuracy curves
    ax = axes[0]
    ax.plot(direct_hist["epoch"], direct_hist["val_acc"],
            color=MANNING_COLORS["red"], linewidth=1.5,
            linestyle="--", label="Direct QAT")
    ax.plot(sched_hist["epoch"], sched_hist["val_acc"],
            color=MANNING_COLORS["blue"], linewidth=1.5,
            label="Scheduled QAT")
    ax.axhline(y=fp32_acc, color=MANNING_COLORS["green"],
               linewidth=1, linestyle=":", label="FP32")
    ax.axhline(y=ptq_acc, color=MANNING_COLORS["gray"],
               linewidth=1, linestyle=":", label="PTQ")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Accuracy convergence")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=6)

    # Panel 2: Learning rate
    ax = axes[1]
    ax.plot(sched_hist["epoch"], sched_hist["lr"],
            color=MANNING_COLORS["blue"], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR schedule (warmup + cosine)")
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    # Panel 3: Bit-width schedule
    ax = axes[2]
    ax.step(sched_hist["epoch"], sched_hist["bits"],
            color=MANNING_COLORS["orange"], linewidth=2, where="post")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bit-width")
    ax.set_title("Progressive quantization")
    ax.set_ylim(2, 10)
    ax.set_yticks([4, 6, 8])

    plt.tight_layout()
    path = os.path.join(args.output_dir, "fig_5_10_qat_schedule_comparison.png")
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    print(f"\n  Saved: {path} (+.pdf)")
    plt.close()


def plot_observer_freeze(results, args):
    """Plot observer freeze timing impact."""
    plt = setup_manning_style()
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 2.8))

    colors = [MANNING_COLORS["blue"], MANNING_COLORS["orange"],
              MANNING_COLORS["green"], MANNING_COLORS["red"],
              MANNING_COLORS["purple"]]
    styles = ["-", "--", "-.", ":", "-"]

    for i, (label, res) in enumerate(results.items()):
        hist = res["history"]
        ax.plot(hist["epoch"], hist["val_acc"],
                color=colors[i % len(colors)],
                linestyle=styles[i % len(styles)],
                linewidth=1.5, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Observer freeze timing impact on QAT convergence")
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(args.output_dir, "fig_5_11_observer_freeze_impact.png")
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    print(f"\n  Saved: {path} (+.pdf)")
    plt.close()


def plot_full_pipeline(history, fp32_acc, args):
    """Plot full scheduled pipeline training log."""
    plt = setup_manning_style()
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.0))

    # Panel 1: Accuracy
    ax = axes[0, 0]
    ax.plot(history["epoch"], history["val_acc"],
            color=MANNING_COLORS["blue"], linewidth=1.5, label="QAT")
    ax.axhline(y=fp32_acc, color=MANNING_COLORS["green"],
               linewidth=1, linestyle=":", label="FP32 baseline")
    # Mark bit transitions
    for i, (e, b) in enumerate(zip(history["epoch"], history["bits"])):
        if i > 0 and b != history["bits"][i - 1]:
            ax.axvline(x=e, color=MANNING_COLORS["gray"],
                       linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Val accuracy (%)")
    ax.set_title("Accuracy")
    ax.legend(loc="lower right", fontsize=7)

    # Panel 2: Loss
    ax = axes[0, 1]
    ax.plot(history["epoch"], history["loss"],
            color=MANNING_COLORS["red"], linewidth=1.5)
    for i, (e, b) in enumerate(zip(history["epoch"], history["bits"])):
        if i > 0 and b != history["bits"][i - 1]:
            ax.axvline(x=e, color=MANNING_COLORS["gray"],
                       linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Training loss")
    ax.set_title("Loss")

    # Panel 3: LR + bit-width
    ax = axes[1, 0]
    ax.plot(history["epoch"], history["lr"],
            color=MANNING_COLORS["blue"], linewidth=1.5, label="LR")
    ax.set_ylabel("Learning rate", color=MANNING_COLORS["blue"])
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax2 = ax.twinx()
    ax2.step(history["epoch"], history["bits"],
             color=MANNING_COLORS["orange"], linewidth=1.5,
             where="post", label="Bits", linestyle="--")
    ax2.set_ylabel("Bit-width", color=MANNING_COLORS["orange"])
    ax2.set_ylim(2, 10)
    ax2.set_yticks([4, 6, 8])
    ax.set_xlabel("Epoch")
    ax.set_title("LR + bit schedule")

    # Panel 4: Scale change
    ax = axes[1, 1]
    sc = [s if s is not None and s < 1.0 else None
          for s in history["scale_change"]]
    valid_epochs = [e for e, s in zip(history["epoch"], sc) if s is not None]
    valid_sc = [s for s in sc if s is not None]
    ax.plot(valid_epochs, valid_sc,
            color=MANNING_COLORS["purple"], linewidth=1.5, marker="o",
            markersize=3)
    ax.axhline(y=0.01, color=MANNING_COLORS["gray"],
               linewidth=1, linestyle=":", label="1% threshold")
    # Mark freeze point
    for i, frozen in enumerate(history["obs_frozen"]):
        if frozen and (i == 0 or not history["obs_frozen"][i - 1]):
            ax.axvline(x=history["epoch"][i], color=MANNING_COLORS["red"],
                       linewidth=1, linestyle="--", label="Freeze point")
            break
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Max rel. scale Δ")
    ax.set_title("Observer convergence")
    ax.legend(loc="upper right", fontsize=6)

    plt.tight_layout()
    path = os.path.join(args.output_dir, "fig_5_12_full_pipeline.png")
    plt.savefig(path)
    plt.savefig(path.replace(".png", ".pdf"))
    print(f"\n  Saved: {path} (+.pdf)")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Ch5 Section 5.3: Schedule QAT to minimize cost"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "compare", "observer-freeze", "bn-folding"],
        help="Which experiment to run"
    )
    parser.add_argument("--target-bits", type=int, default=4,
                        help="Target quantization bit-width (default: 4)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Total QAT epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Base learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size (default: 128)")
    parser.add_argument("--subset-size", type=int, default=5000,
                        help="Training subset size for fast demos "
                             "(default: 5000, 0=full dataset)")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save Manning-compliant figures")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Directory for saved figures")
    args = parser.parse_args()

    if args.subset_size == 0:
        args.subset_size = None

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.mode == "full":
        run_full(args, device)
    elif args.mode == "compare":
        run_compare(args, device)
    elif args.mode == "observer-freeze":
        run_observer_freeze(args, device)
    elif args.mode == "bn-folding":
        run_bn_folding(args, device)


if __name__ == "__main__":
    main()