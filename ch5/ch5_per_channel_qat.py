#!/usr/bin/env python3
"""
Chapter 5, Section 5.4: Per-Channel Strategies for Convolution and Linear Layers

Companion script demonstrating per-channel QAT strategies. Shows why per-channel
fake quantization matters during training (not just inference), how scales evolve
per-channel, and the measurable accuracy impact on real architectures.

Usage:
    python ch5_per_channel_qat.py --mode gradient-analysis
    python ch5_per_channel_qat.py --mode scale-evolution
    python ch5_per_channel_qat.py --mode accuracy-compare
    python ch5_per_channel_qat.py --mode conv-axis
    python ch5_per_channel_qat.py --mode all
    python ch5_per_channel_qat.py --mode all --save-plots

Requirements:
    pip install torch torchvision matplotlib
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# ============================================================
# Manning-compliant plot styling
# ============================================================
MANNING_COLORS = {
    'blue': '#3574A7',
    'orange': '#E87B2F',
    'green': '#429E45',
    'red': '#C63535',
    'purple': '#8B5FB3',
    'gray': '#6B6B6B',
    'light_blue': '#A8C8E8',
    'light_orange': '#F4C08E',
}

def setup_manning_style():
    """Apply Manning-compliant matplotlib styling."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 150,
        'figure.figsize': (5.6, 3.5),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    return plt


# ============================================================
# Core fake quantizers: per-tensor vs per-channel
# ============================================================

class FakeQuantizePerTensor(torch.autograd.Function):
    """Per-tensor fake quantization with STE."""
    @staticmethod
    def forward(ctx, x, scale, q_min, q_max):
        x_q = torch.clamp(torch.round(x / scale), q_min, q_max)
        x_dq = x_q * scale
        # Save mask for STE: gradients pass only where not clipped
        mask = (x_q > q_min) & (x_q < q_max)
        ctx.save_for_backward(mask)
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask.float(), None, None, None


class FakeQuantizePerChannel(torch.autograd.Function):
    """Per-channel fake quantization with STE along output channel axis."""
    @staticmethod
    def forward(ctx, x, scales, q_min, q_max, axis):
        # Reshape scales for broadcasting
        shape = [1] * x.dim()
        shape[axis] = -1
        scales_bc = scales.view(shape)

        x_q = torch.clamp(torch.round(x / scales_bc), q_min, q_max)
        x_dq = x_q * scales_bc
        mask = (x_q > q_min) & (x_q < q_max)
        ctx.save_for_backward(mask)
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask.float(), None, None, None, None


def compute_per_tensor_scale(weight, bits=8):
    """Compute per-tensor scale from weight abs max."""
    q_max = (1 << (bits - 1)) - 1
    abs_max = weight.detach().abs().max()
    return abs_max / q_max if abs_max > 0 else torch.tensor(1.0)


def compute_per_channel_scales(weight, bits=8, axis=0):
    """Compute per-channel scales along given axis."""
    q_max = (1 << (bits - 1)) - 1
    # Reshape to [num_channels, -1] and take max per channel
    if axis == 0:
        flat = weight.detach().reshape(weight.shape[0], -1)
    else:
        # For axis=1, transpose first
        perm = list(range(weight.dim()))
        perm[0], perm[axis] = perm[axis], perm[0]
        flat = weight.detach().permute(perm).reshape(weight.shape[axis], -1)
    abs_max = flat.abs().amax(dim=1)
    scales = abs_max / q_max
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    return scales


# ============================================================
# Fake-quantized layers with granularity control
# ============================================================

class FQLinear(nn.Module):
    """Linear layer with configurable per-tensor or per-channel fake quantization."""
    def __init__(self, in_features, out_features, bits=8, per_channel=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bits = bits
        self.per_channel = per_channel
        self.q_max = (1 << (bits - 1)) - 1
        self.q_min = -self.q_max

    def forward(self, x):
        w = self.linear.weight
        if self.per_channel:
            scales = compute_per_channel_scales(w, self.bits, axis=0)
            w_fq = FakeQuantizePerChannel.apply(
                w, scales, self.q_min, self.q_max, 0)
        else:
            scale = compute_per_tensor_scale(w, self.bits)
            w_fq = FakeQuantizePerTensor.apply(
                w, scale, self.q_min, self.q_max)
        return nn.functional.linear(x, w_fq, self.linear.bias)


class FQConv2d(nn.Module):
    """Conv2d with configurable per-tensor or per-channel fake quantization."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 padding=0, bits=8, per_channel=False, quant_axis=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=padding)
        self.bits = bits
        self.per_channel = per_channel
        self.quant_axis = quant_axis  # 0=output channels, 1=input channels
        self.q_max = (1 << (bits - 1)) - 1
        self.q_min = -self.q_max

    def forward(self, x):
        w = self.conv.weight
        if self.per_channel:
            scales = compute_per_channel_scales(w, self.bits, axis=self.quant_axis)
            w_fq = FakeQuantizePerChannel.apply(
                w, scales, self.q_min, self.q_max, self.quant_axis)
        else:
            scale = compute_per_tensor_scale(w, self.bits)
            w_fq = FakeQuantizePerTensor.apply(
                w, scale, self.q_min, self.q_max)
        return nn.functional.conv2d(x, w_fq, self.conv.bias,
                                    self.conv.stride, self.conv.padding)


# ============================================================
# Model definitions
# ============================================================

class SmallCNN(nn.Module):
    """Small CNN for FashionMNIST with configurable quantization granularity."""
    def __init__(self, num_classes=10, bits=8, per_channel=False):
        super().__init__()
        self.conv1 = FQConv2d(1, 32, 3, padding=1,
                              bits=bits, per_channel=per_channel)
        self.conv2 = FQConv2d(32, 64, 3, padding=1,
                              bits=bits, per_channel=per_channel)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = FQLinear(64 * 7 * 7, 128, bits=bits, per_channel=per_channel)
        self.fc2 = FQLinear(128, num_classes, bits=bits, per_channel=per_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================
# Data loading
# ============================================================

def get_fashion_mnist(subset_size=5000, batch_size=128):
    """Load FashionMNIST with optional subset for faster experiments."""
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_set = datasets.FashionMNIST(
        './data', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(
        './data', train=False, download=True, transform=transform)

    if subset_size and subset_size < len(train_set):
        torch.manual_seed(42)
        indices = torch.randperm(len(train_set))[:subset_size]
        train_set = torch.utils.data.Subset(train_set, indices)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ============================================================
# Experiment 1: Gradient flow analysis
# ============================================================

def gradient_analysis(save_plots=False):
    """
    Compare gradient flow through per-tensor vs per-channel fake quantization.
    Shows that per-channel preserves more gradients (fewer clipped channels).
    """
    print("=" * 60)
    print("EXPERIMENT 1: Gradient Flow — Per-Tensor vs Per-Channel")
    print("=" * 60)

    # Use ResNet-18's first conv for a real-world weight distribution
    from torchvision.models import resnet18
    model = resnet18(weights='DEFAULT')
    weight = model.conv1.weight.detach().clone()  # [64, 3, 7, 7]
    weight.requires_grad = True

    print(f"\nLayer: ResNet-18 conv1, shape {list(weight.shape)}")
    print(f"  64 output channels × (3 × 7 × 7) = {weight.numel()} parameters")

    bits = 4  # INT4 makes the difference dramatic
    q_max = (1 << (bits - 1)) - 1
    q_min = -q_max

    # --- Per-tensor path ---
    w_pt = weight.clone().detach().requires_grad_(True)
    scale_pt = compute_per_tensor_scale(w_pt, bits)
    w_pt_fq = FakeQuantizePerTensor.apply(w_pt, scale_pt, q_min, q_max)
    loss_pt = w_pt_fq.pow(2).sum()  # dummy loss
    loss_pt.backward()
    grad_pt = w_pt.grad.clone()

    # --- Per-channel path ---
    w_pc = weight.clone().detach().requires_grad_(True)
    scales_pc = compute_per_channel_scales(w_pc, bits, axis=0)
    w_pc_fq = FakeQuantizePerChannel.apply(w_pc, scales_pc, q_min, q_max, 0)
    loss_pc = w_pc_fq.pow(2).sum()
    loss_pc.backward()
    grad_pc = w_pc.grad.clone()

    # Analyze per-channel gradient survival
    print(f"\nINT{bits} fake quantization gradient analysis:")
    print("-" * 55)

    # Per-channel: fraction of gradients that are zero (clipped by STE)
    per_ch_grad_pt = grad_pt.reshape(64, -1)
    per_ch_grad_pc = grad_pc.reshape(64, -1)
    zero_frac_pt = (per_ch_grad_pt == 0).float().mean(dim=1)
    zero_frac_pc = (per_ch_grad_pc == 0).float().mean(dim=1)

    print(f"Per-tensor:  avg {zero_frac_pt.mean():.1%} gradients clipped per channel")
    print(f"Per-channel: avg {zero_frac_pc.mean():.1%} gradients clipped per channel")
    print(f"Gradient survival improvement: {(1 - zero_frac_pc.mean()) / (1 - zero_frac_pt.mean()):.2f}×")

    # Show worst-affected channels
    print(f"\nWorst 5 channels (% gradients clipped):")
    print(f"  {'Channel':>8}  {'Per-tensor':>12}  {'Per-channel':>12}")
    worst_pt = zero_frac_pt.argsort(descending=True)[:5]
    for ch in worst_pt:
        print(f"  {ch.item():>8}  {zero_frac_pt[ch]:.1%}{' ':>6}  {zero_frac_pc[ch]:.1%}")

    # Gradient magnitude analysis
    grad_mag_pt = per_ch_grad_pt.abs().mean(dim=1)
    grad_mag_pc = per_ch_grad_pc.abs().mean(dim=1)
    print(f"\nGradient magnitude per channel:")
    print(f"  Per-tensor  — mean: {grad_mag_pt.mean():.6f}, "
          f"std: {grad_mag_pt.std():.6f}, min: {grad_mag_pt.min():.6f}")
    print(f"  Per-channel — mean: {grad_mag_pc.mean():.6f}, "
          f"std: {grad_mag_pc.std():.6f}, min: {grad_mag_pc.min():.6f}")

    # Scale comparison
    print(f"\nScale analysis:")
    print(f"  Per-tensor: 1 scale = {scale_pt.item():.6f}")
    print(f"  Per-channel: {len(scales_pc)} scales, "
          f"range [{scales_pc.min():.6f}, {scales_pc.max():.6f}]")
    print(f"  Scale ratio (max/min): {scales_pc.max() / scales_pc.min():.1f}×")

    if save_plots:
        plt = setup_manning_style()
        fig, axes = plt.subplots(1, 3, figsize=(5.6, 3.0))
        fig.subplots_adjust(wspace=0.45, bottom=0.22)

        channels = np.arange(64)

        # Panel 1: Clipped gradient fraction per channel
        h1 = axes[0].bar(channels, zero_frac_pt.numpy(), alpha=0.7, width=0.8,
                         color=MANNING_COLORS['blue'])
        h2 = axes[0].bar(channels, zero_frac_pc.numpy(), alpha=0.7, width=0.4,
                         color=MANNING_COLORS['orange'])
        axes[0].set_xlabel('Output channel')
        axes[0].set_ylabel('Frac. gradients clipped')
        axes[0].set_title('STE clipping (INT4)', fontsize=9, pad=4)
        axes[0].set_xlim(-1, 64)

        # Panel 2: Per-channel scales — annotate dashed line directly
        axes[1].bar(channels, scales_pc.numpy(), color=MANNING_COLORS['green'],
                    alpha=0.8, width=0.8)
        pt_y = scale_pt.item()
        axes[1].axhline(y=pt_y, color=MANNING_COLORS['red'],
                        linestyle='--', linewidth=1.5)
        axes[1].annotate('Per-tensor scale', xy=(32, pt_y),
                         fontsize=7, color=MANNING_COLORS['red'],
                         ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.15',
                                   fc='white', ec='none', alpha=0.85))
        axes[1].set_xlabel('Output channel')
        axes[1].set_ylabel('Scale value')
        axes[1].set_title('Per-channel scales vs\nper-tensor scale', fontsize=9, pad=4)
        axes[1].set_xlim(-1, 64)

        # Panel 3: Gradient magnitude per channel
        axes[2].scatter(channels, grad_mag_pt.numpy(), s=14, alpha=0.7,
                       color=MANNING_COLORS['blue'],
                       zorder=3, edgecolors='none')
        axes[2].scatter(channels, grad_mag_pc.numpy(), s=14, alpha=0.7,
                       color=MANNING_COLORS['orange'],
                       zorder=3, edgecolors='none')
        axes[2].set_xlabel('Output channel')
        axes[2].set_ylabel('Mean |gradient|')
        axes[2].set_title('Gradient magnitude', fontsize=9, pad=4)
        axes[2].set_xlim(-1, 64)

        # Shared legend below all panels
        fig.legend([h1, h2], ['Per-tensor', 'Per-channel'],
                   loc='lower center', ncol=2, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, 0.02))

        plt.savefig('fig_5_12_gradient_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig('fig_5_12_gradient_analysis.pdf', bbox_inches='tight')
        print("\n→ Saved fig_5_12_gradient_analysis.png/pdf")
        plt.close()

    return zero_frac_pt, zero_frac_pc


# ============================================================
# Experiment 2: Per-channel scale evolution during QAT
# ============================================================

def scale_evolution(save_plots=False):
    """
    Track how per-channel scales evolve during QAT training.
    Shows that channels adapt independently.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Per-Channel Scale Evolution During QAT")
    print("=" * 60)

    train_loader, test_loader = get_fashion_mnist(subset_size=5000)
    bits = 4

    model = SmallCNN(bits=bits, per_channel=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Track scales every epoch for conv1 (32 output channels)
    epochs = 15
    scale_history = []

    print(f"\nTraining SmallCNN (per-channel INT{bits}) for {epochs} epochs...")
    print(f"Tracking conv1 scales: 32 output channels\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Record per-channel scales for conv1
        with torch.no_grad():
            w = model.conv1.conv.weight
            scales = compute_per_channel_scales(w, bits, axis=0)
            scale_history.append(scales.clone().numpy())

        if epoch % 3 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / len(train_loader)
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for bx, by in test_loader:
                    pred = model(bx).argmax(dim=1)
                    correct += (pred == by).sum().item()
                    total += by.size(0)
            acc = correct / total * 100
            print(f"  Epoch {epoch:2d}: loss={avg_loss:.4f}, acc={acc:.1f}%, "
                  f"scale range=[{scales.min():.5f}, {scales.max():.5f}]")

    scale_history = np.array(scale_history)  # [epochs, 32]

    # Analysis: how much do scales change?
    initial_scales = scale_history[0]
    final_scales = scale_history[-1]
    relative_change = np.abs(final_scales - initial_scales) / (initial_scales + 1e-10)
    print(f"\nScale evolution (conv1, 32 output channels):")
    print(f"  Mean relative change: {relative_change.mean():.1%}")
    print(f"  Max relative change:  {relative_change.max():.1%} (channel {relative_change.argmax()})")
    print(f"  Min relative change:  {relative_change.min():.1%} (channel {relative_change.argmin()})")

    # Coefficient of variation across channels at each epoch
    cv = scale_history.std(axis=1) / scale_history.mean(axis=1)
    print(f"  Scale CV (cross-channel): epoch 0 = {cv[0]:.3f}, epoch {epochs-1} = {cv[-1]:.3f}")

    if save_plots:
        plt = setup_manning_style()
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))

        # Panel 1: Scale trajectories for all 32 channels
        colors = plt.cm.viridis(np.linspace(0, 1, 32))
        for ch in range(32):
            axes[0].plot(range(epochs), scale_history[:, ch],
                        color=colors[ch], alpha=0.5, linewidth=0.8)
        # Highlight channels with largest and smallest final scales
        max_ch = final_scales.argmax()
        min_ch = final_scales.argmin()
        axes[0].plot(range(epochs), scale_history[:, max_ch],
                    color=MANNING_COLORS['red'], linewidth=2,
                    label=f'Ch {max_ch} (max)')
        axes[0].plot(range(epochs), scale_history[:, min_ch],
                    color=MANNING_COLORS['blue'], linewidth=2,
                    label=f'Ch {min_ch} (min)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Scale value')
        axes[0].set_title('Conv1 per-channel scale evolution')
        axes[0].legend(loc='best')

        # Panel 2: Scale distribution at epoch 0 vs final
        channels = np.arange(32)
        w = 0.35
        axes[1].bar(channels - w/2, initial_scales, w, alpha=0.7,
                   color=MANNING_COLORS['blue'], label='Epoch 0')
        axes[1].bar(channels + w/2, final_scales, w, alpha=0.7,
                   color=MANNING_COLORS['orange'], label=f'Epoch {epochs-1}')
        axes[1].set_xlabel('Output channel')
        axes[1].set_ylabel('Scale value')
        axes[1].set_title('Scale redistribution during QAT')
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('fig_5_13_scale_evolution.png', dpi=300, bbox_inches='tight')
        plt.savefig('fig_5_13_scale_evolution.pdf', bbox_inches='tight')
        print("\n→ Saved fig_5_13_scale_evolution.png/pdf")
        plt.close()

    return scale_history


# ============================================================
# Experiment 3: Accuracy comparison — per-tensor vs per-channel QAT
# ============================================================

def accuracy_compare(save_plots=False):
    """
    Train SmallCNN with per-tensor vs per-channel QAT and compare accuracy.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Per-Tensor vs Per-Channel QAT Accuracy")
    print("=" * 60)

    train_loader, test_loader = get_fashion_mnist(subset_size=5000)
    epochs = 15
    bits = 4

    results = {}

    for name, per_channel in [('per-tensor', False), ('per-channel', True)]:
        print(f"\nTraining SmallCNN with {name} INT{bits} QAT...")
        torch.manual_seed(42)
        model = SmallCNN(bits=bits, per_channel=per_channel)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        train_accs, test_accs, losses = [], [], []

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            correct_train, total_train = 0, 0
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                correct_train += (out.argmax(1) == by).sum().item()
                total_train += by.size(0)

            losses.append(epoch_loss / len(train_loader))
            train_accs.append(correct_train / total_train * 100)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for bx, by in test_loader:
                    pred = model(bx).argmax(1)
                    correct += (pred == by).sum().item()
                    total += by.size(0)
            test_accs.append(correct / total * 100)

            if epoch % 3 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch:2d}: loss={losses[-1]:.4f}, "
                      f"test_acc={test_accs[-1]:.2f}%")

        results[name] = {
            'train_accs': train_accs,
            'test_accs': test_accs,
            'losses': losses,
            'best_acc': max(test_accs),
            'final_acc': test_accs[-1],
        }

    # Also train FP32 baseline
    print(f"\nTraining FP32 baseline...")
    torch.manual_seed(42)

    class BaselineCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    fp32_model = BaselineCNN()
    optimizer = optim.Adam(fp32_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    fp32_accs = []
    for epoch in range(epochs):
        fp32_model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(fp32_model(bx), by)
            loss.backward()
            optimizer.step()
        fp32_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                correct += (fp32_model(bx).argmax(1) == by).sum().item()
                total += by.size(0)
        fp32_accs.append(correct / total * 100)
    results['FP32'] = {'test_accs': fp32_accs, 'best_acc': max(fp32_accs),
                       'final_acc': fp32_accs[-1]}
    print(f"  FP32 best: {max(fp32_accs):.2f}%")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  {'Method':<20} {'Best Acc':>10} {'Final Acc':>10} {'Gap vs FP32':>12}")
    print(f"  {'-'*52}")
    fp32_best = results['FP32']['best_acc']
    for name in ['FP32', 'per-tensor', 'per-channel']:
        r = results[name]
        gap = r['best_acc'] - fp32_best
        print(f"  {name:<20} {r['best_acc']:>9.2f}% {r['final_acc']:>9.2f}% {gap:>+11.2f}%")
    print(f"{'='*60}")
    print(f"  Per-channel advantage: "
          f"{results['per-channel']['best_acc'] - results['per-tensor']['best_acc']:+.2f}%")

    if save_plots:
        plt = setup_manning_style()
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.5))

        ep = range(epochs)
        axes[0].plot(ep, results['FP32']['test_accs'], '--',
                    color=MANNING_COLORS['gray'], linewidth=1.5, label='FP32')
        axes[0].plot(ep, results['per-tensor']['test_accs'],
                    color=MANNING_COLORS['blue'], linewidth=1.5, label='Per-tensor QAT')
        axes[0].plot(ep, results['per-channel']['test_accs'],
                    color=MANNING_COLORS['orange'], linewidth=1.5, label='Per-channel QAT')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Test accuracy (%)')
        axes[0].set_title(f'INT{bits} QAT: Per-tensor vs per-channel')
        axes[0].legend(loc='lower right')

        axes[1].plot(ep, results['per-tensor']['losses'],
                    color=MANNING_COLORS['blue'], linewidth=1.5, label='Per-tensor')
        axes[1].plot(ep, results['per-channel']['losses'],
                    color=MANNING_COLORS['orange'], linewidth=1.5, label='Per-channel')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Training loss')
        axes[1].set_title(f'INT{bits} QAT training loss')
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('fig_5_14_accuracy_compare.png', dpi=300, bbox_inches='tight')
        plt.savefig('fig_5_14_accuracy_compare.pdf', bbox_inches='tight')
        print("\n→ Saved fig_5_14_accuracy_compare.png/pdf")
        plt.close()

    return results


# ============================================================
# Experiment 4: Conv axis question — which dimension to quantize along
# ============================================================

def conv_axis_analysis(save_plots=False):
    """
    Demonstrate the conv weight axis question: output channels (axis 0)
    vs input channels (axis 1). Uses ResNet-18 conv layers.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Convolution Quantization Axis Analysis")
    print("=" * 60)

    from torchvision.models import resnet18
    model = resnet18(weights='DEFAULT')

    layers = [
        ('conv1',          model.conv1.weight),           # [64, 3, 7, 7]
        ('layer1.0.conv1', model.layer1[0].conv1.weight), # [64, 64, 3, 3]
        ('layer2.0.conv1', model.layer2[0].conv1.weight), # [128, 64, 3, 3]
        ('layer3.0.conv1', model.layer3[0].conv1.weight), # [256, 128, 3, 3]
        ('layer4.0.conv1', model.layer4[0].conv1.weight), # [512, 256, 3, 3]
    ]

    print(f"\nComparing axis 0 (output channels) vs axis 1 (input channels)")
    print(f"{'Layer':<20} {'Shape':<20} {'MSE axis=0':>12} {'MSE axis=1':>12} {'Winner':>8}")
    print("-" * 75)

    bits = 4
    q_max = (1 << (bits - 1)) - 1
    q_min = -q_max

    axis0_mses, axis1_mses = [], []
    layer_names = []

    for name, weight in layers:
        # Axis 0: per output channel
        scales_0 = compute_per_channel_scales(weight, bits, axis=0)
        shape_0 = [1] * weight.dim()
        shape_0[0] = -1
        w_fq_0 = torch.clamp(torch.round(weight / scales_0.view(shape_0)),
                              q_min, q_max) * scales_0.view(shape_0)
        mse_0 = ((weight - w_fq_0) ** 2).mean().item()

        # Axis 1: per input channel
        scales_1 = compute_per_channel_scales(weight, bits, axis=1)
        shape_1 = [1] * weight.dim()
        shape_1[1] = -1
        w_fq_1 = torch.clamp(torch.round(weight / scales_1.view(shape_1)),
                              q_min, q_max) * scales_1.view(shape_1)
        mse_1 = ((weight - w_fq_1) ** 2).mean().item()

        winner = "axis=0" if mse_0 <= mse_1 else "axis=1"
        ratio = max(mse_0, mse_1) / min(mse_0, mse_1)
        print(f"{name:<20} {str(list(weight.shape)):<20} "
              f"{mse_0:>12.2e} {mse_1:>12.2e} {winner:>8} ({ratio:.1f}×)")

        axis0_mses.append(mse_0)
        axis1_mses.append(mse_1)
        layer_names.append(name)

    print(f"\n  Axis 0 (output channel) wins in {sum(a0 <= a1 for a0, a1 in zip(axis0_mses, axis1_mses))}/{len(layers)} layers")
    print(f"  This confirms the standard convention: quantize along the output channel axis.")
    print(f"\n  Why? Each output channel computes one independent feature via a dot product")
    print(f"  over all input channels. Giving each feature its own scale ensures the full")
    print(f"  integer range represents that feature's learned weight distribution.")

    # Also show why depthwise conv is special
    print(f"\n  Special case — depthwise convolution:")
    # Simulate depthwise conv weight
    dw_weight = torch.randn(128, 1, 3, 3) * 0.1
    dw_weight[0] *= 10  # Make one channel an outlier
    scales_dw_0 = compute_per_channel_scales(dw_weight, bits, axis=0)
    shape_dw = [1] * dw_weight.dim(); shape_dw[0] = -1
    w_fq_dw = torch.clamp(torch.round(dw_weight / scales_dw_0.view(shape_dw)),
                           q_min, q_max) * scales_dw_0.view(shape_dw)
    mse_dw = ((dw_weight - w_fq_dw) ** 2).mean().item()
    print(f"  Depthwise [128, 1, 3, 3]: axis 0 = axis 1 (only 1 input channel per group)")
    print(f"  MSE: {mse_dw:.2e}  |  128 scales for 128 independent filters")

    if save_plots:
        plt = setup_manning_style()
        fig, ax = plt.subplots(1, 1, figsize=(5.6, 2.5))

        x = np.arange(len(layers))
        w = 0.35
        ax.bar(x - w/2, axis0_mses, w, color=MANNING_COLORS['blue'],
               label='Axis 0 (output ch.)', alpha=0.8)
        ax.bar(x + w/2, axis1_mses, w, color=MANNING_COLORS['orange'],
               label='Axis 1 (input ch.)', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('.', '\n.') for n in layer_names],
                           fontsize=7)
        ax.set_ylabel('MSE (INT4)')
        ax.set_title('Quantization MSE by axis — ResNet-18')
        ax.legend()
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig('fig_5_15_conv_axis.png', dpi=300, bbox_inches='tight')
        plt.savefig('fig_5_15_conv_axis.pdf', bbox_inches='tight')
        print("\n→ Saved fig_5_15_conv_axis.png/pdf")
        plt.close()

    return axis0_mses, axis1_mses


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Chapter 5.4: Per-Channel QAT Strategies')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['gradient-analysis', 'scale-evolution',
                                 'accuracy-compare', 'conv-axis', 'all'],
                        help='Experiment to run')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save Manning-compliant figures')
    parser.add_argument('--bits', type=int, default=4,
                        help='Bit-width for experiments (default: 4)')
    args = parser.parse_args()

    print("Chapter 5.4: Per-Channel Strategies for Convolution and Linear Layers")
    print("=" * 65)

    if args.mode in ('gradient-analysis', 'all'):
        gradient_analysis(save_plots=args.save_plots)

    if args.mode in ('scale-evolution', 'all'):
        scale_evolution(save_plots=args.save_plots)

    if args.mode in ('accuracy-compare', 'all'):
        accuracy_compare(save_plots=args.save_plots)

    if args.mode in ('conv-axis', 'all'):
        conv_axis_analysis(save_plots=args.save_plots)

    print("\nDone.")


if __name__ == '__main__':
    main()
