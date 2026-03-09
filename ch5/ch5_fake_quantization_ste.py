#!/usr/bin/env python3
"""
Chapter 5.2 Companion Script: Fake Quantization and Straight-Through Estimators

This script demonstrates the core machinery of Quantization-Aware Training (QAT):
- The zero-gradient problem with true quantization
- Straight-through estimator (STE) implementation
- Fake quantization operators
- Observer-based scale computation
- Complete QAT training on real CNN models
- Weight distribution evolution visualization

Usage:
    # Run all demonstrations
    python ch5_fake_quantization_ste.py --all

    # Individual experiments
    python ch5_fake_quantization_ste.py --demo-gradient-problem
    python ch5_fake_quantization_ste.py --demo-ste-variants
    python ch5_fake_quantization_ste.py --visualize-clipped-ste --save-plots
    python ch5_fake_quantization_ste.py --visualize-weight-adaptation --save-plots
    python ch5_fake_quantization_ste.py --qat-cnn --bits 4 --epochs 20

    # Save visualizations as PNG and editable PDF
    python ch5_fake_quantization_ste.py --all --save-plots

Requirements:
    pip install torch torchvision matplotlib numpy tqdm
"""

import argparse
import copy
import math
import os
import sys
from typing import Tuple, Optional, Dict, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from tqdm import tqdm

# Optional imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization disabled.")

try:
    import torchvision
    import torchvision.transforms as transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("Warning: torchvision not available. Using synthetic data.")


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def save_figure(fig, base_path: str, formats: List[str] = ['png', 'pdf']):
    """Save figure in multiple formats (PNG for viewing, PDF for editing)."""
    for fmt in formats:
        path = f"{base_path}.{fmt}"
        fig.savefig(path, dpi=150 if fmt == 'png' else 300, 
                   bbox_inches='tight', format=fmt)
        print(f"   Saved: {path}")


# =============================================================================
# SECTION 1: The Gradient Problem
# =============================================================================

def demonstrate_gradient_problem():
    """
    Listing 5.6: Demonstrating the zero-gradient problem
    
    Shows that true quantization gradients are zero, preventing learning.
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION: The Zero-Gradient Problem")
    print("=" * 70)
    
    # Example 1: Basic gradient through round()
    print("\n1. Gradient through torch.round():")
    print("-" * 40)
    
    x = torch.tensor([0.3, 0.7, 1.2, 1.8], requires_grad=True)
    scale = 0.5
    
    # Forward pass with true quantization
    q = torch.round(x / scale) * scale
    
    print(f"   Input:     {[f'{v:.2f}' for v in x.detach().tolist()]}")
    print(f"   Scale:     {scale}")
    print(f"   Quantized: {[f'{v:.1f}' for v in q.detach().tolist()]}")
    
    # Backward pass
    loss = q.sum()
    loss.backward()
    
    print(f"   Gradient:  {x.grad.tolist()}")
    print(f"   ⚠ All gradients are ZERO - no learning signal!")
    
    # Example 2: Effect on a simple optimization
    print("\n2. Effect on optimization (100 steps):")
    print("-" * 40)
    
    x_opt = torch.tensor([0.3], requires_grad=True)
    target = torch.tensor([1.5])
    optimizer = torch.optim.SGD([x_opt], lr=0.1)
    
    initial_x = x_opt.item()
    
    for _ in range(100):
        optimizer.zero_grad()
        q_opt = torch.round(x_opt / scale) * scale
        loss = (q_opt - target) ** 2
        loss.backward()
        optimizer.step()
    
    final_x = x_opt.item()
    
    print(f"   Target:    {target.item()}")
    print(f"   Initial x: {initial_x:.4f}")
    print(f"   Final x:   {final_x:.4f}")
    print(f"   ⚠ x didn't move! Zero gradients prevent optimization.")
    
    return {"initial": initial_x, "final": final_x, "target": target.item()}


# =============================================================================
# SECTION 2: Straight-Through Estimator Implementation
# =============================================================================

class FakeQuantizeSTE(torch.autograd.Function):
    """
    Listing 5.7: Fake quantization with straight-through estimator
    
    Forward: true quantization (round, clamp, scale)
    Backward: STE (gradient = 1 for unclipped, 0 for clipped)
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, 
                zero_point: torch.Tensor, q_min: int, q_max: int) -> torch.Tensor:
        # Quantize: scale to integer range, round, clip
        x_int = torch.round(x / scale + zero_point)
        x_int_clamped = torch.clamp(x_int, q_min, q_max)
        
        # Dequantize: back to floating point
        x_q = (x_int_clamped - zero_point) * scale
        
        # Save for backward (needed for clipping gradient mask)
        ctx.save_for_backward(x_int, torch.tensor([q_min, q_max]))
        
        return x_q
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_int, bounds = ctx.saved_tensors
        q_min, q_max = bounds[0].item(), bounds[1].item()
        
        # Clipped STE: pass gradient through, but zero for clipped values
        mask = (x_int >= q_min) & (x_int <= q_max)
        grad_input = grad_output * mask.float()
        
        return grad_input, None, None, None, None


def fake_quantize(x: torch.Tensor, scale: torch.Tensor, 
                  zero_point: torch.Tensor, q_min: int, q_max: int) -> torch.Tensor:
    """Convenience wrapper for fake quantization with STE."""
    return FakeQuantizeSTE.apply(x, scale, zero_point, q_min, q_max)


class BasicSTE(torch.autograd.Function):
    """Basic STE: gradient = 1 everywhere (no clipping awareness)."""
    
    @staticmethod
    def forward(ctx, x, scale, zero_point, q_min, q_max):
        x_int = torch.round(x / scale + zero_point)
        x_int = torch.clamp(x_int, q_min, q_max)
        return (x_int - zero_point) * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def demonstrate_ste_variants():
    """
    Listing 5.8 & 5.13: Verify STE gradient flow and compare variants
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION: STE Variants Comparison")
    print("=" * 70)
    
    scale = torch.tensor(0.5)
    zero_point = torch.tensor(0.0)
    q_min, q_max = -8, 7  # INT4 symmetric
    
    test_values = [-5.0, -2.0, 0.0, 1.5, 3.0, 5.0]
    
    print(f"\nQuantization config: scale={scale.item()}, INT4 range=[{q_min}, {q_max}]")
    print(f"Representable range: [{q_min * scale.item()}, {q_max * scale.item()}]")
    print("\n" + "-" * 70)
    print(f"{'Input':>8} | {'Quantized':>10} | {'Basic STE':>10} | {'Clipped STE':>12} | {'Status':>10}")
    print("-" * 70)
    
    results = []
    
    for val in test_values:
        x = torch.tensor([val], requires_grad=True)
        
        # Forward pass
        q = fake_quantize(x, scale, zero_point, q_min, q_max)
        
        # Backward with clipped STE
        loss = q.sum()
        loss.backward()
        clipped_grad = x.grad.item()
        
        # Backward with basic STE
        x_basic = torch.tensor([val], requires_grad=True)
        q_basic = BasicSTE.apply(x_basic, scale, zero_point, q_min, q_max)
        q_basic.sum().backward()
        basic_grad = x_basic.grad.item()
        
        # Determine if clipped
        x_int = val / scale.item() + zero_point.item()
        is_clipped = x_int < q_min or x_int > q_max
        status = "CLIPPED" if is_clipped else "OK"
        
        results.append({
            'input': val,
            'quantized': q.item(),
            'basic_grad': basic_grad,
            'clipped_grad': clipped_grad,
            'is_clipped': is_clipped
        })
        
        print(f"{val:>8.2f} | {q.item():>10.2f} | {basic_grad:>10.2f} | {clipped_grad:>12.2f} | {status:>10}")
    
    print("-" * 70)
    print("\nKey insight: Clipped STE zeros gradient for out-of-range values,")
    print("preventing the optimizer from pushing them further out of range.")
    
    return results


def demonstrate_ste_optimization():
    """Show that STE enables optimization where true gradients fail."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: STE Enables Optimization")
    print("=" * 70)
    
    scale = torch.tensor(0.5)
    zero_point = torch.tensor(0.0)
    q_min, q_max = -8, 7
    target = torch.tensor([1.5])
    
    x_ste = torch.tensor([0.3], requires_grad=True)
    optimizer_ste = torch.optim.SGD([x_ste], lr=0.1)
    
    print("\nOptimizing x to make quantized(x) ≈ 1.5")
    print(f"Initial x = 0.3, target = 1.5")
    print("\n" + "-" * 50)
    
    history = []
    for step in range(20):
        optimizer_ste.zero_grad()
        q = fake_quantize(x_ste, scale, zero_point, q_min, q_max)
        loss = (q - target) ** 2
        loss.backward()
        optimizer_ste.step()
        
        history.append({
            'step': step,
            'x': x_ste.item(),
            'q': q.item(),
            'loss': loss.item(),
            'grad': x_ste.grad.item() if x_ste.grad is not None else 0
        })
        
        if step < 5 or step >= 15:
            print(f"Step {step:2d}: x={x_ste.item():6.3f}, q(x)={q.item():5.2f}, "
                  f"loss={loss.item():.4f}, grad={x_ste.grad.item():+.3f}")
        elif step == 5:
            print("   ...")
    
    print("-" * 50)
    print(f"\n✓ STE enabled x to move from 0.3 to {x_ste.item():.3f}")
    print(f"  Quantized value reached {q.item():.2f} (target: 1.5)")
    
    return history


# =============================================================================
# SECTION 3: Observer Pattern
# =============================================================================

class MinMaxObserver(nn.Module):
    """Tracks running min/max for scale computation."""
    
    def __init__(self, averaging_method: str = 'ema', ema_decay: float = 0.9):
        super().__init__()
        self.averaging_method = averaging_method
        self.ema_decay = ema_decay
        
        self.register_buffer('min_val', torch.tensor(float('inf')))
        self.register_buffer('max_val', torch.tensor(float('-inf')))
        self.register_buffer('num_batches', torch.tensor(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_min = x.detach().min()
            batch_max = x.detach().max()
            
            if self.averaging_method == 'ema':
                if self.num_batches == 0:
                    self.min_val.copy_(batch_min)
                    self.max_val.copy_(batch_max)
                else:
                    self.min_val.copy_(
                        self.ema_decay * self.min_val + 
                        (1 - self.ema_decay) * batch_min
                    )
                    self.max_val.copy_(
                        self.ema_decay * self.max_val + 
                        (1 - self.ema_decay) * batch_max
                    )
            else:
                self.min_val.copy_(torch.min(self.min_val, batch_min))
                self.max_val.copy_(torch.max(self.max_val, batch_max))
            
            self.num_batches += 1
        
        return x
    
    def compute_qparams(self, symmetric: bool = False, 
                        bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        if symmetric:
            q_max = 2 ** (bits - 1) - 1
            abs_max = torch.max(self.min_val.abs(), self.max_val.abs())
            scale = abs_max / q_max
            zero_point = torch.tensor(0, device=scale.device)
        else:
            q_min, q_max = 0, 2 ** bits - 1
            scale = (self.max_val - self.min_val) / (q_max - q_min)
            zero_point = q_min - torch.round(self.min_val / scale.clamp(min=1e-8))
        
        return scale.clamp(min=1e-8), zero_point
    
    def reset(self):
        self.min_val.fill_(float('inf'))
        self.max_val.fill_(float('-inf'))
        self.num_batches.zero_()


# =============================================================================
# SECTION 4: Fake-Quantized Layers
# =============================================================================

class FakeQuantizedLinear(nn.Module):
    """Linear layer with fake quantization for QAT."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 bits: int = 8, weight_symmetric: bool = True,
                 activation_symmetric: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bits = bits
        self.weight_symmetric = weight_symmetric
        self.activation_symmetric = activation_symmetric
        
        if weight_symmetric:
            self.w_q_min = -(2 ** (bits - 1))
            self.w_q_max = 2 ** (bits - 1) - 1
        else:
            self.w_q_min = 0
            self.w_q_max = 2 ** bits - 1
        
        if activation_symmetric:
            self.a_q_min = -(2 ** (bits - 1))
            self.a_q_max = 2 ** (bits - 1) - 1
        else:
            self.a_q_min = 0
            self.a_q_max = 2 ** bits - 1
        
        self.activation_observer = MinMaxObserver(averaging_method='ema')
    
    def _compute_weight_qparams_per_channel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.linear.weight
        
        if self.weight_symmetric:
            abs_max = w.abs().max(dim=1)[0].clamp(min=1e-8)
            scale = abs_max / self.w_q_max
            zero_point = torch.zeros_like(scale)
        else:
            w_min = w.min(dim=1)[0]
            w_max = w.max(dim=1)[0]
            scale = (w_max - w_min).clamp(min=1e-8) / (self.w_q_max - self.w_q_min)
            zero_point = self.w_q_min - torch.round(w_min / scale)
        
        return scale, zero_point
    
    def _fake_quantize_weights(self) -> torch.Tensor:
        w = self.linear.weight
        scale, zero_point = self._compute_weight_qparams_per_channel()
        
        w_q = torch.zeros_like(w)
        for i in range(w.shape[0]):
            w_q[i] = fake_quantize(
                w[i], scale[i], zero_point[i], self.w_q_min, self.w_q_max
            )
        
        return w_q
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self._fake_quantize_weights()
        
        if self.training:
            self.activation_observer(x)
        
        a_scale, a_zp = self.activation_observer.compute_qparams(
            symmetric=self.activation_symmetric, bits=self.bits
        )
        x_q = fake_quantize(x, a_scale, a_zp, self.a_q_min, self.a_q_max)
        
        return F.linear(x_q, w_q, self.linear.bias)


class FakeQuantizedConv2d(nn.Module):
    """Convolutional layer with fake quantization for QAT."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True,
                 bits: int = 8, weight_symmetric: bool = True,
                 activation_symmetric: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bits = bits
        self.weight_symmetric = weight_symmetric
        self.activation_symmetric = activation_symmetric
        
        if weight_symmetric:
            self.w_q_min = -(2 ** (bits - 1))
            self.w_q_max = 2 ** (bits - 1) - 1
        else:
            self.w_q_min = 0
            self.w_q_max = 2 ** bits - 1
        
        if activation_symmetric:
            self.a_q_min = -(2 ** (bits - 1))
            self.a_q_max = 2 ** (bits - 1) - 1
        else:
            self.a_q_min = 0
            self.a_q_max = 2 ** bits - 1
        
        self.activation_observer = MinMaxObserver(averaging_method='ema')
    
    def _fake_quantize_weights(self) -> torch.Tensor:
        w = self.conv.weight  # [out_channels, in_channels, H, W]
        w_flat = w.view(w.shape[0], -1)
        
        if self.weight_symmetric:
            abs_max = w_flat.abs().max(dim=1)[0].clamp(min=1e-8)
            scale = abs_max / self.w_q_max
            zero_point = torch.zeros_like(scale)
        else:
            w_min = w_flat.min(dim=1)[0]
            w_max = w_flat.max(dim=1)[0]
            scale = (w_max - w_min).clamp(min=1e-8) / (self.w_q_max - self.w_q_min)
            zero_point = self.w_q_min - torch.round(w_min / scale)
        
        w_q = torch.zeros_like(w)
        for i in range(w.shape[0]):
            w_q[i] = fake_quantize(
                w[i], scale[i], zero_point[i], self.w_q_min, self.w_q_max
            )
        
        return w_q
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = self._fake_quantize_weights()
        
        if self.training:
            self.activation_observer(x)
        
        a_scale, a_zp = self.activation_observer.compute_qparams(
            symmetric=self.activation_symmetric, bits=self.bits
        )
        x_q = fake_quantize(x, a_scale, a_zp, self.a_q_min, self.a_q_max)
        
        return F.conv2d(x_q, w_q, self.conv.bias, 
                       self.conv.stride, self.conv.padding)


# =============================================================================
# SECTION 5: Numerical Stability
# =============================================================================

class StableFakeQuantize(nn.Module):
    """Numerically stable fake quantization with safeguards."""
    
    def __init__(self, bits: int = 8, symmetric: bool = True,
                 eps: float = 1e-8, max_scale: float = 1e6):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.eps = eps
        self.max_scale = max_scale
        
        if symmetric:
            self.q_min = -(2 ** (bits - 1))
            self.q_max = 2 ** (bits - 1) - 1
        else:
            self.q_min = 0
            self.q_max = 2 ** bits - 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        abs_max = x.abs().max()
        if abs_max < self.eps:
            return x
        
        if self.symmetric:
            scale = abs_max / self.q_max
        else:
            x_min, x_max = x.min(), x.max()
            range_val = x_max - x_min
            if range_val < self.eps:
                return x
            scale = range_val / (self.q_max - self.q_min)
        
        scale = scale.clamp(min=self.eps, max=self.max_scale)
        
        if self.symmetric:
            zero_point = torch.tensor(0.0, device=x.device)
        else:
            zero_point = self.q_min - torch.round(x_min / scale)
            zero_point = zero_point.clamp(self.q_min, self.q_max)
        
        return fake_quantize(x, scale, zero_point, self.q_min, self.q_max)


def demonstrate_numerical_stability():
    """Demonstrate numerical stability safeguards."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Numerical Stability Safeguards")
    print("=" * 70)
    
    stable_fq = StableFakeQuantize(bits=8, symmetric=True)
    
    test_cases = [
        ("Normal tensor", torch.randn(10) * 2),
        ("Near-zero tensor", torch.randn(10) * 1e-10),
        ("Constant tensor", torch.ones(10) * 5.0),
        ("Mixed with outlier", torch.cat([torch.randn(9), torch.tensor([1e6])])),
    ]
    
    print("\n" + "-" * 70)
    for name, tensor in test_cases:
        result = stable_fq(tensor)
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        status = "✓ OK" if not (has_nan or has_inf) else "✗ FAIL"
        
        print(f"\n{name}:")
        print(f"   Input range:  [{tensor.min().item():.2e}, {tensor.max().item():.2e}]")
        print(f"   Output range: [{result.min().item():.2e}, {result.max().item():.2e}]")
        print(f"   Status: {status}")
    
    print("-" * 70)


# =============================================================================
# SECTION 6: Clipped STE Gradient Visualization
# =============================================================================

def visualize_clipped_ste_gradient(bits: int = 4, scale: float = 0.5,
                                    save_path: Optional[str] = None):
    """
    Figure 5.8: Visualize the clipped STE gradient as a function of input value.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualization.")
        return None
    
    print("\n" + "=" * 70)
    print(f"VISUALIZATION: Clipped STE Gradient Function (INT{bits})")
    print("=" * 70)
    
    q_min = -(2 ** (bits - 1))
    q_max = 2 ** (bits - 1) - 1
    float_min = q_min * scale
    float_max = q_max * scale
    
    print(f"\nQuantization parameters:")
    print(f"   Bit-width: {bits}")
    print(f"   Scale: {scale}")
    print(f"   Integer range: [{q_min}, {q_max}]")
    print(f"   Float range: [{float_min}, {float_max}]")
    
    x_range = np.linspace(float_min * 1.5, float_max * 1.5, 1000)
    
    gradients_clipped = []
    gradients_basic = []
    quantized_values = []
    
    for x_val in x_range:
        x = torch.tensor([x_val], requires_grad=True)
        scale_t = torch.tensor(scale)
        zero_point = torch.tensor(0.0)
        
        q = fake_quantize(x, scale_t, zero_point, q_min, q_max)
        q.backward()
        gradients_clipped.append(x.grad.item())
        quantized_values.append(q.item())
        gradients_basic.append(1.0)
    
    gradients_clipped = np.array(gradients_clipped)
    gradients_basic = np.array(gradients_basic)
    quantized_values = np.array(quantized_values)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top plot: Quantization function
    ax1 = axes[0]
    ax1.plot(x_range, x_range, 'b--', alpha=0.5, linewidth=1.5, label='Identity (no quantization)')
    ax1.plot(x_range, quantized_values, 'b-', linewidth=2, label='Quantized output')
    ax1.axvspan(float_min, float_max, alpha=0.2, color='green', label='Quantizable range')
    ax1.axvspan(x_range.min(), float_min, alpha=0.2, color='red')
    ax1.axvspan(float_max, x_range.max(), alpha=0.2, color='red', label='Clipped range')
    
    grid_points = np.arange(q_min, q_max + 1) * scale
    for gp in grid_points:
        ax1.axhline(gp, color='gray', alpha=0.3, linestyle=':', linewidth=0.5)
    
    ax1.set_ylabel('Output Value', fontsize=12)
    ax1.set_title(f'Fake Quantization Function (INT{bits}, scale={scale})', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_range.min(), x_range.max())
    
    # Bottom plot: Gradient comparison
    ax2 = axes[1]
    ax2.plot(x_range, gradients_basic, 'r--', linewidth=2, alpha=0.7, 
             label='Basic STE (gradient = 1 everywhere)')
    ax2.fill_between(x_range, 0, gradients_clipped, alpha=0.3, color='green',
                     where=(gradients_clipped > 0.5))
    ax2.plot(x_range, gradients_clipped, 'g-', linewidth=2.5, 
             label='Clipped STE (gradient = 0 for clipped values)')
    
    ax2.axvspan(float_min, float_max, alpha=0.1, color='green')
    ax2.axvspan(x_range.min(), float_min, alpha=0.1, color='red')
    ax2.axvspan(float_max, x_range.max(), alpha=0.1, color='red')
    ax2.axvline(float_min, color='darkred', linestyle='-', linewidth=2, alpha=0.8)
    ax2.axvline(float_max, color='darkred', linestyle='-', linewidth=2, alpha=0.8)
    
    ax2.annotate('Gradient = 0\n(clipped low)', 
                 xy=(float_min * 1.25, 0.5), fontsize=10, ha='center', color='darkred')
    ax2.annotate('Gradient = 1\n(pass-through)', 
                 xy=((float_min + float_max) / 2, 0.5), fontsize=10, ha='center', color='darkgreen')
    ax2.annotate('Gradient = 0\n(clipped high)', 
                 xy=(float_max * 1.25, 0.5), fontsize=10, ha='center', color='darkred')
    
    ax2.set_xlabel('Input Value', fontsize=12)
    ax2.set_ylabel('Gradient (∂q/∂x)', fontsize=12)
    ax2.set_title('STE Gradient Comparison', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim(-0.1, 1.3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    plt.show()
    
    return fig


# =============================================================================
# SECTION 7: Real CNN Model for QAT
# =============================================================================

class SmallCNN(nn.Module):
    """
    A real CNN architecture for demonstrating QAT.
    
    Architecture:
        Conv2d(1, 32, 3) -> ReLU -> MaxPool
        Conv2d(32, 64, 3) -> ReLU -> MaxPool
        Linear(1600, 128) -> ReLU
        Linear(128, 10)
    
    This is a realistic model similar to what you'd use for MNIST/FashionMNIST.
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class QATSmallCNN(nn.Module):
    """
    The same CNN architecture with fake quantization inserted.
    
    This demonstrates exactly where fake quantizers are placed:
    - Before each Conv2d (quantize activations)
    - Inside each Conv2d (quantize weights)
    - Before each Linear (quantize activations)
    - Inside each Linear (quantize weights)
    """
    
    def __init__(self, num_classes: int = 10, bits: int = 8):
        super().__init__()
        self.bits = bits
        
        # Fake-quantized layers
        self.conv1 = FakeQuantizedConv2d(1, 32, kernel_size=3, padding=1, bits=bits)
        self.conv2 = FakeQuantizedConv2d(32, 64, kernel_size=3, padding=1, bits=bits)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = FakeQuantizedLinear(64 * 7 * 7, 128, bits=bits)
        self.fc2 = FakeQuantizedLinear(128, num_classes, bits=bits)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    @classmethod
    def from_float_model(cls, float_model: SmallCNN, bits: int = 8):
        """Convert a trained FP32 model to QAT model, copying weights."""
        qat_model = cls(num_classes=float_model.fc2.linear.out_features 
                       if hasattr(float_model.fc2, 'linear') else float_model.fc2.out_features,
                       bits=bits)
        
        # Copy conv1 weights
        qat_model.conv1.conv.weight.data = float_model.conv1.weight.data.clone()
        if float_model.conv1.bias is not None:
            qat_model.conv1.conv.bias.data = float_model.conv1.bias.data.clone()
        
        # Copy conv2 weights
        qat_model.conv2.conv.weight.data = float_model.conv2.weight.data.clone()
        if float_model.conv2.bias is not None:
            qat_model.conv2.conv.bias.data = float_model.conv2.bias.data.clone()
        
        # Copy fc1 weights
        qat_model.fc1.linear.weight.data = float_model.fc1.weight.data.clone()
        if float_model.fc1.bias is not None:
            qat_model.fc1.linear.bias.data = float_model.fc1.bias.data.clone()
        
        # Copy fc2 weights
        qat_model.fc2.linear.weight.data = float_model.fc2.weight.data.clone()
        if float_model.fc2.bias is not None:
            qat_model.fc2.linear.bias.data = float_model.fc2.bias.data.clone()
        
        return qat_model


def print_model_structure(model: nn.Module, title: str):
    """Print model structure showing fake quantization insertion points."""
    print(f"\n{title}")
    print("=" * 60)
    
    for name, module in model.named_modules():
        if name == '':
            continue
        indent = "  " * name.count('.')
        
        if isinstance(module, FakeQuantizedConv2d):
            conv = module.conv
            print(f"{indent}{name}: FakeQuantizedConv2d("
                  f"in={conv.in_channels}, out={conv.out_channels}, "
                  f"kernel={conv.kernel_size}, bits={module.bits})")
            print(f"{indent}  ├── Weight quantization: per-channel symmetric")
            print(f"{indent}  └── Activation quantization: per-tensor asymmetric")
        elif isinstance(module, FakeQuantizedLinear):
            linear = module.linear
            print(f"{indent}{name}: FakeQuantizedLinear("
                  f"in={linear.in_features}, out={linear.out_features}, "
                  f"bits={module.bits})")
            print(f"{indent}  ├── Weight quantization: per-channel symmetric")
            print(f"{indent}  └── Activation quantization: per-tensor asymmetric")
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            if isinstance(module, nn.Conv2d):
                print(f"{indent}{name}: Conv2d(in={module.in_channels}, "
                      f"out={module.out_channels}, kernel={module.kernel_size})")
            else:
                print(f"{indent}{name}: Linear(in={module.in_features}, "
                      f"out={module.out_features})")
        elif isinstance(module, (nn.ReLU, nn.MaxPool2d)):
            print(f"{indent}{name}: {module.__class__.__name__}")


# =============================================================================
# SECTION 8: Weight Adaptation Visualization with Real Model
# =============================================================================

def visualize_weight_adaptation(epochs: int = 50, bits: int = 4,
                                 save_path: Optional[str] = None,
                                 device: str = 'cpu'):
    """
    Figure 5.9: Visualize how weights adapt to quantization grid during QAT.
    
    Trains a real CNN on synthetic data and tracks weight distribution evolution.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available. Skipping visualization.")
        return None
    
    print("\n" + "=" * 70)
    print(f"VISUALIZATION: Weight Adaptation During QAT (INT{bits})")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Create a real CNN model
    float_model = SmallCNN(num_classes=10)
    qat_model = QATSmallCNN.from_float_model(float_model, bits=bits)
    qat_model = qat_model.to(device)
    
    print_model_structure(qat_model, "QAT Model Architecture")
    
    # Create synthetic MNIST-like data
    print("\nGenerating synthetic training data...")
    num_samples = 2000
    X = torch.randn(num_samples, 1, 28, 28)  # MNIST-like dimensions
    y = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Track weight distributions from conv1 (most visible effect)
    weight_history = []
    snapshot_epochs = [0, epochs // 4, epochs // 2, 3 * epochs // 4, epochs - 1]
    
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 40)
    
    for epoch in range(epochs):
        # Snapshot weights at specific epochs
        if epoch in snapshot_epochs:
            # Collect all conv weights for visualization
            w_conv1 = qat_model.conv1.conv.weight.detach().clone().flatten().cpu().numpy()
            weight_history.append((epoch, w_conv1.copy()))
        
        # Training
        qat_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = qat_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch}: loss = {epoch_loss / num_batches:.4f}")
    
    # Final snapshot if not already captured
    if epochs - 1 not in snapshot_epochs:
        w_final = qat_model.conv1.conv.weight.detach().clone().flatten().cpu().numpy()
        weight_history.append((epochs - 1, w_final.copy()))
    
    # Create the visualization (matching the uploaded figure style)
    fig, axes = plt.subplots(1, len(weight_history), figsize=(16, 4))
    
    # Compute global weight range for consistent grid lines
    w_all = np.concatenate([h[1] for h in weight_history])
    abs_max = np.abs(w_all).max()
    
    # INT4 quantization grid
    q_max = 2 ** (bits - 1) - 1
    scale = abs_max / q_max
    grid_lines = np.arange(-q_max, q_max + 1) * scale
    
    for idx, (ax, (epoch, weights)) in enumerate(zip(axes, weight_history)):
        # Histogram with style matching the uploaded figure
        ax.hist(weights, bins=50, density=True, alpha=0.7, 
                color='steelblue', edgecolor='white', linewidth=0.3)
        
        # Quantization grid lines (red dashed)
        for gl in grid_lines:
            if abs(gl) <= abs_max * 1.2:
                ax.axvline(gl, color='red', alpha=0.5, linestyle='--', linewidth=1)
        
        ax.set_title(f'Epoch {epoch}', fontsize=12)
        ax.set_xlabel('Weight Value', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Density', fontsize=10)
        
        ax.set_xlim(-abs_max * 1.3, abs_max * 1.3)
    
    fig.suptitle(f'Weight Distribution Evolution During QAT (INT{bits})\n'
                 'Red dashed lines = quantization grid points', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path, formats=['png', 'pdf'])
    
    plt.show()
    
    return {'model': qat_model, 'weight_history': weight_history, 'fig': fig}


# =============================================================================
# SECTION 9: Complete QAT Training with Real Model
# =============================================================================

def run_qat_cnn(bits: int = 4, epochs: int = 20, 
                save_plots: bool = False,
                device: str = 'cpu'):
    """
    Complete QAT experiment on a real CNN with visualization.
    """
    print("\n" + "=" * 70)
    print(f"QAT EXPERIMENT: SmallCNN with INT{bits} Quantization")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Load FashionMNIST or create synthetic data
    if HAS_TORCHVISION:
        print("\nLoading FashionMNIST dataset...")
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            train_dataset = torchvision.datasets.FashionMNIST(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.FashionMNIST(
                root='./data', train=False, download=True, transform=transform
            )
            # Use subset for faster training
            train_dataset = Subset(train_dataset, range(5000))
            test_dataset = Subset(test_dataset, range(1000))
            print(f"   Train samples: {len(train_dataset)}")
            print(f"   Test samples: {len(test_dataset)}")
        except:
            print("   Could not load FashionMNIST. Using synthetic data.")
            HAS_TORCHVISION_DATA = False
    else:
        HAS_TORCHVISION_DATA = False
    
    if not HAS_TORCHVISION or not 'train_dataset' in dir():
        print("\nGenerating synthetic MNIST-like data...")
        X_train = torch.randn(5000, 1, 28, 28)
        y_train = torch.randint(0, 10, (5000,))
        X_test = torch.randn(1000, 1, 28, 28)
        y_test = torch.randint(0, 10, (1000,))
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Step 1: Create and evaluate FP32 model
    print("\n" + "-" * 40)
    print("Step 1: Training FP32 baseline model...")
    print("-" * 40)
    
    fp32_model = SmallCNN(num_classes=10).to(device)
    print_model_structure(fp32_model, "FP32 Model Architecture")
    
    optimizer = torch.optim.Adam(fp32_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train FP32 model briefly
    fp32_model.train()
    for epoch in range(5):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = fp32_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 2 == 0:
            print(f"   Epoch {epoch}: loss = {epoch_loss / len(train_loader):.4f}")
    
    # Evaluate FP32
    fp32_model.eval()
    fp32_correct = 0
    fp32_total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = fp32_model(batch_x)
            _, predicted = outputs.max(1)
            fp32_correct += (predicted == batch_y).sum().item()
            fp32_total += batch_y.size(0)
    fp32_acc = 100.0 * fp32_correct / fp32_total
    print(f"\n   FP32 Baseline Accuracy: {fp32_acc:.2f}%")
    
    # Step 2: Convert to QAT model
    print("\n" + "-" * 40)
    print(f"Step 2: Converting to QAT model (INT{bits})...")
    print("-" * 40)
    
    qat_model = QATSmallCNN.from_float_model(fp32_model, bits=bits).to(device)
    print_model_structure(qat_model, f"QAT Model Architecture (INT{bits})")
    
    # Step 3: QAT fine-tuning with weight tracking
    print("\n" + "-" * 40)
    print(f"Step 3: QAT fine-tuning for {epochs} epochs...")
    print("-" * 40)
    
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=1e-4)  # Lower LR for fine-tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Track metrics and weights
    history = {'train_loss': [], 'val_acc': [], 'lr': []}
    weight_history = []
    snapshot_epochs = [0, epochs // 4, epochs // 2, 3 * epochs // 4, epochs - 1]
    
    for epoch in range(epochs):
        # Snapshot weights
        if epoch in snapshot_epochs:
            w = qat_model.conv1.conv.weight.detach().clone().flatten().cpu().numpy()
            weight_history.append((epoch, w.copy()))
        
        # Training
        qat_model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = qat_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        qat_model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = qat_model(batch_x)
                _, predicted = outputs.max(1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        val_acc = 100.0 * val_correct / val_total
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_acc'].append(val_acc)
        history['lr'].append(scheduler.get_last_lr()[0])
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"   Epoch {epoch}: loss = {train_loss / len(train_loader):.4f}, "
                  f"val_acc = {val_acc:.2f}%")
    
    # Final weight snapshot
    if epochs - 1 not in snapshot_epochs:
        w = qat_model.conv1.conv.weight.detach().clone().flatten().cpu().numpy()
        weight_history.append((epochs - 1, w.copy()))
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"FP32 Baseline:      {fp32_acc:.2f}%")
    print(f"After QAT (INT{bits}):  {history['val_acc'][-1]:.2f}%")
    print(f"Change:             {history['val_acc'][-1] - fp32_acc:+.2f}%")
    
    # Visualizations
    if save_plots and HAS_MATPLOTLIB:
        # Plot 1: Weight evolution
        fig1, axes = plt.subplots(1, len(weight_history), figsize=(16, 4))
        
        w_all = np.concatenate([h[1] for h in weight_history])
        abs_max = np.abs(w_all).max()
        q_max = 2 ** (bits - 1) - 1
        scale = abs_max / q_max
        grid_lines = np.arange(-q_max, q_max + 1) * scale
        
        for idx, (ax, (epoch, weights)) in enumerate(zip(axes, weight_history)):
            ax.hist(weights, bins=50, density=True, alpha=0.7, 
                    color='steelblue', edgecolor='white', linewidth=0.3)
            for gl in grid_lines:
                if abs(gl) <= abs_max * 1.2:
                    ax.axvline(gl, color='red', alpha=0.5, linestyle='--', linewidth=1)
            ax.set_title(f'Epoch {epoch}', fontsize=12)
            ax.set_xlabel('Weight Value', fontsize=10)
            if idx == 0:
                ax.set_ylabel('Density', fontsize=10)
            ax.set_xlim(-abs_max * 1.3, abs_max * 1.3)
        
        fig1.suptitle(f'Weight Distribution Evolution During QAT (INT{bits})\n'
                     'Red dashed lines = quantization grid points', fontsize=14, y=1.02)
        plt.tight_layout()
        save_figure(fig1, 'qat_weight_evolution', formats=['png', 'pdf'])
        plt.show()
        
        # Plot 2: Training curves
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history['train_loss'], 'b-', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title('QAT Training Loss', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history['val_acc'], 'g-', linewidth=2, label=f'QAT INT{bits}')
        ax2.axhline(y=fp32_acc, color='r', linestyle='--', linewidth=2, label='FP32 Baseline')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
        ax2.set_title('QAT Validation Accuracy', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig2, 'qat_training_curves', formats=['png', 'pdf'])
        plt.show()
    
    return {
        'fp32_model': fp32_model,
        'qat_model': qat_model,
        'fp32_acc': fp32_acc,
        'qat_acc': history['val_acc'][-1],
        'history': history,
        'weight_history': weight_history
    }


# =============================================================================
# SECTION 10: Comprehensive Demo
# =============================================================================

def run_all_demos(save_plots: bool = False):
    """Run all demonstrations in sequence."""
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  Chapter 5.2: Fake Quantization and Straight-Through Estimators  ".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Demo 1: Gradient problem
    demonstrate_gradient_problem()
    
    # Demo 2: STE variants
    demonstrate_ste_variants()
    
    # Demo 3: STE enables optimization
    demonstrate_ste_optimization()
    
    # Demo 4: Numerical stability
    demonstrate_numerical_stability()
    
    # Demo 5: Clipped STE gradient visualization
    if HAS_MATPLOTLIB:
        save_path = 'clipped_ste_gradient' if save_plots else None
        visualize_clipped_ste_gradient(bits=4, scale=0.5, save_path=save_path)
    
    # Demo 6: Weight adaptation on real CNN
    if HAS_MATPLOTLIB:
        save_path = 'weight_adaptation' if save_plots else None
        visualize_weight_adaptation(epochs=50, bits=4, save_path=save_path, device=device)
    
    print("\n" + "=" * 70)
    print("All demonstrations complete!")
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chapter 5.2: Fake Quantization and Straight-Through Estimators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ch5_fake_quantization_ste.py --all
    python ch5_fake_quantization_ste.py --all --save-plots
    python ch5_fake_quantization_ste.py --demo-gradient-problem
    python ch5_fake_quantization_ste.py --demo-ste-variants
    python ch5_fake_quantization_ste.py --visualize-clipped-ste --save-plots
    python ch5_fake_quantization_ste.py --visualize-weight-adaptation --save-plots
    python ch5_fake_quantization_ste.py --qat-cnn --bits 4 --epochs 20 --save-plots
        """
    )
    
    # Demo options
    parser.add_argument('--all', action='store_true',
                        help='Run all demonstrations')
    parser.add_argument('--demo-gradient-problem', action='store_true',
                        help='Demonstrate the zero-gradient problem')
    parser.add_argument('--demo-ste-variants', action='store_true',
                        help='Compare STE variants')
    parser.add_argument('--demo-numerical-stability', action='store_true',
                        help='Demonstrate numerical stability safeguards')
    parser.add_argument('--visualize-clipped-ste', action='store_true',
                        help='Visualize clipped STE gradient function')
    parser.add_argument('--visualize-weight-adaptation', action='store_true',
                        help='Visualize weight adaptation during QAT')
    
    # QAT experiments
    parser.add_argument('--qat-cnn', action='store_true',
                        help='Run complete QAT experiment on SmallCNN')
    
    # Configuration
    parser.add_argument('--bits', type=int, default=4,
                        help='Bit-width for quantization (default: 4)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of QAT epochs (default: 20)')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots as PNG and editable PDF')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    if args.all:
        run_all_demos(save_plots=args.save_plots)
    
    if args.demo_gradient_problem:
        demonstrate_gradient_problem()
    
    if args.demo_ste_variants:
        demonstrate_ste_variants()
        demonstrate_ste_optimization()
    
    if args.demo_numerical_stability:
        demonstrate_numerical_stability()
    
    if args.visualize_clipped_ste:
        save_path = 'clipped_ste_gradient' if args.save_plots else None
        visualize_clipped_ste_gradient(bits=args.bits, scale=0.5, save_path=save_path)
    
    if args.visualize_weight_adaptation:
        save_path = 'weight_adaptation' if args.save_plots else None
        visualize_weight_adaptation(epochs=args.epochs, bits=args.bits, 
                                   save_path=save_path, device=device)
    
    if args.qat_cnn:
        run_qat_cnn(bits=args.bits, epochs=args.epochs, 
                   save_plots=args.save_plots, device=device)


if __name__ == "__main__":
    main()