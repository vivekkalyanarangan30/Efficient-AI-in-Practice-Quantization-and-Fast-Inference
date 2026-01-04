#!/usr/bin/env python3
"""
Chapter 4.4 Companion Script: Bias Correction and Cross-Layer Equalization

This script demonstrates:
1. How quantization errors accumulate through network layers
2. Bias correction to fix systematic output shifts
3. Cross-layer equalization to balance weight ranges
4. The complete PTQ pipeline combining both techniques

Usage:
    python ch4_bias_correction_equalization.py --model resnet18 --demo all
    python ch4_bias_correction_equalization.py --model resnet18 --demo accumulation
    python ch4_bias_correction_equalization.py --model resnet18 --demo bias_correction
    python ch4_bias_correction_equalization.py --model resnet18 --demo equalization

Requirements:
    pip install torch torchvision matplotlib numpy
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict


# ============================================================================
# Quantization Utilities
# ============================================================================

def symmetric_quantize(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-tensor symmetric quantization."""
    abs_max = tensor.abs().max()
    q_max = 2**(bits-1) - 1
    scale = abs_max / q_max
    if scale == 0:
        return tensor.clone()
    q = torch.round(tensor / scale).clamp(-q_max, q_max)
    return q * scale


def per_channel_quantize(tensor: torch.Tensor, bits: int = 8, axis: int = 0) -> torch.Tensor:
    """Per-channel symmetric quantization."""
    q_max = 2**(bits-1) - 1
    
    # Move target axis to position 0
    tensor_t = tensor.transpose(0, axis) if axis != 0 else tensor
    
    # Reshape to [channels, -1]
    shape = tensor_t.shape
    tensor_2d = tensor_t.reshape(shape[0], -1)
    
    # Compute per-channel scales
    channel_max = tensor_2d.abs().max(dim=1, keepdim=True).values
    channel_max = torch.clamp(channel_max, min=1e-8)
    scales = channel_max / q_max
    
    # Quantize
    q = torch.round(tensor_2d / scales).clamp(-q_max, q_max)
    dq = q * scales
    
    # Reshape back
    dq = dq.reshape(shape)
    if axis != 0:
        dq = dq.transpose(0, axis)
    
    return dq


# ============================================================================
# Demo 1: Error Accumulation Through Layers
# ============================================================================

def demo_error_accumulation():
    """
    Demonstrates how small per-layer quantization errors compound through
    a deep network, even when each layer appears well-calibrated.
    """
    print("\n" + "=" * 70)
    print("DEMO: Quantization Error Accumulation")
    print("=" * 70)
    
    # Simulate multiplicative error accumulation
    print("\n1. Multiplicative Error Accumulation Simulation")
    print("-" * 50)
    
    biases = [0.001, 0.002, 0.005]  # 0.1%, 0.2%, 0.5% per layer
    depths = [10, 50, 100, 200]
    
    print(f"{'Bias/Layer':<12} ", end="")
    for d in depths:
        print(f"{'Depth ' + str(d):>12}", end="")
    print()
    print("-" * 60)
    
    for bias in biases:
        print(f"{bias*100:.1f}%         ", end="")
        for depth in depths:
            accumulated = (1 + bias) ** depth - 1
            print(f"{accumulated*100:>+11.1f}%", end="")
        print()
    
    # Now demonstrate with actual neural network layers
    print("\n2. Actual Layer-by-Layer Error Propagation")
    print("-" * 50)
    
    class SimpleChain(nn.Module):
        """Chain of linear layers for error propagation demo."""
        def __init__(self, hidden_dim: int, num_layers: int):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) 
                for _ in range(num_layers)
            ])
            # Initialize with small weights to prevent explosion
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
        
        def forward(self, x: torch.Tensor, return_intermediates: bool = False):
            intermediates = []
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x))
                if return_intermediates:
                    intermediates.append((f"layer_{i}", x.clone()))
            if return_intermediates:
                return x, intermediates
            return x
    
    torch.manual_seed(42)
    model = SimpleChain(hidden_dim=256, num_layers=10)
    model.eval()
    
    x = torch.randn(32, 256)
    
    # FP32 forward pass
    with torch.no_grad():
        _, fp32_intermediates = model(x, return_intermediates=True)
    
    # Quantized forward pass
    with torch.no_grad():
        x_q = x.clone()
        q_intermediates = []
        
        for i, layer in enumerate(model.layers):
            # Quantize weights
            w_q = symmetric_quantize(layer.weight, bits=8)
            
            # Forward with quantized weights
            x_q = F.linear(x_q, w_q, layer.bias)
            x_q = symmetric_quantize(x_q, bits=8)  # Quantize activations
            x_q = F.relu(x_q)
            
            q_intermediates.append((f"layer_{i}", x_q.clone()))
    
    # Analyze error propagation
    print(f"{'Layer':<10} {'Mean Error':>12} {'RMS Error':>12} {'Relative':>10}")
    print("-" * 50)
    
    for (name_fp, fp_act), (name_q, q_act) in zip(fp32_intermediates, q_intermediates):
        mean_err = (q_act - fp_act).mean().item()
        rms_err = torch.sqrt(((q_act - fp_act)**2).mean()).item()
        rel_err = rms_err / (fp_act.abs().mean().item() + 1e-8)
        print(f"{name_fp:<10} {mean_err:>+12.6f} {rms_err:>12.6f} {rel_err:>9.2%}")
    
    print("\nKey insight: Mean error (bias) grows through layers even though")
    print("each layer is independently well-quantized. This motivates bias correction.")


# ============================================================================
# Demo 2: Bias Correction
# ============================================================================

class BiasCorrector:
    """
    Estimates and applies bias correction for quantized layers.
    """
    
    def __init__(self, quantize_fn=symmetric_quantize):
        self.quantize_fn = quantize_fn
        self.corrections: Dict[str, torch.Tensor] = {}
    
    def estimate_correction(
        self, 
        layer_name: str,
        layer: nn.Module,
        calibration_inputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """Estimate bias correction for a single layer."""
        accumulated_diff = None
        num_samples = 0
        
        with torch.no_grad():
            w_fp = layer.weight
            w_q = self.quantize_fn(w_fp)
            
            for x in calibration_inputs:
                if isinstance(layer, nn.Conv2d):
                    y_fp = F.conv2d(x, w_fp, bias=None, 
                                   stride=layer.stride, padding=layer.padding)
                    y_q = F.conv2d(x, w_q, bias=None,
                                  stride=layer.stride, padding=layer.padding)
                else:
                    y_fp = F.linear(x, w_fp, bias=None)
                    y_q = F.linear(x, w_q, bias=None)
                
                diff = y_fp - y_q
                
                if diff.dim() == 4:  # Conv output [B, C, H, W]
                    channel_diff = diff.mean(dim=(0, 2, 3))
                else:  # Linear output [B, C]
                    channel_diff = diff.mean(dim=0)
                
                if accumulated_diff is None:
                    accumulated_diff = channel_diff
                else:
                    accumulated_diff += channel_diff
                num_samples += 1
        
        correction = accumulated_diff / num_samples
        self.corrections[layer_name] = correction
        return correction
    
    def apply_correction(self, layer_name: str, layer: nn.Module):
        """Apply correction to layer's bias."""
        if layer_name not in self.corrections:
            raise ValueError(f"No correction for {layer_name}")
        
        correction = self.corrections[layer_name]
        
        if layer.bias is None:
            layer.bias = nn.Parameter(correction.clone())
        else:
            layer.bias.data += correction


def demo_bias_correction():
    """
    Demonstrates bias correction on a simple network.
    """
    print("\n" + "=" * 70)
    print("DEMO: Bias Correction")
    print("=" * 70)
    
    # Create a simple model
    torch.manual_seed(42)
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
            self.fc = nn.Linear(64 * 4 * 4, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = SimpleCNN()
    model.eval()
    
    # Generate calibration data
    calibration_data = [torch.randn(8, 3, 32, 32) for _ in range(5)]
    
    # Collect layer inputs via hooks
    layer_inputs = {
        'conv1': [], 'conv2': [], 'conv3': [], 'fc': []
    }
    
    def make_hook(name):
        def hook(module, input, output):
            layer_inputs[name].append(input[0].detach().clone())
        return hook
    
    hooks = [
        model.conv1.register_forward_hook(make_hook('conv1')),
        model.conv2.register_forward_hook(make_hook('conv2')),
        model.conv3.register_forward_hook(make_hook('conv3')),
        model.fc.register_forward_hook(make_hook('fc')),
    ]
    
    with torch.no_grad():
        for x in calibration_data:
            _ = model(x)
    
    for h in hooks:
        h.remove()
    
    # Estimate and apply corrections
    corrector = BiasCorrector()
    
    print("\nBias Correction Analysis")
    print("-" * 60)
    print(f"{'Layer':<10} {'Correction Mean':>15} {'Correction Max':>15}")
    print("-" * 60)
    
    layers = [
        ('conv1', model.conv1),
        ('conv2', model.conv2),
        ('conv3', model.conv3),
        ('fc', model.fc),
    ]
    
    for name, layer in layers:
        if layer_inputs[name]:
            correction = corrector.estimate_correction(name, layer, layer_inputs[name][:3])
            print(f"{name:<10} {correction.mean():>+15.6f} {correction.abs().max():>15.6f}")
    
    # Demonstrate error reduction
    print("\nError Reduction After Bias Correction")
    print("-" * 60)
    
    test_input = calibration_data[0][:1]  # Single sample
    
    for name, layer in layers:
        if not layer_inputs[name]:
            continue
            
        x = layer_inputs[name][0][:1]
        w_q = symmetric_quantize(layer.weight)
        
        if isinstance(layer, nn.Conv2d):
            y_fp = F.conv2d(x, layer.weight, layer.bias, 
                          stride=layer.stride, padding=layer.padding)
            y_q = F.conv2d(x, w_q, layer.bias,
                         stride=layer.stride, padding=layer.padding)
            
            correction = corrector.corrections.get(name)
            if correction is not None:
                bias_corrected = layer.bias + correction if layer.bias is not None else correction
                y_q_corrected = F.conv2d(x, w_q, bias_corrected,
                                        stride=layer.stride, padding=layer.padding)
        else:
            y_fp = F.linear(x, layer.weight, layer.bias)
            y_q = F.linear(x, w_q, layer.bias)
            
            correction = corrector.corrections.get(name)
            if correction is not None:
                bias_corrected = layer.bias + correction if layer.bias is not None else correction
                y_q_corrected = F.linear(x, w_q, bias_corrected)
        
        error_before = (y_q - y_fp).mean().item()
        error_after = (y_q_corrected - y_fp).mean().item() if correction is not None else error_before
        
        reduction = (1 - abs(error_after) / (abs(error_before) + 1e-10)) * 100
        print(f"{name:<10} Before: {error_before:>+.6f}  After: {error_after:>+.6f}  Reduction: {reduction:>5.1f}%")


# ============================================================================
# Demo 3: Cross-Layer Equalization
# ============================================================================

def compute_equalization_scale(
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute per-channel equalization scales for adjacent layers.
    
    For Conv2d: weight1 [out1, in1, kH, kW], weight2 [out2, in2, kH, kW]
    We need out1 == in2 for equalization to work.
    
    range1: per output-channel max of weight1 (shape [out1])
    range2: per input-channel max of weight2 (shape [in2])
    """
    # Get output channels of layer1 and input channels of layer2
    out_channels_1 = weight1.shape[0]
    in_channels_2 = weight2.shape[1]
    
    # Sanity check
    assert out_channels_1 == in_channels_2, \
        f"Channel mismatch: layer1 out={out_channels_1}, layer2 in={in_channels_2}"
    
    # Reshape to 2D: [channels, -1]
    if weight1.dim() == 4:
        # Conv: [out, in, H, W] -> [out, in*H*W]
        w1_2d = weight1.reshape(weight1.shape[0], -1)
    else:
        w1_2d = weight1
    
    if weight2.dim() == 4:
        # Conv: [out, in, H, W] -> reshape to get per-input-channel stats
        # Transpose to [in, out, H, W] then reshape to [in, out*H*W]
        w2_transposed = weight2.transpose(0, 1)
        w2_2d = w2_transposed.reshape(weight2.shape[1], -1)
    else:
        # Linear: [out, in] -> transpose to [in, out]
        w2_2d = weight2.t()
    
    # Per output-channel range of W1: max over all elements in that output channel
    range1 = w1_2d.abs().max(dim=1).values  # Shape: [out1]
    
    # Per input-channel range of W2: max over all elements using that input channel
    range2 = w2_2d.abs().max(dim=1).values  # Shape: [in2]
    
    # Clamp for stability
    range1 = torch.clamp(range1, min=eps)
    range2 = torch.clamp(range2, min=eps)
    
    # Geometric mean balances the ranges
    scale = torch.sqrt(range2 / range1)
    scale = torch.clamp(scale, min=0.01, max=100.0)
    
    return scale


def apply_equalization(
    layer1: nn.Module,
    layer2: nn.Module,
    scale: torch.Tensor
) -> None:
    """Apply equalization to layer pair."""
    with torch.no_grad():
        if isinstance(layer1, nn.Conv2d):
            scale_shape = [scale.shape[0], 1, 1, 1]
            layer1.weight.mul_(scale.view(*scale_shape))
        else:
            layer1.weight.mul_(scale.view(-1, 1))
        
        if layer1.bias is not None:
            layer1.bias.mul_(scale)
        
        if isinstance(layer2, nn.Conv2d):
            scale_shape = [1, scale.shape[0], 1, 1]
            layer2.weight.div_(scale.view(*scale_shape))
        else:
            layer2.weight.div_(scale.view(1, -1))


def demo_cross_layer_equalization():
    """
    Demonstrates cross-layer equalization on a simple network.
    """
    print("\n" + "=" * 70)
    print("DEMO: Cross-Layer Equalization")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Create a simple model with intentionally imbalanced layers
    # Note: For equalization, layer1.out_channels must equal layer2.in_channels
    class ImbalancedNet(nn.Module):
        def __init__(self):
            super().__init__()
            # conv1: 3 -> 32, conv2: 32 -> 32, conv3: 32 -> 64
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)  # Changed to 32->32
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # Changed to 32->64
            
            # Make layer weights intentionally imbalanced
            with torch.no_grad():
                self.conv1.weight.mul_(0.1)   # Small weights
                self.conv2.weight.mul_(10.0)  # Large weights
                self.conv3.weight.mul_(0.5)   # Medium weights
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x
    
    model = ImbalancedNet()
    model.eval()
    
    # Record pre-equalization statistics
    print("\nPre-Equalization Weight Ranges")
    print("-" * 50)
    
    pre_stats = {}
    for name, layer in [('conv1', model.conv1), ('conv2', model.conv2), ('conv3', model.conv3)]:
        max_range = layer.weight.abs().max().item()
        pre_stats[name] = max_range
        print(f"{name}: {max_range:.4f}")
    
    # Store original output for verification
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        y_original = model(x)
    
    # Apply equalization
    print("\nApplying Equalization")
    print("-" * 50)
    
    # Only equalize pairs where out_channels == in_channels of next layer
    # conv1: out=32, conv2: in=32 ✓
    # conv2: out=32, conv3: in=32 ✓
    pairs = [
        ('conv1', model.conv1, 'conv2', model.conv2),
        ('conv2', model.conv2, 'conv3', model.conv3),
    ]
    
    for name1, layer1, name2, layer2 in pairs:
        pre_range1 = pre_stats[name1]
        pre_range2 = pre_stats[name2]
        pre_ratio = max(pre_range1, pre_range2) / min(pre_range1, pre_range2)
        
        scale = compute_equalization_scale(layer1.weight, layer2.weight)
        apply_equalization(layer1, layer2, scale)
        
        post_range1 = layer1.weight.abs().max().item()
        post_range2 = layer2.weight.abs().max().item()
        post_ratio = max(post_range1, post_range2) / min(post_range1, post_range2)
        
        improvement = (1 - post_ratio / pre_ratio) * 100
        print(f"{name1} → {name2}: {pre_ratio:.2f}x → {post_ratio:.2f}x ({improvement:+.1f}% improvement)")
        
        # Update pre_stats for next iteration
        pre_stats[name1] = post_range1
        pre_stats[name2] = post_range2
    
    # Verify output unchanged
    with torch.no_grad():
        y_equalized = model(x)
    
    max_diff = (y_equalized - y_original).abs().max().item()
    print(f"\nOutput difference after equalization: {max_diff:.2e}")
    print("(Should be ~0 for numerical precision)")
    
    # Post-equalization statistics
    print("\nPost-Equalization Weight Ranges")
    print("-" * 50)
    for name, layer in [('conv1', model.conv1), ('conv2', model.conv2), ('conv3', model.conv3)]:
        max_range = layer.weight.abs().max().item()
        print(f"{name}: {max_range:.4f}")
    
    # Demonstrate quantization improvement
    print("\nQuantization Error Comparison")
    print("-" * 50)
    
    # Reset model for comparison
    model_original = ImbalancedNet()
    model_equalized = ImbalancedNet()
    
    # Apply equalization to equalized model
    for name1, layer1, name2, layer2 in [
        ('conv1', model_equalized.conv1, 'conv2', model_equalized.conv2),
        ('conv2', model_equalized.conv2, 'conv3', model_equalized.conv3),
    ]:
        scale = compute_equalization_scale(layer1.weight, layer2.weight)
        apply_equalization(layer1, layer2, scale)
    
    # Compare quantization error
    for name, layer_orig, layer_eq in [
        ('conv1', model_original.conv1, model_equalized.conv1),
        ('conv2', model_original.conv2, model_equalized.conv2),
        ('conv3', model_original.conv3, model_equalized.conv3),
    ]:
        w_orig = layer_orig.weight
        w_eq = layer_eq.weight
        
        # Quantize
        w_orig_q = symmetric_quantize(w_orig)
        w_eq_q = symmetric_quantize(w_eq)
        
        # Compute MSE
        mse_orig = ((w_orig - w_orig_q)**2).mean().item()
        mse_eq = ((w_eq - w_eq_q)**2).mean().item()
        
        improvement = (1 - mse_eq / mse_orig) * 100 if mse_orig > 0 else 0
        print(f"{name}: Original MSE={mse_orig:.2e}, Equalized MSE={mse_eq:.2e} ({improvement:+.1f}%)")


# ============================================================================
# Demo 4: Complete Pipeline
# ============================================================================

def demo_complete_pipeline():
    """
    Demonstrates the complete PTQ pipeline with equalization and bias correction.
    """
    print("\n" + "=" * 70)
    print("DEMO: Complete PTQ Pipeline")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # Create a simple CNN with compatible channels for equalization
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, padding=1)   # 32->32 for equalization
            self.conv3 = nn.Conv2d(32, 128, 3, padding=1)  # 32->128
            self.fc = nn.Linear(128 * 4 * 4, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # Create three versions of the model
    model_baseline = SimpleCNN()
    model_equalized = SimpleCNN()
    model_full = SimpleCNN()
    
    # Copy weights
    model_equalized.load_state_dict(model_baseline.state_dict())
    model_full.load_state_dict(model_baseline.state_dict())
    
    # Generate test data
    x_test = torch.randn(32, 3, 32, 32)
    
    # Get FP32 baseline output
    model_baseline.eval()
    with torch.no_grad():
        y_fp32 = model_baseline(x_test)
    
    print("\n[Step 1] Baseline Analysis")
    print("-" * 50)
    
    # Measure naive quantization error
    def measure_quantized_error(model, x, y_target):
        """Simulate quantized inference and measure error."""
        model.eval()
        with torch.no_grad():
            # Quantize weights
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.weight.data = symmetric_quantize(module.weight.data)
            
            y_q = model(x)
            mae = (y_q - y_target).abs().mean().item()
            return mae
    
    # Naive PTQ error
    model_naive = SimpleCNN()
    model_naive.load_state_dict(model_baseline.state_dict())
    naive_error = measure_quantized_error(model_naive, x_test, y_fp32)
    print(f"Naive PTQ MAE: {naive_error:.6f}")
    
    print("\n[Step 2] Apply Equalization")
    print("-" * 50)
    
    # Equalize conv pairs (only where channels are compatible)
    # conv1: 3->32, conv2: 32->32 ✓ (out1=32, in2=32)
    # conv2: 32->32, conv3: 32->128 ✓ (out2=32, in3=32)
    pairs = [
        ('conv1', model_equalized.conv1, 'conv2', model_equalized.conv2),
        ('conv2', model_equalized.conv2, 'conv3', model_equalized.conv3),
    ]
    
    for name1, layer1, name2, layer2 in pairs:
        pre_range1 = layer1.weight.abs().max().item()
        pre_range2 = layer2.weight.abs().max().item()
        
        scale = compute_equalization_scale(layer1.weight, layer2.weight)
        apply_equalization(layer1, layer2, scale)
        
        post_range1 = layer1.weight.abs().max().item()
        post_range2 = layer2.weight.abs().max().item()
        
        pre_ratio = max(pre_range1, pre_range2) / min(pre_range1, pre_range2)
        post_ratio = max(post_range1, post_range2) / min(post_range1, post_range2)
        
        print(f"{name1} → {name2}: ratio {pre_ratio:.2f}x → {post_ratio:.2f}x")
    
    # Equalization-only error
    model_eq_only = SimpleCNN()
    model_eq_only.load_state_dict(model_equalized.state_dict())
    eq_error = measure_quantized_error(model_eq_only, x_test, y_fp32)
    print(f"\nEqualization-only PTQ MAE: {eq_error:.6f}")
    
    print("\n[Step 3] Apply Full Pipeline (Equalization + Bias Correction)")
    print("-" * 50)
    
    # First equalize
    pairs = [
        ('conv1', model_full.conv1, 'conv2', model_full.conv2),
        ('conv2', model_full.conv2, 'conv3', model_full.conv3),
    ]
    
    for name1, layer1, name2, layer2 in pairs:
        scale = compute_equalization_scale(layer1.weight, layer2.weight)
        apply_equalization(layer1, layer2, scale)
    
    # Collect layer inputs
    calibration_data = [torch.randn(8, 3, 32, 32) for _ in range(5)]
    layer_inputs = {'conv1': [], 'conv2': [], 'conv3': [], 'fc': []}
    
    def make_hook(name):
        def hook(module, input, output):
            layer_inputs[name].append(input[0].detach().clone())
        return hook
    
    hooks = [
        model_full.conv1.register_forward_hook(make_hook('conv1')),
        model_full.conv2.register_forward_hook(make_hook('conv2')),
        model_full.conv3.register_forward_hook(make_hook('conv3')),
        model_full.fc.register_forward_hook(make_hook('fc')),
    ]
    
    model_full.eval()
    with torch.no_grad():
        for x in calibration_data:
            _ = model_full(x)
    
    for h in hooks:
        h.remove()
    
    # Apply bias corrections
    corrector = BiasCorrector()
    
    for name, layer in [
        ('conv1', model_full.conv1),
        ('conv2', model_full.conv2),
        ('conv3', model_full.conv3),
        ('fc', model_full.fc),
    ]:
        if layer_inputs[name]:
            correction = corrector.estimate_correction(name, layer, layer_inputs[name][:3])
            corrector.apply_correction(name, layer)
            print(f"Applied correction to {name}: mean={correction.mean():.6f}")
    
    # Full pipeline error
    full_error = measure_quantized_error(model_full, x_test, y_fp32)
    print(f"\nFull pipeline PTQ MAE: {full_error:.6f}")
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"Naive PTQ:              {naive_error:.6f}")
    print(f"Equalization only:      {eq_error:.6f} ({(1-eq_error/naive_error)*100:+.1f}%)")
    print(f"Equalization + BiasCorr: {full_error:.6f} ({(1-full_error/naive_error)*100:+.1f}%)")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chapter 4.4: Bias Correction and Cross-Layer Equalization"
    )
    parser.add_argument(
        '--demo', 
        choices=['all', 'accumulation', 'bias_correction', 'equalization', 'pipeline'],
        default='all',
        help='Which demo to run'
    )
    parser.add_argument(
        '--model',
        default='resnet18',
        help='Model to use (currently unused, demos use simple models)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Chapter 4.4: Bias Correction and Cross-Layer Equalization")
    print("=" * 70)
    
    if args.demo in ['all', 'accumulation']:
        demo_error_accumulation()
    
    if args.demo in ['all', 'bias_correction']:
        demo_bias_correction()
    
    if args.demo in ['all', 'equalization']:
        demo_cross_layer_equalization()
    
    if args.demo in ['all', 'pipeline']:
        demo_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("Demos complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()