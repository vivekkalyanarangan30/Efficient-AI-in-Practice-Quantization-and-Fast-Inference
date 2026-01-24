#!/usr/bin/env python3
"""
Chapter 4.4: Cross-Layer Equalization - Multi-Architecture Analysis

Tests equalization across different architectures to validate whether
the technique generalizes beyond ResNet.

Architectures tested:
- ResNet-18 (CNN, BatchNorm)
- MobileNetV2 (CNN, BatchNorm, inverted residuals)
- BERT-base (Transformer, LayerNorm)
- ViT-B/16 (Vision Transformer, LayerNorm)

Usage:
    python ch4_equalization_multi_arch.py --all
    python ch4_equalization_multi_arch.py --arch resnet18
    python ch4_equalization_multi_arch.py --arch mobilenetv2
    python ch4_equalization_multi_arch.py --arch bert
    python ch4_equalization_multi_arch.py --arch vit

Requirements:
    pip install torch torchvision transformers matplotlib pandas
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Data Classes for Results
# ============================================================================

@dataclass
class LayerPairStats:
    """Statistics for a layer pair before/after equalization."""
    name: str
    range1_before: float
    range2_before: float
    ratio_before: float
    range1_after: float
    range2_after: float
    ratio_after: float
    mse_before: float
    mse_after: float
    output_preserved: bool


@dataclass 
class ArchitectureResults:
    """Complete results for one architecture."""
    name: str
    num_params: int
    layer_pairs: List[LayerPairStats]
    naive_mae: float
    equalized_mae: float
    improvement_pct: float
    equalization_valid: bool  # Did equalization preserve FP32 output?


# ============================================================================
# Quantization Utilities
# ============================================================================

def symmetric_quantize(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Per-tensor symmetric quantization."""
    abs_max = tensor.abs().max()
    if abs_max == 0:
        return tensor.clone()
    q_max = 2**(bits-1) - 1
    scale = abs_max / q_max
    q = torch.round(tensor / scale).clamp(-q_max, q_max)
    return q * scale


def compute_weight_mse(weight: torch.Tensor) -> float:
    """Compute MSE between original and quantized weight."""
    w_q = symmetric_quantize(weight)
    return ((weight - w_q) ** 2).mean().item()


# ============================================================================
# Equalization Core Functions
# ============================================================================

def compute_equalization_scale(
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute per-channel equalization scales.
    Works for both Conv2d and Linear layers.
    """
    # Handle different tensor shapes
    if weight1.dim() == 4:  # Conv2d: [out, in, H, W]
        w1_2d = weight1.reshape(weight1.shape[0], -1)
    elif weight1.dim() == 2:  # Linear: [out, in]
        w1_2d = weight1
    else:
        raise ValueError(f"Unsupported weight1 dim: {weight1.dim()}")
    
    if weight2.dim() == 4:  # Conv2d
        w2_transposed = weight2.transpose(0, 1)
        w2_2d = w2_transposed.reshape(weight2.shape[1], -1)
    elif weight2.dim() == 2:  # Linear
        w2_2d = weight2.t()
    else:
        raise ValueError(f"Unsupported weight2 dim: {weight2.dim()}")
    
    # Per-channel ranges
    range1 = w1_2d.abs().max(dim=1).values
    range2 = w2_2d.abs().max(dim=1).values
    
    range1 = torch.clamp(range1, min=eps)
    range2 = torch.clamp(range2, min=eps)
    
    scale = torch.sqrt(range2 / range1)
    scale = torch.clamp(scale, min=0.01, max=100.0)
    
    return scale


def apply_equalization(
    layer1: nn.Module,
    layer2: nn.Module,
    scale: torch.Tensor
) -> None:
    """Apply equalization scales to layer pair in-place."""
    with torch.no_grad():
        # Scale layer1 output channels
        if isinstance(layer1, nn.Conv2d):
            layer1.weight.mul_(scale.view(-1, 1, 1, 1))
        elif isinstance(layer1, nn.Linear):
            layer1.weight.mul_(scale.view(-1, 1))
        
        if layer1.bias is not None:
            layer1.bias.mul_(scale)
        
        # Scale layer2 input channels
        if isinstance(layer2, nn.Conv2d):
            layer2.weight.div_(scale.view(1, -1, 1, 1))
        elif isinstance(layer2, nn.Linear):
            layer2.weight.div_(scale.view(1, -1))


def get_layer_pair_stats(
    name: str,
    layer1: nn.Module,
    layer2: nn.Module,
    layer1_orig: nn.Module,
    layer2_orig: nn.Module
) -> LayerPairStats:
    """Compute statistics for a layer pair."""
    # Before equalization (from original)
    range1_before = layer1_orig.weight.abs().max().item()
    range2_before = layer2_orig.weight.abs().max().item()
    ratio_before = max(range1_before, range2_before) / max(min(range1_before, range2_before), 1e-8)
    mse_before = compute_weight_mse(layer1_orig.weight) + compute_weight_mse(layer2_orig.weight)
    
    # After equalization
    range1_after = layer1.weight.abs().max().item()
    range2_after = layer2.weight.abs().max().item()
    ratio_after = max(range1_after, range2_after) / max(min(range1_after, range2_after), 1e-8)
    mse_after = compute_weight_mse(layer1.weight) + compute_weight_mse(layer2.weight)
    
    return LayerPairStats(
        name=name,
        range1_before=range1_before,
        range2_before=range2_before,
        ratio_before=ratio_before,
        range1_after=range1_after,
        range2_after=range2_after,
        ratio_after=ratio_after,
        mse_before=mse_before,
        mse_after=mse_after,
        output_preserved=True  # Will be updated later
    )


# ============================================================================
# Architecture-Specific Handlers
# ============================================================================

class ResNetHandler:
    """Handler for ResNet architectures."""
    
    @staticmethod
    def load_model():
        import torchvision.models as models
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        return model
    
    @staticmethod
    def fuse_bn(model: nn.Module) -> nn.Module:
        """Fuse BatchNorm into Conv layers."""
        model = deepcopy(model)
        
        def fuse_conv_bn(conv, bn):
            gamma = bn.weight.data
            beta = bn.bias.data
            mean = bn.running_mean
            var = bn.running_var
            eps = bn.eps
            
            std = torch.sqrt(var + eps)
            scale_factor = gamma / std
            
            fused_weight = conv.weight.data * scale_factor.view(-1, 1, 1, 1)
            if conv.bias is not None:
                fused_bias = (conv.bias.data - mean) * scale_factor + beta
            else:
                fused_bias = -mean * scale_factor + beta
            
            fused_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size,
                stride=conv.stride, padding=conv.padding, 
                dilation=conv.dilation, groups=conv.groups, bias=True
            )
            fused_conv.weight.data = fused_weight
            fused_conv.bias.data = fused_bias
            return fused_conv
        
        # Fuse conv1-bn1
        model.conv1 = fuse_conv_bn(model.conv1, model.bn1)
        model.bn1 = nn.Identity()
        
        # Fuse within blocks
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in layer:
                block.conv1 = fuse_conv_bn(block.conv1, block.bn1)
                block.bn1 = nn.Identity()
                block.conv2 = fuse_conv_bn(block.conv2, block.bn2)
                block.bn2 = nn.Identity()
                if block.downsample is not None:
                    block.downsample[0] = fuse_conv_bn(block.downsample[0], block.downsample[1])
                    block.downsample[1] = nn.Identity()
        
        return model
    
    @staticmethod
    def get_equalization_pairs(model: nn.Module) -> List[Tuple[str, nn.Module, nn.Module]]:
        """Get layer pairs that can be equalized."""
        pairs = []
        for layer_name, layer in [('layer1', model.layer1), ('layer2', model.layer2),
                                   ('layer3', model.layer3), ('layer4', model.layer4)]:
            for i, block in enumerate(layer):
                # conv1 -> conv2 within block (ReLU between them)
                if block.conv1.weight.shape[0] == block.conv2.weight.shape[1]:
                    pairs.append((f'{layer_name}.{i}', block.conv1, block.conv2))
        return pairs
    
    @staticmethod
    def get_test_input():
        return torch.randn(4, 3, 224, 224)


class MobileNetV2Handler:
    """Handler for MobileNetV2."""
    
    @staticmethod
    def load_model():
        import torchvision.models as models
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        return model
    
    @staticmethod
    def fuse_bn(model: nn.Module) -> nn.Module:
        """Fuse BatchNorm in MobileNetV2."""
        model = deepcopy(model)
        
        def fuse_conv_bn(conv, bn):
            gamma = bn.weight.data
            beta = bn.bias.data
            mean = bn.running_mean
            var = bn.running_var
            eps = bn.eps
            
            std = torch.sqrt(var + eps)
            scale_factor = gamma / std
            
            fused_weight = conv.weight.data * scale_factor.view(-1, 1, 1, 1)
            
            if conv.bias is not None:
                fused_bias = (conv.bias.data - mean) * scale_factor + beta
            else:
                fused_bias = -mean * scale_factor + beta
            
            fused_conv = nn.Conv2d(
                conv.in_channels, conv.out_channels, conv.kernel_size,
                stride=conv.stride, padding=conv.padding,
                dilation=conv.dilation, groups=conv.groups, bias=True
            )
            fused_conv.weight.data = fused_weight
            fused_conv.bias.data = fused_bias
            return fused_conv
        
        # Fuse in features - handle ConvBNActivation blocks
        for i, layer in enumerate(model.features):
            if hasattr(layer, '__getitem__'):
                # Sequential-like block
                j = 0
                while j < len(layer):
                    if isinstance(layer[j], nn.Conv2d) and j + 1 < len(layer):
                        if isinstance(layer[j + 1], nn.BatchNorm2d):
                            fused = fuse_conv_bn(layer[j], layer[j + 1])
                            layer[j] = fused
                            layer[j + 1] = nn.Identity()
                    j += 1
            elif hasattr(layer, 'conv'):
                # InvertedResidual block
                for j in range(len(layer.conv)):
                    sublayer = layer.conv[j]
                    if isinstance(sublayer, nn.Conv2d) and j + 1 < len(layer.conv):
                        next_sub = layer.conv[j + 1]
                        if isinstance(next_sub, nn.BatchNorm2d):
                            layer.conv[j] = fuse_conv_bn(sublayer, next_sub)
                            layer.conv[j + 1] = nn.Identity()
        
        return model
    
    @staticmethod
    def get_equalization_pairs(model: nn.Module) -> List[Tuple[str, nn.Module, nn.Module]]:
        """
        Get equalizable pairs in MobileNetV2.
        
        MobileNetV2 InvertedResidual structure:
        - expand: 1x1 conv (channels -> expand_ratio * channels)
        - depthwise: 3x3 conv with groups=channels (can't equalize across this)
        - project: 1x1 conv (expand_ratio * channels -> out_channels)
        
        We can try to equalize expand -> depthwise input, but depthwise breaks it.
        Alternative: look for consecutive pointwise convs in the overall architecture.
        """
        pairs = []
        
        # Find consecutive 1x1 convs (pointwise) that can be equalized
        all_convs = []
        for i, layer in enumerate(model.features):
            if hasattr(layer, 'conv'):
                for j, sublayer in enumerate(layer.conv):
                    if isinstance(sublayer, nn.Conv2d) and sublayer.groups == 1:
                        all_convs.append((f'features.{i}.conv.{j}', sublayer))
        
        # Look for pairs where output of one matches input of next
        for i in range(len(all_convs) - 1):
            name1, conv1 = all_convs[i]
            name2, conv2 = all_convs[i + 1]
            
            # Check if they're in the same block and channels match
            if conv1.weight.shape[0] == conv2.weight.shape[1]:
                # Check kernel sizes - prefer 1x1 convs for equalization
                if conv1.kernel_size == (1, 1) and conv2.kernel_size == (1, 1):
                    pairs.append((f'{name1}->{name2}', conv1, conv2))
        
        # If no 1x1 pairs found, try any matching pairs within blocks
        if len(pairs) == 0:
            for i, layer in enumerate(model.features):
                if hasattr(layer, 'conv'):
                    convs_in_block = [(j, m) for j, m in enumerate(layer.conv) 
                                      if isinstance(m, nn.Conv2d) and m.groups == 1]
                    for k in range(len(convs_in_block) - 1):
                        j1, c1 = convs_in_block[k]
                        j2, c2 = convs_in_block[k + 1]
                        if c1.weight.shape[0] == c2.weight.shape[1]:
                            pairs.append((f'block{i}.{j1}->{j2}', c1, c2))
        
        return pairs
    
    @staticmethod
    def get_test_input():
        return torch.randn(4, 3, 224, 224)


class BERTHandler:
    """Handler for BERT transformer."""
    
    @staticmethod
    def load_model():
        from transformers import BertModel
        model = BertModel.from_pretrained('bert-base-uncased')
        return model
    
    @staticmethod
    def fuse_bn(model: nn.Module) -> nn.Module:
        """BERT uses LayerNorm, not BatchNorm. LayerNorm cannot be fused."""
        return deepcopy(model)
    
    @staticmethod
    def get_equalization_pairs(model: nn.Module) -> List[Tuple[str, nn.Module, nn.Module]]:
        """
        Get equalizable pairs in BERT.
        
        BERT structure per layer:
        - Attention: Q, K, V projections -> attention -> output projection
        - FFN: intermediate (expand) -> GELU -> output (contract)
        
        LayerNorm between blocks breaks scale-invariance for cross-block equalization.
        Within FFN, we can try intermediate -> output (GELU is approximately scale-equivariant).
        """
        pairs = []
        
        for i, layer in enumerate(model.encoder.layer):
            # FFN: intermediate -> output
            ffn_inter = layer.intermediate.dense  # [768 -> 3072]
            ffn_out = layer.output.dense          # [3072 -> 768]
            
            if ffn_inter.weight.shape[0] == ffn_out.weight.shape[1]:
                pairs.append((f'layer{i}.ffn', ffn_inter, ffn_out))
        
        return pairs
    
    @staticmethod
    def get_test_input():
        return {
            'input_ids': torch.randint(0, 30000, (4, 128)),
            'attention_mask': torch.ones(4, 128, dtype=torch.long)
        }


class ViTHandler:
    """Handler for Vision Transformer."""
    
    @staticmethod
    def load_model():
        import torchvision.models as models
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        return model
    
    @staticmethod
    def fuse_bn(model: nn.Module) -> nn.Module:
        """ViT uses LayerNorm, not BatchNorm."""
        return deepcopy(model)
    
    @staticmethod
    def get_equalization_pairs(model: nn.Module) -> List[Tuple[str, nn.Module, nn.Module]]:
        """Get equalizable pairs in ViT."""
        pairs = []
        
        for i, block in enumerate(model.encoder.layers):
            # MLP: fc1 -> GELU -> fc2
            mlp = block.mlp
            if hasattr(mlp, 'linear_1') and hasattr(mlp, 'linear_2'):
                fc1 = mlp.linear_1  # This is actually just '0' in Sequential
                fc2 = mlp.linear_2
            else:
                # Try sequential indexing
                fc1 = mlp[0]
                fc2 = mlp[3]  # Skip GELU and Dropout
            
            if isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear):
                if fc1.weight.shape[0] == fc2.weight.shape[1]:
                    pairs.append((f'block{i}.mlp', fc1, fc2))
        
        return pairs
    
    @staticmethod
    def get_test_input():
        return torch.randn(4, 3, 224, 224)


# ============================================================================
# Main Analysis Functions
# ============================================================================

def analyze_architecture(handler_class, arch_name: str) -> Optional[ArchitectureResults]:
    """Run complete analysis on one architecture."""
    print(f"\n{'=' * 70}")
    print(f"Analyzing: {arch_name}")
    print('=' * 70)
    
    try:
        # Load model
        print("Loading model...", end=" ")
        model_orig = handler_class.load_model()
        model_orig.eval()
        num_params = sum(p.numel() for p in model_orig.parameters())
        print(f"✓ ({num_params/1e6:.1f}M parameters)")
        
        # Fuse BN if applicable
        print("Fusing BatchNorm...", end=" ")
        model_fused = handler_class.fuse_bn(model_orig)
        model_fused.eval()
        print("✓")
        
        # Verify fusion
        test_input = handler_class.get_test_input()
        with torch.no_grad():
            if isinstance(test_input, dict):
                y_orig = model_orig(**test_input).last_hidden_state
                y_fused = model_fused(**test_input).last_hidden_state
            else:
                y_orig = model_orig(test_input)
                y_fused = model_fused(test_input)
        
        fusion_diff = (y_orig - y_fused).abs().max().item()
        print(f"Fusion verification: max diff = {fusion_diff:.2e}", end=" ")
        if fusion_diff < 1e-4:
            print("✓")
        else:
            print("(large diff - LayerNorm not fused)")
        
        # Get equalization pairs
        pairs = handler_class.get_equalization_pairs(model_fused)
        print(f"Found {len(pairs)} equalizable layer pairs")
        
        if len(pairs) == 0:
            print("No equalizable pairs found. Skipping equalization analysis.")
            return None
        
        # Store original pairs for comparison
        model_orig_fused = handler_class.fuse_bn(model_orig)
        pairs_orig = handler_class.get_equalization_pairs(model_orig_fused)
        
        # Naive PTQ baseline
        model_naive = deepcopy(model_fused)
        for module in model_naive.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.data = symmetric_quantize(module.weight.data)
        
        with torch.no_grad():
            if isinstance(test_input, dict):
                y_naive = model_naive(**test_input).last_hidden_state
                y_fp32_ref = model_fused(**test_input).last_hidden_state
            else:
                y_naive = model_naive(test_input)
                y_fp32_ref = model_fused(test_input)
        
        naive_mae = (y_naive - y_fp32_ref).abs().mean().item()
        
        # Apply equalization
        model_eq = deepcopy(model_fused)
        pairs_eq = handler_class.get_equalization_pairs(model_eq)
        
        layer_stats = []
        for (name, l1, l2), (_, l1_orig, l2_orig) in zip(pairs_eq, pairs_orig):
            try:
                scale = compute_equalization_scale(l1.weight, l2.weight)
                
                # Get stats before applying
                stats = get_layer_pair_stats(name, l1, l2, l1_orig, l2_orig)
                
                # Apply equalization
                apply_equalization(l1, l2, scale)
                
                # Update stats after
                stats.range1_after = l1.weight.abs().max().item()
                stats.range2_after = l2.weight.abs().max().item()
                stats.ratio_after = max(stats.range1_after, stats.range2_after) / max(min(stats.range1_after, stats.range2_after), 1e-8)
                stats.mse_after = compute_weight_mse(l1.weight) + compute_weight_mse(l2.weight)
                
                layer_stats.append(stats)
            except Exception as e:
                print(f"  Warning: Could not equalize {name}: {e}")
        
        # Verify equalization preserved output
        with torch.no_grad():
            if isinstance(test_input, dict):
                y_eq_fp32 = model_eq(**test_input).last_hidden_state
                y_ref_for_eq = model_fused(**test_input).last_hidden_state
                eq_diff = (y_eq_fp32 - y_ref_for_eq).abs().max().item()
            else:
                y_eq_fp32 = model_eq(test_input)
                y_ref_for_eq = model_fused(test_input)
                eq_diff = (y_eq_fp32 - y_ref_for_eq).abs().max().item()
        
        equalization_valid = eq_diff < 1e-3
        print(f"Equalization output diff: {eq_diff:.2e}", end=" ")
        print("✓" if equalization_valid else "✗ (equalization changed output!)")
        
        # Quantize equalized model
        for module in model_eq.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.data = symmetric_quantize(module.weight.data)
        
        with torch.no_grad():
            if isinstance(test_input, dict):
                y_eq = model_eq(**test_input).last_hidden_state
                y_ref_final = model_fused(**test_input).last_hidden_state
                eq_mae = (y_eq - y_ref_final).abs().mean().item()
            else:
                y_eq = model_eq(test_input)
                y_ref_final = model_fused(test_input)
                eq_mae = (y_eq - y_ref_final).abs().mean().item()
        
        improvement = (1 - eq_mae / naive_mae) * 100 if naive_mae > 0 else 0
        
        return ArchitectureResults(
            name=arch_name,
            num_params=num_params,
            layer_pairs=layer_stats,
            naive_mae=naive_mae,
            equalized_mae=eq_mae,
            improvement_pct=improvement,
            equalization_valid=equalization_valid
        )
        
    except Exception as e:
        print(f"Error analyzing {arch_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_results_table(results: List[ArchitectureResults]):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY: Cross-Architecture Equalization Results")
    print("=" * 80)
    
    print(f"\n{'Architecture':<20} {'Params':>10} {'Pairs':>8} {'Naive MAE':>12} {'Equal MAE':>12} {'Improvement':>12} {'Valid':>8}")
    print("-" * 84)
    
    for r in results:
        if r is None:
            continue
        valid_str = "✓" if r.equalization_valid else "✗"
        print(f"{r.name:<20} {r.num_params/1e6:>9.1f}M {len(r.layer_pairs):>8} {r.naive_mae:>12.6f} {r.equalized_mae:>12.6f} {r.improvement_pct:>+11.1f}% {valid_str:>8}")
    
    print("\n" + "-" * 84)
    print("Per-Layer Pair Analysis:")
    print("-" * 84)
    
    for r in results:
        if r is None or len(r.layer_pairs) == 0:
            continue
        
        print(f"\n{r.name}:")
        print(f"  {'Layer':<20} {'Ratio Before':>14} {'Ratio After':>14} {'MSE Change':>14}")
        print(f"  {'-'*66}")
        
        for lp in r.layer_pairs[:5]:  # Show first 5
            mse_change = (lp.mse_after / lp.mse_before - 1) * 100 if lp.mse_before > 0 else 0
            print(f"  {lp.name:<20} {lp.ratio_before:>13.2f}x {lp.ratio_after:>13.2f}x {mse_change:>+13.1f}%")
        
        if len(r.layer_pairs) > 5:
            print(f"  ... and {len(r.layer_pairs) - 5} more pairs")


def generate_comparison_plot(results: List[ArchitectureResults], output_path: str = "equalization_comparison.png"):
    """Generate comparison visualization."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot generation")
        return
    
    valid_results = [r for r in results if r is not None and len(r.layer_pairs) > 0]
    
    if len(valid_results) == 0:
        print("No valid results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: MAE comparison (bar chart)
    ax1 = axes[0, 0]
    x = np.arange(len(valid_results))
    width = 0.35
    
    naive_maes = [r.naive_mae for r in valid_results]
    eq_maes = [r.equalized_mae for r in valid_results]
    names = [r.name for r in valid_results]
    
    bars1 = ax1.bar(x - width/2, naive_maes, width, label='Naive PTQ', color='#ff6b6b')
    bars2 = ax1.bar(x + width/2, eq_maes, width, label='Equalized PTQ', color='#4ecdc4')
    
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('End-to-End Quantization Error by Architecture')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax1.legend()
    ax1.set_yscale('log')
    
    # Add improvement percentages
    for i, r in enumerate(valid_results):
        if r.improvement_pct > 0:
            ax1.annotate(f'+{r.improvement_pct:.1f}%', 
                        xy=(i + width/2, eq_maes[i]), 
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=9, color='green')
        else:
            ax1.annotate(f'{r.improvement_pct:.1f}%',
                        xy=(i + width/2, eq_maes[i]),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=9, color='red')
    
    # Plot 2: Ratio improvement distribution
    ax2 = axes[0, 1]
    
    for r in valid_results:
        ratios_before = [lp.ratio_before for lp in r.layer_pairs]
        ratios_after = [lp.ratio_after for lp in r.layer_pairs]
        
        ax2.scatter(ratios_before, ratios_after, label=r.name, alpha=0.7, s=50)
    
    # Add diagonal line (no change)
    max_ratio = max(max(lp.ratio_before for r in valid_results for lp in r.layer_pairs), 
                   max(lp.ratio_after for r in valid_results for lp in r.layer_pairs))
    ax2.plot([1, max_ratio], [1, max_ratio], 'k--', alpha=0.3, label='No change')
    ax2.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Perfect balance')
    
    ax2.set_xlabel('Weight Ratio Before Equalization')
    ax2.set_ylabel('Weight Ratio After Equalization')
    ax2.set_title('Layer Pair Weight Ratios')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Plot 3: MSE change distribution
    ax3 = axes[1, 0]
    
    all_mse_changes = []
    all_arch_labels = []
    
    for r in valid_results:
        for lp in r.layer_pairs:
            if lp.mse_before > 0:
                mse_change = (lp.mse_after / lp.mse_before - 1) * 100
                all_mse_changes.append(mse_change)
                all_arch_labels.append(r.name)
    
    # Box plot
    arch_names = list(set(all_arch_labels))
    data_by_arch = [[c for c, l in zip(all_mse_changes, all_arch_labels) if l == name] for name in arch_names]
    
    bp = ax3.boxplot(data_by_arch, labels=arch_names, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(arch_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_ylabel('MSE Change (%)')
    ax3.set_title('Quantization MSE Change per Layer Pair')
    ax3.tick_params(axis='x', rotation=15)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Key Findings:\n\n"
    
    for r in valid_results:
        avg_ratio_before = np.mean([lp.ratio_before for lp in r.layer_pairs])
        avg_ratio_after = np.mean([lp.ratio_after for lp in r.layer_pairs])
        avg_mse_change = np.mean([(lp.mse_after/lp.mse_before - 1)*100 for lp in r.layer_pairs if lp.mse_before > 0])
        
        summary_text += f"{r.name}:\n"
        summary_text += f"  • Avg ratio: {avg_ratio_before:.2f}x → {avg_ratio_after:.2f}x\n"
        summary_text += f"  • Avg MSE change: {avg_mse_change:+.1f}%\n"
        summary_text += f"  • End-to-end improvement: {r.improvement_pct:+.1f}%\n"
        summary_text += f"  • Equalization valid: {'Yes' if r.equalization_valid else 'NO'}\n\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Architecture Equalization Analysis"
    )
    parser.add_argument('--all', action='store_true', help='Run all architectures')
    parser.add_argument('--arch', choices=['resnet18', 'mobilenetv2', 'bert', 'vit'],
                       help='Specific architecture to analyze')
    parser.add_argument('--plot', type=str, default='equalization_comparison.png',
                       help='Output path for comparison plot')
    
    args = parser.parse_args()
    
    handlers = {
        'resnet18': (ResNetHandler, 'ResNet-18'),
        'mobilenetv2': (MobileNetV2Handler, 'MobileNetV2'),
        'bert': (BERTHandler, 'BERT-base'),
        'vit': (ViTHandler, 'ViT-B/16'),
    }
    
    if args.all or (args.arch is None):
        archs_to_run = list(handlers.keys())
    else:
        archs_to_run = [args.arch]
    
    print("=" * 70)
    print("Chapter 4.4: Cross-Layer Equalization")
    print("Multi-Architecture Analysis")
    print("=" * 70)
    
    results = []
    for arch in archs_to_run:
        handler_class, arch_name = handlers[arch]
        result = analyze_architecture(handler_class, arch_name)
        results.append(result)
    
    # Print summary
    print_results_table(results)
    
    # Generate plot
    generate_comparison_plot(results, args.plot)
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()