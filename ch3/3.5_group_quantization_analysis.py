#!/usr/bin/env python3
"""
Group Quantization Analysis for Section 3.5

This script analyzes group quantization trade-offs using real transformer weights.
It generates figures and measurements referenced in the chapter.

Usage:
    python group_quantization_analysis.py --model meta-llama/Llama-2-7b-hf --bits 4
    python group_quantization_analysis.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --bits 4
    
For quick testing without downloading large models:
    python group_quantization_analysis.py --synthetic --shape 4096 4096 --bits 4
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from pathlib import Path


# =============================================================================
# Core Quantization Implementation
# =============================================================================

@dataclass
class GroupQuantizedTensor:
    """A tensor stored in group-quantized format."""
    quantized: torch.Tensor  # Shape: [out_features, in_features], dtype: int8
    scales: torch.Tensor     # Shape: [out_features, num_groups], dtype: float16
    group_size: int
    bits: int = 4
    
    def dequantize(self) -> torch.Tensor:
        """Reconstruct the floating-point tensor."""
        out_features, in_features = self.quantized.shape
        num_groups = self.scales.shape[1]
        
        q_grouped = self.quantized.reshape(out_features, num_groups, self.group_size)
        scales_expanded = self.scales.unsqueeze(-1)
        dequantized = q_grouped.float() * scales_expanded
        
        return dequantized.reshape(out_features, in_features)
    
    def memory_bytes(self) -> dict:
        """Calculate memory usage breakdown."""
        values_per_byte = 8 // self.bits
        quantized_bytes = self.quantized.numel() // values_per_byte
        scale_bytes = self.scales.numel() * 2  # FP16
        
        return {
            'quantized_bytes': quantized_bytes,
            'scale_bytes': scale_bytes,
            'total_bytes': quantized_bytes + scale_bytes
        }


def group_quantize(
    tensor: torch.Tensor,
    group_size: int = 128,
    bits: int = 4
) -> GroupQuantizedTensor:
    """
    Quantize a 2D tensor using group-wise symmetric quantization.
    """
    out_features, in_features = tensor.shape
    assert in_features % group_size == 0, \
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    
    num_groups = in_features // group_size
    q_max = (1 << (bits - 1)) - 1
    
    # Reshape and compute per-group scales
    tensor_grouped = tensor.reshape(out_features, num_groups, group_size)
    abs_max = tensor_grouped.abs().amax(dim=2)
    scales = torch.clamp(abs_max / q_max, min=1e-8)
    
    # Quantize
    scales_expanded = scales.unsqueeze(-1)
    quantized = torch.round(tensor_grouped / scales_expanded)
    quantized = quantized.clamp(-q_max - 1, q_max).to(torch.int8)
    quantized = quantized.reshape(out_features, in_features)
    
    return GroupQuantizedTensor(
        quantized=quantized,
        scales=scales.to(torch.float16),
        group_size=group_size,
        bits=bits
    )


def block_quantize(
    tensor: torch.Tensor,
    block_size: Tuple[int, int] = (32, 32),
    bits: int = 4
) -> dict:
    """
    Quantize a 2D tensor using block-wise symmetric quantization.
    """
    rows, cols = tensor.shape
    block_rows, block_cols = block_size
    assert rows % block_rows == 0 and cols % block_cols == 0
    
    q_max = (1 << (bits - 1)) - 1
    num_row_blocks = rows // block_rows
    num_col_blocks = cols // block_cols
    
    # Reshape to blocks
    tensor_blocked = tensor.reshape(
        num_row_blocks, block_rows, num_col_blocks, block_cols
    ).permute(0, 2, 1, 3)
    
    # Per-block scales and quantization
    abs_max = tensor_blocked.abs().amax(dim=(2, 3))
    scales = torch.clamp(abs_max / q_max, min=1e-8)
    
    scales_expanded = scales[:, :, None, None]
    quantized = torch.round(tensor_blocked / scales_expanded)
    quantized = quantized.clamp(-q_max - 1, q_max).to(torch.int8)
    
    # Reconstruct for error measurement
    dequantized = quantized.float() * scales_expanded
    dequantized = dequantized.permute(0, 2, 1, 3).reshape(rows, cols)
    
    mse = ((tensor - dequantized) ** 2).mean().item()
    
    return {
        'quantized': quantized,
        'scales': scales,
        'block_size': block_size,
        'mse': mse,
        'num_scales': scales.numel()
    }


# =============================================================================
# INT4 Packing Utilities
# =============================================================================

def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack pairs of INT4 values into bytes."""
    assert values.shape[-1] % 2 == 0
    values = values.clamp(-8, 7)
    values_unsigned = (values + 8).to(torch.uint8)
    
    *batch_dims, last_dim = values.shape
    values_paired = values_unsigned.reshape(*batch_dims, last_dim // 2, 2)
    packed = (values_paired[..., 0] << 4) | values_paired[..., 1]
    
    return packed


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack bytes into pairs of INT4 values."""
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    
    *batch_dims, last_dim = packed.shape
    unpacked = torch.stack([high, low], dim=-1)
    unpacked = unpacked.reshape(*batch_dims, last_dim * 2)
    
    return unpacked.to(torch.int8) - 8


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_group_sizes(
    weight: torch.Tensor,
    bits: int = 4,
    group_sizes: Optional[List[int]] = None
) -> List[Dict]:
    """
    Analyze quantization error and memory for different group sizes.
    """
    out_features, in_features = weight.shape
    
    if group_sizes is None:
        # Default: powers of 2 from 32 to full channel
        group_sizes = [g for g in [32, 64, 128, 256, 512, 1024, 2048, 4096, in_features]
                      if in_features % g == 0 and g <= in_features]
    
    results = []
    fp16_bytes = weight.numel() * 2
    
    for group_size in group_sizes:
        if in_features % group_size != 0:
            continue
            
        quantized = group_quantize(weight, group_size=group_size, bits=bits)
        reconstructed = quantized.dequantize()
        
        mse = ((weight - reconstructed) ** 2).mean().item()
        max_error = (weight - reconstructed).abs().max().item()
        mem = quantized.memory_bytes()
        
        results.append({
            'group_size': group_size,
            'mse': mse,
            'max_error': max_error,
            'num_scales': quantized.scales.numel(),
            'total_bytes': mem['total_bytes'],
            'scale_overhead_pct': mem['scale_bytes'] / mem['total_bytes'] * 100,
            'compression_ratio': fp16_bytes / mem['total_bytes']
        })
    
    return results


def compare_group_vs_block(
    weight: torch.Tensor,
    bits: int = 4
) -> Dict[str, List[Dict]]:
    """
    Compare 1D group quantization with 2D block quantization.
    """
    group_results = []
    block_results = []
    
    fp16_bytes = weight.numel() * 2
    
    # Group quantization
    for group_size in [32, 64, 128]:
        q = group_quantize(weight, group_size=group_size, bits=bits)
        reconstructed = q.dequantize()
        mse = ((weight - reconstructed) ** 2).mean().item()
        mem = q.memory_bytes()
        
        group_results.append({
            'config': f'g={group_size}',
            'mse': mse,
            'num_scales': q.scales.numel(),
            'compression': fp16_bytes / mem['total_bytes']
        })
    
    # Block quantization
    for block_size in [(128, 128), (64, 64), (32, 32)]:
        result = block_quantize(weight, block_size=block_size, bits=bits)
        quantized_bytes = weight.numel() // 2
        scale_bytes = result['num_scales'] * 2
        total_bytes = quantized_bytes + scale_bytes
        
        block_results.append({
            'config': str(block_size),
            'mse': result['mse'],
            'num_scales': result['num_scales'],
            'compression': fp16_bytes / total_bytes
        })
    
    return {'group': group_results, 'block': block_results}


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_granularity_tradeoff(results: List[Dict], output_path: str = 'granularity_tradeoff.png'):
    """
    Create the main granularity trade-off figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    group_sizes = [r['group_size'] for r in results]
    mses = [r['mse'] for r in results]
    overheads = [r['scale_overhead_pct'] for r in results]
    compressions = [r['compression_ratio'] for r in results]
    
    # Left: MSE vs Group Size
    ax1 = axes[0]
    ax1.semilogy(group_sizes, mses, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Group Size', fontsize=12)
    ax1.set_ylabel('Mean Squared Error', fontsize=12)
    ax1.set_title('Quantization Error vs Group Size (INT4)', fontsize=14)
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=128, color='r', linestyle='--', alpha=0.7, label='GPTQ/AWQ default')
    ax1.legend()
    
    # Right: Compression vs Overhead
    ax2 = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for i, r in enumerate(results):
        ax2.scatter(r['scale_overhead_pct'], r['compression_ratio'], 
                   c=[colors[i]], s=150, zorder=3)
        ax2.annotate(f'g={r["group_size"]}', 
                    (r['scale_overhead_pct'], r['compression_ratio']),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    
    ax2.set_xlabel('Scale Overhead (%)', fontsize=12)
    ax2.set_ylabel('Compression Ratio', fontsize=12)
    ax2.set_title('Compression vs Metadata Overhead', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_memory_layout_visualization(output_path: str = 'memory_layout.png'):
    """
    Visualize how group quantization aligns with memory layout.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Row-major memory layout
    ax1 = axes[0]
    W = np.arange(12).reshape(3, 4)
    
    im1 = ax1.imshow(W, cmap='Blues', aspect='equal')
    ax1.set_title('Weight Matrix W[3, 4]\n(Row-Major Layout)', fontsize=12)
    ax1.set_xlabel('Input Features (fast axis, consecutive in memory)')
    ax1.set_ylabel('Output Features')
    
    # Add value annotations
    for i in range(3):
        for j in range(4):
            ax1.text(j, i, f'{W[i,j]}', ha='center', va='center', fontsize=12)
    
    # Add memory order arrows
    ax1.annotate('', xy=(3.5, 0), xytext=(-0.5, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(1.5, -0.7, 'Memory order →', ha='center', color='red', fontsize=10)
    
    # Right: Grouped along input dimension
    ax2 = axes[1]
    group_size = 2
    colors = plt.cm.Set3(np.linspace(0, 1, 6))
    
    for i in range(3):
        for g in range(2):
            for j in range(2):
                col = g * 2 + j
                rect = plt.Rectangle((col - 0.4, i - 0.4), 0.8, 0.8, 
                                     facecolor=colors[i * 2 + g], alpha=0.7)
                ax2.add_patch(rect)
                ax2.text(col, i, f'{W[i, col]}', ha='center', va='center', fontsize=12)
    
    ax2.set_xlim(-0.6, 3.6)
    ax2.set_ylim(-0.6, 2.6)
    ax2.invert_yaxis()
    ax2.set_title('Grouped Along Input Dim (group_size=2)\nEach color = one group with one scale', fontsize=12)
    ax2.set_xlabel('Input Features')
    ax2.set_ylabel('Output Features')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(3))
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_packing_diagram(output_path: str = 'int4_packing.png'):
    """
    Visualize INT4 packing into bytes.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Original values
    values = [-8, 7, -3, 4, 0, -1, 2, 5]
    
    # Draw original values
    for i, v in enumerate(values):
        rect = plt.Rectangle((i * 1.2, 2), 1, 0.8, facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(i * 1.2 + 0.5, 2.4, str(v), ha='center', va='center', fontsize=11)
    ax.text(-0.8, 2.4, 'INT4:', ha='right', va='center', fontsize=11, fontweight='bold')
    
    # Draw packed bytes
    packed_bytes = [(0, 15, 'val₀<<4 | val₁'), (1, 92, 'val₂<<4 | val₃'), 
                    (2, 135, 'val₄<<4 | val₅'), (3, 173, 'val₆<<4 | val₇')]
    
    for i, (idx, val, label) in enumerate(packed_bytes):
        rect = plt.Rectangle((i * 2.4, 0), 2, 0.8, facecolor='lightgreen', edgecolor='black')
        ax.add_patch(rect)
        ax.text(i * 2.4 + 1, 0.4, f'0x{val:02X}', ha='center', va='center', fontsize=11)
        ax.text(i * 2.4 + 1, -0.3, label, ha='center', va='top', fontsize=9, color='gray')
        
        # Draw arrows from pairs to packed byte
        ax.annotate('', xy=(i * 2.4 + 0.5, 0.8), xytext=(i * 2 * 1.2 + 0.5, 2),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        ax.annotate('', xy=(i * 2.4 + 1.5, 0.8), xytext=(i * 2 * 1.2 + 1.7, 2),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    
    ax.text(-0.8, 0.4, 'Packed:', ha='right', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(-1.5, 10)
    ax.set_ylim(-0.8, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('INT4 Packing: Two 4-bit values per byte', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def load_weight_tensor(args) -> torch.Tensor:
    """Load weight tensor from model or generate synthetic."""
    if args.synthetic:
        print(f"Using synthetic weights: shape={args.shape}")
        torch.manual_seed(42)
        return torch.randn(args.shape[0], args.shape[1]) * 0.02
    
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    
    # Extract q_proj from middle layer
    num_layers = len(model.model.layers)
    layer_idx = num_layers // 2
    weight = model.model.layers[layer_idx].self_attn.q_proj.weight.data.float()
    
    print(f"Extracted layer {layer_idx} q_proj: shape={weight.shape}")
    return weight


def main():
    parser = argparse.ArgumentParser(description='Group Quantization Analysis')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                       help='HuggingFace model to analyze')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic weights instead of loading model')
    parser.add_argument('--shape', type=int, nargs=2, default=[4096, 4096],
                       help='Shape for synthetic weights')
    parser.add_argument('--bits', type=int, default=4, help='Quantization bits')
    parser.add_argument('--output-dir', type=str, default='./figures',
                       help='Output directory for figures')
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load weights
    weight = load_weight_tensor(args)
    
    # Run analysis
    print(f"\n{'='*60}")
    print(f"INT{args.bits} Group Quantization Analysis")
    print(f"Weight shape: {weight.shape}")
    print(f"{'='*60}\n")
    
    # 1. Analyze group sizes
    results = analyze_group_sizes(weight, bits=args.bits)
    
    print("Group Size Analysis:")
    print("-" * 75)
    print(f"{'Group Size':>12} {'MSE':>12} {'Max Error':>12} {'Overhead':>10} {'Compression':>12}")
    print("-" * 75)
    for r in results:
        print(f"{r['group_size']:>12} {r['mse']:>12.2e} {r['max_error']:>12.4f} "
              f"{r['scale_overhead_pct']:>9.1f}% {r['compression_ratio']:>11.2f}x")
    
    # 2. Compare group vs block
    print("\n\nGroup vs Block Comparison:")
    print("-" * 65)
    comparison = compare_group_vs_block(weight, bits=args.bits)
    
    print(f"{'Scheme':<15} {'Config':<15} {'MSE':>12} {'Scales':>12} {'Compression':>10}")
    print("-" * 65)
    for r in comparison['group']:
        print(f"{'Group (1D)':<15} {r['config']:<15} {r['mse']:>12.2e} {r['num_scales']:>12,} {r['compression']:>9.2f}x")
    for r in comparison['block']:
        print(f"{'Block (2D)':<15} {r['config']:<15} {r['mse']:>12.2e} {r['num_scales']:>12,} {r['compression']:>9.2f}x")
    
    # 3. Generate figures
    print("\n\nGenerating figures...")
    plot_granularity_tradeoff(results, str(output_dir / 'granularity_tradeoff.png'))
    plot_memory_layout_visualization(str(output_dir / 'memory_layout.png'))
    plot_packing_diagram(str(output_dir / 'int4_packing.png'))
    
    # 4. Verify packing
    print("\n\nINT4 Packing Verification:")
    original = torch.tensor([[-8, 7, -3, 4, 0, -1, 2, 5]], dtype=torch.int8)
    packed = pack_int4(original)
    unpacked = unpack_int4(packed)
    print(f"Original:  {original.tolist()[0]}")
    print(f"Packed:    {['0x{:02X}'.format(x) for x in packed[0].tolist()]}")
    print(f"Unpacked:  {unpacked.tolist()[0]}")
    print(f"Roundtrip: {'PASS' if torch.equal(original, unpacked) else 'FAIL'}")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete. Figures saved to {output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
