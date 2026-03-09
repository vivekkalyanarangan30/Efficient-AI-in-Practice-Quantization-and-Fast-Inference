#!/usr/bin/env python3
"""
Chapter 4: PTQ Quality Analysis by Bit-Width

This script analyzes how quantization quality degrades as bit-width decreases,
helping practitioners understand the INT8 → INT4 → INT2 cliff that determines
whether PTQ will succeed or QAT is required.

Usage:
    python ch4/ptq_bitwidth_analysis.py
    python ch4/ptq_bitwidth_analysis.py --save-plot

Output:
    - Console table showing MSE, SNR, and PTQ viability by bit-width
    - Optional: Visualization saved to ch4/figures/bitwidth_snr_analysis.png
"""

import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def simulate_ptq_quality(bits: int, weight_shape: tuple = (1000, 1000), 
                         weight_scale: float = 0.02) -> dict:
    """
    Simulate PTQ quality at a given bit-width using per-channel symmetric quantization.
    
    Args:
        bits: Target bit-width (e.g., 8, 6, 4, 3, 2)
        weight_shape: Shape of simulated weight tensor
        weight_scale: Standard deviation of weight distribution
        
    Returns:
        Dictionary with MSE, SNR, and assessment
    """
    torch.manual_seed(42)
    
    # Simulate typical weight distribution (normal, centered at zero)
    weights = torch.randn(weight_shape) * weight_scale
    
    # Compute quantization parameters
    q_max = (1 << (bits - 1)) - 1  # 127 for 8-bit, 7 for 4-bit, 1 for 2-bit
    
    # Per-channel symmetric quantization (industry standard for weights)
    channel_max = weights.abs().max(dim=1, keepdim=True).values
    channel_max = torch.clamp(channel_max, min=1e-8)  # Avoid division by zero
    scales = channel_max / q_max
    
    # Quantize and dequantize
    q_weights = torch.round(weights / scales).clamp(-q_max, q_max)
    dq_weights = q_weights * scales
    
    # Compute metrics
    mse = ((weights - dq_weights) ** 2).mean().item()
    signal_power = (weights ** 2).mean().item()
    snr = 10 * np.log10(signal_power / (mse + 1e-10))
    
    # Determine assessment
    if snr > 40:
        assessment = "Excellent"
        viability = "PTQ safe"
    elif snr > 30:
        assessment = "Good"
        viability = "PTQ likely works"
    elif snr > 20:
        assessment = "Marginal"
        viability = "PTQ risky, try group quant"
    else:
        assessment = "Poor"
        viability = "QAT required"
    
    return {
        'bits': bits,
        'levels': 1 << bits,
        'mse': mse,
        'snr': snr,
        'assessment': assessment,
        'viability': viability,
        'noise_ratio': 10 ** (-snr / 10)  # Noise power / signal power
    }


def create_bitwidth_visualization(results: list, save_path: str = None):
    """
    Create a visualization showing SNR degradation and PTQ viability zones.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    bits = [r['bits'] for r in results]
    snrs = [r['snr'] for r in results]
    noise_ratios = [r['noise_ratio'] for r in results]
    
    # Color mapping based on viability
    colors = []
    for r in results:
        if r['assessment'] == 'Excellent':
            colors.append('#2ecc71')  # Green
        elif r['assessment'] == 'Good':
            colors.append('#f39c12')  # Orange
        elif r['assessment'] == 'Marginal':
            colors.append('#e74c3c')  # Red
        else:
            colors.append('#8e44ad')  # Purple
    
    # Left plot: SNR by bit-width with zones
    ax1.bar(bits, snrs, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add zone boundaries
    ax1.axhline(y=40, color='#2ecc71', linestyle='--', alpha=0.7, label='PTQ Safe (>40 dB)')
    ax1.axhline(y=30, color='#f39c12', linestyle='--', alpha=0.7, label='PTQ Marginal (>30 dB)')
    ax1.axhline(y=20, color='#e74c3c', linestyle='--', alpha=0.7, label='QAT Zone (<20 dB)')
    
    # Add SNR values on bars
    for i, (b, s) in enumerate(zip(bits, snrs)):
        ax1.text(b, s + 1.5, f'{s:.1f} dB', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Bit-Width', fontsize=12)
    ax1.set_ylabel('Signal-to-Noise Ratio (dB)', fontsize=12)
    ax1.set_title('Quantization Quality by Bit-Width', fontsize=14, fontweight='bold')
    ax1.set_xticks(bits)
    ax1.set_xticklabels([f'INT{b}' for b in bits])
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, max(snrs) + 10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right plot: Noise-to-signal ratio (intuitive interpretation)
    # Show as "1 in X" ratio for easier understanding
    inverse_ratios = [1/nr if nr > 0 else float('inf') for nr in noise_ratios]
    
    bars = ax2.bar(bits, inverse_ratios, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_yscale('log')
    
    # Add ratio labels
    for i, (b, ir) in enumerate(zip(bits, inverse_ratios)):
        if ir >= 1000:
            label = f'1:{ir/1000:.0f}K'
        else:
            label = f'1:{ir:.0f}'
        ax2.text(b, ir * 1.5, label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Bit-Width', fontsize=12)
    ax2.set_ylabel('Signal-to-Noise Ratio (linear scale)', fontsize=12)
    ax2.set_title('How Much Signal vs Noise?', fontsize=14, fontweight='bold')
    ax2.set_xticks(bits)
    ax2.set_xticklabels([f'INT{b}' for b in bits])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add interpretation text box
    textstr = 'Higher = More signal preserved\nINT8: Noise is negligible\nINT4: Noise is significant\nINT2: Noise dominates signal'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        print(f"Saved PNG to {save_path}")
        pdf_path = str(Path(save_path).with_suffix('.pdf'))
        plt.savefig(pdf_path, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved PDF to {pdf_path}")
    
    plt.show()


def print_results_table(results: list):
    """Print formatted results table."""
    print("\nPTQ Quality Analysis by Bit-Width")
    print("=" * 75)
    print(f"{'Bits':<6} {'Levels':<8} {'MSE':<12} {'SNR (dB)':<10} {'Noise Ratio':<14} {'Assessment'}")
    print("-" * 75)
    
    for r in results:
        noise_str = f"1:{1/r['noise_ratio']:.0f}" if r['noise_ratio'] > 0 else "N/A"
        print(f"{r['bits']:<6} {r['levels']:<8} {r['mse']:<12.2e} {r['snr']:<10.1f} {noise_str:<14} {r['viability']}")
    
    print("-" * 75)
    print("\nKey Insight: SNR drops ~6 dB per bit removed (noise power quadruples)")
    print("The INT8→INT4 transition crosses from 'safe' to 'marginal' territory.\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze PTQ quality by bit-width')
    parser.add_argument('--save-plot', action='store_true', 
                        help='Save visualization to ch4/figures/')
    parser.add_argument('--bits', nargs='+', type=int, default=[8, 6, 4, 3, 2],
                        help='Bit-widths to analyze (default: 8 6 4 3 2)')
    args = parser.parse_args()
    
    # Run analysis
    results = [simulate_ptq_quality(b) for b in sorted(args.bits, reverse=True)]
    
    # Print table
    print_results_table(results)
    
    # Create visualization
    save_path = 'ch4/figures/bitwidth_snr_analysis.png' if args.save_plot else None
    create_bitwidth_visualization(results, save_path)


if __name__ == '__main__':
    main()