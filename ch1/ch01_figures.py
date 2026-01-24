#!/usr/bin/env python3
"""
Chapter 1 Figures: The Efficiency Crisis
Generates publication-quality diagrams for the quantization book.

Figures:
1. Energy cost hierarchy - the memory wall visualization
2. Memory traffic breakdown - where the 15GB comes from  
3. Floating point vs integer number lines - the core insight
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# Set up publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def figure1_energy_hierarchy():
    """
    Figure 1.1: The Memory Wall
    
    Shows the massive energy gap between compute and memory access.
    Key insight: fetching data costs 100-1000x more than computing on it.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data: operation types and their energy costs in picojoules
    operations = [
        'INT8\nMultiply-Add',
        'FP16\nMultiply-Add', 
        'FP32\nMultiply-Add',
        'L1 Cache\nRead',
        'L2 Cache\nRead',
        'HBM/DRAM\nRead'
    ]
    
    energy_pj = [0.3, 1.5, 4.0, 3, 20, 500]
    
    # Color scheme: green for compute, yellow-orange-red for memory
    colors = ['#4CAF50', '#66BB6A', '#81C784', '#FFC107', '#FF9800', '#F44336']
    
    # Create bars
    bars = ax.bar(operations, energy_pj, color=colors, edgecolor='black', linewidth=1.2)
    
    # Use log scale to show the range
    ax.set_yscale('log')
    ax.set_ylim(0.1, 1000)
    
    # Add value labels on bars
    for bar, val in zip(bars, energy_pj):
        height = bar.get_height()
        ax.annotate(f'{val} pJ',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
    
    # Add multiplier annotations
    ax.annotate('', xy=(5, 500), xytext=(0, 0.3),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=2))
    ax.text(2.5, 15, '~1,700×\nenergy gap', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#333',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#333', alpha=0.9))
    
    # Dividing line between compute and memory
    ax.axvline(x=2.5, color='#666', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(1, 0.15, 'COMPUTE', ha='center', fontsize=12, fontweight='bold', color='#2E7D32')
    ax.text(4.5, 0.15, 'MEMORY ACCESS', ha='center', fontsize=12, fontweight='bold', color='#C62828')
    
    # Labels and title
    ax.set_ylabel('Energy per Operation (picojoules, log scale)', fontsize=12)
    ax.set_title('The Memory Wall: Why Data Movement Dominates Inference Cost', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Clean up
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_1_energy_hierarchy.png')
    plt.savefig(OUTPUT_DIR / 'fig1_1_energy_hierarchy.pdf')
    print(f"Saved: fig1_1_energy_hierarchy.png/pdf")
    plt.close()


def figure2_memory_traffic():
    """
    Figure 1.2: Memory Traffic Scaling with Context Length
    
    Shows how memory traffic per token grows as context expands.
    Key insight: weights are constant, but KV cache explodes with context.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    context_lengths = ['512', '2K', '8K', '32K', '128K']
    weights = [14, 14, 14, 14, 14]  # constant
    kv_cache = [0.25, 1, 4, 16, 64]  # scales with context
    
    x = np.arange(len(context_lengths))
    width = 0.5
    
    bars1 = ax.bar(x, weights, width, label='Model Weights (constant)', color='#E53935')
    bars2 = ax.bar(x, kv_cache, width, bottom=weights, label='KV Cache (scales with context)', color='#FFC107')
    
    ax.set_xlabel('Context Length (tokens)', fontsize=12)
    ax.set_ylabel('Memory Traffic per Token (GB)', fontsize=12)
    ax.set_title('Memory Traffic Scales with Context Length\n(7B parameter model at FP16)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(context_lengths)
    ax.legend(loc='upper left', fontsize=10)
    
    # Add total labels on top of bars
    for i, (w, kv) in enumerate(zip(weights, kv_cache)):
        total = w + kv
        ax.annotate(f'{total:.0f} GB', xy=(i, total), xytext=(0, 5),
                    textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    
    # Add annotation highlighting the growth
    ax.annotate('', xy=(4, 78), xytext=(0, 14.25),
                arrowprops=dict(arrowstyle='->', color='#333', lw=2, 
                               connectionstyle='arc3,rad=0.2'))
    ax.text(2.5, 55, '5.5× more traffic\nat 128K context', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#333', alpha=0.9))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_ylim(0, 90)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_2_memory_traffic.png')
    plt.savefig(OUTPUT_DIR / 'fig1_2_memory_traffic.pdf')
    print(f"Saved: fig1_2_memory_traffic.png/pdf")
    plt.close()


def figure3_fp_vs_int_numberline():
    """
    Figure 1.3: Floating Point vs Integer Number Lines
    
    The core visual insight of quantization: FP has adaptive (wasteful) spacing,
    integers have uniform (efficient) spacing that you control.
    """
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.3, 1], hspace=0.4)
    
    # --- Top panel: Floating Point representation ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-0.5, 1.5)
    ax1.axis('off')
    
    # Title
    ax1.text(0, 1.3, 'FLOATING POINT: Adaptive Precision (Variable Spacing)', 
             ha='center', fontsize=13, fontweight='bold', color='#C62828')
    
    # Draw the number line
    ax1.axhline(y=0.5, xmin=0.05, xmax=0.95, color='black', linewidth=2)
    
    # Generate FP16-like tick positions (denser near zero)
    # Simulate: more values near zero, fewer far from zero
    fp_ticks_neg = -np.concatenate([
        np.linspace(0.01, 0.1, 20),
        np.linspace(0.1, 0.5, 10),
        np.linspace(0.5, 1.0, 5),
        np.linspace(1.0, 1.8, 3)
    ])
    fp_ticks_pos = np.concatenate([
        np.linspace(0.01, 0.1, 20),
        np.linspace(0.1, 0.5, 10),
        np.linspace(0.5, 1.0, 5),
        np.linspace(1.0, 1.8, 3)
    ])
    fp_ticks = np.concatenate([fp_ticks_neg, [0], fp_ticks_pos])
    
    # Draw ticks
    for tick in fp_ticks:
        ax1.plot([tick, tick], [0.4, 0.6], color='#C62828', linewidth=1, alpha=0.7)
    
    # Highlight dense region
    ax1.fill_between([-0.15, 0.15], 0.2, 0.8, color='#FFCDD2', alpha=0.5)
    ax1.annotate('Dense:\n~50% of values\nconcentrated here', 
                 xy=(0, 0.3), xytext=(0, -0.1),
                 ha='center', fontsize=10, color='#C62828',
                 arrowprops=dict(arrowstyle='->', color='#C62828'))
    
    # Highlight sparse regions
    ax1.annotate('Sparse', xy=(-1.3, 0.5), xytext=(-1.3, 0.9),
                 ha='center', fontsize=10, color='#666',
                 arrowprops=dict(arrowstyle='->', color='#666'))
    ax1.annotate('Sparse', xy=(1.3, 0.5), xytext=(1.3, 0.9),
                 ha='center', fontsize=10, color='#666',
                 arrowprops=dict(arrowstyle='->', color='#666'))
    
    # Labels
    for val in [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]:
        ax1.text(val, 0.15, str(val), ha='center', fontsize=9)
    
    # --- Middle panel: The transformation ---
    ax_mid = fig.add_subplot(gs[1])
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    ax_mid.axis('off')
    
    ax_mid.annotate('', xy=(0.5, 0.2), xytext=(0.5, 0.8),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=3))
    ax_mid.text(0.5, 0.5, 'QUANTIZATION\n\n① Choose range to cover\n② Divide into 256 equal steps\n③ Map each value to nearest step', 
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=2))
    
    # --- Bottom panel: Integer representation ---
    ax2 = fig.add_subplot(gs[2])
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-0.5, 1.5)
    ax2.axis('off')
    
    # Title
    ax2.text(0, 1.3, 'INTEGER (INT8): Uniform Precision (Fixed Spacing)', 
             ha='center', fontsize=13, fontweight='bold', color='#2E7D32')
    
    # Draw the number line
    ax2.axhline(y=0.5, xmin=0.05, xmax=0.95, color='black', linewidth=2)
    
    # INT8 has 256 uniformly spaced values
    # If we map range [-1.5, 1.5] to INT8, step size = 3.0/256 ≈ 0.0117
    int_range = (-1.5, 1.5)
    num_ticks = 64  # Show subset for clarity
    int_ticks = np.linspace(int_range[0], int_range[1], num_ticks)
    
    # Draw ticks - all evenly spaced
    for tick in int_ticks:
        ax2.plot([tick, tick], [0.4, 0.6], color='#2E7D32', linewidth=1, alpha=0.7)
    
    # Highlight uniform spacing
    ax2.annotate('', xy=(-0.5, 0.75), xytext=(0.5, 0.75),
                 arrowprops=dict(arrowstyle='<->', color='#2E7D32', lw=2))
    ax2.text(0, 0.9, 'Same spacing everywhere', ha='center', fontsize=10, color='#2E7D32')
    
    # Show the range choice
    ax2.fill_between([int_range[0], int_range[1]], 0.35, 0.65, color='#C8E6C9', alpha=0.3)
    ax2.plot([int_range[0], int_range[0]], [0.2, 0.8], 'k--', linewidth=2)
    ax2.plot([int_range[1], int_range[1]], [0.2, 0.8], 'k--', linewidth=2)
    ax2.text(int_range[0], 0.1, 'min\n(you choose)', ha='center', fontsize=9)
    ax2.text(int_range[1], 0.1, 'max\n(you choose)', ha='center', fontsize=9)
    
    # Labels
    for val in [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]:
        ax2.text(val, 0.15, str(val), ha='center', fontsize=9)
    
    # Bottom annotation
    ax2.text(0, -0.35, 'Key insight: You choose where to spend your 256 values.\n'
             'Place the grid where your weights actually live.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#2E7D32'))
    
    plt.savefig(OUTPUT_DIR / 'fig1_3_fp_vs_int_numberline.png')
    plt.savefig(OUTPUT_DIR / 'fig1_3_fp_vs_int_numberline.pdf')
    print(f"Saved: fig1_3_fp_vs_int_numberline.png/pdf")
    plt.close()


if __name__ == '__main__':
    print("Generating Chapter 1 figures...")
    print("=" * 50)
    
    figure1_energy_hierarchy()
    figure2_memory_traffic()
    figure3_fp_vs_int_numberline()
    
    print("=" * 50)
    print(f"All figures saved to: {OUTPUT_DIR.absolute()}")
