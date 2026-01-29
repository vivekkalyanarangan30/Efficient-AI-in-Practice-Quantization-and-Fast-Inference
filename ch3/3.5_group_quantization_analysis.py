"""
Manning-compliant visualization functions for Group Quantization Analysis
Chapter 3: Granularity Choices (Section 3.5)

Replace the visualization functions in group_quantization_analysis.py with these.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

# =============================================================================
# MANNING STYLE SETUP
# =============================================================================

MANNING_PALETTE = {
    'blue_l2': '#6BA5D7',
    'blue_l3': '#0060B1',
    'green_l2': '#C2E373',
    'green_l3': '#80C21D',
    'orange_l3': '#E37B45',
    'red_l3': '#D31518',
    'purple_l3': '#773B9A',
    'yellow_l2': '#FEF180',
    'black_l2': '#808080',
    'black_l3': '#4D4D4D',
}

def setup_manning_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
        'figure.dpi': 300,
        'savefig.dpi': 300,
    })

setup_manning_style()


def save_manning_formats(fig, base_path: str):
    """Save in all Manning-required formats."""
    base = base_path.rsplit('.', 1)[0]  # Remove extension if present
    fig.savefig(f'{base}.svg', format='svg', bbox_inches='tight')
    fig.savefig(f'{base}.png', format='png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{base}.pdf', format='pdf', bbox_inches='tight')
    print(f"Saved: {base}.svg, .png, .pdf")


# =============================================================================
# FIGURE 1: Granularity Trade-off
# =============================================================================

def plot_granularity_tradeoff(results: List[Dict], output_path: str = 'CH03_F05_GranularityTradeoff'):
    """
    Create the main granularity trade-off figure.
    Manning compliant: 5.5" x 2.5"
    """
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    
    group_sizes = [r['group_size'] for r in results]
    mses = [r['mse'] for r in results]
    overheads = [r['scale_overhead_pct'] for r in results]
    compressions = [r['compression_ratio'] for r in results]
    
    # Left: MSE vs Group Size
    ax1 = axes[0]
    ax1.semilogy(group_sizes, mses, color=MANNING_PALETTE['blue_l3'], 
                 marker='o', linewidth=1.5, markersize=5)
    ax1.set_xlabel('Group Size')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('Quantization Error vs Group Size (INT4)', fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.axvline(x=128, color=MANNING_PALETTE['red_l3'], linestyle='--', 
                alpha=0.7, linewidth=1, label='GPTQ/AWQ default')
    ax1.legend(loc='upper left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: Compression vs Overhead
    ax2 = axes[1]
    
    # Color gradient based on group size
    norm_sizes = np.log2(group_sizes) / np.log2(max(group_sizes))
    colors = [plt.cm.viridis(n) for n in norm_sizes]
    
    for i, r in enumerate(results):
        ax2.scatter(r['scale_overhead_pct'], r['compression_ratio'], 
                   c=[colors[i]], s=40, zorder=3, edgecolors='black', linewidth=0.5)
        ax2.annotate(f'g={r["group_size"]}', 
                    (r['scale_overhead_pct'], r['compression_ratio']),
                    textcoords="offset points", xytext=(3, 3), fontsize=5)
    
    ax2.set_xlabel('Scale Overhead (%)')
    ax2.set_ylabel('Compression Ratio')
    ax2.set_title('Compression vs Metadata Overhead', fontweight='bold')
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_manning_formats(fig, output_path)
    plt.close()


# =============================================================================
# FIGURE 2: Memory Layout Visualization
# =============================================================================

def plot_memory_layout_visualization(output_path: str = 'CH03_F06_MemoryLayout'):
    """
    Visualize how group quantization aligns with memory layout.
    Manning compliant: 5.5" x 2.5"
    """
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    
    # Left: Row-major memory layout
    ax1 = axes[0]
    W = np.arange(12).reshape(3, 4)
    
    im1 = ax1.imshow(W, cmap='Blues', aspect='equal')
    ax1.set_title('Weight Matrix W[3, 4]\n(Row-Major Layout)', fontweight='bold')
    ax1.set_xlabel('Input Features (consecutive in memory)')
    ax1.set_ylabel('Output Features')
    
    for i in range(3):
        for j in range(4):
            ax1.text(j, i, f'{W[i,j]}', ha='center', va='center', fontsize=7)
    
    ax1.annotate('', xy=(3.5, 0), xytext=(-0.5, 0),
                arrowprops=dict(arrowstyle='->', color=MANNING_PALETTE['red_l3'], lw=1.5))
    ax1.text(1.5, -0.6, 'Memory order →', ha='center', 
             color=MANNING_PALETTE['red_l3'], fontsize=6)
    
    # Right: Grouped along input dimension
    ax2 = axes[1]
    group_size = 2
    
    # Use Manning-compatible colors
    group_colors = [
        MANNING_PALETTE['blue_l2'], MANNING_PALETTE['green_l2'],
        MANNING_PALETTE['yellow_l2'], MANNING_PALETTE['orange_l3'],
        MANNING_PALETTE['purple_l3'], MANNING_PALETTE['red_l3'],
    ]
    
    for i in range(3):
        for g in range(2):
            for j in range(2):
                col = g * 2 + j
                rect = plt.Rectangle((col - 0.4, i - 0.4), 0.8, 0.8, 
                                     facecolor=group_colors[i * 2 + g], 
                                     alpha=0.7, edgecolor='black', linewidth=0.5)
                ax2.add_patch(rect)
                ax2.text(col, i, f'{W[i, col]}', ha='center', va='center', fontsize=7)
    
    ax2.set_xlim(-0.6, 3.6)
    ax2.set_ylim(-0.6, 2.6)
    ax2.invert_yaxis()
    ax2.set_title('Grouped Along Input Dim (g=2)\nEach color = one scale', fontweight='bold')
    ax2.set_xlabel('Input Features')
    ax2.set_ylabel('Output Features')
    ax2.set_xticks(range(4))
    ax2.set_yticks(range(3))
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    save_manning_formats(fig, output_path)
    plt.close()


# =============================================================================
# FIGURE 3: INT4 Packing Diagram
# =============================================================================

def plot_packing_diagram(output_path: str = 'CH03_F07_INT4Packing'):
    """
    Visualize INT4 packing into bytes.
    Manning compliant: 5.5" x 2.0"
    """
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    
    values = [-8, 7, -3, 4, 0, -1, 2, 5]
    
    # Draw original INT4 values
    for i, v in enumerate(values):
        rect = plt.Rectangle((i * 0.65, 1.4), 0.55, 0.5, 
                             facecolor=MANNING_PALETTE['blue_l2'], 
                             edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(i * 0.65 + 0.275, 1.65, str(v), ha='center', va='center', fontsize=7)
    ax.text(-0.3, 1.65, 'INT4:', ha='right', va='center', fontsize=7, fontweight='bold')
    
    # Draw packed bytes
    packed_info = [(0, 0x0F), (1, 0x5C), (2, 0x87), (3, 0xAD)]
    
    for i, (idx, val) in enumerate(packed_info):
        rect = plt.Rectangle((i * 1.3, 0.2), 1.1, 0.5, 
                             facecolor=MANNING_PALETTE['green_l2'], 
                             edgecolor='black', linewidth=0.5)
        ax.add_patch(rect)
        ax.text(i * 1.3 + 0.55, 0.45, f'0x{val:02X}', ha='center', va='center', fontsize=7)
        
        # Arrows from pairs to packed byte
        ax.annotate('', xy=(i * 1.3 + 0.35, 0.7), xytext=(i * 2 * 0.65 + 0.275, 1.4),
                   arrowprops=dict(arrowstyle='->', color=MANNING_PALETTE['black_l2'], lw=0.8))
        ax.annotate('', xy=(i * 1.3 + 0.75, 0.7), xytext=((i * 2 + 1) * 0.65 + 0.275, 1.4),
                   arrowprops=dict(arrowstyle='->', color=MANNING_PALETTE['black_l2'], lw=0.8))
    
    ax.text(-0.3, 0.45, 'Packed:', ha='right', va='center', fontsize=7, fontweight='bold')
    
    ax.set_xlim(-0.8, 5.5)
    ax.set_ylim(-0.1, 2.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('INT4 Packing: Two 4-bit values per byte', fontsize=8, fontweight='bold', pad=10)
    
    plt.tight_layout()
    save_manning_formats(fig, output_path)
    plt.close()


# =============================================================================
# USAGE
# =============================================================================

if __name__ == '__main__':
    print("Manning-compliant visualization functions loaded.")
    print("Replace the plot_* functions in group_quantization_analysis.py with these.")
    print()
    print("Functions available:")
    print("  - plot_granularity_tradeoff(results, output_path)")
    print("  - plot_memory_layout_visualization(output_path)")
    print("  - plot_packing_diagram(output_path)")
    print()
    print("Demo: generating figures with sample data...")
    
    # Demo with sample data
    sample_results = [
        {'group_size': 32, 'mse': 1.2e-6, 'scale_overhead_pct': 12.5, 'compression_ratio': 3.56},
        {'group_size': 64, 'mse': 1.8e-6, 'scale_overhead_pct': 6.25, 'compression_ratio': 3.76},
        {'group_size': 128, 'mse': 2.9e-6, 'scale_overhead_pct': 3.13, 'compression_ratio': 3.88},
        {'group_size': 256, 'mse': 5.1e-6, 'scale_overhead_pct': 1.56, 'compression_ratio': 3.94},
        {'group_size': 512, 'mse': 9.8e-6, 'scale_overhead_pct': 0.78, 'compression_ratio': 3.97},
        {'group_size': 1024, 'mse': 1.9e-5, 'scale_overhead_pct': 0.39, 'compression_ratio': 3.98},
    ]
    
    from pathlib import Path
    Path('figures').mkdir(exist_ok=True)
    
    plot_granularity_tradeoff(sample_results, 'figures/CH03_F05_GranularityTradeoff')
    plot_memory_layout_visualization('figures/CH03_F06_MemoryLayout')
    plot_packing_diagram('figures/CH03_F07_INT4Packing')
    
    print("\nDone! Check ./figures/ directory.")