#!/usr/bin/env python3
"""
Chapter 1 Figures: The Efficiency Crisis
Manning-compliant publication-quality diagrams for the quantization book.

MANNING COMPLIANCE (per Graphics Guidelines updated 7/31/25):
- Max width: 5.6 inches (403.2 pts), Max height: 7 inches (504 pts)
- Fonts: Arial 7pt body, 8pt headings
- Colors: USE MANNING PALETTE - colors work in grayscale
- Don't REFER to colors in text (e.g., "the red line indicates...")
- Output: SVG (editable) + PNG (300 DPI reference)
- File naming: CH01_F01_ExternalID format
- Use #A, #B, #C instead of cueballs for steps

MANNING COLOR PALETTE (from page 5 of guidelines):
- Level 1: Lightest (pastels)
- Level 2: Light-medium  
- Level 3: Medium-dark
- Level 4: Darkest

Colors chosen to have good grayscale separation when printed B&W.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

# ============================================================================
# OFFICIAL MANNING COLOR PALETTE (from Graphics Guidelines page 5)
# RGB values extracted from the document
# ============================================================================

MANNING_PALETTE = {
    # Black family (grayscale)
    'black_l1': '#C0C0C0',  # K=25
    'black_l2': '#808080',  # K=50
    'black_l3': '#4D4D4D',  # K=75
    'black_l4': '#000000',  # K=100
    
    # Brown family
    'brown_l1': '#ECD5B8',  # R=236 G=213 B=184
    'brown_l2': '#D1A672',  # R=209 G=166 B=114
    'brown_l3': '#834622',  # R=131 G=70 B=34
    'brown_l4': '#591F06',  # R=89 G=31 B=6
    
    # Pink family
    'pink_l1': '#FFC7D8',   # R=255 G=199 B=216
    'pink_l2': '#E69EC5',   # R=230 G=158 B=197
    'pink_l3': '#D7519D',   # R=215 G=81 B=157
    'pink_l4': '#9D0064',   # R=157 G=0 B=100
    
    # Red family
    'red_l1': '#F9CBCD',    # R=249 G=203 B=205
    'red_l2': '#F46E60',    # R=244 G=110 B=96
    'red_l3': '#D31518',    # R=211 G=21 B=24
    'red_l4': '#691210',    # R=105 G=18 B=16
    
    # Orange family
    'orange_l1': '#FEE3AC', # R=254 G=227 B=172
    'orange_l2': '#FFB458', # R=255 G=180 B=88
    'orange_l3': '#E37B45', # R=227 G=123 B=69
    'orange_l4': '#CC4E01', # R=204 G=78 B=1
    
    # Yellow family
    'yellow_l1': '#FEFAD5', # R=254 G=250 B=213
    'yellow_l2': '#FEF180', # R=254 G=241 B=128
    'yellow_l3': '#E6CB00', # R=230 G=203 B=0
    'yellow_l4': '#CCA000', # R=204 G=160 B=0
    
    # Green family
    'green_l1': '#DDF8CD',  # R=221 G=248 B=205
    'green_l2': '#C2E373',  # R=194 G=227 B=115
    'green_l3': '#80C21D',  # R=128 G=194 B=29
    'green_l4': '#0A8902',  # R=10 G=137 B=2
    
    # Blue family
    'blue_l1': '#C5DFEF',   # R=197 G=223 B=239
    'blue_l2': '#6BA5D7',   # R=107 G=165 B=215
    'blue_l3': '#0060B1',   # R=0 G=96 B=177
    'blue_l4': '#002D8B',   # R=0 G=45 B=139
    
    # Purple family
    'purple_l1': '#E8E6FD', # R=232 G=230 B=253
    'purple_l2': '#D4ABFD', # R=212 G=171 B=253
    'purple_l3': '#773B9A', # R=119 G=59 B=154
    'purple_l4': '#491F6E', # R=73 G=31 B=110
}

# ============================================================================
# MANNING STYLE DEFAULTS
# ============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 7,                  # Manning: 7pt for body text
    'axes.titlesize': 8,             # Manning: 8pt for headings
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,              # Manning: 300 DPI minimum
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)


def save_manning_formats(fig, base_name):
    """Save figure in Manning-required formats with proper naming."""
    # SVG for editability (Manning preferred)
    fig.savefig(OUTPUT_DIR / f'{base_name}.svg', format='svg')
    # PNG at 300 DPI for reference
    fig.savefig(OUTPUT_DIR / f'{base_name}.png', format='png', dpi=300)
    # PDF as backup editable format
    fig.savefig(OUTPUT_DIR / f'{base_name}.pdf', format='pdf')
    print(f"Saved: {base_name}.svg, .png, .pdf")


def figure1_energy_hierarchy():
    """
    Figure 1.1: The Memory Wall
    
    Shows the massive energy gap between compute and memory access.
    Key insight: fetching data costs 100-1000x more than computing on it.
    
    Color strategy: 
    - Compute operations: Green family (L1 → L3) - efficiency/good
    - Memory operations: Orange/Red family (L2 → L4) - cost/bottleneck
    - These have excellent grayscale separation per Manning palette
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    
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
    
    # Manning palette colors with good grayscale separation
    # Compute: Green family (light to dark = fast to slower)
    # Memory: Orange→Red family (showing increasing cost)
    colors = [
        MANNING_PALETTE['green_l1'],   # INT8 - lightest green (most efficient)
        MANNING_PALETTE['green_l2'],   # FP16 - medium green
        MANNING_PALETTE['green_l3'],   # FP32 - darker green
        MANNING_PALETTE['orange_l2'],  # L1 Cache - light orange
        MANNING_PALETTE['orange_l3'],  # L2 Cache - medium orange  
        MANNING_PALETTE['red_l3'],     # HBM/DRAM - dark red (most expensive)
    ]
    
    # Create bars
    bars = ax.bar(operations, energy_pj, color=colors, 
                  edgecolor='black', linewidth=0.75)
    
    # Add subtle hatching to memory bars for extra B&W differentiation
    bars[4].set_hatch('///')
    bars[5].set_hatch('xxx')
    
    # Use log scale to show the range
    ax.set_yscale('log')
    ax.set_ylim(0.1, 1000)
    
    # Add value labels on bars
    for bar, val in zip(bars, energy_pj):
        height = bar.get_height()
        # Position label above bar
        ax.annotate(f'{val} pJ',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=7, fontweight='bold')
    
    # Add multiplier annotation - vertical arrow on right side to avoid ALL label overlaps
    # Position it to the right of the last bar
    arrow_x = 5.7
    ax.annotate('', xy=(arrow_x, 500), xytext=(arrow_x, 0.3),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
                annotation_clip=False)
    ax.text(arrow_x + 0.15, 8, '~1,700x\nenergy\ngap', ha='left', va='center',
            fontsize=7, fontweight='bold', color='black', clip_on=False)
    
    # Expand x-axis to make room for the arrow
    ax.set_xlim(-0.6, 6.5)
    
    # Dividing line between compute and memory
    ax.axvline(x=2.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Section labels - position in whitespace above the bars
    ax.text(1, 700, 'COMPUTE', ha='center', fontsize=7, 
            fontweight='bold', color='black')
    ax.text(3.5, 700, 'MEMORY ACCESS', ha='center', fontsize=7, 
            fontweight='bold', color='black')
    
    # Labels and title
    ax.set_ylabel('Energy per Operation (picojoules, log scale)', fontsize=7)
    ax.set_title('The Memory Wall: Why Data Movement Dominates Inference Cost', 
                 fontsize=8, fontweight='bold', pad=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    save_manning_formats(fig, 'CH01_F01_EnergyHierarchy')
    plt.close()


def figure2_memory_traffic():
    """
    Figure 1.2: Memory Traffic Scaling with Context Length
    
    Shows how memory traffic per token grows as context expands.
    Key insight: weights are constant, but KV cache explodes with context.
    
    Color strategy:
    - Model Weights (constant): Blue L3 - solid, stable
    - KV Cache (growing): Yellow L3 - attention-grabbing, growth
    - Blue and Yellow have excellent grayscale separation (see palette page 5)
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    
    context_lengths = ['512', '2K', '8K', '32K', '128K']
    weights = [14, 14, 14, 14, 14]  # constant
    kv_cache = [0.25, 1, 4, 16, 64]  # scales with context
    
    x = np.arange(len(context_lengths))
    width = 0.6
    
    # Blue L3 for weights (converts to dark gray)
    # Yellow L3 for KV cache (converts to light gray) - great separation!
    bars1 = ax.bar(x, weights, width, 
                   label='Model Weights (constant)', 
                   color=MANNING_PALETTE['blue_l3'], 
                   edgecolor='black', linewidth=0.75)
    bars2 = ax.bar(x, kv_cache, width, bottom=weights, 
                   label='KV Cache (scales with context)', 
                   color=MANNING_PALETTE['yellow_l3'], 
                   edgecolor='black', linewidth=0.75,
                   hatch='///')  # Pattern for extra B&W clarity
    
    ax.set_xlabel('Context Length (tokens)', fontsize=7)
    ax.set_ylabel('Memory Traffic per Token (GB)', fontsize=7)
    ax.set_title('Memory Traffic Scales with Context Length\n(7B parameter model at FP16)', 
                 fontsize=8, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(context_lengths)
    
    # Legend with pattern indicators visible
    ax.legend(loc='upper left', fontsize=7, framealpha=0.95)
    
    # Add total labels on top of bars
    for i, (w, kv) in enumerate(zip(weights, kv_cache)):
        total = w + kv
        ax.annotate(f'{total:.0f} GB', xy=(i, total), xytext=(0, 3),
                    textcoords='offset points', ha='center', 
                    fontsize=7, fontweight='bold')
    
    # Growth annotation
    ax.annotate('', xy=(4, 78), xytext=(0.3, 20),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5, 
                               connectionstyle='arc3,rad=0.2'))
    ax.text(2.5, 55, '5.5x more traffic\nat 128K context', ha='center', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                      edgecolor='black', alpha=0.95))
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 90)
    
    plt.tight_layout()
    save_manning_formats(fig, 'CH01_F02_MemoryTraffic')
    plt.close()


def figure3_fp_vs_int_numberline():
    """
    Figure 1.3: Floating Point vs Integer Number Lines
    
    The core visual insight of quantization: FP has adaptive (wasteful) spacing,
    integers have uniform (efficient) spacing that you control.
    
    Color strategy:
    - Floating point: Red L3 (problem/inefficiency)
    - Integer/Quantized: Green L3 (solution/efficiency)
    - Highlight boxes: Yellow L1/L2 (attention)
    """
    fig = plt.figure(figsize=(5.5, 6.5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 0.35, 1], hspace=0.3)
    
    # --- Top panel: Floating Point representation ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-0.5, 1.6)
    ax1.axis('off')
    
    # Title
    ax1.text(0, 1.35, 'FLOATING POINT: Adaptive Precision (Variable Spacing)', 
             ha='center', fontsize=8, fontweight='bold')
    
    # Draw the number line
    ax1.axhline(y=0.5, xmin=0.05, xmax=0.95, color='black', linewidth=1.5)
    
    # Generate FP16-like tick positions (denser near zero)
    fp_ticks_neg = -np.concatenate([
        np.linspace(0.01, 0.1, 15),
        np.linspace(0.1, 0.5, 8),
        np.linspace(0.5, 1.0, 4),
        np.linspace(1.0, 1.8, 2)
    ])
    fp_ticks_pos = np.concatenate([
        np.linspace(0.01, 0.1, 15),
        np.linspace(0.1, 0.5, 8),
        np.linspace(0.5, 1.0, 4),
        np.linspace(1.0, 1.8, 2)
    ])
    fp_ticks = np.concatenate([fp_ticks_neg, [0], fp_ticks_pos])
    
    # Draw ticks using Red L3 (the "problem" color)
    for tick in fp_ticks:
        ax1.plot([tick, tick], [0.4, 0.6], 
                 color=MANNING_PALETTE['red_l3'], linewidth=0.8, alpha=0.8)
    
    # Highlight dense region with Yellow L1 (attention)
    ax1.fill_between([-0.15, 0.15], 0.25, 0.75, 
                     color=MANNING_PALETTE['yellow_l1'], alpha=0.7)
    ax1.annotate('Dense region:\n~50% of values', 
                 xy=(0, 0.75), xytext=(0, 1.1),
                 ha='center', fontsize=7,
                 arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    # Sparse region annotations
    ax1.annotate('Sparse', xy=(-1.3, 0.5), xytext=(-1.3, 0.95),
                 ha='center', fontsize=7, color=MANNING_PALETTE['black_l3'],
                 arrowprops=dict(arrowstyle='->', color=MANNING_PALETTE['black_l3'], lw=0.8))
    ax1.annotate('Sparse', xy=(1.3, 0.5), xytext=(1.3, 0.95),
                 ha='center', fontsize=7, color=MANNING_PALETTE['black_l3'],
                 arrowprops=dict(arrowstyle='->', color=MANNING_PALETTE['black_l3'], lw=0.8))
    
    # Number labels
    for val in [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]:
        ax1.text(val, 0.15, str(val), ha='center', fontsize=7)
    
    # --- Middle panel: The transformation ---
    ax_mid = fig.add_subplot(gs[1])
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)
    ax_mid.axis('off')
    
    # Transformation arrow
    ax_mid.annotate('', xy=(0.5, 0.15), xytext=(0.5, 0.85),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Use #A, #B, #C format per Manning cueball guidelines
    ax_mid.text(0.5, 0.5, 
                'QUANTIZATION\n\n'
                '#A  Choose range to cover\n'
                '#B  Divide into 256 equal steps\n'
                '#C  Map each value to nearest step', 
                ha='center', va='center', fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', 
                          facecolor=MANNING_PALETTE['yellow_l1'], 
                          edgecolor='black', linewidth=1.5))
    
    # --- Bottom panel: Integer representation ---
    ax2 = fig.add_subplot(gs[2])
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-0.5, 1.5)
    ax2.axis('off')
    
    # Title
    ax2.text(0, 1.25, 'INTEGER (INT8): Uniform Precision (Fixed Spacing)', 
             ha='center', fontsize=8, fontweight='bold')
    
    # Draw the number line
    ax2.axhline(y=0.5, xmin=0.05, xmax=0.95, color='black', linewidth=1.5)
    
    # INT8 range
    int_range = (-1.5, 1.5)
    num_ticks = 48
    int_ticks = np.linspace(int_range[0], int_range[1], num_ticks)
    
    # Draw ticks using Green L3 (the "solution" color)
    for tick in int_ticks:
        ax2.plot([tick, tick], [0.4, 0.6], 
                 color=MANNING_PALETTE['green_l3'], linewidth=0.8, alpha=0.8)
    
    # Highlight uniform spacing
    ax2.annotate('', xy=(-0.5, 0.8), xytext=(0.5, 0.8),
                 arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax2.text(0, 0.95, 'Same spacing everywhere', ha='center', fontsize=7)
    
    # Range boundaries with Green L1 fill
    ax2.fill_between([int_range[0], int_range[1]], 0.35, 0.65, 
                     color=MANNING_PALETTE['green_l1'], alpha=0.4)
    ax2.plot([int_range[0], int_range[0]], [0.2, 0.8], 'k--', linewidth=1.5)
    ax2.plot([int_range[1], int_range[1]], [0.2, 0.8], 'k--', linewidth=1.5)
    ax2.text(int_range[0], 1.0, 'min', ha='center', fontsize=6, fontweight='bold')
    ax2.text(int_range[1], 1.0, 'max', ha='center', fontsize=6, fontweight='bold')
    
    # Number labels
    for val in [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]:
        ax2.text(val, 0.15, str(val), ha='center', fontsize=7)
    
    # Key insight box
    ax2.text(0, -0.35, 
             'Key insight: You choose where to spend your 256 values.\n'
             'Place the grid where your weights actually live.',
             ha='center', fontsize=7, style='italic',
             bbox=dict(boxstyle='round,pad=0.4', 
                       facecolor=MANNING_PALETTE['green_l1'], 
                       edgecolor='black', linewidth=1))
    
    save_manning_formats(fig, 'CH01_F03_FPvsIntNumberline')
    plt.close()


def verify_grayscale_compatibility():
    """Generate grayscale versions for print verification."""
    from PIL import Image
    
    for fig_name in ['CH01_F01_EnergyHierarchy', 'CH01_F02_MemoryTraffic', 
                     'CH01_F03_FPvsIntNumberline']:
        png_path = OUTPUT_DIR / f'{fig_name}.png'
        if png_path.exists():
            img = Image.open(png_path).convert('L')
            gray_path = OUTPUT_DIR / f'{fig_name}_grayscale.png'
            img.save(gray_path)
            print(f"Grayscale verification: {gray_path}")


if __name__ == '__main__':
    print("Generating Manning-compliant Chapter 1 figures...")
    print("=" * 60)
    print("Using OFFICIAL MANNING COLOR PALETTE")
    print("Target specs:")
    print("  - Max width: 5.5 inches (within 5.6 inch limit)")
    print("  - Resolution: 300 DPI")
    print("  - Fonts: Arial 7pt body, 8pt headings")
    print("  - Colors: Manning palette (grayscale-compatible)")
    print("  - File naming: CH01_F0N_ExternalID")
    print("=" * 60)
    
    figure1_energy_hierarchy()
    figure2_memory_traffic()
    figure3_fp_vs_int_numberline()
    
    print("=" * 60)
    print("Generating grayscale verification images...")
    try:
        verify_grayscale_compatibility()
    except ImportError:
        print("  (PIL not available - skipping grayscale verification)")
    
    print("=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR.absolute()}")
    print("\nManning delivery checklist:")
    print("  [x] SVG files for editability")
    print("  [x] PNG files at 300 DPI for reference")
    print("  [x] PDF files as backup")
    print("  [x] File naming: CH01_F0N_ExternalID")
    print("  [x] No color REFERENCES in text")
    print("  [x] Colors from official Manning palette")
    print("  [x] Pattern differentiation for print clarity")
    print("  [x] #A, #B, #C labels (not cueballs)")