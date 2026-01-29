"""
Manning-compliant Figure: Outlier Channel Analysis Across BERT Layers

Two-panel visualization showing:
1. Left: Hottest channel ratio (max/median) per layer
2. Right: Number of outlier channels (>5x median) per layer

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# =============================================================================
# MANNING STYLE SETUP
# =============================================================================

MANNING_PALETTE = {
    'blue_l3': '#0060B1',    # Normal
    'green_l3': '#80C21D',   # Moderate  
    'orange_l3': '#E37B45',  # High
    'red_l3': '#D31518',     # Severe
    'red_l1': '#F9CBCD',     # Light red for highlight zone
    'black_l2': '#808080',
}

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

# =============================================================================
# DATA
# =============================================================================

layers = list(range(12))
hottest_ratios = [4.2, 4.3, 16.0, 14.3, 15.2, 13.6, 7.8, 5.8, 4.6, 9.2, 36.1, 5.6]
outlier_counts = [0, 0, 1, 3, 8, 4, 1, 2, 0, 3, 2, 1]
hottest_channels = [1431, 1061, 1072, 1793, 2856, 2739, 2089, 294, 2598, 2184, 1046, 2120]

# =============================================================================
# FIGURE CODE
# =============================================================================

# Color coding using Manning palette
def get_ratio_color(r):
    if r > 20: return MANNING_PALETTE['red_l3']
    if r > 10: return MANNING_PALETTE['orange_l3']
    if r > 5: return MANNING_PALETTE['green_l3']
    return MANNING_PALETTE['blue_l3']

def get_count_color(c):
    if c >= 5: return MANNING_PALETTE['red_l3']
    if c >= 3: return MANNING_PALETTE['orange_l3']
    if c >= 1: return MANNING_PALETTE['green_l3']
    return MANNING_PALETTE['blue_l3']

colors_ratio = [get_ratio_color(r) for r in hottest_ratios]
colors_count = [get_count_color(c) for c in outlier_counts]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.8))

# --- Left panel: Hottest Channel Ratio ---
bars1 = ax1.bar(layers, hottest_ratios, color=colors_ratio, 
                edgecolor='black', linewidth=0.5)
ax1.axhline(y=5, color=MANNING_PALETTE['black_l2'], linestyle='--', 
            linewidth=1, label='5x threshold')
ax1.set_xlabel('BERT Layer')
ax1.set_ylabel('Hottest Channel Ratio (max / median)')
ax1.set_title('Outlier Severity by Layer', fontweight='bold')
ax1.set_xticks(layers)
ax1.set_ylim(0, 42)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Annotate extreme outlier (layer 10)
ax1.annotate(f'Ch {hottest_channels[10]}\n36.1x', 
             xy=(10, 36.1), xytext=(8, 39),
             fontsize=6, ha='center',
             arrowprops=dict(arrowstyle='->', color=MANNING_PALETTE['red_l3'], lw=1),
             color=MANNING_PALETTE['red_l3'], fontweight='bold')

# Legend
legend_elements = [
    Patch(facecolor=MANNING_PALETTE['red_l3'], label='Severe (>20x)'),
    Patch(facecolor=MANNING_PALETTE['orange_l3'], label='High (10-20x)'),
    Patch(facecolor=MANNING_PALETTE['green_l3'], label='Moderate (5-10x)'),
    Patch(facecolor=MANNING_PALETTE['blue_l3'], label='Normal (<5x)')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=5)

# --- Right panel: Number of Outlier Channels ---
bars2 = ax2.bar(layers, outlier_counts, color=colors_count, 
                edgecolor='black', linewidth=0.5)
ax2.set_xlabel('BERT Layer')
ax2.set_ylabel('Outlier Channels (>5x median)')
ax2.set_title('Outlier Channel Count by Layer', fontweight='bold')
ax2.set_xticks(layers)
ax2.set_ylim(0, 10)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Annotate most problematic layer
ax2.annotate('8 channels', 
             xy=(4, 8), xytext=(6, 9),
             fontsize=6, ha='center',
             arrowprops=dict(arrowstyle='->', color=MANNING_PALETTE['red_l3'], lw=1),
             color=MANNING_PALETTE['red_l3'], fontweight='bold')

# Value labels on non-zero bars
for bar, count in zip(bars2, outlier_counts):
    if count > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                str(count), ha='center', va='bottom', fontsize=6, fontweight='bold')

# Highlight concentration zone
ax2.axvspan(1.5, 5.5, alpha=0.15, color=MANNING_PALETTE['red_l1'])
ax2.text(3.5, 9.5, 'Middle layers concentrate outliers', 
         ha='center', fontsize=6, style='italic', color=MANNING_PALETTE['black_l2'])

plt.tight_layout()

# Save in Manning formats
fig.savefig('figures/CH03_F04_OutlierChannelAnalysis.svg', format='svg')
fig.savefig('figures/CH03_F04_OutlierChannelAnalysis.png', format='png', dpi=300)
fig.savefig('figures/CH03_F04_OutlierChannelAnalysis.pdf', format='pdf')
print("Saved: figures/CH03_F04_OutlierChannelAnalysis.svg, .png, .pdf")
plt.show()