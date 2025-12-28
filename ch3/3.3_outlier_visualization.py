"""
Figure 3.A: Outlier Channel Analysis Across BERT Layers

This script generates a two-panel visualization showing:
1. Left: Hottest channel ratio (max/median) per layer
2. Right: Number of outlier channels (>5x median) per layer

Place this figure after the outlier channel analysis output in section 3.3.3
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from actual BERT analysis
layers = list(range(12))
layer_names = [f'Layer {i}' for i in layers]

# Hottest channel ratio (max_channel_val / median)
hottest_ratios = [4.2, 4.3, 16.0, 14.3, 15.2, 13.6, 7.8, 5.8, 4.6, 9.2, 36.1, 5.6]

# Number of outlier channels (>5x median)
outlier_counts = [0, 0, 1, 3, 8, 4, 1, 2, 0, 3, 2, 1]

# Hottest channel indices (for annotation)
hottest_channels = [1431, 1061, 1072, 1793, 2856, 2739, 2089, 294, 2598, 2184, 1046, 2120]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Color scheme
colors_ratio = ['#d62728' if r > 20 else '#ff7f0e' if r > 10 else '#2ca02c' if r > 5 else '#1f77b4' 
                for r in hottest_ratios]
colors_count = ['#d62728' if c >= 5 else '#ff7f0e' if c >= 3 else '#2ca02c' if c >= 1 else '#1f77b4' 
                for c in outlier_counts]

# Left panel: Hottest Channel Ratio
bars1 = ax1.bar(layers, hottest_ratios, color=colors_ratio, edgecolor='black', linewidth=0.5)
ax1.axhline(y=5, color='gray', linestyle='--', linewidth=1, label='5x threshold')
ax1.set_xlabel('BERT Layer', fontsize=12)
ax1.set_ylabel('Hottest Channel Ratio (max / median)', fontsize=12)
ax1.set_title('Outlier Severity by Layer', fontsize=14, fontweight='bold')
ax1.set_xticks(layers)
ax1.set_ylim(0, 42)

# Annotate the extreme outlier (layer 10)
ax1.annotate(f'Channel {hottest_channels[10]}\n36.1x!', 
             xy=(10, 36.1), xytext=(8, 38),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
             color='#d62728', fontweight='bold')

# Add legend for severity levels
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', label='Severe (>20x)'),
    Patch(facecolor='#ff7f0e', label='High (10-20x)'),
    Patch(facecolor='#2ca02c', label='Moderate (5-10x)'),
    Patch(facecolor='#1f77b4', label='Normal (<5x)')
]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)

# Right panel: Number of Outlier Channels
bars2 = ax2.bar(layers, outlier_counts, color=colors_count, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('BERT Layer', fontsize=12)
ax2.set_ylabel('Number of Outlier Channels (>5x median)', fontsize=12)
ax2.set_title('Outlier Channel Count by Layer', fontsize=14, fontweight='bold')
ax2.set_xticks(layers)
ax2.set_ylim(0, 10)

# Annotate the layer with most outliers (layer 4)
ax2.annotate('8 channels\n(most problematic)', 
             xy=(4, 8), xytext=(6, 9),
             fontsize=9, ha='center',
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
             color='#d62728', fontweight='bold')

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars2, outlier_counts)):
    if count > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

# Highlight the "danger zone" layers
ax2.axvspan(1.5, 5.5, alpha=0.1, color='red', label='Concentration zone')
ax2.text(3.5, 9.5, 'Middle layers concentrate outliers', 
         ha='center', fontsize=9, style='italic', color='#666666')

plt.tight_layout()
plt.savefig('figure_3a_outlier_channel_analysis.png', dpi=150, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('figure_3a_outlier_channel_analysis.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: figure_3a_outlier_channel_analysis.png and .pdf")
plt.show()