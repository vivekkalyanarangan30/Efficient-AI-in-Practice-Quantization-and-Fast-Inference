"""
Building Quantization from First Principles

Replace the visualization classes in the notebook with these versions.
All figures comply with Manning Graphics Guidelines (updated 7/31/25).

MANNING COMPLIANCE:
- Max width: 5.5 inches (within 5.6" limit)
- Fonts: Arial 7pt body, 8pt headings
- Colors: Official Manning palette with grayscale compatibility
- Patterns/hatching for B&W differentiation
- 300 DPI output

USAGE:
    python ch02_manning_visualizations.py

Or in notebook:
    from ch02_manning_visualizations import QuantVis, ErrorAnalyzer, plot_output_fidelity
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path

# Try to import torch, fall back to numpy-based simulation
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Note: torch not found, using numpy for demo data generation")

# =============================================================================
# MANNING STYLE CONFIGURATION
# =============================================================================

# Official Manning Color Palette (from Graphics Guidelines page 5)
MANNING_PALETTE = {
    'blue_l1': '#C5DFEF', 'blue_l2': '#6BA5D7', 'blue_l3': '#0060B1', 'blue_l4': '#002D8B',
    'green_l1': '#DDF8CD', 'green_l2': '#C2E373', 'green_l3': '#80C21D', 'green_l4': '#0A8902',
    'red_l1': '#F9CBCD', 'red_l2': '#F46E60', 'red_l3': '#D31518', 'red_l4': '#691210',
    'orange_l1': '#FEE3AC', 'orange_l2': '#FFB458', 'orange_l3': '#E37B45', 'orange_l4': '#CC4E01',
    'yellow_l1': '#FEFAD5', 'yellow_l2': '#FEF180', 'yellow_l3': '#E6CB00', 'yellow_l4': '#CCA000',
    'purple_l1': '#E8E6FD', 'purple_l2': '#D4ABFD', 'purple_l3': '#773B9A', 'purple_l4': '#491F6E',
    'black_l1': '#C0C0C0', 'black_l2': '#808080', 'black_l3': '#4D4D4D', 'black_l4': '#000000',
}

def setup_manning_style():
    """Configure matplotlib for Manning compliance."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })
    # Use a clean style instead of seaborn's default
    sns.set_theme(style="whitegrid")

setup_manning_style()

def save_manning_figure(fig, filename, output_dir="figures"):
    """Save figure in Manning-required formats."""
    Path(output_dir).mkdir(exist_ok=True)
    fig.savefig(f"{output_dir}/{filename}.svg", format='svg')
    fig.savefig(f"{output_dir}/{filename}.png", format='png', dpi=300)
    fig.savefig(f"{output_dir}/{filename}.pdf", format='pdf')
    print(f"Saved: {filename}.svg, .png, .pdf")


# =============================================================================
# QUANTVIS CLASS - Manning Compliant
# =============================================================================

class QuantVis:
    @staticmethod
    def _to_numpy(tensor):
        """Convert torch tensor or numpy array to numpy."""
        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            return tensor.detach().numpy().flatten()
        return np.asarray(tensor).flatten()
    
    @staticmethod
    def plot_grid_mapping(tensor, scale, zero_point, q_min, q_max, 
                          title="Quantization Grid", save_as=None):
        """
        Visualizes how continuous values map to the discrete integer grid.
        Manning compliant: 5.5" x 2", proper fonts and colors.
        """
        data = QuantVis._to_numpy(tensor)
        
        # Calculate grid points in real space
        integers = np.arange(q_min, q_max + 1)
        grid_points = scale * (integers - zero_point)
        
        fig, ax = plt.subplots(figsize=(5.5, 2.0))
        
        # Plot Data Density - Blue L2 for visibility in grayscale
        sns.stripplot(x=data, color=MANNING_PALETTE['blue_l2'], alpha=0.4, 
                      jitter=True, orient='h', ax=ax, size=3)
        
        # Plot Grid Points - Red L3 (distinct from blue in grayscale)
        ax.vlines(grid_points, -0.5, 0.5, colors=MANNING_PALETTE['red_l3'], 
                  alpha=0.5, linewidth=0.5)
        
        # Highlight Zero - Green L3 with dashed line for B&W differentiation
        zero_idx = np.abs(grid_points).argmin()
        ax.axvline(grid_points[zero_idx], color=MANNING_PALETTE['green_l3'], 
                   linewidth=2, linestyle='--', label='Real Zero (0.0)')
        
        # Title and labels
        ax.set_title(f"{title}\nScale: {scale:.6f} | Zero-Point: {zero_point} | Range: [{q_min}, {q_max}]",
                     fontsize=8, fontweight='bold')
        ax.set_xlabel("Real Value (Float32)")
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=6)
        ax.set_xlim(min(data.min(), grid_points.min()) - 0.1, 
                    max(data.max(), grid_points.max()) + 0.1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_as:
            save_manning_figure(fig, save_as)
        plt.show()

    @staticmethod
    def plot_quantization_error(original, quantized, 
                                 title="Quantization Error Analysis", save_as=None):
        """
        Plots the staircase function and error histogram.
        Manning compliant: 5.5" x 3", two-panel layout.
        """
        orig = QuantVis._to_numpy(original)
        quant = QuantVis._to_numpy(quantized)
        
        idx = np.argsort(orig)
        orig_sorted = orig[idx]
        quant_sorted = quant[idx]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.0))
        
        # Panel 1: Staircase Function
        ax1.plot(orig_sorted, orig_sorted, color=MANNING_PALETTE['black_l2'], 
                 linestyle='--', alpha=0.5, linewidth=1, label='Ideal')
        ax1.step(orig_sorted, quant_sorted, color=MANNING_PALETTE['red_l3'], 
                 where='mid', linewidth=1.5, label='Quantized')
        ax1.set_title(f"{title}: Staircase", fontsize=8, fontweight='bold')
        ax1.set_xlabel("Input (Float32)")
        ax1.set_ylabel("Output (De-quantized)")
        ax1.legend(fontsize=6)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Panel 2: Error Histogram
        error = quant - orig
        sns.histplot(error, kde=True, ax=ax2, color=MANNING_PALETTE['purple_l3'], 
                     bins=30, alpha=0.7)
        ax2.axvline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_title(f"Granular Error Distribution\nMSE: {np.mean(error**2):.6f}", 
                      fontsize=8, fontweight='bold')
        ax2.set_xlabel("Error (Quantized - Original)")
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_as:
            save_manning_figure(fig, save_as)
        plt.show()


# =============================================================================
# COMPARATOR CLASS - Manning Compliant
# =============================================================================

class Comparator:
    @staticmethod
    def compare_schemes(data, sym_q, aff_q, title, save_as=None):
        """
        Compares Symmetric vs Affine quantization.
        Manning compliant: 5.5" x 3.5", side-by-side panels.
        
        Args:
            data: numpy array or torch tensor
            sym_q: Symmetric quantizer (must have .scale and .quantize())
            aff_q: Affine quantizer (must have .scale, .zero_point, and .quantize())
        """
        # Convert to numpy
        if HAS_TORCH and hasattr(data, 'detach'):
            data_np = data.detach().numpy().flatten()
        else:
            data_np = np.asarray(data).flatten()
        
        data_sym = sym_q.quantize(data_np)
        data_aff = aff_q.quantize(data_np)
        
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.5), sharey=True)
        fig.suptitle(f"Comparison: {title}", fontsize=8, fontweight='bold')
        
        # Panel 1: Symmetric Strategy - Orange for visibility
        sns.histplot(data_sym.flatten(), ax=axes[0], color=MANNING_PALETTE['orange_l3'], 
                     bins=50, binrange=(-128, 127), alpha=0.7)
        axes[0].set_title(f"Symmetric (Signed Int8)\nRange: [-127, 127] | Z=0", fontsize=7)
        axes[0].set_xlabel("Integer Index")
        axes[0].set_xlim(-135, 135)
        
        # Wasted space visualization (for ReLU data)
        if data_np.min() >= 0:
            ylim = axes[0].get_ylim()
            rect = patches.Rectangle((-128, 0), 128, ylim[1] * 0.9,
                                      linewidth=1, edgecolor=MANNING_PALETTE['red_l3'], 
                                      facecolor=MANNING_PALETTE['red_l1'], 
                                      alpha=0.3, hatch='///')
            axes[0].add_patch(rect)
            axes[0].text(-64, ylim[1] * 0.45, "WASTED\nBIT SPACE\n(The 'ReLU Trap')", 
                         ha='center', fontsize=6, fontweight='bold',
                         color=MANNING_PALETTE['red_l4'])
        
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        
        # Panel 2: Affine Strategy - Blue for contrast
        sns.histplot(data_aff.flatten(), ax=axes[1], color=MANNING_PALETTE['blue_l3'], 
                     bins=50, binrange=(0, 255), alpha=0.7)
        axes[1].set_title(f"Asymmetric (Unsigned Int8)\nRange: [0, 255] | Z={aff_q.zero_point}", 
                          fontsize=7)
        axes[1].set_xlabel("Integer Index")
        axes[1].set_xlim(-10, 265)
        
        # Zero point marker
        if aff_q.zero_point != 0:
            axes[1].axvline(aff_q.zero_point, color=MANNING_PALETTE['green_l3'], 
                            linestyle='--', linewidth=2, label=f'Z={aff_q.zero_point}')
            axes[1].legend(fontsize=6)
        
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_as:
            save_manning_figure(fig, save_as)
        plt.show()
        
        # Textual Analysis
        print(f"--- Analysis for {title} ---")
        print(f"Symmetric: Scale={sym_q.scale:.5f}, Range=[{data_sym.min():.0f}, {data_sym.max():.0f}]")
        print(f"Affine:    Scale={aff_q.scale:.5f}, Range=[{data_aff.min():.0f}, {data_aff.max():.0f}]")
        if data_np.min() >= 0 and sym_q.scale > 0 and aff_q.scale > 0:
            print(f"Resolution Gain: Affine is {sym_q.scale / aff_q.scale:.2f}x more precise")
        print()


# =============================================================================
# ERROR ANALYZER CLASS - Manning Compliant
# =============================================================================

class ErrorAnalyzer:
    @staticmethod
    def analyze_tradeoff(data, title="Error Trade-off Analysis", save_as=None):
        """
        Visualizes granular vs overload error trade-off.
        Manning compliant: 5.5" x 3", two-panel layout.
        """
        # Convert to numpy if torch tensor
        if HAS_TORCH and isinstance(data, torch.Tensor):
            data_np = data.detach().numpy()
        else:
            data_np = np.asarray(data)
        
        max_val = np.abs(data_np).max()
        thresholds = np.linspace(max_val * 0.1, max_val * 1.2, 50)
        
        granular_errs, overload_errs, total_errs = [], [], []
        q_max = 127
        
        for thresh in thresholds:
            scale = thresh / q_max
            data_clipped = np.clip(data_np, -thresh, thresh)
            data_q = np.round(data_clipped / scale) * scale
            
            err = data_q - data_np
            overload_component = data_clipped - data_np
            granular_component = data_q - data_clipped
            
            total_errs.append(np.mean(err**2))
            overload_errs.append(np.mean(overload_component**2))
            granular_errs.append(np.mean(granular_component**2))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 3.0))
        
        best_idx = np.argmin(total_errs)
        best_thresh = thresholds[best_idx]
        
        # Panel 1: Histogram with clipping boundary
        sns.histplot(data_np, bins=80, color=MANNING_PALETTE['black_l2'], 
                     ax=ax1, alpha=0.6)
        ax1.axvline(best_thresh, color=MANNING_PALETTE['red_l3'], linestyle='--', 
                    linewidth=1.5, label=f'Clip: ±{best_thresh:.1f}')
        ax1.axvline(-best_thresh, color=MANNING_PALETTE['red_l3'], linestyle='--', 
                    linewidth=1.5)
        
        # Shade overload regions
        ylim = ax1.get_ylim()
        rect1 = patches.Rectangle((best_thresh, 0), max_val - best_thresh, ylim[1], 
                                   color=MANNING_PALETTE['red_l1'], alpha=0.4)
        rect2 = patches.Rectangle((-max_val, 0), max_val - best_thresh, ylim[1], 
                                   color=MANNING_PALETTE['red_l1'], alpha=0.4)
        ax1.add_patch(rect1)
        ax1.add_patch(rect2)
        
        ax1.set_title(f"{title}\nData Distribution & Clip Boundary", fontsize=8, fontweight='bold')
        ax1.set_xlabel("Real Value")
        ax1.legend(fontsize=6, loc='upper right')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Panel 2: Error curves
        ax2.plot(thresholds, granular_errs, color=MANNING_PALETTE['green_l3'], 
                 linewidth=1.5, label='Granular Error')
        ax2.plot(thresholds, overload_errs, color=MANNING_PALETTE['red_l3'], 
                 linewidth=1.5, label='Overload Error')
        ax2.plot(thresholds, total_errs, color=MANNING_PALETTE['black_l3'], 
                 linewidth=1.5, linestyle='--', label='Total Error')
        ax2.plot(best_thresh, total_errs[best_idx], 'ko', markersize=6, 
                 label='Optimal')
        
        ax2.set_title("Error Trade-off Curves", fontsize=8, fontweight='bold')
        ax2.set_xlabel("Clipping Threshold")
        ax2.set_ylabel("MSE")
        ax2.set_yscale('log')
        ax2.legend(fontsize=6, loc='upper right')
        ax2.grid(True, which="both", alpha=0.3, linewidth=0.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_as:
            save_manning_figure(fig, save_as)
        plt.show()
        
        print(f"--- Trade-off Analysis: {title} ---")
        print(f"Optimal Threshold: {best_thresh:.4f} ({best_thresh/max_val:.1%} of range)")
        print(f"  Granular Error: {granular_errs[best_idx]:.6f}")
        print(f"  Overload Error: {overload_errs[best_idx]:.6f}")
        print()


# =============================================================================
# OUTPUT FIDELITY PLOT - Manning Compliant
# =============================================================================

def plot_output_fidelity(output_fp32, output_8bit, title="FP32 vs INT8 Output", save_as=None):
    """
    Scatter plot comparing FP32 baseline to quantized output.
    Manning compliant: 5.5" x 4".
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    
    # Convert to numpy
    if HAS_TORCH and isinstance(output_fp32, torch.Tensor):
        fp32_vals = output_fp32.detach().numpy().flatten()
        int8_vals = output_8bit.detach().numpy().flatten()
    else:
        fp32_vals = np.asarray(output_fp32).flatten()
        int8_vals = np.asarray(output_8bit).flatten()
    
    ax.scatter(fp32_vals, int8_vals, alpha=0.4, s=3, 
               color=MANNING_PALETTE['blue_l3'], label='Samples')
    ax.plot([fp32_vals.min(), fp32_vals.max()], 
            [fp32_vals.min(), fp32_vals.max()], 
            color=MANNING_PALETTE['red_l3'], linestyle='--', 
            linewidth=1.5, label='Ideal (y=x)')
    
    ax.set_title(f"{title}\n(Hybrid Consensus Scheme)", fontsize=8, fontweight='bold')
    ax.set_xlabel("FP32 Output")
    ax.set_ylabel("INT8 Output")
    ax.legend(fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_as:
        save_manning_figure(fig, save_as)
    plt.show()


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("Generating Manning-Compliant Chapter 2 Figures...")
    print("=" * 50)
    
    # Create output directory
    Path("figures").mkdir(exist_ok=True)
    
    # Use numpy for data generation (works with or without torch)
    np.random.seed(42)
    
    # --- Figure 1: Grid Mapping (Symmetric) ---
    print("\n[1/5] Grid Mapping (Symmetric Weights)...")
    sample_weights = np.random.randn(100) * 0.5
    scale_w = np.abs(sample_weights).max() / 127
    QuantVis.plot_grid_mapping(
        sample_weights, scale_w, 0, -127, 127,
        title="Weight Quantization (Symmetric)",
        save_as="CH02_F01_GridMapping_Symmetric"
    )
    
    # --- Figure 2: Grid Mapping (Asymmetric) ---
    print("[2/5] Grid Mapping (Asymmetric Activations)...")
    sample_activations = np.maximum(np.random.randn(100), 0)  # ReLU
    r_min, r_max = sample_activations.min(), sample_activations.max()
    scale_a = (r_max - r_min) / 255
    zero_point = int(round(-r_min / scale_a)) if scale_a > 0 else 0
    QuantVis.plot_grid_mapping(
        sample_activations, scale_a, zero_point, 0, 255,
        title="Activation Quantization (Asymmetric)",
        save_as="CH02_F02_GridMapping_Asymmetric"
    )
    
    # --- Figure 3: Quantization Error (Staircase) ---
    print("[3/5] Quantization Error Analysis...")
    original = np.linspace(-1, 1, 200)
    scale = 2.0 / 255
    quantized = np.round(original / scale) * scale
    QuantVis.plot_quantization_error(
        original, quantized,
        title="INT8 Quantization",
        save_as="CH02_F03_QuantizationError"
    )
    
    # --- Figure 4: Symmetric vs Asymmetric Comparison (ReLU) ---
    print("[4/6] Symmetric vs Asymmetric Comparison...")
    relu_activations = np.maximum(np.random.randn(1000), 0)  # ReLU output
    
    # Create simple quantizer classes for the comparison
    class SimpleSymmetricQ:
        def __init__(self):
            self.scale = None
        def calibrate(self, data):
            self.scale = np.abs(data).max() / 127
        def quantize(self, data):
            return np.round(np.clip(data, -127*self.scale, 127*self.scale) / self.scale)
    
    class SimpleAffineQ:
        def __init__(self):
            self.scale = None
            self.zero_point = 0
        def calibrate(self, data):
            r_min, r_max = data.min(), data.max()
            self.scale = (r_max - r_min) / 255 if r_max > r_min else 1.0
            self.zero_point = int(round(-r_min / self.scale)) if self.scale > 0 else 0
        def quantize(self, data):
            return np.round(data / self.scale) + self.zero_point
    
    sym_q = SimpleSymmetricQ()
    sym_q.calibrate(relu_activations)
    aff_q = SimpleAffineQ()
    aff_q.calibrate(relu_activations)
    
    Comparator.compare_schemes(
        relu_activations, sym_q, aff_q,
        title="ReLU Activations (Skewed/Positive)",
        save_as="CH02_F04_SymmetricVsAsymmetric"
    )
    
    # --- Figure 5: Error Trade-off (LLM-like activations) ---
    print("[5/6] Error Trade-off Analysis...")
    n_samples = 5000
    normal_data = np.random.randn(n_samples)
    outliers_pos = np.random.rand(int(n_samples * 0.005)) * 20.0 + 5.0
    outliers_neg = np.random.rand(int(n_samples * 0.005)) * -20.0 - 5.0
    llm_activations = np.concatenate([normal_data, outliers_pos, outliers_neg])
    np.random.shuffle(llm_activations)
    
    ErrorAnalyzer.analyze_tradeoff(
        llm_activations,
        title="Heavy-Tailed Data (LLM-like)",
        save_as="CH02_F05_ErrorTradeoff"
    )
    
    # --- Figure 6: Output Fidelity ---
    print("[6/6] Output Fidelity Plot...")
    output_fp32 = np.random.randn(500) * 2
    noise = np.random.randn(500) * 0.05
    output_int8 = output_fp32 + noise
    plot_output_fidelity(
        output_fp32, output_int8,
        title="FP32 vs INT8 Output",
        save_as="CH02_F06_OutputFidelity"
    )
    
    print("\n" + "=" * 50)
    print("All figures saved to ./figures/")
    print("Files: .svg (editable), .png (300 DPI), .pdf")