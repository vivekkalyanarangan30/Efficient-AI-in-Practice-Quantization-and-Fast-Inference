#!/usr/bin/env python3
"""
Section 4.3 Companion Script: Range Estimation Methods for PTQ Calibration

This script demonstrates the three fundamental range estimation strategies:
1. Absolute Maximum (absmax) - Simple but sensitive to outliers
2. Percentile Clipping - Trades bounded clipping for better resolution
3. MSE-Optimal - Directly minimizes reconstruction error

Usage:
    python range_estimation_demo.py --model bert --method all
    python range_estimation_demo.py --model resnet --percentile 99.9
    python range_estimation_demo.py --model synthetic --sweep
    
Requirements:
    pip install torch transformers torchvision matplotlib numpy
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CalibrationResult:
    """Container for calibration results from different methods."""
    method: str
    range_val: float
    scale: float
    clipped_pct: float
    mse: float
    snr_db: float


class AbsMaxObserver:
    """Simplest range estimation: use absolute maximum."""
    
    def __init__(self):
        self.max_val = 0.0
    
    def observe(self, tensor: torch.Tensor):
        self.max_val = max(self.max_val, tensor.abs().max().item())
    
    def compute_range(self) -> float:
        return self.max_val


class PercentileObserver:
    """Range estimation using percentile clipping."""
    
    def __init__(self, percentile: float = 99.99):
        self.percentile = percentile
        self.values = []
    
    def observe(self, tensor: torch.Tensor):
        self.values.append(tensor.abs().flatten().cpu())
    
    def compute_range(self) -> float:
        all_vals = torch.cat(self.values)
        return torch.quantile(all_vals, self.percentile / 100).item()


class MSEOptimalObserver:
    """Range estimation that minimizes quantization MSE."""
    
    def __init__(self, num_bins: int = 2048, num_candidates: int = 200):
        self.num_bins = num_bins
        self.num_candidates = num_candidates
        self.histogram = None
        self.max_val = 0.0
    
    def observe(self, tensor: torch.Tensor):
        abs_vals = tensor.abs().float()
        curr_max = abs_vals.max().item()
        
        if curr_max > self.max_val:
            self.max_val = curr_max
        
        hist = torch.histc(abs_vals.flatten(), bins=self.num_bins, 
                          min=0, max=self.max_val)
        
        if self.histogram is None:
            self.histogram = hist.cpu()
        else:
            if len(hist) == len(self.histogram):
                self.histogram += hist.cpu()
    
    def compute_range(self) -> float:
        candidates = torch.linspace(0.5 * self.max_val, self.max_val, 
                                   self.num_candidates)
        
        bin_width = self.max_val / self.num_bins
        bin_centers = torch.arange(self.num_bins) * bin_width + bin_width / 2
        
        best_mse = float('inf')
        best_range = self.max_val
        
        for r in candidates:
            scale = r / 127
            clipped = torch.clamp(bin_centers, 0, r.item())
            quantized = torch.round(clipped / scale) * scale
            squared_error = (bin_centers - quantized) ** 2
            mse = (squared_error * self.histogram).sum() / self.histogram.sum()
            
            if mse < best_mse:
                best_mse = mse
                best_range = r.item()
        
        return best_range


class EntropyObserver:
    """TensorRT-style entropy (KL-divergence) calibration."""
    
    def __init__(self, num_bins: int = 2048, num_quant_bins: int = 128):
        self.num_bins = num_bins
        self.num_quant_bins = num_quant_bins
        self.histogram = None
        self.max_val = 0.0
    
    def observe(self, tensor: torch.Tensor):
        abs_vals = tensor.abs().float()
        curr_max = abs_vals.max().item()
        
        if curr_max > self.max_val:
            self.max_val = curr_max
        
        hist = torch.histc(abs_vals.flatten(), bins=self.num_bins,
                          min=0, max=self.max_val)
        
        if self.histogram is None:
            self.histogram = hist.cpu()
        else:
            if len(hist) == len(self.histogram):
                self.histogram += hist.cpu()
    
    def compute_range(self) -> float:
        hist = self.histogram.float()
        hist = hist / hist.sum()
        
        best_divergence = float('inf')
        best_threshold_bin = self.num_bins
        
        for threshold_bin in range(self.num_quant_bins, self.num_bins + 1):
            reference = hist[:threshold_bin].clone()
            if threshold_bin < self.num_bins:
                reference[-1] += hist[threshold_bin:].sum()
            
            if reference.sum() == 0:
                continue
            
            quantized = self._quantize_histogram(reference, self.num_quant_bins)
            expanded = self._expand_histogram(quantized, threshold_bin)
            divergence = self._kl_divergence(reference, expanded)
            
            if divergence < best_divergence:
                best_divergence = divergence
                best_threshold_bin = threshold_bin
        
        bin_width = self.max_val / self.num_bins
        return best_threshold_bin * bin_width
    
    def _quantize_histogram(self, hist: torch.Tensor, num_bins: int) -> torch.Tensor:
        original_bins = len(hist)
        bin_size = original_bins / num_bins
        result = torch.zeros(num_bins)
        for i in range(num_bins):
            start = int(i * bin_size)
            end = int((i + 1) * bin_size)
            result[i] = hist[start:end].sum()
        return result
    
    def _expand_histogram(self, hist: torch.Tensor, target_bins: int) -> torch.Tensor:
        num_bins = len(hist)
        bin_size = target_bins / num_bins
        result = torch.zeros(target_bins)
        for i in range(num_bins):
            start = int(i * bin_size)
            end = int((i + 1) * bin_size)
            if end > start:
                result[start:end] = hist[i] / (end - start)
        return result
    
    def _kl_divergence(self, p: torch.Tensor, q: torch.Tensor) -> float:
        eps = 1e-10
        p = p + eps
        q = q + eps
        return (p * torch.log(p / q)).sum().item()


def create_observers(percentile: float = 99.99) -> Dict[str, object]:
    """Create all observer types."""
    return {
        'absmax': AbsMaxObserver(),
        'percentile': PercentileObserver(percentile),
        'mse': MSEOptimalObserver(),
        'entropy': EntropyObserver()
    }


def quantize_tensor(tensor: torch.Tensor, range_val: float, bits: int = 8) -> torch.Tensor:
    """Symmetric quantization with given range."""
    q_max = (1 << (bits - 1)) - 1
    scale = range_val / q_max
    clipped = tensor.clamp(-range_val, range_val)
    quantized = torch.round(clipped / scale)
    dequantized = quantized * scale
    return dequantized


def compute_metrics(original: torch.Tensor, range_val: float) -> CalibrationResult:
    """Compute calibration metrics for a given range."""
    quantized = quantize_tensor(original, range_val)
    error = original - quantized
    mse = (error ** 2).mean().item()
    clipped_mask = original.abs() > range_val
    clipped_pct = clipped_mask.float().mean().item() * 100
    signal_power = (original ** 2).mean().item()
    noise_power = mse
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    scale = range_val / 127
    
    return CalibrationResult(
        method='',
        range_val=range_val,
        scale=scale,
        clipped_pct=clipped_pct,
        mse=mse,
        snr_db=snr_db
    )


def analyze_distribution(tensor: torch.Tensor) -> Dict:
    """Compute distribution statistics for analysis."""
    abs_vals = tensor.abs()
    return {
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'abs_max': abs_vals.max().item(),
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'median': tensor.median().item(),
        'p90': torch.quantile(abs_vals, 0.90).item(),
        'p95': torch.quantile(abs_vals, 0.95).item(),
        'p99': torch.quantile(abs_vals, 0.99).item(),
        'p999': torch.quantile(abs_vals, 0.999).item(),
        'p9999': torch.quantile(abs_vals, 0.9999).item(),
        'outlier_ratio': abs_vals.max().item() / (torch.quantile(abs_vals, 0.99).item() + 1e-10)
    }


def collect_bert_activations(num_samples: int = 10) -> torch.Tensor:
    """Collect activations from BERT model."""
    from transformers import AutoModel, AutoTokenizer
    
    print("Loading bert-base-uncased...")
    model = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model.eval()
    
    activations = []
    
    def hook(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        activations.append(out.detach())
    
    layer_idx = min(6, len(model.encoder.layer) - 1)
    handle = model.encoder.layer[layer_idx].output.register_forward_hook(hook)
    
    texts = [
        "Machine learning models require careful optimization for deployment.",
        "The quick brown fox jumps over the lazy dog.",
        "Quantization reduces memory footprint while maintaining accuracy.",
        "Natural language processing has advanced significantly in recent years.",
        "Deep neural networks learn hierarchical representations of data.",
        "Transfer learning enables efficient model adaptation.",
        "Attention mechanisms capture long-range dependencies in sequences.",
        "Efficient inference is critical for production systems.",
        "Model compression techniques include pruning and quantization.",
        "Hardware accelerators optimize matrix multiplication operations.",
    ][:num_samples]
    
    print(f"Collecting activations from {len(texts)} samples...")
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', padding=True,
                              truncation=True, max_length=128)
            model(**inputs)
    
    handle.remove()
    all_acts = torch.cat([a.flatten() for a in activations])
    print(f"Collected {len(all_acts):,} activation values")
    return all_acts


def collect_resnet_activations(num_samples: int = 50) -> torch.Tensor:
    """Collect activations from ResNet-18."""
    from torchvision import models
    
    print("Loading ResNet-18...")
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.eval()
    
    activations = []
    
    def hook(module, input, output):
        activations.append(output.detach())
    
    handle = model.layer4.register_forward_hook(hook)
    
    print(f"Collecting activations from {num_samples} random images...")
    with torch.no_grad():
        for _ in range(num_samples):
            x = torch.randn(1, 3, 224, 224)
            model(x)
    
    handle.remove()
    all_acts = torch.cat([a.flatten() for a in activations])
    print(f"Collected {len(all_acts):,} activation values")
    return all_acts


def run_calibration_comparison(activations: torch.Tensor, 
                               percentile: float = 99.99) -> Dict[str, CalibrationResult]:
    """Run all calibration methods and compare results."""
    observers = create_observers(percentile)
    
    print("\nCalibrating observers...")
    for name, obs in observers.items():
        obs.observe(activations)
    
    results = {}
    for name, obs in observers.items():
        range_val = obs.compute_range()
        metrics = compute_metrics(activations, range_val)
        metrics.method = name
        results[name] = metrics
    
    return results


def print_results(results: Dict[str, CalibrationResult], stats: Dict, percentile: float):
    """Pretty print calibration results."""
    print("\n" + "=" * 70)
    print("DISTRIBUTION STATISTICS")
    print("=" * 70)
    print(f"  Absolute max:     {stats['abs_max']:.4f}")
    print(f"  Mean:             {stats['mean']:.4f}")
    print(f"  Std:              {stats['std']:.4f}")
    print(f"  90th percentile:  {stats['p90']:.4f}")
    print(f"  99th percentile:  {stats['p99']:.4f}")
    print(f"  99.9th percentile:{stats['p999']:.4f}")
    print(f"  Outlier ratio:    {stats['outlier_ratio']:.2f}x (max / p99)")
    
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    print(f"{'Method':<15} {'Range':<10} {'Scale':<12} {'Clipped':<10} {'MSE':<12} {'SNR (dB)':<10}")
    print("-" * 70)
    
    sorted_results = sorted(results.values(), key=lambda x: x.mse)
    
    for r in sorted_results:
        print(f"{r.method:<15} {r.range_val:<10.4f} {r.scale:<12.6f} "
              f"{r.clipped_pct:<10.4f}% {r.mse:<12.6f} {r.snr_db:<10.2f}")
    
    best = sorted_results[0]
    worst = sorted_results[-1]
    
    print("-" * 70)
    print(f"\nBest method: {best.method} (MSE: {best.mse:.6f})")
    print(f"Improvement over worst: {worst.mse / best.mse:.1f}x lower MSE")
    
    if stats['outlier_ratio'] > 3:
        print(f"\n⚠️  High outlier ratio ({stats['outlier_ratio']:.1f}x) detected!")
        print("   Recommendation: Use percentile or MSE-optimal calibration")
    else:
        print(f"\n✓  Moderate outlier ratio ({stats['outlier_ratio']:.1f}x)")
        print("   AbsMax calibration may be acceptable")


def run_percentile_sweep(activations: torch.Tensor) -> Dict[float, CalibrationResult]:
    """Sweep through percentiles to find optimal value."""
    percentiles = [99.0, 99.5, 99.9, 99.95, 99.99, 100.0]
    results = {}
    abs_vals = activations.abs()
    
    for pct in percentiles:
        if pct == 100.0:
            range_val = abs_vals.max().item()
        else:
            range_val = torch.quantile(abs_vals, pct / 100).item()
        
        metrics = compute_metrics(activations, range_val)
        metrics.method = f"p{pct}"
        results[pct] = metrics
    
    return results


def print_percentile_sweep(results: Dict[float, CalibrationResult]):
    """Print percentile sweep results."""
    print("\n" + "=" * 70)
    print("PERCENTILE SWEEP ANALYSIS")
    print("=" * 70)
    print(f"{'Percentile':<12} {'Range':<10} {'Clipped':<10} {'MSE':<12} {'SNR (dB)':<10}")
    print("-" * 70)
    
    sorted_pcts = sorted(results.keys())
    best_pct = min(results.keys(), key=lambda p: results[p].mse)
    
    for pct in sorted_pcts:
        r = results[pct]
        marker = " ← BEST" if pct == best_pct else ""
        print(f"{pct:<12.2f} {r.range_val:<10.4f} {r.clipped_pct:<10.4f}% "
              f"{r.mse:<12.6f} {r.snr_db:<10.2f}{marker}")
    
    print("-" * 70)
    print(f"\nOptimal percentile: {best_pct}%")


def visualize_results(activations: torch.Tensor, 
                      results: Dict[str, CalibrationResult],
                      output_path: str = "range_estimation_analysis.png"):
    """Create visualization of calibration results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    abs_vals = activations.abs().numpy()
    
    # 1. Histogram with range markers
    ax1 = axes[0, 0]
    ax1.hist(abs_vals, bins=200, density=True, alpha=0.7, color='steelblue')
    
    colors = {'absmax': 'red', 'percentile': 'green', 'mse': 'orange', 'entropy': 'purple'}
    for name, result in results.items():
        ax1.axvline(result.range_val, color=colors[name], linestyle='--', 
                   linewidth=2, label=f'{name}: {result.range_val:.2f}')
    
    ax1.set_xlabel('Absolute Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Activation Distribution with Calibrated Ranges')
    ax1.legend()
    ax1.set_xlim(0, abs_vals.max() * 1.1)
    
    # 2. MSE comparison
    ax2 = axes[0, 1]
    methods = list(results.keys())
    mses = [results[m].mse for m in methods]
    bars = ax2.bar(methods, mses, color=[colors[m] for m in methods])
    ax2.set_ylabel('MSE')
    ax2.set_title('Quantization MSE by Method')
    ax2.set_yscale('log')
    for bar, mse in zip(bars, mses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{mse:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Clipping percentage
    ax3 = axes[1, 0]
    clipped = [results[m].clipped_pct for m in methods]
    bars = ax3.bar(methods, clipped, color=[colors[m] for m in methods])
    ax3.set_ylabel('Clipped Values (%)')
    ax3.set_title('Percentage of Values Clipped')
    for bar, pct in zip(bars, clipped):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{pct:.3f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. SNR comparison
    ax4 = axes[1, 1]
    snrs = [results[m].snr_db for m in methods]
    bars = ax4.bar(methods, snrs, color=[colors[m] for m in methods])
    ax4.set_ylabel('SNR (dB)')
    ax4.set_title('Signal-to-Noise Ratio (higher is better)')
    for bar, snr in zip(bars, snrs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{snr:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate range estimation methods for PTQ calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python range_estimation_demo.py --model bert
    python range_estimation_demo.py --model resnet --percentile 99.9
    python range_estimation_demo.py --model bert --sweep
    python range_estimation_demo.py --model synthetic --outlier-ratio 5.0
        """
    )
    
    parser.add_argument('--model', type=str, default='bert',
                       choices=['bert', 'resnet', 'synthetic'],
                       help='Model to collect activations from')
    parser.add_argument('--percentile', type=float, default=99.99,
                       help='Percentile for percentile-based calibration')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples for calibration')
    parser.add_argument('--sweep', action='store_true',
                       help='Run percentile sweep analysis')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--outlier-ratio', type=float, default=5.0,
                       help='Outlier ratio for synthetic data')
    parser.add_argument('--output', type=str, default='range_estimation_analysis.png',
                       help='Output path for visualization')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SECTION 4.3: RANGE ESTIMATION METHODS FOR PTQ CALIBRATION")
    print("=" * 70)
    
    if args.model == 'bert':
        activations = collect_bert_activations(num_samples=args.num_samples)
    elif args.model == 'resnet':
        activations = collect_resnet_activations(num_samples=args.num_samples)
    else:
        print(f"\nGenerating synthetic data with outlier ratio {args.outlier_ratio}x...")
        n = 100000
        main_dist = torch.randn(n) * 1.0
        n_outliers = int(n * 0.001)
        outlier_magnitude = torch.quantile(main_dist.abs(), 0.99) * args.outlier_ratio
        outliers = torch.randn(n_outliers) * 0.1 + outlier_magnitude
        activations = torch.cat([main_dist, outliers, -outliers])
        print(f"Generated {len(activations):,} synthetic activation values")
    
    stats = analyze_distribution(activations)
    results = run_calibration_comparison(activations, args.percentile)
    print_results(results, stats, args.percentile)
    
    if args.sweep:
        sweep_results = run_percentile_sweep(activations)
        print_percentile_sweep(sweep_results)
    
    if args.visualize:
        visualize_results(activations, results, args.output)
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. ABSMAX: Zero clipping but outliers waste precision
   → Use for weights and well-behaved CNN activations

2. PERCENTILE: Trade bounded clipping for better resolution
   → Default choice for transformer activations (99.99% or 99.9%)

3. MSE-OPTIMAL: Directly minimize reconstruction error
   → Best theoretical accuracy, higher calibration cost

4. ENTROPY/KL: Preserve distribution shape
   → TensorRT default, good for classification tasks

The outlier ratio (max / p99) predicts which method will win:
   - < 3x: AbsMax acceptable
   - 3-10x: Percentile recommended  
   - > 10x: MSE-optimal or aggressive percentile (99.0%)
    """)


if __name__ == "__main__":
    main()