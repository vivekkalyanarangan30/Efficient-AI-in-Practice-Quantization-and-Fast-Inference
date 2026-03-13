#!/usr/bin/env python3
"""
Chapter 4.4: PTQ Validation - End-to-End Experiment

Validates Post-Training Quantization on real models:
  1. ResNet-18 on ImageNet validation
  2. BERT-base on SST-2 sentiment

What PTQ guarantees:
  ✓ Size reduction (~4x for INT8)
  ✓ Accuracy preservation (with good calibration)

What PTQ does NOT guarantee without runtime support:
  ✗ Latency improvement (requires INT8 kernels - see Chapter 6)

Outputs:
  - Console summary table
  - figures/CH04_F05_PTQValidation.png  (reference raster, 300 DPI)
  - figures/CH04_F05_PTQValidation.pdf  (vector, editable text)

Usage:
    python ch4/ptq_validation.py --model resnet --num-samples 500
    python ch4/ptq_validation.py --model bert --num-samples 500
    python ch4/ptq_validation.py --model all --visualize
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
import argparse
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Manning compliance constants
# ---------------------------------------------------------------------------
MANNING_MAX_WIDTH_IN = 5.6       # Max figure width in inches
MANNING_MAX_HEIGHT_IN = 7.0      # Max figure height in inches
MANNING_FONT_FAMILY = 'Arial'    # Required font
MANNING_FONT_MIN_PT = 7          # Minimum font size
MANNING_HEADING_PT = 8           # Heading font size
MANNING_DPI_PNG = 300            # PNG reference raster DPI

# Manning palette – chosen from different columns/levels for grayscale      #A
# separation (verified against the grayscale palette page).
MANNING_COLORS = {
    'fp32':       '#80C21D',  # Green Level 3   – grayscale ≈ light
    'int8':       '#002D8B',  # Blue Level 4    – grayscale ≈ very dark
    'resnet':     '#D31518',  # Red Level 3     – grayscale ≈ dark-mid
    'bert':       '#E37B45',  # Orange Level 3  – grayscale ≈ mid-light
    'absmax':     '#D31518',  # Red Level 3
    'percentile': '#80C21D',  # Green Level 3
    'mse':        '#002D8B',  # Blue Level 4
}

# Hatch patterns for grayscale differentiation in B&W print                #B
HATCH_PATTERNS = {
    'fp32':       '',
    'int8':       '///',
    'resnet':     '',
    'bert':       '///',
    'absmax':     '',
    'percentile': '///',
    'mse':        '...',
}


def set_manning_style():
    """Configure matplotlib defaults for Manning compliance."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [MANNING_FONT_FAMILY, 'Helvetica', 'DejaVu Sans'],
        'font.size': MANNING_FONT_MIN_PT,
        'axes.titlesize': MANNING_HEADING_PT,
        'axes.labelsize': MANNING_FONT_MIN_PT,
        'xtick.labelsize': MANNING_FONT_MIN_PT,
        'ytick.labelsize': MANNING_FONT_MIN_PT,
        'legend.fontsize': MANNING_FONT_MIN_PT,
        'figure.dpi': 100,
        'savefig.dpi': MANNING_DPI_PNG,
        'pdf.fonttype': 42,      # TrueType in PDF → editable text          #C
        'ps.fonttype': 42,
    })


def _save_manning_figure(fig, save_path: str):
    """Save a figure as both PNG (raster) and PDF (vector / editable)."""
    base = Path(save_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    stem = base.stem  # e.g. CH04_F05_PTQValidation

    png_path = base.parent / f'{stem}.png'
    pdf_path = base.parent / f'{stem}.pdf'

    fig.savefig(
        str(png_path), dpi=MANNING_DPI_PNG,
        bbox_inches='tight', facecolor='white',
    )
    print(f"  Saved PNG (reference raster) → {png_path}")

    fig.savefig(
        str(pdf_path), format='pdf',
        bbox_inches='tight', facecolor='white',
    )
    print(f"  Saved PDF (vector / editable) → {pdf_path}")


# ============================================================================
# Calibration Methods (from Section 4.3)
# ============================================================================

def compute_scale_absmax(tensor: torch.Tensor) -> Tuple[float, Dict]:
    """
    Absolute maximum calibration.
    Simple but wastes precision if outliers exist.
    """
    abs_max = tensor.abs().max().item()
    scale = abs_max / 127 if abs_max > 0 else 1.0

    stats = {
        'range': abs_max,
        'clipped_pct': 0.0,
        'method': 'absmax',
    }
    return scale, stats


def compute_scale_percentile(
    tensor: torch.Tensor, percentile: float = 99.99,
) -> Tuple[float, Dict]:
    """
    Percentile calibration.
    Clips outliers for better precision on typical values.
    """
    # Sample for large tensors to avoid memory issues
    if tensor.numel() > 5_000_000:
        flat = tensor.flatten()
        indices = torch.randperm(tensor.numel())[:1_000_000]
        sample = flat[indices].abs().float()
    else:
        sample = tensor.abs().float()

    range_val = torch.quantile(sample, percentile / 100).item()
    scale = range_val / 127 if range_val > 0 else 1.0

    # Calculate how much we're clipping
    clipped = (tensor.abs() > range_val).float().mean().item() * 100

    stats = {
        'range': range_val,
        'clipped_pct': clipped,
        'method': f'percentile_{percentile}',
    }
    return scale, stats


def compute_scale_mse(
    tensor: torch.Tensor, num_candidates: int = 100,
) -> Tuple[float, Dict]:
    """
    MSE-optimal calibration.
    Finds the scale that minimizes reconstruction error.
    """
    max_val = tensor.abs().max().item()
    if max_val == 0:
        return 1.0, {'range': 0, 'clipped_pct': 0, 'method': 'mse'}

    best_mse = float('inf')
    best_scale = max_val / 127
    best_range = max_val

    # Grid search over candidate ranges
    for r in np.linspace(0.5 * max_val, max_val, num_candidates):
        scale = r / 127
        q = torch.round(tensor / scale).clamp(-127, 127)
        reconstructed = q * scale
        mse = ((tensor - reconstructed) ** 2).mean().item()

        if mse < best_mse:
            best_mse = mse
            best_scale = scale
            best_range = r

    clipped = (tensor.abs() > best_range).float().mean().item() * 100

    stats = {
        'range': best_range,
        'clipped_pct': clipped,
        'mse': best_mse,
        'method': 'mse',
    }
    return best_scale, stats


# ============================================================================
# Quantized Layers (Simulated INT8 Storage)
# ============================================================================

class QuantizedLinear(nn.Module):
    """
    Linear layer with INT8 weight storage.

    Storage: Weights as INT8 (1 byte) + scale as FP32 (4 bytes)
    Compute: Dequantizes to FP32 for matmul (no speedup without INT8 kernels)
    """
    def __init__(self, original: nn.Linear, scale: float):
        super().__init__()
        # Quantize weights to INT8
        q_weight = torch.round(original.weight.data / scale).clamp(-127, 127)
        self.register_buffer('q_weight', q_weight.to(torch.int8))
        self.register_buffer(
            'scale', torch.tensor(scale, dtype=torch.float32),
        )

        if original.bias is not None:
            self.register_buffer('bias', original.bias.data.clone())
        else:
            self.bias = None

        # Track original size for comparison
        self.original_weight_bytes = original.weight.numel() * 4   # FP32
        self.quantized_weight_bytes = q_weight.numel() * 1         # INT8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize for computation (real INT8 kernels skip this – Ch 6)
        weight = self.q_weight.float() * self.scale
        return F.linear(x, weight, self.bias)


class QuantizedConv2d(nn.Module):
    """Conv2d with INT8 weight storage."""
    def __init__(self, original: nn.Conv2d, scale: float):
        super().__init__()
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups

        q_weight = torch.round(original.weight.data / scale).clamp(-127, 127)
        self.register_buffer('q_weight', q_weight.to(torch.int8))
        self.register_buffer(
            'scale', torch.tensor(scale, dtype=torch.float32),
        )

        if original.bias is not None:
            self.register_buffer('bias', original.bias.data.clone())
        else:
            self.bias = None

        self.original_weight_bytes = original.weight.numel() * 4
        self.quantized_weight_bytes = q_weight.numel() * 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.q_weight.float() * self.scale
        return F.conv2d(
            x, weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups,
        )


# ============================================================================
# Model Quantization
# ============================================================================

def quantize_model(model: nn.Module, method: str = 'mse') -> nn.Module:
    """
    Quantize model weights to INT8 using specified calibration method.

    Args:
        model: FP32 model
        method: 'absmax', 'percentile', or 'mse'

    Returns:
        Model with INT8 weights (simulated)
    """
    if method == 'absmax':
        get_scale = lambda t: compute_scale_absmax(t)[0]
    elif method == 'percentile':
        get_scale = lambda t: compute_scale_percentile(t, 99.99)[0]
    elif method == 'mse':
        get_scale = lambda t: compute_scale_mse(t)[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    model_q = deepcopy(model)

    def replace_layers(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                scale = get_scale(child.weight.data)
                setattr(module, name, QuantizedLinear(child, scale))
            elif isinstance(child, nn.Conv2d):
                scale = get_scale(child.weight.data)
                setattr(module, name, QuantizedConv2d(child, scale))
            else:
                replace_layers(child)

    replace_layers(model_q)
    return model_q


def analyze_calibration(model: nn.Module, method: str) -> Dict:
    """Analyze quantization error for a calibration method."""
    if method == 'absmax':
        get_scale_stats = compute_scale_absmax
    elif method == 'percentile':
        get_scale_stats = lambda t: compute_scale_percentile(t, 99.99)
    elif method == 'mse':
        get_scale_stats = compute_scale_mse
    else:
        raise ValueError(f"Unknown method: {method}")

    total_mse = 0
    total_elements = 0
    max_error = 0
    layer_errors = []

    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            scale, _ = get_scale_stats(param.data)

            # Simulate quantization
            q = torch.round(param.data / scale).clamp(-127, 127)
            reconstructed = q * scale

            # Measure error
            error = (param.data - reconstructed).abs()
            layer_mse = (error ** 2).mean().item()
            layer_max = error.max().item()

            layer_errors.append({
                'name': name,
                'mse': layer_mse,
                'max_error': layer_max,
                'numel': param.numel(),
            })

            total_mse += layer_mse * param.numel()
            total_elements += param.numel()
            max_error = max(max_error, layer_max)

    return {
        'method': method,
        'avg_mse': total_mse / total_elements if total_elements > 0 else 0,
        'max_error': max_error,
        'layer_errors': layer_errors,
    }


# ============================================================================
# Size Measurement
# ============================================================================

def get_model_size_mb(model: nn.Module, quantized: bool = False) -> float:
    """
    Calculate model size in MB.

    For quantized models: INT8 weights + FP32 scales/biases
    For FP32 models: All FP32
    """
    total_bytes = 0

    for module in model.modules():
        if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
            total_bytes += module.q_weight.numel() * 1   # INT8 weights
            total_bytes += 4                              # FP32 scale
            if module.bias is not None:
                total_bytes += module.bias.numel() * 4
        elif isinstance(module, nn.Linear):
            total_bytes += module.weight.numel() * 4
            if module.bias is not None:
                total_bytes += module.bias.numel() * 4
        elif isinstance(module, nn.Conv2d):
            total_bytes += module.weight.numel() * 4
            if module.bias is not None:
                total_bytes += module.bias.numel() * 4
        elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            for param in module.parameters():
                total_bytes += param.numel() * 4

    return total_bytes / (1024 * 1024)


# ============================================================================
# Experiments
# ============================================================================

def run_resnet_experiment(device: str = 'cuda', num_samples: int = 500):
    """ResNet-18 on ImageNet validation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: ResNet-18 on ImageNet")
    print("=" * 70)

    from torchvision import models, transforms
    from datasets import load_dataset

    # Load data
    print("\n[1/4] Loading ImageNet validation (streaming)...")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
        ),
    ])

    hf_dataset = load_dataset(
        "imagenet-1k", split="validation",
        streaming=True, token=True,
    )

    samples = []
    print(f"  Downloading {num_samples} samples...")
    for i, item in enumerate(hf_dataset):
        if i >= num_samples:
            break
        samples.append(item)
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{num_samples}")

    class ImageNetDataset(Dataset):
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            img = self.samples[idx]['image'].convert('RGB')
            return self.transform(img), self.samples[idx]['label']

    dataset = ImageNetDataset(samples, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"  Loaded {len(dataset)} samples")

    # Load model
    print("\n[2/4] Loading ResNet-18...")
    model_fp32 = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1,
    )
    model_fp32.eval()
    model_fp32.to(device)

    # FP32 Baseline
    print("\n[3/4] Evaluating FP32 baseline...")
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_fp32(inputs)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    fp32_acc = correct / total
    fp32_size = get_model_size_mb(model_fp32)

    print(f"  Accuracy: {fp32_acc*100:.2f}%")
    print(f"  Size: {fp32_size:.2f} MB")

    # Quantization experiments
    print("\n[4/4] Quantizing with different calibration methods...")

    results = {
        'FP32': {'accuracy': fp32_acc, 'size_mb': fp32_size},
    }

    calibration_analysis = {}

    for method in ['absmax', 'percentile', 'mse']:
        print(f"\n  {method.upper()}:")

        analysis = analyze_calibration(model_fp32, method)
        calibration_analysis[method] = analysis
        print(f"    Quantization MSE: {analysis['avg_mse']:.2e}")

        model_int8 = quantize_model(model_fp32, method)
        model_int8.to(device)

        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_int8(inputs)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        int8_acc = correct / total
        int8_size = get_model_size_mb(model_int8, quantized=True)

        delta = (int8_acc - fp32_acc) * 100
        compression = fp32_size / int8_size

        print(f"    Accuracy: {int8_acc*100:.2f}% ({delta:+.2f}%)")
        print(f"    Size: {int8_size:.2f} MB ({compression:.1f}x smaller)")

        results[f'INT8_{method}'] = {
            'accuracy': int8_acc,
            'size_mb': int8_size,
            'calibration_mse': analysis['avg_mse'],
        }

    return results, calibration_analysis


def run_bert_experiment(device: str = 'cuda', num_samples: int = 500):
    """BERT-base on SST-2 sentiment classification."""
    print("\n" + "=" * 70)
    print("EXPERIMENT: BERT-base on SST-2")
    print("=" * 70)

    from transformers import BertForSequenceClassification, BertTokenizer
    from datasets import load_dataset

    print("\n[1/4] Loading BERT and SST-2...")

    model_name = "textattack/bert-base-uncased-SST-2"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model_fp32 = BertForSequenceClassification.from_pretrained(model_name)
    model_fp32.eval()
    model_fp32.to(device)

    dataset = load_dataset('glue', 'sst2', split='validation')
    indices = list(range(min(num_samples, len(dataset))))

    print(f"  Model: {model_name}")
    print(f"  Dataset: {len(indices)} samples")

    # FP32 Baseline
    print("\n[2/4] Evaluating FP32 baseline...")

    correct, total = 0, 0
    with torch.no_grad():
        for i in indices:
            inputs = tokenizer(
                dataset[i]['sentence'], padding='max_length',
                truncation=True, max_length=128, return_tensors='pt',
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model_fp32(**inputs)
            pred = outputs.logits.argmax(dim=-1).item()

            if pred == dataset[i]['label']:
                correct += 1
            total += 1

    fp32_acc = correct / total
    fp32_size = get_model_size_mb(model_fp32)

    print(f"  Accuracy: {fp32_acc*100:.2f}%")
    print(f"  Size: {fp32_size:.2f} MB")

    # Calibration analysis
    print("\n[3/4] Analyzing calibration methods...")

    calibration_analysis = {}
    for method in ['absmax', 'percentile', 'mse']:
        analysis = analyze_calibration(model_fp32, method)
        calibration_analysis[method] = analysis
        print(f"  {method}: MSE = {analysis['avg_mse']:.2e}")

    print("\n[4/4] Quantizing with best method (MSE)...")

    results = {
        'FP32': {'accuracy': fp32_acc, 'size_mb': fp32_size},
    }

    model_int8 = quantize_model(model_fp32, 'mse')
    model_int8.to(device)

    correct, total = 0, 0
    with torch.no_grad():
        for i in indices:
            inputs = tokenizer(
                dataset[i]['sentence'], padding='max_length',
                truncation=True, max_length=128, return_tensors='pt',
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model_int8(**inputs)
            pred = outputs.logits.argmax(dim=-1).item()

            if pred == dataset[i]['label']:
                correct += 1
            total += 1

    int8_acc = correct / total
    int8_size = get_model_size_mb(model_int8, quantized=True)

    delta = (int8_acc - fp32_acc) * 100
    compression = fp32_size / int8_size

    print(f"  Accuracy: {int8_acc*100:.2f}% ({delta:+.2f}%)")
    print(f"  Size: {int8_size:.2f} MB ({compression:.1f}x smaller)")

    results['INT8_mse'] = {
        'accuracy': int8_acc,
        'size_mb': int8_size,
        'calibration_mse': calibration_analysis['mse']['avg_mse'],
    }

    return results, calibration_analysis


# ============================================================================
# Visualization (Manning-compliant)
# ============================================================================

def plot_results(
    resnet_results, bert_results,
    calibration_resnet, calibration_bert,
    output_path: str = 'figures/CH04_F05_PTQValidation.png',
):
    """
    Generate Manning-compliant comparison plots.

    Saves both PNG (raster reference) and PDF (vector / editable text).

    Manning compliance:
    - Figure fits within 5.6 × 7 inches
    - Arial font, ≥ 7 pt everywhere
    - Hatch patterns for grayscale differentiation
    - PDF uses TrueType (fonttype 42) so text is editable
    - Sentence-case headings
    """
    set_manning_style()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(MANNING_MAX_WIDTH_IN, 5.5),                          #D
    )

    # ------------------------------------------------------------------
    # Panel 1: ResNet accuracy
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    methods = list(resnet_results.keys())
    accs = [r['accuracy'] * 100 for r in resnet_results.values()]
    colors = [
        MANNING_COLORS['fp32'] if 'FP32' in m else MANNING_COLORS['int8']
        for m in methods
    ]
    hatches = [
        HATCH_PATTERNS['fp32'] if 'FP32' in m else HATCH_PATTERNS['int8']
        for m in methods
    ]

    bars = ax.bar(
        methods, accs, color=colors, hatch=hatches,
        edgecolor='white', linewidth=0.5,
    )
    ax.axhline(
        y=resnet_results['FP32']['accuracy'] * 100,
        color=MANNING_COLORS['fp32'], linestyle='--', alpha=0.5,
    )
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('ResNet-18: accuracy comparison')                     #E
    ax.set_ylim([min(accs) - 2, max(accs) + 2])
    ax.tick_params(axis='x', rotation=15)

    fp32_acc = resnet_results['FP32']['accuracy'] * 100
    for i, (m, a) in enumerate(zip(methods, accs)):
        if 'INT8' in m:
            delta = a - fp32_acc
            ax.text(
                i, a + 0.3, f'{delta:+.2f}%', ha='center',
                fontsize=MANNING_FONT_MIN_PT,
                color='#333333',
            )

    # ------------------------------------------------------------------
    # Panel 2: ResNet size
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    sizes = [r['size_mb'] for r in resnet_results.values()]
    bars = ax.bar(
        methods, sizes, color=colors, hatch=hatches,
        edgecolor='white', linewidth=0.5,
    )
    ax.set_ylabel('Size (MB)')
    ax.set_title('ResNet-18: model size')
    ax.tick_params(axis='x', rotation=15)

    fp32_size = resnet_results['FP32']['size_mb']
    for i, (m, s) in enumerate(zip(methods, sizes)):
        if 'INT8' in m:
            ratio = fp32_size / s
            ax.text(
                i, s + 1, f'{ratio:.1f}x', ha='center',
                fontsize=MANNING_FONT_MIN_PT,
                color='#333333',
            )

    # ------------------------------------------------------------------
    # Panel 3: BERT accuracy
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    methods_b = list(bert_results.keys())
    accs_b = [r['accuracy'] * 100 for r in bert_results.values()]
    colors_b = [
        MANNING_COLORS['fp32'] if 'FP32' in m else MANNING_COLORS['bert']
        for m in methods_b
    ]
    hatches_b = [
        HATCH_PATTERNS['fp32'] if 'FP32' in m else HATCH_PATTERNS['bert']
        for m in methods_b
    ]

    bars = ax.bar(
        methods_b, accs_b, color=colors_b, hatch=hatches_b,
        edgecolor='white', linewidth=0.5,
    )
    ax.axhline(
        y=bert_results['FP32']['accuracy'] * 100,
        color=MANNING_COLORS['fp32'], linestyle='--', alpha=0.5,
    )
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('BERT-base: accuracy comparison')
    ax.set_ylim([min(accs_b) - 2, max(accs_b) + 2])

    fp32_acc_b = bert_results['FP32']['accuracy'] * 100
    for i, (m, a) in enumerate(zip(methods_b, accs_b)):
        if 'INT8' in m:
            delta = a - fp32_acc_b
            ax.text(
                i, a + 0.3, f'{delta:+.2f}%', ha='center',
                fontsize=MANNING_FONT_MIN_PT,
                color='#333333',
            )

    # ------------------------------------------------------------------
    # Panel 4: Calibration method comparison (grouped bars)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    cal_methods = ['absmax', 'percentile', 'mse']
    x = np.arange(len(cal_methods))
    width = 0.35

    resnet_mse = [calibration_resnet[m]['avg_mse'] for m in cal_methods]
    bert_mse = [calibration_bert[m]['avg_mse'] for m in cal_methods]

    ax.bar(
        x - width / 2, resnet_mse, width,
        label='ResNet-18',
        color=MANNING_COLORS['resnet'],
        hatch=HATCH_PATTERNS['resnet'],
        edgecolor='white', linewidth=0.5,
    )
    ax.bar(
        x + width / 2, bert_mse, width,
        label='BERT-base',
        color=MANNING_COLORS['bert'],
        hatch=HATCH_PATTERNS['bert'],
        edgecolor='white', linewidth=0.5,
    )

    ax.set_ylabel('Quantization MSE')
    ax.set_title('Calibration method: quantization error')
    ax.set_xticks(x)
    ax.set_xticklabels(['AbsMax', 'Percentile', 'MSE-optimal'])
    ax.legend(fontsize=MANNING_FONT_MIN_PT)
    ax.set_yscale('log')
    ax.annotate(
        'Lower is better', xy=(0.02, 0.92), xycoords='axes fraction',
        fontsize=MANNING_FONT_MIN_PT, color='#333333',
    )

    plt.tight_layout()

    # --- Dual save: PNG + PDF ---
    _save_manning_figure(fig, output_path)
    plt.close()


# ============================================================================
# Summary
# ============================================================================

def print_summary(resnet_results, bert_results):
    """Print final summary table."""

    print("\n" + "=" * 75)
    print("SUMMARY: PTQ Validation Results")
    print("=" * 75)

    print(
        f"\n{'Model':<12} {'Config':<15} {'Accuracy':>10} "
        f"{'Δ Acc':>10} {'Size (MB)':>12} {'Compression':>12}"
    )
    print("-" * 75)

    # ResNet
    fp32 = resnet_results['FP32']
    print(
        f"{'ResNet-18':<12} {'FP32':<15} "
        f"{fp32['accuracy']*100:>9.2f}% {'-':>10} "
        f"{fp32['size_mb']:>11.2f} {'-':>12}"
    )

    for name, r in resnet_results.items():
        if 'INT8' in name:
            delta = (r['accuracy'] - fp32['accuracy']) * 100
            comp = fp32['size_mb'] / r['size_mb']
            method = name.replace('INT8_', '')
            print(
                f"{'':<12} {f'INT8 ({method})':<15} "
                f"{r['accuracy']*100:>9.2f}% {delta:>+9.2f}% "
                f"{r['size_mb']:>11.2f} {comp:>11.1f}x"
            )

    print("-" * 75)

    # BERT
    fp32 = bert_results['FP32']
    print(
        f"{'BERT-base':<12} {'FP32':<15} "
        f"{fp32['accuracy']*100:>9.2f}% {'-':>10} "
        f"{fp32['size_mb']:>11.2f} {'-':>12}"
    )

    for name, r in bert_results.items():
        if 'INT8' in name:
            delta = (r['accuracy'] - fp32['accuracy']) * 100
            comp = fp32['size_mb'] / r['size_mb']
            method = name.replace('INT8_', '')
            print(
                f"{'':<12} {f'INT8 ({method})':<15} "
                f"{r['accuracy']*100:>9.2f}% {delta:>+9.2f}% "
                f"{r['size_mb']:>11.2f} {comp:>11.1f}x"
            )

    print("=" * 75)

    # Important note about latency
    print("\n" + "-" * 75)
    print("NOTE ON LATENCY:")
    print("-" * 75)
    print("This experiment validates accuracy and size reduction from PTQ.")
    print("Latency improvement requires runtime support for INT8 compute:")
    print("  - PyTorch: torch.ao.quantization with FBGEMM/QNNPACK backends")
    print("  - TensorRT: Native INT8 execution on NVIDIA GPUs")
    print("  - ONNX Runtime: Quantized operators with hardware acceleration")
    print("See Chapter 6 for deploying quantized models with actual speedup.")
    print("-" * 75)


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description='Chapter 4.4: PTQ Validation',
    )
    parser.add_argument(
        '--model', type=str, default='all',
        choices=['resnet', 'bert', 'all'],
    )
    parser.add_argument('--num-samples', type=int, default=500)
    parser.add_argument(
        '--device', type=str, default='cuda',
        choices=['cpu', 'cuda'],
    )
    parser.add_argument(
        '--output-dir', type=str,
        default=str(script_dir / 'figures'),
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate Manning-compliant PNG + PDF plots',
    )
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print("=" * 70)
    print("Chapter 4.4: PTQ Validation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Samples: {args.num_samples}")

    resnet_results, calibration_resnet = None, None
    bert_results, calibration_bert = None, None

    if args.model in ['resnet', 'all']:
        resnet_results, calibration_resnet = run_resnet_experiment(
            device, args.num_samples,
        )

    if args.model in ['bert', 'all']:
        bert_results, calibration_bert = run_bert_experiment(
            device, args.num_samples,
        )

    if resnet_results and bert_results:
        print_summary(resnet_results, bert_results)

        if args.visualize:
            output_path = (
                Path(args.output_dir) / 'CH04_F05_PTQValidation.png'
            )
            plot_results(
                resnet_results, bert_results,
                calibration_resnet, calibration_bert,
                str(output_path),
            )

        # Save JSON results
        json_path = Path(args.output_dir) / 'ptq_validation.json'
        json_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            'resnet': resnet_results,
            'bert': bert_results,
            'calibration': {
                'resnet': {
                    m: {
                        'avg_mse': c['avg_mse'],
                        'max_error': c['max_error'],
                    }
                    for m, c in calibration_resnet.items()
                },
                'bert': {
                    m: {
                        'avg_mse': c['avg_mse'],
                        'max_error': c['max_error'],
                    }
                    for m, c in calibration_bert.items()
                },
            },
        }
        with open(str(json_path), 'w') as f:
            json.dump(results, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
