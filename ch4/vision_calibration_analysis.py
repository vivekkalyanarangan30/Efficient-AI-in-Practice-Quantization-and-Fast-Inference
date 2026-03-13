"""
Chapter 4: Vision Calibration Analysis
======================================

This script demonstrates how different image conditions affect activation
ranges in a CNN, and why calibration data must cover this variation.

Approach: Take real images and apply controlled transformations that
demonstrably affect activation magnitudes:
- Original (baseline)
- High contrast (histogram stretching)
- Low contrast (compressed dynamic range)  
- Edge enhanced (sharpening)
- Saturated colors

Uses STL-10 dataset (96x96 images) for clearer visualization than CIFAR-10.

Outputs:
- Console table showing activation ranges by condition
- Figure: figures/CH04_F04_VisionCalibrationGrid.png  (reference raster)
- Figure: figures/CH04_F04_VisionCalibrationGrid.pdf  (vector, editable text)

Usage:
    python ch4/vision_calibration_analysis.py
    python ch4/vision_calibration_analysis.py --save-plot
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import STL10
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from collections import defaultdict
from pathlib import Path
import argparse
from PIL import Image, ImageEnhance, ImageFilter
import os


# ---------------------------------------------------------------------------
# Manning compliance constants
# ---------------------------------------------------------------------------
MANNING_MAX_WIDTH_IN = 5.6       # Max figure width in inches
MANNING_MAX_HEIGHT_IN = 7.0      # Max figure height in inches
MANNING_FONT_FAMILY = 'Arial'    # Required font
MANNING_FONT_MIN_PT = 7          # Minimum font size
MANNING_HEADING_PT = 8           # Heading font size
MANNING_DPI_PNG = 300            # PNG reference raster DPI

# Manning color palette – chosen for grayscale separation                   #A
# Using colors from different palette columns and levels so they remain
# distinguishable after grayscale conversion.
MANNING_COLORS = {
    'green_l3':  '#80C21D',   # Green Level 3  – grayscale ≈ light
    'black_k50': '#808080',   # Black K=50     – grayscale ≈ mid-gray
    'red_l3':    '#D31518',   # Red Level 3    – grayscale ≈ dark-mid
    'blue_l4':   '#002D8B',   # Blue Level 4   – grayscale ≈ very dark
    'orange_l3': '#E37B45',   # Orange Level 3 – grayscale ≈ mid-light
}

# Hatch patterns for grayscale differentiation (print is B&W)              #B
HATCH_PATTERNS = ['', '///', '\\\\\\', '...', 'xx']


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


def download_sample_images(n_images=5):
    """
    Load sample images from STL-10 dataset.
    STL-10 has 96x96 images which are much clearer than CIFAR's 32x32.
    """
    print("Loading STL-10 dataset (96x96 images)...")

    dataset = STL10(
        root='./data',
        split='test',
        download=True,
        transform=transforms.ToTensor()
    )

    # Select diverse images (different classes)
    selected = []
    seen_classes = set()

    for idx in range(len(dataset)):
        img, label = dataset[idx]
        if label not in seen_classes and len(selected) < n_images:
            img_pil = TF.to_pil_image(img)
            selected.append((img_pil, label, idx))
            seen_classes.add(label)

        if len(selected) >= n_images:
            break

    return selected


def apply_image_conditions(images):
    """
    Apply controlled transformations that demonstrably affect activation ranges.

    These transformations are designed to create meaningful differences in
    how the CNN processes the images.
    """
    conditions = {
        'original': [],
        'low_contrast': [],
        'high_contrast': [],
        'edge_enhanced': [],
        'saturated': [],
    }

    for img_pil, label, idx in images:
        # Original - baseline
        conditions['original'].append((img_pil.copy(), label))

        # Low contrast - compress dynamic range significantly
        # Reduces activation magnitudes by reducing input signal variance
        enhancer = ImageEnhance.Contrast(img_pil)
        low_contrast = enhancer.enhance(0.2)  # Very low contrast
        conditions['low_contrast'].append((low_contrast, label))

        # High contrast - stretch histogram aggressively
        # Increases activation magnitudes by amplifying input signal
        high_contrast = enhancer.enhance(3.0)  # 3x contrast
        # Also boost brightness slightly to push values higher
        bright_enhancer = ImageEnhance.Brightness(high_contrast)
        high_contrast = bright_enhancer.enhance(1.2)
        conditions['high_contrast'].append((high_contrast, label))

        # Edge enhanced - strong sharpening
        # High frequency content increases activations in conv layers
        edge_enhanced = img_pil.filter(ImageFilter.SHARPEN)
        edge_enhanced = edge_enhanced.filter(ImageFilter.SHARPEN)
        edge_enhanced = edge_enhanced.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edge_enhanced = ImageEnhance.Contrast(edge_enhanced).enhance(2.0)
        conditions['edge_enhanced'].append((edge_enhanced, label))

        # Saturated - extreme color saturation + contrast
        # Creates large per-channel variations
        saturator = ImageEnhance.Color(img_pil)
        saturated = saturator.enhance(4.0)  # 4x saturation
        saturated = ImageEnhance.Contrast(saturated).enhance(2.0)
        conditions['saturated'].append((saturated, label))

    return conditions


def measure_activation_ranges(model, images_pil, device='cpu'):
    """
    Measure activation ranges through ResNet-18 for a list of PIL images.
    """
    model = model.to(device)
    model.eval()

    # ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Track max activations across all images
    layer_maxes = defaultdict(float)
    layer_means = defaultdict(list)

    def make_hook(name):
        def hook(module, input, output):
            layer_maxes[name] = max(
                layer_maxes[name], output.abs().max().item()
            )
            layer_means[name].append(output.abs().mean().item())
        return hook

    handles = [
        model.layer1.register_forward_hook(make_hook('layer1')),
        model.layer2.register_forward_hook(make_hook('layer2')),
        model.layer3.register_forward_hook(make_hook('layer3')),
        model.layer4.register_forward_hook(make_hook('layer4')),
    ]

    for img_pil, label in images_pil:
        tensor = preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(tensor)

    for h in handles:
        h.remove()

    results = {}
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        results[layer] = {
            'abs_max': layer_maxes[layer],
            'mean': (
                np.mean(layer_means[layer]) if layer_means[layer] else 0
            ),
        }

    return results


def create_visualization(conditions_data, activation_results, save_path=None):
    """
    Create a grid visualization showing sample images and their activation
    ranges.  Outputs both PNG (raster reference) and PDF (vector / editable
    text) when *save_path* is provided.

    Manning compliance:
    - Figure fits within 5.6 × 7 inches
    - Arial font, ≥ 7 pt everywhere
    - Hatch patterns for grayscale differentiation
    - PDF uses TrueType (fonttype 42) so text is editable
    - File naming follows CH04_F04_… convention
    """
    set_manning_style()

    fig = plt.figure(figsize=(MANNING_MAX_WIDTH_IN, 5.5))              #D
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.30)

    # === Left panel: image grid ===
    ax_images = fig.add_subplot(gs[0])
    ax_images.set_title(
        'Sample images by condition',                                  #E
        fontsize=MANNING_HEADING_PT, fontweight='bold', pad=8,
        fontfamily=MANNING_FONT_FAMILY,
    )

    condition_order = [
        'original', 'low_contrast', 'high_contrast',
        'edge_enhanced', 'saturated',
    ]
    condition_labels = [
        'Original', 'Low contrast', 'High contrast',
        'Edge enhanced', 'Saturated',
    ]
    n_conditions = len(condition_order)
    n_samples = min(4, len(conditions_data.get('original', [])))

    # Build a composite numpy image grid
    img_size = 96
    padding = 6
    grid_h = n_conditions * (img_size + padding) - padding
    grid_w = n_samples * (img_size + padding) - padding
    grid_img = np.ones((grid_h, grid_w, 3))

    for row_idx, condition in enumerate(condition_order):
        if condition not in conditions_data:
            continue
        samples = conditions_data[condition][:n_samples]
        for col_idx, (img_pil, _label) in enumerate(samples):
            img_resized = img_pil.resize(
                (img_size, img_size), Image.LANCZOS
            )
            img_np = np.clip(np.array(img_resized) / 255.0, 0, 1)
            y0 = row_idx * (img_size + padding)
            x0 = col_idx * (img_size + padding)
            grid_img[y0:y0 + img_size, x0:x0 + img_size] = img_np

    ax_images.imshow(grid_img, interpolation='lanczos')
    ax_images.set_xticks([])
    ax_images.set_yticks(
        [(img_size + padding) * i + img_size // 2
         for i in range(n_conditions)]
    )
    ax_images.set_yticklabels(
        condition_labels,
        fontsize=MANNING_FONT_MIN_PT,
        fontfamily=MANNING_FONT_FAMILY,
    )
    ax_images.tick_params(left=False)
    for spine in ax_images.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')

    # === Right panel: grouped bar chart ===
    ax_bars = fig.add_subplot(gs[1])
    ax_bars.set_title(
        'Layer-wise activation maximums',
        fontsize=MANNING_HEADING_PT, fontweight='bold', pad=8,
        fontfamily=MANNING_FONT_FAMILY,
    )

    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    x = np.arange(len(layers))
    width = 0.15

    # Map conditions → Manning palette colours + hatches
    palette_keys = list(MANNING_COLORS.values())
    for idx, condition in enumerate(condition_order):
        if condition not in activation_results:
            continue
        values = [
            activation_results[condition][layer]['abs_max']
            for layer in layers
        ]
        offset = (idx - n_conditions / 2 + 0.5) * width
        ax_bars.bar(
            x + offset, values, width,
            label=condition_labels[idx],
            color=palette_keys[idx],
            hatch=HATCH_PATTERNS[idx],                                 #F
            edgecolor='white',
            linewidth=0.5,
        )

    ax_bars.set_xlabel(
        'ResNet-18 layer',
        fontsize=MANNING_FONT_MIN_PT,
        fontfamily=MANNING_FONT_FAMILY,
    )
    ax_bars.set_ylabel(
        'Activation maximum',
        fontsize=MANNING_FONT_MIN_PT,
        fontfamily=MANNING_FONT_FAMILY,
    )
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(
        ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4'],
        fontsize=MANNING_FONT_MIN_PT,
        fontfamily=MANNING_FONT_FAMILY,
    )
    ax_bars.legend(
        loc='upper left',
        fontsize=MANNING_FONT_MIN_PT,
        framealpha=0.95,
    )
    ax_bars.grid(axis='y', alpha=0.3, linestyle='--')
    ax_bars.set_axisbelow(True)

    # Annotate the activation range ratio at layer 4
    layer4_vals = {
        cond: activation_results[cond]['layer4']['abs_max']
        for cond in condition_order
        if cond in activation_results
    }
    max_val = max(layer4_vals.values())
    min_val = min(layer4_vals.values())
    ratio = max_val / min_val if min_val > 0 else 0
    max_cond = max(layer4_vals, key=layer4_vals.get)
    min_cond = min(layer4_vals, key=layer4_vals.get)

    ax_bars.annotate(
        f'{ratio:.1f}x range\n({min_cond} \u2192 {max_cond})',
        xy=(3.3, max_val * 0.85),
        fontsize=MANNING_FONT_MIN_PT,
        fontfamily=MANNING_FONT_FAMILY,
        color='#333333',
        ha='left',
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor='#FEE3AC',       # Manning Yellow Level 1
            alpha=0.85,
            edgecolor='#CCA000',
        ),
    )

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save outputs: PNG (raster reference) + PDF (vector / editable text)
    # ------------------------------------------------------------------
    if save_path:
        base = Path(save_path)
        base.parent.mkdir(parents=True, exist_ok=True)

        # Derive Manning-convention stems
        stem = base.stem                        # e.g. CH04_F04_VisionCalibrationGrid
        png_path = base.parent / f'{stem}.png'
        pdf_path = base.parent / f'{stem}.pdf'

        fig.savefig(
            str(png_path),
            dpi=MANNING_DPI_PNG,
            bbox_inches='tight',
            facecolor='white',
        )
        print(f"Saved PNG (reference raster) → {png_path}")

        fig.savefig(
            str(pdf_path),
            format='pdf',
            bbox_inches='tight',
            facecolor='white',
        )                                                              #G
        print(f"Saved PDF (vector / editable) → {pdf_path}")

    plt.show()
    return fig


def print_results_table(activation_results):
    """Print formatted table of activation results."""

    condition_order = [
        'original', 'low_contrast', 'high_contrast',
        'edge_enhanced', 'saturated',
    ]
    layers = ['layer1', 'layer2', 'layer3', 'layer4']

    print("\nActivation Ranges by Image Condition (ResNet-18):")
    print("=" * 73)
    print(
        f"{'Condition':<15} | {'Layer1':>10} | {'Layer2':>10} "
        f"| {'Layer3':>10} | {'Layer4':>10}"
    )
    print("-" * 73)

    min_vals = {layer: float('inf') for layer in layers}
    max_vals = {layer: 0 for layer in layers}

    for condition in condition_order:
        if condition not in activation_results:
            continue
        stats = activation_results[condition]
        row = f"{condition:<15}"
        for layer in layers:
            val = stats[layer]['abs_max']
            row += f" | {val:>10.2f}"
            min_vals[layer] = min(min_vals[layer], val)
            max_vals[layer] = max(max_vals[layer], val)
        print(row)

    print("-" * 73)

    ratio_row = f"{'Max/Min Ratio':<15}"
    for layer in layers:
        ratio = (
            max_vals[layer] / min_vals[layer]
            if min_vals[layer] > 0 else 0
        )
        ratio_row += f" | {ratio:>9.1f}x"
    print(ratio_row)

    print("\n" + "=" * 73)
    layer4_ratio = (
        max_vals['layer4'] / min_vals['layer4']
        if min_vals['layer4'] > 0 else 0
    )
    print(
        f"Key Insight: Activation range varies by "
        f"{layer4_ratio:.1f}x in the final layer."
    )
    print(
        "Calibrating only on 'easy' images will clip "
        "activations from challenging ones."
    )
    print("=" * 73)


def main():
    parser = argparse.ArgumentParser(
        description='Vision calibration analysis'
    )
    parser.add_argument(
        '--save-plot', action='store_true',
        help='Save PNG + PDF to figures/',
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device to run on (cpu/cuda)',
    )
    parser.add_argument(
        '--n-images', type=int, default=5,
        help='Number of base images to use',
    )
    args = parser.parse_args()

    print("Loading sample images...")
    base_images = download_sample_images(n_images=args.n_images)
    print(f"Loaded {len(base_images)} base images")

    print("\nApplying image transformations...")
    conditions_data = apply_image_conditions(base_images)

    for cond, imgs in conditions_data.items():
        print(f"  {cond}: {len(imgs)} images")

    print("\nLoading ResNet-18...")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    print("\nMeasuring activation ranges...")
    activation_results = {}
    for condition, images in conditions_data.items():
        activation_results[condition] = measure_activation_ranges(
            model, images, device=args.device
        )

    print_results_table(activation_results)

    # Manning file naming: CH04_F04_VisionCalibrationGrid
    save_path = (
        'figures/CH04_F04_VisionCalibrationGrid.png'
        if args.save_plot else None
    )
    create_visualization(conditions_data, activation_results, save_path)


if __name__ == '__main__':
    main()