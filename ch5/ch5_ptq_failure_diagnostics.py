#!/usr/bin/env python3
"""
Chapter 5: PTQ Failure Diagnostics

This script measures when post-training quantization falls short, providing
the empirical foundation for deciding when QAT is necessary.

Quantization: W+A (weight + activation), matching real integer hardware where
both operands of the matmul/conv are quantized.
  - Weights: per-channel symmetric, quantized statically.
  - Activations: per-tensor asymmetric (with zero-point), quantized dynamically
    via forward hooks.  Asymmetric is correct for activations because post-ReLU
    values are non-negative and post-LayerNorm/GELU values are skewed.

Datasets:
- Vision: ImageNette (10-class ImageNet subset, ~1.5GB) - realistic ImageNet distribution
- NLP: GLUE SST-2 (sentiment classification) - standard BERT benchmark (fine-tuned model)

Diagnostics included:
1. Accuracy across bit-widths (INT8, INT4)
2. Model size sensitivity analysis  
3. Per-class accuracy degradation
4. Layer sensitivity analysis
5. Confidence distribution analysis

Usage:
    # Vision: ResNet-18 on ImageNette
    python ch5_ptq_failure_diagnostics.py --task vision --model resnet18 --all
    
    # NLP: BERT on SST-2
    python ch5_ptq_failure_diagnostics.py --task nlp --model bert-base-uncased --all
    
    # Quick bit-width sweep
    python ch5_ptq_failure_diagnostics.py --task vision --bitwidth-sweep
    
    # Layer sensitivity at INT4
    python ch5_ptq_failure_diagnostics.py --task vision --layer-sensitivity --bits 4

Requirements:
    pip install torch torchvision transformers datasets numpy matplotlib tqdm tabulate
"""

import argparse
import copy
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Conditional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    def tabulate(data, headers, tablefmt=None):
        lines = [" | ".join(str(h) for h in headers)]
        lines.append("-" * len(lines[0]))
        for row in data:
            lines.append(" | ".join(str(x) for x in row))
        return "\n".join(lines)


# ============================================================================
# Quantization Utilities
# ============================================================================

@dataclass
class QuantConfig:
    """Configuration for quantization.
    
    Weights: per-channel symmetric — weights are centered around zero,
             so symmetric quantization uses the full integer range.
    Activations: per-tensor asymmetric — activations are often non-negative
             (post-ReLU) or skewed (post-LayerNorm/GELU), so asymmetric
             quantization with a zero-point offset avoids wasting codes
             on values that never appear.
    
    Real integer hardware (TensorRT, XNNPACK, etc.) quantizes BOTH operands
    of the matmul/conv. This config drives both sides.
    """
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = True   # for weights; activations are always per-tensor
    channel_axis: int = 0
    
    @property
    def q_max(self) -> int:
        if self.symmetric:
            return (1 << (self.bits - 1)) - 1
        return (1 << self.bits) - 1
    
    @property
    def q_min(self) -> int:
        if self.symmetric:
            return -self.q_max
        return 0


def compute_scale(tensor: torch.Tensor, config: QuantConfig) -> torch.Tensor:
    """Compute quantization scale factor (per-channel for weights)."""
    if config.per_channel and tensor.dim() > 1:
        reduce_dims = [i for i in range(tensor.dim()) if i != config.channel_axis]
        abs_max = tensor.abs().amax(dim=reduce_dims, keepdim=True)
    else:
        abs_max = tensor.abs().amax()
    
    abs_max = torch.clamp(abs_max, min=1e-8)
    scale = abs_max / config.q_max
    return scale


def quantize_dequantize_weight(tensor: torch.Tensor, config: QuantConfig) -> torch.Tensor:
    """Quantize-dequantize a weight tensor (per-channel symmetric)."""
    scale = compute_scale(tensor, config)
    q_tensor = torch.clamp(torch.round(tensor / scale), config.q_min, config.q_max)
    return q_tensor * scale


def quantize_dequantize_activation(tensor: torch.Tensor, config: QuantConfig) -> torch.Tensor:
    """Quantize-dequantize an activation tensor (per-tensor asymmetric).
    
    Activations are quantized asymmetrically: the observed [min, max] range
    maps to the full unsigned integer range [0, 2^b - 1].  This is correct
    because activations are often non-negative (post-ReLU) or skewed
    (post-LayerNorm/GELU).  Symmetric quantization would waste half the
    codes on negative values that rarely or never appear.
    
    Per-tensor because activations are dynamic — per-channel scales would
    require calibration data collected in advance.
    """
    a_min = tensor.min()
    a_max = tensor.max()
    
    # Unsigned integer range for asymmetric quantization
    q_min = 0
    q_max = (1 << config.bits) - 1
    
    # scale and zero-point
    scale = (a_max - a_min) / max(q_max - q_min, 1)
    scale = torch.clamp(scale, min=1e-8)
    zero_point = torch.clamp(torch.round(q_min - a_min / scale), q_min, q_max)
    
    # quantize → dequantize
    q_tensor = torch.clamp(torch.round(tensor / scale + zero_point), q_min, q_max)
    return (q_tensor - zero_point) * scale


class _ActivationQuantHook:
    """Forward pre-hook: quantizes layer inputs (activations) before execution.
    
    On real integer hardware, both operands of the matmul/conv are in low
    precision. This hook simulates the activation side by intercepting
    inputs and applying per-tensor asymmetric quantize-dequantize.
    """
    def __init__(self, config: QuantConfig):
        self.config = config
    
    def __call__(self, module, inputs):
        quantized = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor) and inp.is_floating_point():
                quantized.append(quantize_dequantize_activation(inp, self.config))
            else:
                quantized.append(inp)
        return tuple(quantized)


def apply_ptq_to_model(model: nn.Module, config: QuantConfig) -> nn.Module:
    """Apply PTQ (weights + activations) to all Linear and Conv2d layers.
    
    Weights are quantized statically (per-channel). Activations are quantized
    dynamically via forward hooks (per-tensor). This matches what real integer
    hardware does: both operands of the matmul/conv are in low precision.
    """
    model = copy.deepcopy(model)
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Static weight quantization
            if hasattr(module, 'weight') and module.weight is not None:
                with torch.no_grad():
                    q_weight = quantize_dequantize_weight(module.weight, config)
                    module.weight.copy_(q_weight)
            
            # Dynamic activation quantization
            module.register_forward_pre_hook(_ActivationQuantHook(config))
    
    return model


def quantize_single_layer(model: nn.Module, layer_name: str, config: QuantConfig) -> nn.Module:
    """Quantize a single layer (weights + activations) in the model."""
    model = copy.deepcopy(model)
    
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight') and module.weight is not None:
                with torch.no_grad():
                    q_weight = quantize_dequantize_weight(module.weight, config)
                    module.weight.copy_(q_weight)
            module.register_forward_pre_hook(_ActivationQuantHook(config))
            break
    
    return model


# ============================================================================
# Vision Data Loading - ImageNette
# ============================================================================

def get_imagenette_loader(
    batch_size: int = 64,
    num_workers: int = 4,
    num_samples: Optional[int] = None,
    split: str = 'validation'
) -> Tuple[DataLoader, int]:
    """
    Load ImageNette dataset - a 10-class subset of ImageNet.
    
    Classes: tench, English springer, cassette player, chain saw, church,
             French horn, garbage truck, gas pump, golf ball, parachute
             
    Returns: (DataLoader, num_classes)
    """
    import urllib.request
    import tarfile
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    
    # Download and extract if needed
    data_dir = "./data/imagenette2"
    split_dir = "train" if split == "train" else "val"
    full_path = os.path.join(data_dir, split_dir)
    
    if not os.path.exists(full_path):
        print("Downloading ImageNette dataset...")
        os.makedirs("./data", exist_ok=True)
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
        tar_path = "./data/imagenette2.tgz"
        
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, tar_path)
        
        print("Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall("./data")
        os.remove(tar_path)
        print("Download complete!")
    else:
        print(f"Using existing ImageNette dataset at {data_dir}")
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Load dataset
    dataset = ImageFolder(full_path, transform=transform)
    
    # Limit samples if needed
    if num_samples and num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:num_samples].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return loader, 10  # ImageNette has 10 classes


# ============================================================================
# NLP Data Loading - GLUE SST-2
# ============================================================================

def get_sst2_loader(
    tokenizer,
    batch_size: int = 32,
    num_samples: Optional[int] = None,
    max_length: int = 128,
    split: str = 'validation'
) -> Tuple[DataLoader, int]:
    """
    Load GLUE SST-2 dataset for sentiment classification.
    
    Returns: (DataLoader, num_classes)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    print(f"Loading SST-2 {split} split...")
    dataset = load_dataset("glue", "sst2", split=split)
    
    if num_samples and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    def tokenize_fn(examples):
        return tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=['sentence', 'idx'])
    tokenized.set_format('torch')
    
    loader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=False
    )
    
    return loader, 2


# ============================================================================
# Model Loading
# ============================================================================

# ImageNette label indices correspond to these ImageNet class indices
IMAGENETTE_TO_IMAGENET = {
    0: 0,    # tench
    1: 217,  # English springer
    2: 482,  # cassette player
    3: 491,  # chain saw
    4: 497,  # church
    5: 566,  # French horn
    6: 569,  # garbage truck
    7: 571,  # gas pump
    8: 574,  # golf ball
    9: 701,  # parachute
}

IMAGENET_TO_IMAGENETTE = {v: k for k, v in IMAGENETTE_TO_IMAGENET.items()}


class ImageNetteWrapper(nn.Module):
    """Wrapper that maps ImageNet model outputs to ImageNette 10 classes."""
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        # Create indices for gathering
        self.register_buffer(
            'imagenet_indices',
            torch.tensor([IMAGENETTE_TO_IMAGENET[i] for i in range(10)])
        )
    
    def forward(self, x):
        # Get full ImageNet logits
        logits = self.base_model(x)
        # Select only the 10 ImageNette class logits
        return logits[:, self.imagenet_indices]


def load_vision_model(model_name: str, num_classes: int = 1000, pretrained: bool = True) -> nn.Module:
    """Load a vision model, with ImageNette wrapper if needed."""
    from torchvision import models
    
    model_map = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'mobilenet_v3_large': models.mobilenet_v3_large,
        'efficientnet_b0': models.efficientnet_b0,
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_map.keys())}")
    
    model_fn = model_map[model_name]
    
    try:
        weights = 'IMAGENET1K_V1'
        model = model_fn(weights=weights if pretrained else None)
    except TypeError:
        model = model_fn(pretrained=pretrained)
    
    # Wrap model to output only ImageNette classes if needed
    if num_classes == 10:
        model = ImageNetteWrapper(model)
    
    return model


def load_nlp_model(model_name: str = 'bert-base-uncased', num_classes: int = 2):
    """Load a BERT model for sequence classification."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")
    
    # Use a fine-tuned model for SST-2 to get meaningful results
    if model_name == 'bert-base-uncased':
        print(f"Loading fine-tuned BERT model for SST-2...")
        # Use a model that's actually trained on SST-2
        model_name = 'textattack/bert-base-uncased-SST-2'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    else:
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
    
    return model, tokenizer


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_quantizable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Get all quantizable layers (Linear, Conv2d) from a model."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layers.append((name, module))
    return layers


# ============================================================================
# Evaluation Functions
# ============================================================================

@torch.no_grad()
def evaluate_vision_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating"
) -> float:
    """Evaluate vision model top-1 accuracy."""
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    for images, labels in tqdm(data_loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_nlp_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    desc: str = "Evaluating"
) -> float:
    """Evaluate NLP model accuracy."""
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    for batch in tqdm(data_loader, desc=desc, leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = outputs.logits.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_per_class_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    task: str = 'vision'
) -> torch.Tensor:
    """Evaluate accuracy per class."""
    model.eval()
    model.to(device)
    
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    
    for batch in tqdm(data_loader, desc="Per-class eval", leave=False):
        if task == 'vision':
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
        else:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        _, predicted = outputs.max(1)
        
        for label, pred in zip(labels.cpu(), predicted.cpu()):
            total[label] += 1
            correct[label] += (pred == label).item()
    
    accuracy = torch.where(total > 0, correct / total * 100, torch.zeros_like(correct))
    return accuracy


@torch.no_grad()
def evaluate_confidence_distribution(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    task: str = 'vision'
) -> Dict[str, np.ndarray]:
    """Analyze confidence distribution."""
    model.eval()
    model.to(device)
    
    correct_confidences = []
    incorrect_confidences = []
    
    for batch in tqdm(data_loader, desc="Confidence analysis", leave=False):
        if task == 'vision':
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
        else:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        
        probs = F.softmax(outputs, dim=1)
        max_probs, preds = probs.max(1)
        correct_mask = (preds == labels)
        
        correct_confidences.extend(max_probs[correct_mask].cpu().tolist())
        incorrect_confidences.extend(max_probs[~correct_mask].cpu().tolist())
    
    return {
        'correct': np.array(correct_confidences),
        'incorrect': np.array(incorrect_confidences)
    }


# ============================================================================
# Diagnostic Functions
# ============================================================================

@dataclass
class BitwidthSweepResult:
    """Results from bit-width sweep analysis."""
    model_name: str
    task: str
    fp32_accuracy: float
    results: Dict[int, float] = field(default_factory=dict)
    
    def to_table(self) -> str:
        headers = ["Bits", "Accuracy", "Δ from FP32", "Status"]
        rows = [["FP32", f"{self.fp32_accuracy:.2f}%", "—", "Baseline"]]
        
        for bits in sorted(self.results.keys(), reverse=True):
            acc = self.results[bits]
            delta = acc - self.fp32_accuracy
            
            if abs(delta) < 1.0:
                status = "✓ Acceptable"
            elif abs(delta) < 3.0:
                status = "⚠ Marginal"
            elif abs(delta) < 10.0:
                status = "✗ Severe"
            else:
                status = "✗ Catastrophic"
            
            rows.append([f"INT{bits}", f"{acc:.2f}%", f"{delta:+.2f}%", status])
        
        return tabulate(rows, headers, tablefmt="simple")


def run_bitwidth_sweep(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    eval_fn: Callable,
    bits_list: List[int] = [8, 4],
    model_name: str = "model",
    task: str = "vision"
) -> BitwidthSweepResult:
    """Sweep across bit-widths and measure accuracy."""
    print(f"\n{'='*60}")
    print(f"Bit-Width Sweep: {model_name} ({task})")
    print(f"{'='*60}")
    
    # FP32 baseline
    fp32_acc = eval_fn(model, data_loader, device, "FP32 baseline")
    print(f"FP32 baseline: {fp32_acc:.2f}%")
    
    result = BitwidthSweepResult(
        model_name=model_name,
        task=task,
        fp32_accuracy=fp32_acc
    )
    
    for bits in bits_list:
        config = QuantConfig(bits=bits, per_channel=True)
        q_model = apply_ptq_to_model(model, config)
        acc = eval_fn(q_model, data_loader, device, f"INT{bits}")
        result.results[bits] = acc
        delta = acc - fp32_acc
        print(f"INT{bits}: {acc:.2f}% (Δ = {delta:+.2f}%)")
    
    return result


@dataclass
class LayerSensitivityResult:
    """Results from layer sensitivity analysis."""
    model_name: str
    bits: int
    baseline_accuracy: float
    layer_sensitivities: Dict[str, float] = field(default_factory=dict)
    
    def get_top_sensitive(self, n: int = 10) -> List[Tuple[str, float]]:
        sorted_layers = sorted(
            self.layer_sensitivities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_layers[:n]
    
    def to_table(self, top_n: int = 15) -> str:
        headers = ["Layer", "Accuracy Drop", "Status"]
        rows = []
        
        for layer_name, drop in self.get_top_sensitive(top_n):
            if drop > 2.0:
                status = "⚠ High sensitivity"
            elif drop > 0.5:
                status = "Moderate"
            elif drop > 0.1:
                status = "Low"
            else:
                status = "Negligible"
            
            rows.append([layer_name[:40], f"{drop:.2f}%", status])
        
        return tabulate(rows, headers, tablefmt="simple")


def run_layer_sensitivity_analysis(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    eval_fn: Callable,
    bits: int = 4,
    model_name: str = "model",
    max_layers: int = 10
) -> LayerSensitivityResult:
    """Quantize one layer at a time and measure accuracy impact."""
    print(f"\n{'='*60}")
    print(f"Layer Sensitivity Analysis: {model_name} @ INT{bits}")
    print(f"{'='*60}")
    
    baseline_acc = eval_fn(model, data_loader, device, "Baseline")
    print(f"Baseline accuracy: {baseline_acc:.2f}%")
    
    result = LayerSensitivityResult(
        model_name=model_name,
        bits=bits,
        baseline_accuracy=baseline_acc  
    )
    
    layers = get_quantizable_layers(model)
    if len(layers) > max_layers:
        print(f"Model has {len(layers)} layers, sampling {max_layers} for analysis")
        step = len(layers) // max_layers
        layers = layers[::step][:max_layers]
    
    print(f"Analyzing {len(layers)} quantizable layers...")
    
    config = QuantConfig(bits=bits, per_channel=True)
    
    for layer_name, _ in tqdm(layers, desc="Layer sensitivity"):
        q_model = quantize_single_layer(model, layer_name, config)
        acc = eval_fn(q_model, data_loader, device)
        sensitivity = baseline_acc - acc
        result.layer_sensitivities[layer_name] = sensitivity
    
    return result


@dataclass 
class PerClassResult:
    """Results from per-class accuracy analysis."""
    model_name: str
    bits: int
    fp32_per_class: np.ndarray
    quantized_per_class: np.ndarray
    
    @property
    def degradation(self) -> np.ndarray:
        return self.fp32_per_class - self.quantized_per_class
    
    def get_worst_classes(self, n: int = 10) -> List[Tuple[int, float, float, float]]:
        deg = self.degradation
        worst_indices = np.argsort(deg)[-n:][::-1]
        
        results = []
        for idx in worst_indices:
            results.append((
                int(idx),
                float(self.fp32_per_class[idx]),
                float(self.quantized_per_class[idx]),
                float(deg[idx])
            ))
        return results
    
    def compute_statistics(self) -> Dict[str, float]:
        deg = self.degradation
        return {
            'mean': float(np.mean(deg)),
            'std': float(np.std(deg)),
            'max': float(np.max(deg)),
            'pct_within_2pct': float(np.mean(np.abs(deg - np.mean(deg)) < 2.0) * 100)
        }


def run_per_class_analysis(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    bits: int = 4,
    num_classes: int = 10,
    model_name: str = "model",
    task: str = 'vision'
) -> PerClassResult:
    """Analyze per-class accuracy degradation."""
    print(f"\n{'='*60}")
    print(f"Per-Class Accuracy Analysis: {model_name} @ INT{bits}")
    print(f"{'='*60}")
    
    print("Computing FP32 per-class accuracy...")
    fp32_per_class = evaluate_per_class_accuracy(model, data_loader, device, num_classes, task)
    
    print(f"Computing INT{bits} per-class accuracy...")
    config = QuantConfig(bits=bits, per_channel=True)
    q_model = apply_ptq_to_model(model, config)
    quant_per_class = evaluate_per_class_accuracy(q_model, data_loader, device, num_classes, task)
    
    result = PerClassResult(
        model_name=model_name,
        bits=bits,
        fp32_per_class=fp32_per_class.numpy(),
        quantized_per_class=quant_per_class.numpy()
    )
    
    stats = result.compute_statistics()
    print(f"\nDegradation Statistics:")
    print(f"  Mean: {stats['mean']:.2f}%")
    print(f"  Std:  {stats['std']:.2f}%")
    print(f"  Max:  {stats['max']:.2f}%")
    print(f"  Classes within ±2% of mean: {stats['pct_within_2pct']:.1f}%")
    
    return result


@dataclass
class ConfidenceResult:
    """Results from confidence distribution analysis."""
    model_name: str
    bits: int
    fp32_confidence: Dict[str, np.ndarray]
    quantized_confidence: Dict[str, np.ndarray]
    
    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        def stats(conf: np.ndarray) -> Dict[str, float]:
            if len(conf) == 0:
                return {'mean': 0, 'median': 0, 'below_50': 0, 'below_70': 0}
            return {
                'mean': float(np.mean(conf)),
                'median': float(np.median(conf)),
                'below_50': float(np.mean(conf < 0.5) * 100),
                'below_70': float(np.mean(conf < 0.7) * 100)
            }
        
        return {
            'fp32': stats(self.fp32_confidence['correct']),
            'quantized': stats(self.quantized_confidence['correct'])
        }


def run_confidence_analysis(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    bits: int = 4,
    model_name: str = "model",
    task: str = 'vision'
) -> ConfidenceResult:
    """Analyze confidence distribution changes after quantization."""
    print(f"\n{'='*60}")
    print(f"Confidence Distribution Analysis: {model_name} @ INT{bits}")
    print(f"{'='*60}")
    
    print("Analyzing FP32 confidence distribution...")
    fp32_conf = evaluate_confidence_distribution(model, data_loader, device, task)
    
    print(f"Analyzing INT{bits} confidence distribution...")
    config = QuantConfig(bits=bits, per_channel=True)
    q_model = apply_ptq_to_model(model, config)
    quant_conf = evaluate_confidence_distribution(q_model, data_loader, device, task)
    
    result = ConfidenceResult(
        model_name=model_name,
        bits=bits,
        fp32_confidence=fp32_conf,
        quantized_confidence=quant_conf
    )
    
    stats = result.compute_statistics()
    print(f"\nConfidence Statistics (correct predictions):")
    print(f"{'Metric':<25} {'FP32':>10} {'INT'+str(bits):>10}")
    print("-" * 47)
    print(f"{'Mean confidence':<25} {stats['fp32']['mean']:>10.3f} {stats['quantized']['mean']:>10.3f}")
    print(f"{'Median confidence':<25} {stats['fp32']['median']:>10.3f} {stats['quantized']['median']:>10.3f}")
    print(f"{'Below 50% confidence':<25} {stats['fp32']['below_50']:>9.1f}% {stats['quantized']['below_50']:>9.1f}%")
    print(f"{'Below 70% confidence':<25} {stats['fp32']['below_70']:>9.1f}% {stats['quantized']['below_70']:>9.1f}%")
    
    return result





def run_model_size_comparison(
    data_loader: DataLoader,
    device: torch.device,
    bits: int = 4,
    num_classes: int = 10
) -> Dict[str, Dict[str, float]]:
    """Compare PTQ performance across models of different sizes."""
    print(f"\n{'='*60}")
    print(f"Model Size Comparison @ INT{bits}")
    print(f"{'='*60}")
    
    models_to_test = [
        ('mobilenet_v3_small', 'MobileNetV3-S'),
        ('mobilenet_v2', 'MobileNetV2'),
        ('resnet18', 'ResNet-18'),
        ('resnet50', 'ResNet-50'),
    ]
    
    results = {}
    
    for model_key, model_name in models_to_test:
        print(f"\nEvaluating {model_name}...")
        try:
            model = load_vision_model(model_key, num_classes=num_classes)
            num_params = count_parameters(model)
            
            fp32_acc = evaluate_vision_accuracy(model, data_loader, device, f"{model_name} FP32")
            
            config = QuantConfig(bits=bits, per_channel=True)
            q_model = apply_ptq_to_model(model, config)
            quant_acc = evaluate_vision_accuracy(q_model, data_loader, device, f"{model_name} INT{bits}")
            
            degradation = fp32_acc - quant_acc
            
            results[model_name] = {
                'num_params': num_params,
                'fp32_acc': fp32_acc,
                'quant_acc': quant_acc,
                'degradation': degradation
            }
            
            print(f"  Parameters: {num_params/1e6:.1f}M")
            print(f"  FP32: {fp32_acc:.2f}%, INT{bits}: {quant_acc:.2f}% (Δ = {degradation:+.2f}%)")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Print summary table
    print(f"\n{'Model':<20} {'Params':<10} {'FP32':<10} {'INT'+str(bits):<10} {'Δ':<10}")
    print("-" * 60)
    for model_name, data in sorted(results.items(), key=lambda x: x[1]['degradation'], reverse=True):
        print(f"{model_name:<20} {data['num_params']/1e6:<10.1f}M {data['fp32_acc']:<10.2f}% {data['quant_acc']:<10.2f}% {data['degradation']:+.2f}%")
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_bitwidth_sweep(result: BitwidthSweepResult, output_path: Optional[str] = None):
    """Plot accuracy vs bit-width."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bits = sorted(result.results.keys(), reverse=True)
    accuracies = [result.results[b] for b in bits]
    
    bits = [32] + bits
    accuracies = [result.fp32_accuracy] + accuracies
    
    colors = []
    for i, b in enumerate(bits):
        if b == 32:
            colors.append('green')
        else:
            delta = abs(result.results[b] - result.fp32_accuracy)
            if delta < 1:
                colors.append('blue')
            elif delta < 5:
                colors.append('orange')
            else:
                colors.append('red')
    
    bars = ax.bar([str(b) for b in bits], accuracies, color=colors, edgecolor='black')
    
    ax.axhline(y=result.fp32_accuracy, color='green', linestyle='--', alpha=0.7, label='FP32 baseline')
    ax.axhline(y=result.fp32_accuracy - 1, color='blue', linestyle=':', alpha=0.5, label='1% degradation')
    ax.axhline(y=result.fp32_accuracy - 5, color='orange', linestyle=':', alpha=0.5, label='5% degradation')
    
    ax.set_xlabel('Bit Width', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'PTQ Accuracy vs Bit-Width: {result.model_name}', fontsize=14)
    ax.legend(loc='lower left')
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_layer_sensitivity(result: LayerSensitivityResult, output_path: Optional[str] = None, top_n: int = 20):
    """Plot layer sensitivity analysis."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_layers = result.get_top_sensitive(top_n)
    layer_names = [name[:30] for name, _ in top_layers]
    sensitivities = [sens for _, sens in top_layers]
    
    colors = ['red' if s > 2 else 'orange' if s > 0.5 else 'green' for s in sensitivities]
    
    y_pos = np.arange(len(layer_names))
    ax.barh(y_pos, sensitivities, color=colors, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title(f'Layer Sensitivity to INT{result.bits} Quantization: {result.model_name}', fontsize=14)
    
    legend_elements = [
        mpatches.Patch(facecolor='red', edgecolor='black', label='High (>2%)'),
        mpatches.Patch(facecolor='orange', edgecolor='black', label='Moderate (0.5-2%)'),
        mpatches.Patch(facecolor='green', edgecolor='black', label='Low (<0.5%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chapter 5: PTQ Failure Diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Vision: ResNet-18 on ImageNette (recommended starting point)
    python ch5_ptq_failure_diagnostics.py --task vision --model resnet18 --all
    
    # NLP: BERT on SST-2
    python ch5_ptq_failure_diagnostics.py --task nlp --model bert-base-uncased --all
    
    # Quick bit-width sweep on vision
    python ch5_ptq_failure_diagnostics.py --task vision --bitwidth-sweep
    
    # Layer sensitivity at INT4
    python ch5_ptq_failure_diagnostics.py --task vision --layer-sensitivity --bits 4
    
    # Model size comparison
    python ch5_ptq_failure_diagnostics.py --task vision --model-comparison --bits 4
        """
    )
    
    # Task selection
    parser.add_argument('--task', type=str, default='vision', choices=['vision', 'nlp'],
                        help='Task type: vision (ImageNette) or nlp (SST-2)')
    
    # Model selection
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model name (vision: resnet18/50/mobilenet_v2, nlp: bert-base-uncased)')
    
    # Analysis selection
    parser.add_argument('--all', action='store_true', help='Run all diagnostics')
    parser.add_argument('--bitwidth-sweep', action='store_true', help='Run bit-width sweep')
    parser.add_argument('--layer-sensitivity', action='store_true', help='Run layer sensitivity analysis')
    parser.add_argument('--per-class', action='store_true', help='Run per-class accuracy analysis')
    parser.add_argument('--confidence', action='store_true', help='Run confidence distribution analysis')
    parser.add_argument('--model-comparison', action='store_true', help='Compare multiple model sizes')
    
    # Quantization settings
    parser.add_argument('--bits', type=int, default=4, help='Bit-width for single analyses (default: 4)')
    parser.add_argument('--bits-list', type=int, nargs='+', default=[8, 4],
                        help='Bit-widths for sweep')
    
    # Data settings
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples (default: full validation set)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loading workers')
    
    # Output settings
    parser.add_argument('--output-dir', type=str, default='./ptq_diagnostics', help='Output directory')
    parser.add_argument('--save-plots', action='store_true', help='Save plots')
    parser.add_argument('--save-json', action='store_true', help='Save results to JSON')
    
    # Device
    parser.add_argument('--device', type=str, default=None, help='Device (default: auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Setup output directory
    if args.save_plots or args.save_json:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine analyses to run
    run_bitwidth = args.bitwidth_sweep or args.all
    run_layer = args.layer_sensitivity or args.all
    run_class = args.per_class or args.all
    run_conf = args.confidence or args.all
    run_compare = args.model_comparison
    
    if not any([run_bitwidth, run_layer, run_class, run_conf, run_compare]):
        print("\nNo analysis specified. Running bit-width sweep as default...")
        run_bitwidth = True
    
    results = {}
    
    # Load data and model based on task
    if args.task == 'vision':
        print("\n" + "="*60)
        print("Loading ImageNette dataset...")
        print("="*60)
        data_loader, num_classes = get_imagenette_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_samples=args.num_samples
        )
        
        print(f"\nLoading model: {args.model}")
        model = load_vision_model(args.model, num_classes=num_classes)
        print(f"Parameters: {count_parameters(model):,}")
        
        eval_fn = evaluate_vision_accuracy
        
    else:  # nlp
        print("\n" + "="*60)
        print("Loading BERT and SST-2 dataset...")
        print("="*60)
        model, tokenizer = load_nlp_model(args.model)
        print(f"Parameters: {count_parameters(model):,}")
        
        data_loader, num_classes = get_sst2_loader(
            tokenizer,
            batch_size=args.batch_size,
            num_samples=args.num_samples
        )
        
        eval_fn = evaluate_nlp_accuracy
    
    # Run analyses
    if run_bitwidth:
        result = run_bitwidth_sweep(
            model, data_loader, device, eval_fn,
            bits_list=args.bits_list,
            model_name=args.model,
            task=args.task
        )
        results['bitwidth_sweep'] = asdict(result)
        print(f"\n{result.to_table()}")
        
        if args.save_plots:
            plot_bitwidth_sweep(result, os.path.join(args.output_dir, f'{args.model}_bitwidth_sweep.png'))
    
    if run_layer:
        result = run_layer_sensitivity_analysis(
            model, data_loader, device, eval_fn,
            bits=args.bits,
            model_name=args.model
        )
        results['layer_sensitivity'] = asdict(result)
        print(f"\nTop 15 Most Sensitive Layers:\n{result.to_table(15)}")
        
        if args.save_plots:
            plot_layer_sensitivity(result, os.path.join(args.output_dir, f'{args.model}_layer_sensitivity.png'))
    
    if run_class:
        result = run_per_class_analysis(
            model, data_loader, device,
            bits=args.bits,
            num_classes=num_classes,
            model_name=args.model,
            task=args.task
        )
        results['per_class'] = {
            'statistics': result.compute_statistics(),
            'worst_classes': result.get_worst_classes(10)
        }
        
        print(f"\nWorst {min(10, num_classes)} Classes:")
        for class_idx, fp32_acc, quant_acc, delta in result.get_worst_classes(min(10, num_classes)):
            print(f"  Class {class_idx}: {fp32_acc:.1f}% → {quant_acc:.1f}% (Δ = {delta:+.1f}%)")
    
    if run_conf:
        result = run_confidence_analysis(
            model, data_loader, device,
            bits=args.bits,
            model_name=args.model,
            task=args.task
        )
        results['confidence'] = result.compute_statistics()
    
    if run_compare and args.task == 'vision':
        results['model_comparison'] = run_model_size_comparison(
            data_loader, device,
            bits=args.bits,
            num_classes=num_classes
        )
    
    # Save results
    if args.save_json:
        json_path = os.path.join(args.output_dir, f'{args.model}_{args.task}_diagnostics.json')
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(json_path, 'w') as f:
            json.dump(convert(results), f, indent=2)
        print(f"\nSaved results to {json_path}")
    
    print("\n" + "="*60)
    print("Diagnostics complete!")
    print("="*60)


if __name__ == '__main__':
    main()