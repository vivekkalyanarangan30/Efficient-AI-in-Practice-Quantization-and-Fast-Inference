import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Load a real model practitioners use
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Hook to capture intermediate activations
activation_stats = {}

def capture_hook(name):
    def hook(module, input, output):
        # Handle different output types
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output
        activation_stats[name] = {
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'abs_max': tensor.abs().max().item()
        }
    return hook

# Register hook on a feed-forward layer
layer = model.encoder.layer[6].intermediate.dense
handle = layer.register_forward_hook(capture_hook('ffn_intermediate'))

# Two very different inputs
inputs_technical = tokenizer(
    "The quantum chromodynamics coupling constant varies logarithmically.",
    return_tensors="pt"
)
inputs_casual = tokenizer(
    "Hey, what's up?",
    return_tensors="pt"
)

# Forward pass 1: Technical text
with torch.no_grad():
    _ = model(**inputs_technical)
stats_technical = activation_stats['ffn_intermediate'].copy()

# Forward pass 2: Casual text  
with torch.no_grad():
    _ = model(**inputs_casual)
stats_casual = activation_stats['ffn_intermediate'].copy()

handle.remove()

print("FFN Intermediate Layer Statistics:")
print(f"  Technical text: max={stats_technical['abs_max']:.2f}, "
      f"std={stats_technical['std']:.3f}")
print(f"  Casual text:    max={stats_casual['abs_max']:.2f}, "
      f"std={stats_casual['std']:.3f}")
print(f"  Range ratio:    {stats_technical['abs_max']/stats_casual['abs_max']:.2f}x")


##############################
print("\n"*3)

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

def analyze_activation_distribution(model, tokenizer, text, layer_idx=6):
    """Analyze the distribution of activations in a specific layer."""
    
    activations = []
    
    def hook(module, input, output):
        if isinstance(output, tuple):
            activations.append(output[0].detach().cpu())
        else:
            activations.append(output.detach().cpu())
    
    layer = model.encoder.layer[layer_idx].intermediate.dense
    handle = layer.register_forward_hook(hook)
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        _ = model(**inputs)
    
    handle.remove()
    
    act = activations[0].numpy().flatten()
    
    # Compute distribution statistics
    percentiles = [50, 90, 95, 99, 99.9, 100]
    abs_act = np.abs(act)
    
    print(f"Activation Distribution Analysis (n={len(act):,} values):")
    print("-" * 50)
    for p in percentiles:
        val = np.percentile(abs_act, p)
        print(f"  {p:>5.1f}th percentile: {val:>8.3f}")
    
    # Outlier analysis
    p99 = np.percentile(abs_act, 99)
    p100 = np.percentile(abs_act, 100)
    outlier_ratio = p100 / p99
    
    print("-" * 50)
    print(f"  Outlier stretch (max / 99th): {outlier_ratio:.1f}x")
    
    # What happens to scale factor?
    scale_with_outliers = p100 / 127
    scale_without_outliers = p99 / 127
    
    print(f"\n  Scale factor with outliers:    {scale_with_outliers:.4f}")
    print(f"  Scale factor at 99th pctl:     {scale_without_outliers:.4f}")
    print(f"  Resolution loss from outliers: {scale_with_outliers/scale_without_outliers:.1f}x")
    
    return act

# Analyze with real text
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model.eval()

# Use a typical input
text = """Machine learning models are increasingly deployed in production 
environments where latency and memory constraints are critical. Quantization 
offers a practical path to efficiency without architectural changes."""

act = analyze_activation_distribution(model, tokenizer, text)

#############################
print("\n"*3)

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple

class StaticActivationCalibrator:
    """
    Calibrates activation ranges using a dataset before deployment.
    The scales are then fixed during inference.
    """
    
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.observers: Dict[str, ActivationObserver] = {}
        self.handles = []
        
        # Initialize observers for each layer
        for name in layer_names:
            self.observers[name] = ActivationObserver()
    
    def _get_layer(self, name: str) -> nn.Module:
        """Navigate to a layer by dot-separated name."""
        parts = name.split('.')
        module = self.model
        for part in parts:
            module = getattr(module, part)
        return module
    
    def calibrate(self, calibration_loader, num_batches: int = 100):
        """Run calibration data through the model to collect statistics."""
        
        # Register hooks
        for name in self.layer_names:
            layer = self._get_layer(name)
            observer = self.observers[name]
            
            def make_hook(obs):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        tensor = output[0]
                    else:
                        tensor = output
                    obs.observe(tensor.detach())
                return hook
            
            handle = layer.register_forward_hook(make_hook(observer))
            self.handles.append(handle)
        
        # Run calibration
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(calibration_loader):
                if batch_idx >= num_batches:
                    break
                _ = self.model(**batch)
        
        # Remove hooks
        for handle in self.handles:
            handle.remove()
        self.handles = []
        
        return self.get_scales()
    
    def get_scales(self, method: str = 'minmax') -> Dict[str, Tuple[float, int]]:
        """
        Compute scale and zero-point for each layer.
        
        Args:
            method: 'minmax' or 'percentile_99' or 'percentile_999'
        """
        scales = {}
        for name, obs in self.observers.items():
            if method == 'minmax':
                scale, zp = obs.get_scale_zeropoint(symmetric=False)
            elif method == 'percentile_99':
                scale, zp = obs.get_scale_zeropoint_percentile(99.0)
            elif method == 'percentile_999':
                scale, zp = obs.get_scale_zeropoint_percentile(99.9)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            scales[name] = (scale, zp)
        
        return scales


class ActivationObserver:
    """Collects running statistics for activation quantization."""
    
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.all_values = []  # For percentile computation
        self.count = 0
        
    def observe(self, tensor: torch.Tensor):
        """Update statistics with new observations."""
        self.min_val = min(self.min_val, tensor.min().item())
        self.max_val = max(self.max_val, tensor.max().item())
        
        # Store a subsample for percentile computation
        flat = tensor.flatten()
        if len(flat) > 10000:
            indices = torch.randperm(len(flat))[:10000]
            flat = flat[indices]
        self.all_values.extend(flat.tolist())
        self.count += tensor.numel()
    
    def get_scale_zeropoint(self, bits: int = 8, symmetric: bool = False):
        """Compute quantization parameters from collected statistics."""
        if symmetric:
            abs_max = max(abs(self.min_val), abs(self.max_val))
            q_max = (1 << (bits - 1)) - 1  # 127 for int8
            scale = abs_max / q_max if abs_max > 0 else 1.0
            zero_point = 0
        else:
            q_min, q_max = 0, (1 << bits) - 1  # 0-255 for uint8
            scale = (self.max_val - self.min_val) / (q_max - q_min)
            if scale == 0:
                scale = 1.0
            zero_point = int(round(q_min - self.min_val / scale))
        
        return scale, zero_point
    
    def get_scale_zeropoint_percentile(self, percentile: float, bits: int = 8):
        """Compute scale using percentile instead of absolute max."""
        import numpy as np
        values = np.array(self.all_values)
        
        p_low = np.percentile(values, 100 - percentile)
        p_high = np.percentile(values, percentile)
        
        q_min, q_max = 0, (1 << bits) - 1
        scale = (p_high - p_low) / (q_max - q_min)
        if scale == 0:
            scale = 1.0
        zero_point = int(round(q_min - p_low / scale))
        
        return scale, zero_point

class DynamicQuantizer:
    """
    Quantizes activations dynamically at runtime.
    Scale factors are computed for each forward pass.
    """
    
    def __init__(self, bits: int = 8, symmetric: bool = True):
        self.bits = bits
        self.symmetric = symmetric
        
        if symmetric:
            self.q_min = -(1 << (bits - 1)) + 1  # -127
            self.q_max = (1 << (bits - 1)) - 1    # 127
        else:
            self.q_min = 0
            self.q_max = (1 << bits) - 1          # 255
    
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """
        Dynamically quantize a tensor.
        
        Returns:
            (quantized_tensor, scale, zero_point)
        """
        if self.symmetric:
            abs_max = tensor.abs().max().item()
            scale = abs_max / self.q_max if abs_max > 0 else 1.0
            zero_point = 0
        else:
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            scale = (max_val - min_val) / (self.q_max - self.q_min)
            if scale == 0:
                scale = 1.0
            zero_point = int(round(self.q_min - min_val / scale))
        
        # Quantize
        q_tensor = torch.clamp(
            torch.round(tensor / scale + zero_point),
            self.q_min,
            self.q_max
        ).to(torch.int8)
        
        return q_tensor, scale, zero_point
    
    def dequantize(self, q_tensor: torch.Tensor, scale: float, 
                   zero_point: int) -> torch.Tensor:
        """Reconstruct the floating-point tensor."""
        return (q_tensor.float() - zero_point) * scale


# Demonstration: dynamic vs static on varying inputs
def compare_dynamic_static():
    """
    Show how dynamic quantization adapts while static may struggle.
    """
    torch.manual_seed(42)
    
    # Simulate calibration on "typical" data
    calibration_data = torch.randn(1000) * 2.0  # std=2
    
    # Static scale from calibration
    static_scale = calibration_data.abs().max().item() / 127
    
    # Test on different input distributions
    test_cases = [
        ("Typical (std=2)", torch.randn(1000) * 2.0),
        ("Narrow (std=0.5)", torch.randn(1000) * 0.5),
        ("Wide (std=5)", torch.randn(1000) * 5.0),
        ("With outliers", torch.cat([torch.randn(990) * 2.0, 
                                     torch.randn(10) * 20.0])),
    ]
    
    dynamic_q = DynamicQuantizer(bits=8, symmetric=True)
    
    print("Comparing Static vs Dynamic Quantization:")
    print("=" * 65)
    
    for name, data in test_cases:
        # Static quantization error
        q_static = torch.clamp(torch.round(data / static_scale), -127, 127)
        recon_static = q_static * static_scale
        mse_static = ((data - recon_static) ** 2).mean().item()
        
        # Dynamic quantization error
        q_dynamic, dyn_scale, _ = dynamic_q.quantize(data)
        recon_dynamic = q_dynamic.float() * dyn_scale
        mse_dynamic = ((data - recon_dynamic) ** 2).mean().item()
        
        print(f"\n{name}:")
        print(f"  Static scale: {static_scale:.4f}, MSE: {mse_static:.6f}")
        print(f"  Dynamic scale: {dyn_scale:.4f}, MSE: {mse_dynamic:.6f}")
        print(f"  Dynamic improvement: {mse_static/mse_dynamic:.2f}x")

compare_dynamic_static()


#############################
print("\n"*3)

class PerTokenQuantizer:
    """
    Quantizes activations with a separate scale per token.
    Shape: [batch, seq_len, hidden_dim] -> scale per [batch, seq_len]
    """
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.q_max = (1 << (bits - 1)) - 1  # 127 for symmetric
    
    def quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize with per-token scales.
        
        Args:
            tensor: Shape [batch, seq_len, hidden_dim]
            
        Returns:
            (quantized_tensor, scales) where scales has shape [batch, seq_len, 1]
        """
        # Compute abs max per token
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)  # [B, S, 1]
        
        # Compute scales
        scales = abs_max / self.q_max
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        
        # Quantize
        q_tensor = torch.clamp(
            torch.round(tensor / scales),
            -self.q_max,
            self.q_max
        ).to(torch.int8)
        
        return q_tensor, scales
    
    def dequantize(self, q_tensor: torch.Tensor, 
                   scales: torch.Tensor) -> torch.Tensor:
        """Reconstruct from quantized tensor and per-token scales."""
        return q_tensor.float() * scales


def compare_all_strategies():
    """
    Compare static, dynamic per-tensor, and dynamic per-token 
    on transformer activations with heterogeneous token magnitudes.
    """
    torch.manual_seed(42)
    batch, seq_len, hidden = 2, 128, 768
    
    # Simulate transformer hidden states
    hidden_states = torch.randn(batch, seq_len, hidden)
    
    # Make some tokens "important" (larger activations)
    important_tokens = [0, 10, 50, 100]
    for t in important_tokens:
        hidden_states[:, t, :] *= 5.0
    
    # 1. Static quantization (calibrated on similar but different data)
    calibration_data = torch.randn(batch, seq_len, hidden)
    for t in [0, 20, 60, 110]:  # Different important tokens
        calibration_data[:, t, :] *= 5.0
    static_scale = calibration_data.abs().max().item() / 127
    
    q_static = torch.clamp(torch.round(hidden_states / static_scale), -127, 127)
    recon_static = q_static * static_scale
    mse_static = ((hidden_states - recon_static) ** 2).mean().item()
    
    # 2. Dynamic per-tensor
    dynamic_scale = hidden_states.abs().max().item() / 127
    q_dynamic = torch.clamp(torch.round(hidden_states / dynamic_scale), -127, 127)
    recon_dynamic = q_dynamic * dynamic_scale
    mse_dynamic = ((hidden_states - recon_dynamic) ** 2).mean().item()
    
    # 3. Dynamic per-token
    per_token_q = PerTokenQuantizer(bits=8)
    q_token, scales_token = per_token_q.quantize(hidden_states)
    recon_token = per_token_q.dequantize(q_token, scales_token)
    mse_token = ((hidden_states - recon_token) ** 2).mean().item()
    
    print("Comparing All Activation Quantization Strategies:")
    print("=" * 60)
    print(f"\n1. Static (calibrated offline):")
    print(f"   Scale: {static_scale:.4f} (fixed)")
    print(f"   MSE: {mse_static:.6f}")
    
    print(f"\n2. Dynamic per-tensor:")
    print(f"   Scale: {dynamic_scale:.4f} (adapts per input)")
    print(f"   MSE: {mse_dynamic:.6f}")
    print(f"   vs Static: {mse_static/mse_dynamic:.2f}x better")
    
    print(f"\n3. Dynamic per-token:")
    print(f"   Scale range: [{scales_token.min():.4f}, {scales_token.max():.4f}]")
    print(f"   MSE: {mse_token:.6f}")
    print(f"   vs Static: {mse_static/mse_token:.2f}x better")
    print(f"   vs Dynamic per-tensor: {mse_dynamic/mse_token:.2f}x better")
    
    # Show the scale distribution
    print(f"\n   Scale breakdown by token type:")
    normal_scales = scales_token[0, [1,2,3,4,5], 0].mean().item()
    important_scales = scales_token[0, [0,10,50,100], 0].mean().item()
    print(f"     Normal tokens (avg):    {normal_scales:.4f}")
    print(f"     Important tokens (avg): {important_scales:.4f}")
    print(f"     Ratio: {important_scales/normal_scales:.1f}x")

compare_all_strategies()

#############################
print("\n"*3)

import torch
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict
import numpy as np

def identify_outlier_channels(model_name: str = "bert-base-uncased",
                              num_samples: int = 100):
    """
    Analyze which channels consistently produce large activations.
    """
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # Track max values per channel across samples
    channel_maxes = defaultdict(list)
    
    # Sample diverse inputs
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming industries worldwide.",
        "In 1969, humans first walked on the moon.",
        "The Pythagorean theorem states that a² + b² = c².",
        "Climate change poses significant challenges to ecosystems.",
        # ... more diverse samples would be used in practice
    ] * (num_samples // 5)
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            # Get max absolute value per channel (hidden dim)
            # tensor shape: [batch, seq, hidden]
            channel_max = tensor.abs().amax(dim=(0, 1))  # [hidden]
            channel_maxes[name].append(channel_max.detach().cpu())
        return hook
    
    # Register hooks on feed-forward layers
    handles = []
    for i, layer in enumerate(model.encoder.layer):
        h = layer.intermediate.dense.register_forward_hook(
            hook_fn(f'layer_{i}_ffn')
        )
        handles.append(h)
    
    # Run inference
    with torch.no_grad():
        for text in sample_texts[:num_samples]:
            inputs = tokenizer(text, return_tensors="pt", 
                             padding=True, truncation=True)
            _ = model(**inputs)
    
    # Clean up
    for h in handles:
        h.remove()
    
    # Analyze outlier patterns
    print("Outlier Channel Analysis:")
    print("=" * 60)
    
    for layer_name in sorted(channel_maxes.keys()):
        maxes = torch.stack(channel_maxes[layer_name])  # [samples, channels]
        
        # Average max per channel across samples
        avg_max = maxes.mean(dim=0)
        
        # Find channels that are consistently large
        global_median = avg_max.median().item()
        outlier_threshold = global_median * 5  # 5x median = outlier
        
        outlier_channels = (avg_max > outlier_threshold).sum().item()
        max_channel_idx = avg_max.argmax().item()
        max_channel_val = avg_max[max_channel_idx].item()
        
        print(f"\n{layer_name}:")
        print(f"  Median channel max: {global_median:.2f}")
        print(f"  Outlier channels (>5x median): {outlier_channels}")
        print(f"  Hottest channel: {max_channel_idx} "
              f"(avg max: {max_channel_val:.2f}, "
              f"ratio: {max_channel_val/global_median:.1f}x)")

# Run analysis
identify_outlier_channels()

#############################
print("\n"*3)

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Callable
import torch
import torch.nn as nn

class QuantStrategy(Enum):
    STATIC_PERTENSOR = "static_pertensor"
    STATIC_PERCENTILE = "static_percentile"
    DYNAMIC_PERTENSOR = "dynamic_pertensor"
    DYNAMIC_PERTOKEN = "dynamic_pertoken"
    MIXED_PRECISION = "mixed_precision"

@dataclass
class ActivationQuantConfig:
    """Configuration for activation quantization."""
    strategy: QuantStrategy
    bits: int = 8
    symmetric: bool = True
    percentile: float = 99.9  # For percentile-based calibration
    outlier_threshold: float = 6.0  # For mixed-precision
    calibration_samples: int = 512

class ActivationQuantizer:
    """
    Production activation quantizer supporting multiple strategies.
    """
    
    def __init__(self, config: ActivationQuantConfig):
        self.config = config
        self.calibrated = False
        self.scales: Dict[str, torch.Tensor] = {}
        self.zero_points: Dict[str, torch.Tensor] = {}
        self.outlier_masks: Dict[str, torch.Tensor] = {}
        
        # Statistics collection during calibration
        self.observers: Dict[str, ActivationObserver] = {}
    
    def add_observer(self, name: str):
        """Add an observer for a layer."""
        self.observers[name] = ActivationObserver()
    
    def observe(self, name: str, tensor: torch.Tensor):
        """Record statistics during calibration."""
        if name in self.observers:
            self.observers[name].observe(tensor)
    
    def compute_scales(self):
        """Compute quantization parameters from observed statistics."""
        
        for name, observer in self.observers.items():
            if self.config.strategy == QuantStrategy.STATIC_PERTENSOR:
                scale, zp = observer.get_scale_zeropoint(
                    bits=self.config.bits,
                    symmetric=self.config.symmetric
                )
                self.scales[name] = torch.tensor(scale)
                self.zero_points[name] = torch.tensor(zp)
                
            elif self.config.strategy == QuantStrategy.STATIC_PERCENTILE:
                scale, zp = observer.get_scale_zeropoint_percentile(
                    percentile=self.config.percentile,
                    bits=self.config.bits
                )
                self.scales[name] = torch.tensor(scale)
                self.zero_points[name] = torch.tensor(zp)
                
            elif self.config.strategy == QuantStrategy.MIXED_PRECISION:
                # Identify outlier channels
                channel_maxes = torch.stack(observer.channel_maxes).mean(dim=0)
                median_max = channel_maxes.median()
                outlier_mask = channel_maxes > (median_max * self.config.outlier_threshold)
                self.outlier_masks[name] = outlier_mask
                
                # Compute scale for non-outlier channels
                normal_vals = torch.tensor(observer.all_values)
                scale = normal_vals.abs().max() / 127
                self.scales[name] = torch.tensor(scale)
                self.zero_points[name] = torch.tensor(0)
        
        self.calibrated = True
    
    def quantize(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize a tensor using the configured strategy.
        
        For static strategies: uses pre-computed scales
        For dynamic strategies: computes scales on-the-fly
        """
        
        if self.config.strategy in [QuantStrategy.STATIC_PERTENSOR,
                                    QuantStrategy.STATIC_PERCENTILE]:
            return self._static_quantize(name, tensor)
        
        elif self.config.strategy == QuantStrategy.DYNAMIC_PERTENSOR:
            return self._dynamic_pertensor_quantize(tensor)
        
        elif self.config.strategy == QuantStrategy.DYNAMIC_PERTOKEN:
            return self._dynamic_pertoken_quantize(tensor)
        
        elif self.config.strategy == QuantStrategy.MIXED_PRECISION:
            return self._mixed_precision_quantize(name, tensor)
    
    def _static_quantize(self, name: str, tensor: torch.Tensor):
        """Static quantization with pre-computed scales."""
        assert self.calibrated, "Must calibrate before static quantization"
        
        scale = self.scales[name]
        zp = self.zero_points[name]
        
        q_min = -127 if self.config.symmetric else 0
        q_max = 127 if self.config.symmetric else 255
        
        q_tensor = torch.clamp(
            torch.round(tensor / scale + zp),
            q_min, q_max
        ).to(torch.int8)
        
        return q_tensor, scale, zp
    
    def _dynamic_pertensor_quantize(self, tensor: torch.Tensor):
        """Dynamic per-tensor quantization."""
        abs_max = tensor.abs().max()
        scale = abs_max / 127 if abs_max > 0 else torch.tensor(1.0)
        
        q_tensor = torch.clamp(
            torch.round(tensor / scale),
            -127, 127
        ).to(torch.int8)
        
        return q_tensor, scale, torch.tensor(0)
    
    def _dynamic_pertoken_quantize(self, tensor: torch.Tensor):
        """
        Dynamic per-token quantization.
        Expects tensor of shape [batch, seq_len, hidden_dim].
        """
        # Scale per token position
        abs_max = tensor.abs().amax(dim=-1, keepdim=True)  # [B, S, 1]
        scales = abs_max / 127
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        
        q_tensor = torch.clamp(
            torch.round(tensor / scales),
            -127, 127
        ).to(torch.int8)
        
        return q_tensor, scales, torch.zeros_like(scales)
    
    def _mixed_precision_quantize(self, name: str, tensor: torch.Tensor):
        """Mixed-precision quantization with outlier handling."""
        assert self.calibrated, "Must calibrate for mixed precision"
        
        outlier_mask = self.outlier_masks.get(name)
        
        if outlier_mask is None or outlier_mask.sum() == 0:
            # No outliers: standard quantization
            return self._dynamic_pertensor_quantize(tensor)
        
        # Return tensor with outlier info for downstream handling
        return tensor, self.scales[name], outlier_mask


def create_quantized_model_wrapper(model: nn.Module, 
                                   config: ActivationQuantConfig) -> nn.Module:
    """
    Create a wrapper that applies activation quantization to a model.
    
    This is a simplified version - production implementations would
    fuse quantization into the model graph for efficiency.
    """
    
    quantizer = ActivationQuantizer(config)
    
    class QuantizedWrapper(nn.Module):
        def __init__(self, base_model, quant):
            super().__init__()
            self.base_model = base_model
            self.quantizer = quant
            self.hooks = []
            
        def calibrate(self, dataloader, num_batches: int = None):
            """Run calibration on a dataset."""
            num_batches = num_batches or self.quantizer.config.calibration_samples
            
            # Setup observers
            for name, module in self.base_model.named_modules():
                if isinstance(module, nn.Linear):
                    self.quantizer.add_observer(name)
            
            # Run calibration
            self.base_model.eval()
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= num_batches:
                        break
                    # Forward pass triggers hooks
                    _ = self.base_model(**batch)
            
            # Compute scales
            self.quantizer.compute_scales()
            
        def forward(self, *args, **kwargs):
            # In production, quantization would be fused into layers
            return self.base_model(*args, **kwargs)
    
    return QuantizedWrapper(model, quantizer)