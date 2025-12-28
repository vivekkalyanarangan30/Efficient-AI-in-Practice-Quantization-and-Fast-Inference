import torch
import numpy as np

def analyze_kv_asymmetry():
    """
    Demonstrate the key/value asymmetry using simulated patterns
    based on published observations from real models.
    """
    torch.manual_seed(42)
    batch, heads, seq_len, head_dim = 1, 8, 512, 128
    
    # Simulate Keys: channel-wise outliers (consistent across tokens)
    keys = torch.randn(batch, heads, seq_len, head_dim) * 0.5
    outlier_channels = [7, 23, 64, 89, 120]  # Same channels always hot
    for ch in outlier_channels:
        keys[..., ch] *= torch.tensor([8.0, 12.0, 15.0, 10.0, 7.0])[
            outlier_channels.index(ch)
        ]
    
    # Simulate Values: token-wise variation (no consistent channel pattern)
    values = torch.randn(batch, heads, seq_len, head_dim) * 0.5
    # Attention sink tokens (early tokens) have larger magnitudes
    values[:, :, :4, :] *= 3.0
    # Some random tokens also spike
    spike_tokens = torch.randint(0, seq_len, (20,))
    for t in spike_tokens:
        values[:, :, t, :] *= 2.5
    
    # Analyze Keys: check channel consistency across tokens
    key_channel_max = keys.abs().amax(dim=(0, 1, 2))  # Max across batch, heads, seq
    key_median = key_channel_max.median()
    key_outlier_channels = (key_channel_max > key_median * 5).sum().item()
    
    # Analyze Values: check token consistency across channels  
    value_token_max = values.abs().amax(dim=(0, 1, 3))  # Max across batch, heads, channels
    value_median = value_token_max.median()
    value_outlier_tokens = (value_token_max > value_median * 2).sum().item()
    
    print("Key/Value Asymmetry Analysis")
    print("=" * 60)
    print(f"\nKeys (channel-wise patterns):")
    print(f"  Outlier channels (>5x median): {key_outlier_channels}")
    print(f"  Max/Median ratio: {key_channel_max.max() / key_median:.1f}x")
    
    print(f"\nValues (token-wise patterns):")
    print(f"  Outlier tokens (>2x median): {value_outlier_tokens}")
    print(f"  Max/Median ratio: {value_token_max.max() / value_median:.1f}x")
    
    # Helper: compute quantization MSE
    def quantize_mse(tensor, scales):
        q = torch.clamp(torch.round(tensor / scales), -127, 127)
        return ((tensor - q * scales) ** 2).mean().item()
    
    # Per-channel scales: one scale per channel (last dim)
    def get_per_channel_scales(t):
        abs_max = t.abs().amax(dim=(0, 1, 2), keepdim=True)  # [1,1,1,head_dim]
        scales = abs_max / 127
        return torch.where(scales == 0, torch.ones_like(scales), scales)
    
    # Per-token scales: one scale per token position
    def get_per_token_scales(t):
        abs_max = t.abs().amax(dim=-1, keepdim=True)  # [B,H,S,1]
        scales = abs_max / 127
        return torch.where(scales == 0, torch.ones_like(scales), scales)
    
    # Compare BOTH granularities on BOTH tensor types
    key_pc_scales = get_per_channel_scales(keys)
    key_pt_scales = get_per_token_scales(keys)
    val_pc_scales = get_per_channel_scales(values)
    val_pt_scales = get_per_token_scales(values)
    
    key_mse_per_channel = quantize_mse(keys, key_pc_scales)
    key_mse_per_token = quantize_mse(keys, key_pt_scales)
    val_mse_per_channel = quantize_mse(values, val_pc_scales)
    val_mse_per_token = quantize_mse(values, val_pt_scales)
    
    print(f"\nQuantization Error Comparison (MSE):")
    print(f"  Keys:")
    print(f"    Per-channel: {key_mse_per_channel:.6f}")
    print(f"    Per-token:   {key_mse_per_token:.6f}")
    print(f"    Winner: {'Per-channel' if key_mse_per_channel < key_mse_per_token else 'Per-token'} "
          f"({max(key_mse_per_channel, key_mse_per_token) / min(key_mse_per_channel, key_mse_per_token):.1f}x better)")
    
    print(f"  Values:")
    print(f"    Per-channel: {val_mse_per_channel:.6f}")
    print(f"    Per-token:   {val_mse_per_token:.6f}")
    print(f"    Winner: {'Per-channel' if val_mse_per_channel < val_mse_per_token else 'Per-token'} "
          f"({max(val_mse_per_channel, val_mse_per_token) / min(val_mse_per_channel, val_mse_per_token):.1f}x better)")

analyze_kv_asymmetry()

#######################
print("\n"*3)

import torch
import math

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0):
    """Apply Rotary Position Embeddings."""
    batch, heads, seq_len, dim = x.shape
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(positions.float(), inv_freq)
    cos, sin = freqs.cos().unsqueeze(0).unsqueeze(0), freqs.sin().unsqueeze(0).unsqueeze(0)
    
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack([rotated_even, rotated_odd], dim=-1).flatten(start_dim=-2)

def demonstrate_rope_effect():
    """Show how RoPE dilutes outlier structure."""
    torch.manual_seed(42)
    keys = torch.randn(1, 1, 100, 64)
    outlier_channels = [7, 23, 45]
    for ch in outlier_channels:
        keys[..., ch] *= 10.0
    
    # Analyze before RoPE
    pre_channel_max = keys.abs().amax(dim=(0, 1, 2))
    pre_outlier_ratio = pre_channel_max.max() / pre_channel_max.median()
    
    # Apply RoPE
    keys_post = apply_rope(keys, torch.arange(100))
    post_channel_max = keys_post.abs().amax(dim=(0, 1, 2))
    post_outlier_ratio = post_channel_max.max() / post_channel_max.median()
    
    print(f"Outlier ratio before RoPE: {pre_outlier_ratio:.1f}x")
    print(f"Outlier ratio after RoPE:  {post_outlier_ratio:.1f}x")
    print(f"RoPE dilutes structure by: {pre_outlier_ratio/post_outlier_ratio:.1f}x")

demonstrate_rope_effect()

#######################
print("\n"*3)

import torch
from typing import Tuple, NamedTuple
from dataclasses import dataclass

class QuantizedTensor(NamedTuple):
    data: torch.Tensor      # Quantized values
    scales: torch.Tensor    # Per-channel or per-token scales

@dataclass
class KVCacheConfig:
    key_bits: int = 4
    value_bits: int = 4
    residual_length: int = 128  # Recent tokens kept in FP16
    
class MixedPrecisionKVCache:
    """
    KV cache with asymmetric quantization: per-channel for keys, per-token for values.
    """
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, 
                 config: KVCacheConfig = None):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.config = config or KVCacheConfig()
        
        self.key_q_max = (1 << (self.config.key_bits - 1)) - 1
        self.value_q_max = (1 << (self.config.value_bits - 1)) - 1
        self.caches = [None] * num_layers
        
    def _quantize_keys_per_channel(self, keys: torch.Tensor) -> QuantizedTensor:
        """Per-channel quantization for keys."""
        # Scale per channel dimension (last axis)
        abs_max = keys.abs().amax(dim=(0, 1, 2), keepdim=True)
        scales = abs_max / self.key_q_max
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        q_keys = torch.clamp(torch.round(keys / scales), 
                             -self.key_q_max, self.key_q_max).to(torch.int8)
        return QuantizedTensor(q_keys, scales.squeeze())
    
    def _quantize_values_per_token(self, values: torch.Tensor) -> QuantizedTensor:
        """Per-token quantization for values."""
        # Scale per token position
        abs_max = values.abs().amax(dim=-1, keepdim=True)
        scales = abs_max / self.value_q_max
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        q_values = torch.clamp(torch.round(values / scales),
                               -self.value_q_max, self.value_q_max).to(torch.int8)
        return QuantizedTensor(q_values, scales.squeeze(-1))
    
    def update(self, layer_idx: int, new_keys: torch.Tensor, 
               new_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return full KV for attention."""
        if self.caches[layer_idx] is None:
            self.caches[layer_idx] = {
                'q_keys': None, 'q_values': None,
                'residual_keys': new_keys, 'residual_values': new_values
            }
        else:
            cache = self.caches[layer_idx]
            cache['residual_keys'] = torch.cat([cache['residual_keys'], new_keys], dim=2)
            cache['residual_values'] = torch.cat([cache['residual_values'], new_values], dim=2)
            
            # Quantize overflow when residual exceeds threshold
            residual_len = cache['residual_keys'].shape[2]
            if residual_len > self.config.residual_length:
                overflow = residual_len - self.config.residual_length
                
                # Quantize with appropriate granularity
                new_q_keys = self._quantize_keys_per_channel(
                    cache['residual_keys'][:, :, :overflow, :])
                new_q_values = self._quantize_values_per_token(
                    cache['residual_values'][:, :, :overflow, :])
                
                # Update residual
                cache['residual_keys'] = cache['residual_keys'][:, :, overflow:, :]
                cache['residual_values'] = cache['residual_values'][:, :, overflow:, :]
                
                # Append to quantized cache
                if cache['q_keys'] is None:
                    cache['q_keys'], cache['q_values'] = new_q_keys, new_q_values
                else:
                    cache['q_keys'] = QuantizedTensor(
                        torch.cat([cache['q_keys'].data, new_q_keys.data], dim=2),
                        cache['q_keys'].scales)
                    cache['q_values'] = QuantizedTensor(
                        torch.cat([cache['q_values'].data, new_q_values.data], dim=2),
                        torch.cat([cache['q_values'].scales, new_q_values.scales], dim=2))
        
        return self._get_full_cache(layer_idx)
    
    def _get_full_cache(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and concatenate all cache components."""
        cache = self.caches[layer_idx]
        if cache['q_keys'] is None:
            return cache['residual_keys'], cache['residual_values']
        
        # Dequantize
        deq_keys = cache['q_keys'].data.float() * cache['q_keys'].scales
        deq_values = cache['q_values'].data.float() * cache['q_values'].scales.unsqueeze(-1)
        
        return (torch.cat([deq_keys, cache['residual_keys']], dim=2),
                torch.cat([deq_values, cache['residual_values']], dim=2))

# Quick demonstration
print("Mixed-Precision KV Cache Demo")
print("=" * 50)

cache = MixedPrecisionKVCache(num_layers=4, num_heads=8, head_dim=64,
                               config=KVCacheConfig(key_bits=4, value_bits=4, residual_length=64))

torch.manual_seed(42)
for step in range(256):
    k = torch.randn(1, 8, 1, 64)
    v = torch.randn(1, 8, 1, 64)
    k[..., [7, 23]] *= 5.0  # Channel outliers in keys
    for layer in range(4):
        full_k, full_v = cache.update(layer, k, v)

print(f"Sequence length: {full_k.shape[2]}")
print(f"Keys shape: {full_k.shape}, Values shape: {full_v.shape}")

#########################
print("\n"*3)

class DenseAndSparseQuantizer:
    """
    Dense-and-sparse quantization for ultra-low precision (INT2).
    Based on KVQuant: isolate 1% outliers in FP16, quantize rest aggressively.
    """
    def __init__(self, bits: int = 2, outlier_percentile: float = 99.0):
        self.bits = bits
        self.q_max = (1 << (bits - 1)) - 1
        self.outlier_percentile = outlier_percentile
        
    def quantize(self, tensor: torch.Tensor):
        """Returns (dense_quantized, scales, sparse_values, sparse_indices)."""
        # Find outlier threshold
        threshold = torch.quantile(tensor.abs().flatten(), self.outlier_percentile / 100.0)
        outlier_mask = tensor.abs() > threshold
        
        # Sparse: store outliers separately
        sparse_values = tensor[outlier_mask]
        sparse_indices = outlier_mask.nonzero(as_tuple=False)
        
        # Dense: quantize non-outliers
        tensor_clean = tensor.clone()
        tensor_clean[outlier_mask] = 0.0
        
        abs_max = tensor_clean.abs().amax(dim=(0, 1, 2), keepdim=True)
        scales = abs_max / self.q_max
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        
        dense_q = torch.clamp(torch.round(tensor_clean / scales),
                              -self.q_max, self.q_max).to(torch.int8)
        
        return dense_q, scales, sparse_values, sparse_indices
    
    def memory_efficiency(self, original_numel: int, sparse_count: int) -> float:
        """Calculate compression ratio."""
        original_bytes = original_numel * 2  # FP16
        dense_bytes = original_numel * (self.bits / 8)
        sparse_bytes = sparse_count * (2 + 8)  # FP16 value + INT64 index
        return original_bytes / (dense_bytes + sparse_bytes)

# Demo
torch.manual_seed(42)
tensor = torch.randn(1, 8, 1024, 64)
tensor[..., [7, 23]] *= 15.0  # Outliers

quantizer = DenseAndSparseQuantizer(bits=2)
dense_q, scales, sparse_vals, sparse_idx = quantizer.quantize(tensor)

print(f"Dense-and-Sparse INT2 Quantization")
print(f"  Sparse outliers: {len(sparse_vals):,} ({len(sparse_vals)/tensor.numel()*100:.1f}%)")
print(f"  Compression ratio: {quantizer.memory_efficiency(tensor.numel(), len(sparse_vals)):.2f}x")