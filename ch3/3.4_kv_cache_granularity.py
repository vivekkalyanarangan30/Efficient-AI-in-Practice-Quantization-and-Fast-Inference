#!/usr/bin/env python3
"""
KV Cache Granularity Verification - Chapter 3.4
================================================
Verifies the key/value asymmetry and optimal quantization granularity.
Does NOT cover production concerns (attention sinks, three-tier architecture) - see Chapter 7.

Usage:
    python kv_cache_granularity.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python kv_cache_granularity.py --model gpt2
"""

import torch
from typing import Dict, Tuple, List
import argparse
import warnings
warnings.filterwarnings('ignore')


def get_device():
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model(model_name: str):
    """Load model and extract architecture info."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    device = get_device()
    model = model.to(device).eval()
    
    config = model.config
    info = {
        'num_layers': getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 12)),
        'num_heads': getattr(config, 'num_attention_heads', getattr(config, 'n_head', 12)),
        'num_kv_heads': getattr(config, 'num_key_value_heads', 
                                getattr(config, 'num_attention_heads', 12)),
        'hidden_dim': getattr(config, 'hidden_size', getattr(config, 'n_embd', 768)),
    }
    info['head_dim'] = info['hidden_dim'] // info['num_heads']
    
    print(f"  Layers: {info['num_layers']}, KV Heads: {info['num_kv_heads']}, Head dim: {info['head_dim']}")
    
    return model, tokenizer, device, info


def extract_kv_cache(model, tokenizer, device, text: str) -> Dict[int, Dict[str, torch.Tensor]]:
    """Extract KV cache from a forward pass."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    
    kv_cache = {}
    for layer_idx, layer_kv in enumerate(outputs.past_key_values):
        keys, values = layer_kv[0], layer_kv[1]
        kv_cache[layer_idx] = {
            'keys': keys.detach().cpu(),
            'values': values.detach().cpu()
        }
    
    return kv_cache


# =============================================================================
# Claim 1: Keys have channel-wise structure, values have token-wise structure
# =============================================================================

def analyze_asymmetry(keys: torch.Tensor, values: torch.Tensor) -> Dict:
    """
    Analyze the statistical structure of keys and values.
    
    Keys should show channel-wise outliers (same dimensions large across tokens).
    Values should show token-wise variation (certain tokens large across channels).
    """
    if keys.dim() == 3:
        keys, values = keys.unsqueeze(0), values.unsqueeze(0)
    
    # Keys: channel-wise analysis
    # Find max absolute value per channel (aggregating across batch, heads, seq)
    key_channel_max = keys.abs().amax(dim=(0, 1, 2))  # [head_dim]
    key_median = key_channel_max.median()
    key_outliers = (key_channel_max > key_median * 3).sum().item()
    key_ratio = (key_channel_max.max() / key_median).item()
    
    # Measure channel consistency: correlation of channel magnitudes across adjacent tokens
    key_per_token = keys.abs().amax(dim=(0, 1))  # [seq_len, head_dim]
    correlations = []
    for i in range(min(20, key_per_token.shape[0] - 1)):
        corr = torch.corrcoef(torch.stack([key_per_token[i], key_per_token[i+1]]))[0, 1]
        if not torch.isnan(corr):
            correlations.append(corr.item())
    key_consistency = sum(correlations) / len(correlations) if correlations else 0
    
    # Values: token-wise analysis
    # Find max absolute value per token (aggregating across batch, heads, channels)
    val_token_max = values.abs().amax(dim=(0, 1, 3))  # [seq_len]
    val_median = val_token_max.median()
    val_outliers = (val_token_max > val_median * 2).sum().item()
    val_ratio = (val_token_max.max() / val_median).item()
    
    return {
        'key_outlier_channels': key_outliers,
        'key_max_median_ratio': key_ratio,
        'key_channel_consistency': key_consistency,
        'val_outlier_tokens': val_outliers,
        'val_max_median_ratio': val_ratio,
    }


# =============================================================================
# Claim 2: Per-channel wins for keys, per-token wins for values
# =============================================================================

def compare_granularity(keys: torch.Tensor, values: torch.Tensor, bits: int = 4) -> Dict:
    """
    Compare per-channel vs per-token quantization for both keys and values.
    
    Returns MSE for each combination and declares winners.
    """
    if keys.dim() == 3:
        keys, values = keys.unsqueeze(0), values.unsqueeze(0)
    
    q_max = (1 << (bits - 1)) - 1
    
    def quantize_mse(tensor: torch.Tensor, scales: torch.Tensor) -> float:
        """Compute MSE from quantization with given scales."""
        q = torch.clamp(torch.round(tensor / scales), -q_max, q_max)
        return ((tensor - q * scales) ** 2).mean().item()
    
    def per_channel_scales(t: torch.Tensor) -> torch.Tensor:
        """One scale per channel (head_dim dimension)."""
        scales = t.abs().amax(dim=(0, 1, 2), keepdim=True) / q_max
        return torch.where(scales == 0, torch.ones_like(scales), scales)
    
    def per_token_scales(t: torch.Tensor) -> torch.Tensor:
        """One scale per token position."""
        scales = t.abs().amax(dim=-1, keepdim=True) / q_max
        return torch.where(scales == 0, torch.ones_like(scales), scales)
    
    results = {}
    
    # Test both strategies on keys
    k_pc_mse = quantize_mse(keys, per_channel_scales(keys))
    k_pt_mse = quantize_mse(keys, per_token_scales(keys))
    k_winner = 'per_channel' if k_pc_mse < k_pt_mse else 'per_token'
    k_ratio = max(k_pc_mse, k_pt_mse) / (min(k_pc_mse, k_pt_mse) + 1e-10)
    
    results['keys'] = {
        'per_channel_mse': k_pc_mse,
        'per_token_mse': k_pt_mse,
        'winner': k_winner,
        'improvement': k_ratio
    }
    
    # Test both strategies on values
    v_pc_mse = quantize_mse(values, per_channel_scales(values))
    v_pt_mse = quantize_mse(values, per_token_scales(values))
    v_winner = 'per_channel' if v_pc_mse < v_pt_mse else 'per_token'
    v_ratio = max(v_pc_mse, v_pt_mse) / (min(v_pc_mse, v_pt_mse) + 1e-10)
    
    results['values'] = {
        'per_channel_mse': v_pc_mse,
        'per_token_mse': v_pt_mse,
        'winner': v_winner,
        'improvement': v_ratio
    }
    
    return results


# =============================================================================
# Main verification
# =============================================================================

def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_verification(model_name: str = "gpt2"):
    """Run all granularity verification tests."""
    
    print_section("KV CACHE GRANULARITY VERIFICATION (Chapter 3.4)")
    print(f"Model: {model_name}")
    
    model, tokenizer, device, info = load_model(model_name)
    
    # Test text for KV cache extraction
    test_text = """The transformer architecture has fundamentally changed natural language processing. 
    Self-attention allows models to weigh the importance of different parts of the input. 
    Modern large language models like GPT and Llama build on these foundations."""
    
    print_section("EXTRACTING KV CACHE")
    kv_cache = extract_kv_cache(model, tokenizer, device, test_text)
    seq_len = kv_cache[0]['keys'].shape[-2]
    print(f"Sequence length: {seq_len}")
    print(f"Keys shape: {kv_cache[0]['keys'].shape}")
    print(f"Values shape: {kv_cache[0]['values'].shape}")
    
    # Select layers to analyze: first, middle, last
    layers = [0, info['num_layers'] // 2, info['num_layers'] - 1]
    
    # =========================================================================
    # Claim 1: Asymmetry analysis
    # =========================================================================
    print_section("CLAIM 1: Keys have channel-wise structure, Values have token-wise structure")
    
    print("\n{:<8} {:<12} {:<12} {:<15} {:<12}".format(
        "Layer", "Key Outliers", "Key Ratio", "Key Consistency", "Val Outliers"))
    print("-" * 65)
    
    consistencies = []
    for layer_idx in layers:
        keys = kv_cache[layer_idx]['keys']
        values = kv_cache[layer_idx]['values']
        result = analyze_asymmetry(keys, values)
        consistencies.append(result['key_channel_consistency'])
        
        print("{:<8} {:<12} {:<12.1f}x {:<15.3f} {:<12}".format(
            layer_idx,
            result['key_outlier_channels'],
            result['key_max_median_ratio'],
            result['key_channel_consistency'],
            result['val_outlier_tokens']
        ))
    
    avg_consistency = sum(consistencies) / len(consistencies)
    claim1_pass = avg_consistency > 0.5
    print(f"\nAverage key channel consistency: {avg_consistency:.3f}")
    print(f"VERDICT: {'✓ CONFIRMED' if claim1_pass else '✗ NOT CONFIRMED'} "
          f"(threshold: 0.5)")
    
    # =========================================================================
    # Claim 2: Granularity comparison
    # =========================================================================
    print_section("CLAIM 2: Per-channel wins for Keys, Per-token wins for Values")
    
    key_wins = 0
    val_wins = 0
    total_tests = 0
    
    for bits in [8, 4]:
        print(f"\n--- INT{bits} ---")
        print("{:<8} {:<20} {:<20}".format("Layer", "Keys Winner", "Values Winner"))
        print("-" * 50)
        
        for layer_idx in layers:
            keys = kv_cache[layer_idx]['keys']
            values = kv_cache[layer_idx]['values']
            result = compare_granularity(keys, values, bits)
            
            total_tests += 1
            if result['keys']['winner'] == 'per_channel':
                key_wins += 1
            if result['values']['winner'] == 'per_token':
                val_wins += 1
            
            print("{:<8} {:<20} {:<20}".format(
                layer_idx,
                f"{result['keys']['winner']} ({result['keys']['improvement']:.1f}x)",
                f"{result['values']['winner']} ({result['values']['improvement']:.1f}x)"
            ))
    
    print(f"\nKeys: per-channel wins {key_wins}/{total_tests}")
    print(f"Values: per-token wins {val_wins}/{total_tests}")
    
    claim2_keys = key_wins > total_tests // 2
    claim2_vals = val_wins > total_tests // 2
    
    print(f"\nVERDICT Keys: {'✓ CONFIRMED' if claim2_keys else '✗ NOT CONFIRMED'}")
    print(f"VERDICT Values: {'✓ CONFIRMED' if claim2_vals else '✗ NOT CONFIRMED'}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print_section("SUMMARY")
    
    claims = [
        ("1. Keys have channel-wise structure", claim1_pass),
        ("2a. Per-channel wins for keys", claim2_keys),
        ("2b. Per-token wins for values", claim2_vals),
    ]
    
    for name, result in claims:
        status = "✓" if result else "✗"
        print(f"  {name:<40} {status}")
    
    confirmed = sum(1 for _, r in claims if r)
    print(f"\n  {confirmed}/{len(claims)} claims confirmed on {model_name}")
    
    print("\n" + "=" * 70)
    print("  Note: Production concerns (attention sinks, three-tier architecture)")
    print("  are covered in Chapter 7.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify KV cache granularity claims")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Model to test (e.g., gpt2, TinyLlama/TinyLlama-1.1B-Chat-v1.0)")
    args = parser.parse_args()
    run_verification(args.model)
