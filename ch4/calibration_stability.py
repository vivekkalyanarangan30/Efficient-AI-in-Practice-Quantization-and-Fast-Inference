"""
Chapter 4: Calibration Stability Analysis
=========================================

This script demonstrates how scale factor estimates stabilize as calibration
sample size increases. The key insight: you need enough samples to discover
the true activation range, but beyond ~128-512 diverse samples, adding more
rarely changes the estimated scale factors.

Approach:
- Generate a large pool of diverse calibration samples
- Incrementally add samples and track how the estimated scale factor changes
- Show that the estimate converges and stabilizes

Outputs:
- Console table showing scale factor convergence
- Visualization of convergence curve (optional)

Usage:
    python ch4/calibration_stability.py
    python ch4/calibration_stability.py --save-plot
"""

import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


def generate_diverse_texts(n_texts: int = 1000) -> List[str]:
    """
    Generate a large pool of diverse calibration texts.
    
    We create variety across:
    - Length (short to long)
    - Style (factual, casual, technical, code)
    - Special characters and formatting
    """
    texts = []
    
    # Short factual (low activation potential)
    short_factual = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What year did WWII end?",
        "How many continents exist?",
        "What is photosynthesis?",
        "Who painted the Mona Lisa?",
        "What is the speed of light?",
        "When was the internet invented?",
        "What causes earthquakes?",
        "Who discovered gravity?",
    ] * 20
    
    # Technical/formal (medium activation potential)
    technical = [
        "The eigenvalue decomposition of a symmetric matrix yields orthogonal eigenvectors with real eigenvalues.",
        "Gradient descent optimization converges to local minima under appropriate learning rate schedules.",
        "The convolutional neural network applies learned filters across spatial dimensions of the input tensor.",
        "Backpropagation computes gradients efficiently via the chain rule of calculus.",
        "Transformer architectures employ multi-head self-attention mechanisms for sequence modeling.",
        "The softmax function normalizes logits into a valid probability distribution.",
        "Batch normalization stabilizes training by normalizing intermediate activations.",
        "Residual connections enable gradient flow in very deep neural networks.",
        "The attention mechanism computes weighted sums based on query-key compatibility.",
        "Layer normalization applies normalization across feature dimensions independently per sample.",
    ] * 15
    
    # Long context (higher activation potential due to length)
    long_context = [
        """The following is a comprehensive analysis of machine learning optimization techniques. 
        Stochastic gradient descent remains the workhorse of deep learning, though numerous variants 
        have emerged including Adam, AdaGrad, and RMSprop. Each optimizer makes different assumptions 
        about the loss landscape geometry and adapts learning rates accordingly. The choice of optimizer 
        can significantly impact both convergence speed and final model quality.""",
        
        """Consider the implications of quantization on neural network inference. When we reduce 
        weight precision from 32-bit floating point to 8-bit integers, we introduce quantization 
        error that propagates through the network. The magnitude of this error depends critically 
        on how we choose the scale factors that map floating point ranges to integer ranges.""",
        
        """The transformer architecture has revolutionized natural language processing since its 
        introduction in 2017. Key innovations include positional encodings to inject sequence order 
        information, multi-head attention to capture different types of relationships, and layer 
        normalization for training stability. These components work together to enable effective 
        modeling of long-range dependencies in sequential data.""",
    ] * 30
    
    # Code snippets (often high activation due to special tokens)
    code = [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    return quicksort([x for x in arr if x < pivot]) + [pivot] + quicksort([x for x in arr if x > pivot])",
        "SELECT users.name, COUNT(orders.id) as order_count\nFROM users\nLEFT JOIN orders ON users.id = orders.user_id\nGROUP BY users.id\nHAVING order_count > 5;",
        "import torch\nimport torch.nn as nn\n\nclass Attention(nn.Module):\n    def __init__(self, dim):\n        super().__init__()\n        self.qkv = nn.Linear(dim, dim * 3)",
        "async function fetchData(url) {\n  const response = await fetch(url);\n  if (!response.ok) throw new Error('Failed');\n  return response.json();\n}",
        "fn fibonacci(n: u64) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n-1) + fibonacci(n-2)\n    }\n}",
    ] * 25
    
    # Adversarial/edge cases (potentially highest activations)
    adversarial = [
        "!!!!!!!!!! URGENT ALERT !!!!!!!!!! CRITICAL WARNING !!!!!!!!!!",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "🔥🚀💯🎉✨🌟⭐️🏆🥇🎯💪🎊🎁🎈" * 5,
        "1234567890" * 20,
        "!@#$%^&*()_+-=[]{}|;':\",./<>?" * 5,
        "<<<<<<< HEAD\n=======\n>>>>>>> branch",
        "\t\t\t\t\n\n\n\n\t\t\t\t",
        "NULL; DROP TABLE users; --",
        "ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ" * 3,
        "∑∏∫∂∇∆√∛∜≈≠≤≥±∓" * 5,
    ] * 20
    
    # Combine all
    texts = short_factual + technical + long_context + code + adversarial
    
    # Shuffle to mix categories
    np.random.seed(42)
    np.random.shuffle(texts)
    
    return texts[:n_texts]


def measure_scale_factor_incremental(model, tokenizer, texts: List[str], 
                                     layer_idx: int = 6,
                                     checkpoints: List[int] = None) -> Dict:
    """
    Measure how scale factor estimates change as we add more samples.
    
    Returns dict mapping sample_count -> scale_factor
    """
    if checkpoints is None:
        checkpoints = [8, 16, 32, 64, 128, 256, 512]
    
    model.eval()
    device = next(model.parameters()).device
    
    # Storage for results
    results = {}
    running_max = 0.0
    
    # Hook to capture activations
    activation_max = [0.0]  # Use list to allow modification in nested function
    
    def hook_fn(module, input, output):
        activation_max[0] = output.abs().max().item()
    
    # Register hook on target layer
    if hasattr(model, 'encoder'):  # BERT-style
        target_layer = model.encoder.layer[layer_idx].intermediate.dense
    else:
        raise ValueError("Unknown model architecture - expected BERT-style encoder")
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    try:
        for i, text in enumerate(texts):
            # Process this sample
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True,
                max_length=256,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items() 
                     if k in ['input_ids', 'attention_mask']}
            
            with torch.no_grad():
                _ = model(**inputs)
            
            # Update running max
            running_max = max(running_max, activation_max[0])
            
            # Record at checkpoints
            sample_count = i + 1
            if sample_count in checkpoints:
                results[sample_count] = running_max
                
    finally:
        handle.remove()
    
    return results


def compute_stability_metrics(incremental_results: Dict) -> Dict:
    """
    Compute stability metrics from incremental results.
    
    Shows how much the scale factor estimate changes between checkpoints.
    """
    checkpoints = sorted(incremental_results.keys())
    
    metrics = {}
    final_value = incremental_results[checkpoints[-1]]
    
    for i, cp in enumerate(checkpoints):
        value = incremental_results[cp]
        
        # Percent of final value discovered
        pct_of_final = (value / final_value * 100) if final_value > 0 else 0
        
        # Change from previous checkpoint
        if i > 0:
            prev_value = incremental_results[checkpoints[i-1]]
            change = ((value - prev_value) / prev_value * 100) if prev_value > 0 else 0
        else:
            change = None
        
        metrics[cp] = {
            'scale_factor': value,
            'pct_of_final': pct_of_final,
            'change_from_prev': change
        }
    
    return metrics


def print_stability_table(metrics: Dict):
    """Print formatted stability analysis table."""
    
    checkpoints = sorted(metrics.keys())
    
    print("\nCalibration Scale Factor Convergence:")
    print("=" * 70)
    print(f"{'Samples':<10} | {'Scale Factor':>12} | {'% of Final':>12} | {'Δ from Prev':>12} | Status")
    print("-" * 70)
    
    for cp in checkpoints:
        m = metrics[cp]
        
        # Determine convergence status
        if m['pct_of_final'] >= 99:
            status = "Converged ✓"
        elif m['pct_of_final'] >= 95:
            status = "Nearly stable"
        elif m['pct_of_final'] >= 90:
            status = "Approaching"
        else:
            status = "Still discovering"
        
        change_str = f"{m['change_from_prev']:>+11.1f}%" if m['change_from_prev'] is not None else "        --"
        
        print(f"{cp:<10} | {m['scale_factor']:>12.2f} | {m['pct_of_final']:>11.1f}% | {change_str} | {status}")
    
    print("-" * 70)
    
    # Find convergence point (first checkpoint at 95%+ of final)
    convergence_point = None
    for cp in checkpoints:
        if metrics[cp]['pct_of_final'] >= 95:
            convergence_point = cp
            break
    
    if convergence_point:
        print(f"\nConvergence point: ~{convergence_point} samples (95% of final scale factor)")
    
    print("\nInterpretation:")
    print("  - Scale factor = max activation observed (determines quantization range)")
    print("  - '% of Final' shows how much of the true range you've discovered")
    print("  - Once at 95-99%, adding more samples rarely changes the estimate")


def print_reference_info():
    """Print reference information about calibration sizes in practice."""
    
    print("=" * 70)
    print("Industry Reference - Default Calibration Sizes:")
    print("-" * 70)
    print("  GPTQ (original paper):     128 samples")
    print("  AWQ:                       128-256 samples")
    print("  llama.cpp:                 128-512 samples")
    print("  TensorRT:                  500-1000 samples (vision)")
    print("-" * 70)
    print("Key insight: Scale factor estimation (finding max/percentile)")
    print("stabilizes quickly with diverse samples. Beyond 128-512,")
    print("adding more data rarely changes quantization parameters.")
    print("=" * 70)


def create_convergence_plot(metrics: Dict, save_path: str = None):
    """Create visualization of scale factor convergence."""
    
    checkpoints = sorted(metrics.keys())
    scale_factors = [metrics[cp]['scale_factor'] for cp in checkpoints]
    pct_of_final = [metrics[cp]['pct_of_final'] for cp in checkpoints]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Scale factor vs samples
    ax1.plot(checkpoints, scale_factors, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=scale_factors[-1], color='r', linestyle='--', alpha=0.7, label='Final value')
    ax1.set_xlabel('Number of Calibration Samples', fontsize=11)
    ax1.set_ylabel('Scale Factor (Max Activation)', fontsize=11)
    ax1.set_title('Scale Factor Convergence', fontsize=12, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Right: Percent of final discovered
    ax2.bar(range(len(checkpoints)), pct_of_final, color='steelblue', edgecolor='white')
    ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% threshold')
    ax2.axhline(y=99, color='green', linestyle='--', alpha=0.7, label='99% threshold')
    ax2.set_xlabel('Checkpoint', fontsize=11)
    ax2.set_ylabel('% of Final Scale Factor', fontsize=11)
    ax2.set_title('Range Discovery Progress', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(checkpoints)))
    ax2.set_xticklabels([str(cp) for cp in checkpoints])
    ax2.set_ylim(0, 105)
    ax2.legend(loc='lower right')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nSaved figure to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Calibration stability analysis')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                       help='Model to analyze')
    parser.add_argument('--layer', type=int, default=6,
                       help='Layer index to analyze')
    parser.add_argument('--n-samples', type=int, default=512,
                       help='Total number of calibration samples to use')
    parser.add_argument('--save-plot', action='store_true',
                       help='Save convergence plot')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip model analysis, just show reference info')
    args = parser.parse_args()
    
    print_reference_info()
    
    if args.skip_analysis:
        return
    
    print(f"\nRunning convergence analysis on {args.model}...")
    
    # Generate diverse texts
    print(f"Generating {args.n_samples} diverse calibration texts...")
    texts = generate_diverse_texts(args.n_samples)
    print(f"  Created texts with mixed: factual, technical, code, adversarial")
    
    try:
        from transformers import AutoModel, AutoTokenizer
        
        print(f"\nLoading model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)
        model.eval()
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f"Model loaded on {device}")
        
        # Define checkpoints
        checkpoints = [8, 16, 32, 64, 128, 256, 512]
        checkpoints = [cp for cp in checkpoints if cp <= args.n_samples]
        
        print(f"\nMeasuring scale factor convergence (layer {args.layer})...")
        print(f"Checkpoints: {checkpoints}")
        
        incremental_results = measure_scale_factor_incremental(
            model, tokenizer, texts,
            layer_idx=args.layer,
            checkpoints=checkpoints
        )
        
        # Compute and display metrics
        metrics = compute_stability_metrics(incremental_results)
        print_stability_table(metrics)
        
        # Optional plot
        if args.save_plot:
            create_convergence_plot(metrics, 'figures/ch4_calibration_convergence.png')
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("\nThe reference information above provides industry-standard guidance.")


if __name__ == '__main__':
    main()