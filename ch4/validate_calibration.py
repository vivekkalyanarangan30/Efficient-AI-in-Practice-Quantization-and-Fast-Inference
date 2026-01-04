"""
Chapter 4: Calibration Validation
=================================

This script validates calibration coverage against production activation
statistics. It implements the feedback loop described in Section 4.2.6:

1. Deploy at full precision first (collect production stats)
2. Compare calibration coverage against production observations
3. Identify gaps where production exceeds calibration
4. Recommend actions to fix coverage gaps

Outputs:
- Validation report showing per-layer coverage
- Recommendations for improving calibration

Usage:
    python ch4/validate_calibration.py --calib-stats calib.json --prod-stats prod.json
    python ch4/validate_calibration.py --demo  # Run demo with example data
"""

import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LayerValidation:
    """Validation result for a single layer."""
    layer: str
    calibration_max: float
    production_max: float
    coverage_ratio: float
    status: str
    gap_pct: Optional[float] = None


@dataclass  
class ValidationReport:
    """Full validation report."""
    passed: bool
    layers: Dict[str, LayerValidation]
    recommendations: List[str]
    summary: Dict


def validate_calibration_coverage(
    calibration_stats: Dict,
    production_stats: Dict,
    margin: float = 1.0,
    critical_margin: float = 0.9
) -> ValidationReport:
    """
    Compare calibration statistics against production observations.
    
    Args:
        calibration_stats: Dict of layer -> {abs_max, p99, mean, ...}
        production_stats: Same format from production data
        margin: Ratio at which coverage is considered OK (1.0 = exact match)
        critical_margin: Ratio below which coverage is critical (0.9 = 10% gap)
    
    Returns:
        ValidationReport with detailed analysis
    """
    layers = {}
    recommendations = []
    all_passed = True
    
    # Track overall statistics
    n_ok = 0
    n_warning = 0
    n_critical = 0
    max_gap = 0.0
    
    for layer in calibration_stats.keys():
        if layer not in production_stats:
            continue
        
        calib_max = calibration_stats[layer].get('abs_max', 0)
        prod_max = production_stats[layer].get('abs_max', 0)
        
        if prod_max == 0:
            coverage = float('inf')
            gap_pct = None
            status = 'OK'
        else:
            coverage = calib_max / prod_max
            gap_pct = (1 - coverage) * 100 if coverage < 1 else None
            
            if coverage >= margin:
                status = 'OK'
                n_ok += 1
            elif coverage >= critical_margin:
                status = 'WARNING'
                n_warning += 1
                all_passed = False
            else:
                status = 'CRITICAL'
                n_critical += 1
                all_passed = False
                
                if gap_pct and gap_pct > max_gap:
                    max_gap = gap_pct
        
        layers[layer] = LayerValidation(
            layer=layer,
            calibration_max=calib_max,
            production_max=prod_max,
            coverage_ratio=coverage,
            status=status,
            gap_pct=gap_pct
        )
        
        # Generate recommendations for gaps
        if status in ['WARNING', 'CRITICAL']:
            severity = 'significantly ' if status == 'CRITICAL' else ''
            recommendations.append(
                f"{layer}: Production max ({prod_max:.1f}) {severity}exceeds "
                f"calibration ({calib_max:.1f}) by {gap_pct:.1f}%. "
                f"Add samples with higher activation magnitudes."
            )
    
    # Summary statistics
    summary = {
        'total_layers': len(layers),
        'ok': n_ok,
        'warning': n_warning,
        'critical': n_critical,
        'max_gap_pct': max_gap,
        'overall_status': 'PASS' if all_passed else 'FAIL'
    }
    
    return ValidationReport(
        passed=all_passed,
        layers=layers,
        recommendations=recommendations,
        summary=summary
    )


def print_validation_report(report: ValidationReport):
    """Print formatted validation report."""
    
    status_icon = '✓' if report.passed else '✗'
    status_color = 'PASS' if report.passed else 'FAIL'
    
    print("\nCalibration Validation Report:")
    print("=" * 74)
    print(f"Overall Status: {status_color} {status_icon}")
    print(f"Layers: {report.summary['ok']} OK, "
          f"{report.summary['warning']} Warning, "
          f"{report.summary['critical']} Critical")
    
    if report.summary['max_gap_pct'] > 0:
        print(f"Maximum Gap: {report.summary['max_gap_pct']:.1f}%")
    
    print("\nPer-Layer Coverage:")
    print("-" * 74)
    print(f"{'Layer':<12} | {'Calib Max':>10} | {'Prod Max':>10} | "
          f"{'Coverage':>10} | {'Status'}")
    print("-" * 74)
    
    # Sort by coverage ratio (worst first)
    sorted_layers = sorted(
        report.layers.values(),
        key=lambda x: x.coverage_ratio
    )
    
    for layer_result in sorted_layers:
        status_icon = {
            'OK': '✓',
            'WARNING': '⚠',
            'CRITICAL': '✗'
        }.get(layer_result.status, '?')
        
        coverage_str = f"{layer_result.coverage_ratio:.1%}"
        
        print(f"{layer_result.layer:<12} | "
              f"{layer_result.calibration_max:>10.1f} | "
              f"{layer_result.production_max:>10.1f} | "
              f"{coverage_str:>10} | "
              f"{layer_result.status} {status_icon}")
    
    if report.recommendations:
        print("\n" + "-" * 74)
        print("Recommendations:")
        print("-" * 74)
        for rec in report.recommendations:
            print(f"  • {rec}")
    
    print("\n" + "=" * 74)


def print_workflow_guidance():
    """Print the recommended validation workflow."""
    
    print("""
Production Validation Workflow
==============================

Step 1: Deploy at Full Precision First
--------------------------------------
Your initial production deployment should run in FP16/FP32 with activation
instrumentation enabled. This establishes ground truth for production patterns.

    # Example instrumentation hook
    def collect_production_stats(model):
        stats = {}
        def hook_fn(name):
            def hook(module, input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if name not in stats:
                    stats[name] = {'abs_max': 0, 'count': 0}
                stats[name]['abs_max'] = max(
                    stats[name]['abs_max'], 
                    tensor.abs().max().item()
                )
                stats[name]['count'] += 1
            return hook
        # Register hooks on key layers...
        return stats

Step 2: Collect Production Statistics
-------------------------------------
Run for sufficient time to capture temporal patterns:
- Minimum: Full day/week cycle
- Recommended: Cover peak traffic periods
- Ideal: Seasonal variation if relevant

Save statistics periodically:
    
    with open('production_stats.json', 'w') as f:
        json.dump(stats, f)

Step 3: Compare Against Calibration
-----------------------------------
Run this validation script:

    python ch4/validate_calibration.py \\
        --calib-stats calibration_stats.json \\
        --prod-stats production_stats.json

Step 4: Iterate Until Coverage Passes
-------------------------------------
For each gap identified:
1. Analyze what input patterns cause high activations
2. Add similar samples to calibration set
3. Re-run calibration
4. Validate again

Step 5: Periodic Revalidation
-----------------------------
Production traffic evolves. Schedule periodic revalidation:
- After major feature changes
- Quarterly traffic reviews
- When accuracy metrics drift
""")


def run_demo():
    """Run demo with example calibration and production stats."""
    
    print("Running demo with example statistics...")
    print("(In practice, these would come from your instrumentation)")
    
    # Example calibration statistics (from calibration run)
    calibration_stats = {
        'layer_0': {'abs_max': 25.4, 'p99': 18.2, 'mean': 4.5},
        'layer_3': {'abs_max': 32.1, 'p99': 24.8, 'mean': 6.2},
        'layer_6': {'abs_max': 42.8, 'p99': 35.1, 'mean': 8.4},
        'layer_9': {'abs_max': 38.2, 'p99': 30.5, 'mean': 7.1},
        'layer_11': {'abs_max': 38.5, 'p99': 31.2, 'mean': 6.8}
    }
    
    # Example production statistics (collected from live traffic)
    # Note: layer_6 shows a significant gap - production exceeds calibration
    production_stats = {
        'layer_0': {'abs_max': 28.1, 'p99': 19.4, 'mean': 4.8},
        'layer_3': {'abs_max': 31.5, 'p99': 25.2, 'mean': 6.5},
        'layer_6': {'abs_max': 51.3, 'p99': 38.7, 'mean': 9.2},  # Gap!
        'layer_9': {'abs_max': 36.8, 'p99': 29.1, 'mean': 6.9},
        'layer_11': {'abs_max': 36.2, 'p99': 29.8, 'mean': 6.5}
    }
    
    # Run validation
    report = validate_calibration_coverage(calibration_stats, production_stats)
    print_validation_report(report)
    
    # Show what the fix would look like
    print("\nAfter adding high-activation samples to calibration:")
    print("-" * 74)
    
    # Fixed calibration stats
    fixed_calibration_stats = calibration_stats.copy()
    fixed_calibration_stats['layer_0'] = {'abs_max': 30.2, 'p99': 20.1, 'mean': 4.9}
    fixed_calibration_stats['layer_6'] = {'abs_max': 55.0, 'p99': 42.0, 'mean': 9.5}
    
    fixed_report = validate_calibration_coverage(fixed_calibration_stats, production_stats)
    print_validation_report(fixed_report)


def main():
    parser = argparse.ArgumentParser(
        description='Validate calibration coverage against production',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run demo with example data
    python ch4/validate_calibration.py --demo
    
    # Validate with your statistics files
    python ch4/validate_calibration.py \\
        --calib-stats calibration_stats.json \\
        --prod-stats production_stats.json
    
    # Show workflow guidance
    python ch4/validate_calibration.py --workflow
"""
    )
    parser.add_argument('--calib-stats', type=str,
                       help='Path to calibration statistics JSON')
    parser.add_argument('--prod-stats', type=str,
                       help='Path to production statistics JSON')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with example data')
    parser.add_argument('--workflow', action='store_true',
                       help='Show recommended validation workflow')
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Coverage ratio for OK status (default: 1.0)')
    args = parser.parse_args()
    
    if args.workflow:
        print_workflow_guidance()
        return
    
    if args.demo:
        run_demo()
        return
    
    if args.calib_stats and args.prod_stats:
        # Load statistics from files
        with open(args.calib_stats) as f:
            calibration_stats = json.load(f)
        with open(args.prod_stats) as f:
            production_stats = json.load(f)
        
        report = validate_calibration_coverage(
            calibration_stats, 
            production_stats,
            margin=args.margin
        )
        print_validation_report(report)
    else:
        print("Usage: Provide --demo, --workflow, or both --calib-stats and --prod-stats")
        print("Run with --help for more information")


if __name__ == '__main__':
    main()