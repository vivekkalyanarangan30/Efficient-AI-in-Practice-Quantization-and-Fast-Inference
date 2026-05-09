"""
Chapter 10, Section 10.1 — Cost-per-million-tokens curve
Companion script for "Efficient AI in Practice: Quantization and Fast Inference"

Mode:
  --mode figure   Render Figure 10.1 (cost curve) as PNG + PDF.

This script does not measure throughput. It plots the cost-per-million-tokens
curve from already-measured TPS values and on-demand cloud prices. Every
input value is sourced explicitly in the Config docstring below; rerunning
the script with different prices/TPS produces a different curve, but the
shape and the regime story is robust to ±25% on either input.

Usage:
  python ch10_cost_curve.py --mode figure

Requires: matplotlib >= 3.8 (Arial fallback to DejaVu Sans on Linux).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
FIG_DIR = SCRIPT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)


# ─── Config ──────────────────────────────────────────────────────────────────
@dataclass
class Config:
    """All inputs to Figure 10.1.

    Pricing: Google Cloud Compute Engine on-demand list rates in `us-central1`,
    pulled from cloud.google.com/compute/vm-instance-pricing at the time of
    writing. Spot, sustained-use, and committed-use discounts move the curves
    proportionally without changing crossover regimes.

    TPS values:
      - c3-standard-22 measurement: §10.5's freshly-converted Q4_K_M GGUF
        landed at 19.94 tok/s (control); the community pre-quantized file
        landed at 21.02 tok/s. The chapter uses 20 tok/s as a representative
        point on a per-VM band of roughly 17–21 tok/s under noisy-neighbor
        variance. See §10.5.
      - g2-standard-4 (1× L4): bandwidth-derived. L4 has 300 GB/s memory
        bandwidth; Q4_K_M Llama-2-7B is ~3.9 GB. At ~60% of bandwidth ceiling
        (the same fraction RTX 3070 mobile hit on Llama-2-7B Q4_0 in
        published llama.cpp benchmarks), single-stream decode lands ~50 tok/s.
        DECA (MICRO 2025) confirms decoding on bandwidth-bound silicon
        recovers 60–85% of theoretical ceiling depending on kernel quality.
    """

    # Pricing (USD/hour, on-demand list, us-central1)
    cpu_hourly_usd: float = 1.1088    # c3-standard-22 (22 vCPU Sapphire Rapids)
    gpu_hourly_usd: float = 0.7068    # g2-standard-4 (4 vCPU + 1× NVIDIA L4)
    cpu_label: str = "c3-standard-22 (22 vCPU Sapphire Rapids)"
    gpu_label: str = "g2-standard-4 (1× NVIDIA L4)"

    # Single-stream decode rate (tok/s) for Llama-2-7B Q4_K_M
    cpu_tps_max: float = 20.0   # c3-standard-22 measurement, §10.5
    gpu_tps_max: float = 50.0   # L4 bandwidth-derived, see docstring

    # Demand sweep (tok/s)
    demand_min: float = 1.0
    demand_max: float = 200.0
    demand_n: int = 600

    # Replica caps (max replicas to plot before truncating)
    cpu_replica_cap: int = 10
    gpu_replica_cap: int = 4

    # Figure dimensions (inches; Manning max width 5.6")
    fig_width: float = 5.4
    fig_height: float = 3.6
    dpi: int = 300

    # Identity
    figure_id: str = "CH10_F01_Kalyanarangan"


CFG = Config()


# ─── Manning style ───────────────────────────────────────────────────────────
def apply_manning_style() -> None:
    """Arial 8pt, fonttype 42, gridded prose-friendly axes."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "lines.linewidth": 1.4,
        "patch.linewidth": 0.6,
        "savefig.dpi": CFG.dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Grayscale-safe palette: distinct in print, distinct in B&W
COLOR_CPU = "#5a5a5a"
COLOR_GPU = "#1a1a1a"
COLOR_ANNOT = "#3a3a3a"


# ─── Cost computation ────────────────────────────────────────────────────────
def replicas_needed(demand_tps: np.ndarray, per_replica_max: float) -> np.ndarray:
    """Minimum integer replicas to absorb sustained demand."""
    return np.ceil(demand_tps / per_replica_max).astype(int)


def cost_per_million(
    demand_tps: np.ndarray,
    hourly_usd: float,
    per_replica_max_tps: float,
) -> np.ndarray:
    """Effective $/M tokens at sustained demand, paying for whole replicas.

    The cost per second equals (replicas × hourly / 3600). The throughput is
    the demand itself (replicas can absorb it; idle headroom does not save
    money). $/M tokens = cost-per-second × 1e6 / demand.
    """
    n = replicas_needed(demand_tps, per_replica_max_tps)
    return n * hourly_usd * 1e6 / (3600.0 * demand_tps)


def saturation_demands(per_replica_max: float, n_max: int) -> List[float]:
    """Demand levels where the curve steps up (replica boundaries)."""
    return [per_replica_max * k for k in range(1, n_max + 1)]


# ─── Figure ──────────────────────────────────────────────────────────────────
def render_cost_curve(cfg: Config = CFG) -> Tuple[List[Path], str]:
    """Render Figure 10.1 as PNG + PDF; return (paths, caption)."""
    apply_manning_style()

    # Demand sweep on a log axis
    d = np.geomspace(cfg.demand_min, cfg.demand_max, cfg.demand_n)

    cpu = cost_per_million(d, cfg.cpu_hourly_usd, cfg.cpu_tps_max)
    gpu = cost_per_million(d, cfg.gpu_hourly_usd, cfg.gpu_tps_max)

    fig, ax = plt.subplots(figsize=(cfg.fig_width, cfg.fig_height))

    ax.plot(d, cpu, color=COLOR_CPU, linestyle="-", linewidth=1.6,
            label=cfg.cpu_label)
    ax.plot(d, gpu, color=COLOR_GPU, linestyle="--", linewidth=1.6,
            label=cfg.gpu_label)

    # Replica-boundary tick marks on each curve (saturation events)
    for x in saturation_demands(cfg.cpu_tps_max, cfg.cpu_replica_cap):
        if x > cfg.demand_max:
            break
        y = cost_per_million(np.array([x]), cfg.cpu_hourly_usd,
                             cfg.cpu_tps_max)[0]
        ax.plot([x], [y], marker="|", color=COLOR_CPU, markersize=5,
                markeredgewidth=0.9)

    for x in saturation_demands(cfg.gpu_tps_max, cfg.gpu_replica_cap):
        if x > cfg.demand_max:
            break
        y = cost_per_million(np.array([x]), cfg.gpu_hourly_usd,
                             cfg.gpu_tps_max)[0]
        ax.plot([x], [y], marker="|", color=COLOR_GPU, markersize=5,
                markeredgewidth=0.9)

    # Annotation 1 — the single-replica regime ratio (price ratio)
    ratio_low = cfg.cpu_hourly_usd / cfg.gpu_hourly_usd
    ax.annotate(
        f"Below saturation: ratio fixed at\n"
        f"{cfg.cpu_hourly_usd:.2f}/{cfg.gpu_hourly_usd:.2f} = "
        f"{ratio_low:.2f}× (price ratio)",
        xy=(3.0, cost_per_million(np.array([3.0]),
                                  cfg.cpu_hourly_usd,
                                  cfg.cpu_tps_max)[0]),
        xytext=(2.0, 6.0),
        fontsize=6.5, color=COLOR_ANNOT,
        arrowprops=dict(arrowstyle="-", color=COLOR_ANNOT,
                        linewidth=0.6, shrinkA=2, shrinkB=2),
    )

    # Annotation 2 — the saturation regime ratio
    cpu_sat_cost = cfg.cpu_hourly_usd * 1e6 / (3600 * cfg.cpu_tps_max)
    gpu_sat_cost = cfg.gpu_hourly_usd * 1e6 / (3600 * cfg.gpu_tps_max)
    ratio_sat = cpu_sat_cost / gpu_sat_cost
    ax.annotate(
        f"At GPU saturation:\n${gpu_sat_cost:.2f}/M (GPU) vs\n"
        f"${cpu_sat_cost:.2f}/M (CPU at GPU's TPS) → {ratio_sat:.1f}×",
        xy=(cfg.gpu_tps_max, gpu_sat_cost),
        xytext=(70, 1.6),
        fontsize=6.5, color=COLOR_ANNOT,
        arrowprops=dict(arrowstyle="-", color=COLOR_ANNOT,
                        linewidth=0.6, shrinkA=2, shrinkB=2),
    )

    # Saturation reference lines (dotted) on both axes
    ax.axvline(cfg.cpu_tps_max, color=COLOR_CPU, linestyle=":",
               linewidth=0.6, alpha=0.6)
    ax.axvline(cfg.gpu_tps_max, color=COLOR_GPU, linestyle=":",
               linewidth=0.6, alpha=0.6)
    ax.text(cfg.cpu_tps_max, cfg.demand_max * 0.005, "  CPU sat",
            fontsize=6, color=COLOR_CPU,
            rotation=90, va="bottom", ha="left")
    ax.text(cfg.gpu_tps_max, cfg.demand_max * 0.005, "  GPU sat",
            fontsize=6, color=COLOR_GPU,
            rotation=90, va="bottom", ha="left")

    # Axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(cfg.demand_min, cfg.demand_max)
    ax.set_ylim(0.8, 400)

    ax.set_xlabel("Sustained demand (tok/s)")
    ax.set_ylabel("Effective cost per million tokens (USD)")

    # Tick formatting — readable powers of 10
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v)}" if v >= 1 else f"{v:g}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${int(v)}" if v >= 1 else f"${v:g}"))

    ax.grid(True, which="both", linestyle="-", alpha=0.25)
    ax.set_axisbelow(True)

    # Legend
    leg = ax.legend(loc="upper right", frameon=True, framealpha=0.9,
                    edgecolor="#bfbfbf")
    leg.get_frame().set_linewidth(0.5)

    plt.tight_layout()

    paths = _save_pair(fig, cfg.figure_id)
    plt.close(fig)

    caption = (
        "Effective cost per million Llama-2-7B Q4_K_M tokens versus "
        "sustained demand, on Google Cloud `c3-standard-22` (CPU-only, "
        "solid line) and `g2-standard-4` with one NVIDIA L4 (dashed line) "
        "in `us-central1` at on-demand list pricing. Each curve declines "
        "hyperbolically until the instance's single-stream throughput "
        "ceiling (CPU at 20 tok/s, GPU at 50 tok/s; tick marks at "
        "successive replica boundaries), then steps up as additional "
        "replicas absorb demand above ceiling. Below saturation the "
        "ratio between curves equals the ratio of hourly prices "
        "(1.57×). At the GPU's saturation the ratio is roughly 3.9×, "
        "and it widens further at higher demand because the CPU instance "
        "saturates at 2.5× lower throughput. The L4 wins on cost per "
        "token at every demand point measured; the gap is bounded "
        "between 1.57× and ~5×."
    )
    return paths, caption


# ─── I/O helpers ─────────────────────────────────────────────────────────────
def _save_pair(fig, stem: str) -> List[Path]:
    """Save matplotlib figure as PNG + PDF using stem as filename basis."""
    png_path = FIG_DIR / f"{stem}.png"
    pdf_path = FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=CFG.dpi)
    fig.savefig(pdf_path)
    return [png_path, pdf_path]


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["figure"],
        default="figure",
        help="What to render. Currently only 'figure' is supported.",
    )
    args = parser.parse_args()

    if args.mode == "figure":
        paths, caption = render_cost_curve()
        print(f"Wrote {len(paths)} files to {FIG_DIR}")
        for p in paths:
            print(f"  {p.relative_to(SCRIPT_DIR)}")
        print()
        print("Suggested caption:")
        print(caption)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())