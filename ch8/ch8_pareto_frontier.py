"""
Figure 8.21 — Sub-8-bit Pareto frontier on OPT-6.7B (WikiText-2)

Generates CH08_F21_Kalyanarangan.{png,pdf} from measured data points
collected across Sections 8.2, 8.3, and 8.4. All perplexity values and
memory sizes are copied verbatim from the chapter's figures and tables.

Data sources:
  - BF16 baseline:         §8.2.4, Figure 8.6         (16.26, 12.40 GB)
  - FP8 E4M3:              §8.2.4, Figure 8.6         (16.26,  6.21 GB)
  - FP4 per-tensor:        §8.3.4, Table 8.4          (26.09,  3.11 GB)
  - FP4 block-16 (FP32):   §8.3.4, Table 8.4          (16.49,  4.66 GB)
  - FP4 block-32 (FP32):   §8.3.4, Table 8.4          (16.54,  3.89 GB)
  - NF4 block-64:          §8.5.1 explicit reference  (16.49,  3.77 GB est.)

NF4 size note: §8.5.1 states "NF4 block-64 achieves 16.49 on the same model"
(OPT-6.7B). Size is computed from §8.4's 4.5 effective bits per weight after
double quantization: 6.7e9 * 4.5 / 8 / 1e9 = 3.77 GB. This matches the
"~3.5 GB for a 7B model" figure cited in §8.1 and §8.5.

Ternary configurations are omitted by design: OPT-6.7B has no trained ternary
checkpoint, and ternary PTQ (§8.5.1) produces perplexities in the 13,000-38,000
range that would fall off the plot. BitNet b1.58 is a different model and is
discussed in prose only.

Layout: Two panels. The left panel shows all points at full y-range so FP4
per-tensor's 60% perplexity cliff is visible. The right panel zooms into the
frontier cluster (16.20-16.65) where the surviving formats sit, revealing the
BF16 -> FP8 -> NF4/FP4-b32 step structure that is invisible at full scale.

Usage:
    python ch8_pareto_frontier.py              # PNG + PDF alongside script
    python ch8_pareto_frontier.py --show       # interactive display
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter


# ---------------------------------------------------------------------------
# Manning figure conventions
# ---------------------------------------------------------------------------

MANNING_COLORS = {
    "on_frontier": "#1F4E79",     # deep blue, dominant points
    "dominated":   "#7F7F7F",     # grey, configurations strictly dominated
    "broken":      "#C0392B",     # red, functionally broken (FP4 per-tensor)
    "frontier":    "#1F4E79",     # frontier line
    "axis":        "#333333",
    "inset_edge":  "#7F7F7F",
}

MANNING_DPI = 300
MANNING_MAX_WIDTH_INCHES = 5.6

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": MANNING_COLORS["axis"],
    "axes.labelcolor": MANNING_COLORS["axis"],
    "xtick.color": MANNING_COLORS["axis"],
    "ytick.color": MANNING_COLORS["axis"],
})


# ---------------------------------------------------------------------------
# Data — every number traceable to a chapter figure or table
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Point:
    label: str
    size_gb: float          # model storage in GB
    perplexity: float       # WikiText-2, 64 seq × 512 tokens
    category: str           # "frontier", "dominated", "broken"
    main_offset: tuple[float, float]
    inset_offset: tuple[float, float] | None   # None = not labeled in inset
    source: str


POINTS: list[Point] = [
    Point(
        label="BF16",
        size_gb=12.40, perplexity=16.26,
        category="frontier",
        main_offset=(0, 10),
        inset_offset=(-6, 10),
        source="§8.2.4 Fig 8.6",
    ),
    Point(
        label="FP8 E4M3",
        size_gb=6.21, perplexity=16.26,
        category="frontier",
        main_offset=(8, 0),
        inset_offset=(8, 8),
        source="§8.2.4 Fig 8.6",
    ),
    Point(
        label="NF4 block-64",
        size_gb=3.77, perplexity=16.49,
        category="frontier",
        main_offset=(-8, -14),
        inset_offset=(10, -22),
        source="§8.5.1 + §8.4 (size est.)",
    ),
    Point(
        label="FP4 block-32",
        size_gb=3.89, perplexity=16.54,
        category="frontier",
        main_offset=(8, -14),
        inset_offset=(8, 8),
        source="§8.3.4 Table 8.4",
    ),
    Point(
        label="FP4 block-16\n(dominated)",
        size_gb=4.66, perplexity=16.49,
        category="dominated",
        main_offset=(8, 0),
        inset_offset=(12, -4),
        source="§8.3.4 Table 8.4",
    ),
    Point(
        label="FP4 per-tensor",
        size_gb=3.11, perplexity=26.09,
        category="broken",
        main_offset=(10, 2),
        inset_offset=None,
        source="§8.3.4 Table 8.4",
    ),
]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CATEGORY_STYLE = {
    "frontier":  dict(color=MANNING_COLORS["on_frontier"], marker="o",
                      s=55, edgecolor="white", linewidth=1.2, zorder=5),
    "dominated": dict(color=MANNING_COLORS["dominated"], marker="s",
                      s=45, edgecolor="white", linewidth=1.2, zorder=4),
    "broken":    dict(color=MANNING_COLORS["broken"], marker="X",
                      s=65, edgecolor="white", linewidth=1.2, zorder=5),
}

CATEGORY_LABEL = {
    "frontier":  "On frontier",
    "dominated": "Dominated",
    "broken":    "Functionally broken",
}


def draw_frontier(ax: plt.Axes) -> None:
    """Connect the frontier points in size order."""
    frontier_pts = sorted(
        [p for p in POINTS if p.category == "frontier"],
        key=lambda p: p.size_gb,
    )
    ax.plot(
        [p.size_gb for p in frontier_pts],
        [p.perplexity for p in frontier_pts],
        color=MANNING_COLORS["frontier"],
        linewidth=1.2,
        linestyle="--",
        alpha=0.55,
        zorder=2,
        label="Pareto frontier",
    )


def draw_points(ax: plt.Axes, with_legend: bool = True) -> None:
    for category, style in CATEGORY_STYLE.items():
        pts = [p for p in POINTS if p.category == category]
        if not pts:
            continue
        ax.scatter(
            [p.size_gb for p in pts],
            [p.perplexity for p in pts],
            label=CATEGORY_LABEL[category] if with_legend else None,
            **style,
        )


def annotate_main(ax: plt.Axes) -> None:
    """Label only the broken outlier on the main panel; the cluster is left
    unlabeled and directs the reader to the inset."""
    outlier = next(p for p in POINTS if p.category == "broken")
    dx, dy = outlier.main_offset
    ax.annotate(
        outlier.label,
        xy=(outlier.size_gb, outlier.perplexity),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8,
        color=MANNING_COLORS["broken"],
        ha="left",
    )
    # Callout pointing to the frontier cluster
    ax.annotate(
        "frontier cluster\n(see zoom, right)",
        xy=(6.5, 16.45),
        xytext=(10, 40),
        textcoords="offset points",
        fontsize=7.5,
        color=MANNING_COLORS["axis"],
        style="italic",
        ha="left",
        arrowprops=dict(
            arrowstyle="-",
            color=MANNING_COLORS["axis"],
            alpha=0.5,
            linewidth=0.7,
        ),
    )


def annotate_inset(ax: plt.Axes) -> None:
    """Labels for all in-range points in the zoom panel."""
    for p in POINTS:
        if p.inset_offset is None:
            continue
        dx, dy = p.inset_offset
        color = (
            MANNING_COLORS["dominated"]
            if p.category == "dominated"
            else MANNING_COLORS["axis"]
        )
        ax.annotate(
            p.label,
            xy=(p.size_gb, p.perplexity),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            ha="left" if dx >= 0 else "right",
        )


def draw_dominance_arrow(ax: plt.Axes) -> None:
    """Dotted arrow from FP4 block-16 to NF4 block-64: same quality, less memory."""
    fp4_b16 = next(p for p in POINTS if p.label.startswith("FP4 block-16"))
    nf4 = next(p for p in POINTS if p.label == "NF4 block-64")
    ax.annotate(
        "",
        xy=(nf4.size_gb, nf4.perplexity),
        xytext=(fp4_b16.size_gb, fp4_b16.perplexity),
        arrowprops=dict(
            arrowstyle="->",
            color=MANNING_COLORS["dominated"],
            linewidth=0.9,
            linestyle=(0, (2, 2)),
            alpha=0.75,
            shrinkA=8, shrinkB=8,
        ),
        zorder=3,
    )


def configure_main_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("Model size (GB)")
    ax.set_ylabel("Perplexity (log scale, lower is better)")
    ax.set_yscale("log")
    ax.set_xlim(2.0, 14.0)
    ax.set_ylim(15.9, 30.0)

    ax.yaxis.set_major_locator(FixedLocator([16, 18, 20, 25, 30]))
    ax.yaxis.set_minor_locator(NullLocator())
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)

    ax.grid(True, which="major", linestyle=":", alpha=0.3, zorder=1)
    ax.set_axisbelow(True)
    ax.set_title("All configurations", fontsize=9, pad=6)


def configure_inset_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("Model size (GB)", fontsize=8)
    ax.set_ylabel("Perplexity (linear)", fontsize=8)
    ax.set_xlim(3.0, 13.8)
    ax.set_ylim(16.05, 16.75)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_locator(FixedLocator([16.20, 16.35, 16.50, 16.65]))
    ax.grid(True, which="major", linestyle=":", alpha=0.3, zorder=1)
    ax.set_axisbelow(True)
    ax.set_title("Frontier cluster (zoom)", fontsize=9, pad=6)

    for spine in ax.spines.values():
        spine.set_edgecolor(MANNING_COLORS["inset_edge"])


def build_figure() -> plt.Figure:
    fig, (ax_main, ax_inset) = plt.subplots(
        1, 2,
        figsize=(MANNING_MAX_WIDTH_INCHES, 3.2),
        dpi=MANNING_DPI,
        gridspec_kw={"width_ratios": [1.0, 1.2], "wspace": 0.35},
    )

    draw_frontier(ax_main)
    draw_points(ax_main, with_legend=True)
    annotate_main(ax_main)
    configure_main_axes(ax_main)
    ax_main.legend(
        loc="center right",
        frameon=True,
        framealpha=0.95,
        edgecolor=MANNING_COLORS["axis"],
        fontsize=7,
    )

    draw_frontier(ax_inset)
    draw_points(ax_inset, with_legend=False)
    annotate_inset(ax_inset)
    configure_inset_axes(ax_inset)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save / show
# ---------------------------------------------------------------------------

def save_or_show(fig: plt.Figure, show: bool) -> None:
    if show:
        plt.show()
        return

    out_dir = Path(__file__).resolve().parent
    stem = "CH08_F21_Kalyanarangan"
    fig.savefig(out_dir / f"{stem}.png", dpi=MANNING_DPI, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    print(f"Wrote {out_dir / stem}.png")
    print(f"Wrote {out_dir / stem}.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true",
                        help="Display interactively instead of writing files")
    args = parser.parse_args()

    fig = build_figure()
    save_or_show(fig, show=args.show)


if __name__ == "__main__":
    main()