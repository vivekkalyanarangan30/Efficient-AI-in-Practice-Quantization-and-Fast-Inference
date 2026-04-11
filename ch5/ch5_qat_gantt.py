#!/usr/bin/env python3
"""Chapter 5, Figure 5.10 -- QAT training-schedule gantt chart.

Replaces the original Mermaid-exported SVG/PNG for CH05_F10.  The Mermaid
gantt exported a ~1900-pixel-wide bar chart with 11px labels; once fit to
Manning's 5.6-inch column, that text shrank to ~3pt and was unreadable in
print.  This script lays out the same four-lane training schedule directly
at Manning dimensions with 7-8pt Arial and clean matplotlib bars.

Run:
    python ch5_qat_gantt.py
Outputs (in ch5/figures/):
    CH05_F10_Kalyanarangan.png
    CH05_F10_Kalyanarangan.pdf
    CH05_F10_Kalyanarangan.svg
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Manning layout constants
# ---------------------------------------------------------------------------
FIGURE_DIR = Path(__file__).parent / "figures"
FIG_BASE = "CH05_F10_Kalyanarangan"

FIG_W = 5.6    # inches (Manning column width)
FIG_H = 3.00   # inches

TITLE_FONT_SIZE = 9.5
SECTION_FONT_SIZE = 8
BAR_FONT_SIZE = 7
TICK_FONT_SIZE = 7
AXIS_LABEL_FONT_SIZE = 8

EDGE_COLOR = "#28253D"
TEXT_COLOR = "#28253D"
GRID_COLOR = "#E6E6EC"
LANE_BG = "#F6F7FB"

# Bar fill palette: subtle tinted fills, grayscale-safe, matches
# ch5_decision_tree.py.
BAR_FILLS = {
    "blue":   "#DCE6F5",
    "yellow": "#FFF2C6",
    "purple": "#E8E0F4",
    "green":  "#D6EDC8",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class Phase:
    label: str
    start: float
    end: float
    color: str


TOTAL_EPOCHS = 20

# Row order is top-to-bottom on the chart.
ROWS: List[Tuple[str, List[Phase]]] = [
    ("Bit-Width", [
        Phase("INT8 Warm-up",      0.0,  5.0, "blue"),
        Phase("INT6 Intermediate", 5.0, 10.0, "yellow"),
        Phase("INT4 Target",      10.0, 20.0, "green"),
    ]),
    ("Learning\nRate", [
        Phase("Warmup (10%\u2192100%)", 0.0,  3.0, "blue"),
        Phase("Cosine Decay",           3.0, 20.0, "purple"),
    ]),
    ("Observers", [
        Phase("Updating (tracking)", 0.0,  7.0, "blue"),
        Phase("Frozen (fixed grid)", 7.0, 20.0, "purple"),
    ]),
    ("BN Folding", [
        Phase("Fold before QAT", 0.0, 1.0, "yellow"),
    ]),
]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": SECTION_FONT_SIZE,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    n_rows = len(ROWS)
    # y = n_rows - 1 for the first row (top), 0 for the last (bottom).
    row_centers = [n_rows - 1 - i for i in range(n_rows)]
    bar_half = 0.28  # bar vertical half-height in data units

    # Alternating lane shading so the eye can track long rows.
    for ri, y in enumerate(row_centers):
        if ri % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, facecolor=LANE_BG, zorder=0)

    # Epoch grid lines (every 2 epochs, light).
    for xg in range(0, TOTAL_EPOCHS + 1, 2):
        ax.axvline(xg, color=GRID_COLOR, linewidth=0.5, zorder=1)

    # Draw bars and labels.
    for ri, (_section_label, phases) in enumerate(ROWS):
        y = row_centers[ri]
        for p in phases:
            bar_width = p.end - p.start
            rect = FancyBboxPatch(
                (p.start, y - bar_half),
                bar_width,
                2 * bar_half,
                boxstyle="round,pad=0,rounding_size=0.10",
                facecolor=BAR_FILLS[p.color],
                edgecolor=EDGE_COLOR,
                linewidth=0.8,
                zorder=2,
            )
            ax.add_patch(rect)

            cx = (p.start + p.end) / 2
            # Label placement rules:
            #   - bar wide enough (>=3.5 epochs): centered inside
            #   - short bar where label is wider than bar (Warmup LR,
            #     3-epoch bar with a 17-char label): drawn above the
            #     bar, left-anchored at the bar start so the label
            #     does not spill past the left axis
            #   - very short bar at the bottom (BN Fold, 1-epoch bar):
            #     drawn to the right of the bar
            if bar_width >= 3.5:
                ax.text(
                    cx, y, p.label,
                    ha="center", va="center",
                    fontsize=BAR_FONT_SIZE, color=TEXT_COLOR,
                    zorder=3,
                )
            elif bar_width >= 2.0:
                ax.text(
                    p.start, y + bar_half + 0.06, p.label,
                    ha="left", va="bottom",
                    fontsize=BAR_FONT_SIZE, color=TEXT_COLOR,
                    zorder=3,
                )
            else:
                ax.text(
                    p.end + 0.25, y, p.label,
                    ha="left", va="center",
                    fontsize=BAR_FONT_SIZE, color=TEXT_COLOR,
                    zorder=3,
                )

    # Axis formatting.
    ax.set_xlim(-0.20, TOTAL_EPOCHS + 0.20)
    ax.set_ylim(-0.70, n_rows - 0.30)
    ax.set_xticks(list(range(0, TOTAL_EPOCHS + 1, 2)))
    ax.tick_params(axis="x", labelsize=TICK_FONT_SIZE, length=2, pad=1,
                   colors=EDGE_COLOR)
    ax.set_xlabel("Epoch", fontsize=AXIS_LABEL_FONT_SIZE,
                  color=TEXT_COLOR, labelpad=2)

    ax.set_yticks(row_centers)
    ax.set_yticklabels(
        [label for label, _ in ROWS],
        fontsize=SECTION_FONT_SIZE,
        color=TEXT_COLOR,
    )
    ax.tick_params(axis="y", length=0, pad=4)

    # Spine styling.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(EDGE_COLOR)
        ax.spines[side].set_linewidth(0.8)

    fig.suptitle(
        "QAT Training Schedule (20 Epochs)",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.97,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{FIG_BASE}.png"
    pdf_path = FIGURE_DIR / f"{FIG_BASE}.pdf"
    svg_path = FIGURE_DIR / f"{FIG_BASE}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:\n  {png_path}\n  {pdf_path}\n  {svg_path}")


if __name__ == "__main__":
    main()
