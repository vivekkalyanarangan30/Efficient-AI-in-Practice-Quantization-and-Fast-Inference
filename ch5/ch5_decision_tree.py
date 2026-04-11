#!/usr/bin/env python3
"""Chapter 5, Figure 5.5 -- PTQ accuracy-gap decision tree.

Replaces the original Mermaid-exported SVG/PNG for CH05_F05.  The
Mermaid PNG was technically high-pixel but, once fit to Manning's
5.6-inch column, its 14px-in-viewBox node text shrank below ~5pt and
became unreadable in print.  This version lays the same decision tree
out directly at Manning dimensions with 8pt Arial and uses matplotlib's
patchA/patchB auto-clipping so every arrow snaps cleanly to the visible
edge of its source/destination node.

Run:
    python ch5_decision_tree.py
Outputs (in ch5/figures/):
    CH05_F05_Kalyanarangan.png
    CH05_F05_Kalyanarangan.pdf
    CH05_F05_Kalyanarangan.svg
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Polygon

# ---------------------------------------------------------------------------
# Manning layout constants
# ---------------------------------------------------------------------------
FIGURE_DIR = Path(__file__).parent / "figures"
FIG_BASE = "CH05_F05_Kalyanarangan"

FIG_W = 5.6   # inches (Manning column width)
FIG_H = 7.0   # inches (Manning column height)

NODE_FONT_SIZE = 8
EDGE_FONT_SIZE = 7
TITLE_FONT_SIZE = 9.5

EDGE_COLOR = "#28253D"
TEXT_COLOR = "#28253D"

# Node styles (subtle tinted fills, grayscale-safe).
STYLE = {
    "start":    {"fill": "#DCE6F5", "edge": EDGE_COLOR, "lw": 1.1},
    "decision": {"fill": "#FFF2C6", "edge": EDGE_COLOR, "lw": 1.1},
    "process":  {"fill": "#E8E0F4", "edge": EDGE_COLOR, "lw": 1.1},
    "end":      {"fill": "#D6EDC8", "edge": EDGE_COLOR, "lw": 1.3},
}

# Z-order layers: arrows beneath nodes, labels on top of arrows.
Z_ARROW = 1
Z_NODE = 2
Z_EDGE_LABEL = 3


# ---------------------------------------------------------------------------
# Node + edge data
# ---------------------------------------------------------------------------
@dataclass
class Node:
    name: str
    x: float
    y: float
    label: str
    kind: str            # one of STYLE keys
    w: float = 2.0
    h: float = 0.9


# Coordinate system: y grows downward after ax.invert_yaxis().
# xlim -0.4..12.1 (span 12.5), ylim -0.2..15.425 (span 15.625).
# Aspect ratio 12.5 / 15.625 = 0.8 -> matches 5.6/7.0 figure aspect so
# each data unit is ~0.45" physical.
#
# Layout rules used when placing nodes:
#   1. Keep each data unit ~0.45" physical (same as v2) so 8pt text stays
#      large enough inside shapes.
#   2. Horizontal gap between siblings on a row >= 0.6 data units (~0.27").
#   3. Every diagonal arrow end-to-end >= 2.8 data units (~1.25") so there
#      is visible shaft after patchA/patchB clipping.
#   4. ACC_CRIT and CONS_LIGHT share x so the long vertical arrow runs
#      down a clean empty column.
#   5. SHIP_PTQ gets its own half-row (y between the ACC_CRIT and
#      QAT_LIKELY rows) on the far left so its two "No" inbound arrows
#      are real diagonals with clear labels.
NODES_LIST: List[Node] = [
    Node("PTQ_GAP",    5.60,  0.90, "PTQ Accuracy Gap",              "start",    w=2.45, h=0.95),
    Node("GAP_2",      5.60,  2.75, "Gap > 2%?",                     "decision", w=2.10, h=1.25),
    Node("GAP_05",     2.30,  4.90, "Gap > 0.5%?",                   "decision", w=2.30, h=1.25),
    Node("UNI_CON",    8.80,  4.90, "Uniform or\nConcentrated?",     "decision", w=2.75, h=1.55),
    # ACC_CRIT shifted right off x=2.3 so the GAP_05->SHIP_PTQ straight
    # line (which passes through x~1.45 at y=7.15) clears its left vertex.
    Node("ACC_CRIT",   2.80,  7.15, "Accuracy\nCritical?",           "decision", w=2.20, h=1.55),
    Node("MODEL_SZ",   5.60,  7.15, "Model Size?",                   "decision", w=2.25, h=1.25),
    Node("MIX_PREC",   9.40,  7.15, "Try Mixed\nPrecision First",    "process",  w=2.45, h=1.20),
    Node("SHIP_PTQ",   0.50,  9.65, "Ship PTQ\nResult",              "end",      w=1.75, h=1.20),
    # QAT_LIKELY pulled right so the ACC_CRIT->CONS_LIGHT near-vertical
    # arrow at x~2.6..2.3 clears its left edge.
    Node("QAT_LIKELY", 4.00,  9.65, "QAT Likely\nNeeded",            "end",      w=2.00, h=1.20),
    Node("CALIB_TUNE", 6.80,  9.65, "Try Calibration\nTuning First", "process",  w=2.55, h=1.20),
    Node("MIX_SUFF",  10.00,  9.65, "Mixed Precision\nSufficient?",  "decision", w=2.80, h=1.55),
    Node("CALIB_IMP",  6.80, 11.85, "Calibration\nImproved?",        "decision", w=2.30, h=1.55),
    Node("SHIP_MIX",  10.70, 11.85, "Ship Mixed\nPrecision",         "end",      w=1.95, h=1.20),
    Node("CONS_LIGHT", 2.30, 14.10, "Consider\nLight QAT",           "end",      w=1.85, h=1.20),
    Node("QAT_REC",    6.80, 14.10, "QAT Recommended",               "end",      w=2.45, h=0.95),
]
NODES = {n.name: n for n in NODES_LIST}


# (src, dst, label, rad)  -- rad > 0 curves one way, < 0 the other, 0 = straight
EDGES: List[Tuple[str, str, Optional[str], float]] = [
    ("PTQ_GAP",    "GAP_2",      None,           0.00),
    ("GAP_2",      "GAP_05",     "No",           0.00),
    ("GAP_2",      "UNI_CON",    "Yes",          0.00),
    ("GAP_05",     "SHIP_PTQ",   "No",           0.00),
    ("GAP_05",     "ACC_CRIT",   "Yes",          0.00),
    ("ACC_CRIT",   "SHIP_PTQ",   "No",           0.00),
    ("ACC_CRIT",   "CONS_LIGHT", "Yes",          0.00),
    ("UNI_CON",    "MODEL_SZ",   "Uniform",      0.00),
    ("UNI_CON",    "MIX_PREC",   "Concentrated", 0.00),
    ("MODEL_SZ",   "QAT_LIKELY", "Medium",       0.00),
    ("MODEL_SZ",   "CALIB_TUNE", "Small <10M /\nLarge >50M", 0.00),
    ("CALIB_TUNE", "CALIB_IMP",  None,           0.00),
    ("CALIB_IMP",  "CONS_LIGHT", "Yes",          0.00),
    ("CALIB_IMP",  "QAT_REC",    "No",           0.00),
    ("MIX_PREC",   "MIX_SUFF",   None,           0.00),
    ("MIX_SUFF",   "SHIP_MIX",   "Yes",          0.00),
    # Slight arc so the arrow visibly curves away from CALIB_IMP's right vertex.
    ("MIX_SUFF",   "QAT_REC",    "No",          -0.18),
]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_node(ax, n: Node) -> Patch:
    style = STYLE[n.kind]
    if n.kind == "decision":
        patch: Patch = Polygon(
            [
                (n.x,            n.y - n.h / 2),
                (n.x + n.w / 2,  n.y),
                (n.x,            n.y + n.h / 2),
                (n.x - n.w / 2,  n.y),
            ],
            closed=True,
            facecolor=style["fill"],
            edgecolor=style["edge"],
            linewidth=style["lw"],
            joinstyle="miter",
            zorder=Z_NODE,
        )
    else:
        patch = FancyBboxPatch(
            (n.x - n.w / 2, n.y - n.h / 2),
            n.w,
            n.h,
            boxstyle="round,pad=0,rounding_size=0.22",
            facecolor=style["fill"],
            edgecolor=style["edge"],
            linewidth=style["lw"],
            zorder=Z_NODE,
        )
    ax.add_patch(patch)
    ax.text(
        n.x, n.y, n.label,
        ha="center", va="center",
        fontsize=NODE_FONT_SIZE, color=TEXT_COLOR,
        linespacing=1.05,
        zorder=Z_NODE + 0.5,
    )
    return patch


def draw_edge(
    ax,
    src: Node,
    dst: Node,
    src_patch: Patch,
    dst_patch: Patch,
    label: Optional[str],
    rad: float = 0.0,
) -> None:
    """Draw an arrow using patchA/patchB so it snaps to each node's visible edge."""
    arrow = FancyArrowPatch(
        (src.x, src.y),
        (dst.x, dst.y),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=6,head_width=4",
        linewidth=0.9,
        color=EDGE_COLOR,
        patchA=src_patch,
        patchB=dst_patch,
        shrinkA=0.5,
        shrinkB=0.5,
        zorder=Z_ARROW,
    )
    ax.add_patch(arrow)

    if not label:
        return

    # Place label at the geometric midpoint of the (straight or curved) path.
    mx = (src.x + dst.x) / 2
    my = (src.y + dst.y) / 2
    if rad != 0.0:
        # For arc3, the midpoint of the bezier is offset perpendicular to the
        # chord by |rad| * (chord_length / 2).  We nudge the label the same
        # way so it sits on the visible arc rather than the chord.
        dx = dst.x - src.x
        dy = dst.y - src.y
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        # Perpendicular (rotated +90 deg in screen coords; y is inverted).
        px = -dy / length
        py = dx / length
        offset = rad * (length / 2.0)
        mx += px * offset
        my += py * offset
    else:
        # Bias straight-line labels slightly toward source so the head end is
        # left free for the arrowhead.
        mx = src.x * 0.55 + dst.x * 0.45
        my = src.y * 0.55 + dst.y * 0.45

    ax.text(
        mx, my, label,
        ha="center", va="center",
        fontsize=EDGE_FONT_SIZE, color=TEXT_COLOR,
        linespacing=1.0,
        bbox=dict(
            boxstyle="round,pad=0.15",
            facecolor="white",
            edgecolor="none",
        ),
        zorder=Z_EDGE_LABEL,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": NODE_FONT_SIZE,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(-0.40, 12.10)
    ax.set_ylim(-0.20, 15.425)
    ax.invert_yaxis()        # top-down flow
    ax.set_aspect("equal")
    ax.axis("off")

    patches: Dict[str, Patch] = {}
    for n in NODES_LIST:
        patches[n.name] = draw_node(ax, n)

    for src_name, dst_name, label, rad in EDGES:
        draw_edge(
            ax,
            NODES[src_name],
            NODES[dst_name],
            patches[src_name],
            patches[dst_name],
            label,
            rad,
        )

    fig.suptitle(
        "PTQ accuracy-gap decision guide",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))

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
