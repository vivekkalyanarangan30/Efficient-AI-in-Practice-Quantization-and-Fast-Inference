#!/usr/bin/env python3
"""Chapter 7, Figure 7.20 -- LLM quantization decision tree.

Replaces the original Mermaid-exported PNG/SVG for CH07_F20.  The
Mermaid flowchart exported a 688 x 1979 viewBox; once fit to Manning's
5.6-inch column that would be ~16 inches tall (over the 7-inch Manning
limit) with ~3pt node text.  This script redraws the same two-stage
decision tree directly at Manning dimensions with 7-8pt Arial and uses
matplotlib's patchA/patchB auto-clipping so every arrow snaps cleanly
to the visible edge of its source/destination node -- the same pattern
used in ch5_decision_tree.py / ch7_llm_int8_flow.py.

The tree has two stages:
  Stage 1 (top):    pick a weight-quantization method
                    (LLM.int8 / AWQ / GPTQ)
  Separator:        "-- Weight method chosen --"
  Stage 2 (bottom): optionally add TurboQuant KV-cache compression

Run:
    python ch7_quant_decision_tree.py
Outputs (in ch7/figures/):
    CH07_F20_Kalyanarangan.png
    CH07_F20_Kalyanarangan.pdf
    CH07_F20_Kalyanarangan.svg
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

# ---------------------------------------------------------------------------
# Manning layout constants
# ---------------------------------------------------------------------------
FIGURE_DIR = Path(__file__).parent / "figures"
FIG_BASE = "CH07_F20_Kalyanarangan"

FIG_W = 5.6   # inches (Manning column width)
FIG_H = 6.90  # inches (just under Manning 7.0-inch height limit)

NODE_FONT_SIZE = 7
EDGE_FONT_SIZE = 6.5
SEP_FONT_SIZE = 7
TITLE_FONT_SIZE = 9.5

EDGE_COLOR = "#28253D"
TEXT_COLOR = "#28253D"
SEP_COLOR = "#6E6A8A"    # muted purple-grey for the stage separator line

# Node styles (subtle tinted fills, grayscale-safe -- same palette as
# ch5_decision_tree.py and ch7_llm_int8_flow.py).
STYLE = {
    "start":    {"fill": "#DCE6F5", "edge": EDGE_COLOR, "lw": 1.1},
    "decision": {"fill": "#FFF2C6", "edge": EDGE_COLOR, "lw": 1.0},
    "process":  {"fill": "#E8E0F4", "edge": EDGE_COLOR, "lw": 1.0},
    "end":      {"fill": "#D6EDC8", "edge": EDGE_COLOR, "lw": 1.2},
}

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


# Coordinate system (y inverted at draw time so y grows downward):
# xlim -0.20..12.20 (span 12.40), ylim -0.20..15.08 (span 15.28).
# Aspect 12.40 / 15.28 = 0.811 matches FIG_W / FIG_H = 5.6 / 6.9
# so set_aspect("equal") leaves no ugly whitespace.  Each data unit
# is ~0.45" physical -- same scale as ch5_decision_tree.py, which
# keeps 7pt node text readable inside node boxes.
#
# Sizing rule for every node: width = (longest text line in 7pt Arial
# data-unit width) + 0.55 padding; height = (n_lines * 0.26) + 0.45
# padding.  Decision nodes render as rounded rectangles (not diamonds)
# so that 3-line subtitles like "Sub-4-bit \u00b7 narrow domain" keep
# their full width -- the original diamond shape pinched the top and
# bottom text rows and clipped wide subtitles.
NODES_LIST: List[Node] = [
    # Stage 1 -- weight-method picker
    Node("START",
         6.00,  0.55,
         "What is your binding constraint?",
         "start",    w=4.40, h=0.85),
    Node("Q1",
         6.00,  2.00,
         "Need to deploy\nimmediately?\nNo calibration data",
         "decision", w=3.10, h=1.30),
    Node("LLM",
         1.90,  4.00,
         "LLM.int8()\n6.4 GB \u00b7 zero setup",
         "end",      w=2.90, h=1.20),
    Node("Q2",
         8.70,  4.00,
         "Is model memory\nthe bottleneck?\nNeed INT4",
         "decision", w=2.70, h=1.30),
    Node("Q3",
         8.70,  5.90,
         "Broad or unpredictable\ndeployment distribution?\nChatbot \u00b7 multi-domain",
         "decision", w=3.90, h=1.30),
    Node("AWQ",
         3.90,  7.85,
         "AWQ 4-bit g128\n3.6 GB \u00b7 domain-robust",
         "end",      w=3.20, h=1.20),
    Node("Q4",
         9.40,  7.85,
         "Need maximum\nper-layer accuracy?\nSub-4-bit \u00b7 narrow domain",
         "decision", w=3.90, h=1.30),
    Node("GPTQ",
         9.40,  9.80,
         "GPTQ 4-bit g128\n3.6 GB \u00b7 Hessian-optimal",
         "end",      w=3.40, h=1.20),
    # Stage 2 -- KV cache follow-up
    Node("Q5",
         6.00, 12.00,
         "Is KV cache memory\nlimiting concurrent users?\nLong context \u00b7 high QPS",
         "decision", w=4.00, h=1.30),
    Node("TQ",
         3.10, 14.10,
         "Add TurboQuant TQ4\n3.9\u00d7 KV compression\n+0.7% perplexity",
         "process",  w=3.00, h=1.30),
    Node("DONE",
         8.80, 14.10,
         "Deploy",
         "end",      w=1.70, h=0.85),
]
NODES = {n.name: n for n in NODES_LIST}


# Stage-1 separator y-position (drawn as a thin dashed horizontal rule
# with an inline label bubble, no node and no arrows).
SEPARATOR_Y = 10.90
SEPARATOR_X_LO = 0.40
SEPARATOR_X_HI = 11.60


# (src, dst, label, rad)
# rad > 0 curves one way, < 0 the other, 0 = straight chord.
EDGES: List[Tuple[str, str, Optional[str], float]] = [
    ("START", "Q1",   None,  0.00),
    ("Q1",    "LLM",  "Yes", 0.00),
    ("Q1",    "Q2",   "No",  0.00),
    ("Q2",    "LLM",  "No",  0.00),   # horizontal left
    ("Q2",    "Q3",   "Yes", 0.00),
    ("Q3",    "AWQ",  "Yes", 0.00),
    ("Q3",    "Q4",   "No",  0.00),
    ("Q4",    "GPTQ", "Yes", 0.00),
    ("Q4",    "AWQ",  "No",  0.00),   # horizontal left
    ("Q5",    "TQ",   "Yes", 0.00),
    ("Q5",    "DONE", "No",  0.00),
    ("TQ",    "DONE", None,  0.00),   # horizontal right
]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_node(ax, n: Node) -> Patch:
    style = STYLE[n.kind]
    # All nodes render as rounded rectangles (pill shape for START /
    # DONE).  Decision nodes used to be diamonds but the tapering
    # points clipped 3-line subtitle text; the yellow fill alone is
    # enough visual signal for "decision point".
    rounding = n.h / 2 if n.name in ("START", "DONE") else 0.22
    patch: Patch = FancyBboxPatch(
        (n.x - n.w / 2, n.y - n.h / 2),
        n.w,
        n.h,
        boxstyle=f"round,pad=0,rounding_size={rounding:.3f}",
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
        linespacing=1.10,
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

    # Place label at a biased midpoint so the arrowhead end stays free.
    mx = src.x * 0.55 + dst.x * 0.45
    my = src.y * 0.55 + dst.y * 0.45
    if rad != 0.0:
        dx = dst.x - src.x
        dy = dst.y - src.y
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        px = -dy / length
        py = dx / length
        offset = rad * (length / 2.0)
        mx += px * offset
        my += py * offset

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


def draw_separator(ax) -> None:
    """Horizontal stage separator with an inline label bubble."""
    # Two dashed rules flanking a centered label.
    ax.plot(
        [SEPARATOR_X_LO, 3.70], [SEPARATOR_Y, SEPARATOR_Y],
        color=SEP_COLOR, linewidth=0.8, linestyle="--",
        zorder=Z_ARROW,
    )
    ax.plot(
        [8.30, SEPARATOR_X_HI], [SEPARATOR_Y, SEPARATOR_Y],
        color=SEP_COLOR, linewidth=0.8, linestyle="--",
        zorder=Z_ARROW,
    )
    ax.text(
        6.00, SEPARATOR_Y, "Weight method chosen",
        ha="center", va="center",
        fontsize=SEP_FONT_SIZE, color=SEP_COLOR,
        fontstyle="italic",
        bbox=dict(
            boxstyle="round,pad=0.25",
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
    # Let the axes fill the figure (no bbox trimming at save time) so
    # the PNG is exactly FIG_W x FIG_H.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.01)
    ax.set_xlim(-0.20, 12.20)
    ax.set_ylim(-0.20, 15.08)
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

    draw_separator(ax)

    fig.suptitle(
        "LLM quantization decision tree",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.985,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{FIG_BASE}.png"
    pdf_path = FIGURE_DIR / f"{FIG_BASE}.pdf"
    svg_path = FIGURE_DIR / f"{FIG_BASE}.svg"
    # Save without bbox_inches="tight" so FIG_W x FIG_H is preserved.
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)
    print(f"Saved:\n  {png_path}\n  {pdf_path}\n  {svg_path}")


if __name__ == "__main__":
    main()
