"""ch11_4_figures.py — Android-specific figures for ch11 §11.4.

Section served: 11.4 (Edge / Mobile tier-up — Android delegate routing).
The cross-modal design-space + cross-device latency views live in
ch11_1_aggregate.py (F0101 / F0104); the Apple cross-platform Whisper /
Llama comparisons live in ch11_3_apple.py (F0304 / F0306). This script
adds one Android-anchored view that the other generators don't cover:
the TFLite delegate-portability matrix on Pixel.

Modes:
  figures   Regenerate CH11_F0401 (delegate-portability matrix) under
            figures/ch11_4/.
  smoke     No-op shape check.

Read-only on results.json; writes only under figures/ch11_4/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse the apple module's shared palette / hatch / save helpers so chapter
# figures retain a consistent look.
import ch11_3_apple as apple

HERE = Path(__file__).resolve().parent
# Data file was renamed results.json -> runs.json (same schema/contents).
RESULTS_JSON = HERE / "runs.json"
FIG_DIR = HERE / "figures" / "ch11_4"
LOG_DIR = HERE / "logs"

SCRIPT_NAME = "ch11_4_figures.py"

DEVICE = "Google Pixel 10 Pro"
DELEGATES = ["xnnpack_1t", "xnnpack_4t", "nnapi", "gpu"]
DELEGATE_LABEL = {
    "xnnpack_1t": "XNNPACK 1T",
    "xnnpack_4t": "XNNPACK 4T",
    "nnapi": "NNAPI",
    "gpu": "GPU",
}
# Cells that were actually attempted on-device and refused by the delegate.
# Empty cells outside this set are "not in the test plan" rather than rejected.
# See caveats.md 2026-05-12T19:01 — NNAPI on Pixel 10 Pro returns
# ANEURALNETWORKS_BAD_DATA for the int8 graph and rejects INT64 tensors for
# the int16x8 graph. Whisper's audio sweep only runs XNNPACK_4T/NNAPI/GPU
# (AudioBenchmark.kt does not include the single-thread variant), so the
# XNNPACK_1T cells for whisper are "—" not "×".
KNOWN_REJECTIONS = {
    ("efficientnet_lite0", "tflite_int8",   "nnapi"),
    ("efficientnet_lite0", "tflite_int16x8", "nnapi"),
}


def _setup_logger(mode: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"ch11_4_figures.{mode}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_DIR / f"ch11_4_figures_{mode}.log", mode="a", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def _load_records() -> list[dict]:
    with RESULTS_JSON.open("r", encoding="utf-8") as f:
        return json.load(f).get("records", [])


def _is_throughput_row(r: dict) -> bool:
    cu = r.get("compute_units") or ""
    return not cu.endswith(("_sustained_300s", "_sustained_60s", "_power_30s"))


def _render_delegate_matrix(records: list[dict], logger: logging.Logger) -> None:
    """F0401 — TFLite delegate × (model, variant) latency matrix on Pixel.

    Rows are model-variant pairs; columns are delegates (XNNPACK 1T/4T, NNAPI,
    GPU). Each cell is the mean latency in milliseconds, with the cell coloured
    by relative speed within its row. Missing combinations (e.g., NNAPI
    rejecting INT8) are rendered as "rejected ✗" and shaded grey.
    """
    pixel = [r for r in records
             if r.get("device", {}).get("name") == DEVICE
             and r.get("backend") == "tflite"
             and r.get("latency_ms")
             and (r.get("latency_ms") or {}).get("mean") is not None
             and _is_throughput_row(r)]
    if not pixel:
        logger.warning("F0401: no Pixel TFLite records found — skipping")
        return

    # Stable row order: model family, then variant precision (fp32 → dynrange → int8 → int16x8).
    model_order = ["efficientnet_lite0", "whisper_tiny"]
    variant_rank = {"fp32": 0, "dynrange": 1, "int8": 2, "int16x8": 3}

    def _row_key(r: dict) -> tuple:
        m = r["model"]
        m_idx = model_order.index(m) if m in model_order else len(model_order)
        v = r["variant"].split("_", 1)[-1]
        return (m_idx, variant_rank.get(v, 99), r["variant"])

    row_keys = sorted({_row_key(r) + (r["model"], r["variant"]) for r in pixel})
    rows = [(rk[-2], rk[-1]) for rk in row_keys]

    matrix = np.full((len(rows), len(DELEGATES)), np.nan)
    for ri, (model, variant) in enumerate(rows):
        for ci, deleg in enumerate(DELEGATES):
            match = [r for r in pixel
                     if r["model"] == model
                     and r["variant"] == variant
                     and r.get("compute_units") == deleg]
            if match:
                matrix[ri, ci] = match[0]["latency_ms"]["mean"]

    fig, ax = plt.subplots(figsize=(7.0, 0.6 * len(rows) + 1.8))

    # Per-row min-max normalisation drives the cell shade (fastest = lightest).
    norm = np.zeros_like(matrix)
    for ri in range(matrix.shape[0]):
        row = matrix[ri]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        lo, hi = finite.min(), finite.max()
        if hi > lo:
            norm[ri] = (row - lo) / (hi - lo)
        else:
            norm[ri] = 0.5
        norm[ri][~np.isfinite(row)] = np.nan

    cmap = plt.cm.Blues
    for ri in range(matrix.shape[0]):
        model, variant = rows[ri]
        for ci in range(matrix.shape[1]):
            val = matrix[ri, ci]
            if not np.isfinite(val):
                rejected = (model, variant, DELEGATES[ci]) in KNOWN_REJECTIONS
                ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                           facecolor="#e8e8e8", edgecolor="white",
                                           linewidth=1.2, zorder=1))
                # "×" (U+00D7) is Arial-safe; "—" (U+2014) is also Arial-safe.
                label = "× rejected" if rejected else "— not in sweep"
                ax.text(ci, ri, label, ha="center", va="center",
                        fontsize=7, color="#666666", zorder=2)
            else:
                shade = 0.20 + 0.55 * norm[ri, ci]  # cap at ~0.75 so text stays legible
                ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1, 1,
                                           facecolor=cmap(shade), edgecolor="white",
                                           linewidth=1.2, zorder=1))
                txt = f"{val:.1f} ms" if val < 100 else f"{val:.0f} ms"
                # White text on dark cells, black on light.
                text_color = "white" if shade > 0.55 else "black"
                ax.text(ci, ri, txt, ha="center", va="center",
                        fontsize=7, color=text_color, zorder=2)

    ax.set_xticks(range(len(DELEGATES)))
    ax.set_xticklabels([DELEGATE_LABEL[d] for d in DELEGATES], fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{m.split('_')[0]} / {v.split('_', 1)[-1]}" for m, v in rows], fontsize=8)
    ax.set_xlim(-0.5, len(DELEGATES) - 0.5)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.invert_yaxis()  # first row at top
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("11.4.1 — Pixel 10 Pro: TFLite delegate × (model / variant) latency matrix",
                 fontsize=10)

    fig.tight_layout()
    apple._save_pair(fig, "CH11_F0401_Kalyanarangan", "ch11_4")
    plt.close(fig)
    logger.info("wrote F0401 delegate-portability matrix (%d rows × %d cols)",
                len(rows), len(DELEGATES))


def mode_figures(args: argparse.Namespace) -> int:
    logger = _setup_logger("figures")
    if not RESULTS_JSON.is_file():
        logger.error("missing results.json at %s", RESULTS_JSON)
        return 1
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    records = _load_records()
    _render_delegate_matrix(records, logger)
    return 0


def mode_smoke(args: argparse.Namespace) -> int:
    logger = _setup_logger("smoke")
    logger.info("smoke OK — DELEGATES=%s rows expected=2 modalities (vision, audio)",
                DELEGATES)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog=SCRIPT_NAME, description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    p_fig = sub.add_parser("figures", help="render F0401 + F0402")
    p_fig.set_defaults(func=mode_figures)
    p_smoke = sub.add_parser("smoke", help="no-op shape check")
    p_smoke.set_defaults(func=mode_smoke)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
