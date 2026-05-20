"""ch11_1_aggregate.py — Sec 11.1 (read-only aggregator, self-contained).

Section served: 11.1 (design space, sustained, budget).

Modes:
  summary             Print table per (model, modality) of (backend, variant, device)
                      combinations. Echoed to stdout and to run_report.md.
  design-space        11.1.1 — latency × power scatter; one point per record where
                      both populated; marker shape by device.class; hatch by backend.
                      Records with null power on a separate strip with explicit label.
  sustained-curve     11.1.2 — throughput-over-time for records with sustained populated.
  budget-calculator   CLI: --p95-ms / --power-mw filter; prints subset and writes JSON.
  budget-figure       11.1.3 — shaded budget-feasible region overlay on 11.1.1.
  figures             design-space + sustained-curve + budget-figure.
  all                 summary + figures.

This script never produces measurements. It is a read-only consumer of
results.json and only writes figures + run_report.md (when summary runs).

Invocation:
  python ch11_1_aggregate.py summary
  python ch11_1_aggregate.py design-space
  python ch11_1_aggregate.py budget-calculator --p95-ms 33 --power-mw 5000
  python ch11_1_aggregate.py all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Manning-compatible palette mirrored from ch10: muted hues that keep separation
# in grayscale (Level 2-3 of the Manning color sheet). Each backend gets its
# own colour AND each variant family its own hatch so the figures decode
# unambiguously in print B&W (use hatch) and in eBook colour (use both).
BACKEND_COLORS = {
    "tflite":  "#319974",   # teal-green, Manning green L3
    "coreml":  "#7E76B0",   # muted purple, Manning purple L2-3
    "mlx":     "#D67430",   # orange, Manning orange L3
    "mps":     "#3A6FA8",   # muted blue, Manning blue L2-3
    "prepost": "#888888",   # neutral grey
}
BACKEND_FALLBACK = "#444444"

VARIANT_HATCH_RULES = (
    # (substring tested against variant.lower(), hatch)
    ("int16x8",      "xxxx"),
    ("int4",         "\\\\"),
    ("palettize",    "\\\\"),
    ("int8",         "////"),
    ("dynrange",     "...."),
    ("dyn_range",    "...."),
)
# B&W-PRIMARY ENCODING. Per Manning Graphics Guidelines, colour cannot carry
# meaning ("should not be used for teaching"). The figures must remain
# decodable when printed in greyscale, so the *primary* axis is marker shape
# (backend) and marker size (device class); colour is decorative only.
BACKEND_MARKERS = {
    "tflite":  "o",   # circle
    "coreml":  "s",   # square
    "mlx":     "^",   # triangle-up
    "mps":     "D",   # diamond
    "prepost": "X",   # filled X
}
BACKEND_MARKER_FALLBACK = "P"

# Device-class encoding: marker SIZE (small/large), readable in B&W.
DEVICE_SIZES = {
    "laptop": 70,
    "phone":  140,
    "sbc":    100,
    "nuc":    110,
    "jetson": 130,
}
# Retained for back-compat with code paths that still want a shape per
# device class (e.g. 11.1.2 sustained line plot, where the marker also tags
# the device class). For the design-space scatter, shape now means backend.
DEVICE_MARKERS = {"laptop": "o", "phone": "s", "sbc": "^", "nuc": "v", "jetson": "D"}

# Friendly device-class display labels for legends. Singular, lower-case.
DEVICE_CLASS_LABELS = {
    "laptop": "MacBook (laptop)",
    "phone":  "Mobile (phone)",
    "sbc":    "SBC",
    "nuc":    "Mini-PC (NUC)",
    "jetson": "Jetson",
}

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "results.json"
RUN_REPORT = HERE / "run_report.md"
FIG_DIR = HERE / "figures" / "ch11_1"
LOG_DIR = HERE / "logs"

SCRIPT_NAME = "ch11_1_aggregate.py"


def _setup_logger(mode: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"ch11_1_aggregate.{mode}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(LOG_DIR / f"ch11_1_aggregate_{mode}.log", mode="a", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def _save_pair(fig, name: str, section: str = "ch11_1") -> tuple[Path, Path]:
    out_dir = HERE / "figures" / section
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"; pdf = out_dir / f"{name}.pdf"
    cap = out_dir / f"{name.split('_Kalyanarangan')[0]}_caption.md"
    fig.savefig(png, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    if not cap.exists():
        cap.write_text(f"Caption skeleton for {name}. Hatched markers; B&W; flesh out post-run.\n")
    return png, pdf


def _load_records() -> list[dict]:
    if not RESULTS_JSON.exists():
        return []
    data = json.loads(RESULTS_JSON.read_text())
    return data.get("records", [])


# --------------------------------------------------------------------------- #
# Summary                                                                     #
# --------------------------------------------------------------------------- #
def mode_summary(args: argparse.Namespace) -> int:
    logger = _setup_logger("summary")
    recs = _load_records()
    grouped: dict[tuple, list[dict]] = {}
    for r in recs:
        key = (r.get("model"), r.get("modality"))
        grouped.setdefault(key, []).append(r)
    lines = ["# ch11 results.json summary", f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_", ""]
    if not recs:
        lines.append("(no records yet)")
    for (model, mod), rs in sorted(grouped.items(), key=lambda x: (x[0][0] or "", x[0][1] or "")):
        lines.append(f"## {model} ({mod})")
        lines.append("| backend | variant | compute_units | device | latency_p50 | tput | acc | power_mw | n |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        cnt: dict[tuple, int] = {}
        for r in rs:
            k = (r.get("backend"), r.get("variant"), r.get("compute_units"), (r.get("device") or {}).get("name"))
            cnt[k] = cnt.get(k, 0) + 1
        for r in rs:
            lat = (r.get("latency_ms") or {}).get("p50")
            tput = (r.get("throughput") or {}).get("samples_per_sec") or (r.get("throughput") or {}).get("tokens_per_sec")
            acc = (r.get("accuracy") or {}).get("value")
            pw = (r.get("power_mw") or {}).get("mean")
            cell_lat = f"{lat:.2f} ms" if lat is not None else "-"
            cell_tput = f"{tput:.2f}" if tput is not None else "-"
            cell_acc = f"{acc:.3f}" if acc is not None else "-"
            cell_pw = f"{int(pw)}" if pw is not None else "-"
            count = cnt[(r.get("backend"), r.get("variant"), r.get("compute_units"),
                         (r.get("device") or {}).get("name"))]
            lines.append(
                f"| {r.get('backend')} | {r.get('variant')} | {r.get('compute_units') or '-'} | "
                f"{(r.get('device') or {}).get('name')} | {cell_lat} | {cell_tput} | "
                f"{cell_acc} | {cell_pw} | {count} |"
            )
        lines.append("")
    text = "\n".join(lines)
    print(text)
    # Append to existing run_report.md without clobbering.
    existing = RUN_REPORT.read_text() if RUN_REPORT.exists() else ""
    sep = "\n\n---\n\n"
    if "# ch11 results.json summary" in existing:
        # Replace section
        head = existing.split("# ch11 results.json summary")[0].rstrip()
        RUN_REPORT.write_text(head + sep + text + "\n")
    else:
        RUN_REPORT.write_text((existing + sep if existing else "") + text + "\n")
    logger.info("wrote %s; %d records summarized", RUN_REPORT, len(recs))
    return 0


# --------------------------------------------------------------------------- #
# Design space (11.1.1)                                                       #
# --------------------------------------------------------------------------- #


def _backend_color(backend: str | None) -> str:
    return BACKEND_COLORS.get(backend, BACKEND_FALLBACK)


def _variant_hatch(variant: str | None) -> str:
    """Map a variant string (fp32 / int8 / dynrange / int16x8 / coreml_int4_palettize / …)
    to one of the Manning-safe hatch patterns. Floats / fp16 baselines get no hatch
    so they stand out as the "reference" point in B&W."""
    if not variant:
        return ""
    v = variant.lower()
    for needle, hatch in VARIANT_HATCH_RULES:
        if needle in v:
            return hatch
    return ""  # fp32 / fp16 / baseline / unknown


DEVICE_CLASS_MARKERS_F0101 = {"laptop": "o", "phone": "s"}
DEVICE_CLASS_MARKER_FALLBACK = "^"
F0101_MARKER_SIZE = 110  # matplotlib scatter `s` (points^2); single, consistent.
# Single colour per device class — used by figures that switch to
# device-class encoding (11.1.1, 11.1.3) so legend swatches and data points
# never mismatch when multiple backends share a device class.
DEVICE_CLASS_COLORS = {
    "laptop": "#7E76B0",
    "phone":  "#319974",
    "sbc":    "#D67430",
    "nuc":    "#3A6FA8",
    "jetson": "#666666",
}
DEVICE_CLASS_COLOR_FALLBACK = "#888888"

# Cross-modal encoding for the chapter-wide design-space view: marker shape
# carries modality (so vision / audio / LLM are unambiguous in B&W); facecolor
# carries the device name so M3 / iPhone / Pixel are distinguishable.
MODALITY_MARKERS = {"vision": "o", "audio": "^", "text": "s"}
MODALITY_MARKER_FALLBACK = "P"
DEVICE_NAME_COLORS = {
    "MacBook Air M3":      "#7E76B0",  # purple
    "iPhone 16":           "#D67430",  # orange
    "Google Pixel 10 Pro": "#319974",  # green
}
DEVICE_NAME_COLOR_FALLBACK = "#888888"


def _latency_for_record(rec: dict) -> float | None:
    """Pick the latency the user actually experiences for each modality:
      - text (LLM)   → TTFT (time to first token) — the perceived response time
      - vision/audio → p50 (or mean) — the per-inference time
    Falling back to mean if p50 is missing so older records still plot.

    For LLM records without an explicit `ttft_ms` field, fall back to p50
    if and only if the record is a *prefill-only* conversion — iPhone Core ML's
    `prefill1` / `prefill128` variants measure per-prefill-call latency on
    Xcode Performance Reports, which IS time-to-first-token semantically for
    a prompt of that bucket length. MLX records (Mac) record steady-state
    per-token latency (closer to TPOT), so they are deliberately NOT
    fallback-eligible — returning None there keeps them out of TTFT-axis
    figures rather than mis-crediting per-token speed as TTFT.
    """
    lm = rec.get("latency_ms") or {}
    if rec.get("modality") == "text":
        ttft = lm.get("ttft_ms")
        if ttft is not None:
            return ttft
        if rec.get("backend") == "coreml" and "prefill" in (rec.get("variant") or ""):
            return lm.get("p50") or lm.get("mean")
        return None
    return lm.get("p50") or lm.get("mean")


def _design_space_axes(ax, recs: list[dict], *, marker_by: str = "backend") -> dict:
    """Plot points; return enough info to build a rich legend afterwards.

    `marker_by`:
      - "backend"       — marker shape from BACKEND_MARKERS, size from
                          DEVICE_SIZES, variant hatch on the fill. Used by
                          11.1.3 (budget figure).
      - "device_class"  — marker shape from DEVICE_CLASS_MARKERS_F0101
                          (laptop=circle, phone=square), uniform size, no
                          hatch. Used by 11.1.1 (vision-only legacy) and
                          11.1.3 (budget figure, vision-only).
      - "modality"      — marker shape from MODALITY_MARKERS
                          (vision=circle, audio=triangle, text=square),
                          facecolor from DEVICE_NAME_COLORS (one colour per
                          device.name), uniform size, no hatch. Used by
                          11.1.1's cross-modal expansion. The per-modality
                          latency definition matters here: LLM records use
                          TTFT (perceived response), others use p50/mean.

    Power records often live under a suffixed compute_units
    (`*_power_30s`, `*_sustained_300s`) so we pre-build a
    (model, variant, backend, device.name, base_cu) → power lookup that ignores
    the suffix, then attach to latency-bearing records.
    """
    def _strip_suffix(cu: str | None) -> str | None:
        if cu is None: return None
        for sfx in ("_sustained_300s", "_sustained_60s", "_power_30s", "_power"):
            if cu.endswith(sfx): return cu[: -len(sfx)]
        return cu
    power_index: dict[tuple, dict] = {}
    for r in recs:
        pw = r.get("power_mw") or {}
        if pw.get("source") and pw.get("mean") is not None:
            key = (r.get("model"), r.get("variant"), r.get("backend"),
                   (r.get("device") or {}).get("name"), _strip_suffix(r.get("compute_units")))
            power_index[key] = pw

    backends_seen: set[str] = set()
    classes_seen: set[str] = set()
    combos_seen: set[tuple[str, str]] = set()  # all (backend, device-class) present
    plotted_combos: set[tuple[str, str]] = set()  # only combos drawn on the main axes (with power)
    plotted_points: list[dict] = []  # one entry per scatter point on the main axes
    device_names_by_class: dict[str, set[str]] = {}
    modalities_seen: set[str] = set()  # cross-modal mode
    device_names_seen: set[str] = set()  # cross-modal mode
    null_power = []
    seen = set()  # avoid double-plotting when both lat-only and pw-only records exist
    for r in recs:
        if marker_by == "modality":
            lat = _latency_for_record(r)
        else:
            lat = (r.get("latency_ms") or {}).get("p50") or (r.get("latency_ms") or {}).get("mean")
        if lat is None: continue
        klass = (r.get("device") or {}).get("class") or "laptop"
        backend = r.get("backend") or "prepost"
        variant = r.get("variant") or ""
        modality = r.get("modality") or "vision"
        dname = (r.get("device") or {}).get("name") or ""
        if marker_by == "device_class":
            marker = DEVICE_CLASS_MARKERS_F0101.get(klass, DEVICE_CLASS_MARKER_FALLBACK)
            size = F0101_MARKER_SIZE
            hatch = ""  # Drop hatches per user feedback (simpler read).
            color = DEVICE_CLASS_COLORS.get(klass, DEVICE_CLASS_COLOR_FALLBACK)
        elif marker_by == "modality":
            marker = MODALITY_MARKERS.get(modality, MODALITY_MARKER_FALLBACK)
            size = F0101_MARKER_SIZE
            hatch = ""
            color = DEVICE_NAME_COLORS.get(dname, DEVICE_NAME_COLOR_FALLBACK)
        else:
            marker = BACKEND_MARKERS.get(backend, BACKEND_MARKER_FALLBACK)
            size = DEVICE_SIZES.get(klass, 80)
            hatch = _variant_hatch(variant)
            color = _backend_color(backend)
        cu_base = _strip_suffix(r.get("compute_units"))
        # In cross-modal mode the prompt_length matters for dedup (an LLM
        # record can have the same model/variant/backend/device/cu base but
        # differ on prompt length, which we want to surface as separate points).
        if marker_by == "modality":
            prompt = (r.get("throughput") or {}).get("prompt_length")
            key = (r.get("model"), variant, backend, dname, cu_base, prompt)
        else:
            key = (r.get("model"), variant, backend, dname, cu_base)
        if key in seen: continue
        seen.add(key)
        backends_seen.add(backend)
        classes_seen.add(klass)
        modalities_seen.add(modality)
        device_names_seen.add(dname)
        combos_seen.add((backend, klass))
        pw = (r.get("power_mw") or {}) or power_index.get(key[:5]) or {}
        if pw.get("mean") is not None:
            ax.scatter([lat], [pw["mean"]], marker=marker, s=size,
                       facecolor=color, hatch=hatch,
                       edgecolor="black", linewidths=0.7, zorder=3)
            plotted_combos.add((backend, klass))
            device_names_by_class.setdefault(klass, set()).add(dname)
            plotted_points.append({
                "lat": lat, "power": pw["mean"],
                "klass": klass, "backend": backend, "modality": modality,
                "variant": variant, "device_name": dname,
                "prompt_length": (r.get("throughput") or {}).get("prompt_length"),
            })
        else:
            null_power.append((lat, klass, backend, hatch))
    return {
        "backends": sorted(backends_seen),
        "classes": sorted(classes_seen),
        "combos":   sorted(combos_seen),
        "modalities": sorted(modalities_seen),
        "device_names": sorted(device_names_seen),
        "plotted_combos": sorted(plotted_combos),
        "plotted_points": plotted_points,
        "device_names_by_class": {k: sorted(v) for k, v in device_names_by_class.items()},
        "null_power": null_power,
    }


def _annotate_points_no_overlap(ax, points: list[dict], label_fn) -> None:
    """Place a short text label next to each scatter point. Tries a fixed
    set of candidate offsets and accepts the first whose pixel bbox does
    not overlap any previously-placed label bbox. Falls back to the first
    candidate if all overlap. Pure heuristic — works well for ≤ ~10 points;
    not a general label-placement solver.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    candidates = [
        (7, 4), (7, -4), (-7, 4), (-7, -4),
        (10, 10), (10, -10), (-10, 10), (-10, -10),
        (14, 0), (-14, 0), (0, 14), (0, -14),
        (16, 12), (16, -12), (-16, 12), (-16, -12),
    ]
    placed: list = []
    for pt in points:
        lbl = label_fn(pt)
        if not lbl: continue
        chosen_txt = None
        for dx, dy in candidates:
            txt = ax.annotate(
                lbl, xy=(pt["lat"], pt["power"]),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=7, color="#333333",
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
            )
            fig.canvas.draw()
            bbox = txt.get_window_extent(renderer=renderer)
            if not any(bbox.overlaps(b) for b in placed):
                chosen_txt = (txt, bbox); break
            txt.remove()
        if chosen_txt is None:
            dx, dy = candidates[0]
            txt = ax.annotate(
                lbl, xy=(pt["lat"], pt["power"]),
                xytext=(dx, dy), textcoords="offset points",
                fontsize=7, color="#333333", ha="left", va="bottom",
            )
            fig.canvas.draw()
            chosen_txt = (txt, txt.get_window_extent(renderer=renderer))
        placed.append(chosen_txt[1])


def _short_variant_label(variant: str | None) -> str:
    """Compact label for in-figure annotation. Strips backend prefix and
    collapses verbose suffixes."""
    if not variant: return ""
    v = variant.lower()
    if v.startswith(("coreml_", "tflite_", "mlx_", "mps_", "litertlm_")):
        v = v.split("_", 1)[1]
    short = {
        "fp16": "fp16", "fp32": "fp32",
        "int8": "int8",
        "int8_linear": "int8-lin",
        "int8_weight_only": "int8-wo",
        "palettize_4bit": "4-bit pal",
        "palettize_6bit": "6-bit pal",
        "dynrange": "dyn-range", "dyn_range": "dyn-range",
        "int16x8": "int16×8",
    }
    return short.get(v, v)


# Friendly OS hint by device class — used by the budget figure legend so a
# reader can tell iPhone (iOS) from Pixel (Android) without consulting the
# caption. Mirrors the data: phone class is currently Android-only here.
DEVICE_CLASS_OS_HINT = {"phone": "Android"}


def _build_cross_modal_legend_handles(plotted_combos: list[tuple[str, str]]) -> list:
    """One legend row per actually-rendered (modality, device.name) combo.

    Each swatch uses the same marker shape AND facecolor that the
    corresponding scatter point uses — so the reader matches "green
    square in legend" ↔ "green square on chart" without mentally
    composing two separate encodings.
    """
    from matplotlib.lines import Line2D
    import math
    ms = math.sqrt(F0101_MARKER_SIZE)
    MOD_LABEL = {"vision": "vision (p50)", "audio": "audio (p50)", "text": "LLM (TTFT)"}
    DEV_TAG = {
        "MacBook Air M3": "M3",
        "iPhone 16": "iPhone 16",
        "Google Pixel 10 Pro": "Pixel 10 Pro",
    }
    handles: list = []
    for modality, dname in plotted_combos:
        handles.append(Line2D(
            [0], [0], linestyle="none",
            marker=MODALITY_MARKERS.get(modality, MODALITY_MARKER_FALLBACK),
            markersize=ms,
            markerfacecolor=DEVICE_NAME_COLORS.get(dname, DEVICE_NAME_COLOR_FALLBACK),
            markeredgecolor="black", markeredgewidth=0.8,
            label=f"{DEV_TAG.get(dname, dname)} · {MOD_LABEL.get(modality, modality)}",
        ))
    return handles


def _build_device_class_legend_handles(plotted_combos: list[tuple[str, str]],
                                       device_names_by_class: dict[str, list[str]] | None = None) -> list:
    """Simple Line2D legend: one row per device class actually plotted.
    Marker shape and fill colour both come from the DEVICE_CLASS_* tables,
    matching exactly what the scatter draws when `marker_by='device_class'`.
    markersize matches the scatter `s` via sqrt() so legend swatches look the
    same size as the data points.

    If a single device.name is present for a class, label the row with that
    exact device (e.g. "Pixel 10 Pro (Android)") so iPhone-vs-Android is
    never ambiguous. Otherwise fall back to the generic class label.
    """
    from matplotlib.lines import Line2D
    import math
    ms = math.sqrt(F0101_MARKER_SIZE)
    classes = sorted({klass for _backend, klass in plotted_combos})
    handles: list = []
    for klass in classes:
        names = (device_names_by_class or {}).get(klass) or []
        if len(names) == 1 and names[0]:
            os_hint = DEVICE_CLASS_OS_HINT.get(klass)
            label = f"{names[0]} ({os_hint})" if os_hint else names[0]
        else:
            label = DEVICE_CLASS_LABELS.get(klass, klass)
        handles.append(Line2D(
            [0], [0], linestyle="none",
            marker=DEVICE_CLASS_MARKERS_F0101.get(klass, DEVICE_CLASS_MARKER_FALLBACK),
            markersize=ms,
            markerfacecolor=DEVICE_CLASS_COLORS.get(klass, DEVICE_CLASS_COLOR_FALLBACK),
            markeredgecolor="black", markeredgewidth=0.8,
            label=label,
        ))
    return handles


def _variant_family(variant: str | None) -> str | None:
    """Short legend label for a variant; collapses backend-specific names."""
    if not variant: return None
    v = variant.lower()
    if "int16x8" in v: return "INT16×8"
    if "int4" in v or "palettize" in v: return "INT4 / palettize"
    if "int8" in v: return "INT8"
    if "dyn" in v: return "Dyn-range"
    if "fp16" in v or "fp32" in v: return "Float baseline"
    return None


class _LegendSpec:
    """Plain holder passed as a legend handle; the custom handler reads off it."""
    __slots__ = ("marker", "color", "hatch", "edgewidth", "size_frac", "label")
    def __init__(self, marker, color, hatch, edgewidth, size_frac, label):
        self.marker = marker
        self.color = color
        self.hatch = hatch
        self.edgewidth = edgewidth
        self.size_frac = size_frac
        self.label = label
    def get_label(self): return self.label


class _HatchedMarkerHandler:
    """Custom legend handler that renders the actual marker shape as a
    `PathPatch`, with hatch fill — which Line2D markers do not support.
    Used so the legend swatches for same-backend-different-device combos can
    be told apart by *both* size and hatch, exactly mirroring how the points
    look on the plot.
    """
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        from matplotlib.markers import MarkerStyle
        from matplotlib.patches import PathPatch
        from matplotlib.transforms import Affine2D
        spec: _LegendSpec = orig_handle
        cx = handlebox.xdescent + handlebox.width / 2
        cy = handlebox.ydescent + handlebox.height / 2
        scale = min(handlebox.width, handlebox.height) * spec.size_frac
        ms = MarkerStyle(spec.marker)
        path = ms.get_path().transformed(ms.get_transform())
        tx = Affine2D().scale(scale).translate(cx, cy)
        patch = PathPatch(path, transform=tx + handlebox.get_transform(),
                          facecolor=spec.color, hatch=spec.hatch,
                          edgecolor="black", linewidth=spec.edgewidth)
        handlebox.add_artist(patch)
        return patch


# Representative hatch picked per device class for the *legend swatch* only.
# Plot points use _variant_hatch() (per-variant) as before; in the legend we
# pick one hatch that visually separates same-backend rows at a glance:
#   laptop → no hatch (solid)
#   phone  → dense crosshatch
# These were chosen so the laptop/phone distinction is readable in B&W.
LEGEND_DEVICE_HATCH = {
    "laptop": "",
    "phone":  "xxxx",
    "sbc":    "////",
    "nuc":    "....",
    "jetson": "\\\\\\\\",
}


def _build_legend_handles(info: dict, *, include_variants: bool = True) -> list:
    """Single unified legend, B&W-primary.

    One row per (backend, device-class) combination present in the data.
    Each swatch is the actual marker shape (backend) at a size proportional
    to the device class, filled with the backend's decorative colour, and
    overlaid with a device-class hatch so two rows with the same backend
    but different device class look unmistakably different — in colour *and*
    in greyscale.

    `include_variants` is retained for caller compatibility but is ignored —
    variant differences are conveyed by hatch on the points themselves and
    explained in the figure caption.
    """
    handles: list = []
    combos = info.get("combos") or [(b, k) for b in info["backends"] for k in info["classes"]]
    LEGEND_SIZE_FRAC = {"laptop": 0.55, "phone": 0.95, "sbc": 0.7,
                         "nuc": 0.75, "jetson": 0.85}
    LEGEND_EDGEW     = {"laptop": 0.7,  "phone": 1.4,  "sbc": 1.0,
                         "nuc": 1.0,    "jetson": 1.2}
    for backend, klass in combos:
        spec = _LegendSpec(
            marker    = BACKEND_MARKERS.get(backend, BACKEND_MARKER_FALLBACK),
            color     = _backend_color(backend),
            hatch     = LEGEND_DEVICE_HATCH.get(klass, ""),
            edgewidth = LEGEND_EDGEW.get(klass, 0.7),
            size_frac = LEGEND_SIZE_FRAC.get(klass, 0.7),
            label     = f"{backend} · {DEVICE_CLASS_LABELS.get(klass, klass)}",
        )
        handles.append(spec)
    return handles


# Public so call sites can wire it into legend(handler_map=...)
LEGEND_HANDLER_MAP = {_LegendSpec: _HatchedMarkerHandler()}


def _draw_null_power_strip(ax_null, info, ax_main, *, marker_by: str = "backend"):
    null_power = info["null_power"]
    if not null_power:
        ax_null.text(0.5, 0.5, "(no null-power records)", transform=ax_null.transAxes,
                     ha="center", va="center", fontsize=7, color="#888888")
        ax_null.set_yticks([])
        return
    ys = [0] * len(null_power)
    for (lat, klass, backend, hatch), y in zip(null_power, ys):
        if marker_by == "device_class":
            marker = DEVICE_CLASS_MARKERS_F0101.get(klass, DEVICE_CLASS_MARKER_FALLBACK)
            size = F0101_MARKER_SIZE
            hatch_use = ""
            face = DEVICE_CLASS_COLORS.get(klass, DEVICE_CLASS_COLOR_FALLBACK)
        else:
            marker = BACKEND_MARKERS.get(backend, BACKEND_MARKER_FALLBACK)
            size = max(45, DEVICE_SIZES.get(klass, 80) * 0.55)
            hatch_use = hatch
            face = _backend_color(backend)
        ax_null.scatter([lat], [y],
                        marker=marker,
                        facecolor=face, hatch=hatch_use,
                        edgecolor="black", linewidths=0.7,
                        s=size, zorder=3)
    ax_null.set_yticks([0]); ax_null.set_yticklabels(["power not\nmeasured"])
    ax_null.set_xlim(ax_main.get_xlim())


def mode_design_space(args: argparse.Namespace) -> int:
    """11.1.1 — cross-modal design space (latency × power across vision, audio, LLM).

    Every record with both a latency reading and a paired power reading is
    plotted. Modality drives marker shape (vision=circle, audio=triangle,
    LLM=square); device.name drives facecolour. Latency is per-modality:
    p50 for vision/audio, TTFT for LLMs (the perceived response time —
    LLM p50 across full prefill+generation would dwarf everything else
    and break the cross-workload comparison). Records without a paired
    power measurement (the iPhone Core ML reports, the Mac vision +
    audio records that lack the `*_power_30s` sustained variant) are
    excluded; their count is logged.
    """
    logger = _setup_logger("design-space")
    recs = _load_records()
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    info = _design_space_axes(ax, recs, marker_by="modality")
    ax.set_xlabel("Latency to first useful output (ms) — p50 for vision/audio, TTFT for LLM")
    ax.set_ylabel("Mean power (mW)")
    ax.set_title("11.1.1 — Cross-modal design space: latency × power on shipped devices")
    ax.set_xscale("log")
    ax.grid(True, axis="y", linestyle=":", color="#cccccc", linewidth=0.5, zorder=0)

    # One cluster annotation per (device.name, modality) group — placed at
    # the geometric centroid (log-x mean) and nudged up-right. LLM clusters
    # collapse the prompt-sweep into a single "prompts (32, 256, 1024)" tag
    # rather than three overlapping "prompt=N" labels on near-adjacent dots.
    import math
    clusters: dict[tuple[str, str], list[dict]] = {}
    for pt in info["plotted_points"]:
        clusters.setdefault((pt["device_name"], pt["modality"]), []).append(pt)
    DEVICE_TAG = {"MacBook Air M3": "M3", "iPhone 16": "iPhone", "Google Pixel 10 Pro": "Pixel"}
    MOD_TAG = {
        "vision": "vision (EffNet-Lite0)",
        "audio":  "audio (Whisper-tiny)",
        "text":   "LLM (Llama-3.2-1B)",
    }
    for (dname, modality), pts in clusters.items():
        n = len(pts)
        x_log = sum(math.log10(p["lat"]) for p in pts) / n
        y = sum(p["power"] for p in pts) / n
        if modality == "text":
            prompts = sorted(
                {p.get("prompt_length") for p in pts if p.get("prompt_length") is not None}
            )
            if prompts:
                joined = ", ".join(str(int(p)) for p in prompts)
                suffix = f" · prompts ({joined})"
            else:
                suffix = ""
        else:
            suffix = f" · {n} variants" if n > 1 else ""
        # LLM clusters sit at the rightmost end of the log-x axis, so a
        # rightward-flowing label runs off the chart. Anchor those above
        # the centroid (ha=center, larger up-offset). Other clusters keep
        # the original up-right placement.
        if modality == "text":
            ax.annotate(f"{DEVICE_TAG.get(dname, dname)} {MOD_TAG.get(modality, modality)}{suffix}",
                        xy=(10 ** x_log, y), xytext=(0, 14),
                        textcoords="offset points",
                        fontsize=7, color="#333333",
                        ha="center", va="bottom")
        else:
            ax.annotate(f"{DEVICE_TAG.get(dname, dname)} {MOD_TAG.get(modality, modality)}{suffix}",
                        xy=(10 ** x_log, y), xytext=(12, 12),
                        textcoords="offset points",
                        fontsize=7, color="#333333",
                        ha="left", va="bottom")

    # One legend row per actually-rendered (modality, device.name) combo —
    # matches the chart's encoding exactly so "find this marker in the
    # legend" works without any mental composition.
    DEV_ORDER = ["MacBook Air M3", "iPhone 16", "Google Pixel 10 Pro"]
    MOD_ORDER = ["vision", "audio", "text"]
    raw_combos = {(p["modality"], p["device_name"]) for p in info["plotted_points"]}
    plotted_combos = sorted(
        raw_combos,
        key=lambda mc: (DEV_ORDER.index(mc[1]) if mc[1] in DEV_ORDER else 99,
                        MOD_ORDER.index(mc[0]) if mc[0] in MOD_ORDER else 99),
    )
    handles = _build_cross_modal_legend_handles(plotted_combos)
    if handles:
        ax.legend(handles=handles, fontsize=8,
                  loc="center left", bbox_to_anchor=(1.02, 0.5),
                  framealpha=0.95, edgecolor="#cccccc",
                  handletextpad=0.8, borderpad=0.6, labelspacing=1.0, ncol=1)
    else:
        ax.text(0.5, 0.5, "no records with latency + power",
                transform=ax.transAxes, ha="center", va="center", color="#888888")
    fig.tight_layout()
    _save_pair(fig, "CH11_F0101_Kalyanarangan", "ch11_1"); plt.close(fig)
    logger.info("wrote 11.1.1 (cross-modal); plotted_points=%d modalities=%s devices=%s (null-power omitted=%d)",
                len(info["plotted_points"]),
                info.get("modalities"), info.get("device_names"),
                len(info["null_power"]))
    return 0


# --------------------------------------------------------------------------- #
# Sustained curve (11.1.2)                                                    #
# --------------------------------------------------------------------------- #
def mode_sustained_curve(args: argparse.Namespace) -> int:
    """11.1.2 — thermal retention bars.

    The chapter's §11.1.2 thesis is "does the device hold its first-30 s
    throughput across a longer window, or does the SoC knee under thermal
    pressure?". The previous line-chart formulation mixed vision
    samples/s with audio segments/s with LLM tokens/s on the same y axis
    — three different units — which collapsed every non-vision line to a
    flat near-zero and was unreadable.

    Reframe: one bar per sustained record, height = throughput retention
    `100 · last_30s / first_30s`. Retention is unit-free, so vision /
    audio / LLM coexist honestly. A horizontal reference line at 100 %
    marks the no-throttle baseline; bars below the line are the
    throttling configs and the bar height tells the reader how much
    survived the window.
    """
    logger = _setup_logger("sustained-curve")
    raw = [r for r in _load_records() if r.get("sustained")]
    seen: set[tuple] = set()
    recs: list[dict] = []
    for r in raw:
        key = (
            (r.get("device") or {}).get("name"),
            r.get("model"),
            r.get("variant"),
            r.get("compute_units"),
            (r.get("sustained") or {}).get("window_s"),
        )
        if key in seen: continue
        seen.add(key)
        recs.append(r)

    rows: list[dict] = []
    for r in recs:
        sus = r["sustained"]
        first = sus.get("throughput_first_30s")
        last  = sus.get("throughput_last_30s")
        if not first or not last: continue
        retention = 100.0 * last / first
        cu = r.get("compute_units") or "-"
        for sfx in ("_sustained_300s", "_sustained_60s"):
            if cu.endswith(sfx):
                cu = cu[: -len(sfx)]
        dname = (r.get("device") or {}).get("name") or "?"
        rows.append({
            "device_name": dname,
            "model": r.get("model"),
            "modality": r.get("modality") or "vision",
            "variant": _short_variant_label(r.get("variant")) or r.get("variant"),
            "cu": cu,
            "window_s": sus.get("window_s") or 300,
            "retention": retention,
            "first": first, "last": last,
        })

    # Stable ordering: device then modality then retention asc (worst-first
    # within each device-modality group reads as a thermal severity ranking).
    DEV_ORDER = ["MacBook Air M3", "iPhone 16", "Google Pixel 10 Pro"]
    MOD_ORDER = ["vision", "audio", "text"]
    rows.sort(key=lambda x: (
        DEV_ORDER.index(x["device_name"]) if x["device_name"] in DEV_ORDER else 99,
        MOD_ORDER.index(x["modality"]) if x["modality"] in MOD_ORDER else 99,
        x["retention"],
    ))

    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.45 * len(rows) + 1.5)))
    if rows:
        y = np.arange(len(rows))
        # Colour per device (matches F0101 / F0104), hatch per modality.
        MOD_HATCH = {"vision": "", "audio": "...", "text": "xxx"}
        bars = ax.barh(
            y, [r["retention"] for r in rows],
            color=[DEVICE_NAME_COLORS.get(r["device_name"], DEVICE_NAME_COLOR_FALLBACK) for r in rows],
            hatch=[MOD_HATCH.get(r["modality"], "") for r in rows],
            edgecolor="black", linewidth=0.5, zorder=3,
        )
        ax.axvline(100.0, color="#333333", linewidth=1.0, linestyle="--", zorder=2,
                   label="no-throttle baseline (100 %)")
        # Per-bar text: retention % + the absolute first→last (so the reader
        # can see whether the bar's "100 %" sits on real work or a sluggish
        # config that didn't have anywhere to fall).
        for bar, r in zip(bars, rows):
            retention = r["retention"]
            unit = {"vision": "samples/s", "audio": "segments/s", "text": "tok/s"}.get(r["modality"], "ops/s")
            x_text = max(retention, 100.0) + 2.0
            # Split into two lines so the throughput pair doesn't run past
            # the right xlim (the 154 % bar is the widest case).
            ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                    f"{retention:.1f} %\n{r['first']:.1f} → {r['last']:.1f} {unit}",
                    va="center", fontsize=7, color="#222222")

        ax.set_yticks(y)
        DEV_TAG = {"MacBook Air M3": "M3", "iPhone 16": "iPhone",
                   "Google Pixel 10 Pro": "Pixel"}
        MOD_TAG = {"vision": "vision", "audio": "audio", "text": "LLM"}
        ax.set_yticklabels(
            [f"{DEV_TAG.get(r['device_name'], r['device_name'])} · {MOD_TAG.get(r['modality'], r['modality'])} · "
             f"{r['variant']} · {r['cu']} ({r['window_s']}s)" for r in rows],
            fontsize=7,
        )
        ax.invert_yaxis()  # first row at top
        # Leave headroom on the right for annotation text — bars hover near
        # 100% so we want the axis to start a bit below the lowest retention
        # and stretch out enough that the descriptor text fits.
        min_ret = min(r["retention"] for r in rows)
        ax.set_xlim(left=max(0, min_ret - 5), right=180)
        ax.set_xlabel("Throughput retention (% of first 30 s, sustained over window)")
        ax.set_title("11.1.2 — Thermal retention across modalities × devices "
                     "(100 % = no throttle; lower bars throttled)")
        ax.grid(True, axis="x", linestyle=":", color="#cccccc", linewidth=0.5, zorder=0)

        # Compact legend for the device colors + the baseline.
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_handles = []
        for dname in DEV_ORDER:
            if any(r["device_name"] == dname for r in rows):
                legend_handles.append(Patch(
                    facecolor=DEVICE_NAME_COLORS.get(dname, DEVICE_NAME_COLOR_FALLBACK),
                    edgecolor="black", linewidth=0.6,
                    label=DEV_TAG.get(dname, dname),
                ))
        for mod in MOD_ORDER:
            if any(r["modality"] == mod for r in rows):
                legend_handles.append(Patch(
                    facecolor="#dddddd", edgecolor="black", linewidth=0.6,
                    hatch=MOD_HATCH.get(mod, ""),
                    label=MOD_TAG.get(mod, mod),
                ))
        legend_handles.append(Line2D([0], [0], linestyle="--", color="#333333",
                                     linewidth=1.0, label="100 % baseline"))
        ax.legend(handles=legend_handles, fontsize=7,
                  loc="lower right", framealpha=0.95, edgecolor="#cccccc",
                  ncol=2, columnspacing=1.0, handletextpad=0.6, borderpad=0.5)
    else:
        ax.text(0.5, 0.5, "data not available", transform=ax.transAxes, ha="center", va="center")
    fig.tight_layout(); _save_pair(fig, "CH11_F0102_Kalyanarangan", "ch11_1"); plt.close(fig)
    logger.info("wrote 11.1.2 (retention bars) with %d sustained recs (dedup of %d)",
                len(rows), len(raw))
    return 0


# --------------------------------------------------------------------------- #
# Budget calculator                                                           #
# --------------------------------------------------------------------------- #
def mode_budget_calculator(args: argparse.Namespace) -> int:
    logger = _setup_logger("budget-calculator")
    recs = _load_records()
    out = []
    for r in recs:
        lat = (r.get("latency_ms") or {}).get("p95")
        pw = (r.get("power_mw") or {}).get("mean")
        if lat is None: continue
        if args.p95_ms is not None and lat > args.p95_ms: continue
        if args.power_mw is not None and (pw is None or pw > args.power_mw): continue
        out.append({"model": r.get("model"), "variant": r.get("variant"),
                    "backend": r.get("backend"),
                    "device": (r.get("device") or {}).get("name"),
                    "compute_units": r.get("compute_units"),
                    "p95_ms": lat, "power_mw_mean": pw})
    print(json.dumps({"constraints": {"p95_ms": args.p95_ms, "power_mw": args.power_mw},
                      "matches": out}, indent=2))
    logger.info("budget: %d matches under p95<=%s power_mean<=%s", len(out), args.p95_ms, args.power_mw)
    return 0


# --------------------------------------------------------------------------- #
# Budget figure (11.1.3)                                                      #
# --------------------------------------------------------------------------- #
def mode_budget_figure(args: argparse.Namespace) -> int:
    logger = _setup_logger("budget-figure")
    recs = [r for r in _load_records() if r.get("model") == "efficientnet_lite0"]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    info = _design_space_axes(ax, recs, marker_by="device_class")
    p95 = args.p95_ms if args.p95_ms is not None else 33
    pw = args.power_mw if args.power_mw is not None else 5000

    # Tinted feasible region — light Manning-green so it survives B&W as a
    # light-grey wash; black hatched border highlights the boundary.
    from matplotlib.patches import Rectangle
    xlim = ax.get_xlim()
    if xlim[0] <= 0:  # log scale safety
        xlim = (max(0.01, xlim[0]), xlim[1])
    x0 = xlim[0]
    rect_fill = Rectangle((x0, 0), p95 - x0, pw,
                          facecolor="#319974", alpha=0.12, edgecolor="none",
                          zorder=0)
    rect_edge = Rectangle((x0, 0), p95 - x0, pw,
                          fill=False, hatch="//", edgecolor="#319974",
                          linewidth=0.8, zorder=1)
    ax.add_patch(rect_fill)
    ax.add_patch(rect_edge)
    ax.axvline(p95, color="#319974", linewidth=0.8, linestyle=":", zorder=1)
    ax.axhline(pw, color="#319974", linewidth=0.8, linestyle=":", zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("p95 latency (ms)")
    ax.set_ylabel("Mean power (mW)")
    ax.set_title(f"11.1.3 — EfficientNet-Lite0 budget-feasible region (p95 ≤ {p95} ms, mean power ≤ {pw} mW)")
    ax.grid(True, axis="y", linestyle=":", color="#cccccc", linewidth=0.5, zorder=0)

    # Annotate each plotted point with a short variant label so multiple
    # MacBook dots are not ambiguous (e.g. "fp16" vs "4-bit pal").
    _annotate_points_no_overlap(ax, info["plotted_points"],
                                lambda pt: _short_variant_label(pt["variant"]))

    region_handle = Rectangle((0, 0), 1, 1,
                              facecolor="#319974", alpha=0.25, edgecolor="#319974",
                              hatch="//", linewidth=0.8,
                              label=f"budget-feasible (≤{p95} ms, ≤{pw} mW)")
    handles = [region_handle] + _build_device_class_legend_handles(
        info["plotted_combos"], info["device_names_by_class"])
    ax.legend(handles=handles, fontsize=8, loc="best",
              framealpha=0.95, edgecolor="#cccccc",
              handletextpad=0.8, borderpad=0.6)
    fig.tight_layout()
    _save_pair(fig, "CH11_F0103_Kalyanarangan", "ch11_1"); plt.close(fig)
    logger.info("wrote 11.1.3 budget figure (legend entries=%d, null-power omitted=%d)",
                len(handles), len(info["null_power"]))
    return 0


# --------------------------------------------------------------------------- #
# Device latency (11.1.4) — latency-only cross-device view, no power axis      #
# --------------------------------------------------------------------------- #
F0104_DEVICE_MARKERS = {
    "MacBook Air M3":       "o",  # circle (Apple laptop)
    "iPhone 16":            "s",  # square (Apple phone)
    "Google Pixel 10 Pro":  "D",  # diamond (Android phone) — B&W-decodable
}
F0104_DEVICE_COLORS = {
    "MacBook Air M3":       "#7E76B0",  # purple (matches F0101 laptop)
    "iPhone 16":            "#D67430",  # orange (distinct from Pixel)
    "Google Pixel 10 Pro":  "#319974",  # green (matches F0101 phone)
}


def mode_device_latency(args: argparse.Namespace) -> int:
    """11.1.4 — cross-modal, cross-device latency comparison.

    For each (modality, device) pair we plot one canonical point: the
    fastest variant the device actually shipped with on its preferred
    compute unit. Latency is per-modality (TTFT for LLM, p50 for
    vision/audio), so the user is reading "time to first useful output"
    on a single log axis spanning ~0.5 ms (M3 ANE vision) to ~430 ms
    (Pixel LiteRT-LM LLM TTFT).

    Why per-modality canonical configs rather than one global rule: each
    modality has its own platform-native fast path. Vision lights up
    iPhone/Mac ANE only when computeUnits=`all` exposes it; LLM on
    Apple goes through MLX (Mac) or Core ML prefill (iPhone); Android
    routes all three through TFLite XNNPACK except LLM which uses
    LiteRT-LM. Forcing a single backend filter would mis-credit the
    devices.
    """
    logger = _setup_logger("device-latency")
    recs = _load_records()

    # Canonical pick rule per (modality, device.name). The match function
    # returns True for records that count as canonical for that pair; we
    # then take the lowest-latency match. For LLM we further pin to the
    # smallest prompt length (32) so cross-device comparison is
    # apples-to-apples on the TTFT metric.
    def _is_canonical(r: dict, modality: str, dname: str) -> bool:
        backend = r.get("backend")
        cu = r.get("compute_units") or ""
        # Skip the suffixed sustained / power records — they're snapshots
        # of the throughput / power side of the same underlying variant.
        if cu.endswith(("_sustained_300s", "_sustained_60s", "_power_30s", "_power")):
            return False
        if r.get("modality") != modality:
            return False
        if dname == "MacBook Air M3":
            if modality in ("vision", "audio"):
                return backend == "coreml" and cu == "all"
            if modality == "text":
                return backend == "mlx" and (r.get("throughput") or {}).get("prompt_length") == 32
        if dname == "iPhone 16":
            if modality in ("vision", "audio"):
                return backend == "coreml" and cu == "all"
            if modality == "text":
                # iPhone has prefill-only conversions; pick the shortest prefill
                # (`prefill1`) so TTFT is a per-prefill-call latency comparable
                # to MLX/LiteRT-LM TTFT.
                return backend == "coreml" and "prefill1_" in (r.get("variant") or "")
        if dname == "Google Pixel 10 Pro":
            if modality in ("vision", "audio"):
                return backend == "tflite" and cu == "xnnpack_4t"
            if modality == "text":
                return (backend == "litertlm"
                        and (r.get("throughput") or {}).get("prompt_length") == 32
                        and cu == "gpu")
        return False

    modalities_order = ["vision", "audio", "text"]
    devices_order = ["MacBook Air M3", "iPhone 16", "Google Pixel 10 Pro"]
    canon: list[dict] = []
    for modality in modalities_order:
        for dname in devices_order:
            best = None
            for r in recs:
                if not _is_canonical(r, modality, dname):
                    continue
                lat = _latency_for_record(r)
                if lat is None: continue
                if best is None or lat < best["lat"]:
                    best = {
                        "modality": modality, "device_name": dname,
                        "variant": r.get("variant") or "",
                        "lat": lat,
                    }
            if best: canon.append(best)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if not canon:
        ax.text(0.5, 0.5, "no canonical-config records found",
                transform=ax.transAxes, ha="center", va="center", color="#888888")
        fig.tight_layout()
        _save_pair(fig, "CH11_F0104_Kalyanarangan", "ch11_1"); plt.close(fig)
        logger.info("wrote 11.1.4 (empty)")
        return 0

    x_pos = {m: i for i, m in enumerate(modalities_order)}
    MOD_X_LABEL = {"vision": "Vision\n(EfficientNet-Lite0, p50)",
                    "audio":  "Audio\n(Whisper-tiny encoder, p50)",
                    "text":   "LLM\n(Llama-3.2-1B, TTFT @ prompt=32)"}
    import math
    ms_legend = math.sqrt(F0101_MARKER_SIZE)
    width = 0.22
    for dname in devices_order:
        pts = [p for p in canon if p["device_name"] == dname]
        if not pts: continue
        # Cluster device markers within each modality bin so they don't overlap.
        i_dev = devices_order.index(dname) - 1  # -1 / 0 / +1
        xs = [x_pos[p["modality"]] + i_dev * width for p in pts]
        ys = [p["lat"] for p in pts]
        ax.scatter(xs, ys,
                   marker=F0104_DEVICE_MARKERS.get(dname, "P"),
                   s=F0101_MARKER_SIZE,
                   facecolor=F0104_DEVICE_COLORS.get(dname, "#888888"),
                   edgecolor="black", linewidths=0.7, zorder=3,
                   label=dname)
        # Per-point annotation: short variant label so the reader can see
        # which quantization config is being credited to each device.
        # In the LLM column the rightmost marker (iPhone, +width offset)
        # would push its label past the right xlim — flip those labels
        # to the LEFT of the marker (ha=right, negative x offset).
        for x, y, p in zip(xs, ys, pts):
            if p["modality"] == "text":
                ax.annotate(_short_variant_label(p["variant"]),
                            xy=(x, y), xytext=(-7, 4), textcoords="offset points",
                            fontsize=7, color="#333333", ha="right")
            else:
                ax.annotate(_short_variant_label(p["variant"]),
                            xy=(x, y), xytext=(7, 4), textcoords="offset points",
                            fontsize=7, color="#333333")

    ax.set_xticks([x_pos[m] for m in modalities_order])
    ax.set_xticklabels([MOD_X_LABEL[m] for m in modalities_order], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Latency to first useful output (ms, log)")
    ax.set_title("11.1.4 — Cross-modal latency on shipped devices (canonical per-modality config)")
    ax.grid(True, axis="y", which="both", linestyle=":", color="#cccccc", linewidth=0.5, zorder=0)

    from matplotlib.lines import Line2D
    handles = []
    for dname in devices_order:
        if not any(p["device_name"] == dname for p in canon): continue
        handles.append(Line2D(
            [0], [0], linestyle="none",
            marker=F0104_DEVICE_MARKERS.get(dname, "P"),
            markersize=ms_legend,
            markerfacecolor=F0104_DEVICE_COLORS.get(dname, "#888888"),
            markeredgecolor="black", markeredgewidth=0.8,
            label=dname,
        ))
    ax.legend(handles=handles, fontsize=8, loc="best",
              framealpha=0.95, edgecolor="#cccccc",
              handletextpad=0.8, borderpad=0.6, labelspacing=1.0)
    fig.tight_layout()
    _save_pair(fig, "CH11_F0104_Kalyanarangan", "ch11_1"); plt.close(fig)
    logger.info("wrote 11.1.4 (cross-modal); %d points across %d devices × %d modalities",
                len(canon),
                len({p["device_name"] for p in canon}),
                len({p["modality"] for p in canon}))
    return 0


def mode_figures(args: argparse.Namespace) -> int:
    rc = 0
    for fn in [mode_design_space, mode_sustained_curve, mode_budget_figure,
               mode_device_latency]:
        sub = fn(args)
        if sub: rc = sub
    return rc


def mode_all(args: argparse.Namespace) -> int:
    rc = 0
    for fn in [mode_summary, mode_figures]:
        sub = fn(args)
        if sub: rc = sub
    return rc


def mode_smoke(args: argparse.Namespace) -> int:
    """Smoke for the aggregator: run summary + figures over whatever's in results.json."""
    logger = _setup_logger("smoke")
    rc = mode_summary(args) | mode_figures(args)
    logger.info("ch11_1 smoke complete (rc=%d)", rc)
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="all",
                   choices=["summary", "design-space", "sustained-curve",
                            "budget-calculator", "budget-figure",
                            "device-latency", "figures", "all", "smoke"])
    p.add_argument("--p95-ms", type=float, default=None)
    p.add_argument("--power-mw", type=float, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke: args.mode = "smoke"
    dispatch = {
        "summary": mode_summary, "design-space": mode_design_space,
        "sustained-curve": mode_sustained_curve,
        "budget-calculator": mode_budget_calculator,
        "budget-figure": mode_budget_figure,
        "device-latency": mode_device_latency,
        "figures": mode_figures, "all": mode_all, "smoke": mode_smoke,
    }
    return dispatch[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
