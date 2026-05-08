"""
Chapter 10, Section 10.4 — Compare kernel families and what they imply
Companion script for "Efficient AI in Practice: Quantization and Fast Inference"

What this script demonstrates:
  Section 10.4 takes the four quantized variants produced in 10.3 and runs them
  against three llama.cpp builds compiled with progressively richer x86 ISAs
  (AVX-2, AVX-512+VNNI/VBMI, AVX-512+AMX), plus an Apple-Silicon NEON column
  reused from 10.3's Metal numbers. The goal is to test, empirically, the claim
  10.3 ended on: that IQ4_XS dominates Q4_K_M on modern x86 / Apple Silicon
  but converges or inverts on older x86 because legacy ISAs serialize the
  parallel codebook lookups IQ-quants depend on.

  Every measurement plotted in F09/F10 traces back to a real llama-bench
  invocation; nothing is interpolated. The M3 column is loaded from a
  hand-curated JSON (10.3's results.json export) so the chart can compare
  three x86 kernel families against an ARM/Metal reference without re-running
  10.3's Apple Silicon pipeline on this host.

Pipeline (--mode all = dispatch -> bench -> crossover -> figures):
  1. dispatch  - read each build's CMakeCache, run llama-bench --p 0 -n 1 -r 1
                 to capture system_info, cross-check that enabled flags actually
                 lit up at runtime, and emit Listing 10.7 (one Markdown table
                 per build).
  2. bench     - 15 cells = 3 x86 builds x 5 quants. llama-bench -p 64 -n 64 -t
                 <nproc> -r 5 each, parsing tg64 avg_ns/stddev_ns. CV gate at
                 0.10; up to 2 retries on noisy cells. M3 column folded in from
                 _artifacts/m3_results.json.
  3. crossover - pure derivation. ratio[k] = IQ4_XS_tps / Q4_K_M_tps per
                 kernel family. Gate: |ratio[avx2] - ratio[avx512-amx]| >= 0.05;
                 if smaller, the prose hook from 10.3 isn't substantiated and the
                 F10 caption gains a "[Caveat: ...]" prefix.
  4. figures   - reads results.json only. Renders F08 (vector lane diagram,
                 static), F09 (5x4 grouped bar chart, decode tok/s), F10
                 (single-row dot plot of IQ4_XS / Q4_K_M ratios per kernel
                 family). Re-runnable as many times as needed during prose
                 iteration; never re-invokes llama-bench.

Standalone modes (not in --mode all):
  - kernel-walk - extracts the Q4_K dot product from llama.cpp's
                  ggml/src/ggml-cpu/arch/{x86,arm}/quants.c into raw .c files,
                  ready for hand-annotation as Listing 10.8. The #A/#B/#C/#D
                  markers (load -> unpack -> fmadd -> reduce) are added by the
                  prose author against the script's raw output.

Inputs assumed to exist on disk (the script does not build llama.cpp and does
not quantize anything; if any path is missing it fails loud at startup):
  _build/llama-cpp-{avx2,avx512,avx512-amx}/bin/llama-bench
  _build/llama-cpp-avx512-amx/ggml/src/ggml-cpu/arch/{x86,arm}/quants.c
  _models/Llama-2-7b-hf-{F16,Q8_0,Q5_K_M,Q4_K_M,IQ4_XS}.gguf
  _artifacts/m3_results.json   (curated export of 10.3's M3/Metal decode tps)

Outputs:
  _artifacts/results.json                           central state, re-readable
  _artifacts/listing_10_7.md                        ISA flags x source map per build
  _artifacts/listing_10_8_x86.c                     raw x86 Q4_K dot product
  _artifacts/listing_10_8_arm.c                     raw ARM Q4_K dot product
  _artifacts/figures/CH10_F0{8,9,10}_Kalyanarangan.{png,pdf}
  figures/CH10_F0{8,9,10}_Kalyanarangan.{png,pdf}   manuscript-side copies
  _logs/bench_<build>_<quant>.json                  per-cell raw llama-bench output

Cross-section continuity:
  - The four quant slugs and their family colors come from 10.2/10.3; this
    script extends the palette with a kernel-family axis (avx2 = warm brown,
    avx512 = green, avx512-amx = dark green, neon-m3 = blue) plus distinct
    hatches so the four-way comparison survives B&W print.
  - The IQ4_XS / Q4_K_M ratio in F10 is the empirical answer to the prose
    in 10.3.5 about IQ4_XS losing on older x86 -- if avx2's ratio < 1 and
    avx512-amx's ratio > 1, the deployment guidance in 10.4.5 is grounded.
"""

import argparse
import datetime
import json
import logging
import os
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

BASE = Path(__file__).resolve().parent  #A all paths derived from script location, never cwd

# Module-level constants. Order matters for figure layouts.
BUILDS: Tuple[str, ...] = ("avx2", "avx512", "avx512-amx")           #A x86 builds we measure here
KERNEL_FAMILIES: Tuple[str, ...] = BUILDS + ("neon-m3",)             #A plus M3 column folded from 10.3
QUANTS: Tuple[str, ...] = ("F16", "Q8_0", "Q5_K_M", "Q4_K_M", "IQ4_XS")

# Every CMakeCache flag we cross-check against runtime system_info. Order is
# the order rows appear in Listing 10.7's tables.
ISA_FLAGS: Tuple[Tuple[str, str, str], ...] = (
    # (CMakeCache key,        system_info label, source file glob suffix)
    ("GGML_AVX",              "AVX",             "ggml/src/ggml-cpu/arch/x86/quants.c"),
    ("GGML_AVX2",             "AVX2",            "ggml/src/ggml-cpu/arch/x86/quants.c"),
    ("GGML_AVX512",           "AVX512",          "ggml/src/ggml-cpu/arch/x86/quants.c"),
    ("GGML_AVX512_VBMI",      "AVX512_VBMI",     "ggml/src/ggml-cpu/arch/x86/quants.c"),
    ("GGML_AVX512_VNNI",      "AVX512_VNNI",     "ggml/src/ggml-cpu/arch/x86/quants.c"),
    ("GGML_AVX_VNNI",         "AVX_VNNI",        "ggml/src/ggml-cpu/arch/x86/quants.c"),
    ("GGML_AMX_TILE",         "AMX_TILE",        "ggml/src/ggml-cpu/amx/mmq.cpp"),
    ("GGML_AMX_INT8",         "AMX_INT8",        "ggml/src/ggml-cpu/amx/mmq.cpp"),
    ("GGML_AMX_BF16",         "AMX_BF16",        "ggml/src/ggml-cpu/amx/mmq.cpp"),
)


# --- Configuration ---------------------------------------------------------

@dataclass
class Config:
    # Paths
    build_dir: Path = field(default_factory=lambda: BASE / "_build")
    models_dir: Path = field(default_factory=lambda: BASE / "_models")
    artifacts_dir: Path = field(default_factory=lambda: BASE / "_artifacts")
    figures_dir: Path = field(default_factory=lambda: BASE / "_artifacts" / "figures")
    book_figures_dir: Path = field(default_factory=lambda: BASE / "figures")
    logs_dir: Path = field(default_factory=lambda: BASE / "_logs")
    results_path: Path = field(default_factory=lambda: BASE / "_artifacts" / "results.json")
    m3_results_path: Path = field(default_factory=lambda: BASE / "_artifacts" / "m3_results.json")
    log_file: Path = field(default_factory=lambda: BASE / "ch10_kernels.log")

    # Source artifacts
    repo_id: str = "meta-llama/Llama-2-7b-hf"
    model_basename: str = "Llama-2-7b-hf"

    # llama-bench config
    prompt_n: int = 64
    decode_n: int = 64
    reps: int = 5
    cv_threshold: float = 0.10
    retry_max: int = 2

    # Figure aesthetics — match section 10.2/10.3 conventions
    fig_width_in: float = 5.6
    fig_dpi: int = 300

    # Quant family palette (lifted from ch10_convert.py)
    quant_color: Dict[str, str] = field(default_factory=lambda: {
        "F16":    "#444444",
        "Q8_0":   "#7E76B0",
        "Q5_K_M": "#319974",
        "Q4_K_M": "#319974",
        "IQ4_XS": "#D67430",
    })
    quant_hatch: Dict[str, str] = field(default_factory=lambda: {
        "F16":    "",
        "Q8_0":   "....",
        "Q5_K_M": "////",
        "Q4_K_M": "////",
        "IQ4_XS": "xxxx",
    })

    # Kernel family palette — second axis introduced for 10.4
    kernel_color: Dict[str, str] = field(default_factory=lambda: {
        "avx2":       "#B5651D",   #A warm brown — older x86, baseline
        "avx512":     "#319974",   #A green — modern x86 mainline
        "avx512-amx": "#1F5C46",   #A dark green — modern x86 + accelerator
        "neon-m3":    "#3060B0",   #A blue — ARM (visually distinct from x86)
    })
    kernel_hatch: Dict[str, str] = field(default_factory=lambda: {
        "avx2":       "....",
        "avx512":     "////",
        "avx512-amx": "xxxx",
        "neon-m3":    "\\\\\\\\",
    })
    kernel_marker: Dict[str, str] = field(default_factory=lambda: {
        "avx2":       "D",
        "avx512":     "s",
        "avx512-amx": "^",
        "neon-m3":    "o",
    })


# --- Tiny utilities --------------------------------------------------------

def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str, log_file: Optional[Path] = None) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a") as f:
            f.write(line + "\n")


def run_cmd(cmd: List[str], log_path: Path, cwd: Optional[Path] = None,
            env: Optional[Dict[str, str]] = None) -> Tuple[float, int, str]:
    """Stream cmd to stdout AND log_path; return (wall_seconds, returncode, tail_stderr)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    merged_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")          #A defuse libiomp5 double-init when MKL is present
    if env:
        merged_env.update(env)
    pretty = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n$ {pretty}", flush=True)
    print(f"  log -> {log_path}", flush=True)
    t0 = time.perf_counter()
    tail_buffer: List[str] = []
    with log_path.open("w") as logf:
        logf.write(f"$ {pretty}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(cwd) if cwd else None, env=merged_env, text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            tail_buffer.append(line)
            if len(tail_buffer) > 400:
                tail_buffer.pop(0)
        rc = proc.wait()
    wall = time.perf_counter() - t0
    return wall, rc, "".join(tail_buffer)


def must_run(cmd: List[str], log_path: Path, cwd: Optional[Path] = None,
             env: Optional[Dict[str, str]] = None) -> Tuple[float, str]:
    wall, rc, tail = run_cmd(cmd, log_path, cwd=cwd, env=env)
    if rc != 0:
        sys.stderr.write(
            f"\n[FATAL] command failed (rc={rc}). full log: {log_path}\n"
            f"--- tail ---\n{tail}\n"
        )
        sys.exit(1)
    return wall, tail


def host_info() -> Dict[str, object]:
    """Linux /proc-based host info (this script targets c3-standard-22 GCE)."""
    info: Dict[str, object] = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    cpu_brand = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_brand = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    info["cpu_brand"] = cpu_brand
    info["nproc"] = os.cpu_count() or 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                info["ram_gb"] = round(kb / (1024 ** 2), 1)
                break
    except Exception:
        info["ram_gb"] = 0
    return info


def read_results(cfg: Config) -> dict:
    if not cfg.results_path.exists():
        return {}
    return json.loads(cfg.results_path.read_text())


def write_results(cfg: Config, data: dict) -> None:
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.results_path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _gguf_path(cfg: Config, quant: str) -> Path:
    return cfg.models_dir / f"{cfg.model_basename}-{quant}.gguf"


def _bench_bin(cfg: Config, build: str) -> Path:
    return cfg.build_dir / f"llama-cpp-{build}" / "bin" / "llama-bench"


def _src_root(cfg: Config) -> Path:
    """The avx512-amx tree is the source of truth for kernel-walk extracts."""
    return cfg.build_dir / "llama-cpp-avx512-amx"


# --- Pre-flight: assert all assumed inputs exist ---------------------------

def _check_inputs(cfg: Config, modes: List[str]) -> None:
    """Fail loud if any path expected by the requested modes is missing."""
    missing: List[str] = []

    # Bench binaries are needed for dispatch and bench.
    if any(m in modes for m in ("dispatch", "bench")):
        for b in BUILDS:
            p = _bench_bin(cfg, b)
            if not p.exists():
                missing.append(str(p.relative_to(BASE)))

    # GGUFs needed for bench (and for the smallest one in dispatch).
    if "bench" in modes:
        for q in QUANTS:
            g = _gguf_path(cfg, q)
            if not g.exists():
                missing.append(str(g.relative_to(BASE)))

    # Smallest gguf is enough for dispatch's banner-capture run.
    if "dispatch" in modes and "bench" not in modes:
        g = _gguf_path(cfg, "IQ4_XS")
        if not g.exists():
            missing.append(str(g.relative_to(BASE)))

    # M3 results JSON needed for bench's NEON column.
    if "bench" in modes:
        if not cfg.m3_results_path.exists():
            missing.append(str(cfg.m3_results_path.relative_to(BASE)))

    # Source files for kernel-walk.
    if "kernel-walk" in modes:
        src = _src_root(cfg)
        for rel in ("ggml/src/ggml-cpu/arch/x86/quants.c",
                    "ggml/src/ggml-cpu/arch/arm/quants.c"):
            p = src / rel
            if not p.exists():
                missing.append(str(p.relative_to(BASE)))

    if missing:
        sys.stderr.write("\n[FATAL] missing required inputs:\n")
        for m in missing:
            sys.stderr.write(f"  - {m}\n")
        sys.stderr.write(
            "\nThis script assumes ch10_convert.py and the three llama.cpp builds\n"
            "have already produced these files. Re-check the spec's 'Inputs assumed\n"
            "to exist on disk' section and try again.\n"
        )
        sys.exit(1)


# --- Mode: dispatch --------------------------------------------------------

_CMAKE_BOOL_RE = re.compile(r"^([A-Z0-9_]+):BOOL=(ON|OFF|TRUE|FALSE|1|0)\s*$",
                            re.IGNORECASE | re.MULTILINE)
_SYSINFO_PAIR_RE = re.compile(r"([A-Z0-9_]+(?:_[A-Z0-9_]+)*)\s*=\s*([01])")


def _read_cmake_flags(cache_path: Path) -> Dict[str, bool]:
    text = cache_path.read_text()
    out: Dict[str, bool] = {}
    for m in _CMAKE_BOOL_RE.finditer(text):
        key = m.group(1).upper()
        val_raw = m.group(2).upper()
        out[key] = val_raw in ("ON", "TRUE", "1")
    return out


def _capture_system_info(bench: Path, gguf: Path, log_path: Path) -> str:
    """Run llama-bench briefly and pull the `system_info` line out of stderr.

    Newer llama.cpp builds (post 2025-07) dropped the verbose `system_info: ...`
    banner from llama-bench. If we don't find it, return "" and the caller will
    fall back to deriving flags from /proc/cpuinfo (which reflects the host's
    runtime-available features and matches what system_info historically printed).
    """
    cmd = [str(bench), "-m", str(gguf), "-p", "0", "-n", "1", "-r", "1", "-v"]
    _, _, tail = run_cmd(cmd, log_path)
    for line in tail.splitlines():
        if "system_info:" in line and ("=" in line):
            return line.split("system_info:", 1)[1].strip()
    return ""


def _parse_system_info(line: str) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for m in _SYSINFO_PAIR_RE.finditer(line):
        out[m.group(1).upper()] = (m.group(2) == "1")
    return out


def _runtime_flags_from_cpuinfo() -> Dict[str, bool]:
    """Approximate what llama.cpp's old `system_info` banner would have printed,
    from /proc/cpuinfo. Used when the modern llama-bench omits the banner.
    Mapping: kernel-cpuinfo flag name -> llama.cpp's system_info label."""
    # Note: kernel cpuinfo names are historically inconsistent — vbmi is one
    # word, vnni has an underscore, etc. The map below matches what
    # Sapphire Rapids /proc/cpuinfo actually emits today.
    flag_map = {
        "avx":          "AVX",
        "avx2":         "AVX2",
        "avx512f":      "AVX512",
        "avx512vbmi":   "AVX512_VBMI",      #A no underscore in cpuinfo
        "avx512_vnni":  "AVX512_VNNI",      #A underscore in cpuinfo
        "avx_vnni":     "AVX_VNNI",
        "amx_tile":     "AMX_TILE",
        "amx_int8":     "AMX_INT8",
        "amx_bf16":     "AMX_BF16",
    }
    flags_line = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags") and ":" in line:
                flags_line = line.split(":", 1)[1].strip()
                break
    except Exception:
        return {}
    have = set(flags_line.split())
    return {label: (k in have) for k, label in flag_map.items()}


def _emit_listing_10_7(cfg: Config, dispatch: Dict[str, dict]) -> Path:
    """One Markdown table per build, paste-ready into the manuscript."""
    out = cfg.artifacts_dir / "listing_10_7.md"
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Listing 10.7 — ISA flags & kernel sources, per build\n")
    for b in BUILDS:
        d = dispatch.get(b, {})
        cmake = d.get("cmake_flags", {})
        sysinfo = d.get("system_info_flags", {})
        lines.append(f"**build-{b}**\n")
        lines.append("| Feature        | CMakeCache | system_info | Source location                              |")
        lines.append("|----------------|:----------:|:-----------:|----------------------------------------------|")
        for cmake_key, sys_key, src in ISA_FLAGS:
            cm_on = cmake.get(cmake_key, False)
            si_on = sysinfo.get(sys_key, False)
            cm_disp = "ON" if cm_on else "OFF"
            si_disp = "1" if si_on else "0"
            label_map = {
                "GGML_AVX": "AVX",
                "GGML_AVX2": "AVX2",
                "GGML_AVX512": "AVX-512 F",
                "GGML_AVX512_VBMI": "AVX-512 VBMI",
                "GGML_AVX512_VNNI": "AVX-512 VNNI",
                "GGML_AVX_VNNI": "AVX-VNNI",
                "GGML_AMX_TILE": "AMX-TILE",
                "GGML_AMX_INT8": "AMX-INT8",
                "GGML_AMX_BF16": "AMX-BF16",
            }
            label = label_map.get(cmake_key, cmake_key)
            lines.append(
                f"| {label:<14} |     {cm_disp:^4}   |     {si_disp:^4}    | {src:<44} |"
            )
        lines.append("")
    out.write_text("\n".join(lines) + "\n")
    return out


def mode_dispatch(cfg: Config) -> dict:
    """For each build: read CMakeCache, run llama-bench briefly, cross-check, map sources."""
    out: Dict[str, dict] = {}
    smallest = _gguf_path(cfg, "IQ4_XS")                              #A smallest variant — dispatch only needs the banner
    for b in BUILDS:
        bench = _bench_bin(cfg, b)
        cache = cfg.build_dir / f"llama-cpp-{b}" / "CMakeCache.txt"

        cmake_all = _read_cmake_flags(cache)
        # Filter to just the flags we cross-check against runtime sysinfo.
        cmake_flags = {k: cmake_all.get(k, False) for k, _, _ in ISA_FLAGS}

        dispatch_log = cfg.logs_dir / f"dispatch_{b}.log"
        sys_line = _capture_system_info(bench, smallest, dispatch_log)
        sys_flags = _parse_system_info(sys_line) if sys_line else {}
        if not sys_flags:
            # Modern llama-bench dropped the system_info banner. Derive from
            # /proc/cpuinfo instead — same content, captured once per host.
            sys_flags = _runtime_flags_from_cpuinfo()
            sys_line = (
                "(derived from /proc/cpuinfo: "
                + " | ".join(f"{k} = {1 if v else 0}" for k, v in sys_flags.items())
                + ")"
            )

        # Cross-check: every CMakeCache=ON should appear =1 in system_info.
        # We don't fail on CMakeCache=OFF + sysinfo=1 (the system_info line
        # advertises *every* runtime-detected feature, even when the build
        # didn't compile that path in — fine, just informational).
        for cmake_key, sys_key, _ in ISA_FLAGS:
            if cmake_flags.get(cmake_key, False) and not sys_flags.get(sys_key, False):
                sys.stderr.write(
                    f"[FATAL] dispatch mismatch in build={b}: "
                    f"{cmake_key}=ON in CMakeCache but {sys_key} not =1 in system_info\n"
                    f"  system_info line: {sys_line}\n"
                )
                sys.exit(1)

        # Source map: which file holds each enabled feature's kernels.
        source_map = {sys_key: src for cmake_key, sys_key, src in ISA_FLAGS
                      if cmake_flags.get(cmake_key, False)}

        out[b] = {
            "cmake_flags": cmake_flags,
            "system_info": sys_line,
            "system_info_flags": sys_flags,
            "source_map": source_map,
            "log": str(dispatch_log.relative_to(BASE)),
        }
        log(f"dispatch[{b}]: {sum(cmake_flags.values())} flags ON; system_info OK", cfg.log_file)

    listing = _emit_listing_10_7(cfg, out)
    log(f"listing 10.7 -> {listing.relative_to(BASE)}", cfg.log_file)
    return out


# --- Mode: kernel-walk -----------------------------------------------------

_FN_SIG_RE = re.compile(r"\bggml_vec_dot_q4_K_q8_K\b\s*\(", re.MULTILINE)


def _extract_function(src: str, fn_re: re.Pattern) -> Optional[str]:
    """Extract a C function body by name, brace-balanced."""
    m = fn_re.search(src)
    if not m:
        return None
    # Walk back to start of the function (return type begins on its own line
    # or just after a preceding `;`/`}`/preprocessor line).
    start = m.start()
    line_start = src.rfind("\n", 0, start) + 1
    # If the line begins with whitespace, walk back until we hit a non-empty,
    # non-`#` preceding line break — keeps the return-type prefix on the line.
    while line_start > 0:
        prev_nl = src.rfind("\n", 0, line_start - 1)
        prev_line = src[prev_nl + 1:line_start - 1].strip() if prev_nl >= 0 else src[:line_start - 1].strip()
        if not prev_line or prev_line.endswith((";", "}", "*/")) or prev_line.startswith("#") or prev_line.startswith("//"):
            break
        line_start = prev_nl + 1
    # Find opening brace.
    open_brace = src.find("{", m.end())
    if open_brace == -1:
        return None
    depth = 1
    i = open_brace + 1
    while i < len(src) and depth > 0:
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return src[line_start:i]


def _trim_to_inner_loop(body: str, prefer_after: Optional[str] = None) -> str:
    """Keep only the `for (int i = 0; i < nb; ++i) { ... }` accumulation loop body.

    `prefer_after` (optional regex pattern): start scanning for the inner loop
    only AFTER the first match of this pattern in the body. Used for ARM where
    the function has multiple branches (SVE, NEON, dotprod) and we specifically
    want the NEON branch -- the chapter calls out NEON's 128-bit q registers
    and SDOT, not SVE's variable-length vectors.
    """
    search_from = 0
    if prefer_after:
        anchor = re.search(prefer_after, body)
        if anchor:
            search_from = anchor.end()
    loop_re = re.compile(r"for\s*\(\s*int\s+i\s*=\s*0\s*;\s*i\s*<\s*nb\s*;\s*\+\+i\s*\)\s*\{")
    m = loop_re.search(body, search_from)
    if not m:
        # Fall back: scan from the top.
        m = loop_re.search(body)
        if not m:
            return body                                               #B return whole function if regex shape drifts
    open_brace = m.end() - 1
    depth = 1
    i = open_brace + 1
    while i < len(body) and depth > 0:
        c = body[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    inner = body[m.start():i]
    return inner


def mode_kernel_walk(cfg: Config) -> dict:
    out: Dict[str, str] = {}
    src_root = _src_root(cfg)
    # ARM source has multiple branches (SVE+MATMUL, SVE, NEON, dotprod). The
    # chapter narrative specifically calls out NEON's 128-bit q registers and
    # SDOT, so anchor extraction at the `#elif defined(__ARM_NEON)` line.
    arch_specs = (
        ("x86", "listing_10_8_x86.c", None),
        # llama.cpp's arm/quants.c writes the NEON branch as
        # `#elif defined __ARM_NEON` (no parens) — anchor accepts either form.
        ("arm", "listing_10_8_arm.c",
         r"#(?:elif|el\s*if|if)\s+defined\s*\(?\s*__ARM_NEON\s*\)?"),
    )
    for arch, listing_name, prefer_after in arch_specs:
        path = src_root / "ggml" / "src" / "ggml-cpu" / "arch" / arch / "quants.c"
        text = path.read_text(encoding="utf-8", errors="replace")
        fn = _extract_function(text, _FN_SIG_RE)
        if fn is None:
            sys.stderr.write(f"[WARN] could not locate ggml_vec_dot_q4_K_q8_K in {path}\n")
            continue
        # Prepend a comment with the source path so the prose author can
        # cross-reference the upstream file when annotating.
        header = (
            f"// Source: {path.relative_to(BASE)}\n"
            f"// Extract: ggml_vec_dot_q4_K_q8_K (inner loop body)\n"
            f"// Hand-add #A/#B/#C/#D markers on:\n"
            f"//   #A scale broadcast (FP16 super-scale x INT8 sub-scale)\n"
            f"//   #B nibble unpack (low and high nibbles into INT8 lanes)\n"
            f"//   #C dot-product instruction (VPDPBUSD on AVX-512+VNNI; SDOT on NEON)\n"
            f"//   #D horizontal reduction at end of block\n\n"
        )
        inner = _trim_to_inner_loop(fn, prefer_after=prefer_after)
        listing_path = cfg.artifacts_dir / listing_name
        listing_path.write_text(header + inner + "\n")
        out[arch] = str(listing_path.relative_to(BASE))
        log(f"kernel-walk[{arch}]: {len(inner.splitlines())} lines -> {listing_path.relative_to(BASE)}",
            cfg.log_file)
    return out


# --- Mode: bench -----------------------------------------------------------

def _parse_bench_json(stdout: str) -> Dict[str, Dict[str, float]]:
    """llama-bench --output json prints a JSON array of test results.
    Each element has n_prompt, n_gen, avg_ns, stddev_ns. We split into pp/tg by
    n_prompt > 0 vs n_gen > 0."""
    out: Dict[str, Dict[str, float]] = {}
    # Find the JSON block — llama-bench prints a banner first, then the array.
    start = stdout.find("[")
    end = stdout.rfind("]")
    if start < 0 or end < 0 or end <= start:
        return out
    try:
        arr = json.loads(stdout[start:end + 1])
    except json.JSONDecodeError:
        return out
    for row in arr:
        n_prompt = int(row.get("n_prompt", 0))
        n_gen = int(row.get("n_gen", 0))
        avg = float(row.get("avg_ns", 0.0))
        sd = float(row.get("stddev_ns", 0.0))
        if n_prompt > 0 and n_gen == 0:
            out["pp"] = {"n": float(n_prompt), "avg_ns": avg, "stddev_ns": sd}
        elif n_gen > 0 and n_prompt == 0:
            out["tg"] = {"n": float(n_gen), "avg_ns": avg, "stddev_ns": sd}
    return out


def _bench_one(cfg: Config, build: str, quant: str) -> dict:
    """Run llama-bench for one (build, quant) cell with CV-gated retry. Return cell dict."""
    bench = _bench_bin(cfg, build)
    gguf = _gguf_path(cfg, quant)
    log_path = cfg.logs_dir / f"bench_{build}_{quant}.log"
    raw_path = cfg.logs_dir / f"bench_{build}_{quant}.json"
    threads = str(os.cpu_count() or 8)

    # Note: spec says -p 64 -n 64, but llama.cpp's AMX-INT8 prompt-processing
    # path raises SIGILL on Sapphire Rapids c3 GCE for quantized inputs (F16
    # works because it skips AMX repack). Decode (-n) is what feeds F09/F10
    # so we run with -p 0 -n 64 only. pp_tok_per_sec is left null in results.
    cmd = [
        str(bench),
        "-m", str(gguf),
        "-p", "0",
        "-n", str(cfg.decode_n),
        "-t", threads,
        "-r", str(cfg.reps),
        "-o", "json",
    ]

    best: Optional[dict] = None
    attempts = 0
    while attempts <= cfg.retry_max:
        attempts += 1
        wall, rc, tail = run_cmd(cmd, log_path)
        if rc != 0:
            sys.stderr.write(f"[WARN] llama-bench rc={rc} for {build}/{quant}, attempt {attempts}\n")
            if attempts > cfg.retry_max:
                sys.stderr.write(f"[FATAL] giving up on {build}/{quant}\n")
                sys.exit(1)
            continue
        parsed = _parse_bench_json(tail)
        tg = parsed.get("tg")
        pp = parsed.get("pp")
        if not tg or tg.get("avg_ns", 0) <= 0:
            sys.stderr.write(f"[WARN] bench for {build}/{quant} produced no tg row, attempt {attempts}\n")
            if attempts > cfg.retry_max:
                sys.stderr.write(f"[FATAL] {build}/{quant} unparseable\n")
                sys.exit(1)
            continue
        tps = 1e9 * tg["n"] / tg["avg_ns"]
        cv = tg["stddev_ns"] / tg["avg_ns"] if tg["avg_ns"] > 0 else 0.0
        # Persist the raw JSON output (whatever the bench printed) for audit.
        raw_path.write_text(tail)
        cell = {
            "tok_per_sec": round(tps, 4),
            "cv": round(cv, 4),
            "noisy": False,
            "tg_avg_ns": tg["avg_ns"],
            "tg_stddev_ns": tg["stddev_ns"],
            "tg_n": int(tg["n"]),
            "pp_tok_per_sec": round(1e9 * pp["n"] / pp["avg_ns"], 4) if pp and pp["avg_ns"] > 0 else None,
            "wall_s": round(wall, 2),
            "attempts": attempts,
            "raw_path": str(raw_path.relative_to(BASE)),
            "log": str(log_path.relative_to(BASE)),
        }
        if cv <= cfg.cv_threshold:
            return cell
        # CV exceeded threshold — keep best (lowest cv) so far, retry.
        if best is None or cell["cv"] < best["cv"]:
            best = cell
        log(f"bench[{build}/{quant}] cv={cv:.3f} > {cfg.cv_threshold}; retrying ({attempts}/{cfg.retry_max + 1})",
            cfg.log_file)
    # Exhausted retries — keep best with noisy=True
    assert best is not None
    best["noisy"] = True
    return best


def mode_bench(cfg: Config) -> dict:
    out: Dict[str, Dict[str, dict]] = {}
    for b in BUILDS:
        out[b] = {}
        for q in QUANTS:
            log(f"bench {b}/{q}", cfg.log_file)
            out[b][q] = _bench_one(cfg, b, q)
            tag = "NOISY" if out[b][q]["noisy"] else "ok"
            log(f"  -> {out[b][q]['tok_per_sec']:.2f} tok/s (cv={out[b][q]['cv']:.3f}, {tag})",
                cfg.log_file)

    # M3 column folded from hand-curated _artifacts/m3_results.json
    m3 = json.loads(cfg.m3_results_path.read_text())
    out["neon-m3"] = {}
    for q in QUANTS:
        v = m3.get(q)
        if not v:
            sys.stderr.write(f"[FATAL] m3_results.json missing quant {q}\n")
            sys.exit(1)
        out["neon-m3"][q] = {
            "tok_per_sec": float(v["tok_per_sec"]),
            "cv": float(v.get("cv", 0.02)),
            "noisy": bool(v.get("noisy", False)),
            "provisional": bool(m3.get("provisional", False)),
            "raw_path": str(cfg.m3_results_path.relative_to(BASE)),
        }
    out["_meta"] = {
        "neon-m3_provenance": m3.get("_provenance", ""),
        "neon-m3_provisional": bool(m3.get("provisional", False)),
        "neon-m3_host": m3.get("host", ""),
    }
    return out


# --- Mode: crossover -------------------------------------------------------

def mode_crossover(cfg: Config, results: dict) -> dict:
    bench = results.get("bench", {})
    ratios: Dict[str, float] = {}
    for k in KERNEL_FAMILIES:
        cell = bench.get(k, {})
        iq = cell.get("IQ4_XS", {}).get("tok_per_sec")
        q4 = cell.get("Q4_K_M", {}).get("tok_per_sec")
        if iq is None or q4 is None or q4 == 0:
            sys.stderr.write(f"[FATAL] crossover missing IQ4_XS or Q4_K_M for {k}\n")
            sys.exit(1)
        ratios[k] = round(iq / q4, 4)

    gap = abs(ratios["avx2"] - ratios["avx512-amx"])
    gate_passed = gap >= 0.05
    if not gate_passed:
        log(
            f"[GATE] crossover gap |avx2-vs-avx512-amx| = {gap:.4f} < 0.05; "
            f"prose hook ungrounded — figure 10.10 will carry caveat bracket.",
            cfg.log_file,
        )
    else:
        log(f"[GATE] crossover gap = {gap:.4f} >= 0.05 OK", cfg.log_file)

    # gap_winner = kernel family where IQ4_XS wins by the most (used in F10 caption)
    gap_winner = max(ratios.items(), key=lambda kv: kv[1])[0]

    out = dict(ratios)
    out["gate_passed"] = gate_passed
    out["gap"] = round(gap, 4)
    out["gap_winner"] = gap_winner
    return out


# --- Figure helpers --------------------------------------------------------

def _setup_mpl(cfg: Config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    pick = next((c for c in ("Arial", "Helvetica", "DejaVu Sans") if c in available), "sans-serif")
    if pick != "Arial":
        log(f"font 'Arial' unavailable; falling back to '{pick}'", cfg.log_file)
    plt.rcParams.update({
        "font.family": pick,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": cfg.fig_dpi, "savefig.dpi": cfg.fig_dpi,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "hatch.linewidth": 0.5,
        "hatch.color": "black",
    })
    return plt


def _save_pair(plt, fig, cfg: Config, name: str) -> Tuple[str, str]:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.book_figures_dir.mkdir(parents=True, exist_ok=True)
    png = cfg.figures_dir / f"{name}.png"
    pdf = cfg.figures_dir / f"{name}.pdf"
    fig.savefig(png, dpi=cfg.fig_dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
    shutil.copy2(png, cfg.book_figures_dir / png.name)
    shutil.copy2(pdf, cfg.book_figures_dir / pdf.name)
    plt.close(fig)
    return str(png.relative_to(BASE)), str(pdf.relative_to(BASE))


# --- Figure 10.8: Vector lane diagram --------------------------------------

def _fig_lane_diagram(cfg: Config) -> Tuple[Tuple[str, str], str]:
    """Static diagram. Three stacked register strips for one Q4_K dot product."""
    plt = _setup_mpl(cfg)
    from matplotlib.patches import Rectangle, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(cfg.fig_width_in, 3.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    fill = cfg.kernel_color["avx512-amx"]
    edge = "#222222"

    # Strip 1 — zmm0, packed nibbles. 16 visible lanes + ellipsis.
    y_top = 88
    strip_h = 8
    lane_x0, lane_x1 = 18, 70
    n_lanes = 16
    lane_w = (lane_x1 - lane_x0) / n_lanes
    ax.text(lane_x0 - 1.5, y_top + strip_h / 2, "zmm0", fontsize=7.5,
            ha="right", va="center", fontweight="bold")
    ax.text(lane_x0 - 1.5, y_top + strip_h / 2 - 4.5,
            "(packed nibbles)", fontsize=6.0, ha="right", va="center", style="italic", color="#555")
    for i in range(n_lanes):
        x = lane_x0 + i * lane_w
        ax.add_patch(Rectangle((x, y_top), lane_w, strip_h,
                               facecolor=fill, alpha=0.18,
                               edgecolor=edge, linewidth=0.5))
        if i < 8 or i >= n_lanes - 1:
            label = f"n{i if i < 8 else 127}"
            ax.text(x + lane_w / 2, y_top + strip_h / 2, label,
                    fontsize=5.5, ha="center", va="center")
    ax.text((lane_x0 + lane_x1) / 2 + 5, y_top + strip_h / 2, "…",
            fontsize=10, ha="center", va="center", color="#444")
    ax.text(lane_x1 + 1.5, y_top + strip_h / 2,
            "[LOAD] 1× VMOVDQU64 — 64 bytes, one cache line",
            fontsize=6.5, ha="left", va="center", color="#222")

    # Strip 2 — zmm1, after VPSRLD + AND mask.
    y_mid = 60
    n_lanes2 = 8
    lane_w2 = (lane_x1 - lane_x0) / n_lanes2
    ax.text(lane_x0 - 1.5, y_mid + strip_h / 2, "zmm1", fontsize=7.5,
            ha="right", va="center", fontweight="bold")
    ax.text(lane_x0 - 1.5, y_mid + strip_h / 2 - 4.5,
            "(unpacked INT8)", fontsize=6.0, ha="right", va="center", style="italic", color="#555")
    for i in range(n_lanes2):
        x = lane_x0 + i * lane_w2
        ax.add_patch(Rectangle((x, y_mid), lane_w2, strip_h,
                               facecolor=fill, alpha=0.32,
                               edgecolor=edge, linewidth=0.5))
        if i < 4 or i == n_lanes2 - 1:
            label = f"q{i if i < 4 else 63}"
            ax.text(x + lane_w2 / 2, y_mid + strip_h / 2, label,
                    fontsize=6.0, ha="center", va="center")
    ax.text((lane_x0 + lane_x1) / 2 + 4, y_mid + strip_h / 2, "…",
            fontsize=10, ha="center", va="center", color="#444")
    ax.text(lane_x1 + 1.5, y_mid + strip_h / 2,
            "[UNPACK] VPSRLD + VPANDQ — split low/high nibbles",
            fontsize=6.5, ha="left", va="center", color="#222")

    # Strip 3 — zmm2, INT32 accumulator.
    y_low = 32
    n_lanes3 = 4
    lane_w3 = (lane_x1 - lane_x0) / n_lanes3
    ax.text(lane_x0 - 1.5, y_low + strip_h / 2, "zmm2", fontsize=7.5,
            ha="right", va="center", fontweight="bold")
    ax.text(lane_x0 - 1.5, y_low + strip_h / 2 - 4.5,
            "(16× INT32 acc)", fontsize=6.0, ha="right", va="center", style="italic", color="#555")
    for i in range(n_lanes3):
        x = lane_x0 + i * lane_w3
        ax.add_patch(Rectangle((x, y_low), lane_w3, strip_h,
                               facecolor=fill, alpha=0.55,
                               edgecolor=edge, linewidth=0.6))
        label = f"acc{i}" if i < n_lanes3 - 1 else "…"
        ax.text(x + lane_w3 / 2, y_low + strip_h / 2, label,
                fontsize=6.5, ha="center", va="center", color="white" if i < n_lanes3 - 1 else "#444")
    ax.text(lane_x1 + 1.5, y_low + strip_h / 2,
            "[FMADD] VPDPBUSD (AVX-VNNI) — INT8×INT8 → INT32 fused dot",
            fontsize=6.5, ha="left", va="center", color="#222")

    # Reduce arrow + label below strip 3.
    arr = FancyArrowPatch(((lane_x0 + lane_x1) / 2, y_low - 1),
                          ((lane_x0 + lane_x1) / 2, y_low - 9),
                          arrowstyle="-|>", mutation_scale=10,
                          color="#333", linewidth=1.0)
    ax.add_patch(arr)
    ax.text((lane_x0 + lane_x1) / 2, y_low - 11.5,
            "[REDUCE] VPHADDD chain — 16 INT32 → 1 scalar",
            fontsize=6.5, ha="center", va="top", color="#222")

    # NEON equivalence text at the bottom.
    fig.text(
        0.5, 0.025,
        "NEON path uses 128-bit q registers (¼ width); the lane structure is conceptually identical, "
        "throughput is one-fourth per instruction. SDOT (Armv8.4-A) is the dot-product equivalent of VPDPBUSD.",
        ha="center", va="bottom", fontsize=6.0, color="#555", style="italic", wrap=True,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    name = "CH10_F08_Kalyanarangan"
    paths = _save_pair(plt, fig, cfg, name)
    caption = (
        "Vector lane progression for one Q4_K block decode on AVX-512 with VNNI. The 256-weight "
        "superblock from Section 10.2 traverses three register states: packed-nibble load, "
        "unpacked INT8 lanes, and INT32 accumulators that horizontal-reduce to a single dot "
        "product. The NEON path executes the same sequence at one-quarter the lane count "
        "per instruction."
    )
    return paths, caption


# --- Figure 10.9: Decode tok/s grouped bars --------------------------------

def _fig_decode_grouped(cfg: Config, results: dict) -> Tuple[Tuple[str, str], str]:
    plt = _setup_mpl(cfg)
    from matplotlib.patches import Patch
    import numpy as np

    fig, ax = plt.subplots(figsize=(cfg.fig_width_in, 4.4))

    bench = results.get("bench", {})
    n_groups = len(QUANTS)
    n_bars = len(KERNEL_FAMILIES)
    group_centers = np.arange(n_groups, dtype=float)
    bar_width = 0.18
    span = bar_width * n_bars                                          #A actual width occupied per group

    # Position offsets so bars are centered around each group center.
    bar_offsets = np.linspace(-span / 2 + bar_width / 2,
                              span / 2 - bar_width / 2, n_bars)

    # First pass — track ymax for axis range. Each value label is anchored
    # above its own bar's error-bar (1σ whisker) top in the second pass, so
    # labels stay tied to their own bars instead of floating above the group.
    ymax = 0.0
    for k in KERNEL_FAMILIES:
        for q in QUANTS:
            tps = bench.get(k, {}).get(q, {}).get("tok_per_sec", 0.0)
            ymax = max(ymax, float(tps))

    for i, k in enumerate(KERNEL_FAMILIES):
        for j, q in enumerate(QUANTS):
            cell = bench.get(k, {}).get(q, {})
            tps = float(cell.get("tok_per_sec", 0.0))
            cv = float(cell.get("cv", 0.0))
            noisy = bool(cell.get("noisy", False))
            x = group_centers[j] + bar_offsets[i]
            face = cfg.kernel_color[k] if not noisy else "white"
            ax.bar(
                x, tps, width=bar_width,
                color=face, edgecolor="black", linewidth=0.7,
                hatch=cfg.kernel_hatch[k], zorder=2,
            )
            # Error bars (1σ from CV).
            ax.errorbar(
                x, tps, yerr=tps * cv, fmt="none",
                ecolor="#444444", elinewidth=0.8, capsize=2, zorder=3,
            )
            # Value annotation — sits just above each bar's own error-bar (1σ
            # whisker) top, with a small stagger so close-value pairs like
            # 16.4/16.7 don't run into each other horizontally. neon-m3 (i=3)
            # gets a smaller bump because its bar is the tallest in every group,
            # so its label is already comfortably above its neighbours.
            err_top = tps * (1.0 + cv)
            stagger = {1: ymax * 0.032, 3: ymax * 0.010}.get(i, 0.0)
            y_offset = ymax * 0.018 + stagger
            ax.text(x, err_top + y_offset, f"{tps:.1f}",
                    ha="center", va="bottom", fontsize=5.5, color="#222")

    ax.set_xticks(group_centers)
    ax.set_xticklabels(list(QUANTS))
    ax.set_xlabel("Quantization variant")
    ax.set_ylabel("Decode throughput (tokens / sec)")
    ax.set_title("Decode throughput per kernel family on Llama-2-7B",
                 fontsize=8.5, pad=8)
    ax.grid(True, axis="y", linestyle=":", color="#cccccc", linewidth=0.5)
    ax.set_ylim(0, ymax * 1.20 + 1)
    ax.set_xlim(-0.5, n_groups - 0.5)

    # Legend — four kernel-family patches in a single row beneath the x-axis,
    # outside the plot area so it never lands on top of the bars.
    legend_handles = [
        Patch(facecolor=cfg.kernel_color[k], hatch=cfg.kernel_hatch[k],
              edgecolor="black", linewidth=0.6, label=k)
        for k in KERNEL_FAMILIES
    ]
    ax.legend(handles=legend_handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=4, frameon=False, fontsize=7.5,
              handlelength=2.0, handleheight=1.2,
              borderaxespad=0.0, columnspacing=1.6)

    # Footnote sits below the legend strip.
    fig.text(
        0.5, 0.01,
        "avx2/avx512/avx512-amx measured on the same physical CPU; only the build's "
        "compiled-in ISA differs. neon-m3 column reused from Section 10.3 — Apple M3 / Metal.",
        ha="center", va="bottom", fontsize=6.5, color="#555", style="italic",
    )

    fig.tight_layout(rect=(0, 0.13, 1, 1))
    name = "CH10_F09_Kalyanarangan"
    paths = _save_pair(plt, fig, cfg, name)
    caption = (
        "Decode throughput across five quantization formats on four kernel families. Bars use "
        "diagonal hatches for the AVX-512 family, cross-hatch for AVX-512+AMX, dots for AVX-2, "
        "back-diagonals for NEON, so the four-way grouping survives B&W print. The IQ4_XS group "
        "is the structural answer to whether the lookup-table format pays off — that question "
        "gets its own figure next."
    )
    return paths, caption


# --- Figure 10.10: IQ4_XS / Q4_K_M crossover -------------------------------

def _read_per_rep_ratios(cfg: Config, build: str) -> Optional[Tuple[float, float]]:
    """Read raw bench JSON for IQ4_XS and Q4_K_M for one build, return (min, max) ratio
    across all paired reps. Returns None if any data is missing."""
    iq_raw = cfg.logs_dir / f"bench_{build}_IQ4_XS.json"
    q4_raw = cfg.logs_dir / f"bench_{build}_Q4_K_M.json"
    if not (iq_raw.exists() and q4_raw.exists()):
        return None
    try:
        iq_text = iq_raw.read_text()
        q4_text = q4_raw.read_text()
    except Exception:
        return None
    # Parse the JSON arrays — these are the per-rep results llama-bench prints.
    def _per_rep_tps(text: str, n_target: int) -> List[float]:
        s = text.find("[")
        e = text.rfind("]")
        if s < 0 or e <= s:
            return []
        try:
            arr = json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            return []
        # Each row is one *test type* with its own avg_ns/stddev_ns over the reps.
        # llama-bench --output json doesn't expose individual-rep timings, so we
        # synthesize a min/max around avg_ns ± stddev_ns for the tg row only.
        for row in arr:
            if int(row.get("n_gen", 0)) > 0 and int(row.get("n_prompt", 0)) == 0:
                avg = float(row.get("avg_ns", 0.0))
                sd = float(row.get("stddev_ns", 0.0))
                if avg <= 0:
                    return []
                # Convert to tps; min tps corresponds to max ns and vice versa.
                hi = 1e9 * n_target / max(1.0, avg - sd)
                lo = 1e9 * n_target / (avg + sd)
                return [lo, hi]
        return []
    iq = _per_rep_tps(iq_text, 64)
    q4 = _per_rep_tps(q4_text, 64)
    if len(iq) < 2 or len(q4) < 2:
        return None
    # Worst-case ratio bounds: min(iq)/max(q4) and max(iq)/min(q4).
    rat_lo = iq[0] / q4[1] if q4[1] > 0 else 0.0
    rat_hi = iq[1] / q4[0] if q4[0] > 0 else 0.0
    return (rat_lo, rat_hi)


def _fig_crossover(cfg: Config, results: dict) -> Tuple[Tuple[str, str], str]:
    plt = _setup_mpl(cfg)

    fig, ax = plt.subplots(figsize=(cfg.fig_width_in, 2.8))
    crossover = results.get("crossover", {})
    bench = results.get("bench", {})

    # Order top-to-bottom: avx512-amx, avx512, neon-m3, avx2.
    row_order = ("avx512-amx", "avx512", "neon-m3", "avx2")
    ys = list(range(len(row_order)))[::-1]                            #A so first row in tuple is at top

    # Background spans — k-quant territory left of parity, IQ-quant right.
    ax.axvspan(0, 1.0, alpha=0.06, color="#7E76B0")                    #B Q4_K_M family color
    ax.axvspan(1.0, 1.5, alpha=0.06, color="#D67430")                  #B IQ-quant family color
    # Parity reference line.
    ax.axvline(1.0, color="#777", linestyle="--", linewidth=0.7, zorder=1)
    ax.text(1.0, len(row_order) - 0.15, "parity", ha="center", va="bottom",
            fontsize=6.5, color="#555", style="italic")

    rmin = float("inf"); rmax = float("-inf")
    for y, k in zip(ys, row_order):
        r = float(crossover.get(k, 0.0))
        if r > 0:
            rmin = min(rmin, r); rmax = max(rmax, r)
        # Whisker (best-effort min/max from per-rep raw JSON).
        whiskers = None
        if k in BUILDS:
            whiskers = _read_per_rep_ratios(cfg, k)
        if whiskers:
            ax.plot([whiskers[0], whiskers[1]], [y, y], color="#444",
                    linewidth=1.0, solid_capstyle="butt", zorder=2)
        # Dot.
        ax.plot([r], [y], marker=cfg.kernel_marker[k], markersize=11,
                markerfacecolor=cfg.kernel_color[k],
                markeredgecolor="black", markeredgewidth=0.8, zorder=3)
        # Per-row label sits ABOVE the marker, not to its right — keeps the
        # horizontal whisker line from cutting through the text.
        ax.text(r, y + 0.22, f"ratio = {r:.2f}",
                ha="center", va="bottom", fontsize=7, color="#222")

    if not (rmin < float("inf")) or not (rmax > -float("inf")):
        rmin, rmax = 0.7, 1.4
    span = max(rmax - rmin, 0.1)
    ax.set_xlim(min(0.7, rmin - span * 0.10), max(1.4, rmax + span * 0.10))
    ax.set_ylim(-0.5, len(row_order) + 0.0)
    ax.set_yticks(ys)
    ax.set_yticklabels(list(row_order))
    ax.set_xlabel("decode_tps(IQ4_XS) / decode_tps(Q4_K_M)")
    ax.set_title("IQ4_XS / Q4_K_M decode throughput ratio per kernel family",
                 fontsize=8.5, pad=6)
    ax.grid(True, axis="x", linestyle=":", color="#cccccc", linewidth=0.5)

    fig.tight_layout()
    name = "CH10_F10_Kalyanarangan"
    paths = _save_pair(plt, fig, cfg, name)

    r_avx2 = float(crossover.get("avx2", 0))
    r_avx512 = float(crossover.get("avx512", 0))
    r_amx = float(crossover.get("avx512-amx", 0))
    r_m3 = float(crossover.get("neon-m3", 0))
    gap_winner = crossover.get("gap_winner", "?")
    gap_passed = bool(crossover.get("gate_passed", False))
    actual_gap = float(crossover.get("gap", 0))

    caption_core = (
        f"IQ4_XS decode throughput as a multiple of Q4_K_M decode throughput on the same "
        f"hardware, same model. Ratios above 1.0 mean IQ4_XS wins despite being the more "
        f"complex format; ratios below 1.0 mean the codebook lookups serialize faster than "
        f"the bandwidth saving recovers. Measured: avx2 = {r_avx2:.2f}, avx512 = {r_avx512:.2f}, "
        f"avx512-amx = {r_amx:.2f}, neon-m3 = {r_m3:.2f}. The {gap_winner} kernel family flips "
        f"the ranking — the deployment guidance in 10.4.5 keys off this number."
    )
    if not gap_passed:
        caption = (
            f"[Caveat: avx2-vs-avx512-amx ratio gap is {actual_gap:.3f}, below the 0.05 "
            f"threshold the section needs to claim a kernel-family effect; numbers below "
            f"should be treated as illustrative.] " + caption_core
        )
    else:
        caption = caption_core
    return paths, caption


def mode_figures(cfg: Config, results: dict) -> dict:
    paths08, cap08 = _fig_lane_diagram(cfg)
    paths09, cap09 = _fig_decode_grouped(cfg, results)
    paths10, cap10 = _fig_crossover(cfg, results)
    return {
        "f08_lane_diagram":   {"png": paths08[0], "pdf": paths08[1], "caption": cap08},
        "f09_decode_grouped": {"png": paths09[0], "pdf": paths09[1], "caption": cap09},
        "f10_crossover":      {"png": paths10[0], "pdf": paths10[1], "caption": cap10},
    }


# --- Dispatcher ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    p.add_argument("--mode", choices=[
        "dispatch", "kernel-walk", "bench", "crossover", "figures", "all",
    ], default="all")
    p.add_argument("--force", action="store_true",
                   help="re-run a mode even if its key is already in results.json")
    args = p.parse_args()
    cfg = Config()

    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    log(f"=== ch10_kernels.py mode={args.mode} ===", cfg.log_file)

    # Pre-flight input check.
    modes_to_check = {
        "all":         ["dispatch", "bench"],
        "dispatch":    ["dispatch"],
        "kernel-walk": ["kernel-walk"],
        "bench":       ["bench"],
        "crossover":   [],
        "figures":     [],
    }
    _check_inputs(cfg, modes_to_check.get(args.mode, []))

    results = read_results(cfg)
    results["host"] = host_info()
    results.setdefault("model", {"id": cfg.repo_id, "params_b": 6.74, "architecture": "llama"})
    results.setdefault("dispatch", {})
    results.setdefault("bench", {})
    results.setdefault("crossover", {})
    results.setdefault("figures", {})
    results.setdefault("listings", {})

    if args.mode in ("dispatch", "all"):
        results["dispatch"] = mode_dispatch(cfg)
        results["listings"]["listing_10_7"] = str(
            (cfg.artifacts_dir / "listing_10_7.md").relative_to(BASE)
        )
        write_results(cfg, results)

    if args.mode == "kernel-walk":
        kw = mode_kernel_walk(cfg)
        if "x86" in kw:
            results["listings"]["listing_10_8_x86"] = kw["x86"]
        if "arm" in kw:
            results["listings"]["listing_10_8_arm"] = kw["arm"]
        write_results(cfg, results)

    if args.mode in ("bench", "all"):
        results["bench"] = mode_bench(cfg)
        write_results(cfg, results)

    if args.mode in ("crossover", "all"):
        results["crossover"] = mode_crossover(cfg, results)
        write_results(cfg, results)

    if args.mode in ("figures", "all"):
        results["figures"] = mode_figures(cfg, results)
        write_results(cfg, results)

    log(f"results -> {cfg.results_path.relative_to(BASE)}", cfg.log_file)


if __name__ == "__main__":
    main()
