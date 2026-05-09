#!/usr/bin/env python3
"""Section 10.5 companion script — runtime memory & throughput.

Self-contained: clones llama.cpp at a pinned commit, builds an AVX-512+VNNI
binary, downloads two Q4_K_M GGUFs, measures memory footprint and decode
throughput, and renders Figure 10.11, Figure 10.12, and Table 10.2 from a
single results.json artifact.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
MODELS = ROOT / "_models"
FIGS = ROOT / "figures"
RESULTS = ROOT / "results.json"
LLAMA_DIR = ROOT / "llama.cpp"
BUILD_DIR = ROOT / "build-avx512"


@dataclass(frozen=True)
class Config:
    llamacpp_commit: str = "6d57a49"
    llamacpp_repo: str = "https://github.com/ggerganov/llama.cpp.git"
    model_specs: tuple = (
        ("llama-2-7b",
         "TheBloke/Llama-2-7B-GGUF",
         "llama-2-7b.Q4_K_M.gguf"),
        ("llama-3-8b",
         "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
         "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"),
    )
    contexts: tuple = (2048, 4096)
    thread_counts: tuple = (1, 2, 4, 6, 8, 10, 11, 12, 14, 16, 18, 20, 22)
    n_repetitions: int = 5
    n_decode_tokens: int = 64
    threads_sweep_model: str = "llama-2-7b"


CFG = Config()
SPEC_VERSION = "10.5.0"


# ---------- helpers ---------------------------------------------------------

def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_dirs() -> None:
    for d in (LOGS, MODELS, FIGS):
        d.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str] | str, *, log_path: Path, append: bool = True,
         cwd: Optional[Path] = None, check: bool = True,
         env: Optional[dict] = None, capture_stdout_to: Optional[Path] = None) -> int:
    """Run a subprocess streaming stderr (and optionally stdout) to a log file.

    Returns the process return code. Raises CalledProcessError if check and rc!=0.
    """
    if isinstance(cmd, list):
        printable = " ".join(map(str, cmd))
    else:
        printable = str(cmd)
    mode = "a" if append else "w"
    with open(log_path, mode) as logf:
        logf.write(f"\n$ {printable}\n")
        logf.flush()
        if capture_stdout_to is not None:
            with open(capture_stdout_to, "w") as outf:
                proc = subprocess.run(
                    cmd, cwd=str(cwd) if cwd else None, env=env,
                    stdout=outf, stderr=logf, shell=isinstance(cmd, str),
                )
        else:
            proc = subprocess.run(
                cmd, cwd=str(cwd) if cwd else None, env=env,
                stdout=logf, stderr=subprocess.STDOUT, shell=isinstance(cmd, str),
            )
    if check and proc.returncode != 0:
        sys.stderr.write(f"\nCommand failed (rc={proc.returncode}): {printable}\n"
                         f"See {log_path}\n")
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    return proc.returncode


def _read_results() -> dict:
    if RESULTS.exists():
        with open(RESULTS) as f:
            return json.load(f)
    return {"spec_version": SPEC_VERSION}


def _write_results(d: dict) -> None:
    d["generated_utc"] = _utcnow()
    d["spec_version"] = SPEC_VERSION
    tmp = RESULTS.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(d, f, indent=2, sort_keys=False)
    tmp.replace(RESULTS)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _host_info() -> dict:
    cpu_name = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except FileNotFoundError:
        pass
    ram_kb = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_kb = int(line.split()[1])
                    break
    except FileNotFoundError:
        pass
    ram_gib = round(ram_kb / 1024 / 1024, 1)
    vcpus = os.cpu_count() or 0
    physical = vcpus
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        cs = ts = None
        for line in out.splitlines():
            if line.startswith("Core(s) per socket:"):
                cs = int(line.split(":")[1].strip())
            elif line.startswith("Socket(s):"):
                ts = int(line.split(":")[1].strip())
        if cs and ts:
            physical = cs * ts
    except Exception:
        pass
    return {
        "cpu": cpu_name,
        "vcpus": vcpus,
        "physical_cores": physical,
        "ram_gib": ram_gib,
        "kernel": platform.release(),
        "platform": platform.platform(),
    }


# ---------- mode: setup -----------------------------------------------------

def cmd_setup() -> None:
    _ensure_dirs()
    log = LOGS / "build_avx512.log"
    print(f"[setup] log -> {log}")
    log.write_text(f"# setup log {_utcnow()}\n")

    # 1. apt packages
    apt_pkgs = ["build-essential", "cmake", "git", "python3-pip", "libcurl4-openssl-dev"]
    print("[setup] apt-get update + install ...")
    _run(["sudo", "apt-get", "update", "-y"], log_path=log)
    _run(["sudo", "apt-get", "install", "-y", *apt_pkgs], log_path=log)

    # 2. Python packages — Debian 12 enforces PEP 668 so use --break-system-packages
    print("[setup] pip install matplotlib numpy huggingface-hub gguf ...")
    pip_cmd = [sys.executable, "-m", "pip", "install", "--user",
               "--break-system-packages",
               "matplotlib", "numpy", "huggingface-hub", "gguf"]
    try:
        _run(pip_cmd, log_path=log)
    except subprocess.CalledProcessError:
        # Older pip without --break-system-packages flag: retry without it
        with open(log, "a") as f:
            f.write("\n# retrying pip without --break-system-packages\n")
        _run([sys.executable, "-m", "pip", "install", "--user",
              "matplotlib", "numpy", "huggingface-hub", "gguf"], log_path=log)

    # 3. Clone llama.cpp at pinned commit
    if LLAMA_DIR.exists():
        print(f"[setup] llama.cpp/ exists -> fetch + checkout {CFG.llamacpp_commit}")
        _run(["git", "fetch", "--all", "--tags"], log_path=log, cwd=LLAMA_DIR)
        _run(["git", "checkout", CFG.llamacpp_commit], log_path=log, cwd=LLAMA_DIR)
    else:
        print(f"[setup] git clone {CFG.llamacpp_repo}")
        _run(["git", "clone", CFG.llamacpp_repo, str(LLAMA_DIR)], log_path=log)
        _run(["git", "checkout", CFG.llamacpp_commit], log_path=log, cwd=LLAMA_DIR)

    # 4. Build AVX-512+VNNI binary, AMX off
    print(f"[setup] cmake configure -> {BUILD_DIR}")
    cmake_args = [
        "cmake", "-B", str(BUILD_DIR), "-S", str(LLAMA_DIR),
        "-DGGML_AVX2=ON", "-DGGML_AVX512=ON", "-DGGML_AVX512_VNNI=ON",
        "-DGGML_AMX_TILE=OFF", "-DGGML_AMX_INT8=OFF", "-DGGML_AMX_BF16=OFF",
        "-DGGML_NATIVE=OFF", "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_CURL=ON",
    ]
    _run(cmake_args, log_path=log)
    print("[setup] cmake build llama-completion + llama-bench ...")
    # llama-cli at this commit is interactive-only; the binary explicitly directs
    # one-shot usage to llama-completion. Build that target instead.
    _run(["cmake", "--build", str(BUILD_DIR),
          "--target", "llama-completion", "llama-bench", "-j"], log_path=log)

    # Verify build features. At commit 6d57a49 `llama-cli --version` no longer
    # prints the GGML cap line — it only emits version + compiler. The
    # feature line is printed by print_system_info() during model load. So
    # verify configuration from CMakeCache.txt (the compile-time truth) and
    # capture the runtime line later in cmd_memory.
    bin_cli = BUILD_DIR / "bin" / "llama-completion"
    if not bin_cli.exists():
        raise RuntimeError(f"llama-completion not found at {bin_cli}")
    cache = (BUILD_DIR / "CMakeCache.txt").read_text()
    feats: dict[str, str] = {}
    for line in cache.splitlines():
        m = re.match(r"GGML_([A-Z0-9_]+):BOOL=(ON|OFF)", line)
        if m:
            feats[m.group(1)] = "1" if m.group(2) == "ON" else "0"
    if feats.get("AVX512_VNNI") != "1":
        raise RuntimeError(
            f"Build verification failed: AVX512_VNNI != ON in CMakeCache. feats={feats}")
    if feats.get("AMX_INT8", "0") != "0":
        raise RuntimeError(
            f"Build verification failed: AMX_INT8 must be OFF, got {feats.get('AMX_INT8')}")

    feature_summary = ", ".join(f"{k}={v}" for k, v in sorted(feats.items()))

    # 5. Download GGUFs to _models/
    from huggingface_hub import hf_hub_download  # type: ignore

    model_records: list[dict] = []
    for name, repo, fname in CFG.model_specs:
        target = MODELS / fname
        if target.exists() and target.stat().st_size > 0:
            print(f"[setup] {name}: cached at {target} ({target.stat().st_size:,} bytes)")
            with open(log, "a") as f:
                f.write(f"\n# {name}: cached at {target}\n")
        else:
            print(f"[setup] downloading {repo}/{fname} -> {target}")
            with open(log, "a") as f:
                f.write(f"\n# downloading {repo}/{fname}\n")
            try:
                downloaded = hf_hub_download(
                    repo_id=repo, filename=fname,
                    local_dir=str(MODELS),
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {repo}/{fname}: {e}\n"
                    f"Both repos are public Q4_K_M quants — check network."
                )
            dp = Path(downloaded)
            if dp.resolve() != target.resolve():
                if target.exists():
                    target.unlink()
                shutil.copy2(dp, target)
        size = target.stat().st_size
        sha = _sha256(target)
        model_records.append({
            "name": name, "repo": repo, "file": fname,
            "path": str(target.relative_to(ROOT)),
            "size_bytes": size, "sha256": sha,
        })

    results = _read_results()
    results["setup"] = {
        "host": _host_info(),
        "llamacpp_commit": CFG.llamacpp_commit,
        "build_features": feature_summary,
        "models": model_records,
        "timestamp_utc": _utcnow(),
    }
    _write_results(results)
    print(f"[setup] done -> {RESULTS}")


# ---------- mode: memory ----------------------------------------------------

# Defensive regexes (case-insensitive, leading whitespace tolerated, optional prefixes).
# At commit 6d57a49 the binary tags shifted: `model size` -> `file size`,
# `KV self size` -> `llama_kv_cache: size`. Patterns updated accordingly.
_RE_WEIGHTS = re.compile(
    r"(?:file|model)\s*size\s*=\s*([\d.]+)\s*GiB", re.IGNORECASE)
_RE_NLAYER  = re.compile(r"n_layer\s*=\s*(\d+)", re.IGNORECASE)
_RE_NHEAD   = re.compile(r"n_head\s*=\s*(\d+)", re.IGNORECASE)
_RE_NHEADKV = re.compile(r"n_head_kv\s*=\s*(\d+)", re.IGNORECASE)
_RE_HEADDIM = re.compile(r"n_embd_head_k\s*=\s*(\d+)", re.IGNORECASE)
# KV total: at this commit the line is `llama_kv_cache: size = 1024.00 MiB`;
# older versions used `KV self size = ...`. Match both.
_RE_KVTOTAL = re.compile(
    r"(?:KV\s*self\s*size|kv_cache\s*:\s*size)\s*=\s*([\d.]+)\s*MiB",
    re.IGNORECASE,
)
_RE_KVKV    = re.compile(
    r"K\s*\([^)]*\)\s*:\s*([\d.]+)\s*MiB.*?V\s*\([^)]*\)\s*:\s*([\d.]+)\s*MiB",
    re.IGNORECASE | re.DOTALL,
)
_RE_COMPBUF = re.compile(
    r"(?:CPU\s+)?compute\s*buffer\s*size\s*=\s*([\d.]+)\s*MiB",
    re.IGNORECASE,
)


def _parse_memory_log(log_text: str) -> dict:
    """Parse a llama-cli stderr log; raise with a slice if any required field is missing."""

    def _one(rx: re.Pattern, label: str, *, group: int = 1) -> str:
        m = rx.search(log_text)
        if not m:
            # Dump a relevant slice — first 60 lines containing "load" or any header
            head = "\n".join(log_text.splitlines()[:200])
            raise RuntimeError(
                f"Memory regex miss for '{label}'. Pattern={rx.pattern!r}\n"
                f"--- log head (first 200 lines) ---\n{head}\n--- end ---")
        return m.group(group)

    weights_gib = float(_one(_RE_WEIGHTS, "weights model size GiB"))
    n_layers = int(_one(_RE_NLAYER, "n_layer"))
    # For n_head, find a match that is explicitly "n_head =" not "n_head_kv ="
    # We search for a line that does NOT contain n_head_kv before the match.
    n_head = None
    for line in log_text.splitlines():
        # match "n_head = N" but exclude "n_head_kv"
        if re.search(r"n_head\s*=", line, re.IGNORECASE) and \
           not re.search(r"n_head_kv|n_head_k\b", line, re.IGNORECASE):
            mm = _RE_NHEAD.search(line)
            if mm:
                n_head = int(mm.group(1))
                break
    if n_head is None:
        # fall back to first n_head match anyway
        n_head = int(_one(_RE_NHEAD, "n_head"))
    n_head_kv = int(_one(_RE_NHEADKV, "n_head_kv"))
    head_dim = int(_one(_RE_HEADDIM, "n_embd_head_k"))
    kv_total = float(_one(_RE_KVTOTAL, "KV self size MiB"))
    kv_split = _RE_KVKV.search(log_text)
    if kv_split:
        kv_k = float(kv_split.group(1))
        kv_v = float(kv_split.group(2))
    else:
        # Some builds split K and V on different lines: search separately
        m_k = re.search(r"K\s*\([^)]*\)\s*:\s*([\d.]+)\s*MiB", log_text, re.IGNORECASE)
        m_v = re.search(r"V\s*\([^)]*\)\s*:\s*([\d.]+)\s*MiB", log_text, re.IGNORECASE)
        if m_k and m_v:
            kv_k = float(m_k.group(1))
            kv_v = float(m_v.group(1))
        else:
            # Fall back: assume even split
            kv_k = kv_v = kv_total / 2.0
    comp_buf_matches = _RE_COMPBUF.findall(log_text)
    if not comp_buf_matches:
        raise RuntimeError(
            "Memory regex miss for compute buffer.\n"
            f"--- log tail (last 200 lines) ---\n"
            + "\n".join(log_text.splitlines()[-200:]) + "\n--- end ---")
    # Take the largest compute buffer reported (model fwd, not KV-only scratch)
    comp_buf = max(float(x) for x in comp_buf_matches)

    return {
        "weights_gib": weights_gib,
        "weights_mib": weights_gib * 1024.0,
        "n_layers": n_layers,
        "n_head": n_head,
        "n_head_kv": n_head_kv,
        "head_dim": head_dim,
        "kv_total_mib": kv_total,
        "kv_k_mib": kv_k,
        "kv_v_mib": kv_v,
        "compute_buf_mib": comp_buf,
    }


def cmd_memory() -> None:
    _ensure_dirs()
    bin_cli = BUILD_DIR / "bin" / "llama-completion"
    if not bin_cli.exists():
        raise RuntimeError(f"{bin_cli} missing — run --mode setup first")

    results = _read_results()
    mem_section: dict[str, Any] = {}
    for name, _repo, fname in CFG.model_specs:
        gguf = MODELS / fname
        if not gguf.exists():
            raise RuntimeError(f"{gguf} missing — run --mode setup first")
        ctx_records: dict[str, Any] = {}
        arch_record: Optional[dict] = None
        weights_mib_record: Optional[float] = None
        for ctx in CFG.contexts:
            log = LOGS / f"mem_{name}_{ctx}.log"
            print(f"[memory] {name} ctx={ctx} -> {log}")
            cmd = [
                str(bin_cli),
                "-m", str(gguf),
                "-c", str(ctx),
                "-n", "1",
                "-p", "test",
                "--no-warmup",
                "-t", "11",
            ]
            log.write_text(f"# {' '.join(cmd)}\n")
            with open(log, "a") as logf:
                proc = subprocess.run(
                    cmd, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=logf,
                    timeout=180,
                )
            if proc.returncode != 0:
                # Some llama-cli versions return non-zero on EOF after -n 1; still parse if log has data
                with open(log, "a") as logf:
                    logf.write(f"\n# llama-cli exited rc={proc.returncode}; attempting to parse anyway\n")
            text = log.read_text()
            parsed = _parse_memory_log(text)

            kv_analytical = (
                2 * parsed["n_layers"] * parsed["n_head_kv"] * parsed["head_dim"]
                * ctx * 2 / (1024 ** 2)
            )
            match_pct = 100.0 * parsed["kv_total_mib"] / kv_analytical
            err = abs(parsed["kv_total_mib"] - kv_analytical) / kv_analytical
            if err >= 0.02:
                raise RuntimeError(
                    f"[{name} ctx={ctx}] KV mismatch: parsed={parsed['kv_total_mib']:.2f} MiB "
                    f"vs analytical={kv_analytical:.2f} MiB (err={err*100:.2f}%). "
                    f"Regex is wrong, not the model.")

            ctx_records[str(ctx)] = {
                "kv_total_mib": parsed["kv_total_mib"],
                "kv_k_mib": parsed["kv_k_mib"],
                "kv_v_mib": parsed["kv_v_mib"],
                "compute_buf_mib": parsed["compute_buf_mib"],
                "kv_analytical_mib": round(kv_analytical, 4),
                "kv_match_pct": round(match_pct, 4),
            }
            if arch_record is None:
                arch_record = {
                    "n_layers": parsed["n_layers"],
                    "n_head": parsed["n_head"],
                    "n_head_kv": parsed["n_head_kv"],
                    "head_dim": parsed["head_dim"],
                }
                weights_mib_record = round(parsed["weights_mib"], 2)
        mem_section[name] = {
            "arch": arch_record,
            "weights_mib": weights_mib_record,
            "ctx": ctx_records,
        }
    results["memory"] = mem_section
    _write_results(results)
    print(f"[memory] done -> {RESULTS}")


# ---------- mode: threads ---------------------------------------------------

def _try_governor() -> str:
    log = LOGS / "governor.log"
    try:
        proc = subprocess.run(
            ["sudo", "-n", "cpupower", "frequency-set", "-g", "performance"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        log.write_text(proc.stdout or "")
        if proc.returncode == 0:
            return "set:performance"
        return f"failed:rc={proc.returncode}"
    except FileNotFoundError:
        log.write_text("cpupower not found — skipping governor pin\n")
        return "missing:cpupower"


def cmd_threads() -> None:
    _ensure_dirs()
    bin_bench = BUILD_DIR / "bin" / "llama-bench"
    if not bin_bench.exists():
        raise RuntimeError(f"{bin_bench} missing — run --mode setup first")

    name = CFG.threads_sweep_model
    fname = next(f for n, _r, f in CFG.model_specs if n == name)
    gguf = MODELS / fname
    if not gguf.exists():
        raise RuntimeError(f"{gguf} missing — run --mode setup first")

    governor_state = _try_governor()
    print(f"[threads] governor: {governor_state}")

    order = list(CFG.thread_counts)
    rng = random.Random(42)
    rng.shuffle(order)
    print(f"[threads] shuffled order: {order}")

    sweep: dict[str, Any] = {}
    for t in order:
        tt = f"{t:02d}"
        log = LOGS / f"threads_t{tt}.log"
        outj = LOGS / f"threads_t{tt}.json"
        print(f"[threads] t={t} -> {outj}")
        cmd = [
            str(bin_bench),
            "-m", str(gguf),
            "-p", "0",
            "-n", str(CFG.n_decode_tokens),
            "-t", str(t),
            "-r", str(CFG.n_repetitions),
            "--output", "json",
        ]
        log.write_text(f"# {' '.join(cmd)}\n")
        with open(log, "a") as logf, open(outj, "w") as outf:
            proc = subprocess.run(cmd, stdout=outf, stderr=logf)
        if proc.returncode != 0:
            raise RuntimeError(f"llama-bench failed for t={t} rc={proc.returncode}; see {log}")

        with open(outj) as f:
            data = json.load(f)
        if not isinstance(data, list) or not data:
            raise RuntimeError(f"llama-bench JSON unexpected: {data!r} (file={outj})")
        rec = data[0]
        if "avg_ts" not in rec:
            raise RuntimeError(
                f"llama-bench JSON missing avg_ts (t={t}). Raw record:\n{json.dumps(rec, indent=2)}")
        avg_ts = float(rec["avg_ts"])
        std_ts = float(rec.get("stddev_ts", 0.0))
        samples = rec.get("samples_ts")
        if samples is None and "samples_ns" in rec:
            n_gen = float(rec.get("n_gen", CFG.n_decode_tokens))
            samples = [n_gen * 1e9 / s for s in rec["samples_ns"] if s > 0]
        if samples is None:
            samples = []
        sweep[str(t)] = {
            "tok_per_sec": avg_ts,
            "stddev": std_ts,
            "samples": [float(s) for s in samples],
        }

    # Re-order sweep keys ascending for tidy JSON
    sweep_sorted = {str(t): sweep[str(t)] for t in sorted(int(k) for k in sweep)}
    peak_t = max(sweep_sorted, key=lambda k: sweep_sorted[k]["tok_per_sec"])
    peak_v = sweep_sorted[peak_t]["tok_per_sec"]
    t22 = sweep_sorted.get("22")
    t22_record = None
    if t22 is not None:
        regression = 100.0 * (peak_v - t22["tok_per_sec"]) / peak_v
        t22_record = {
            "tok_per_sec": t22["tok_per_sec"],
            "regression_vs_peak_pct": round(regression, 3),
        }

    results = _read_results()
    results["threads"] = {
        "model": name,
        "format": "Q4_K_M",
        "build": "avx512+vnni",
        "n_decode_tokens": CFG.n_decode_tokens,
        "n_repetitions": CFG.n_repetitions,
        "shuffled_order": order,
        "governor": governor_state,
        "sweep": sweep_sorted,
        "peak": {"t": int(peak_t), "tok_per_sec": peak_v},
        "t22": t22_record,
    }
    _write_results(results)
    print(f"[threads] done -> {RESULTS}")


# ---------- mode: figures ---------------------------------------------------

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    })
    return plt


def _save_pair(fig, stem: str) -> list[str]:
    out_png = FIGS / f"{stem}.png"
    out_pdf = FIGS / f"{stem}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    return [str(out_png.relative_to(ROOT)), str(out_pdf.relative_to(ROOT))]


def _fig_memory(results: dict, plt) -> tuple[list[str], str]:
    mem = results["memory"]
    bars: list[tuple[str, float, float, float]] = []  # (label, weights, kv, compute)
    label_order = [
        ("Llama-2-7B, ctx=2048", "llama-2-7b", "2048"),
        ("Llama-2-7B, ctx=4096", "llama-2-7b", "4096"),
        ("Llama-3-8B, ctx=2048", "llama-3-8b", "2048"),
        ("Llama-3-8B, ctx=4096", "llama-3-8b", "4096"),
    ]
    for label, model, ctx in label_order:
        block = mem[model]
        c = block["ctx"][ctx]
        bars.append((label, block["weights_mib"], c["kv_total_mib"], c["compute_buf_mib"]))

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    y = list(range(len(bars)))
    labels = [b[0] for b in bars]
    weights = [b[1] for b in bars]
    kv      = [b[2] for b in bars]
    compbuf = [b[3] for b in bars]

    weights_color = "0.55"
    kv_color      = "0.80"
    comp_color    = "0.92"

    ax.barh(y, weights, color=weights_color, edgecolor="black",
            label="weights", linewidth=0.6)
    ax.barh(y, kv, left=weights, color=kv_color, edgecolor="black",
            hatch="///", label="KV cache", linewidth=0.6)
    bottom2 = [w + k for w, k in zip(weights, kv)]
    ax.barh(y, compbuf, left=bottom2, color=comp_color, edgecolor="black",
            hatch="xxx", label="compute buffer", linewidth=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("MiB")
    ax.set_title("Figure 10.11 Runtime memory footprint, Q4_K_M (weights + KV + compute)")

    # Per-segment annotations. A segment narrower than `min_inline_mib` gets its
    # label pinned to the right of the bar with a leader; otherwise centered inline.
    totals_max = max(w + k + c for w, k, c in zip(weights, kv, compbuf))
    pad = totals_max * 0.012
    min_inline = totals_max * 0.05  # ~5% of widest bar
    ax.set_xlim(0, totals_max * 1.18)

    for i, (w, k, c) in enumerate(zip(weights, kv, compbuf)):
        # Weights — always wide enough for inline
        ax.text(w / 2, i, f"{w:.1f}", va="center", ha="center", fontsize=7)

        # KV
        if k >= min_inline:
            ax.text(w + k / 2, i, f"{k:.1f}", va="center", ha="center", fontsize=7)
            kv_label_outside = False
        else:
            kv_label_outside = True

        # Compute buffer
        comp_inline = c >= min_inline and not kv_label_outside  # avoid stacking outside
        if comp_inline:
            ax.text(w + k + c / 2, i, f"{c:.1f}", va="center", ha="center", fontsize=7)

        # Outside labels (right of bar) for narrow segments
        x_outside = w + k + c + pad
        if kv_label_outside and c >= min_inline:
            # rare: KV narrow but compute wide — label only KV outside
            ax.text(x_outside, i, f"KV {k:.1f}", va="center", ha="left", fontsize=7)
        elif kv_label_outside and c < min_inline:
            ax.text(x_outside, i, f"KV {k:.1f}, comp {c:.1f}",
                    va="center", ha="left", fontsize=7)
        elif not comp_inline:
            ax.text(x_outside, i, f"comp {c:.1f}", va="center", ha="left", fontsize=7)

    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, which="major", linestyle="--", alpha=0.35)
    ax.yaxis.grid(False)

    fig.tight_layout()
    paths = _save_pair(fig, "CH10_F11_Kalyanarangan")
    plt.close(fig)

    # Caption: cite the L3-8B-4096 KV vs L2-7B-2048 KV comparison
    l3_4k_kv = mem["llama-3-8b"]["ctx"]["4096"]["kv_total_mib"]
    l2_2k_kv = mem["llama-2-7b"]["ctx"]["2048"]["kv_total_mib"]
    caption = (
        "**Figure 10.11 Runtime memory footprint for Q4_K_M Llama-2-7B "
        "(MHA, 32 KV heads) and Q4_K_M Llama-3-8B (GQA, 8 KV heads) at "
        "2048 and 4096 context. Weights are nearly fixed across context "
        "lengths; the KV cache (`///` hatching) scales linearly with context "
        "and dominates the footprint at long ctx for the MHA model. The "
        "compute buffer (`xxx` hatching) — activation scratch the runtime "
        "allocates per forward pass — is small in both cases. Llama-3-8B's "
        "4× reduction in KV heads (GQA) cuts the KV cache footprint by 4× "
        "at the same context: the Llama-3-8B 4096-ctx bar's KV segment is "
        f"{l3_4k_kv:.1f} MiB versus {l2_2k_kv:.1f} MiB for the Llama-2-7B "
        "2048-ctx bar despite holding twice as many tokens of state. "
        "Numbers measured from llama-cli stderr on c3-standard-22, llama.cpp "
        f"build at commit {CFG.llamacpp_commit}.**"
    )
    return paths, caption


def _fig_threads(results: dict, plt) -> tuple[list[str], str]:
    th = results["threads"]
    sweep = th["sweep"]
    ts = sorted(int(k) for k in sweep)
    avg = [sweep[str(t)]["tok_per_sec"] for t in ts]
    std = [sweep[str(t)]["stddev"] for t in ts]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    err_low = [min(s, a) for s, a in zip(std, avg)]
    err_high = list(std)
    ax.errorbar(ts, avg, yerr=[err_low, err_high], fmt="-o", color="0.15",
                ecolor="0.45", capsize=2.5, linewidth=1.2, markersize=4.5,
                markerfacecolor="0.15")

    ax.set_xlabel("llama.cpp `-t` thread count")
    ax.set_ylabel("decode throughput (tokens / second)")
    ax.set_title("Figure 10.12 Decode throughput vs thread count, Q4_K_M Llama-2-7B")
    ax.set_xticks(ts)
    ax.set_xlim(0, max(ts) + 1.5)  # extra room on the right so peak label fits
    ax.set_axisbelow(True)
    ymin = 0
    ymax = max(avg) + max(std) * 2.0 + 2.0
    ax.set_ylim(ymin, ymax)

    # Vertical reference lines
    ax.axvline(11, linestyle="--", color="0.50", linewidth=0.9)
    ax.axvline(22, linestyle="--", color="0.50", linewidth=0.9)
    ax.text(11.15, ymax * 0.98, "physical cores",
            color="0.30", fontsize=7, rotation=90, va="top", ha="left")
    ax.text(22.15, ymax * 0.98, "vCPUs (SMT)",
            color="0.30", fontsize=7, rotation=90, va="top", ha="left")

    peak_t = th["peak"]["t"]
    peak_v = th["peak"]["tok_per_sec"]
    # When peak == 22, place the peak callout to the LEFT of the marker so it
    # doesn't fall off the right edge; also skip the redundant t=22 callout.
    if peak_t == 22:
        ax.annotate(
            f"peak: t={peak_t}\n{peak_v:.2f} tok/s",
            xy=(peak_t, peak_v),
            xytext=(peak_t - 5.0, peak_v - ymax * 0.18),
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color="0.30", lw=0.6),
            ha="left",
        )
    else:
        ax.annotate(
            f"peak: t={peak_t}, {peak_v:.2f} tok/s",
            xy=(peak_t, peak_v),
            xytext=(peak_t + 0.5, peak_v + ymax * 0.06),
            fontsize=7,
            arrowprops=dict(arrowstyle="->", color="0.30", lw=0.6),
        )
        if "22" in sweep:
            v22 = sweep["22"]["tok_per_sec"]
            regr = th["t22"]["regression_vs_peak_pct"] if th.get("t22") else 0.0
            ax.annotate(
                f"t=22: {v22:.2f} tok/s\n({regr:+.1f}% vs peak)",
                xy=(22, v22),
                xytext=(22 - 6, v22 - ymax * 0.18),
                fontsize=7,
                arrowprops=dict(arrowstyle="->", color="0.30", lw=0.6),
            )

    fig.tight_layout()
    paths = _save_pair(fig, "CH10_F12_Kalyanarangan")
    plt.close(fig)

    # Caption adapts to observed shape: peak < 22 → regression narrative;
    # peak == 22 → SMT-helps narrative. Either way report the t=11 → t=12 step.
    t11 = sweep.get("11", {}).get("tok_per_sec")
    t12 = sweep.get("12", {}).get("tok_per_sec")
    t22_v = sweep.get("22", {}).get("tok_per_sec", 0.0)
    regr_pct = th["t22"]["regression_vs_peak_pct"] if th.get("t22") else 0.0
    smt_dip_blurb = ""
    if t11 is not None and t12 is not None and t12 < t11:
        dip_pct = 100.0 * (t11 - t12) / t11
        smt_dip_blurb = (
            f"A brief dip at t=12 ({t12:.2f} tok/s, {dip_pct:.1f}% below t=11) "
            "is visible as the first SMT sibling lands on a core whose primary "
            "thread is already saturating its private L1/L2 bandwidth. "
        )
    if peak_t < 22:
        peak_blurb = (
            f"Throughput climbs through the physical-core count, peaks at "
            f"t={peak_t} at {peak_v:.2f} tok/s, and degrades as additional "
            "SMT siblings are recruited because the second hyperthread on "
            "each core fights for the same L1/L2 bandwidth the first is "
            "already saturating. "
            f"Running with `-t $(nproc)` on this host yields {t22_v:.2f} "
            f"tok/s — a {regr_pct:.1f}% regression versus the optimum. "
        )
    else:
        peak_blurb = (
            "Throughput continues to scale through the full vCPU count: "
            f"peak is at t={peak_t} at {peak_v:.2f} tok/s, with no "
            "regression at `-t $(nproc)`. On this VM the decode kernel is "
            "not yet L1/L2-bandwidth-bound at 11 cores, so the SMT siblings "
            "still extract additional throughput from the integer/load "
            "ports rather than fighting for cache. "
        )
    caption = (
        "**Figure 10.12 Decode throughput on c3-standard-22 (Sapphire Rapids, "
        "11 physical cores × 2 SMT) versus llama.cpp `-t` thread count, "
        "AVX-512+VNNI build, Q4_K_M Llama-2-7B, n=64 token generation, "
        f"mean over {CFG.n_repetitions} repetitions, error bars ±1σ. "
        + peak_blurb
        + smt_dip_blurb
        + "The dashed line at t=11 marks the physical core count; the dashed "
        "line at t=22 marks the full vCPU count.**"
    )
    return paths, caption


def _table_arch(results: dict) -> str:
    mem = results["memory"]
    rows = []
    for name in ("llama-2-7b", "llama-3-8b"):
        b = mem[name]
        a = b["arch"]
        c2 = b["ctx"]["2048"]
        c4 = b["ctx"]["4096"]
        display = "Llama-2-7B" if name == "llama-2-7b" else "Llama-3-8B"
        rows.append((display, a["n_layers"], a["n_head"], a["n_head_kv"],
                     a["head_dim"], b["weights_mib"],
                     c2["kv_total_mib"], c4["kv_total_mib"],
                     c4["compute_buf_mib"]))
    md = []
    md.append("##### Table 10.2 Architecture and runtime memory at ctx=4096")
    md.append("")
    md.append("| Model | Layers | Heads | KV heads | head_dim | Weights (MiB) | "
              "KV cache @ 2k (MiB) | KV cache @ 4k (MiB) | Compute buf (MiB) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        md.append(
            f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | "
            f"{r[5]:.2f} | {r[6]:.2f} | {r[7]:.2f} | {r[8]:.2f} |"
        )
    md.append("")
    return "\n".join(md)


def cmd_figures() -> None:
    _ensure_dirs()
    plt = _setup_matplotlib()
    results = _read_results()
    if "memory" not in results or "threads" not in results:
        raise RuntimeError("results.json missing 'memory' or 'threads' — run those modes first")

    f11_paths, f11_cap = _fig_memory(results, plt)
    f12_paths, f12_cap = _fig_threads(results, plt)

    # Captions side-cars (numbers filled in)
    (FIGS / "CH10_F11_Kalyanarangan.caption.txt").write_text(f11_cap + "\n")
    (FIGS / "CH10_F12_Kalyanarangan.caption.txt").write_text(f12_cap + "\n")

    # Table 10.2
    table_md = _table_arch(results)
    table_path = FIGS / "CH10_T2_Kalyanarangan.md"
    table_path.write_text(table_md + "\n")

    results["figures"] = [
        {"id": "10.11", "stem": "CH10_F11_Kalyanarangan",
         "paths": f11_paths, "caption": f11_cap},
        {"id": "10.12", "stem": "CH10_F12_Kalyanarangan",
         "paths": f12_paths, "caption": f12_cap},
    ]
    results["tables"] = [
        {"id": "10.2", "path": str(table_path.relative_to(ROOT))},
    ]
    _write_results(results)
    print(f"[figures] done — {len(results['figures'])} figures, "
          f"{len(results['tables'])} tables -> {RESULTS}")


# ---------- mode: all -------------------------------------------------------

def cmd_all() -> None:
    cmd_setup()
    cmd_memory()
    cmd_threads()
    cmd_figures()


# ---------- entrypoint ------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True,
                   choices=("setup", "memory", "threads", "figures", "all"))
    args = p.parse_args()
    {
        "setup": cmd_setup,
        "memory": cmd_memory,
        "threads": cmd_threads,
        "figures": cmd_figures,
        "all": cmd_all,
    }[args.mode]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
