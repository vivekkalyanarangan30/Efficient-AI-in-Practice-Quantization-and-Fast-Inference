"""
Chapter 10, Section 10.3 — Convert large language models to CPU-friendly artifacts
Companion script for "Efficient AI in Practice: Quantization and Fast Inference"

What this script demonstrates:
  Section 10.3 walks the full HF safetensors -> GGUF F16 -> quantized variants
  pipeline. This script orchestrates that pipeline end-to-end against the same
  reference model used in section 10.2 (meta-llama/Llama-2-7b-hf), runs verification
  (token preview + perplexity) on five artifacts (F16 + four quants), and renders
  the four publication-ready figures Manning needs (Figures 10.4 - 10.7).

  Every measurement plotted in the figures traces back to a real subprocess
  invocation on the host machine - sizes via os.path.getsize, perplexity via
  llama-perplexity, wall-clocks via time.perf_counter. No interpolation, no
  defaults, no estimates.

Pipeline (--mode all):
  1. setup     - clone + build llama.cpp (Metal-enabled) into ch10/_build/llama.cpp/
  2. download  - HF snapshot_download of meta-llama/Llama-2-7b-hf
  3. convert   - convert_hf_to_gguf.py --outtype f16
  4. imatrix   - llama-imatrix on a wikitext-2-raw-v1 slice (CPU)
  5. quantize  - llama-quantize for Q8_0, Q5_K_M, Q4_K_M, IQ4_XS
  6. verify    - llama-cli token preview + llama-perplexity (CPU) for F16 and the four quants
  7. figures   - read results.json and emit CH10_F04..F07_Kalyanarangan.{png,pdf}

Usage:
  python ch10_convert.py --mode all                    # full run
  python ch10_convert.py --mode setup                  # build llama.cpp
  python ch10_convert.py --mode figures                # re-render from results.json

Required Python packages (do NOT auto-install):
  huggingface_hub, gguf (pinned to bundled gguf-py from the cloned llama.cpp clone),
  matplotlib, datasets (optional; raw URL fallback used if missing),
  adjustText (optional; manual offsets used if missing).

Disk and RAM:
  ~46 GB free disk required (HF snapshot 13 GB + F16 GGUF 13 GB + four quants ~18 GB
  + build 2 GB). On a 16 GB Mac, llama-perplexity on F16 mmaps the full 13 GB - close
  other RAM-heavy apps before running. All llama.cpp invocations that contribute
  numbers to results.json run with --n-gpu-layers 0 for CPU-comparable measurements.

Cross-section continuity:
  The Q4_K_M output here corresponds bit-for-bit (modulo build-commit drift) to
  the llama-2-7b.Q4_K_M.gguf used in section 10.2 / Figure 10.3. The quant family
  palette and hatch conventions match section 10.2's Figure 10.2.
"""

import argparse
import dataclasses
import datetime
import json
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

BASE = Path(__file__).resolve().parent  #A all paths derived from script location, never cwd


# --- Configuration ---------------------------------------------------------

@dataclass
class Config:
    # Paths
    build_dir: Path = field(default_factory=lambda: BASE / "_build" / "llama.cpp")
    models_dir: Path = field(default_factory=lambda: BASE / "_models")
    artifacts_dir: Path = field(default_factory=lambda: BASE / "_artifacts")
    figures_dir: Path = field(default_factory=lambda: BASE / "_artifacts" / "figures")
    book_figures_dir: Path = field(default_factory=lambda: BASE / "figures")  #A also drop figures next to F02/F03
    logs_dir: Path = field(default_factory=lambda: BASE / "_logs")
    results_path: Path = field(default_factory=lambda: BASE / "_artifacts" / "results.json")
    log_file: Path = field(default_factory=lambda: BASE / "ch10_convert.log")

    # Source artifacts
    repo_id: str = "meta-llama/Llama-2-7b-hf"
    llama_cpp_repo: str = "https://github.com/ggml-org/llama.cpp"

    # Verification fixtures
    verification_prompt: str = "The Apollo program was the third United States human spaceflight program carried out by"
    n_predict: int = 64
    seed: int = 42
    perplexity_chunks: int = 4   #A 4 x 512-token chunks ~ 2K tokens; keeps the F16 PPL in budget on a 16 GB MBA

    # Quantization targets
    quant_targets: Tuple[Tuple[str, str, int], ...] = field(default_factory=lambda: (
        ("Q8_0",   "q8_0",   7),    #A (ftype-name, slug, expected ftype code)
        ("Q5_K_M", "q5_k_m", 17),
        ("Q4_K_M", "q4_k_m", 15),
        ("IQ4_XS", "iq4_xs", 30),
    ))

    # Figure aesthetics — match section 10.2 conventions
    fig_width_in: float = 5.6                  #A Manning max width
    fig_dpi: int = 300
    family_color: Dict[str, str] = field(default_factory=lambda: {
        "f16":      "#444444",                  #A neutral grey baseline
        "legacy":   "#7E76B0",                  #A Q8_0
        "k_quant":  "#319974",                  #A Q5_K_M, Q4_K_M
        "iq_quant": "#D67430",                  #A IQ4_XS
    })
    family_hatch: Dict[str, str] = field(default_factory=lambda: {
        "f16":      "",
        "legacy":   "....",
        "k_quant":  "////",
        "iq_quant": "xxxx",
    })
    family_marker: Dict[str, str] = field(default_factory=lambda: {
        "f16":      "o",
        "legacy":   "D",
        "k_quant":  "s",
        "iq_quant": "^",
    })


# Variant -> (display name, family key)
VARIANT_FAMILY: Dict[str, Tuple[str, str]] = {
    "f16":    ("F16",     "f16"),
    "q8_0":   ("Q8_0",    "legacy"),
    "q5_k_m": ("Q5_K_M",  "k_quant"),
    "q4_k_m": ("Q4_K_M",  "k_quant"),
    "iq4_xs": ("IQ4_XS",  "iq_quant"),
}

# imatrix calibration text source (raw wikitext-2 test split, MIT-licensed)
WIKITEXT_RAW_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/"
    "resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
)


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
    merged_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")          #A defuse libiomp5 double-init when the conda env has MKL
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
            if len(tail_buffer) > 200:
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
    info: Dict[str, object] = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    try:
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        info["cpu_brand"] = cpu
    except Exception:
        info["cpu_brand"] = ""
    try:
        ram_b = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        info["ram_gb"] = round(ram_b / (1024 ** 3), 1)
    except Exception:
        info["ram_gb"] = 0
    try:
        info["macos"] = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
    except Exception:
        info["macos"] = ""
    return info


def read_results(cfg: Config) -> dict:
    if not cfg.results_path.exists():
        return {}
    return json.loads(cfg.results_path.read_text())


def write_results(cfg: Config, data: dict) -> None:
    cfg.results_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.results_path.write_text(json.dumps(data, indent=2, sort_keys=True))


# --- Mode: setup -----------------------------------------------------------

def cmd_path(cfg: Config, name: str) -> Path:
    """Locate a built llama.cpp binary inside _build/llama.cpp/build/bin/."""
    p = cfg.build_dir / "build" / "bin" / name
    return p


def mode_setup(cfg: Config, force: bool = False) -> dict:
    cfg.build_dir.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.build_dir.exists():
        log(f"cloning llama.cpp into {cfg.build_dir}", cfg.log_file)
        must_run(["git", "clone", cfg.llama_cpp_repo, str(cfg.build_dir)],
                 cfg.logs_dir / "git_clone.log")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(cfg.build_dir), text=True).strip()
    log(f"llama.cpp commit: {commit}", cfg.log_file)

    bins = ["llama-quantize", "llama-imatrix", "llama-completion", "llama-perplexity"]  #B llama-completion is the one-shot generator on recent llama.cpp; llama-cli has gone interactive-only
    needs_build = force or any(not cmd_path(cfg, b).exists() for b in bins)
    if needs_build:
        log("configuring cmake (Metal=ON, Release)", cfg.log_file)
        must_run([
            "cmake", "-B", "build",
            "-DGGML_METAL=ON",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=ON",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ], cfg.logs_dir / "cmake_configure.log", cwd=cfg.build_dir)

        log("building llama.cpp targets", cfg.log_file)
        must_run([
            "cmake", "--build", "build", "--config", "Release",
            "-j", str(max(2, os.cpu_count() or 4)),
            "--target", *bins,
        ], cfg.logs_dir / "cmake_build.log", cwd=cfg.build_dir)
    else:
        log("llama.cpp binaries already built; skipping (use --force to rebuild)", cfg.log_file)

    convert_py = cfg.build_dir / "convert_hf_to_gguf.py"
    if not convert_py.exists():
        sys.stderr.write(f"convert_hf_to_gguf.py not found at {convert_py}\n")
        sys.exit(1)

    return {
        "commit": commit,
        "build_flags": "-DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release",
        "convert_hf_to_gguf_path": str(convert_py.relative_to(cfg.build_dir)),
        "bin_dir": str((cfg.build_dir / "build" / "bin").relative_to(BASE)),
    }


# --- Mode: download --------------------------------------------------------

def repo_local_dir(cfg: Config) -> Path:
    return cfg.models_dir / cfg.repo_id.replace("/", "__")


def mode_download(cfg: Config) -> dict:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError:
        sys.stderr.write("huggingface_hub not installed. pip install huggingface_hub\n")
        sys.exit(1)
    target = repo_local_dir(cfg)
    target.mkdir(parents=True, exist_ok=True)
    log(f"downloading {cfg.repo_id} -> {target}", cfg.log_file)
    try:
        path = snapshot_download(
            repo_id=cfg.repo_id,
            local_dir=str(target),
            allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"],  #A skip .bin .pth duplicates
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "403" in msg or "gated" in msg.lower():
            sys.stderr.write(
                f"\n[AUTH] Hugging Face refused access to {cfg.repo_id}.\n"
                f"       Run `huggingface-cli login` with a token that has gate access,\n"
                f"       and confirm you accepted the model license on the model page.\n"
                f"       Original error: {e}\n"
            )
            sys.exit(1)
        raise
    return {"local_dir": str(Path(path).relative_to(BASE) if Path(path).is_absolute() else path)}


# --- Mode: convert ---------------------------------------------------------

def f16_path(cfg: Config) -> Path:
    return cfg.models_dir / f"{cfg.repo_id.split('/')[-1]}-F16.gguf"


def mode_convert(cfg: Config) -> dict:
    convert_py = cfg.build_dir / "convert_hf_to_gguf.py"
    if not convert_py.exists():
        sys.stderr.write(f"missing {convert_py}; run --mode setup first\n")
        sys.exit(1)
    src = repo_local_dir(cfg)
    if not src.exists() or not any(src.glob("*.safetensors")):
        sys.stderr.write(f"safetensors missing in {src}; run --mode download first\n")
        sys.exit(1)
    out = f16_path(cfg)
    log(f"converting {src.name} -> {out.name} (F16)", cfg.log_file)
    wall, _ = must_run([
        sys.executable, str(convert_py),
        str(src),
        "--outtype", "f16",
        "--outfile", str(out),
    ], cfg.logs_dir / "convert_f16.log")
    size_mb = round(os.path.getsize(out) / (1024 ** 2), 2)
    return {
        "wall_s": round(wall, 2),
        "out_path": str(out.relative_to(BASE)),
        "size_mb": size_mb,
        "log": str((cfg.logs_dir / "convert_f16.log").relative_to(BASE)),
    }


# --- Mode: imatrix ---------------------------------------------------------

def imatrix_text_path(cfg: Config) -> Path:
    return cfg.models_dir / "imatrix_calibration.txt"


def imatrix_gguf_path(cfg: Config) -> Path:
    return cfg.models_dir / "imatrix.gguf"


def _fetch_wikitext_text(cfg: Config, target_words: int = 1500) -> str:
    """Pull a wikitext-2-raw-v1 test slice. Try `datasets` first; fall back to raw URL."""
    try:
        from datasets import load_dataset                         #A optional path: cleaner, slower first run
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text_lines = [r["text"] for r in ds if r["text"].strip()]
        text = "\n".join(text_lines)
    except Exception as e:
        log(f"datasets path unavailable ({e}); using raw URL fallback", cfg.log_file)
        try:
            import pyarrow.parquet as pq                          #A still part of pyarrow which datasets depends on
            tmp = cfg.models_dir / "_wikitext_test.parquet"
            cfg.models_dir.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(WIKITEXT_RAW_URL, tmp)
            tbl = pq.read_table(tmp)
            text = "\n".join(s for s in tbl.column("text").to_pylist() if s.strip())
        except Exception as e2:
            sys.stderr.write(f"failed to fetch wikitext: {e2}\n"); sys.exit(1)
    # Trim to roughly target_words for a fast imatrix run.
    words = text.split()
    return " ".join(words[:target_words])


def mode_imatrix(cfg: Config) -> dict:
    text = _fetch_wikitext_text(cfg)
    txt = imatrix_text_path(cfg)
    txt.write_text(text, encoding="utf-8")
    log(f"wrote calibration text ({len(text.split())} words) -> {txt.name}", cfg.log_file)

    out = imatrix_gguf_path(cfg)
    if out.exists():
        out.unlink()                                              #A always rebuild for a fresh wall-clock measurement
    binary = cmd_path(cfg, "llama-imatrix")
    if not binary.exists():
        sys.stderr.write(f"missing {binary}; run --mode setup\n"); sys.exit(1)

    log("generating importance matrix on wikitext-2 slice (CPU only)", cfg.log_file)
    wall, _ = must_run([
        str(binary),
        "-m", str(f16_path(cfg)),
        "-f", str(txt),
        "-o", str(out),
        "--n-gpu-layers", "0",                                    #B CPU-only for reproducible numbers
        "--chunks", "10",
        "-c", "512",
    ], cfg.logs_dir / "imatrix.log")

    size_kb = round(os.path.getsize(out) / 1024.0, 2)
    return {
        "wall_s": round(wall, 2),
        "out_path": str(out.relative_to(BASE)),
        "size_kb": size_kb,
        "log": str((cfg.logs_dir / "imatrix.log").relative_to(BASE)),
        "calibration_words": len(text.split()),
    }


# --- Mode: quantize --------------------------------------------------------

def quant_out_path(cfg: Config, slug: str) -> Path:
    return cfg.models_dir / f"{cfg.repo_id.split('/')[-1]}-{slug.upper()}.gguf"


def mode_quantize(cfg: Config) -> Dict[str, dict]:
    binary = cmd_path(cfg, "llama-quantize")
    src = f16_path(cfg)
    if not src.exists():
        sys.stderr.write(f"missing {src}; run --mode convert first\n"); sys.exit(1)
    imat = imatrix_gguf_path(cfg)

    out: Dict[str, dict] = {}
    for ftype, slug, expected_code in cfg.quant_targets:
        target = quant_out_path(cfg, slug)
        if target.exists():
            target.unlink()
        log(f"quantizing -> {ftype}", cfg.log_file)
        cmd: List[str] = [str(binary)]
        if ftype == "IQ4_XS":
            if not imat.exists():
                sys.stderr.write("imatrix.gguf missing; IQ4_XS refuses to run\n"); sys.exit(1)
            cmd += ["--imatrix", str(imat)]                      #B IQ-quants depend on the importance matrix
        cmd += [str(src), str(target), ftype]
        wall, _ = must_run(cmd, cfg.logs_dir / f"quantize_{slug}.log")
        size_mb = round(os.path.getsize(target) / (1024 ** 2), 2)
        out[f"quantize_{slug}"] = {
            "wall_s": round(wall, 2),
            "out_path": str(target.relative_to(BASE)),
            "size_mb": size_mb,
            "ftype_code": expected_code,                          #C confirmed against gguf-py LlamaFileType
            "log": str((cfg.logs_dir / f"quantize_{slug}.log").relative_to(BASE)),
        }
    return out


# --- Mode: verify ----------------------------------------------------------

def _parse_completion_perf(text: str) -> Dict[str, float]:
    """Pull inference timing out of llama-completion's `common_perf_print` block.

    Lines we care about:
      common_perf_print:        eval time =    3259.73 ms /    63 runs   (   51.74 ms per token,    19.33 tokens per second)
      common_perf_print: prompt eval time =     183.50 ms /    18 tokens (   10.19 ms per token,    98.09 tokens per second)
      common_perf_print:       total time =    3449.10 ms /    81 tokens
    """
    import re
    out: Dict[str, float] = {}
    for raw in text.splitlines():
        if "common_perf_print:" not in raw:
            continue
        s = raw.split("common_perf_print:", 1)[1].strip()
        if s.startswith("eval time"):
            m = re.search(
                r"=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?\(\s*([\d.]+)\s*ms per token.*?([\d.]+)\s*tokens per second",
                s,
            )
            if m:
                out["eval_ms"] = float(m.group(1))
                out["eval_runs"] = int(m.group(2))
                out["decode_ms_per_token"] = float(m.group(3))
                out["decode_tokens_per_sec"] = float(m.group(4))
        elif s.startswith("total time"):
            m = re.search(r"=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens", s)
            if m:
                out["total_ms"] = float(m.group(1))
                out["total_tokens"] = int(m.group(2))
        elif s.startswith("prompt eval time"):
            m = re.search(
                r"=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?\(\s*([\d.]+)\s*ms per token.*?([\d.]+)\s*tokens per second",
                s,
            )
            if m:
                out["prompt_eval_ms"] = float(m.group(1))
                out["prompt_eval_tokens"] = int(m.group(2))
                out["prompt_ms_per_token"] = float(m.group(3))
                out["prompt_tokens_per_sec"] = float(m.group(4))
    return out


def _parse_first_64_tokens(stdout: str, prompt: str, n_target: int = 64) -> str:
    """llama-completion prints generated tokens after a banner of system/sampler info.
    The generation is bookended by `generate: n_ctx = ... n_predict = N\\n\\n` and a
    `common_perf_print:` block. Slice between them, fall back to scanning if the
    delimiters drift across versions."""
    txt = stdout
    start_marker = "generate: n_ctx"
    if start_marker in txt:
        txt = txt.split(start_marker, 1)[1]
        # consume the rest of that intro line and the blank line that follows
        nl = txt.find("\n")
        if nl >= 0:
            txt = txt[nl + 1:].lstrip("\n")
    # cut off everything after the perf/memory block
    for marker in ("\ncommon_perf_print", "\ncommon_memory_breakdown_print",
                   "\nllama_perf", "\nllama_print_timings"):
        if marker in txt:
            txt = txt.split(marker, 1)[0]
    # If --no-display-prompt is set the slice should already start at the generation,
    # but older builds occasionally emit a prompt prefix before generation. Only strip
    # the prompt if it appears at the very start (don't rfind — the model often parrots
    # the prompt mid-completion and aggressive trimming would chop the real generation).
    stripped_lead = txt.lstrip()
    if stripped_lead.startswith(prompt):
        lead_len = len(txt) - len(stripped_lead)
        txt = txt[lead_len + len(prompt):]
    return txt.strip()


def _parse_perplexity(stderr_or_stdout: str) -> Optional[float]:
    """Parse the final perplexity number printed by llama-perplexity."""
    # llama-perplexity prints lines like:
    #   [1]4.4321,[2]4.5512,[3]...
    # and at the end:
    #   Final estimate: PPL = 5.6789 +/- 0.12345
    val: Optional[float] = None
    for line in stderr_or_stdout.splitlines():
        line = line.strip()
        if "Final estimate" in line and "PPL" in line:
            try:
                # "Final estimate: PPL = 5.6789 +/- 0.12345"
                rhs = line.split("PPL", 1)[1]
                rhs = rhs.split("=", 1)[1] if "=" in rhs else rhs
                tok = rhs.strip().split()[0]
                val = float(tok)
            except Exception:
                continue
        elif line.startswith("[") and "]" in line:
            # Per-chunk live values; keep last as a fallback.
            try:
                tok = line.rsplit(",", 1)[-1]
                tok = tok.split("]", 1)[1]
                val = float(tok)
            except Exception:
                continue
    return val


def _run_token_preview(cfg: Config, model_path: Path, slug: str) -> str:
    """Run llama-completion (one-shot). On the b9010-d05fe1d7d commit llama-cli is
    interactive-only — its source explicitly says "use llama-completion instead" when
    --no-conversation is requested, so we route through llama-completion. Token preview
    runs with Metal so F16 on a 16-24 GB Mac finishes without paging the 13 GB model.
    Perplexity runs separately stay CPU-only — those are the numbers plotted in Figure
    10.5. The Manning prose should note that Metal output may drift bit-for-bit across
    reruns even with --seed 42 --temp 0."""
    binary = cmd_path(cfg, "llama-completion")
    log_path = cfg.logs_dir / f"verify_tokens_{slug}.log"
    cmd = [
        str(binary),
        "-m", str(model_path),
        "-p", cfg.verification_prompt,
        "-n", str(cfg.n_predict),
        "--seed", str(cfg.seed),
        "--temp", "0",
        "-no-cnv",                                                #B one-shot completion mode; without it llama-completion launches a chat REPL
        "--no-display-prompt",                                    #B suppress echoing the prompt for a clean parse
    ]
    wall, rc, tail = run_cmd(cmd, log_path)
    if rc != 0:
        sys.stderr.write(f"llama-completion failed (rc={rc}); see {log_path}\n"); sys.exit(1)
    text = log_path.read_text()
    return _parse_first_64_tokens(text, cfg.verification_prompt, cfg.n_predict)


def _backfill_inference_perf(cfg: Config, results: dict) -> None:
    """If verify[*]['inference_perf'] is missing, parse it from the existing token-preview logs."""
    for slug in ("f16", "q8_0", "q5_k_m", "q4_k_m", "iq4_xs"):
        v = results.get("verify", {}).get(slug)
        if not v:
            continue
        if v.get("inference_perf"):
            continue
        log_path = cfg.logs_dir / f"verify_tokens_{slug}.log"
        if not log_path.exists():
            continue
        perf = _parse_completion_perf(log_path.read_text())
        if perf:
            v["inference_perf"] = perf


def _run_perplexity(cfg: Config, model_path: Path, slug: str) -> float:
    binary = cmd_path(cfg, "llama-perplexity")
    txt_path = imatrix_text_path(cfg)
    if not txt_path.exists():
        sys.stderr.write(f"missing {txt_path}; run --mode imatrix first to materialize calibration text\n"); sys.exit(1)
    log_path = cfg.logs_dir / f"verify_ppl_{slug}.log"
    cmd = [
        str(binary),
        "-m", str(model_path),
        "-f", str(txt_path),
        "--n-gpu-layers", "0",                                    #B CPU-comparable across variants
        "-c", "512",
        "--chunks", str(cfg.perplexity_chunks),
    ]
    wall, rc, tail = run_cmd(cmd, log_path)
    if rc != 0:
        sys.stderr.write(f"llama-perplexity failed (rc={rc}); see {log_path}\n"); sys.exit(1)
    text = log_path.read_text()
    val = _parse_perplexity(text)
    if val is None:
        sys.stderr.write(f"could not parse PPL from {log_path}\n--- tail ---\n{tail}\n"); sys.exit(1)
    return val


def _tensor_type_histogram(model_path: Path) -> Tuple[Dict[str, int], int, str]:
    """Re-parse the GGUF and return (histogram, file_type_code, file_type_label)."""
    from gguf import GGUFReader
    from gguf.constants import LlamaFileType
    reader = GGUFReader(str(model_path))
    hist: Dict[str, int] = {}
    for t in reader.tensors:
        name = t.tensor_type.name
        hist[name] = hist.get(name, 0) + 1
    code = -1
    label = ""
    for f in reader.fields.values():
        if f.name == "general.file_type":
            try:
                code = int(f.parts[f.data[0]][0])
            except Exception:
                pass
            break
    try:
        label = LlamaFileType(code).name
    except Exception:
        label = f"UNKNOWN({code})"
    return hist, code, label


def mode_verify(cfg: Config, results: dict) -> dict:
    src = f16_path(cfg)
    if not src.exists():
        sys.stderr.write(f"missing {src}; run --mode convert first\n"); sys.exit(1)
    verify: Dict[str, dict] = {}

    # F16 first (lossless anchor for figures 10.5 and 10.7).
    log("verifying F16 (token preview + perplexity)", cfg.log_file)
    f16_tokens = _run_token_preview(cfg, src, "f16")
    f16_perf = _parse_completion_perf((cfg.logs_dir / "verify_tokens_f16.log").read_text())
    f16_ppl = _run_perplexity(cfg, src, "f16")
    verify["f16"] = {
        "size_mb": round(os.path.getsize(src) / (1024 ** 2), 2),
        "first_64_tokens": f16_tokens,
        "inference_perf": f16_perf,                                #B Metal-accelerated decode timing for Figure 10.8
        "wikitext_ppl": round(f16_ppl, 4),
    }

    # Quants
    for ftype, slug, expected_code in cfg.quant_targets:
        path = quant_out_path(cfg, slug)
        if not path.exists():
            sys.stderr.write(f"missing {path}; run --mode quantize first\n"); sys.exit(1)
        log(f"verifying {ftype}", cfg.log_file)
        hist, code, label = _tensor_type_histogram(path)
        if code != expected_code:
            sys.stderr.write(
                f"[FATAL] {ftype} ftype mismatch: expected {expected_code}, got {code} ({label})\n"
                f"        check llama.cpp commit drift and re-confirm against gguf-py LlamaFileType\n"
            )
            sys.exit(1)
        tokens = _run_token_preview(cfg, path, slug)
        perf = _parse_completion_perf((cfg.logs_dir / f"verify_tokens_{slug}.log").read_text())
        ppl = _run_perplexity(cfg, path, slug)
        verify[slug] = {
            "size_mb": round(os.path.getsize(path) / (1024 ** 2), 2),
            "file_type_code": code,
            "file_type_label": label,
            "tensor_type_histogram": hist,
            "first_64_tokens": tokens,
            "inference_perf": perf,                                #B Metal decode ms/token for Figure 10.8
            "wikitext_ppl": round(ppl, 4),
        }

    # Markdown summary table to stdout.
    print("\n## Verification summary\n")
    header = "| variant | size MB | bpw* | wall-s | wikitext PPL | first-64 (truncated) |"
    sep = "|---|---:|---:|---:|---:|---|"
    print(header); print(sep)
    steps = results.get("steps", {})
    n_params = 6_738_415_616  #A llama-2-7B parameter count from the model config; used only for bpw display
    for slug in ("f16", "q8_0", "q5_k_m", "q4_k_m", "iq4_xs"):
        v = verify.get(slug, {})
        size_mb = v.get("size_mb", 0)
        bpw = (size_mb * 1024 * 1024 * 8) / n_params if n_params else 0
        wall = steps.get(f"convert_f16" if slug == "f16" else f"quantize_{slug}", {}).get("wall_s", 0)
        toks = v.get("first_64_tokens", "")
        toks_short = (toks[:60] + "...") if len(toks) > 60 else toks
        toks_short = toks_short.replace("|", "\\|").replace("\n", " ")
        print(f"| {VARIANT_FAMILY[slug][0]} | {size_mb:.1f} | {bpw:.2f} | {wall:.1f} | {v.get('wikitext_ppl', 0):.3f} | {toks_short} |")
    print()
    return verify


# --- Mode: figures ---------------------------------------------------------

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
        "pdf.fonttype": 42, "ps.fonttype": 42,                    #B keeps text editable in vector PDF
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
    # Also drop into ch10/figures/ alongside F02/F03 for the prose author.
    shutil.copy2(png, cfg.book_figures_dir / png.name)
    shutil.copy2(pdf, cfg.book_figures_dir / pdf.name)
    plt.close(fig)
    return str(png.relative_to(BASE)), str(pdf.relative_to(BASE))


def _family_for(slug: str) -> str:
    return VARIANT_FAMILY[slug][1]


def _fig_pipeline(cfg: Config) -> Tuple[Tuple[str, str], str]:
    """Figure 10.4 — pipeline block diagram (no measurements)."""
    plt = _setup_mpl(cfg)
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
    fig, ax = plt.subplots(figsize=(cfg.fig_width_in, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    NEUTRAL = "#DDDDDD"
    NEUTRAL_EDGE = "#666666"

    # Layout: process row across the top, output column on the right.
    # The family colors on the four output boxes match section 10.2 conventions
    # (purple/green/orange = legacy/k-quant/IQ-quant) — the legend below is what
    # tells the reader those colors are not arbitrary.
    boxes = [
        ("HF\nsafetensors",      8,  78, 14, 11, NEUTRAL,                       "",                         NEUTRAL_EDGE),
        ("convert_hf_to_gguf.py",30, 78, 22, 11, NEUTRAL,                       "",                         NEUTRAL_EDGE),
        ("F16 GGUF",             54, 78, 14, 11, NEUTRAL,                       "",                         NEUTRAL_EDGE),
        ("llama-quantize",       78, 78, 16, 11, NEUTRAL,                       "",                         NEUTRAL_EDGE),
        # outputs stacked vertically, well to the right of the quantize hub
        ("Q8_0",                 95, 95,  9,  6, cfg.family_color["legacy"],    cfg.family_hatch["legacy"],   "#3A3460"),
        ("Q5_K_M",               95, 85,  9,  6, cfg.family_color["k_quant"],   cfg.family_hatch["k_quant"],  "#1F5C46"),
        ("Q4_K_M",               95, 75,  9,  6, cfg.family_color["k_quant"],   cfg.family_hatch["k_quant"],  "#1F5C46"),
        ("IQ4_XS",               95, 65,  9,  6, cfg.family_color["iq_quant"],  cfg.family_hatch["iq_quant"], "#7B411A"),
        ("llama-imatrix",        54, 40, 16, 11, NEUTRAL,                       "",                         NEUTRAL_EDGE),
        ("imatrix.gguf",         78, 40, 14, 11, NEUTRAL,                       "",                         NEUTRAL_EDGE),
    ]
    coords = {}
    import matplotlib.patheffects as path_effects                  #A let labels punch through hatches with a thin outline
    for (label, cx, cy, w, h, face, hatch, edge) in boxes:
        x = cx - w/2; y = cy - h/2
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=1.2",
            linewidth=0.9, facecolor=face, edgecolor=edge,
            hatch=hatch if hatch else None,
        )
        ax.add_patch(patch)
        is_quant = bool(hatch)
        text = ax.text(cx, cy, label, ha="center", va="center",
                       fontsize=7.5 if is_quant else 7.0,
                       color="white" if is_quant else "black",
                       fontweight="bold" if is_quant else "normal", zorder=3)
        if is_quant:                                              #B white text + black outline = legible over any hatch fill
            text.set_path_effects([
                path_effects.Stroke(linewidth=1.6, foreground="black"),
                path_effects.Normal(),
            ])
        coords[label] = (cx, cy, w, h)

    def arrow(p1, p2, dashed=False, color="#333333", lw=0.9, rad=0.0, zorder=1):
        style = "-" if not dashed else (0, (3, 2))
        cs = f"arc3,rad={rad}"
        arr = FancyArrowPatch(
            p1, p2,
            connectionstyle=cs,
            arrowstyle="-|>", mutation_scale=7,
            linestyle=style, color=color, linewidth=lw, zorder=zorder,
        )
        ax.add_patch(arr)

    def edge(box, side):
        cx, cy, w, h = coords[box]
        if side == "right":  return (cx + w/2, cy)
        if side == "left":   return (cx - w/2, cy)
        if side == "top":    return (cx, cy + h/2)
        if side == "bottom": return (cx, cy - h/2)

    arrow(edge("HF\nsafetensors", "right"),       edge("convert_hf_to_gguf.py", "left"))
    arrow(edge("convert_hf_to_gguf.py", "right"), edge("F16 GGUF", "left"))
    arrow(edge("F16 GGUF", "right"),              edge("llama-quantize", "left"))
    qx = coords["llama-quantize"][0] + coords["llama-quantize"][2] / 2
    qy = coords["llama-quantize"][1]
    arrow((qx, qy), edge("Q8_0",   "left"), rad=-0.20)
    arrow((qx, qy), edge("Q5_K_M", "left"), rad=-0.06)
    arrow((qx, qy), edge("Q4_K_M", "left"), rad= 0.06)
    arrow((qx, qy), edge("IQ4_XS", "left"), rad= 0.20)

    arrow(edge("F16 GGUF", "bottom"),       edge("llama-imatrix", "top"))
    arrow(edge("llama-imatrix", "right"),   edge("imatrix.gguf", "left"))
    arrow(edge("imatrix.gguf", "top"), edge("IQ4_XS", "bottom"),
          dashed=True, color=cfg.family_color["iq_quant"], lw=1.1, rad=-0.25, zorder=2)
    ax.text(91, 50, "required\n(IQ-quants only)", ha="center", va="center",
            fontsize=6.2, color=cfg.family_color["iq_quant"], style="italic")

    # Family legend — explains BOTH the colors and the hatches in one place.
    legend_handles = [
        Patch(facecolor=cfg.family_color["legacy"],   hatch=cfg.family_hatch["legacy"],
              edgecolor="black", linewidth=0.6, label="Legacy quants (Q8_0)"),
        Patch(facecolor=cfg.family_color["k_quant"],  hatch=cfg.family_hatch["k_quant"],
              edgecolor="black", linewidth=0.6, label="k-quants (Q5_K_M, Q4_K_M)"),
        Patch(facecolor=cfg.family_color["iq_quant"], hatch=cfg.family_hatch["iq_quant"],
              edgecolor="black", linewidth=0.6, label="IQ-quants (IQ4_XS)"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False,
              fontsize=6.8, handlelength=2.2, handleheight=1.4, columnspacing=1.6)

    fig.tight_layout()
    name = "CH10_F04_Kalyanarangan"
    paths = _save_pair(plt, fig, cfg, name)
    caption = (
        "Conversion pipeline from a Hugging Face safetensors checkpoint to four quantized "
        "GGUF artifacts. Output boxes use the section 10.2 family colors (purple = legacy, "
        "green = k-quant, orange = IQ-quant) and matching hatches (dots, diagonals, cross). "
        "The dashed arrow marks the imatrix dependency that only IQ4_XS requires."
    )
    return paths, caption


def _fig_size_quality(cfg: Config, results: dict) -> Tuple[Tuple[str, str], str]:
    """Figure 10.5 — file size vs wikitext perplexity scatter."""
    plt = _setup_mpl(cfg)
    fig, ax = plt.subplots(figsize=(cfg.fig_width_in, 3.6))

    verify = results.get("verify", {})
    order = ["f16", "q8_0", "q5_k_m", "q4_k_m", "iq4_xs"]
    pts = []
    for slug in order:
        v = verify.get(slug, {})
        if not v or "size_mb" not in v or "wikitext_ppl" not in v:
            log(f"figure 10.5: variant {slug} missing data; skipping point", cfg.log_file)
            continue
        pts.append((slug, v["size_mb"], v["wikitext_ppl"]))

    # Frontier line in size order (largest to smallest).
    pts_sorted = sorted(pts, key=lambda p: -p[1])
    if len(pts_sorted) >= 2:
        ax.plot(
            [p[1] for p in pts_sorted], [p[2] for p in pts_sorted],
            linestyle=":", color="#888888", linewidth=0.8, zorder=1,
        )

    # Markers, fills, hatches
    for (slug, size_mb, ppl) in pts:
        fam = _family_for(slug)
        ax.scatter(
            [size_mb], [ppl],
            s=110, c=cfg.family_color[fam], marker=cfg.family_marker[fam],
            edgecolors="black", linewidths=0.8,
            hatch=cfg.family_hatch[fam] if cfg.family_hatch[fam] else None,
            zorder=3,
        )

    # Hand-tuned label offsets — every label sits clear of the frontier line and other markers.
    label_offsets_data = {
        "f16":    (-180,  0.000),     #B label to the left of F16 so it stays inside the axes
        "q8_0":   (   0,  0.018),     #B above marker
        "q5_k_m": (   0,  0.018),     #B above marker
        "q4_k_m": ( 200,  0.000),     #B to the right of marker
        "iq4_xs": (-180,  0.000),     #B to the left of marker
    }
    label_align = {
        "f16":    ("right", "center"),
        "q8_0":   ("center", "bottom"),
        "q5_k_m": ("center", "bottom"),
        "q4_k_m": ("left", "center"),
        "iq4_xs": ("right", "center"),
    }
    for (slug, size_mb, ppl) in pts:
        dx, dy = label_offsets_data.get(slug, (80, 0.025))
        ha, va = label_align.get(slug, ("left", "center"))
        ax.text(size_mb + dx, ppl + dy, VARIANT_FAMILY[slug][0],
                fontsize=7.5, color="black", ha=ha, va=va)

    ax.set_xlabel("File size (MB)")
    ax.set_ylabel("Wikitext-2 perplexity (lower is better)")

    if pts:
        ymin = min(p[2] for p in pts); ymax = max(p[2] for p in pts)
        pad = (ymax - ymin) * 0.30 + 0.05
        ax.set_ylim(ymin - pad * 0.5, ymax + pad)
        xmin = min(p[1] for p in pts); xmax = max(p[1] for p in pts)
        xpad = (xmax - xmin) * 0.10
        ax.set_xlim(xmin - xpad - 600, xmax + xpad + 600)

    # Family legend — same color/marker/hatch encoding as the markers themselves.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=cfg.family_marker["f16"], color="none",
               markerfacecolor=cfg.family_color["f16"], markeredgecolor="black",
               markersize=8, label="F16 (lossless baseline)"),
        Line2D([0], [0], marker=cfg.family_marker["legacy"], color="none",
               markerfacecolor=cfg.family_color["legacy"], markeredgecolor="black",
               markersize=8, label="Legacy quants"),
        Line2D([0], [0], marker=cfg.family_marker["k_quant"], color="none",
               markerfacecolor=cfg.family_color["k_quant"], markeredgecolor="black",
               markersize=8, label="k-quants"),
        Line2D([0], [0], marker=cfg.family_marker["iq_quant"], color="none",
               markerfacecolor=cfg.family_color["iq_quant"], markeredgecolor="black",
               markersize=8, label="IQ-quants"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False,
              fontsize=6.8, handlelength=1.6, borderaxespad=0.4)

    ax.grid(True, axis="y", linestyle=":", color="#cccccc", linewidth=0.5)
    fig.tight_layout()
    name = "CH10_F05_Kalyanarangan"
    paths = _save_pair(plt, fig, cfg, name)
    caption = (
        "File size (megabytes) versus wikitext-2 perplexity for the five reference "
        "artifacts produced by the conversion pipeline. Marker shape and color encode "
        "quant family per the legend. Lower and to the left is better."
    )
    return paths, caption


def _fig_walltime(cfg: Config, results: dict) -> Tuple[Tuple[str, str], str]:
    """Figure 10.6 — quantize wall-clock bars with imatrix overhead stacked on IQ4_XS."""
    plt = _setup_mpl(cfg)
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(cfg.fig_width_in, 3.6))

    steps = results.get("steps", {})
    bars = [
        ("Q8_0",   "q8_0",   "legacy"),
        ("Q5_K_M", "q5_k_m", "k_quant"),
        ("Q4_K_M", "q4_k_m", "k_quant"),
        ("IQ4_XS", "iq4_xs", "iq_quant"),
    ]
    xs = list(range(len(bars)))
    base_vals: List[float] = []
    for (label, slug, fam) in bars:
        s = steps.get(f"quantize_{slug}", {})
        base_vals.append(float(s.get("wall_s", 0.0)))
    imat_wall = float(steps.get("imatrix", {}).get("wall_s", 0.0))

    bar_width = 0.60
    for i, (label, slug, fam) in enumerate(bars):
        ax.bar(
            i, base_vals[i], width=bar_width,
            color=cfg.family_color[fam], edgecolor="black", linewidth=0.7,
            hatch=cfg.family_hatch[fam], zorder=2,
        )
    iq_idx = 3
    if imat_wall > 0:
        ax.bar(
            iq_idx, imat_wall, width=bar_width, bottom=base_vals[iq_idx],
            color=cfg.family_color["iq_quant"], edgecolor="black", linewidth=0.7,
            hatch="///", alpha=0.7, zorder=2,
        )

    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars])
    ax.set_xlabel("Quantization target")                          #B explicit so readers know rows are quant variants, not stages
    ax.set_ylabel("Wall-clock seconds (F16 → variant)")           #B clarify that bars are the conversion step alone
    ax.set_title("Conversion time per quantized variant (one-time, F16 → target)",  #B explicit title so this isn't confused with inference TAT in F08
                 fontsize=8.5, pad=8)
    ax.grid(True, axis="y", linestyle=":", color="#cccccc", linewidth=0.5)

    ymax_data = max(base_vals[i] + (imat_wall if i == iq_idx else 0) for i in range(len(bars)))
    pad = ymax_data * 0.03 + 1
    for i, (label, slug, fam) in enumerate(bars):
        total = base_vals[i] + (imat_wall if i == iq_idx else 0)
        ax.text(i, total + pad, f"{total:.1f}s", ha="center", va="bottom", fontsize=7)
        if i == iq_idx and imat_wall > 0:
            mid = base_vals[i] + imat_wall / 2
            ax.text(i + bar_width / 2 + 0.02, mid, f"(imatrix: {imat_wall:.1f}s)",
                    ha="left", va="center", fontsize=6.5, style="italic", color="#444444")

    f16_wall = float(steps.get("convert_f16", {}).get("wall_s", 0.0))
    ax.text(
        0.02, 0.97,
        f"F16 conversion (one-time, upstream): {f16_wall:.1f}s",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=6.8, color="#444444",
        bbox=dict(facecolor="white", edgecolor="#bbbbbb", linewidth=0.5, pad=2),
    )

    # Family legend + the imatrix overlay so the stacked top of IQ4_XS is explicit.
    legend_handles = [
        Patch(facecolor=cfg.family_color["legacy"],   hatch=cfg.family_hatch["legacy"],
              edgecolor="black", linewidth=0.6, label="Legacy quants"),
        Patch(facecolor=cfg.family_color["k_quant"],  hatch=cfg.family_hatch["k_quant"],
              edgecolor="black", linewidth=0.6, label="k-quants"),
        Patch(facecolor=cfg.family_color["iq_quant"], hatch=cfg.family_hatch["iq_quant"],
              edgecolor="black", linewidth=0.6, label="IQ-quants"),
        Patch(facecolor=cfg.family_color["iq_quant"], hatch="///",
              edgecolor="black", linewidth=0.6, alpha=0.7,
              label="imatrix calibration overhead"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              bbox_to_anchor=(0.02, 0.92), frameon=False,
              fontsize=6.8, handlelength=2.0, handleheight=1.2,
              borderaxespad=0.2, labelspacing=0.5)

    ax.set_ylim(0, ymax_data * 1.20 + 1)
    fig.tight_layout()
    name = "CH10_F06_Kalyanarangan"
    paths = _save_pair(plt, fig, cfg, name)
    caption = (
        "llama-quantize wall-clock seconds per output variant. The IQ4_XS bar stacks the "
        "imatrix calibration time (overlay hatch) onto its base quantization, since the "
        "imatrix is a hard prerequisite for IQ-quant quality. The F16 conversion time "
        "noted at the top-left is a one-time upstream cost shared by all four variants."
    )
    return paths, caption


def _fig_token_preview(cfg: Config, results: dict) -> Tuple[Tuple[str, str], str]:
    """Figure 10.7 — token preview side-by-side comparison."""
    plt = _setup_mpl(cfg)
    verify = results.get("verify", {})
    order = ["f16", "q8_0", "q5_k_m", "q4_k_m", "iq4_xs"]
    rows = []
    for slug in order:
        v = verify.get(slug, {})
        if "first_64_tokens" not in v:
            log(f"figure 10.7: variant {slug} missing tokens; skipping row", cfg.log_file)
            continue
        rows.append((slug, v["first_64_tokens"]))

    # The mono cell at 5.6" wide x ~0.78 axes wide can hold ~52 chars at 6.5pt.
    # Cap rows at four wrapped lines and size the figure so each cell visibly contains them.
    wrap_cols = 52                                                #A char budget per line in the tokens cell
    line_h_in = 0.16                                              #B printed inch per text line at 6.5pt mono
    max_lines = 4                                                 #B firm cap so the table cell fully contains the text

    # Pre-wrap so we can size each row to its actual content height.
    wrapped_per_row: List[List[str]] = []
    for _, tokens in rows:
        wlines: List[str] = []
        for para in tokens.splitlines():
            wlines.extend(textwrap.wrap(para, width=wrap_cols) or [""])
        if len(wlines) > max_lines:
            wlines = wlines[:max_lines]
            wlines[-1] = wlines[-1].rstrip()[: max(0, len(wlines[-1]) - 1)] + "…"
        wrapped_per_row.append(wlines)

    row_text_h_in = [max(1, len(w)) * line_h_in + 0.18 for w in wrapped_per_row]   #C cell height = lines + padding
    rows_total_h = sum(row_text_h_in)
    header_h_in = 0.70                                            #D space for two-line prompt header
    fig_h = header_h_in + rows_total_h + 0.18

    fig, ax = plt.subplots(figsize=(cfg.fig_width_in, fig_h))
    ax.axis("off")

    # Prompt header — wrap so the long prompt doesn't bleed past the figure edge.
    prompt_text = cfg.verification_prompt
    wrapped_prompt = textwrap.wrap(f'Prompt: "{prompt_text}"', width=82)
    ax.text(
        0.0, 1.0 - 0.012,
        "\n".join(wrapped_prompt),
        transform=ax.transAxes, ha="left", va="top", fontsize=7,
        family="DejaVu Sans Mono",
    )

    var_x = 0.0; var_w = 0.18
    tok_x = var_x + var_w + 0.02
    tok_w = 1.0 - tok_x

    # Convert per-row inch heights to axes-fraction heights.
    rows_axes_h = [h_in / fig_h for h_in in row_text_h_in]
    top = 1.0 - (header_h_in / fig_h)

    mono_font = "DejaVu Sans Mono"
    import matplotlib.patheffects as path_effects
    cursor_y = top
    for i, (slug, tokens) in enumerate(rows):
        fam = _family_for(slug)
        cell_h = rows_axes_h[i]
        y_top = cursor_y
        y_bot = y_top - cell_h
        cursor_y = y_bot

        # Variant cell with family color + hatch
        rect = plt.Rectangle(
            (var_x, y_bot), var_w, y_top - y_bot,
            facecolor=cfg.family_color[fam], edgecolor="black", linewidth=0.6,
            hatch=cfg.family_hatch[fam] if cfg.family_hatch[fam] else None,
            transform=ax.transAxes, zorder=1,
        )
        ax.add_patch(rect)
        text = ax.text(
            var_x + var_w / 2, (y_top + y_bot) / 2,
            VARIANT_FAMILY[slug][0],
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=2,
        )
        text.set_path_effects([
            path_effects.Stroke(linewidth=2.0, foreground="black"),
            path_effects.Normal(),
        ])
        # Tokens cell: white background, mono. Top-aligned text so wrapped lines don't drop below.
        ax.add_patch(plt.Rectangle(
            (tok_x, y_bot), tok_w, y_top - y_bot,
            facecolor="white", edgecolor="#bbbbbb", linewidth=0.5,
            transform=ax.transAxes, zorder=1,
        ))
        wrapped = "\n".join(wrapped_per_row[i])
        ax.text(
            tok_x + 0.012, y_top - 0.012, wrapped,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, family=mono_font, color="#111111", linespacing=1.15,
        )

    fig.tight_layout()
    name = "CH10_F07_Kalyanarangan"
    paths = _save_pair(plt, fig, cfg, name)
    caption = (
        "First 64 tokens generated by each variant from the prompt shown, with --seed 42 "
        "and --temp 0 for reproducibility. The F16 row is the lossless reference; "
        "degradation in the lower rows manifests as token-level divergence rather than "
        "collapse for this prompt size."
    )
    return paths, caption


def _fig_inference_tat(cfg: Config, results: dict) -> Tuple[Tuple[str, str], str]:
    """Figure 10.8 — inference turnaround per variant (ms per generated token).

    Source: each verify_tokens_<slug>.log captured `eval time = N ms / R runs (X ms per
    token, Y tokens per second)` from llama-completion. Token preview ran with default
    -ngl (Metal) so these are real-world Apple-Silicon decode rates, not the CPU-only
    perplexity numbers from F05.
    """
    plt = _setup_mpl(cfg)
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(cfg.fig_width_in, 3.6))

    verify = results.get("verify", {})
    bars = [
        ("F16",    "f16",    "f16"),
        ("Q8_0",   "q8_0",   "legacy"),
        ("Q5_K_M", "q5_k_m", "k_quant"),
        ("Q4_K_M", "q4_k_m", "k_quant"),
        ("IQ4_XS", "iq4_xs", "iq_quant"),
    ]
    rows: List[Tuple[str, str, str, float, float]] = []
    for label, slug, fam in bars:
        perf = verify.get(slug, {}).get("inference_perf") or {}
        ms = perf.get("decode_ms_per_token")
        tps = perf.get("decode_tokens_per_sec")
        if ms is None or tps is None:
            log(f"figure 10.8: variant {slug} missing inference_perf; skipping bar", cfg.log_file)
            continue
        rows.append((label, slug, fam, float(ms), float(tps)))

    xs = list(range(len(rows)))
    bar_width = 0.62
    for i, (label, slug, fam, ms, tps) in enumerate(rows):
        ax.bar(
            i, ms, width=bar_width,
            color=cfg.family_color[fam], edgecolor="black", linewidth=0.7,
            hatch=cfg.family_hatch[fam] if cfg.family_hatch[fam] else None, zorder=2,
        )

    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows])
    ax.set_xlabel("Inference target")
    ax.set_ylabel("Decode time per token (ms, lower is better)")
    ax.set_title("Inference turnaround per variant (Metal, --temp 0, 64-token completion)",
                 fontsize=8.5, pad=8)
    ax.grid(True, axis="y", linestyle=":", color="#cccccc", linewidth=0.5)

    if rows:
        ymax = max(r[3] for r in rows)
        ax.set_ylim(0, ymax * 1.22 + 1)
        pad = ymax * 0.03 + 0.5
        for i, (label, slug, fam, ms, tps) in enumerate(rows):
            ax.text(i, ms + pad, f"{ms:.1f} ms", ha="center", va="bottom", fontsize=7)
            ax.text(i, ms + pad * 3.2, f"({tps:.1f} tok/s)",
                    ha="center", va="bottom", fontsize=6.3, color="#555555", style="italic")

    legend_handles = [
        Patch(facecolor=cfg.family_color["f16"], edgecolor="black", linewidth=0.6,
              label="F16 (lossless baseline)"),
        Patch(facecolor=cfg.family_color["legacy"],   hatch=cfg.family_hatch["legacy"],
              edgecolor="black", linewidth=0.6, label="Legacy quants"),
        Patch(facecolor=cfg.family_color["k_quant"],  hatch=cfg.family_hatch["k_quant"],
              edgecolor="black", linewidth=0.6, label="k-quants"),
        Patch(facecolor=cfg.family_color["iq_quant"], hatch=cfg.family_hatch["iq_quant"],
              edgecolor="black", linewidth=0.6, label="IQ-quants"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False,
              fontsize=6.8, handlelength=2.0, handleheight=1.2,
              borderaxespad=0.4, labelspacing=0.5)

    # Footnote — keep readers from generalizing this bar order to non-Metal hardware.
    fig.text(
        0.5, 0.005,
        "Note: ranking is for Apple Silicon + Metal, where decode is memory-bandwidth bound and the IQ4_XS codebook lookup is effectively free. "
        "On CPUs without efficient gather (older x86), the IQ4_XS / Q4_K_M gap may shrink or invert.",
        ha="center", va="bottom", fontsize=6.0, color="#555555", style="italic", wrap=True,
    )

    fig.tight_layout(rect=(0, 0.04, 1, 1))                        #B reserve room for the footnote so tight_layout doesn't overlap it
    name = "CH10_F08_Kalyanarangan"
    paths = _save_pair(plt, fig, cfg, name)
    caption = (
        "Inference turnaround time per generated token, measured during the same 64-token "
        "completion that produced Figure 10.7. The bars run in the opposite direction from "
        "Figure 10.6: smaller-bit-width variants decode faster because the per-token cost "
        "is dominated by memory bandwidth from weight tensors, and the quantized variants "
        "stream less data. Decode runs use Metal on Apple Silicon — on CPUs without "
        "efficient gather instructions the IQ4_XS / Q4_K_M ordering may shrink or invert."
    )
    return paths, caption


def mode_figures(cfg: Config, results: dict) -> dict:
    _backfill_inference_perf(cfg, results)                        #A pull decode timing from existing logs into results.json
    paths04, cap04 = _fig_pipeline(cfg)
    paths05, cap05 = _fig_size_quality(cfg, results)
    paths06, cap06 = _fig_walltime(cfg, results)
    paths07, cap07 = _fig_token_preview(cfg, results)
    paths08, cap08 = _fig_inference_tat(cfg, results)
    return {
        "f04_pipeline":      {"png": paths04[0], "pdf": paths04[1], "caption": cap04},
        "f05_size_quality":  {"png": paths05[0], "pdf": paths05[1], "caption": cap05},
        "f06_walltime":      {"png": paths06[0], "pdf": paths06[1], "caption": cap06},
        "f07_token_preview": {"png": paths07[0], "pdf": paths07[1], "caption": cap07},
        "f08_inference_tat": {"png": paths08[0], "pdf": paths08[1], "caption": cap08},
    }


# --- Dispatcher ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    p.add_argument("--mode", choices=[
        "setup", "download", "convert", "imatrix", "quantize", "verify", "figures", "all",
    ], default="all")
    p.add_argument("--force", action="store_true", help="rebuild llama.cpp even if binaries exist")
    args = p.parse_args()
    cfg = Config()

    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    log(f"=== ch10_convert.py mode={args.mode} ===", cfg.log_file)

    if args.mode == "all":
        free = shutil.disk_usage(BASE).free / (1024 ** 3)
        if free < 50:
            sys.stderr.write(f"only {free:.1f} GB free; need >=50 GB before --mode all\n"); sys.exit(1)
        log(f"disk free: {free:.1f} GB OK", cfg.log_file)

    results = read_results(cfg)
    results["host"] = host_info()                                  #B refresh host info on every run so cross-env reruns don't leave stale x86 details from the wrong venv
    results.setdefault("model", {"id": cfg.repo_id, "params_b": 6.74, "architecture": "llama"})
    results.setdefault("steps", {})
    results.setdefault("verify", {})
    results.setdefault("figures", {})

    if args.mode in ("setup", "all"):
        results["llama_cpp"] = mode_setup(cfg, force=args.force)
    if args.mode in ("download", "all"):
        results.setdefault("hf", {}).update(mode_download(cfg))
    if args.mode in ("convert", "all"):
        results["steps"]["convert_f16"] = mode_convert(cfg)
    if args.mode in ("imatrix", "all"):
        results["steps"]["imatrix"] = mode_imatrix(cfg)
    if args.mode in ("quantize", "all"):
        results["steps"].update(mode_quantize(cfg))
    if args.mode in ("verify", "all"):
        results["verify"] = mode_verify(cfg, results)
    if args.mode in ("figures", "all"):
        results["figures"] = mode_figures(cfg, results)

    write_results(cfg, results)
    log(f"results -> {cfg.results_path.relative_to(BASE)}", cfg.log_file)


if __name__ == "__main__":
    main()
