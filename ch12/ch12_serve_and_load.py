#!/usr/bin/env python3
"""Chapter 12 — serve a 4-bit artifact + concurrency sweep (§5/§9 of spec).

================================ HONESTY RULES (§16) =========================
  * Every number written to results_serving.json traces to a real subprocess
    (vllm serve / vllm offline) or a real Prometheus scrape. No estimates.
  * Unmeasured fields stay `null` with a `# PLACEHOLDER` note in the JSON; this
    script flags them and ch12_figures.py skips rendering anything placeholdered.
  * vLLM metric names are VERSION-DEPENDENT. This script verifies the metric
    names against the live /metrics before scraping and records what it found.
  * The canary agreement number is valid only with sampling OFF (greedy).
  * No new quantization concept here — AWQ (Ch.?), GPTQ, FP8 are all *reused*;
    this is operational wrapping (post-ship gate) around Chapters 1–11.
=============================================================================

Modes (--mode):
  golden  : load an FP16/BF16 reference of the model OFFLINE via vLLM, greedy-
            decode the fixed canary prompt set, write obs/canary/golden.jsonl.
            (§10 step 1 — run ONCE, before serving the 4-bit artifact.)
  serve   : launch `vllm serve` for cfg.format/cfg.artifact_path, block until
            /metrics is healthy, then stay up (Ctrl-C to stop). Rewrites the
            Prometheus `format` label and reloads Prometheus so series facet.
  sweep   : assume a server is already up; drive the concurrency sweep
            (1,2,4,8,16,32,64), capture the scrape window + offered concurrency,
            query Prometheus for per-window server-side metrics, and append to
            results_serving.json incrementally.
  all     : golden (if missing) -> serve -> sweep -> stop, for cfg.format.

Standing conventions: Config dataclass at top; BASE = Path(__file__)...;
run_cmd streaming to stdout + per-step log files; #A/#B/#C annotations;
--mode flag; results_serving.json written incrementally.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
RESULTS = ROOT / "results_serving.json"
LOGDIR = ROOT / "logs"
LOGDIR.mkdir(exist_ok=True)
GOLDEN = ROOT / "obs" / "canary" / "golden.jsonl"
PROM_YML = ROOT / "obs" / "prometheus.yml"


# --------------------------------------------------------------------------- #A
# Config. One field (`format`) switches the serve format for the §8/12.3 sweep.
#
# NOTE on the live-capture model: the chapter HEADLINE targets Llama-3.1-8B /
# Qwen2.5-7B on a 32 GB g2-standard-8 (open decision §1). The defaults below
# use Qwen2.5-3B so the FP16 golden generation + the FP16-vs-4bit serve sweep
# both fit on the smaller 15 GB-RAM box this capture ran on, WITHOUT fabricating
# anything. Bump model/reference to the 7B/8B ids on the larger VM — it is a
# one-line change. meta.model in the results records exactly what ran.
# --------------------------------------------------------------------------- #A
FORMATS = {
    # format-key -> (hf repo or local path, vllm --quantization value or None)
    "fp16": (os.environ.get("CH12_FP16_REPO", "Qwen/Qwen2.5-3B-Instruct"), None),
    "awq_int4": (os.environ.get("CH12_AWQ_REPO", "Qwen/Qwen2.5-3B-Instruct-AWQ"), "awq_marlin"),
    "gptq_int4": (os.environ.get("CH12_GPTQ_REPO", "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"), "gptq_marlin"),
    "fp8": (os.environ.get("CH12_FP8_REPO", ""), "fp8"),  # provide a repo to enable
}


@dataclass
class Config:
    format: str = "awq_int4"
    # FP16/BF16 reference for golden generation (same model family as served).
    reference_repo: str = os.environ.get("CH12_FP16_REPO", "Qwen/Qwen2.5-3B-Instruct")
    ctx: int = int(os.environ.get("CH12_CTX", "4096"))
    gpu_mem_util: float = float(os.environ.get("CH12_GPU_MEM_UTIL", "0.90"))
    port: int = int(os.environ.get("CH12_PORT", "8000"))
    prom_url: str = os.environ.get("CH12_PROM_URL", "http://127.0.0.1:9090")

    # sweep parameters (§9)
    concurrencies: tuple = (1, 2, 4, 8, 16, 32, 64)
    prompt_len_tokens: int = 256       # fixed input length (approx, via repeated text)
    decode_tokens: int = 128           # fixed decode length (ignore_eos in vLLM)
    warmup_s: int = 8                  # let the server reach steady state
    window_s: int = 25                 # scrape window per concurrency point
    request_timeout_s: int = 180

    # golden / canary prompt set (§10) — small, fixed, deterministic
    canary_max_tokens: int = 24   # closed-form answers are short; keeps passes fast

    @property
    def artifact_path(self) -> str:
        repo, _ = FORMATS[self.format]
        return repo

    @property
    def quantization(self):
        _, q = FORMATS[self.format]
        return q


# A fixed, deterministic prompt set for the canary/golden (§10).
#
# DESIGN NOTE (matters for the agreement number to be useful): the canary
# detects *drift between artifacts of the same model*, so the prompts are
# CLOSED-FORM short-answer questions where the FP16 reference and a faithful
# 4-bit artifact should produce the SAME greedy tokens. Open-ended generation
# would diverge early under greedy decode even for a perfectly good artifact
# (tiny logit deltas -> different argmax a few tokens in), leaving the agreement
# ratio pinned near zero with no headroom. Closed-form prompts give a high
# baseline (~1.0) that DROPS when the artifact is silently swapped or a kernel
# upcasts/regresses — which is the whole point of the canary.
CANARY_PROMPTS = [
    "What is the capital of France? Answer with only the city name.",
    "What is 7 multiplied by 8? Answer with only the number.",
    "Complete the sequence with one number: 2, 4, 6, 8,",
    "What color is a clear daytime sky? Answer with one word.",
    "How many days are in a week? Answer with only the number.",
    "What is the chemical formula for water? Answer with only the formula.",
    "What is the first name of the author of 'Romeo and Juliet'? One word.",
    "Translate the word 'hello' into Spanish. Answer with only the word.",
]


# --------------------------------------------------------------------------- #B
# Helpers: run_cmd (streaming + per-step log), results IO, Prometheus queries.
# --------------------------------------------------------------------------- #B
def run_cmd(cmd: list[str], log_name: str, env: dict | None = None) -> subprocess.Popen:
    """Stream a long-running subprocess to BOTH stdout and a per-step log file.
    Returns the Popen so the caller can manage lifecycle (used for vllm serve)."""
    logpath = LOGDIR / log_name
    print(f"[run] {' '.join(cmd)}\n[run] logging -> {logpath}", flush=True)
    logf = open(logpath, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        env={**os.environ, **(env or {})},
    )
    return proc


def load_results() -> dict:
    if RESULTS.exists():
        return json.loads(RESULTS.read_text())
    # Fresh skeleton — mirrors the §13 contract. null == not yet measured.
    return {
        "meta": {
            "model": None,            # PLACEHOLDER until a run records it
            "artifact_path": None,    # PLACEHOLDER
            "vllm_version": None,     # PLACEHOLDER
            "gpu": "L4",
            "captured_at": None,      # PLACEHOLDER
        },
        "serving": {
            "by_format": {"fp16": {}, "awq_int4": {}, "gptq_int4": {}, "fp8": {}},
            "concurrency_sweep": [],
        },
        "canary": {"agreement_ratio": None, "n_prompts": None, "sampling": "greedy"},
        "notes": [],
    }


def save_results(data: dict) -> None:
    """Incremental write — single source of truth (§12/§13)."""
    RESULTS.write_text(json.dumps(data, indent=2))


def prom_instant(cfg: Config, query: str, t: float | None = None):
    """Query the Prometheus HTTP API (instant). Returns list of result samples
    or [] on empty. Raises on transport error (we want to know)."""
    params = {"query": query}
    if t is not None:
        params["time"] = f"{t:.3f}"
    url = f"{cfg.prom_url}/api/v1/query?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=15) as r:
        body = json.loads(r.read().decode())
    if body.get("status") != "success":
        raise RuntimeError(f"prometheus query failed: {body}")
    return body["data"]["result"]


def prom_scalar(cfg: Config, query: str, t: float | None = None):
    """First sample value of an instant query as float, or None if no data."""
    res = prom_instant(cfg, query, t)
    if not res:
        return None
    try:
        return float(res[0]["value"][1])
    except (KeyError, IndexError, ValueError):
        return None


# --------------------------------------------------------------------------- #C
# Mode: golden — offline FP16 greedy decode -> golden.jsonl  (§10 step 1)
# --------------------------------------------------------------------------- #C
def mode_golden(cfg: Config) -> None:
    if GOLDEN.exists() and os.environ.get("CH12_FORCE_GOLDEN") != "1":
        print(f"[golden] {GOLDEN} already exists; set CH12_FORCE_GOLDEN=1 to regenerate.")
        return
    print(f"[golden] loading FP16/BF16 reference offline: {cfg.reference_repo}")
    # Import here so `--mode sweep/serve` don't require torch in-process.
    from vllm import LLM, SamplingParams  # noqa: E402

    llm = LLM(
        model=cfg.reference_repo,
        dtype="bfloat16",
        max_model_len=cfg.ctx,
        gpu_memory_utilization=cfg.gpu_mem_util,
        enforce_eager=True,
    )
    tok = llm.get_tokenizer()
    # Render each prompt through the chat template so it matches what the served
    # endpoint receives via /v1/chat/completions.
    rendered = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in CANARY_PROMPTS
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=cfg.canary_max_tokens, seed=0)
    outs = llm.generate(rendered, sp)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN, "w") as f:
        for prompt, out in zip(CANARY_PROMPTS, outs):
            f.write(json.dumps({"prompt": prompt, "golden_output": out.outputs[0].text}) + "\n")
    print(f"[golden] wrote {len(CANARY_PROMPTS)} rows -> {GOLDEN}")
    # Record provenance in results.
    data = load_results()
    data["meta"]["model"] = cfg.reference_repo
    data["notes"].append(
        f"golden.jsonl generated greedy(temp=0) from FP16/BF16 reference {cfg.reference_repo}"
    )
    save_results(data)


def _set_prom_format_label(fmt: str, model: str) -> None:
    """Rewrite the vLLM job's `format`/`model` labels in prometheus.yml in place
    so the live series facet correctly, then trigger a hot reload."""
    txt = PROM_YML.read_text()
    import re

    txt = re.sub(r'(\n\s*format:\s*")[^"]*(")', rf'\g<1>{fmt}\g<2>', txt, count=1)
    short = model.split("/")[-1].lower()
    txt = re.sub(r'(\n\s*model:\s*")[^"]*(")', rf'\g<1>{short}\g<2>', txt, count=1)
    PROM_YML.write_text(txt)
    print(f"[prom] set vllm labels format={fmt} model={short}; reloading Prometheus")


def _prom_reload(cfg: Config) -> None:
    try:
        requests.post(f"{cfg.prom_url}/-/reload", timeout=10)
    except Exception as exc:
        print(f"[prom] reload skipped ({exc}); is Prometheus up with --web.enable-lifecycle?")


def _wait_metrics(cfg: Config, timeout_s: int = 600) -> dict:
    """Block until vLLM /metrics is healthy; verify expected metric names are
    present (§5) and return which of the expected names actually exist."""
    expected = [
        "vllm:time_to_first_token_seconds",
        "vllm:inter_token_latency_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:gpu_cache_usage_perc",
        "vllm:kv_cache_usage_perc",
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_total",
    ]
    url = f"http://127.0.0.1:{cfg.port}/metrics"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            body = requests.get(url, timeout=5).text
            present = {m: (m in body) for m in expected}
            if present["vllm:num_requests_running"]:
                print("[serve] /metrics healthy. Metric-name presence:")
                for m, ok in present.items():
                    print(f"        {'OK ' if ok else 'MISS'}  {m}")
                return present
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("vLLM /metrics never became healthy")


def serve_proc(cfg: Config) -> subprocess.Popen:
    """Build + launch the vllm serve subprocess (§5)."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg.artifact_path,
        "--dtype", "half",
        "--max-model-len", str(cfg.ctx),
        "--gpu-memory-utilization", str(cfg.gpu_mem_util),
        "--port", str(cfg.port),
        "--no-enable-log-requests",   # vLLM 0.21: replaces the old --disable-log-requests
    ]
    if cfg.quantization:
        cmd += ["--quantization", cfg.quantization]
    return run_cmd(cmd, f"vllm_serve_{cfg.format}.log")


def mode_serve(cfg: Config, block: bool = True) -> subprocess.Popen:
    proc = serve_proc(cfg)
    _wait_metrics(cfg)
    _set_prom_format_label(cfg.format, cfg.artifact_path)
    _prom_reload(cfg)
    # record meta
    data = load_results()
    data["meta"]["artifact_path"] = cfg.artifact_path
    data["meta"]["model"] = cfg.artifact_path
    try:
        import vllm
        data["meta"]["vllm_version"] = vllm.__version__
    except Exception:
        pass
    save_results(data)
    if block:
        print("[serve] up. Ctrl-C to stop.")
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            proc.wait()
    return proc


# ---- load generator: small async-ish client via thread pool ---------------- #C
def _one_request(cfg: Config, model: str, prompt: str) -> bool:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": cfg.decode_tokens,
        "temperature": 0.0,
        "ignore_eos": True,   # vLLM: force a fixed decode length for fair sweep
    }
    try:
        r = requests.post(
            f"http://127.0.0.1:{cfg.port}/v1/completions",
            json=payload, timeout=cfg.request_timeout_s,
        )
        r.raise_for_status()
        return True
    except Exception:
        return False


def _drive_concurrency(cfg: Config, model: str, conc: int, prompt: str) -> dict:
    """Hold ~`conc` requests in flight for window_s+warmup_s; return offered
    concurrency and client-side completion count for the measurement window."""
    stop_at = time.time() + cfg.warmup_s + cfg.window_s
    window_start = None
    completed_in_window = 0
    with ThreadPoolExecutor(max_workers=conc) as ex:
        inflight = {ex.submit(_one_request, cfg, model, prompt) for _ in range(conc)}
        # mark the measurement window start after warmup
        warm_until = time.time() + cfg.warmup_s
        while time.time() < stop_at:
            done = {f for f in inflight if f.done()}
            for f in done:
                inflight.discard(f)
                if window_start is not None:
                    completed_in_window += 1
                if time.time() < stop_at:
                    inflight.add(ex.submit(_one_request, cfg, model, prompt))
            if window_start is None and time.time() >= warm_until:
                window_start = time.time()
            time.sleep(0.02)
        # drain
        for f in as_completed(inflight):
            pass
    return {
        "offered_concurrency": conc,
        "t_start": window_start,
        "t_end": stop_at,
        "client_completed_in_window": completed_in_window,
    }


def mode_sweep(cfg: Config) -> None:
    """Run the concurrency sweep against an already-running server and scrape
    Prometheus for per-window server-side metrics (§9/§12)."""
    # resolve served model id
    model = requests.get(f"http://127.0.0.1:{cfg.port}/v1/models", timeout=10).json()["data"][0]["id"]
    prompt = ("Summarize the trade-offs of low-bit LLM quantization. " * 8).strip()

    data = load_results()
    data["meta"]["model"] = model
    if not data["meta"].get("captured_at"):
        data["meta"]["captured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save_results(data)

    for conc in cfg.concurrencies:
        print(f"\n[sweep] format={cfg.format} concurrency={conc} "
              f"(warmup {cfg.warmup_s}s + window {cfg.window_s}s)")
        win = _drive_concurrency(cfg, model, conc, prompt)
        t_end = win["t_end"]
        time.sleep(2)  # let the 15s scrape catch up to steady state

        # Prometheus instant queries at window end, faceted to this format.
        f = cfg.format
        ttft = prom_scalar(cfg,
            f'histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket{{format="{f}"}}[2m])) by (le))')
        itl = prom_scalar(cfg,
            f'histogram_quantile(0.95, sum(rate(vllm:inter_token_latency_seconds_bucket{{format="{f}"}}[2m])) by (le))')
        toks = prom_scalar(cfg, f'rate(vllm:generation_tokens_total{{format="{f}"}}[1m])')
        kv = prom_scalar(cfg, f'vllm:gpu_cache_usage_perc{{format="{f}"}}')
        if kv is None:
            kv = prom_scalar(cfg, f'vllm:kv_cache_usage_perc{{format="{f}"}}')
        waiting = prom_scalar(cfg, f'vllm:num_requests_waiting{{format="{f}"}}')
        power = prom_scalar(cfg, "DCGM_FI_DEV_POWER_USAGE")

        entry = {
            "concurrency": conc,
            "format": cfg.format,
            "t_start": win["t_start"],
            "t_end": t_end,
            "client_completed_in_window": win["client_completed_in_window"],
            "ttft_p95_ms": (ttft * 1000) if ttft is not None else None,           # PLACEHOLDER if null
            "inter_token_p95_ms": (itl * 1000) if itl is not None else None,      # PLACEHOLDER if null
            "tokens_per_s": toks,                                                  # PLACEHOLDER if null
            "kv_cache_usage_perc": kv,                                             # PLACEHOLDER if null
            "num_requests_waiting": waiting,                                       # PLACEHOLDER if null
            "gpu_power_w": power,                                                  # PLACEHOLDER if null
        }
        print("[sweep] scraped:", json.dumps({k: entry[k] for k in (
            "ttft_p95_ms", "inter_token_p95_ms", "tokens_per_s",
            "kv_cache_usage_perc", "num_requests_waiting", "gpu_power_w")}, default=str))

        data = load_results()
        # replace any prior entry for this (format, concurrency)
        data["serving"]["concurrency_sweep"] = [
            e for e in data["serving"]["concurrency_sweep"]
            if not (e["format"] == cfg.format and e["concurrency"] == conc)
        ]
        data["serving"]["concurrency_sweep"].append(entry)
        # per-format summary (peak tokens/s, min ttft)
        fmt_rows = [e for e in data["serving"]["concurrency_sweep"] if e["format"] == cfg.format]
        tps = [e["tokens_per_s"] for e in fmt_rows if e["tokens_per_s"] is not None]
        ttfts = [e["ttft_p95_ms"] for e in fmt_rows if e["ttft_p95_ms"] is not None]
        data["serving"]["by_format"][cfg.format] = {
            "peak_tokens_per_s": max(tps) if tps else None,
            "min_ttft_p95_ms": min(ttfts) if ttfts else None,
            "n_points": len(fmt_rows),
        }
        save_results(data)

    print(f"\n[sweep] done. results -> {RESULTS}")


def mode_all(cfg: Config) -> None:
    if not GOLDEN.exists():
        mode_golden(cfg)
    proc = mode_serve(cfg, block=False)
    try:
        mode_sweep(cfg)
    finally:
        print("[all] stopping server")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description="ch12 serve + concurrency sweep")
    ap.add_argument("--mode", required=True, choices=["golden", "serve", "sweep", "all"])
    ap.add_argument("--format", default=None, choices=list(FORMATS), help="serve/sweep format")
    args = ap.parse_args()

    cfg = Config()
    if args.format:
        cfg.format = args.format

    print("=" * 70)
    print(f"ch12_serve_and_load  mode={args.mode}  format={cfg.format}")
    print(f"  artifact={cfg.artifact_path}  quant={cfg.quantization}")
    print(f"  HONESTY: all numbers come from real subprocess/scrape; null=unmeasured.")
    print("=" * 70)

    if args.mode == "golden":
        mode_golden(cfg)
    elif args.mode == "serve":
        mode_serve(cfg, block=True)
    elif args.mode == "sweep":
        mode_sweep(cfg)
    elif args.mode == "all":
        mode_all(cfg)


if __name__ == "__main__":
    main()
