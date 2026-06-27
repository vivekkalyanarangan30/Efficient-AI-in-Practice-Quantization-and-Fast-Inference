"""Chapter 12 — PRE-SHIP GATE tests (offline, deterministic, hard-assert).

Framing (§0): tests are the pre-ship gate; observability is the post-ship gate.
These tests assert the *config and code* are correct and honest BEFORE the stack
runs — they do not need a GPU, vLLM, or a live server. The post-ship gate
(Prometheus/Grafana/canary) is exercised by the end-to-end run, not here.

Run:  ./.venv/bin/python -m pytest tests/ -q
"""
import importlib.util
import json
from pathlib import Path

import yaml
import pytest

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "obs"


def _load(modpath: Path, name: str):
    import sys
    spec = importlib.util.spec_from_file_location(name, modpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclass decorator looks the module up here
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1. docker-compose: all required services, every image pinned (no :latest).
# --------------------------------------------------------------------------- #
def test_compose_services_and_pinned_tags():
    dc = yaml.safe_load((OBS / "docker-compose.yml").read_text())
    svcs = dc["services"]
    for required in ("dcgm-exporter", "node-exporter", "pushgateway", "prometheus", "grafana"):
        assert required in svcs, f"missing service {required}"
    for name, svc in svcs.items():
        img = svc["image"]
        assert ":" in img and not img.endswith(":latest"), f"{name} image not pinned: {img}"


def test_pushgateway_bound_localhost_only():
    """§11/§15: pushgateway must never be publicly exposed."""
    dc = yaml.safe_load((OBS / "docker-compose.yml").read_text())
    for p in dc["services"]["pushgateway"].get("ports", []):
        assert str(p).startswith("127.0.0.1:"), f"pushgateway port not localhost-bound: {p}"


def test_operator_ports_localhost_only():
    dc = yaml.safe_load((OBS / "docker-compose.yml").read_text())
    for svc in ("grafana", "prometheus", "pushgateway"):
        for p in dc["services"][svc].get("ports", []):
            assert str(p).startswith("127.0.0.1:"), f"{svc} exposes {p} publicly"


# --------------------------------------------------------------------------- #
# 2. prometheus.yml: the five scrape jobs from §7 exist.
# --------------------------------------------------------------------------- #
def test_prometheus_scrape_jobs():
    cfg = yaml.safe_load((OBS / "prometheus.yml").read_text())
    jobs = {j["job_name"] for j in cfg["scrape_configs"]}
    assert {"vllm", "dcgm", "node", "canary", "pushgateway"} <= jobs


def test_pushgateway_honor_labels():
    cfg = yaml.safe_load((OBS / "prometheus.yml").read_text())
    pgw = [j for j in cfg["scrape_configs"] if j["job_name"] == "pushgateway"][0]
    assert pgw.get("honor_labels") is True, "pushgateway scrape must keep M3-pushed labels"


# --------------------------------------------------------------------------- #
# 3. Grafana provisioning + custom dashboard (§8).
# --------------------------------------------------------------------------- #
def test_datasource_default_prometheus():
    ds = yaml.safe_load((OBS / "grafana/provisioning/datasources/prometheus.yml").read_text())
    d = ds["datasources"][0]
    assert d["type"] == "prometheus" and d["isDefault"] is True
    assert d["uid"] == "prometheus"


def test_custom_dashboard_valid_and_four_panels():
    dash = json.loads((OBS / "grafana/dashboards/ch12_quant_payoff.json").read_text())
    panels = dash["panels"]
    assert len(panels) == 4, "§8: four panels only"
    titles = " ".join(p["title"] for p in panels).lower()
    # chapter vocabulary present somewhere across the dashboard (§8)
    for vocab in ("kv-cache", "queue depth", "ttft", "inter-token", "power", "canary"):
        assert vocab in titles or any(
            vocab in (t.get("legendFormat", "") + p.get("description", "")).lower()
            for p in panels for t in p.get("targets", [])
        ), f"missing chapter vocabulary: {vocab}"


def test_custom_dashboard_promql_metric_names():
    dash = json.loads((OBS / "grafana/dashboards/ch12_quant_payoff.json").read_text())
    exprs = " ".join(t["expr"] for p in dash["panels"] for t in p.get("targets", []))
    for metric in (
        "vllm:kv_cache_usage_perc",   # 0.21 name; gpu_cache_usage_perc kept as `or` fallback
        "vllm:num_requests_waiting",
        "vllm:generation_tokens_total",
        "vllm:time_to_first_token_seconds_bucket",
        "vllm:inter_token_latency_seconds_bucket",
        "DCGM_FI_DEV_POWER_USAGE",
        "ch12_canary_agreement_ratio",
    ):
        assert metric in exprs, f"custom dashboard missing PromQL metric {metric}"


def test_latency_panel_converts_seconds_to_ms():
    """§8: vLLM reports seconds; latency exprs must *1000 for the ms panel."""
    dash = json.loads((OBS / "grafana/dashboards/ch12_quant_payoff.json").read_text())
    lat_exprs = [
        t["expr"] for p in dash["panels"] for t in p.get("targets", [])
        if "latency_seconds_bucket" in t["expr"] or "time_to_first_token_seconds_bucket" in t["expr"]
    ]
    assert lat_exprs
    assert all("* 1000" in e for e in lat_exprs), "latency exprs must convert s->ms"


# --------------------------------------------------------------------------- #
# 4. Canary logic — the deterministic snapshot-style test (mirrors 12.1, §10).
# --------------------------------------------------------------------------- #
def test_token_agreement_function():
    cx = _load(OBS / "canary/canary_exporter.py", "canary_exporter")
    assert cx._token_agreement("a b c", "a b c") == 1.0
    assert cx._token_agreement("", "") == 1.0
    # diverges after 2 tokens of 4 -> 0.5
    assert cx._token_agreement("a b X d", "a b c d") == pytest.approx(0.5)
    # length mismatch penalised by max-length denominator
    assert cx._token_agreement("a b", "a b c d") == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# 5. results_serving.json contract (§13) — skeleton has the right shape.
# --------------------------------------------------------------------------- #
def test_results_skeleton_contract():
    sl = _load(ROOT / "scripts/ch12_serve_and_load.py", "ch12_serve_and_load")
    # use a tmp results file by pointing module RESULTS at a non-existent path
    import tempfile, os
    d = sl.load_results()
    assert set(d.keys()) >= {"meta", "serving", "canary"}
    assert set(d["meta"].keys()) >= {"model", "artifact_path", "vllm_version", "gpu", "captured_at"}
    assert "by_format" in d["serving"] and "concurrency_sweep" in d["serving"]
    assert d["canary"]["sampling"] == "greedy"  # §10: greedy only


# --------------------------------------------------------------------------- #
# 6. Figure HONESTY (§16): null fields are NOT rendered; populated ones are.
# --------------------------------------------------------------------------- #
def test_figures_skip_on_null_and_render_on_data(tmp_path, monkeypatch):
    fg = _load(ROOT / "scripts/ch12_figures.py", "ch12_figures")
    cfg = fg.Config()
    monkeypatch.setattr(fg, "FIGDIR", tmp_path)

    # all-null sweep -> figure must be SKIPPED, no file written
    null_data = {"serving": {"concurrency_sweep": [
        {"concurrency": 1, "format": "awq_int4", "kv_cache_usage_perc": None,
         "num_requests_waiting": None, "tokens_per_s": None, "ttft_p95_ms": None}]}}
    paths, cap = fg._fig_kv_queue(null_data, cfg)
    assert paths is None and cap.startswith("SKIPPED")

    # real-shaped data -> figure rendered (this is test fixture data, not chapter data)
    good = {"serving": {"concurrency_sweep": [
        {"concurrency": 1, "format": "awq_int4", "kv_cache_usage_perc": 0.10,
         "num_requests_waiting": 0.0, "tokens_per_s": 50.0, "ttft_p95_ms": 30.0},
        {"concurrency": 8, "format": "awq_int4", "kv_cache_usage_perc": 0.55,
         "num_requests_waiting": 2.0, "tokens_per_s": 300.0, "ttft_p95_ms": 80.0},
    ]}}
    fg._setup_mpl()
    paths, cap = fg._fig_kv_queue(good, cfg)
    assert paths and all(Path(p).exists() for p in paths)
    assert any(p.endswith(".pdf") for p in paths) and any(p.endswith(".png") for p in paths)
