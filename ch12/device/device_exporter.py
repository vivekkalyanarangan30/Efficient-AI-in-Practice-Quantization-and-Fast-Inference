# Honesty rules (Ch.12 M3 device session):
# 1. This exporter ships numbers that other scripts already wrote to
#    results_device.json. It does not synthesize values. If a field is
#    `null`, it is skipped — never sent as 0 or a placeholder.
# 2. Power is labelled `_estimated` in the metric name (Ch.11 caveat).
# 3. Unmeasured fields stay `null` with `# PLACEHOLDER` upstream.
# 4. No new quantization concepts.
"""Push ch12_device_* metrics to the VM Pushgateway under job `ch12_device_vision`.

Reads results_device.json, builds a prometheus_client registry, pushes
once over an authenticated path. The Pushgateway URL and credentials
come from environment variables — never hardcoded, never logged.

    export PUSHGATEWAY_URL="https://vm.example.invalid:9091"
    export PUSHGATEWAY_USER="ch12_device"
    export PUSHGATEWAY_PASS="<from secret manager>"
    python device/device_exporter.py --mode push
"""
from __future__ import annotations

import argparse
import os
import ssl
import sys
import urllib.error
import urllib.request
from base64 import b64encode

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from _common import load_results


METRIC_PREFIX = "ch12_device_"
JOB = "ch12_device_vision"


def _build_registry(results: dict) -> CollectorRegistry:
    reg = CollectorRegistry()
    dev = results.get("device", {})

    stage_g = Gauge(
        f"{METRIC_PREFIX}stage_latency_ms",
        "Per-stage wall time on the M3 device (mean over the sample budget).",
        ["stage"], registry=reg,
    )
    for stage, val in (dev.get("stage_latency_ms") or {}).items():
        if val is None:
            continue
        stage_g.labels(stage=stage).set(float(val))

    if (e2e := dev.get("e2e_p95_ms")) is not None:
        Gauge(f"{METRIC_PREFIX}e2e_p95_ms",
              "End-to-end p95 latency on the M3.", registry=reg).set(float(e2e))

    if (power := dev.get("power_w_estimated")) is not None:
        # `_estimated` in the metric name is load-bearing — Ch.11 honesty rule.
        Gauge(f"{METRIC_PREFIX}power_w_estimated",
              "Estimated combined CPU+GPU+ANE power (powermetrics; not sensor-grade).",
              registry=reg).set(float(power))

    placement_g = Gauge(
        f"{METRIC_PREFIX}op_placement_frac",
        "Fraction of compute ops scheduled on each unit (MLComputePlan).",
        ["unit"], registry=reg,
    )
    for unit, frac in (dev.get("op_placement_frac") or {}).items():
        if frac is None:
            continue
        placement_g.labels(unit=unit).set(float(frac))

    if (ret := dev.get("thermal_retention_pct")) is not None:
        Gauge(f"{METRIC_PREFIX}thermal_retention_pct",
              "Sustained-throughput retention (last-30s / first-30s * 100).",
              registry=reg).set(float(ret))
    return reg


def _basic_auth_handler(url, method, timeout, headers, data):
    """Return a urllib handler that adds Basic auth and enforces TLS verification.

    The handler also disables auto-redirects so a malicious gateway cannot
    bounce credentials off-host (SSRF-style hop trap).

    For self-hosted / private-CA deployments, the operator can set
    `PUSHGATEWAY_CA_BUNDLE` to a PEM file. TLS verification stays ON —
    only the trust anchor changes. We never accept self-signed certs by
    silently disabling verification.
    """
    user = os.environ.get("PUSHGATEWAY_USER", "")
    pw = os.environ.get("PUSHGATEWAY_PASS", "")
    ca_bundle = os.environ.get("PUSHGATEWAY_CA_BUNDLE", "")
    if not user or not pw:
        raise SystemExit("PUSHGATEWAY_USER and PUSHGATEWAY_PASS must be set "
                         "(do not push over an unauthenticated endpoint).")

    if not url.lower().startswith("https://"):
        # Refuse plaintext push for credentials; chapter spec also requires this.
        raise SystemExit("PUSHGATEWAY_URL must be https:// — credentials over TLS only.")

    token = b64encode(f"{user}:{pw}".encode("utf-8")).decode("ascii")
    request_headers = list(headers) + [("Authorization", f"Basic {token}")]

    def handle():
        request = urllib.request.Request(url=url, data=data, method=method)
        for k, v in request_headers:
            request.add_header(k, v)
        # Default ssl.create_default_context() verifies hostname + chain.
        # If a CA bundle was supplied, validate it's a real file we can read
        # before passing it in — fail closed on a bad path rather than
        # silently falling back to the system store.
        if ca_bundle:
            from pathlib import Path as _Path
            cab = _Path(ca_bundle).expanduser().resolve()
            if not cab.is_file():
                raise SystemExit("PUSHGATEWAY_CA_BUNDLE points at a missing file.")
            ctx = ssl.create_default_context(cafile=str(cab))
        else:
            ctx = ssl.create_default_context()
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED

        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *_a, **_kw):  # type: ignore[override]
                return None

        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ctx), _NoRedirect()
        )
        with opener.open(request, timeout=timeout) as resp:
            resp.read()
    return handle


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Push ch12_device_* metrics to Pushgateway")
    ap.add_argument("--mode", choices=["push"], default="push")
    ap.add_argument("--instance", default=os.uname().nodename,
                    help="Prometheus `instance` label (defaults to hostname)")
    args = ap.parse_args(argv)

    url = os.environ.get("PUSHGATEWAY_URL", "")
    if not url:
        raise SystemExit("PUSHGATEWAY_URL is not set; refusing to push.")

    results = load_results()
    reg = _build_registry(results)
    if not list(reg.collect())[0].samples and len(list(reg.collect())) <= 1:
        # Nothing measured yet; don't push an empty registry to a shared gateway.
        print("[device_exporter] no measured metrics in results_device.json yet; "
              "run vision_pipeline / op_placement / thermal_loop first.",
              file=sys.stderr)
        sys.exit(2)

    try:
        push_to_gateway(
            url, job=JOB, registry=reg,
            grouping_key={"instance": args.instance},
            handler=_basic_auth_handler,
            timeout=15,
        )
    except urllib.error.URLError as e:
        # Generic client-facing error; do not leak the URL or auth state into logs.
        raise SystemExit(f"push failed: {e.__class__.__name__}")
    print(f"[device_exporter] pushed metrics to job={JOB} (instance={args.instance})")


if __name__ == "__main__":
    main()
