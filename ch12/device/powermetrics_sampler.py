# Honesty rules (Ch.12 M3 device session):
# 1. powermetrics returns an ESTIMATED power figure. Every number this script
#    writes carries `_estimated` in its name and the caveat in any caption.
#    MetricKit is daily-aggregated, so it cannot replace this signal at the
#    per-run granularity the chapter needs. The Ch.11 constraint stands.
# 2. The number traces to a real `powermetrics` subprocess on this M3.
# 3. Unmeasured fields stay `null` with `# PLACEHOLDER`.
# 4. No new quantization concepts.
"""Sample CPU/GPU/ANE estimated power via `powermetrics` (sudo required).

Runs out-of-band: start this in one terminal, run vision_pipeline.py in
another, stop with Ctrl-C (or use --duration). Records the mean
combined-package estimated power in watts to results_device.json under
`power_w_estimated`.

`powermetrics` requires sudo. Do NOT embed a password here — the operator
either has cached sudo credentials or runs the script under `sudo -E`.

    sudo -E python device/powermetrics_sampler.py --mode run --duration 120
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from _common import LOG_DIR, update_results

POWERMETRICS_BIN = "/usr/bin/powermetrics"            #A absolute path; no PATH lookup
SAMPLE_INTERVAL_MS = 1000                              #A 1 Hz; matches Ch.11 cadence

# Lines look like: "Combined Power (CPU + GPU + ANE): 3456 mW"
_RE_COMBINED = re.compile(
    r"Combined Power.*?:\s*([0-9]+(?:\.[0-9]+)?)\s*mW", re.IGNORECASE
)


def _require_powermetrics() -> None:
    if not Path(POWERMETRICS_BIN).exists():
        raise SystemExit(f"{POWERMETRICS_BIN} not found — this script is M3-only")


def _run_sampler(duration_s: int) -> list[float]:
    cmd = [
        POWERMETRICS_BIN,
        "--samplers", "cpu_power,gpu_power,ane_power",
        "-i", str(SAMPLE_INTERVAL_MS),
        "-n", str(max(1, duration_s)),
        "--show-process-energy",
    ]
    print(f"[powermetrics] launching: {' '.join(cmd)}")
    log_path = LOG_DIR / "powermetrics.log"
    samples_mw: list[float] = []
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                logf.write(line)
                m = _RE_COMBINED.search(line)
                if m:
                    samples_mw.append(float(m.group(1)))
                    sys.stdout.write(f"  combined_estimated={samples_mw[-1]:.1f} mW\n")
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
        finally:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return samples_mw


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="powermetrics-based ESTIMATED power sampler")
    ap.add_argument("--mode", choices=["run"], default="run")
    ap.add_argument("--duration", type=int, default=120,
                    help="seconds to sample at 1 Hz")
    args = ap.parse_args(argv)

    _require_powermetrics()
    if os.geteuid() != 0:
        # Don't try to escalate ourselves — the chapter is explicit that the
        # user runs this via `sudo -E`. Fail closed with a clear message.
        raise SystemExit("powermetrics requires root; rerun under `sudo -E`.")

    samples = _run_sampler(args.duration)
    if not samples:
        raise SystemExit("no Combined Power lines parsed; aborting without writing")

    mean_mw = sum(samples) / len(samples)
    mean_w = mean_mw / 1000.0
    update_results({"device": {"power_w_estimated": round(mean_w, 3)}})
    print(f"\n[powermetrics] mean ESTIMATED combined power: {mean_w:.3f} W "
          f"over {len(samples)} samples (label: estimated, not sensor-grade)")


if __name__ == "__main__":
    main()
