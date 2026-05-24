"""ch11_4_android.py — Android TFLite ingestion (sec 11.2/11.3 mobile half).

Section served: 11.2 (TFLite on-device) + 11.3 (NNAPI delegate). Records produced
here populate the phone-class TFLite slot in the chapter's results.json so the
existing aggregator (ch11_1_aggregate.py) can render the design-space, sustained,
and budget figures with Android points alongside the Mac/iPhone ones.

Modes:
  unpack-artifacts   --zip <path> [--out runs/<name>]
                     Extract the AWS Device Farm "Customer Artifacts" ZIP and
                     locate results-android.json. Stages the JSON under
                     ch11/android/runs/<name>/results-android.json for ingest.
  ingest-apk-results --input <path> [--device-class phone]
                     Parse a results-android.json produced by ch11-bench.apk on
                     a real Android device, validate each record against
                     ResultRecord, upsert into ch11/results.json by dedup key.
  smoke              Run shape checks on the schema with a tiny synthetic
                     payload; never touches results.json.
  figures            Convenience pass-through to `ch11_1_aggregate.py all`.

This script never reaches the network and never executes anything inside the
APK. It is strictly an ingestion + validation helper that lives on the Mac.

Invocation:
  python ch11_4_android.py unpack-artifacts --zip ~/Downloads/run-XYZ.zip
  python ch11_4_android.py ingest-apk-results --input android/runs/run-XYZ/results-android.json
  python ch11_4_android.py figures
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Re-use existing infrastructure rather than duplicating the dataclass / writers.
import ch11_3_apple as apple  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "results.json"
CAVEATS_MD = HERE / "caveats.md"
LOG_DIR = HERE / "logs"
ANDROID_RUNS = HERE / "android" / "runs"

SCRIPT_NAME = "ch11_4_android.py"
SCHEMA_VERSION = "11.0"

# Allow-list of model names the chapter knows about. Records with unknown models
# are rejected so we don't silently pollute the aggregator with stray rows.
KNOWN_MODELS = {
    "efficientnet_lite0", "whisper_tiny", "llama_3_2_1b_instruct",
    # Android-side fallbacks when the Llama-3.2-1B `.task` isn't available
    # (Kaggle license-gate or local conversion failure). LLMBenchmark.kt
    # derives the model name from the .task filename, so adding the model
    # ID here is what gates ingestion.
    "tinyllama_1_1b_chat",  # litert-community/TinyLlama-1.1B-Chat-v1.0
    "gemma_2_2b_it",        # litert-community/Gemma2-2B-IT (HF license-gated)
    "qwen_2_5_1_5b_instruct",  # litert-community/Qwen2.5-1.5B-Instruct (ungated)
}
# litertlm added for the on-device Android LLM path (LiteRT-LM 0.12.0, the
# .litertlm bundle produced by litert-lm-builder). Supersedes the earlier
# MediaPipe LLM Inference path, which only read legacy .task FlatBuffer-ZIP
# bundles and could not load .litertlm output from the chapter's conversion
# pipeline. The Apple-side llama record uses backend="mlx" (laptop) and
# "coreml" (iPhone); Android now uses "litertlm".
KNOWN_BACKENDS = {"tflite", "coreml", "mlx", "mps", "prepost", "litertlm"}
KNOWN_DEVICE_CLASSES = {"laptop", "phone", "sbc", "nuc", "jetson"}

# Allowed top-level keys on a record. Anything outside this set is dropped with
# a warning. This keeps mass-assignment style bugs (CWE-915) out of the merge.
ALLOWED_RECORD_KEYS = {
    "model", "modality", "variant", "backend", "device", "size_bytes", "params",
    "quantization", "compute_units", "latency_ms", "throughput", "accuracy",
    "power_mw", "sustained", "ane_op_coverage", "memory_mb", "prepost",
    "timestamp", "script", "notes", "energy_per_inference_mj",
}


def _setup_logger(mode: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"ch11_4_android.{mode}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_DIR / f"ch11_4_android_{mode}.log", mode="a", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------- #
# unpack-artifacts                                                            #
# --------------------------------------------------------------------------- #
def mode_unpack_artifacts(args: argparse.Namespace) -> int:
    logger = _setup_logger("unpack")
    zip_path = Path(args.zip).expanduser().resolve()
    if not zip_path.is_file():
        logger.error("zip not found: %s", zip_path)
        return 1
    out_name = args.out or zip_path.stem
    out_dir = ANDROID_RUNS / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    found: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            # Safety: skip path-traversal entries (CWE-22). zipfile.extract handles this
            # since Py 3.12, but we double-check.
            if name.startswith("/") or ".." in Path(name).parts:
                logger.warning("skipping suspicious entry: %s", name)
                continue
            if name.endswith("results-android.json") or name.endswith("results-android-error.txt"):
                target = out_dir / Path(name).name
                with zf.open(name) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                found.append(target)
                logger.info("extracted %s", target.relative_to(HERE))

    if not found:
        logger.error("no results-android.json found in %s", zip_path)
        logger.error("zip contents (first 20): %s",
                     zipfile.ZipFile(zip_path).namelist()[:20])
        return 1

    logger.info("done — %d file(s) staged under %s", len(found), out_dir.relative_to(HERE))
    print(f"\nNext: python {SCRIPT_NAME} ingest-apk-results --input "
          f"{(out_dir / 'results-android.json').relative_to(HERE)}")
    return 0


# --------------------------------------------------------------------------- #
# ingest-apk-results                                                          #
# --------------------------------------------------------------------------- #
def mode_ingest(args: argparse.Namespace) -> int:
    logger = _setup_logger("ingest")
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        logger.error("input not found: %s", input_path)
        return 1

    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "records" not in raw:
        logger.error("expected top-level object with 'records' array, got: %s",
                     type(raw).__name__)
        return 1

    schema_in = raw.get("schema_version", "<missing>")
    if schema_in != SCHEMA_VERSION:
        logger.warning("schema_version mismatch: got %r, expected %r — proceeding "
                       "with allow-listed fields only", schema_in, SCHEMA_VERSION)

    records_in = raw["records"]
    if not isinstance(records_in, list):
        logger.error("'records' is not a list")
        return 1

    added = 0
    replaced = 0
    rejected = 0
    for idx, rec in enumerate(records_in):
        try:
            sanitized = _sanitize_record(rec, logger, idx)
        except _RejectRecord as exc:
            logger.warning("record %d rejected: %s", idx, exc)
            rejected += 1
            continue
        # Round-trip through ResultRecord to enforce dataclass shape exactly.
        try:
            rr = apple.ResultRecord(**sanitized)
        except TypeError as exc:
            logger.warning("record %d failed dataclass validation: %s", idx, exc)
            rejected += 1
            continue
        outcome = apple.append_result(rr)
        if outcome == "added":
            added += 1
        else:
            replaced += 1

    logger.info("ingest summary: added=%d replaced=%d rejected=%d (of %d input)",
                added, replaced, rejected, len(records_in))

    if rejected:
        _append_caveat(
            f"ch11_4_android.py ingest-apk-results: {rejected} record(s) "
            f"rejected from {input_path.name}; see logs/ch11_4_android_ingest.log",
        )
    return 0 if rejected == 0 else 2


class _RejectRecord(ValueError):
    """Raised by _sanitize_record when a record cannot be safely admitted."""


def _sanitize_record(rec: Any, logger: logging.Logger, idx: int) -> dict:
    if not isinstance(rec, dict):
        raise _RejectRecord(f"not a JSON object (type={type(rec).__name__})")

    # Drop any fields outside the allow-list. This is the mass-assignment guard.
    unknown = set(rec.keys()) - ALLOWED_RECORD_KEYS
    if unknown:
        logger.debug("record %d dropping unknown fields: %s", idx, sorted(unknown))
    safe = {k: v for k, v in rec.items() if k in ALLOWED_RECORD_KEYS}

    # Required fields
    for k in ("model", "variant", "backend", "device"):
        if k not in safe:
            raise _RejectRecord(f"missing required field {k!r}")
    if safe["model"] not in KNOWN_MODELS:
        raise _RejectRecord(f"unknown model {safe['model']!r} (expected one of {sorted(KNOWN_MODELS)})")
    if safe["backend"] not in KNOWN_BACKENDS:
        raise _RejectRecord(f"unknown backend {safe['backend']!r}")
    if not isinstance(safe["device"], dict) or not safe["device"].get("name"):
        raise _RejectRecord("device.name missing")
    klass = safe["device"].get("class")
    if klass not in KNOWN_DEVICE_CLASSES:
        raise _RejectRecord(f"unknown device.class {klass!r}")

    # Pull out energy_per_inference_mj — the dataclass doesn't have a field for
    # it; keep it stashed in 'notes' so we don't lose the value but also don't
    # break round-trip validation.
    energy = safe.pop("energy_per_inference_mj", None)
    if energy is not None:
        existing_notes = safe.get("notes", "") or ""
        safe["notes"] = (existing_notes + f" [energy_per_inference_mj={energy:.4f}]").strip()

    # Always re-stamp script + timestamp so the merge attribution is correct.
    safe["script"] = SCRIPT_NAME
    safe.setdefault(
        "timestamp",
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    return safe


def _append_caveat(message: str) -> None:
    CAVEATS_MD.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with CAVEATS_MD.open("a", encoding="utf-8") as f:
        f.write(f"- {ts} [{SCRIPT_NAME}] {message}\n")


# --------------------------------------------------------------------------- #
# smoke                                                                       #
# --------------------------------------------------------------------------- #
def mode_smoke(args: argparse.Namespace) -> int:
    logger = _setup_logger("smoke")
    sample = {
        "schema_version": "11.0",
        "records": [
            {
                "model": "efficientnet_lite0",
                "modality": "vision",
                "variant": "tflite_int8",
                "backend": "tflite",
                "compute_units": "nnapi",
                "device": {
                    "name": "Pixel 9",
                    "soc": "Tensor G4",
                    "os": "Android 15",
                    "class": "phone",
                },
                "size_bytes": 5259920,
                "latency_ms": {
                    "p50": 9.1, "p95": 10.2, "mean": 9.4,
                    "n_iters": 200, "warmup_iters": 50,
                    "input_shape": [1, 224, 224, 3],
                },
                "throughput": {"samples_per_sec": 106.0,
                               "tokens_per_sec": None,
                               "prompt_length": None,
                               "generation_length": None},
                "accuracy": {"metric": "top1", "value": 0.66,
                             "secondary": {"top5": 0.88},
                             "dataset": "imagenet-1k-val(100)", "n_samples": 100},
            },
        ],
    }

    failures = 0
    for idx, rec in enumerate(sample["records"]):
        try:
            sanitized = _sanitize_record(rec, logger, idx)
        except _RejectRecord as exc:
            logger.error("smoke: record %d rejected: %s", idx, exc)
            failures += 1
            continue
        try:
            apple.ResultRecord(**sanitized)
        except TypeError as exc:
            logger.error("smoke: record %d failed dataclass: %s", idx, exc)
            failures += 1
    if failures:
        logger.error("smoke FAILED (%d/%d records bad)", failures, len(sample["records"]))
        return 1
    logger.info("smoke OK (%d records validated)", len(sample["records"]))
    return 0


# --------------------------------------------------------------------------- #
# figures                                                                     #
# --------------------------------------------------------------------------- #
def mode_figures(args: argparse.Namespace) -> int:
    logger = _setup_logger("figures")
    # Defer to ch11_1_aggregate which is the canonical figure renderer for
    # the vision-only design-space figures; then run ch11_3_apple figures for
    # the cross-platform Apple/Android comparisons; finally run the
    # Android-only delegate-portability matrix.
    chain = [
        [sys.executable, str(HERE / "ch11_1_aggregate.py"), "all"],
        [sys.executable, str(HERE / "ch11_3_apple.py"), "figures"],
        [sys.executable, str(HERE / "ch11_4_figures.py"), "figures"],
    ]
    rc = 0
    for cmd in chain:
        if not Path(cmd[1]).is_file():
            logger.warning("missing generator: %s — skipping", cmd[1])
            continue
        logger.info("invoking: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(HERE), check=False)
        rc = rc or result.returncode
    return rc


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog=SCRIPT_NAME, description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    p_unpack = sub.add_parser("unpack-artifacts",
                              help="extract AWS Customer Artifacts ZIP")
    p_unpack.add_argument("--zip", required=True, help="path to Customer Artifacts .zip")
    p_unpack.add_argument("--out", default=None, help="staging dir name (default: zip stem)")
    p_unpack.set_defaults(func=mode_unpack_artifacts)

    p_ing = sub.add_parser("ingest-apk-results",
                           help="merge results-android.json into results.json")
    p_ing.add_argument("--input", required=True, help="path to results-android.json")
    p_ing.set_defaults(func=mode_ingest)

    p_smoke = sub.add_parser("smoke", help="schema validation only")
    p_smoke.set_defaults(func=mode_smoke)

    p_fig = sub.add_parser("figures", help="regenerate figures via ch11_1_aggregate")
    p_fig.set_defaults(func=mode_figures)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
