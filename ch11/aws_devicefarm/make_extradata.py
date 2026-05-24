"""Bundle a Llama-3.2-1B `.task` MediaPipe model into an AWS Device Farm
EXTERNAL_DATA archive.

The on-device LLM benchmark (LLMBenchmark.kt) loads any `.task` files it
finds in `/sdcard/Android/data/com.ch11.bench/files/models/`. The Device
Farm testspec.yml's pre_test phase populates that directory by `adb push`-ing
the contents of the EXTERNAL_DATA upload from the host EC2 runner. This
script produces the upload archive.

Usage:
  .venv/bin/python aws_devicefarm/make_extradata.py \\
      --task ~/Downloads/ch11/models/llama_3_2_1b_instruct.task \\
      --out ~/Downloads/ch11/aws_devicefarm/extradata.zip

The script does NOT download the model — Llama is license-gated on Kaggle
(`https://www.kaggle.com/models/google/gemma/...` flow for Llama variants).
Accept the license, download the `.task` bundle, then point this script at
the local file.

Then upload by passing the resulting zip to run.py:
  .venv/bin/python aws_devicefarm/run.py \\
      --project-arn arn:aws:devicefarm:us-west-2:<acct>:project:<uuid> \\
      --device "Google Pixel 10 Pro" \\
      --extra-data ~/Downloads/ch11/aws_devicefarm/extradata.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", action="append", default=[],
                   help="path to a .task file (LLM bundle); may be repeated")
    p.add_argument("--tflite", action="append", default=[],
                   help="path to a .tflite file (e.g. whisper_*.tflite); may "
                        "be repeated. The testspec_llm.yml pre_test phase "
                        "pushes both .task and whisper_*.tflite files.")
    p.add_argument("--out", required=True,
                   help="output zip path (e.g. aws_devicefarm/extradata.zip)")
    args = p.parse_args()

    if not args.task and not args.tflite:
        print("error: must provide at least one --task or --tflite file",
              file=sys.stderr)
        return 1

    out_path = Path(args.out).expanduser().resolve()
    if out_path.exists() and not out_path.is_file():
        print(f"error: --out exists and is not a regular file: {out_path}",
              file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sources: list[Path] = []
    for raw, expected_suffix, flag in (
        [(t, ".task", "--task") for t in args.task]
        + [(t, ".tflite", "--tflite") for t in args.tflite]
    ):
        src = Path(raw).expanduser().resolve()
        if not src.is_file():
            print(f"error: {flag} file not found: {src}", file=sys.stderr)
            return 1
        if src.suffix.lower() != expected_suffix:
            print(f"error: {flag} must point at a {expected_suffix} file: "
                  f"{src.name}", file=sys.stderr)
            return 1
        sources.append(src)

    # Names inside the zip must be flat (the testspec.yml uses `basename` to
    # find them), so we strip any directory prefix and only keep the leaf name.
    # We also guard against name collisions across multiple --task uploads.
    seen: set[str] = set()
    total_bytes = 0
    # ZIP_STORED keeps the .task usable without re-extraction overhead — and
    # the .task internals are already compressed, so DEFLATE would barely save
    # space anyway.
    with zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        for src in sources:
            if src.name in seen:
                print(f"error: duplicate filename in --task list: {src.name}",
                      file=sys.stderr)
                return 1
            seen.add(src.name)
            print(f"  adding {src.name} ({src.stat().st_size / (1024 * 1024):.0f} MiB)")
            zf.write(src, arcname=src.name)
            total_bytes += src.stat().st_size

    out_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"wrote {out_path} ({out_mb:.0f} MiB total)")
    if out_mb > 2048:
        print("warn: archive exceeds Device Farm 2 GiB EXTERNAL_DATA cap; "
              "the upload step in run.py will reject it.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
