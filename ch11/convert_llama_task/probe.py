"""Standalone import probe — same as `convert.py probe` but without the
weight-load or forward-pass stages. Useful as a 30-second smoke test of
the container build itself, before checking that the host's HF cache is
correctly mounted.

  docker run --rm ch11-llama-convert python /work/probe.py
"""
from __future__ import annotations

import os
import sys
import time


def main() -> int:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    t = time.time(); import tensorflow as tf
    print(f"tf {tf.__version__} in {time.time() - t:.1f}s", flush=True)
    t = time.time(); import torch
    print(f"torch {torch.__version__} in {time.time() - t:.1f}s", flush=True)
    t = time.time(); import litert_torch.generative.examples.llama.llama as _llama
    print(f"litert_torch.llama in {time.time() - t:.1f}s", flush=True)
    t = time.time(); from litert_torch.generative.utilities import converter as _conv  # noqa: F401
    print(f"litert_torch.converter in {time.time() - t:.1f}s", flush=True)
    t = time.time(); from mediapipe.tasks.python.genai.bundler import llm_bundler as _b  # noqa: F401
    print(f"mediapipe bundler in {time.time() - t:.1f}s", flush=True)
    print("PROBE OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
