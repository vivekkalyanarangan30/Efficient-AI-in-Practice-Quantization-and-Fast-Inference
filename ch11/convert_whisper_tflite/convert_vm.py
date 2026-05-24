"""Convert Whisper-tiny encoder (HF/openai) -> TFLite for Android.

Mirrors the Apple side's Whisper-tiny encoder benchmark, which records
encoder-only forward-pass latency (no decoder, no full ASR). Produces two
TFLite files for the Android benchmark to sweep:

  models/tflite/whisper_tiny_encoder_fp32.tflite        (~38 MB)
  models/tflite/whisper_tiny_encoder_dynrange.tflite    (~11 MB, dynamic-range int8)

The chapter's Apple side picks {fp16, int8_weight_only, int8_linear,
palettize_4bit, palettize_6bit}; on Android we pick the TFLite-canonical
{fp32, dynrange} pair so AudioBenchmark.kt can sweep them through XNNPACK
/ NNAPI / GPU and report the same axes 11.2 already uses for vision.

Run on the same GCP x86_64 VM that successfully converted Llama
(litert-torch + ai-edge-tensorflow stack already installed). Output is
small (~50 MB total), so push it back to the Mac with scp.

Usage on the VM:
  cd ~/ch11_convert  (or wherever you keep the conversion workdir)
  python convert_vm.py --hf-token <token>   # downloads checkpoint
  # outputs land in ./out/

Then on the Mac:
  scp <vm>:~/ch11_convert/out/whisper_tiny_encoder_*.tflite \\
      ~/Downloads/ch11/models/tflite/

This script intentionally lives OUTSIDE the litert-torch generative
examples path because Whisper's encoder is plain PyTorch — no LLM
decoder, no KV cache, no special chat templating. We use the
direct `ai_edge_torch.convert()` API instead.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def _stage(n: int, name: str) -> float:
    print(f"\nSTAGE {n}: {name}", flush=True)
    return time.time()


def _stage_ok(n: int, t0: float) -> None:
    print(f"STAGE {n} OK ({time.time() - t0:.1f}s)", flush=True)


def _imports():
    """Import precedence matters: TF first to avoid XLA load-order deadlocks."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf   # noqa: F401
    import torch
    import ai_edge_torch
    print(f"  tf={tf.__version__}  torch={torch.__version__}  ai_edge_torch={ai_edge_torch.__version__}",
          flush=True)
    return tf, torch, ai_edge_torch


def _load_whisper_encoder(hf_token: str | None):
    """Load `openai/whisper-tiny` from Hugging Face and extract the encoder.

    The HF Whisper checkpoint contains both encoder and decoder; we only
    need the encoder for the chapter's measurement. We wrap the encoder
    in a tiny module that exposes a single positional forward(features) ->
    last_hidden_state so the converter's tracer doesn't trip on optional
    kwargs that newer HF Whisper introduces.
    """
    from transformers import WhisperFeatureExtractor, WhisperModel
    import torch
    import torch.nn as nn

    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token

    # openai/whisper-tiny is the public checkpoint — same one used by the
    # Apple side's `openai_whisper.load_model("tiny")`. Standard input is
    # 80 mel bins x 3000 frames = 30 s audio at 100 Hz frame rate.
    full = WhisperModel.from_pretrained("openai/whisper-tiny")
    encoder = full.encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Wrap so the exported graph has a single tensor input and a single
    # tensor output — keeps the TFLite signature trivial for the Android
    # interpreter and matches the [1, 80, 3000] -> [1, 1500, 384] shape
    # the Apple records use.
    class EncoderWrap(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, mel):
            out = self.enc(mel)
            return out.last_hidden_state

    return EncoderWrap(encoder).eval()


def _convert(model, ai_edge_torch, out_dir: Path, quantize: bool):
    """Run ai_edge_torch.convert() and save the .tflite file.

    Two passes:
      - fp32: direct convert, no quantization
      - dynrange: PT2E dynamic-range int8 (matches the TFLite "Default
        optimizations" path so it goes through the same fused dispatch
        that EfficientNet-Lite0 uses on the same hardware)
    """
    import torch

    sample_mel = torch.zeros((1, 80, 3000), dtype=torch.float32)
    suffix = "dynrange" if quantize else "fp32"
    out_name = f"whisper_tiny_encoder_{suffix}.tflite"
    out_path = out_dir / out_name

    if quantize:
        # PT2E dynamic-range int8 weights, fp32 activations. Closest TFLite
        # analog to Apple's `int8_weight_only`.
        from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer
        from ai_edge_torch.quantize.pt2e_quantizer_utils import get_symmetric_quantization_config
        from ai_edge_torch.quantize.quant_config import QuantConfig

        qc = PT2EQuantizer().set_global(
            get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
        )
        quant_config = QuantConfig(pt2e_quantizer=qc)
        edge = ai_edge_torch.convert(model, (sample_mel,), quant_config=quant_config)
    else:
        edge = ai_edge_torch.convert(model, (sample_mel,))

    edge.export(str(out_path))
    sz_mb = out_path.stat().st_size / 1e6
    print(f"  wrote {out_path.name}  ({sz_mb:.1f} MB)", flush=True)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN"),
                   help="HF token (read scope). openai/whisper-tiny is public — "
                        "this is only needed if your HF cache requires auth.")
    p.add_argument("--out", default="out",
                   help="output directory (default: ./out)")
    p.add_argument("--variants", default="fp32,dynrange",
                   help="comma-separated variants to produce. Available: "
                        "fp32, dynrange. Default: both.")
    args = p.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}", flush=True)

    t = _stage(1, "import stack")
    _tf, _torch, ai_edge_torch = _imports()
    _stage_ok(1, t)

    t = _stage(2, "load Whisper-tiny encoder")
    model = _load_whisper_encoder(args.hf_token)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  encoder params: {n_params/1e6:.1f} M", flush=True)
    _stage_ok(2, t)

    requested = {v.strip() for v in args.variants.split(",") if v.strip()}
    valid = {"fp32", "dynrange"}
    bad = requested - valid
    if bad:
        raise SystemExit(f"unknown variants: {sorted(bad)} (valid: {sorted(valid)})")

    if "fp32" in requested:
        t = _stage(3, "convert FP32")
        _convert(model, ai_edge_torch, out_dir, quantize=False)
        _stage_ok(3, t)

    if "dynrange" in requested:
        t = _stage(4, "convert dynamic-range INT8 (weights)")
        _convert(model, ai_edge_torch, out_dir, quantize=True)
        _stage_ok(4, t)

    print(f"\nCONVERSION COMPLETE. Output in {out_dir}", flush=True)
    print(f"\nNext (on the Mac):", flush=True)
    print(f"  scp <vm>:{out_dir}/whisper_tiny_encoder_*.tflite \\", flush=True)
    print(f"      ~/Downloads/ch11/models/tflite/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
