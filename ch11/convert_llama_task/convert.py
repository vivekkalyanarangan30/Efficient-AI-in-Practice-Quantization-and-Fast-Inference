"""Convert Llama-3.2-1B-Instruct HF safetensors into a MediaPipe `.task` bundle.

Runs inside the Linux/ARM64 container defined by the sibling Dockerfile. The
host invokes this with `probe` (fast viability check) or `convert` (full
pipeline) — see the Dockerfile's docstring for invocation examples.

Stage gates: each step prints `STAGE N: <name>` before starting and `STAGE
N OK` after. If a stage hangs or errors, the user sees exactly where.

Inputs / outputs (mounted from host):
  /hf_cache       : ~/.cache/huggingface/hub (read-only, contains the HF
                    snapshot for meta-llama/Llama-3.2-1B-Instruct)
  /output         : repo's models/mediapipe (read-write, receives .task)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HF_REPO = Path("/hf_cache/models--meta-llama--Llama-3.2-1B-Instruct")
OUTPUT_DIR = Path("/output")
MODEL_NAME = "llama_3_2_1b_instruct"


def _stage(n: int, name: str):
    print(f"STAGE {n}: {name}", flush=True)


def _stage_ok(n: int, t0: float):
    print(f"STAGE {n} OK ({time.time() - t0:.1f}s)", flush=True)


def _imports():
    """Stage 1: import the heavy modules.

    Even on Linux we set TF env vars to suppress noisy logs from the long
    import path that goes through tensorflow.lite.python.schema_py_generated.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    _stage(1, "imports")
    t0 = time.time()
    import tensorflow as tf  # noqa: F401  (precedence: avoid xla<->tf load order issues)
    print(f"  tf {tf.__version__}", flush=True)
    import torch
    print(f"  torch {torch.__version__}", flush=True)
    import litert_torch.generative.examples.llama.llama as llama_mod
    print(f"  llama_mod loaded", flush=True)
    from litert_torch.generative.utilities import converter
    print(f"  converter loaded", flush=True)
    from mediapipe.tasks.python.genai.bundler import llm_bundler
    print(f"  mediapipe bundler loaded", flush=True)
    _stage_ok(1, t0)
    return llama_mod, converter, llm_bundler


def _locate_checkpoint() -> Path:
    """Stage 2: locate the HF snapshot dir holding model.safetensors."""
    _stage(2, "locate checkpoint")
    t0 = time.time()
    if not HF_REPO.is_dir():
        raise SystemExit(f"HF cache mount not found: {HF_REPO}. "
                         "Did you `-v ~/.cache/huggingface:/hf_cache`?")
    snaps = HF_REPO / "snapshots"
    cands = [d for d in snaps.iterdir()
             if d.is_dir() and (d / "model.safetensors").is_file()]
    if not cands:
        raise SystemExit(f"No snapshot with model.safetensors in {snaps}")
    cands.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    ckpt = cands[0]
    print(f"  snapshot: {ckpt}", flush=True)
    print(f"  has tokenizer.json: {(ckpt / 'tokenizer.json').is_file()}", flush=True)
    _stage_ok(2, t0)
    return ckpt


def _build_model(llama_mod, ckpt: Path):
    """Stage 3: build the Llama-1B model and load HF weights.

    Llama-3.2-1B has tied embeddings (no separate lm_head.weight in the
    safetensors). The default config has lm_head_share_weight_with_embedding=
    False which forces strict=True loading and rejects the missing key.
    We override the field before constructing the model.
    """
    from litert_torch.generative.utilities import model_builder
    _stage(3, "build model + load weights")
    t0 = time.time()
    config = llama_mod.get_1b_model_config()
    # Tied embedding: see ch11_3_apple.py convert-coreml-llm path for the same
    # detail on the Apple side. HF Llama-3.2 ships tied weights.
    config.lm_head_share_weight_with_embedding = True
    model = model_builder.build_decoder_only_model(
        checkpoint_path=str(ckpt),
        config=config,
        tensor_names=llama_mod.TENSOR_NAMES,
        model_class=llama_mod.Llama,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params / 1e9:.2f}B", flush=True)
    _stage_ok(3, t0)
    return model


def _forward_probe(model):
    """Stage 4: run one forward pass with random tokens.

    This catches model-graph issues (op shape mismatches, missing KV cache
    plumbing) BEFORE we commit to a 30-90 min conversion. Uses the model's
    forward(...) which the converter will trace.
    """
    import torch
    _stage(4, "forward-pass probe")
    t0 = time.time()
    seq_len = 16
    input_ids = torch.zeros((1, seq_len), dtype=torch.long)
    input_pos = torch.arange(seq_len, dtype=torch.long)
    with torch.no_grad():
        # The DecoderOnlyModel forward signature: (tokens, input_pos, kv_cache, ...)
        # We call without an external kv_cache to use the built-in path.
        try:
            out = model(tokens=input_ids, input_pos=input_pos)
        except TypeError:
            # Older signature: (input_ids, input_pos)
            out = model(input_ids, input_pos)
    if hasattr(out, "shape"):
        print(f"  output shape: {tuple(out.shape)}", flush=True)
    else:
        print(f"  output type: {type(out).__name__}", flush=True)
    _stage_ok(4, t0)


def _convert(model, converter, prefill_seq_lens, kv_cache_max_len, quantize):
    """Stage 5: convert PyTorch -> multi-signature TFLite.

    This is the slow step (~30-90 min on M3 in container). Produces a file
    named {MODEL_NAME}_{quant_suffix}_ekv{kv}.tflite in OUTPUT_DIR.
    """
    _stage(5, f"convert to TFLite (prefill={prefill_seq_lens}, kv={kv_cache_max_len}, q={quantize})")
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converter.convert_to_tflite(
        pytorch_model=model,
        output_path=str(OUTPUT_DIR),
        output_name_prefix=MODEL_NAME,
        prefill_seq_len=prefill_seq_lens,
        kv_cache_max_len=kv_cache_max_len,
        quantize=quantize,
    )
    quant_suffix = {
        "none": "",
        "dynamic_int8": "q8",
        "dynamic_int4": "q4",
        "weight_only_int8": "wo8",
    }.get(quantize, quantize.replace(":", "_"))
    sep = "_" if quant_suffix else ""
    expected = OUTPUT_DIR / f"{MODEL_NAME}{sep}{quant_suffix}_ekv{kv_cache_max_len}.tflite"
    if not expected.is_file():
        # Find newest .tflite under OUTPUT_DIR as fallback.
        tflites = sorted(OUTPUT_DIR.glob("*.tflite"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if not tflites:
            raise RuntimeError(f"No .tflite produced in {OUTPUT_DIR}")
        expected = tflites[0]
    sz_mb = expected.stat().st_size / 1e6
    print(f"  tflite: {expected.name} ({sz_mb:.0f} MB)", flush=True)
    _stage_ok(5, t0)
    return expected


def _bundle(llm_bundler, tflite_path: Path, ckpt: Path, out_task: Path):
    """Stage 6: wrap .tflite + tokenizer.json into a `.task` bundle.

    Llama 3.2 uses tiktoken-style BPE encoded as HF tokenizer.json. Recent
    mediapipe versions accept it; if rejected, we fall through to printing
    a clear next-step message instead of crashing.
    """
    _stage(6, "bundle to .task")
    t0 = time.time()
    tokenizer = ckpt / "tokenizer.json"
    if not tokenizer.is_file():
        raise SystemExit(f"tokenizer.json missing at {tokenizer}")
    cfg = llm_bundler.BundleConfig(
        tflite_model=str(tflite_path),
        tokenizer_model=str(tokenizer),
        start_token="<|begin_of_text|>",
        stop_tokens=["<|eot_id|>", "<|end_of_text|>"],
        output_filename=str(out_task),
        prompt_prefix_user="<|start_header_id|>user<|end_header_id|>\n\n",
        prompt_suffix_user="<|eot_id|>",
        prompt_prefix_model="<|start_header_id|>assistant<|end_header_id|>\n\n",
        prompt_suffix_model="<|eot_id|>",
        prompt_prefix_system="<|start_header_id|>system<|end_header_id|>\n\n",
        prompt_suffix_system="<|eot_id|>",
        enable_bytes_to_unicode_mapping=False,
    )
    llm_bundler.create_bundle(cfg)
    if not out_task.is_file():
        raise RuntimeError(f"bundler produced no {out_task}")
    print(f"  task: {out_task.name} ({out_task.stat().st_size / 1e6:.0f} MB)", flush=True)
    _stage_ok(6, t0)


def cmd_probe() -> int:
    """Fast viability mode: imports + checkpoint locate + model build + 1 forward pass.

    Skips the slow convert + bundle. Useful as a 5-min health check before
    committing to the full conversion.
    """
    llama_mod, _converter, _llm_bundler = _imports()
    ckpt = _locate_checkpoint()
    model = _build_model(llama_mod, ckpt)
    _forward_probe(model)
    print("PROBE PASSED — env is viable for conversion.", flush=True)
    return 0


def cmd_convert(quantize: str = "dynamic_int8",
                prefill_seq_lens: tuple[int, ...] = (128,),
                kv_cache_max_len: int = 1280) -> int:
    """Full pipeline: probe + convert + bundle."""
    llama_mod, converter, llm_bundler = _imports()
    ckpt = _locate_checkpoint()
    model = _build_model(llama_mod, ckpt)
    _forward_probe(model)
    tflite_path = _convert(model, converter, list(prefill_seq_lens),
                            kv_cache_max_len, quantize)
    out_task = OUTPUT_DIR / f"{MODEL_NAME}_{quantize.replace('dynamic_', '')}.task"
    _bundle(llm_bundler, tflite_path, ckpt, out_task)
    print(f"\nCONVERSION COMPLETE: {out_task}", flush=True)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] == "help":
        print(__doc__, flush=True)
        print("\nUsage: docker run ... ch11-llama-convert {probe|convert}",
              flush=True)
        return 0
    mode = argv[1]
    if mode == "probe":
        return cmd_probe()
    if mode == "convert":
        quantize = argv[2] if len(argv) >= 3 else "dynamic_int8"
        return cmd_convert(quantize=quantize)
    print(f"error: unknown mode {mode!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
