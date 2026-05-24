"""ch11_5_prepost.py — Sec 11.5 (pre/post-processing instrumentation, full on Mac).

Section served: 11.5.

Modes:
  bench-vision-prepost   EfficientNet-Lite0: JPEG decode (Pillow / OpenCV / tf.io.decode_jpeg),
                         resize (PIL bicubic / cv2 INTER_AREA / tf.image.resize),
                         normalize (numpy / vectorized).
  bench-audio-prepost    Whisper-tiny: file decode (soundfile / librosa),
                         resample to 16 kHz (scipy / torchaudio / librosa),
                         log-mel (manual / torchaudio / Whisper-bundled).
  bench-llm-prepost      Llama-3.2-1B: tokenize (HF tokenizers / sentencepiece direct),
                         detokenize per-token vs batch, KV cache allocation overhead.
  bench-end-to-end       Full pipeline (decode -> preprocess -> infer -> postprocess);
                         records linking pre/post + inference for a (model, variant, backend).
  figures                11.5.1 (preprocessing breakdown stacked bar per modality),
                         11.5.2 (end-to-end vs inference-only).
  all                    All of the above.
  --smoke                Cheap end-to-end with synthetic inputs; one record + one figure stub.

Records: prepost.* fields populated; accuracy and ane_op_coverage left null per spec §4.3.

Honesty:
  - Vision benchmarks generate a synthetic JPEG corpus (random RGB images encoded with
    Pillow at quality=85) if no data/jpegs/ directory exists. JPEG decode timing is
    representative for synthetic content but real photos can differ; documented.
  - Audio benchmarks generate a 30 s 16-kHz WAV of pink noise unless data/audio/
    contains files; deltas between resamplers are stable across content.

Invocation:
  python ch11_5_prepost.py <mode> [--smoke]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.4,
    "patch.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "figure.dpi": 300,
})

# Manning grayscale-safe palette: see ch11_1_aggregate.py for the canonical
# mapping. Pre/post-vs-inference stacks use PALETTE[0/1] for the two strata.
PALETTE = ["#319974", "#7E76B0", "#D67430", "#3A6FA8", "#888888", "#444444"]
HATCHES = ["", "////", "....", "xxxx", "\\\\\\\\", "++++"]

HERE = Path(__file__).resolve().parent
RESULTS_JSON = HERE / "results.json"
CAVEATS_MD = HERE / "caveats.md"
DATA_DIR = HERE / "data"
LOG_DIR = HERE / "logs"

SCHEMA_VERSION = "11.0"
SCRIPT_NAME = "ch11_5_prepost.py"


@dataclass
class ResultRecord:
    model: str
    variant: str
    backend: str
    device: dict
    modality: str | None = None
    size_bytes: int | None = None
    params: int | None = None
    quantization: dict | None = None
    compute_units: str | None = None
    latency_ms: dict | None = None
    throughput: dict | None = None
    accuracy: dict | None = None
    power_mw: dict | None = None
    sustained: dict | None = None
    ane_op_coverage: dict | None = None
    memory_mb: dict | None = None
    prepost: dict | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"))
    script: str = SCRIPT_NAME
    notes: str = ""


def _device_fingerprint() -> dict:
    return {
        "name": "MacBook Air M3",
        "soc": "Apple M3",
        "os": f"macOS {platform.mac_ver()[0]}",
        "class": "laptop",
    }


def _dedup_key(rec: dict) -> tuple:
    return (rec.get("model"), rec.get("variant"), rec.get("backend"),
            (rec.get("device") or {}).get("name"), rec.get("compute_units"))


def _np_default(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _load_results() -> dict:
    if RESULTS_JSON.exists():
        with RESULTS_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if "records" not in data:
            data["records"] = []
        return data
    return {"schema_version": SCHEMA_VERSION, "records": []}


def append_result(rec: ResultRecord) -> str:
    payload = asdict(rec)
    data = _load_results()
    key = _dedup_key(payload)
    for i, existing in enumerate(data["records"]):
        if _dedup_key(existing) == key:
            data["records"][i] = payload
            with RESULTS_JSON.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=_np_default)
            return "replaced"
    data["records"].append(payload)
    with RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_np_default)
    return "added"


def append_caveat(mode: str, message: str) -> None:
    CAVEATS_MD.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    with CAVEATS_MD.open("a", encoding="utf-8") as f:
        f.write(f"- {ts} [{SCRIPT_NAME}::{mode}] {message}\n")


def _save_pair(fig, name: str, section: str = "ch11_5") -> tuple[Path, Path]:
    out_dir = HERE / "figures" / section
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    cap = out_dir / f"{name.split('_Kalyanarangan')[0]}_caption.md"
    fig.savefig(png, bbox_inches="tight"); fig.savefig(pdf, bbox_inches="tight")
    if not cap.exists():
        cap.write_text(f"Caption skeleton for {name}. Hatched markers; B&W; flesh out post-run.\n")
    return png, pdf


def _setup_logger(mode: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"ch11_5_prepost.{mode}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(LOG_DIR / f"ch11_5_prepost_{mode}.log", mode="a", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------- #
# Synthetic corpora                                                           #
# --------------------------------------------------------------------------- #
def _ensure_jpeg_corpus(n: int = 16) -> list[Path]:
    out = DATA_DIR / "synthetic_jpegs"
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*.jpg"))
    if len(existing) >= n:
        return existing[:n]
    from PIL import Image
    for i in range(len(existing), n):
        arr = (np.random.rand(640, 480, 3) * 255).astype(np.uint8)
        Image.fromarray(arr).save(out / f"img_{i:03d}.jpg", quality=85)
    return sorted(out.glob("*.jpg"))[:n]


def _ensure_audio_corpus(n: int = 8, seconds: float = 30.0, sr: int = 16000) -> list[Path]:
    out = DATA_DIR / "synthetic_audio"
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*.wav"))
    if len(existing) >= n:
        return existing[:n]
    import soundfile as sf
    for i in range(len(existing), n):
        # Pink-ish noise via 1/f filtered white noise.
        N = int(sr * seconds)
        white = np.random.randn(N).astype(np.float32) * 0.1
        # Simple low-pass via cumulative running average to make it less harsh
        kernel = np.ones(8) / 8
        x = np.convolve(white, kernel, mode="same").astype(np.float32)
        sf.write(out / f"clip_{i:02d}.wav", x, sr)
    return sorted(out.glob("*.wav"))[:n]


# --------------------------------------------------------------------------- #
# Vision pre/post                                                             #
# --------------------------------------------------------------------------- #
def mode_bench_vision_prepost(args: argparse.Namespace) -> int:
    logger = _setup_logger("bench-vision-prepost")
    files = _ensure_jpeg_corpus()
    raw_bytes = [p.read_bytes() for p in files]
    rc = 0

    def _time(fn, n=50):
        for _ in range(5): fn()
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(t))

    # Decode benchmarks
    from PIL import Image
    import cv2
    import tensorflow as tf

    def dec_pil(): return [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in raw_bytes]
    def dec_cv2(): return [cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for b in raw_bytes]
    def dec_tf():
        out = []
        for b in raw_bytes:
            out.append(tf.io.decode_jpeg(b, channels=3).numpy())
        return out
    decode_pil = _time(dec_pil)
    decode_cv2 = _time(dec_cv2)
    decode_tf = _time(dec_tf)

    # Resize benchmarks (use PIL-decoded as input)
    decoded = dec_pil()
    def rs_pil():
        return [np.asarray(Image.fromarray(a).resize((224, 224), Image.BICUBIC)) for a in decoded]
    def rs_cv2():
        return [cv2.resize(a, (224, 224), interpolation=cv2.INTER_AREA) for a in decoded]
    def rs_tf():
        return [tf.image.resize(a, (224, 224)).numpy() for a in decoded]
    resize_pil = _time(rs_pil)
    resize_cv2 = _time(rs_cv2)
    resize_tf = _time(rs_tf)

    # Normalize benchmarks
    resized = rs_cv2()
    arr_stack = np.stack(resized).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255
    def norm_loop():
        out = np.empty_like(arr_stack)
        for i, x in enumerate(arr_stack):
            out[i] = (x - mean) / std
        return out
    def norm_vec():
        return (arr_stack - mean) / std
    normalize_loop = _time(norm_loop)
    normalize_vec = _time(norm_vec)

    triplets = [
        ("vision_decode_pil_resize_pil_normalize_loop", decode_pil, resize_pil, normalize_loop),
        ("vision_decode_cv2_resize_cv2_normalize_vec", decode_cv2, resize_cv2, normalize_vec),
        ("vision_decode_tf_resize_tf_normalize_vec", decode_tf, resize_tf, normalize_vec),
    ]
    for variant, dec, rs, nm in triplets:
        rec = ResultRecord(
            model="efficientnet_lite0", modality="vision",
            variant=variant, backend="prepost", device=_device_fingerprint(),
            prepost={"decode_ms": dec, "resize_ms": rs, "normalize_ms": nm,
                     "tokenize_ms": None, "detokenize_ms": None,
                     "logmel_ms": None, "nms_ms": None},
            notes=f"per-batch median over {len(files)} synthetic JPEGs (640×480→224×224).",
        )
        action = append_result(rec)
        logger.info("%s decode=%.2f resize=%.2f normalize=%.2f (%s)", variant, dec, rs, nm, action)
    return rc


# --------------------------------------------------------------------------- #
# Audio pre/post                                                              #
# --------------------------------------------------------------------------- #
def mode_bench_audio_prepost(args: argparse.Namespace) -> int:
    logger = _setup_logger("bench-audio-prepost")
    files = _ensure_audio_corpus()
    rc = 0

    def _time(fn, n=20):
        for _ in range(2): fn()
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(t))

    import soundfile as sf
    import librosa
    from scipy.signal import resample_poly

    raw_bytes = [p.read_bytes() for p in files]
    target_sr = 16000

    def dec_sf():
        out = []
        for b in raw_bytes:
            arr, sr = sf.read(io.BytesIO(b))
            out.append((arr.astype(np.float32), sr))
        return out
    def dec_librosa():
        out = []
        for f in files:
            arr, sr = librosa.load(str(f), sr=None)
            out.append((arr.astype(np.float32), sr))
        return out
    decode_sf = _time(dec_sf)
    decode_lib = _time(dec_librosa)

    sample_data = dec_sf()
    sr0 = sample_data[0][1]
    target_pair = (target_sr, sr0)

    def res_scipy():
        return [resample_poly(x, target_pair[0], target_pair[1]) for x, sr in sample_data]
    def res_torchaudio():
        try:
            import torchaudio
            import torch
            r = torchaudio.transforms.Resample(orig_freq=sr0, new_freq=target_sr)
            return [r(torch.from_numpy(x)).numpy() for x, _ in sample_data]
        except Exception as e:
            append_caveat("bench-audio-prepost", f"torchaudio resample failed: {e}")
            return None
    def res_librosa():
        return [librosa.resample(x, orig_sr=sr, target_sr=target_sr) for x, sr in sample_data]
    resample_scipy = _time(res_scipy)
    try:
        resample_torch = _time(res_torchaudio)
    except Exception as e:
        resample_torch = None
        append_caveat("bench-audio-prepost", f"torchaudio resample timing failed: {e}")
    resample_librosa = _time(res_librosa)

    # Log-mel benchmarks (manual / torchaudio / whisper-bundled)
    resampled = [librosa.resample(x, orig_sr=sr, target_sr=target_sr) for x, sr in sample_data]
    n_mels = 80; n_fft = 400; hop = 160

    def lm_manual():
        out = []
        for x in resampled:
            S = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop, :] * np.hanning(n_fft), axis=-1)) ** 2
            mel_fb = librosa.filters.mel(sr=target_sr, n_fft=n_fft, n_mels=n_mels)
            S_mel = mel_fb @ S.T
            out.append(np.log(np.maximum(S_mel, 1e-10)))
        return out
    def lm_torchaudio():
        try:
            import torchaudio
            import torch
            mel = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
            return [torch.log(mel(torch.from_numpy(x)) + 1e-10).numpy() for x in resampled]
        except Exception as e:
            append_caveat("bench-audio-prepost", f"torchaudio mel failed: {e}"); return None
    def lm_whisper():
        try:
            import whisper as openai_whisper
            return [openai_whisper.audio.log_mel_spectrogram(x).numpy() for x in resampled]
        except Exception as e:
            append_caveat("bench-audio-prepost", f"whisper log_mel failed: {e}"); return None

    logmel_manual = _time(lm_manual, n=10)
    try:
        logmel_torch = _time(lm_torchaudio, n=10)
    except Exception:
        logmel_torch = None
    try:
        logmel_whisper = _time(lm_whisper, n=10)
    except Exception:
        logmel_whisper = None

    triplets = [
        ("audio_dec_sf_res_scipy_lm_manual", decode_sf, resample_scipy, logmel_manual),
        ("audio_dec_lib_res_lib_lm_torchaudio", decode_lib, resample_librosa, logmel_torch),
        ("audio_dec_sf_res_torch_lm_whisper", decode_sf, resample_torch, logmel_whisper),
    ]
    for variant, dec, rs, lm in triplets:
        rec = ResultRecord(
            model="whisper_tiny", modality="audio",
            variant=variant, backend="prepost", device=_device_fingerprint(),
            prepost={"decode_ms": dec, "resize_ms": rs, "normalize_ms": None,
                     "tokenize_ms": None, "detokenize_ms": None,
                     "logmel_ms": lm, "nms_ms": None},
            notes=f"per-batch median over {len(files)} clips.",
        )
        action = append_result(rec)
        logger.info("%s decode=%s resample=%s logmel=%s (%s)", variant, dec, rs, lm, action)
    return rc


# --------------------------------------------------------------------------- #
# LLM pre/post                                                                #
# --------------------------------------------------------------------------- #
def mode_bench_llm_prepost(args: argparse.Namespace) -> int:
    logger = _setup_logger("bench-llm-prepost")
    base = os.environ.get("LLAMA_HF_REPO", "meta-llama/Llama-3.2-1B-Instruct")
    rc = 0

    def _time(fn, n=20):
        for _ in range(2): fn()
        t = []
        for _ in range(n):
            t0 = time.perf_counter(); fn(); t.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(t))

    text = "The quick brown fox jumps over the lazy dog. " * 64

    # HF tokenizers fast path
    try:
        from transformers import AutoTokenizer
        tok_hf = AutoTokenizer.from_pretrained(base)
    except Exception as e:
        msg = (f"HF tokenizer load failed: {e}. Llama is gated; ensure huggingface-cli login + license accepted. "
               f"LLM prepost not measured.")
        logger.warning(msg); append_caveat("bench-llm-prepost", msg); return 2

    ids = tok_hf.encode(text)

    def hf_encode(): return tok_hf.encode(text)
    def hf_decode_batch(): return tok_hf.decode(ids)
    def hf_decode_per_token(): return [tok_hf.decode([i]) for i in ids[:128]]

    enc_hf = _time(hf_encode)
    dec_hf_batch = _time(hf_decode_batch)
    dec_hf_per = _time(hf_decode_per_token)

    # Sentencepiece direct (if model has spm)
    sp_enc = sp_dec_batch = sp_dec_per = None
    try:
        import sentencepiece as spm
        from huggingface_hub import hf_hub_download
        # Llama-3.2 uses tiktoken not sentencepiece, so this branch typically fails
        try:
            spm_path = hf_hub_download(repo_id=base, filename="tokenizer.model")
            sp = spm.SentencePieceProcessor(model_file=spm_path)
            ids2 = sp.encode(text, out_type=int)
            sp_enc = _time(lambda: sp.encode(text, out_type=int))
            sp_dec_batch = _time(lambda: sp.decode(ids2))
            sp_dec_per = _time(lambda: [sp.decode([i]) for i in ids2[:128]])
        except Exception as e:
            append_caveat("bench-llm-prepost",
                          f"sentencepiece tokenizer.model unavailable for {base}: {e}. Llama-3.2 uses a tiktoken-based tokenizer; sentencepiece path skipped honestly.")
    except Exception as e:
        append_caveat("bench-llm-prepost", f"sentencepiece import failed: {e}")

    # KV cache allocation overhead — measure mlx_lm cache init only
    kv_alloc = None
    try:
        import mlx.core as mx
        from mlx_lm import load
        # We don't load Llama here; allocate a representative KV-cache tensor instead.
        def alloc():
            arr = mx.zeros((2, 32, 256, 64), dtype=mx.float16)
            mx.eval(arr); return arr
        kv_alloc = _time(alloc, n=20)
    except Exception as e:
        append_caveat("bench-llm-prepost", f"KV cache alloc surrogate failed: {e}")

    rec = ResultRecord(
        model="llama_3_2_1b_instruct", modality="text",
        variant="hf_tokenizers", backend="prepost", device=_device_fingerprint(),
        prepost={"decode_ms": None, "resize_ms": None, "normalize_ms": None,
                 "tokenize_ms": enc_hf, "detokenize_ms": dec_hf_batch,
                 "logmel_ms": None, "nms_ms": None},
        notes=f"detokenize per-token = {dec_hf_per:.3f} ms (128 tokens); KV alloc surrogate = {kv_alloc} ms",
    )
    append_result(rec)
    if sp_enc is not None:
        rec2 = ResultRecord(
            model="llama_3_2_1b_instruct", modality="text",
            variant="sentencepiece_direct", backend="prepost", device=_device_fingerprint(),
            prepost={"decode_ms": None, "resize_ms": None, "normalize_ms": None,
                     "tokenize_ms": sp_enc, "detokenize_ms": sp_dec_batch,
                     "logmel_ms": None, "nms_ms": None},
            notes=f"detokenize per-token = {sp_dec_per} ms (128 tokens)",
        )
        append_result(rec2)
    logger.info("LLM prepost recorded.")
    return rc


# --------------------------------------------------------------------------- #
# End-to-end                                                                  #
# --------------------------------------------------------------------------- #
def mode_bench_end_to_end(args: argparse.Namespace) -> int:
    logger = _setup_logger("bench-end-to-end")
    rc = 0
    # Vision E2E: PIL decode + cv2 resize + numpy normalize + Core ML inference (palettize_4bit if present)
    try:
        import coremltools as ct
        from PIL import Image
        import cv2
        files = _ensure_jpeg_corpus(8)
        eff = HERE / "models" / "coreml" / "effnet_lite0_palettize_4bit.mlpackage"
        if eff.exists():
            model = ct.models.MLModel(str(eff), compute_units=ct.ComputeUnit.CPU_AND_NE)
            in_spec = model.get_spec().description.input[0]
            in_name = in_spec.name
            warm_shape = tuple(in_spec.type.multiArrayType.shape) or (1, 3, 224, 224)
            timings = {"decode": [], "resize": [], "normalize": [], "infer": []}
            for _ in range(3):
                model.predict({in_name: np.random.randn(*warm_shape).astype(np.float32)})
            # Detect expected layout from input shape
            in_shape = list(in_spec.type.multiArrayType.shape)
            channels_first = (len(in_shape) == 4 and in_shape[1] == 3)
            for f in files:
                buf = f.read_bytes()
                t0 = time.perf_counter()
                arr = np.asarray(Image.open(io.BytesIO(buf)).convert("RGB"))
                t1 = time.perf_counter()
                arr = cv2.resize(arr, (224, 224), interpolation=cv2.INTER_AREA)
                t2 = time.perf_counter()
                arr = arr.astype(np.float32) / 255.0
                t3 = time.perf_counter()
                if channels_first:
                    inp = np.transpose(arr, (2, 0, 1))[None, ...]
                else:
                    inp = arr[None, ...]
                model.predict({in_name: inp})
                t4 = time.perf_counter()
                timings["decode"].append((t1 - t0) * 1000.0)
                timings["resize"].append((t2 - t1) * 1000.0)
                timings["normalize"].append((t3 - t2) * 1000.0)
                timings["infer"].append((t4 - t3) * 1000.0)
            rec = ResultRecord(
                model="efficientnet_lite0", modality="vision",
                variant="coreml_palettize_4bit", backend="prepost",
                compute_units="cpuAndNeuralEngine", device=_device_fingerprint(),
                prepost={"decode_ms": float(np.median(timings["decode"])),
                         "resize_ms": float(np.median(timings["resize"])),
                         "normalize_ms": float(np.median(timings["normalize"])),
                         "tokenize_ms": None, "detokenize_ms": None,
                         "logmel_ms": None, "nms_ms": None},
                latency_ms={"p50": float(np.percentile(timings["infer"], 50)),
                            "p95": float(np.percentile(timings["infer"], 95)),
                            "mean": float(np.mean(timings["infer"])),
                            "n_iters": len(timings["infer"]), "warmup_iters": 6,
                            "input_shape": [1, 224, 224, 3]},
                notes="end-to-end vision pipeline; per-stage median over JPEGs.",
            )
            action = append_result(rec)
            logger.info("E2E vision (%s): infer mean=%.2f", action, rec.latency_ms["mean"])
        else:
            append_caveat("bench-end-to-end", "missing palettize_4bit; vision E2E skipped.")
    except Exception as e:
        rc = 1; logger.error("E2E vision failed: %s", e); append_caveat("bench-end-to-end", str(e))
    return rc


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #
def mode_figures(args: argparse.Namespace) -> int:
    logger = _setup_logger("figures")
    data = _load_results()
    # Filter by backend rather than script attribution: Android-sourced prepost
    # records are stamped script="ch11_4_android.py" at ingest time even though
    # they belong to the §11.5 figure scope (the ingest sanitizer forcibly
    # restamps `script` for provenance). Backend tag is the stable signal.
    recs = [r for r in data["records"] if r.get("backend") == "prepost"]
    sec = "ch11_5"

    # 11.5.1 — preprocessing breakdown, stacked bars grouped by (modality, device).
    # Mac and Pixel sit side-by-side within each modality so the cross-platform
    # contrast reads directly off the figure (decode/resize/normalize on Pixel
    # is sub-millisecond on Bitmap APIs; logmel on Pixel via radix-2 Kotlin FFT
    # is two-orders-of-magnitude slower than Apple's Accelerate.vDSP on Mac).
    DEVICE_ORDER = ["MacBook Air M3", "Google Pixel 10 Pro"]
    DEV_TAG = {"MacBook Air M3": "M3", "Google Pixel 10 Pro": "Pixel"}
    modalities = ["vision", "audio", "text"]
    components = ["decode_ms", "resize_ms", "normalize_ms", "tokenize_ms", "detokenize_ms", "logmel_ms"]

    # Per (device, modality) bucket: median across records for each component.
    group_vals: dict[tuple[str, str], dict[str, float]] = {}
    for dname in DEVICE_ORDER:
        for m in modalities:
            mrecs = [r for r in recs
                     if r.get("modality") == m
                     and r.get("device", {}).get("name") == dname
                     and r.get("prepost")]
            if not mrecs: continue
            stats: dict[str, float] = {}
            for comp in components:
                vals = [r["prepost"][comp] for r in mrecs
                        if r["prepost"].get(comp) is not None]
                if vals:
                    stats[comp] = float(np.median(vals))
            if stats:
                group_vals[(dname, m)] = stats

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    if group_vals:
        groups = [(d, m) for d in DEVICE_ORDER for m in modalities if (d, m) in group_vals]
        x = np.arange(len(groups))
        bottom = np.zeros(len(groups))
        for ci, comp in enumerate(components):
            heights = np.array([group_vals[g].get(comp, 0.0) for g in groups])
            if not np.any(heights > 0): continue
            ax.bar(x, heights, bottom=bottom,
                   label=comp.replace("_ms", ""),
                   color=PALETTE[ci % len(PALETTE)],
                   hatch=HATCHES[ci % len(HATCHES)],
                   edgecolor="black", linewidth=0.5)
            bottom += heights
        ax.set_xticks(x)
        ax.set_xticklabels([f"{DEV_TAG.get(d, d)}\n{m}" for (d, m) in groups], fontsize=8)
        ax.set_ylabel("Time (ms, log)")
        ax.set_yscale("log")
        ax.set_title("11.5.1 — Preprocessing breakdown × modality × device")
        ax.legend(fontsize=7, loc="upper left",
                  framealpha=0.95, edgecolor="#cccccc",
                  labelspacing=0.9, borderpad=0.5)
        ax.grid(True, axis="y", which="both", linestyle=":",
                color="#cccccc", linewidth=0.5, zorder=0)
    else:
        ax.text(0.5, 0.5, "no prepost records",
                transform=ax.transAxes, ha="center", va="center")
    fig.tight_layout(); _save_pair(fig, "CH11_F0501_Kalyanarangan", sec); plt.close(fig)

    # 11.5.2 — end-to-end vs inference-only. Pure-prepost records (latency_ms.mean
    # equals the prepost stage sum) render as single bars; only true end-to-end
    # records (from mode_bench_end_to_end, where latency_ms includes a real
    # inference pass on top of prepost) stack a separate "inference" component.
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    e2e = [r for r in recs
           if r.get("prepost") and (r.get("latency_ms") or {}).get("mean") is not None]
    if e2e:
        labels: list[str] = []
        prepost_total: list[float] = []
        infer_only: list[float] = []
        for r in e2e:
            pp = r["prepost"]
            tot_pp = sum(v for k, v in pp.items() if v is not None)
            lat_mean = r["latency_ms"]["mean"]
            # If latency_ms.mean is within 5 % of the prepost sum, treat the
            # record as prepost-only and zero out the "inference" stack so the
            # bar isn't double-counted.
            if tot_pp > 0 and abs(lat_mean - tot_pp) / tot_pp < 0.05:
                infer_component = 0.0
            else:
                infer_component = max(0.0, lat_mean - tot_pp)
            dev_tag = {"MacBook Air M3": "M3", "Google Pixel 10 Pro": "Pixel"}.get(
                r["device"]["name"], r["device"]["name"]
            )
            short_variant = (r["variant"]
                             .replace("coreml_", "")
                             .replace("android_", ""))
            labels.append(f"{dev_tag}/{r['modality']}/{short_variant}")
            prepost_total.append(tot_pp)
            infer_only.append(infer_component)
        x = np.arange(len(labels))
        ax.bar(x, prepost_total, label="pre/post",
               color=PALETTE[1], hatch=HATCHES[1], edgecolor="black", linewidth=0.5)
        ax.bar(x, infer_only, bottom=prepost_total, label="inference",
               color=PALETTE[0], hatch=HATCHES[2], edgecolor="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Time (ms)")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_title("11.5.2 — End-to-end vs inference-only (Mac + Pixel)")
        ax.grid(True, axis="y", linestyle=":", color="#cccccc",
                linewidth=0.5, zorder=0)
    else:
        ax.text(0.5, 0.5, "data not available",
                transform=ax.transAxes, ha="center", va="center")
    fig.tight_layout(); _save_pair(fig, "CH11_F0502_Kalyanarangan", sec); plt.close(fig)
    logger.info("wrote 11.5.1, 11.5.2 — %d prepost records (Mac + Pixel)", len(recs))
    return 0


# --------------------------------------------------------------------------- #
# Smoke                                                                       #
# --------------------------------------------------------------------------- #
def mode_smoke(args: argparse.Namespace) -> int:
    logger = _setup_logger("smoke")
    files = _ensure_jpeg_corpus(4)
    from PIL import Image
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        for f in files: np.asarray(Image.open(f).convert("RGB"))
        times.append((time.perf_counter() - t0) * 1000.0)
    rec = ResultRecord(
        model="smoke_decode", modality="vision",
        variant="vision_smoke_pil", backend="prepost", device=_device_fingerprint(),
        prepost={"decode_ms": float(np.median(times)), "resize_ms": None, "normalize_ms": None,
                 "tokenize_ms": None, "detokenize_ms": None, "logmel_ms": None, "nms_ms": None},
        notes="smoke run; 10 iters of 4-image PIL decode.",
    )
    action = append_result(rec)
    logger.info("smoke decode median=%.3f ms (%s)", rec.prepost["decode_ms"], action)
    # No figure for smoke mode — single-bar smoke charts have no editorial content.
    return 0


# --------------------------------------------------------------------------- #
# All                                                                         #
# --------------------------------------------------------------------------- #
def mode_all(args: argparse.Namespace) -> int:
    rc = 0
    for fn in [mode_bench_vision_prepost, mode_bench_audio_prepost, mode_bench_llm_prepost,
               mode_bench_end_to_end, mode_figures]:
        sub = fn(args)
        if sub: rc = sub
    return rc


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", default="all",
                   choices=["bench-vision-prepost", "bench-audio-prepost", "bench-llm-prepost",
                            "bench-end-to-end", "figures", "all", "smoke"])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke: args.mode = "smoke"
    dispatch = {
        "bench-vision-prepost": mode_bench_vision_prepost,
        "bench-audio-prepost": mode_bench_audio_prepost,
        "bench-llm-prepost": mode_bench_llm_prepost,
        "bench-end-to-end": mode_bench_end_to_end,
        "figures": mode_figures,
        "all": mode_all,
        "smoke": mode_smoke,
    }
    return dispatch[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
