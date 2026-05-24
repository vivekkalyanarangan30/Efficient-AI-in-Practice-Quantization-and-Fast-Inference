# Chapter 11 — Targeting edge and mobile devices

Companion code for the chapter. Reproduces every figure (11.1–11.16) and every measured number in the prose, plus the manual procedures the chapter cites (iPhone Performance Reports via Xcode, AWS Device Farm runs on a paired Pixel 10 Pro).

The chapter measures three reference devices: a MacBook Air M3, an iPhone 16, and a Google Pixel 10 Pro. Three models cover the workload mix: EfficientNet-Lite0 (vision), Whisper-tiny encoder (audio), and Llama-3.2-1B-Instruct (LLM). Every record from a measurement run lands in a single chapter-wide `results.json`; the figure generators read from there and never measure.

## How the repo executes

Three execution tiers, run independently in any order:

| Tier | Host | What runs there |
|------|------|-----------------|
| Mac  | M-series macOS 15+ | TFLite Mac host bench (§11.2), Core ML + MLX + MPS (§11.3), prepost (§11.5), figure generation (§11.1–§11.5). |
| Linux container | Docker / Colab GPU | Llama → MediaPipe `.task` → LiteRT-LM `.litertlm` (host-only conversion; runtime ships separately for Android). Whisper → `.tflite` (TF nightly that does not install cleanly on macOS arm64). |
| Android phone | Pixel 10 Pro / Tensor G5 / Android 16 | TFLite + LiteRT-LM + prepost benchmarks via the `ch11-bench` APK, run on a real device through AWS Device Farm. |
| iPhone (manual) | iPhone 16 paired to the Mac via Xcode 16+ | `.mlperfreport` generation through Xcode's Core ML Performance test. Drop reports back into `reports/` for ingest. |

Outputs from all four tiers append to `results.json`. `ch11_1_aggregate.py` is the single read-only consumer that regenerates the chapter's cross-section figures.

## Quick start (smoke path on Mac, ~10 minutes)

```bash
git clone <repo-url>
cd ch11

# Install the Python environment (deps pinned in requirements.txt)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Smoke runs — each writes a few records into results.json
.venv/bin/python ch11_2_tflite.py smoke
.venv/bin/python ch11_3_apple.py smoke
.venv/bin/python ch11_5_prepost.py smoke

# Regenerate figures from whatever results.json now contains
.venv/bin/python ch11_1_aggregate.py figures
.venv/bin/python ch11_2_tflite.py figures
.venv/bin/python ch11_3_apple.py figures
.venv/bin/python ch11_5_prepost.py figures
```

`figures/CH11_F*_Kalyanarangan.{png,pdf}` are the PDF/PNG pairs the manuscript uses; PNGs are linked from the prose, PDFs are the publication source.

## Prerequisites

### Mac side (required for §11.2 host bench, §11.3, §11.5, all figure generators)

- macOS 15.2 or later on Apple M-series silicon
- Python 3.11 (3.10 also works; older Pythons miss some `coremltools` features)
- Xcode 16+ Command Line Tools (`xcode-select --install`)
- ~30 GB free disk for converted artifacts (`models/` after all conversions)

### Linux container side (required for `.litertlm` bundle and Whisper `.tflite`)

- Docker Desktop on Mac, or a Linux box with NVIDIA GPU (recommended; CPU also works, slower)
- Or: Google Colab with the Llama HF license accepted on the model card

### Android side (required for §11.2 Pixel records, §11.4 delegate matrix, §11.5 Pixel prepost)

- Android Studio Iguana (2023.2) or later, or a JDK 17 + Android SDK command-line tools
- A Pixel 10 Pro (or any Android 16 phone exposing the same delegate set) — directly via USB, or rented through AWS Device Farm

### iPhone side (required for §11.3 iPhone column)

- iPhone 16 (or any iOS 17+ device with Neural Engine) paired to the Mac via cable or wireless
- Xcode 16+ for the Performance Reports flow described in `ch11_3_iphone_steps.md`

## Credential setup

Three auth surfaces are needed. **None should be committed to the repo.** Use the standard env-var and credential-cache locations.

| Surface | Used for | How to authenticate |
|---------|----------|---------------------|
| Hugging Face | Llama-3.2-1B-Instruct download (gated), Whisper-tiny weights | `huggingface-cli login` (writes `~/.cache/huggingface/token`, mode 600) |
| Kaggle | EfficientNet-Lite0 weights from Kaggle Models | `~/.kaggle/kaggle.json` with `chmod 600`; or set `EFFNET_LITE0_KAGGLE=<handle>` and let `kagglehub` resolve |
| AWS Device Farm | Pixel 10 Pro run orchestration | Standard AWS chain: `AWS_PROFILE`, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, or `~/.aws/credentials`. Scope the IAM role to `devicefarm:*` only. |

If you run the smoke path only, none of these are needed — the smoke modes use synthetic data and small bundled fixtures.

## Section-by-section execution

### §11.1 Set latency and power budgets

Pure aggregator; never measures. Run after any of the measurement scripts to refresh the cross-modal figures.

```bash
.venv/bin/python ch11_1_aggregate.py figures
```

Writes `figures/CH11_F01..F04_Kalyanarangan.{png,pdf}`.

### §11.2 Drive Android's LiteRT family — TFLite and LiteRT-LM

```bash
# Convert EfficientNet-Lite0 through the four TFLite paths
.venv/bin/python ch11_2_tflite.py convert

# Inspect the produced flatbuffers (figure 11.6 dtype map)
.venv/bin/python ch11_2_tflite.py inspect

# Verify accuracy on 1,000 ImageNet val images (figure 11.5)
.venv/bin/python ch11_2_tflite.py verify-accuracy

# Host-side latency sweep on the M3 (not in figures, used for §11.2 prose)
.venv/bin/python ch11_2_tflite.py bench-host

# Regenerate figures
.venv/bin/python ch11_2_tflite.py figures
```

Pixel records for §11.2 come from the Android benchmark APK (see *Android side*). The Whisper `.tflite` conversion needs the Linux container — see *Llama and Whisper conversion*.

### §11.3 Apple silicon — Core ML and MLX

```bash
# Convert all five Core ML EfficientNet variants
.venv/bin/python ch11_3_apple.py convert-coreml-vision

# Convert Core ML Whisper-tiny encoder
.venv/bin/python ch11_3_apple.py convert-coreml-whisper

# Convert Core ML Llama (prefill-only, INT4/INT8 per-block; iPhone Performance Reports only)
.venv/bin/python ch11_3_apple.py convert-coreml-llm

# Convert MLX Llama (M3 autoregressive decode path)
.venv/bin/python ch11_3_apple.py convert-mlx-llm

# Inspect each artifact (op coverage, weight metadata)
.venv/bin/python ch11_3_apple.py inspect

# Verify accuracy (ImageNet for vision, WER for Whisper)
.venv/bin/python ch11_3_apple.py verify-accuracy

# Benchmark each runtime on the M3
.venv/bin/python ch11_3_apple.py bench-mac-coreml
.venv/bin/python ch11_3_apple.py bench-mac-mlx
.venv/bin/python ch11_3_apple.py bench-mac-mps
.venv/bin/python ch11_3_apple.py bench-mac-sustained
.venv/bin/python ch11_3_apple.py bench-mac-power

# Ingest iPhone .mlperfreport files dropped into reports/
.venv/bin/python ch11_3_apple.py ingest-iphone-report

# Regenerate figures
.venv/bin/python ch11_3_apple.py figures
```

The iPhone column requires the manual Xcode procedure in **`ch11_3_iphone_steps.md`**. Drop the produced `.mlperfreport` files into `reports/` and rerun `ingest-iphone-report`.

### §11.4 Phone, SBC, and small-box tiers

No new measurements — the section uses the Pixel data from §11.2/§11.4 and discusses SBC/small-box tiers from public spec sheets. The optional delegate-portability heatmap (dropped from final prose, kept as a reference view) is regenerated by:

```bash
.venv/bin/python ch11_4_figures.py figures
```

### §11.5 Pre and post-processing

```bash
# Mac-side prepost benchmarks (vision via OpenCV/PIL/TF + Core ML vImage, audio via librosa,
# LLM via HuggingFace tokenizers and sentencepiece)
.venv/bin/python ch11_5_prepost.py bench

# End-to-end vs inference-only stacked bars
.venv/bin/python ch11_5_prepost.py bench-end-to-end

# Regenerate figures 11.15 and 11.16
.venv/bin/python ch11_5_prepost.py figures
```

Pixel-side prepost records come from the Android APK (`PrepostBenchmark.kt`).

## Llama and Whisper conversion (Linux container path)

Two artifacts cannot be produced on macOS arm64 with stock tooling:

1. **Llama-3.2-1B-Instruct `.litertlm` bundle** for LiteRT-LM on Android. Conversion needs `mediapipe-model-maker`, which only ships Linux wheels with x86_64 + CUDA.
2. **Whisper-tiny encoder `.tflite`** in both `fp32` and `dynrange` variants. The TF nightly build with the encoder's required ops is Linux-only.

Two reproducible paths:

**Docker on a Linux box (recommended):**

```bash
# Authenticate Hugging Face first (the model is license-gated)
huggingface-cli login

cd convert_llama_task
docker build -t ch11-convert .
docker run --rm -it --gpus all \
  -v "$PWD/..":/workspace \
  -e HF_TOKEN \
  ch11-convert convert
```

**Colab GPU runtime:**

See `convert_llama_task/COLAB_INSTRUCTIONS.md` for the cell-by-cell runbook. The end state is the same: a `.litertlm` bundle written under `models/litertlm/`.

For Whisper:

```bash
cd convert_whisper_tflite
# Spin up an Ubuntu x86_64 instance (EC2 c5.4xlarge, GCP n2-standard-16, or equivalent)
# Then on that VM:
python convert_vm.py
# scp the resulting whisper_tiny_encoder_{fp32,dynrange}.tflite files back to your Mac
```

## Android APK — build and deploy

The Android benchmark app drives Pixel-side records for vision, audio, LLM, and prepost. Source lives in `android/`. Assets (model files) are populated separately by `_prepare_android_assets.py` to keep them out of git.

```bash
# 1. Stage the assets the APK packages (requires converted artifacts to exist
#    under models/tflite/ and models/litertlm/ from the conversion steps above)
.venv/bin/python _prepare_android_assets.py

# 2. Build both APKs (app + instrumentation test)
cd android
./gradlew assembleDebug assembleAndroidTest
# Outputs:
#   app/build/outputs/apk/debug/app-debug.apk
#   app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
```

Two execution paths:

**Direct USB run:**

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
adb shell am instrument -w \
  com.ch11.bench.test/androidx.test.runner.AndroidJUnitRunner
adb shell run-as com.ch11.bench cat files/results-android.json > results-android.json
.venv/bin/python ch11_4_android.py ingest --input results-android.json
```

**AWS Device Farm run (recommended for reproducibility):**

```bash
# Package the large assets the APK needs but doesn't bundle directly
.venv/bin/python aws_devicefarm/make_extradata.py

# Launch the run (uploads APK + test APK + testspec + extradata, polls, pulls artifacts)
.venv/bin/python aws_devicefarm/run.py

# Ingest the customer-artifacts ZIPs back into results.json
.venv/bin/python ch11_4_android.py ingest --input aws_test_run/
```

Full Device Farm walkthrough — including the testspec rewrite that fixed the "empty artifacts" failure mode — is in **`aws_devicefarm/README.md`** and **`DEVICEFARM_LLM_HANDOFF.md`**.

## iPhone Performance Reports (manual procedure)

Apple does not expose a programmatic Core ML latency-and-power API on iOS, so iPhone records come from Xcode's Core ML Performance test. The six-step procedure — open the `.mlpackage` in Xcode, configure the Performance tab, run on the paired iPhone, copy the produced `.mlperfreport` file into `reports/` — is documented in **`ch11_3_iphone_steps.md`**.

Once reports are in `reports/`:

```bash
.venv/bin/python ch11_3_apple.py ingest-iphone-report
.venv/bin/python ch11_3_apple.py figures   # iPhone columns now populate
```

## Repository layout

```
ch11/
├── README.md                          # this file
├── requirements.txt                   # pinned Python deps
├── results.json                       # measurement records (read-only consumer: ch11_1_aggregate.py)
├── caveats.md                         # measurement irregularities log (every gotcha the prose alludes to)
│
├── ch11_1_aggregate.py                # §11.1 — figures 11.1–11.4
├── ch11_2_tflite.py                   # §11.2 — TFLite conversion + Mac host bench + figures 11.5–11.7
├── ch11_3_apple.py                    # §11.3 — Core ML + MLX + MPS + iPhone ingest + figures 11.8–11.13
├── ch11_4_android.py                  # §11.2 + §11.4 — Android artifact ingest (TFLite + LiteRT-LM)
├── ch11_4_figures.py                  # §11.4 — optional delegate-portability heatmap
├── ch11_5_prepost.py                  # §11.5 — prepost benchmarks + figures 11.15–11.16
│
├── _build_effnet_lite0_savedmodel.py  # EfficientNet-Lite0 from timm → Keras SavedModel
├── _prepare_data.py                   # ImageNet val + LibriSpeech subset stager
├── _prepare_android_assets.py         # stages assets into android/app/src/main/assets/
├── _convert_llama_task.py             # wrapper around the containerized Llama converter
│
├── convert_llama_task/                # containerized Llama → .litertlm conversion
│   ├── Dockerfile
│   ├── convert.py
│   ├── probe.py
│   └── COLAB_INSTRUCTIONS.md
│
├── convert_whisper_tflite/            # Linux-VM Whisper → .tflite conversion
│   └── convert_vm.py
│
├── android/                           # Android benchmark APK (Kotlin)
│   ├── settings.gradle.kts
│   ├── build.gradle.kts
│   ├── gradle.properties
│   ├── gradlew / gradlew.bat
│   └── app/
│       ├── build.gradle.kts
│       └── src/
│           ├── main/                  # MainActivity + Benchmark / AudioBenchmark / LLMBenchmark / PrepostBenchmark / PowerSampler / ResultsWriter / SampleBundle
│           └── androidTest/           # BenchmarkInstrumentationTest
│
├── aws_devicefarm/                    # Device Farm orchestration
│   ├── README.md
│   ├── run.py
│   ├── make_extradata.py
│   └── testspec_llm.yml
│
├── ch11_3_iphone_steps.md             # Xcode → .mlperfreport manual procedure
└── DEVICEFARM_LLM_HANDOFF.md          # Device Farm LLM run debugging notes
```

## Caveats and known measurement irregularities

Every measurement irregularity the chapter cites is logged in **`caveats.md`** with an ISO timestamp, the script and mode that hit it, and the resolution. Examples: the Mali G715 → PowerVR DXT-48-1536 GPU correction on the Tensor G5; the LiteRT-LM `Backend.CPU()` hint being silently routed to GPU; the iPhone Llama prefill-only conversion path. Read it before extending any measurement script.

## Reproducibility expectations

- Latency numbers will not match the book to the third significant figure. Tensor G5 driver versions, macOS minor releases, and iOS minor releases all move kernels around. Match within ±15% on the same chip generation should be expected; gross routing decisions (which compute-unit, which delegate) should reproduce exactly.
- The accuracy column for INT8 variants depends on the calibration dataset. The chapter shipped `data/calib/` empty and the converter fell back to synthetic uniform-random pixels — see §11.2 prose for why this drops top-1 by 17 percentage points without harming top-5. To reproduce production-grade INT8 accuracy, drop a 200-sample ImageNet val shard into `data/calib/` before running `ch11_2_tflite.py convert`.

## Issues

File issues at the chapter's manuscript repository (link in the book), or open a PR against this companion repo with a script change plus a `caveats.md` entry naming the cause.
