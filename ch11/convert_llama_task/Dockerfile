# Linux/ARM64 container for converting Llama-3.2-1B-Instruct PyTorch weights
# into a MediaPipe `.task` bundle. Runs the litert-torch generative example
# (build_1b_model + convert_to_tflite) and then the mediapipe genai bundler.
#
# Why a container: macOS host hits a pywrap_tensorflow <-> jaxlib XLA
# dynamic-link deadlock when litert_torch.generative.examples.llama imports.
# Linux uses ELF dynamic linking and is the platform the litert-torch team
# tests on, so we sidestep the issue by running the conversion in a container.
#
# Build (~10-20 min first time, mostly pip wheel downloads):
#   docker build -t ch11-llama-convert convert_llama_task/
#
# Run import probe (~2 min):
#   docker run --rm ch11-llama-convert probe
#
# Run conversion (~30-90 min). Mounts HF cache (input) + models dir (output):
#   docker run --rm \
#     -v "$HOME/.cache/huggingface:/hf_cache:ro" \
#     -v "$PWD/models/mediapipe:/output" \
#     ch11-llama-convert convert

FROM --platform=linux/arm64 python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Pinned versions. torch + transformers + jax align with the macOS host venv
# where the API surface was already inspected. litert-torch 0.9.0 is the
# package that owns the generative example modules. mediapipe ships the
# genai bundler used in step 2.
#
# Wheels: torch 2.11.0 has manylinux2014_aarch64 wheels. mediapipe 0.10.21
# has cp311 aarch64 wheels. litert-torch is pure Python.
RUN pip install --no-cache-dir \
        "torch==2.11.0" \
        "transformers==5.8.0" \
        "safetensors>=0.4" \
        "sentencepiece" \
        "jax>=0.5" "jaxlib>=0.5" \
        "litert-torch==0.9.0" \
        "mediapipe==0.10.18" \
        "huggingface-hub" \
        "absl-py" \
        "flatbuffers"

WORKDIR /work
COPY convert.py /work/convert.py
COPY probe.py /work/probe.py

# Default mode prints usage. Pass `probe` or `convert` as the first arg.
ENTRYPOINT ["python", "-u", "/work/convert.py"]
CMD ["help"]
