#!/usr/bin/env python3
"""
Chapter 6.3 Companion Script: Run the TensorFlow Path with TF MOT

Section 6.2 quantized models in PyTorch with TorchAO and PT2E. This script
runs the same quantization intent through TensorFlow's post-training
quantization pipeline — accessed via the TFLite converter — and measures
what changes.

Two models:
  - MobileNetV2 (ImageNet-pretrained, evaluated on ImageNette)
  - BERT-base fine-tuned on SST-2 (loaded from HuggingFace, from_pt=True)

Three quantization modes:
  - Dynamic range: weights quantized to INT8 at conversion time,
    activations quantized dynamically at inference. No calibration data
    needed. Analogous to TorchAO's Int8WeightOnlyConfig.
  - Full integer: both weights and activations quantized to INT8 using
    a representative dataset for calibration. Analogous to TorchAO's
    Int8DynamicActivationInt8WeightConfig (but static, not dynamic).
  - Float16: weights stored in FP16. 2× compression with negligible
    accuracy impact. No PyTorch parallel in TorchAO's stable configs.

Usage:
    python ch6_tf_mot_path.py --all --save-plots
    python ch6_tf_mot_path.py --mobilenet --mode all
    python ch6_tf_mot_path.py --bert --mode dynamic
    python ch6_tf_mot_path.py --all --mode full-integer --num-samples 200

Dependencies:
    pip install tensorflow numpy Pillow matplotlib
    pip install "transformers>=4.40,<5.0" datasets    # BERT only; v5+ dropped TF
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Suppress TF verbose logging before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "ch6_outputs"

# ── ImageNette class indices in ImageNet (1000-class) ─────────────────
# ImageNette contains 10 easily-classifiable ImageNet classes.
# We evaluate MobileNetV2's 1000-class predictions against these.
IMAGENETTE_CLASSES = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}
IMAGENETTE_WNIDS = sorted(IMAGENETTE_CLASSES.keys())
IMAGENETTE_TO_INET = [IMAGENETTE_CLASSES[k] for k in IMAGENETTE_WNIDS]


# =====================================================================
# DATA LOADING
# =====================================================================

def load_imagenette(data_dir, split="val", image_size=224, max_samples=None):
    """Load ImageNette validation set, return (images, labels) as numpy arrays.

    Images are preprocessed for MobileNetV2 (224×224, scaled to [-1, 1]).
    Labels are ImageNet class indices (0-999).
    """
    from PIL import Image

    dataset_path = Path(data_dir) / "imagenette2"
    if not dataset_path.exists():
        # Download ImageNette
        import tarfile
        import urllib.request
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
        tgz_path = Path(data_dir) / "imagenette2.tgz"
        tgz_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading ImageNette to {tgz_path}...")
        urllib.request.urlretrieve(url, tgz_path)
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Done.")

    split_dir = dataset_path / split
    images, labels = [], []
    # Sort for reproducibility
    for class_idx, wnid in enumerate(IMAGENETTE_WNIDS):
        class_dir = split_dir / wnid
        if not class_dir.exists():
            continue
        inet_label = IMAGENETTE_TO_INET[class_idx]
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpeg", ".jpg", ".png"):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((image_size, image_size))
                arr = np.array(img, dtype=np.float32)
                # MobileNetV2 preprocessing: scale to [-1, 1]            #A
                arr = (arr / 127.5) - 1.0
                images.append(arr)
                labels.append(inet_label)
            except Exception:
                continue
            if max_samples and len(images) >= max_samples:
                return np.array(images), np.array(labels)

    return np.array(images), np.array(labels)
    # A MobileNetV2 expects inputs in [-1, 1], not [0, 1] or [0, 255].
    #   Using tf.keras.applications.mobilenet_v2.preprocess_input would
    #   do the same transform, but explicit math is clearer for teaching.


def load_sst2(max_samples=None):
    """Load SST-2 validation set, return (texts, labels).

    Returns raw text strings — tokenization happens at conversion/eval time.
    """
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="validation")
    texts = ds["sentence"]
    labels = np.array(ds["label"])
    texts = list(texts)                                                 #A
    if max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]
    return texts, labels
    # A HuggingFace datasets returns a lazy column, not a Python list.
    #   numpy int64 indices (from np.random.choice) fail on lazy columns.
    #   list() materializes it for reliable indexing.


# =====================================================================
# MODEL LOADING AND SAVEDMODEL EXPORT
# =====================================================================

def load_mobilenet_v2(saved_model_dir):
    """Load MobileNetV2 from Keras and export as SavedModel.

    The TFLite converter requires a SavedModel or concrete function.
    tf.keras.applications.MobileNetV2 gives us an ImageNet-pretrained
    model with 3.5M parameters — purpose-built for mobile deployment.
    """
    model = tf.keras.applications.MobileNetV2(                          #A
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    # A MobileNetV2: 3.5M params, 88% Conv2d + DepthwiseConv2d. Unlike
    #   ResNet-18 in section 6.2, TF MOT quantizes ALL these layers — not
    #   just nn.Linear. This is a fundamental difference from TorchAO.

    saved_model_path = str(saved_model_dir / "mobilenet_v2_fp32")

    # Keras 3.x removed model.save() for SavedModel directories.        #B
    # model.export() produces a SavedModel that TFLiteConverter accepts.
    try:
        model.export(saved_model_path)                                  #B
    except (AttributeError, TypeError):
        # Keras 2.x fallback
        model.save(saved_model_path, save_format="tf")
    print(f"  SavedModel exported to {saved_model_path}")
    # B In Keras 2.x, model.save("path/", save_format="tf") created a
    #   SavedModel. Keras 3.x requires model.export("path/") instead.
    #   TFLiteConverter.from_saved_model needs a SavedModel directory.

    # Measure parameter breakdown
    conv_params = 0
    dense_params = 0
    other_params = 0
    for layer in model.layers:
        layer_params = sum(
            int(np.prod(w.shape)) for w in layer.trainable_weights
        )
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            conv_params += layer_params
        elif isinstance(layer, tf.keras.layers.Dense):
            dense_params += layer_params
        else:
            other_params += layer_params

    total = conv_params + dense_params + other_params
    if total > 0:
        print(f"  Architecture: Conv/DW={conv_params/total:.1%}, "
              f"Dense={dense_params/total:.1%}, "
              f"Other={other_params/total:.1%}")
    print(f"  Total trainable params: {total:,}")
    return model, saved_model_path


def load_bert_sst2(saved_model_dir, max_seq_len=128):
    """Load BERT-base fine-tuned on SST-2 and export as SavedModel.

    The same model from section 6.2, but loaded via HuggingFace's TF
    backend. from_pt=True converts PyTorch weights to TF format.

    Critical for TFLite: input shapes must be fixed. We trace the model
    with a concrete batch_size=1 and fixed sequence length. This is a
    key difference from PyTorch, which allows dynamic shapes by default.

    Requires: transformers<5.0 (v5.0+ dropped TensorFlow support).       #C
    Install: pip install "transformers>=4.40,<5.0" datasets
    """
    # C HuggingFace removed TF model classes in transformers 5.0 (2025).
    #   This is a real ecosystem constraint: the same library that provides
    #   the PyTorch model in section 6.2 no longer supports loading it into
    #   TensorFlow. Pin to transformers 4.x for cross-framework workflows.
    try:
        from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    except ImportError:
        print("\n  ERROR: TFAutoModelForSequenceClassification not available.")
        print("  HuggingFace transformers v5+ dropped TensorFlow support.")
        print("  Fix: pip install 'transformers>=4.40,<5.0'")
        print("  Then re-run this script.\n")
        raise SystemExit(1)

    model_name = "textattack/bert-base-uncased-SST-2"
    print(f"  Loading {model_name} (from_pt=True)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(        #A
        model_name, from_pt=True
    )
    # A from_pt=True: the SST-2 checkpoint is PyTorch-native. HuggingFace
    #   automatically converts weights to TF format. This cross-framework
    #   weight transfer is itself a source of potential numerical divergence.

    # Export as SavedModel with fixed input shapes                      #B
    saved_model_path = str(saved_model_dir / "bert_sst2_fp32")

    # Build concrete function with fixed shapes for TFLite
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, max_seq_len], dtype=tf.int32,
                      name="input_ids"),
        tf.TensorSpec(shape=[1, max_seq_len], dtype=tf.int32,
                      name="attention_mask"),
    ])
    def serving_fn(input_ids, attention_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return {"logits": outputs.logits}
    # B TFLite requires static input shapes. We fix batch_size=1 and
    #   sequence_length=128. PyTorch's torch.export also requires shape
    #   specialization, but TorchAO's quantize_() works in eager mode
    #   without it — a usability difference.

    tf.saved_model.save(model, saved_model_path,
                        signatures={"serving_default": serving_fn})
    print(f"  SavedModel exported to {saved_model_path}")
    print(f"  Fixed input shape: [1, {max_seq_len}] (batch=1, seq={max_seq_len})")

    return model, tokenizer, saved_model_path


# =====================================================================
# TFLITE CONVERSION WITH QUANTIZATION
# =====================================================================

def convert_fp32(saved_model_path, output_path):
    """Baseline: convert to TFLite without quantization."""
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
    return tflite_model


def convert_dynamic_range(saved_model_path, output_path):
    """Dynamic range quantization: weights to INT8, activations dynamic.

    This is the simplest path — no calibration data required. The
    converter statically quantizes weight tensors from FP32 to INT8 at
    conversion time. At inference, activation tensors are quantized
    dynamically (per-batch min/max) when entering quantized kernels.

    Analogous to TorchAO's Int8WeightOnlyConfig, but with a key
    difference: TF MOT quantizes ALL layers including Conv2d and
    DepthwiseConv2d, while TorchAO only targets nn.Linear.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]                #A
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
    return tflite_model
    # A Optimize.DEFAULT with no representative_dataset triggers dynamic
    #   range quantization. Weights become INT8, activations stay dynamic.


def convert_full_integer(saved_model_path, output_path,
                         representative_dataset_fn):
    """Full integer quantization: weights AND activations to INT8.

    Requires a representative dataset — a Python generator that yields
    sample inputs. The converter runs these through the model to collect
    activation ranges, then quantizes both weights and activations to INT8.

    This is TF MOT's equivalent of static post-training quantization with
    calibration (Chapter 4). The representative_dataset serves the same
    role as the calibration set in sections 4.2–4.3.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_fn        #A
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
    return tflite_model
    # A The representative_dataset is a callable that yields lists of numpy
    #   arrays (one per model input). The converter consumes all yielded
    #   samples to estimate activation ranges — typically 100-500 samples.
    #   More samples → tighter ranges → better accuracy, with diminishing
    #   returns beyond ~200 for most models.


def convert_full_integer_strict(saved_model_path, output_path,
                                representative_dataset_fn):
    """Full integer quantization with INT8 inputs/outputs.

    Same as full integer, but additionally constrains model inputs and
    outputs to INT8. Required for integer-only hardware (microcontrollers,
    Coral Edge TPU) that cannot handle float I/O.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_fn
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8                            #A
    ]
    converter.inference_input_type = tf.uint8                           #B
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
    return tflite_model
    # A Restrict to INT8-only ops — if any op lacks an INT8 kernel, the
    #   converter raises an error instead of silently falling back to float.
    # B Inputs/outputs also become INT8. The calling code must handle
    #   quantization/dequantization of inputs and outputs manually.


def convert_float16(saved_model_path, output_path):
    """Float16 quantization: weights stored as FP16.

    No calibration data needed. Weights are halved in precision, giving
    ~2× model size reduction with negligible accuracy loss. At inference,
    FP16 weights are upcast to FP32 — no FP16 compute unless the device
    supports it (GPU delegates).

    No direct TorchAO parallel: TorchAO's stable configs focus on INT8.
    PyTorch's native FP16 inference uses model.half(), which converts
    both weights AND computation to FP16.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]                #A
    tflite_model = converter.convert()
    Path(output_path).write_bytes(tflite_model)
    return tflite_model
    # A supported_types=[tf.float16] tells the converter to store weights
    #   in FP16 format. Operations still run in FP32 on CPU.


# =====================================================================
# REPRESENTATIVE DATASET GENERATORS
# =====================================================================

def make_mobilenet_representative_dataset(images, num_calibration=200):
    """Create a representative dataset generator for MobileNetV2.

    The generator must yield lists of numpy arrays, one per model input.
    MobileNetV2 has a single input: [batch, 224, 224, 3] in float32.
    """
    def generator():                                                    #A
        indices = np.random.RandomState(42).choice(
            len(images), size=min(num_calibration, len(images)),
            replace=False
        )
        for i in indices:
            # Single image, batch dimension added                       #B
            yield [images[i:i+1].astype(np.float32)]
    return generator
    # A The generator is a callable (not the iterator itself). The converter
    #   calls it once and consumes all yielded samples.
    # B Each yield must be a list with one element per model input.
    #   Shape must match the model's expected input exactly.


def make_bert_representative_dataset(tokenizer, texts, max_seq_len=128,
                                     num_calibration=200):
    """Create a representative dataset generator for BERT.

    BERT has two inputs: input_ids and attention_mask. The generator
    yields [input_ids_array, attention_mask_array] for each sample.
    """
    def generator():
        indices = np.random.RandomState(42).choice(
            len(texts), size=min(num_calibration, len(texts)),
            replace=False
        )
        for i in indices:
            encoded = tokenizer(
                texts[i],
                max_length=max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            yield [
                encoded["input_ids"].astype(np.int32),                  #A
                encoded["attention_mask"].astype(np.int32),
            ]
    return generator
    # A Input ordering must match the SavedModel's signature. We defined
    #   the serving_fn with (input_ids, attention_mask) — the representative
    #   dataset must yield them in the same order.


# =====================================================================
# TFLITE INFERENCE AND EVALUATION
# =====================================================================

def create_interpreter(tflite_model_path):
    """Create a TFLite interpreter from a .tflite file."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    return interpreter


def evaluate_mobilenet_tflite(interpreter, images, labels, tag=""):
    """Evaluate MobileNetV2 TFLite model on ImageNette.

    Returns top-1 accuracy (within the full 1000-class prediction).
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    input_quant = input_details[0].get("quantization_parameters", {})
    input_scale = input_quant.get("scales", np.array([]))
    input_zp = input_quant.get("zero_points", np.array([]))

    correct = 0
    total = 0
    start_time = time.perf_counter()

    for i in range(len(images)):
        img = images[i:i+1].astype(np.float32)

        # Handle quantized input: dequantize manually if needed         #A
        if input_dtype == np.uint8 and len(input_scale) > 0:
            img = (img / input_scale[0] + input_zp[0]).astype(np.uint8)
        elif input_dtype == np.int8 and len(input_scale) > 0:
            img = (img / input_scale[0] + input_zp[0]).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])

        # Handle quantized output
        output_quant = output_details[0].get(
            "quantization_parameters", {}
        )
        output_scale = output_quant.get("scales", np.array([]))
        output_zp = output_quant.get("zero_points", np.array([]))
        if output.dtype != np.float32 and len(output_scale) > 0:
            output = (output.astype(np.float32) - output_zp[0]) * output_scale[0]

        pred = np.argmax(output[0])
        if pred == labels[i]:
            correct += 1
        total += 1
    # A Full-integer models with INT8 I/O require the caller to handle
    #   quantization/dequantization at the boundaries. The scale and
    #   zero_point are embedded in the TFLite model metadata.

    elapsed = time.perf_counter() - start_time
    accuracy = correct / total if total > 0 else 0.0
    throughput = total / elapsed if elapsed > 0 else 0.0
    print(f"  [{tag}] Accuracy: {accuracy:.2%} "
          f"({correct}/{total}), "
          f"{throughput:.1f} samples/sec, "
          f"{elapsed:.1f}s total")
    return {"accuracy": accuracy, "correct": correct, "total": total,
            "elapsed_sec": elapsed, "throughput": throughput}


def evaluate_bert_tflite(interpreter, tokenizer, texts, labels,
                         max_seq_len=128, tag=""):
    """Evaluate BERT TFLite model on SST-2 validation set."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Map input names to detail indices
    input_map = {d["name"]: d for d in input_details}

    correct = 0
    total = 0
    start_time = time.perf_counter()

    for i in range(len(texts)):
        encoded = tokenizer(
            texts[i],
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        input_ids = encoded["input_ids"].astype(np.int32)
        attention_mask = encoded["attention_mask"].astype(np.int32)

        # Set inputs by matching tensor names                           #A
        for detail in input_details:
            name = detail["name"]
            if "input_ids" in name:
                interpreter.set_tensor(detail["index"], input_ids)
            elif "attention_mask" in name:
                interpreter.set_tensor(detail["index"], attention_mask)

        interpreter.invoke()

        # Get logits
        output = interpreter.get_tensor(output_details[0]["index"])
        pred = np.argmax(output[0])
        if pred == labels[i]:
            correct += 1
        total += 1

        if (i + 1) % 200 == 0:
            print(f"    ... evaluated {i+1}/{len(texts)} samples")
    # A TFLite input tensor names may not match the original model's
    #   argument names exactly. We match by substring to handle naming
    #   variations across TF versions.

    elapsed = time.perf_counter() - start_time
    accuracy = correct / total if total > 0 else 0.0
    throughput = total / elapsed if elapsed > 0 else 0.0
    print(f"  [{tag}] Accuracy: {accuracy:.2%} "
          f"({correct}/{total}), "
          f"{throughput:.1f} samples/sec, "
          f"{elapsed:.1f}s total")
    return {"accuracy": accuracy, "correct": correct, "total": total,
            "elapsed_sec": elapsed, "throughput": throughput}


# =====================================================================
# TFLITE MODEL INSPECTION
# =====================================================================

def inspect_tflite_model(tflite_path, tag="", max_tensors=10):
    """Inspect a TFLite model: size, tensor dtypes, quantization params.

    This is the TFLite equivalent of inspecting AffineQuantizedTensor in
    section 6.2. Instead of looking at tensor subclasses, we examine the
    TFLite flatbuffer's tensor metadata.
    """
    file_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"\n  [{tag}] Model size: {file_size_mb:.2f} MB")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()
    dtype_counts = {}
    quantized_count = 0
    total_count = 0

    for t in tensor_details:
        dtype = str(t["dtype"])
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        total_count += 1
        quant_params = t.get("quantization_parameters", {})
        scales = quant_params.get("scales", np.array([]))
        if len(scales) > 0 and scales[0] != 0.0:
            quantized_count += 1

    print(f"  Tensor dtype distribution: {dict(dtype_counts)}")
    print(f"  Quantized tensors: {quantized_count}/{total_count}")

    # Show sample weight tensors                                        #A
    print(f"  Sample tensors (first {max_tensors} with quant params):")
    shown = 0
    for t in tensor_details:
        quant_params = t.get("quantization_parameters", {})
        scales = quant_params.get("scales", np.array([]))
        zero_points = quant_params.get("zero_points", np.array([]))
        if len(scales) > 0 and scales[0] != 0.0:
            scale_summary = (f"scale={scales[0]:.6f}"
                           if len(scales) == 1
                           else f"scales=[{scales[0]:.6f}..{scales[-1]:.6f}], "
                                f"n_scales={len(scales)}")
            print(f"    {t['name'][:60]:60s} "
                  f"dtype={str(t['dtype']):12s} "
                  f"shape={str(t['shape']):20s} "
                  f"{scale_summary}")
            shown += 1
            if shown >= max_tensors:
                break
    # A In the TFLite flatbuffer, quantization parameters (scale, zero_point)
    #   are stored per-tensor. Per-channel quantization stores one scale per
    #   output channel — visible as n_scales matching the output dimension.

    return {"size_mb": file_size_mb, "dtype_counts": dtype_counts,
            "quantized_tensors": quantized_count,
            "total_tensors": total_count}


# =====================================================================
# MOBILENET EXPERIMENT
# =====================================================================

def run_mobilenet_experiment(args):
    """Full MobileNetV2 quantization experiment on ImageNette."""
    print("\n" + "=" * 70)
    print("MOBILENET-V2 ON IMAGENETTE")
    print("=" * 70)

    saved_model_dir = OUTPUT_DIR / "saved_models"
    tflite_dir = OUTPUT_DIR / "tflite_models"
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    tflite_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    print("\n[1] Loading MobileNetV2...")
    model, saved_model_path = load_mobilenet_v2(saved_model_dir)

    # ── Load evaluation data ──────────────────────────────────────────
    print("\n[2] Loading ImageNette validation set...")
    images, labels = load_imagenette(
        args.data_dir, split="val",
        max_samples=args.num_samples,
    )
    print(f"  Loaded {len(images)} samples")

    results = {"model": "MobileNetV2", "dataset": "ImageNette",
               "num_samples": len(images), "configs": {}}

    # ── FP32 baseline ─────────────────────────────────────────────────
    print("\n[3] Converting FP32 baseline to TFLite...")
    fp32_path = tflite_dir / "mobilenet_v2_fp32.tflite"
    convert_fp32(saved_model_path, fp32_path)
    fp32_info = inspect_tflite_model(fp32_path, "FP32")
    print("\n  Evaluating FP32 baseline...")
    fp32_interp = create_interpreter(str(fp32_path))
    fp32_eval = evaluate_mobilenet_tflite(fp32_interp, images, labels, "FP32")
    results["configs"]["fp32"] = {
        **fp32_info, **fp32_eval, "compression": 1.0
    }
    baseline_size = fp32_info["size_mb"]

    # ── Dynamic range quantization ────────────────────────────────────
    if args.mode in ("dynamic", "all"):
        print("\n[4a] Dynamic range quantization (weights-only INT8)...")
        dyn_path = tflite_dir / "mobilenet_v2_dynamic.tflite"
        convert_dynamic_range(saved_model_path, dyn_path)
        dyn_info = inspect_tflite_model(dyn_path, "Dynamic")
        dyn_interp = create_interpreter(str(dyn_path))
        print("\n  Evaluating...")
        dyn_eval = evaluate_mobilenet_tflite(
            dyn_interp, images, labels, "Dynamic"
        )
        compression = baseline_size / dyn_info["size_mb"]
        results["configs"]["dynamic"] = {
            **dyn_info, **dyn_eval, "compression": compression
        }
        print(f"  Compression: {compression:.2f}×")

    # ── Full integer quantization ─────────────────────────────────────
    if args.mode in ("full-integer", "all"):
        print("\n[4b] Full integer quantization (W8A8 with calibration)...")
        repr_fn = make_mobilenet_representative_dataset(
            images, num_calibration=args.num_calibration
        )
        full_path = tflite_dir / "mobilenet_v2_full_integer.tflite"
        convert_full_integer(saved_model_path, full_path, repr_fn)
        full_info = inspect_tflite_model(full_path, "Full-INT8")
        full_interp = create_interpreter(str(full_path))
        print("\n  Evaluating...")
        full_eval = evaluate_mobilenet_tflite(
            full_interp, images, labels, "Full-INT8"
        )
        compression = baseline_size / full_info["size_mb"]
        results["configs"]["full_integer"] = {
            **full_info, **full_eval, "compression": compression
        }
        print(f"  Compression: {compression:.2f}×")

    # ── Full integer with INT8 I/O (strict) ───────────────────────────
    if args.mode in ("full-integer", "all"):
        print("\n[4c] Full integer with INT8 I/O (edge device target)...")
        repr_fn = make_mobilenet_representative_dataset(
            images, num_calibration=args.num_calibration
        )
        strict_path = tflite_dir / "mobilenet_v2_int8_strict.tflite"
        try:
            convert_full_integer_strict(
                saved_model_path, strict_path, repr_fn
            )
            strict_info = inspect_tflite_model(strict_path, "INT8-strict")
            strict_interp = create_interpreter(str(strict_path))
            print("\n  Evaluating...")
            strict_eval = evaluate_mobilenet_tflite(
                strict_interp, images, labels, "INT8-strict"
            )
            compression = baseline_size / strict_info["size_mb"]
            results["configs"]["int8_strict"] = {
                **strict_info, **strict_eval, "compression": compression
            }
            print(f"  Compression: {compression:.2f}×")
        except Exception as e:
            print(f"  INT8-strict conversion failed: {e}")
            print("  (Some ops may lack INT8 kernel implementations)")
            results["configs"]["int8_strict"] = {"error": str(e)}

    # ── Float16 quantization ──────────────────────────────────────────
    if args.mode in ("fp16", "all"):
        print("\n[4d] Float16 quantization (weights in FP16)...")
        fp16_path = tflite_dir / "mobilenet_v2_fp16.tflite"
        convert_float16(saved_model_path, fp16_path)
        fp16_info = inspect_tflite_model(fp16_path, "FP16")
        fp16_interp = create_interpreter(str(fp16_path))
        print("\n  Evaluating...")
        fp16_eval = evaluate_mobilenet_tflite(
            fp16_interp, images, labels, "FP16"
        )
        compression = baseline_size / fp16_info["size_mb"]
        results["configs"]["fp16"] = {
            **fp16_info, **fp16_eval, "compression": compression
        }
        print(f"  Compression: {compression:.2f}×")

    # ── Summary ───────────────────────────────────────────────────────
    print_summary_table("MobileNetV2 on ImageNette", results)
    return results


# =====================================================================
# BERT EXPERIMENT
# =====================================================================

def run_bert_experiment(args):
    """Full BERT-base SST-2 quantization experiment."""
    print("\n" + "=" * 70)
    print("BERT-BASE ON SST-2")
    print("=" * 70)

    saved_model_dir = OUTPUT_DIR / "saved_models"
    tflite_dir = OUTPUT_DIR / "tflite_models"
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    tflite_dir.mkdir(parents=True, exist_ok=True)

    max_seq_len = 128

    # ── Load model ────────────────────────────────────────────────────
    print("\n[1] Loading BERT-base SST-2...")
    model, tokenizer, saved_model_path = load_bert_sst2(
        saved_model_dir, max_seq_len=max_seq_len
    )

    # ── Load evaluation data ──────────────────────────────────────────
    print("\n[2] Loading SST-2 validation set...")
    texts, labels = load_sst2(max_samples=args.num_samples)
    print(f"  Loaded {len(texts)} samples")

    results = {"model": "BERT-base", "dataset": "SST-2",
               "num_samples": len(texts), "configs": {}}

    # ── FP32 baseline ─────────────────────────────────────────────────
    print("\n[3] Converting FP32 baseline to TFLite...")
    fp32_path = tflite_dir / "bert_sst2_fp32.tflite"
    try:
        convert_fp32(saved_model_path, fp32_path)
    except Exception as e:
        print(f"  FP32 conversion failed: {e}")
        print("  Attempting with experimental converter...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,                              #A
        ]
        tflite_model = converter.convert()
        Path(fp32_path).write_bytes(tflite_model)
    # A BERT uses ops that may not have native TFLite kernels (e.g.,
    #   Einsum, certain gather patterns). SELECT_TF_OPS falls back to
    #   full TF kernels when TFLite built-in ops don't cover the graph.
    #   This produces a larger .tflite file but ensures conversion succeeds.

    fp32_info = inspect_tflite_model(fp32_path, "FP32")
    print("\n  Evaluating FP32 baseline...")
    fp32_interp = create_interpreter(str(fp32_path))
    fp32_eval = evaluate_bert_tflite(
        fp32_interp, tokenizer, texts, labels,
        max_seq_len=max_seq_len, tag="FP32"
    )
    results["configs"]["fp32"] = {
        **fp32_info, **fp32_eval, "compression": 1.0
    }
    baseline_size = fp32_info["size_mb"]

    # ── Dynamic range quantization ────────────────────────────────────
    if args.mode in ("dynamic", "all"):
        print("\n[4a] Dynamic range quantization...")
        dyn_path = tflite_dir / "bert_sst2_dynamic.tflite"
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(
                saved_model_path
            )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            tflite_model = converter.convert()
            Path(dyn_path).write_bytes(tflite_model)

            dyn_info = inspect_tflite_model(dyn_path, "Dynamic")
            dyn_interp = create_interpreter(str(dyn_path))
            print("\n  Evaluating...")
            dyn_eval = evaluate_bert_tflite(
                dyn_interp, tokenizer, texts, labels,
                max_seq_len=max_seq_len, tag="Dynamic"
            )
            compression = baseline_size / dyn_info["size_mb"]
            results["configs"]["dynamic"] = {
                **dyn_info, **dyn_eval, "compression": compression
            }
            print(f"  Compression: {compression:.2f}×")
        except Exception as e:
            print(f"  Dynamic range conversion failed: {e}")
            results["configs"]["dynamic"] = {"error": str(e)}

    # ── Full integer: skipped for transformers ──────────────────────────
    # Full-integer quantization bakes static activation scales into the
    # model at conversion time. CNN activations (bounded by ReLU/ReLU6)
    # are stable enough for static ranges — MobileNetV2 lost only 1.58pp.
    # Transformer activations (layer norm, softmax, GELU) vary too much
    # across inputs for a single static scale to represent faithfully.
    # Dynamic range quantization (above) is the correct TFLite path for
    # transformers: weights are INT8, activations are quantized dynamically
    # at inference time with per-batch ranges.
    if args.mode in ("full-integer", "all"):
        print("\n[4b] Full integer quantization: skipped for BERT.")
        print("  Static activation calibration does not work for")
        print("  transformers — activation distributions are too")
        print("  input-dependent. Use dynamic range instead (above).")
        print("  (MobileNetV2 full-integer works: see --mobilenet)")

    # ── Float16 quantization ──────────────────────────────────────────
    if args.mode in ("fp16", "all"):
        print("\n[4c] Float16 quantization...")
        fp16_path = tflite_dir / "bert_sst2_fp16.tflite"
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(
                saved_model_path
            )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            tflite_model = converter.convert()
            Path(fp16_path).write_bytes(tflite_model)

            fp16_info = inspect_tflite_model(fp16_path, "FP16")
            fp16_interp = create_interpreter(str(fp16_path))
            print("\n  Evaluating...")
            fp16_eval = evaluate_bert_tflite(
                fp16_interp, tokenizer, texts, labels,
                max_seq_len=max_seq_len, tag="FP16"
            )
            compression = baseline_size / fp16_info["size_mb"]
            results["configs"]["fp16"] = {
                **fp16_info, **fp16_eval, "compression": compression
            }
            print(f"  Compression: {compression:.2f}×")
        except Exception as e:
            print(f"  FP16 conversion failed: {e}")
            results["configs"]["fp16"] = {"error": str(e)}

    # ── Logit comparison: verify models produce different numbers ───────
    # When accuracy is identical across configs, verify the underlying
    # logits actually differ — ruling out an evaluation bug.
    print("\n[5] Logit comparison (first 5 samples)...")
    tflite_paths = {}
    if (tflite_dir / "bert_sst2_fp32.tflite").exists():
        tflite_paths["fp32"] = tflite_dir / "bert_sst2_fp32.tflite"
    if (tflite_dir / "bert_sst2_dynamic.tflite").exists():
        tflite_paths["dynamic"] = tflite_dir / "bert_sst2_dynamic.tflite"
    if (tflite_dir / "bert_sst2_full_integer.tflite").exists():
        tflite_paths["full_int"] = tflite_dir / "bert_sst2_full_integer.tflite"
    if (tflite_dir / "bert_sst2_fp16.tflite").exists():
        tflite_paths["fp16"] = tflite_dir / "bert_sst2_fp16.tflite"

    if len(tflite_paths) >= 2:
        # Collect logits from each model for first 5 samples
        all_logits = {}
        for tag, path in tflite_paths.items():
            interp = create_interpreter(str(path))
            input_details = interp.get_input_details()
            output_details = interp.get_output_details()
            logits_list = []
            for i in range(min(5, len(texts))):
                encoded = tokenizer(
                    texts[i], max_length=max_seq_len,
                    padding="max_length", truncation=True,
                    return_tensors="np",
                )
                for detail in input_details:
                    name = detail["name"]
                    if "input_ids" in name:
                        interp.set_tensor(
                            detail["index"],
                            encoded["input_ids"].astype(np.int32))
                    elif "attention_mask" in name:
                        interp.set_tensor(
                            detail["index"],
                            encoded["attention_mask"].astype(np.int32))
                interp.invoke()
                logits = interp.get_tensor(output_details[0]["index"])[0].copy()
                logits_list.append(logits)
            all_logits[tag] = logits_list

        # Print side-by-side comparison
        tags = list(all_logits.keys())
        print(f"\n  {'Sample':<8s}", end="")
        for tag in tags:
            print(f"  {tag:>22s}", end="")
        print(f"  {'max|Δ| vs fp32':>16s}")
        print(f"  {'─'*8}", end="")
        for _ in tags:
            print(f"  {'─'*22}", end="")
        print(f"  {'─'*16}")

        for i in range(min(5, len(texts))):
            label_str = "pos" if labels[i] == 1 else "neg"
            print(f"  {i:<3d}({label_str})", end="")
            for tag in tags:
                lg = all_logits[tag][i]
                print(f"  [{lg[0]:+8.4f}, {lg[1]:+8.4f}]", end="")
            # Max absolute difference vs FP32
            if "fp32" in all_logits:
                fp32_lg = all_logits["fp32"][i]
                max_delta = 0.0
                for tag in tags:
                    if tag != "fp32":
                        delta = np.max(np.abs(
                            np.array(all_logits[tag][i]) - fp32_lg
                        ))
                        max_delta = max(max_delta, delta)
                print(f"  {max_delta:>14.6f}", end="")
            print()

        # Summary: how many predictions differ across configs?
        if "fp32" in all_logits:
            fp32_preds = [np.argmax(lg) for lg in all_logits["fp32"]]
            for tag in tags:
                if tag == "fp32":
                    continue
                tag_preds = [np.argmax(lg) for lg in all_logits[tag]]
                n_differ = sum(
                    1 for a, b in zip(fp32_preds, tag_preds) if a != b
                )
                print(f"  {tag} vs fp32: {n_differ}/{len(fp32_preds)} "
                      f"predictions differ in first 5 samples")

    # ── Summary ───────────────────────────────────────────────────────
    print_summary_table("BERT-base on SST-2", results)
    return results


# =====================================================================
# COMPARISON PLOT (OPTIONAL)
# =====================================================================

def save_comparison_plot(all_results, output_dir):
    """Generate a side-by-side comparison chart: size vs accuracy.

    Manning figure conventions: Arial font, min 7pt, width ≤ 5.6",
    hatch patterns for grayscale safety, no figure numbers in titles.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots")
        return

    for model_name, results in all_results.items():
        configs = results.get("configs", {})
        if len(configs) < 2:
            continue

        names = []
        sizes = []
        accuracies = []
        for cname, cdata in configs.items():
            if "error" in cdata:
                continue
            names.append(cname.replace("_", "\n"))
            sizes.append(cdata.get("size_mb", 0))
            accuracies.append(cdata.get("accuracy", 0) * 100)

        if len(names) < 2:
            continue

        hatches = ["", "//", "\\\\", "xx", "..", "++"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 3.0))

        bars1 = ax1.bar(range(len(names)), sizes,
                        color=["#4878A8", "#6CA87D", "#C87D5A",
                               "#8B6CAE", "#C4A843"][:len(names)],
                        edgecolor="black", linewidth=0.5)
        for bar, h in zip(bars1, hatches):
            bar.set_hatch(h)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, fontsize=7, fontfamily="Arial")
        ax1.set_ylabel("Model size (MB)", fontsize=8, fontfamily="Arial")
        ax1.tick_params(axis="y", labelsize=7)

        bars2 = ax2.bar(range(len(names)), accuracies,
                        color=["#4878A8", "#6CA87D", "#C87D5A",
                               "#8B6CAE", "#C4A843"][:len(names)],
                        edgecolor="black", linewidth=0.5)
        for bar, h in zip(bars2, hatches):
            bar.set_hatch(h)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, fontsize=7, fontfamily="Arial")
        ax2.set_ylabel("Accuracy (%)", fontsize=8, fontfamily="Arial")
        ax2.tick_params(axis="y", labelsize=7)

        # Set y-axis range to show differences clearly
        if accuracies:
            min_acc = min(accuracies)
            ax2.set_ylim(max(0, min_acc - 5), 100)

        fig.suptitle(f"{model_name}",
                     fontsize=9, fontfamily="Arial")
        fig.tight_layout()

        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        plot_path = output_dir / f"ch6_tf_mot_{safe_name}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"  Saved plot: {plot_path}")

        # Also save PDF for Manning vector submission
        pdf_path = plot_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)


# =====================================================================
# SUMMARY TABLE
# =====================================================================

def print_summary_table(title, results):
    """Print a formatted summary table of all quantization configs."""
    configs = results.get("configs", {})
    if not configs:
        return

    print(f"\n{'─' * 70}")
    print(f"  {title}: Quantization Summary")
    print(f"{'─' * 70}")
    print(f"  {'Config':<20s} {'Size (MB)':>10s} {'Compress':>10s} "
          f"{'Accuracy':>10s} {'Δ Acc':>8s}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")

    fp32_acc = configs.get("fp32", {}).get("accuracy", 0)
    for cname, cdata in configs.items():
        if "error" in cdata:
            print(f"  {cname:<20s} {'FAILED':>10s}")
            continue
        size = cdata.get("size_mb", 0)
        compression = cdata.get("compression", 0)
        accuracy = cdata.get("accuracy", 0)
        delta = (accuracy - fp32_acc) * 100  # percentage points
        print(f"  {cname:<20s} {size:>10.2f} {compression:>9.2f}× "
              f"{accuracy:>9.2%} {delta:>+7.2f}pp")
    print(f"{'─' * 70}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chapter 6.3: TF MOT post-training quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ch6_tf_mot_path.py --all --mode all --save-plots
    python ch6_tf_mot_path.py --mobilenet --mode dynamic
    python ch6_tf_mot_path.py --bert --mode full-integer --num-samples 200
    python ch6_tf_mot_path.py --all --mode all --num-calibration 100
        """,
    )

    # Model selection
    parser.add_argument("--all", action="store_true",
                        help="Run both MobileNetV2 and BERT experiments")
    parser.add_argument("--mobilenet", action="store_true",
                        help="Run MobileNetV2 experiment")
    parser.add_argument("--bert", action="store_true",
                        help="Run BERT-base experiment")

    # Quantization mode
    parser.add_argument("--mode", type=str, default="all",
                        choices=["dynamic", "full-integer", "fp16", "all"],
                        help="Quantization mode to apply (default: all)")

    # Data settings
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit eval samples (None = full set)")
    parser.add_argument("--num-calibration", type=int, default=200,
                        help="Samples for representative dataset (default: 200)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data download directory")

    # Output settings
    parser.add_argument("--save-plots", action="store_true",
                        help="Save comparison plots (requires matplotlib)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ./ch6_outputs)")

    args = parser.parse_args()

    # Default output dir
    if args.output_dir:
        global OUTPUT_DIR
        OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Default to --all if no model specified
    if not args.all and not args.mobilenet and not args.bert:
        args.all = True

    all_results = {}

    if args.all or args.mobilenet:
        mob_results = run_mobilenet_experiment(args)
        all_results["MobileNetV2 on ImageNette"] = mob_results

    if args.all or args.bert:
        bert_results = run_bert_experiment(args)
        all_results["BERT-base on SST-2"] = bert_results

    # Save results JSON
    json_path = OUTPUT_DIR / "ch6_tf_mot_results.json"

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert_numpy)
    print(f"\nResults saved to {json_path}")

    if args.save_plots:
        save_comparison_plot(all_results, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()