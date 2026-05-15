# Efficient AI in Practice: Quantization and Fast Inference

Code repository for the book *[Efficient AI in Practice: Quantization and Fast Inference](https://www.manning.com/)*.

<br>

## Table of Contents

| Chapter | Title | Code |
|:-------:|-------|------|
| 1 | [The Efficiency Crisis](ch1/) | [ch01_figures.py](ch1/ch01_figures.py) |
| 2 | [Building Quantization from First Principles](ch2/) | [Notebook](ch2/Building%20Quantization%20from%20First%20Principles.ipynb) &bull; [Script](ch2/build_quantization_from_first_principles.py) |
| 3 | [Granularity Choices](ch3/) | [3.1 ResNet-18 Distributions](ch3/3.1%20resnet18_dist.ipynb) &bull; [3.2 ResNet-18 Quantization](ch3/3.2%20resnet18_quant.ipynb) &bull; [3.3 Activation Quantization](ch3/3.3_activation_quantization.py) &bull; [3.3 Outlier Visualization](ch3/3.3_outlier_visualization.py) &bull; [3.4 KV-Cache Quantization](ch3/3.4_kv_cache_quantization.py) &bull; [3.4 KV-Cache Granularity](ch3/3.4_kv_cache_granularity.py) &bull; [3.5 Group Quantization](ch3/3.5_group_quantization_analysis.py) |
| 4 | [Calibration & Post-Training Quantization](ch4/) | [Range Estimation](ch4/range_estimation_demo.py) &bull; [Calibration Stability](ch4/calibration_stability.py) &bull; [Vision Calibration](ch4/vision_calibration_analysis.py) &bull; [LLM Calibration](ch4/llm_calibration_builder.py) &bull; [Bitwidth Analysis](ch4/ptq_bitwidth_analysis.py) &bull; [Equalization](ch4/ch4_equalization_multi_arch.py) &bull; [Validation](ch4/validate_calibration.py) |
| 5 | [Quantization-Aware Training](ch5/) | [Fake Quantization & STE](ch5/ch5_fake_quantization_ste.py) &bull; [Per-Channel QAT](ch5/ch5_per_channel_qat.py) &bull; [PTQ Failure Diagnostics](ch5/ch5_ptq_failure_diagnostics.py) &bull; [QAT Schedules](ch5/ch5_qat_schedule.py) &bull; [Transformer QAT](ch5/ch5_transformer_qat.py) |
| 6 | [Quantization Pathways](ch6/) | [PyTorch TorchAO Path](ch6/ch6_pytorch_torchao_path.py) &bull; [ONNX Export Path](ch6/ch6_onnx_export_path.py) &bull; [TF MOT Path](ch6/ch6_tf_mot_path.py) &bull; [Verify Equivalence](ch6/ch6_verify_equivalence.py) |
| 7 | [Quantizing Large Language Models in Practice](ch7/) | [7.1 Outlier Profiling](ch7/ch7_outlier_profiling.py) &bull; [7.2 LLM.int8() Decomposition](ch7/ch7_llm_int8_decomposition.py) &bull; [7.2 LLM.int8() Flow](ch7/ch7_llm_int8_flow.py) &bull; [7.3 GPTQ](ch7/ch7_gptq_quantization.py) &bull; [7.4 AWQ](ch7/ch7_awq_quantization.py) &bull; [7.5 TurboQuant KV Cache](ch7/ch7_turboquant_kv_cache.py) &bull; [Decision Tree](ch7/ch7_quant_decision_tree.py) |
| 8 | [Sub-8-bit Formats](ch8/) | [8.2 FP8 Formats](ch8/ch8_fp8_formats.py) &bull; [8.3 FP4 Blockscale](ch8/ch8_fp4_blockscale.py) &bull; [8.4 QLoRA NF4](ch8/ch8_qlora_nf4.py) &bull; [8.5 Ternary 1.58-bit](ch8/ch8_ternary.py) &bull; [Pareto Frontier](ch8/ch8_pareto_frontier.py) |
| 9 | [Deployment Pipelines](ch9/) | [9.2 ORT + Optimum](ch9/ch9_ort_optimum_deployment.py) &bull; [9.3 TensorRT Engines](ch9/ch9_tensorrt_engines.py) &bull; [9.4 OpenVINO](ch9/ch9_openvino_deployment.py) &bull; [9.5 Packaging & Serving](ch9/ch9_packaging_serving.py) |
| 10 | [CPU-Friendly LLM Serving with llama.cpp / GGUF](ch10/) | [10.1 Cost Curve](ch10/ch10_cost_curve.py) &bull; [10.2 GGUF Format](ch10/ch10_gguf_format.py) &bull; [10.3 HF → GGUF Convert](ch10/ch10_convert.py) &bull; [10.4 Kernel Families](ch10/ch10_kernels.py) &bull; [10.5 Runtime & Throughput](ch10/ch10_runtime.py) |

<br>

## Chapter Summaries

### Ch 1 — The Efficiency Crisis
Why quantization matters: energy costs, memory bandwidth bottlenecks, and the fundamental gap between floating-point and integer arithmetic.

### Ch 2 — Building Quantization from First Principles
Symmetric vs. asymmetric quantization, the zero-point nudge, hybrid quantization (symmetric weights + asymmetric activations), integer arithmetic pipelines, and error trade-offs between granular and overload error.

### Ch 3 — Granularity Choices
Per-tensor vs. per-channel vs. per-group quantization, range utilization analysis, KV-cache quantization asymmetries, and outlier handling strategies across ResNet-18 and BERT.

### Ch 4 — Calibration & Post-Training Quantization
Activation range estimation (MinMax, entropy, percentile), calibration set construction for vision and language models, cross-layer equalization, bitwidth selection, and calibration coverage validation.

### Ch 5 — Quantization-Aware Training
Fake quantization nodes, straight-through estimators, observer-based scale computation, progressive quantization schedules, and fine-tuning strategies for CNNs and Transformers.

### Ch 6 — Quantization Pathways
Weight-only quantization via PyTorch TorchAO, ONNX Runtime dynamic and static quantization with mixed-precision analysis, TensorFlow Lite post-training quantization (dynamic, full-integer, float16), and cross-framework numerical equivalence verification.

### Ch 7 — Quantizing Large Language Models in Practice
Activation outlier profiling on OPT, LLM.int8() mixed-precision decomposition, GPTQ Hessian-aware groupwise quantization, AWQ activation-aware weight protection, and TurboQuant KV-cache vector quantization — each method built from scratch and validated against its production library (bitsandbytes, gptqmodel, autoawq).

### Ch 8 — Sub-8-bit Formats
FP8 (E4M3 / E5M2) formats and kernel caveats, FP4 (E2M1) with blockwise scaling, NF4 with QLoRA fine-tuning, and ternary (1.58-bit) BitNet models — covering the encode/decode internals, the sub-8-bit Pareto frontier on OPT-6.7B, and end-to-end perplexity comparisons.

### Ch 9 — Deployment Pipelines
Deploying quantized artifacts through ONNX Runtime + Optimum across CPU / CUDA / TensorRT execution providers, building real INT8 TensorRT engines (implicit vs explicit QDQ), OpenVINO standalone vs via-ORT on Intel silicon, and packaging models for serving with manifests, Triton config.pbtxt, and provenance metadata.

### Ch 10 — CPU-Friendly LLM Serving with llama.cpp / GGUF
The cost-per-million-tokens curve for self-hosted inference, the GGUF v3 file format dissected byte by byte, the safetensors → GGUF → quantized-variant pipeline on Llama-2-7B, x86 kernel families (AVX-2 vs AVX-512+VNNI vs AMX) compared against Apple Silicon NEON/Metal, and runtime memory + throughput measurement.

<br>

## Prerequisites

- Python 3.9+
- Familiarity with deep learning concepts (neural networks, backpropagation)
- Basic PyTorch experience

### Key Dependencies

- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/) (for BERT, TinyLlama examples)
- [TorchVision](https://pytorch.org/vision/) (for ResNet, MobileNetV2, ViT examples)
- Matplotlib / Seaborn (for figures)
- NumPy

<br>

## Hardware Requirements

All code in this repository is designed to run on a standard laptop or desktop. No GPU is required, though a CUDA-capable GPU will speed up the training examples in Chapters 4 and 5.

<br>

## Models Used

The book uses a variety of models to demonstrate quantization across architectures:

| Domain | Models |
|--------|--------|
| Vision | ResNet-18, MobileNetV2, ViT-B/16 |
| Language (small) | BERT, TinyLlama |
| Language (LLM) | OPT-125m / 1.3B / 2.7B / 6.7B, Llama-2-7B, BitNet b1.58 2B4T |

<br>

## Citation

<br>

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Vivek Kalyanarangan
