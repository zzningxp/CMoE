# DartMQ
**DartMQ: Direct Mixed-Precision Quantization for Fixed-Memory-Constrained LLM Deployment**

**DartMQ** is a deterministic, low-overhead framework designed for deploying Large Language Models (LLMs) on resource-constrained embedded systems and consumer-grade GPUs. Unlike traditional quantization methods, DartMQ guarantees that the model fits within a strict, user-defined memory budget while maximizing accuracy.

Our core insight is transforming the precision allocation problem into a **Grouped Knapsack Problem**, solved optimally via **Dynamic Programming (DP)**. This approach eliminates the stochastic search overhead of AutoML methods, reducing search time from hours to milliseconds.

## Key Features

*   **Hard Memory Constraint Satisfaction:** DartMQ prioritizes "fit-first, optimize-second." It guarantees the quantized model will strictly fit within your GPU/VRAM budget (e.g., 4GB, with command args --vram-quota 4.0 ), avoiding Out-Of-Memory (OOM) crashes.
*   **Global Optimality & Determinism:** By using Dynamic Programming, we find the globally optimal mixed-precision configuration. Unlike genetic algorithms, our results are reproducible and deterministic.
*   **Calibrated Sensitivity:** We correct the bias in traditional Hessian estimation (GPTQ loss) by calibrating local reconstruction errors to actual global Perplexity (PPL) impact. This ensures critical layers (like projection layers) are preserved at higher precision.
*   **Unmatched Speed:** Reduce the mixed-precision search overhead by **3–4 orders of magnitude** compared to state-of-the-art AutoML methods (e.g., AMQ).
*   **Export to GGUF Format:** Export the mixture precision quantized model to pt
format, then GGUF format with llama.cpp.

## Methodology

### 1. The Problem: The Memory-Accuracy Staircase
Uniform quantization (e.g., all 4-bit) creates a "dead zone" of unused memory. For example, a model might fit in 3-bit (low accuracy) or not fit in 4-bit (high accuracy), leaving a gap of unused VRAM capacity.

### 2. The DartMQ Solution
We solve this by assigning varying bit-widths (3, 4, 5, 6, 8 bits) to different operators (layers).

1.  **Pre-Quantization Profiling:** We perform a one-time profiling to each operator to global perplexity impact, correcting the variance-induced bias in Hessian estimation.
2.  **Grouped Knapsack Formulation:**
    *   **Goal:** Minimize total weighted quantization error.
    *   **Constraint:** Total VRAM footprint ≤ User-defined Budget ($V_{max}$).
3.  **Dynamic Programming Solver:** We solve the allocation problem deterministically in milliseconds.

## Benchmark Results

DartMQ performs near mixed-precision (AMQ) methods under identical memory constraints, but with significantly faster search time.

## vs AMQ 
AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models: http://arxiv.org/abs/2509.12019, EMNLP 2025 Oral



## Dependencies

```bash
conda create -n dmq python=3.11
conda activate dmq
conda install pytorch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 pytorch-cuda=12.8 -c pytorch -c nvidia
pip install datasets==4.4.1
pip install transformers==4.57.5
pip install accelerate==1.12.0
pip install sentencepiece==0.2.0
pip install protobuf==6.33.3
pip install matplotlib==3.10.0
pip install lap==0.5.12
pip install peft==0.14.0
```
Note: please modify the version of some packages for your own environment.

## Supported Models

Dense Models:

Llama-2-7B / Llama-2-13B / Llama3-8B / Qwen3-8B / Qwen3-4B

## Quick Start

```
python run_reconstruct.py $MODEL_PATH wikitext2 --mixqdense --nsamples 64 --vram-quota 3 --not-quant-lm-head --recompute-pre-quant
```

## Evaluation

bash run.sh
```
# python
python eval_cmoe.py $MODEL_PATH 
```

## Code sources

Framework code is referenced from: https://github.com/JarvisPei/CMoE

GPTQ code is referenced from: https://github.com/cat538/MxMoE

Important code is inspired from: https://github.com/xuyuzhuang11/CAMERA

## Cite

If you found this work useful, please consider citing:

```

```
