#!/bin/bash
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=gguf_%j.out
#SBATCH --error=gguf_%j.err
#SBATCH -w gn[8]

# 设置 CUDA 环境（即使 perplexity 可能只用 CPU，保留以防需要 GPU 加载模型）
export CUDA_HOME=/gf3/softwares/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


# python run_reconstruct.py /gf3/home/mqt/Eagle/models/Qwen3-8B wikitext2 \
#   --mixqdense \
#   --nsamples 64 \
#   --vram-quota 4 \
#   --export-gptq-data \
#   --gptq-export-dir gptq_export_Q4

# python run_reconstruct.py /gf3/home/mqt/Eagle/models/Llama3.1-8B wikitext2 \
#   --mixqdense \
#   --nsamples 64 \
#   --vram-quota 999 \
#   --profile-only-quant-layers -1 \
#   --export-gptq-data \
#   --gptq-export-dir gptq_export_4

# python convert_hf_to_gguf.py /gf3/home/mqt/Eagle/Eagle3-llama/llama.cpp/DartMQ/model/carved_llama_e1a1_None --outfile /gf3/home/mqt/Eagle/Eagle3-llama/llama.cpp/DartMQ/model/carved_llama_e1a1_None/llama3_8B_gptq.gguf

# /gf3/home/mqt/Eagle/Eagle3-llama/llama.cpp/build/bin/llama-quantize /gf3/home/mqt/Eagle/Eagle3-llama/llama.cpp/DartMQ/model/carved_llama_e1a1_None/llama3_8B_gptq.gguf /gf3/home/mqt/Eagle/Eagle3-llama/llama.cpp/DartMQ/model/carved_llama_e1a1_None/llama3_8B_gptq_q4_1.gguf Q4_1