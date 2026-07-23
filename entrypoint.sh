#!/bin/sh
# Entrypoint wrapper: applies runtime env, then exec run.py.
#
# Default config is now Path C: main = Qwen3.5-35B-A3B-GPTQ-Int4 (~23GB int4),
# guard = Qwen3Guard-Gen-4B fp16 (~8GB), vLLM gpu_mem_util=0.70, max_model_len=4096.
# The int4 (Qwen3-30B-A3B) path is still selectable by overriding these env at
# docker run time (the vllm_engine.py Qwen3.5 kwargs are env-gated OFF unless
# VLLM_MAMBA_CACHE_MODE / VLLM_LANGUAGE_MODEL_ONLY are set).
set -e

# vLLM pool. Qwen3.5 int4 weights ~22.7GB + KV ~3.8GB; on H100 40GB, 0.70
# gives a ~28GB pool that fits weights+KV with the guard (8.8GB, separate HF
# process outside vLLM's pool) plus ~2GB headroom. KV budget at 32x4096 = 2.68GB.
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.70}"
# 4096 fits the smaller KV budget (Qwen3.5 is hybrid: only 10/40 layers carry
# KV). Answers in this benchmark are short (H_clear <=600 tok, refusals tiny).
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_MAX_SEQS="${VLLM_MAX_SEQS:-32}"
# Main weights size for run.py's fail-fast assert. Qwen3.5 GPTQ-Int4 = 22.73 GiB.
export TRUSTW_MAIN_WEIGHTS_GB="${TRUSTW_MAIN_WEIGHTS_GB:-23}"
export TRUSTW_GUARD_WEIGHTS_GB="${TRUSTW_GUARD_WEIGHTS_GB:-8}"

# Qwen3.5-specific (Path C). Env-gated in vllm_engine.py; setting them here
# activates the Qwen3.5 load path. Omit/blank to fall back to plain int4.
export VLLM_MAMBA_CACHE_MODE="${VLLM_MAMBA_CACHE_MODE:-align}"
export VLLM_LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-1}"
export VLLM_QUANT="${VLLM_QUANT:-moe_wna16}"

exec python3 /app/run.py "$@"
