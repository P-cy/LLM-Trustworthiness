#!/bin/sh
# Entrypoint wrapper: applies CONFIG_A runtime env, then exec run.py.
# CONFIG=A: int4 ~16GB main + 4B guard ~8GB. Default and only config.
set -e

export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.68}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export VLLM_MAX_SEQS="${VLLM_MAX_SEQS:-32}"
export TRUSTW_MAIN_WEIGHTS_GB="${TRUSTW_MAIN_WEIGHTS_GB:-16}"
export TRUSTW_GUARD_WEIGHTS_GB="${TRUSTW_GUARD_WEIGHTS_GB:-8}"

exec python3 /app/run.py "$@"
