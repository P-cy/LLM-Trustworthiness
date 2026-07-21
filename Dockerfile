# syntax=docker/dockerfile:1
# LLM Trustworthiness Challenge - self-contained inference image.
# Target: linux/amd64, runtime GPU H100 40GB, NO internet at runtime.
#
# Weights are DOWNLOADED AT BUILD TIME from HuggingFace (build host needs
# internet) and baked into the image, so runtime is fully offline.
#
# CONFIG=A: main = RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16
#           (int4 / compressed-tensors, ~16GB), vLLM gpu_mem_util=0.68,
#           max_num_seqs=32, guard = Qwen3Guard-Gen-4B fp16 (~8GB).

ARG MAIN_REPO=RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16
ARG GUARD_REPO=Qwen/Qwen3Guard-Gen-4B
ARG HF_TOKEN=

# ---- Stage 1: download weights (needs internet, NOT set offline here) ----
FROM python:3.11-slim AS weights
ARG MAIN_REPO
ARG GUARD_REPO
ARG HF_TOKEN
RUN pip install --no-cache-dir "huggingface_hub[cli]" hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN --mount=type=cache,target=/root/.cache/huggingface \
    if [ -n "$HF_TOKEN" ]; then export HF_TOKEN; fi && \
    echo "downloading main=$MAIN_REPO guard=$GUARD_REPO" && \
    hf download "$MAIN_REPO" --local-dir /models/main && \
    hf download "$GUARD_REPO" --local-dir /models/guard && \
    echo "weights downloaded:" && du -sh /models/*

# ---- Stage 2: runtime base (offline) ----
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libglib2.0-0 libsm6 libxext6 libxrender1 \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

# vLLM 0.24 (Jul 2026) — latest. Pinned exact because the image is built once and
# run as-is on the judge H100 (no surprise from a newer patch). 0.24 includes the
# Marlin MoE kernel fixes needed for compressed-tensors int4 on Qwen3-30B-A3B.
RUN pip install --no-cache-dir \
        "vllm==0.24.0" \
        "transformers>=4.51" \
        "accelerate>=1.6" \
        "compressed-tensors>=0.9.0" \
        "sentencepiece" \
        "protobuf" \
        "pydantic" \
        "numpy<2"

WORKDIR /app
COPY run.py /app/run.py
COPY entrypoint.sh /app/entrypoint.sh
COPY src /app/src
COPY prompts /app/prompts
RUN chmod +x /app/entrypoint.sh

# Bake the downloaded weights into the final (offline) image.
COPY --from=weights /models/main /models/main
COPY --from=weights /models/guard /models/guard

RUN mkdir -p /model/test /result /benchmark_lib

ENV TRUSTW_CONFIG=A \
    MAIN_MODEL_DIR=/models/main \
    GUARD_MODEL_DIR=/models/guard \
    TRUSTW_PROMPT_DIR=/app/prompts \
    TRUSTW_POLICY_PATH=/app/prompts/policy.md \
    TRUSTW_SEED=1337

ENTRYPOINT ["/app/entrypoint.sh"]
