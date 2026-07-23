# syntax=docker/dockerfile:1
# LLM Trustworthiness Challenge - self-contained inference image.
# Target: linux/amd64, runtime GPU H100 40GB, NO internet at runtime.
#
# Weights are DOWNLOADED AT BUILD TIME from HuggingFace (build host needs
# internet) and baked into the image, so runtime is fully offline.
#
# Path C (default): main = Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
#           (GPTQ-Int4 / moe_wna16, ~23GB), vLLM gpu_mem_util=0.70,
#           max_model_len=4096, max_num_seqs=32, guard = Qwen3Guard-Gen-4B
#           fp16 (~8GB). Qwen3.5 is a hybrid (Gated DeltaNet) multimodal MoE;
#           the runtime activates mamba_cache_mode=align + language_model_only
#           via env (see entrypoint.sh). The previous int4 path
#           (RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16) is still
#           buildable by overriding --build-arg MAIN_REPO.

ARG MAIN_REPO=Qwen/Qwen3.5-35B-A3B-GPTQ-Int4
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
# Install vllm FIRST, alone, so pip resolves its own deps without the
# mistral-common[image] extra self-referential cycle against mistral-common
# 1.11.6 (a build-time reproducibility bug: the [image] extra of
# mistral-common 1.11.6 depends on mistral-common==1.11.6 itself). vllm 0.24
# runs fine on numpy 2 (working H200 venv uses numpy 2.3.5 + torch 2.11).
RUN pip install --no-cache-dir "vllm==0.24.0"
RUN pip install --no-cache-dir \
        "transformers>=4.51" \
        "accelerate>=1.6" \
        "compressed-tensors>=0.9.0" \
        "sentencepiece" \
        "protobuf" \
        "pydantic"
# MUST come AFTER the transformers/accelerate install above: accelerate->torch->
# cuda-toolkit==13.0.2 pulls nvidia-cuda-runtime + nvidia-cuda-nvrtc back down to
# 13.0.96/13.0.88, which mismatches vllm's bundled nvidia-cuda-nvcc 13.2.86.
# flashinfer's cccl then errors "CUDA compiler and CUDA toolkit headers are
# incompatible" at JIT, failing vLLM init -> vllm_ok=False -> all rows refusal ->
# H=0.000. Downgrading nvcc to 13.0 instead fails the other way (ptxas 13.0 only
# supports PTX 9.0; sm_90a needs 9.2). So the correct fix is: let all the 13.0
# installs happen, THEN force cudart + nvrtc back UP to 13.2.86 to match nvcc.
# --no-deps so we don't disturb the rest of the vllm dep tree.
RUN pip install --no-cache-dir --force-reinstall --no-deps \
        "nvidia-cuda-runtime==13.2.86" \
        "nvidia-cuda-nvrtc==13.2.86"

# vLLM 0.24 depends on the nvidia-cuda-nvcc wheel, which installs the full CUDA
# toolkit (bin/nvcc + ptxas + nvlink, include/cuda*.h, lib/libcudart) into
# site-packages/nvidia/cu13. vLLM does NOT auto-set CUDA_HOME and the
# python:3.11-slim base has no /usr/local/cuda, so flashinfer/CuTeDSL JIT at
# profile_run fails: "Could not find nvcc and default cuda_home=/usr/local/cuda
# doesnt exist" -> vLLM init fails -> vllm_ok=False -> all rows refusal -> H=0.000.
# Fix: point CUDA_HOME+PATH+LD_LIBRARY_PATH at the wheel-bundled toolkit. No apt,
# no base change (smallest possible fix).
ENV NVCC_WHEEL_DIR=/usr/local/lib/python3.11/site-packages/nvidia/cu13
ENV CUDA_HOME=/usr/local/lib/python3.11/site-packages/nvidia/cu13
ENV PATH=/usr/local/lib/python3.11/site-packages/nvidia/cu13/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cu13/lib
# Sanity (now that PATH has nvcc): the whole toolkit must be 13.2 — nvcc 13.2.86
# AND the cudart headers it sees must also be 13.2 (CUDART_VERSION==13020), not the
# 13.0 headers accelerate's cuda-toolkit==13.0.2 dep left behind. flashinfer's
# bundled cccl checks CUDACC==CUDART at JIT; a mismatch fails vLLM init -> H=0.000.
# So fail the BUILD early if either the nvcc binary or the cudart header is <13.2.
RUN nvcc --version | grep "release 13.2" && \
    python3 -c "import re,sys; h=open('/usr/local/lib/python3.11/site-packages/nvidia/cu13/include/cuda_runtime_api.h').read(); v=re.search(r'#define CUDART_VERSION\s+(\d+)', h); print('CUDART', v.group(1) if v else 'NOT FOUND'); assert v and int(v.group(1))>=13020, 'cudart headers still <13.2 (accelerate reverted them)'" && \
    test -x "${NVCC_WHEEL_DIR}/bin/nvcc" && test -f "${NVCC_WHEEL_DIR}/include/cuda_runtime.h"

# flashinfer JIT hardcodes `-L$cuda_home/lib64` and `-L$cuda_home/lib64/stubs`
# then links `-lcudart -lcuda` (cpp_ext.py:254-255). But the nvidia-cuda-nvcc
# wheel ships the libs at `lib/` (lib/libcudart.so.13 etc.), NOT `lib64/`, and
# ships NO `libcuda.so` (the driver lib — only present at runtime via the
# nvidia container runtime bind-mount) and NO unversioned `libcudart.so`.
# Without lib64/ + the symlinks, the flashinfer ninja LINK step fails:
#   `collect2: error: ld returned 1 exit status` (can't find -lcudart/-lcuda)
# -> vLLM init fails -> vllm_ok=False -> all rows refusal -> H=0.000.
# Fix: create the lib64/ layout flashinfer expects, symlinking to the real
# libs in lib/, and a libcuda.so stub pointing at the host path the nvidia
# runtime bind-mounts at runtime (/usr/lib/x86_64-linux-gnu/libcuda.so).
# The libcuda.so symlink is dangling at BUILD time (no GPU here) but resolves
# at RUNTIME on the judge GPU; that's fine — the .so is only needed for the
# JIT link step, which runs at runtime when the driver IS mounted.
RUN set -e && \
    cd "${NVCC_WHEEL_DIR}" && \
    mkdir -p lib64/stubs && \
    for so in libcudart libcublas libcublasLt libcufft libcurand libcusolver libcusparse libnvrtc libnvJitLink; do \
        if [ -f "lib/${so}.so.13" ]; then \
            ln -sf "../lib/${so}.so.13" "lib64/${so}.so"; \
            ln -sf "../lib/${so}.so.13" "lib64/${so}.so.13"; \
        fi; \
    done && \
    ln -sf "/usr/lib/x86_64-linux-gnu/libcuda.so" lib64/stubs/libcuda.so && \
    echo "lib64 layout for flashinfer JIT link:" && ls -la lib64/ lib64/stubs/

# ninja sanity: flashinfer's JIT (top-k/top-p sampling + GDN kernels) shells out
# to the bare `ninja` executable via subprocess.run(["ninja",...]) at profile_run
# (flashinfer/jit/cpp_ext.py:run_ninja). vllm==0.24.0 depends on the `ninja` PyPI
# package, which installs the executable into <python-prefix>/bin/ninja =
# /usr/local/bin/ninja (on PATH here). If it is NOT found, vLLM init fails AFTER
# weights load -> vllm_ok=False -> all rows refusal -> H=0.000 (this exact
# false-failure was hit on the bare-metal smoke host). Fail the BUILD if the
# image lacks a resolvable ninja, so a ninja regression can never silently tank
# the judge run.
RUN command -v ninja && ninja --version

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
