#!/usr/bin/env bash
# Pre-download model weights into the build context so the Docker image is
# fully self-contained (no internet at runtime).
#
# Run this on the BUILD host (the Linux machine with internet + enough disk),
# BEFORE `docker build`. It populates ./models/ in the repo root.
#
# Requirements on the build host:
#   - huggingface-cli (pip install -U "huggingface_hub[cli]")
#   - internet
#   - disk: ~16GB (AWQ 30B) + ~8GB (Guard 4B)
#
# Env overrides:
#   MAIN_REPO  (default Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ)
#   GUARD_REPO (default Qwen/Qwen3Guard-Gen-4B)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
MODELS="$ROOT/models"

MAIN_REPO="${MAIN_REPO:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
GUARD_REPO="${GUARD_REPO:-Qwen/Qwen3Guard-Gen-4B}"

mkdir -p "$MODELS"

echo "==> ensuring hf CLI"
command -v hf >/dev/null || pip install -U "huggingface_hub[cli]"

export HF_HUB_ENABLE_HF_TRANSFER=1

echo "==> downloading main model (FP8): $MAIN_REPO"
hf download "$MAIN_REPO" \
  --local-dir "$MODELS/Qwen3-30B-A3B-Instruct-2507-FP8"

echo "==> downloading guard model: $GUARD_REPO"
hf download "$GUARD_REPO" \
  --local-dir "$MODELS/Qwen3Guard-Gen-4B"

echo "==> done. models at $MODELS"
du -sh "$MODELS"/*
