#!/usr/bin/env bash
# Run this ON YOUR MAC. Only the CODE tarball (small, ~50KB) is needed from the
# B200 host -- weights are downloaded from HuggingFace at build time, which is
# much faster than scp'ing 38GB over SSH.
#
# Steps:
#   1. untar the code into ~/Downloads/testllm_build/testllm/
#   2. docker build (linux/amd64) - downloads ~38GB weights from HF, bakes them in
#   3. tag for the REAL registry (no -sample)
#   4. login + push + logout
#
# Usage on Mac (two options):
#   A) From inside the testllm dir (after scp):
#        cd ~/Downloads/2026-trustworthy/testllm
#        bash mac_build_push.sh
#   B) From anywhere with a code tarball:
#        bash mac_build_push.sh /path/to/testllm_code.tar
#
# Env (optional):
#   CONFIG     A (int4 main, default) or B (FP8 main + 0.6B guard)
#   REG_TAG    tag name (default: full-int4 for A, full-fp8 for B)
#   REG_USER   registry usercode (default: masterpp.ymjz)
#   REG_PATH   2026-trustworthy (real) or 2026-trustworthy-sample (sandbox)
#   HF_TOKEN   optional HF token for higher download rate limits
set -euo pipefail

CONFIG="${CONFIG:-A}"
REG_USER="${REG_USER:-masterpp.ymjz}"
REG_PATH="${REG_PATH:-2026-trustworthy}"   # REAL submission (no -sample)
REG_TAG="${REG_TAG:-$([ "$CONFIG" = "B" ] && echo full-fp8 || echo full-int4)}"
IMAGE_LOCAL="trustworthy-full:local"
REG_FULL="registry.ai.in.th/${REG_PATH}/cf39c3a7/${REG_USER}:${REG_TAG}"

if [[ $# -ge 1 && -f "$1" ]]; then
  echo "==> untar code from $1"
  BUILD_DIR="$(mktemp -d)"
  tar -xf "$1" -C "$BUILD_DIR"
  cd "$BUILD_DIR/testllm"
else
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  echo "==> using project at $HERE"
  cd "$HERE"
fi
echo "    contents:"; ls -la

echo "==> [2/4] docker build (linux/amd64)"
echo "    This DOWNLOADS ~38GB from HuggingFace and bakes weights into the image."
echo "    May take 10-30 min depending on your internet. Make sure Docker Desktop is up:"
echo "      open -a Docker   (if not already)"
echo "    If HF download is slow/rate-limited, set: export HF_TOKEN=hf_xxx  and rerun."
# Pass HF_TOKEN through to the build so HF downloads are faster if you have one.
BUILD_ARGS=(--platform linux/amd64 --build-arg TRUSTW_CONFIG="$CONFIG" -t "$IMAGE_LOCAL")
if [[ -n "${HF_TOKEN:-}" ]]; then
  BUILD_ARGS+=(--build-arg HF_TOKEN="$HF_TOKEN")
fi
docker build "${BUILD_ARGS[@]}" .

echo "==> [3/4] tag -> $REG_FULL"
docker image tag "$IMAGE_LOCAL" "$REG_FULL"

echo "==> [4/4] login + push + logout"
docker login registry.ai.in.th
#   username: masterpp.ymjz   (then password)
docker image push "$REG_FULL"
docker logout registry.ai.in.th

echo
echo "DONE. Image pushed: $REG_FULL"
echo "Now go to the dashboard, Select Images -> pick '$REG_TAG', submit."
