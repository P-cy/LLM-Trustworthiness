#!/usr/bin/env bash
# build_push_linux.sh: build the image on a Linux host (linux/amd64 native)
# and push to the registry. Run this on a Linux box where `docker build` works
# (has CAP_SYS_ADMIN). The image is ~20-35GB so building on Linux (native
# amd64, no QEMU emulation) is far faster than the Mac, and the Mac upload is
# avoided entirely.
#
# PREREQ: this host has internet (weights download from HF at build time) and
# Docker with build working. If you're on the B200 box where `docker build`
# fails (seccomp blocks unshare/mount), run this on a different Linux host, or
# fix the B200 container's seccomp profile first.
#
# Usage (on the Linux build host, from the testllm dir):
#   CONFIG=A bash build_push_linux.sh
#   CONFIG=B bash build_push_linux.sh
#
# Env:
#   CONFIG     A (int4 main, default) or B (FP8 main + 0.6B guard)
#   REG_TAG    tag (default: full-int4 for A, full-fp8 for B)
#   REG_USER   registry usercode (default: masterpp.ymjz)
#   REG_PATH   2026-trustworthy (real) or 2026-trustworthy-sample (sandbox)
#   HF_TOKEN   optional HF token for faster/uncapped downloads
set -euo pipefail

CONFIG="${CONFIG:-A}"
REG_USER="${REG_USER:-masterpp.ymjz}"
REG_PATH="${REG_PATH:-2026-trustworthy}"
REG_TAG="${REG_TAG:-$([ "$CONFIG" = "B" ] && echo full-fp8 || echo full-int4)}"
REG_FULL="registry.ai.in.th/${REG_PATH}/cf39c3a7/${REG_USER}:${REG_TAG}"
IMAGE_LOCAL="trustworthy-full:local"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
echo "==> building in $HERE (CONFIG=$CONFIG)"
ls -la

echo "==> docker build (linux/amd64, native). Downloads weights from HF."
BUILD_ARGS=(--platform linux/amd64 --build-arg TRUSTW_CONFIG="$CONFIG" -t "$IMAGE_LOCAL")
if [[ -n "${HF_TOKEN:-}" ]]; then
  BUILD_ARGS+=(--build-arg HF_TOKEN="$HF_TOKEN")
fi
docker build "${BUILD_ARGS[@]}" .

echo "==> tag -> $REG_FULL"
docker image tag "$IMAGE_LOCAL" "$REG_FULL"

echo "==> login + push + logout"
docker login registry.ai.in.th
docker image push "$REG_FULL"
docker logout registry.ai.in.th

echo
echo "DONE: $REG_FULL"
echo "Submit via dashboard: Select Images -> '$REG_TAG' -> submit."
