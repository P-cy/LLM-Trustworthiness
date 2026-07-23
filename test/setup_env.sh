#!/usr/bin/env bash
# setup_env.sh — install the minimal ML stack to run the guard probe + (optionally)
# the full pipeline on a bare-metal GPU host (no docker needed).
#
# What it installs:
#   - python3.11 venv at ./.venv_guard (reusable for both guard probe + pipeline)
#   - torch (CUDA build matching the host), transformers>=4.51, accelerate,
#     sentencepiece, pydantic, numpy<2
#   - (OPTIONAL, set INSTALL_VLLM=1) vllm==0.24.0 — only needed for the FULL
#     pipeline test (uses the 30B main model). Skip for the guard-only probe.
#
# Models are NOT included — point MAIN_MODEL_DIR / GUARD_MODEL_DIR at wherever
# you have them on the host (defaults: ./models/Qwen3-30B-A3B-Instruct-2507-FP8
# and ./models/Qwen3Guard-Gen-4B).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

echo "==> creating venv (.venv_guard) with python3.11"
/usr/bin/python3.11 -m venv .venv_guard
. .venv_guard/bin/activate
pip install --quiet --upgrade pip

echo "==> installing torch + transformers + accelerate + sentencepiece"
pip install --quiet "torch" "transformers>=4.51" "accelerate" "sentencepiece" "pydantic" "numpy<2"

if [[ "${INSTALL_VLLM:-0}" == "1" ]]; then
  echo "==> INSTALL_VLLM=1 -> installing vllm==0.24.0 (needed for FULL pipeline test)"
  pip install --quiet "vllm==0.24.0" "compressed-tensors>=0.9.0"
fi

echo
echo "==> verify imports"
python -c "
import torch, transformers, accelerate
print(f'torch {torch.__version__} cuda={torch.cuda.is_available()} gpus={torch.cuda.device_count()}')
print(f'transformers {transformers.__version__}')
try:
    import vllm; print(f'vllm {vllm.__version__}')
except ImportError:
    print('vllm: not installed (ok for guard-only test)')
"
echo
echo "setup_env.sh DONE."
echo "Next:"
echo "  guard probe:   python _guard_probe.py"
echo "  full pipeline:  python -m src.pipeline ...  (or set up a run.py wrapper)"
