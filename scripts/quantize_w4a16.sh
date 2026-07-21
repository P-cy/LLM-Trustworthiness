#!/usr/bin/env bash
# OPTIONAL: quantize Qwen3-30B-A3B-Instruct-2507 to INT4 w4a16 with
# llm-compressor, to run on the B200 box (has GPU + internet). Use this ONLY if
# the RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16 (the default CONFIG_A
# main model) turns out unsuitable at runtime. Otherwise skip this script.
#
# Output: ./models/Qwen3-30B-A3B-Instruct-2507-w4a16-local/
#
# Run on the B200 box (GPU + internet):
#   bash scripts/quantize_w4a16.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="$ROOT/models/Qwen3-30B-A3B-Instruct-2507-w4a16-local"
SRC="Qwen/Qwen3-30B-A3B-Instruct-2507"

pip install -q "llmcompressor>=0.16" "compressed-tensors" "transformers" accelerate

python3 - "$SRC" "$OUT" <<'PY'
import sys
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

src, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, torch_dtype="auto", device_map="auto")
# w4a16: 4-bit weights, 16-bit activations. Ignore MoE router/experts not in
# the ignore list; llmcompressor handles MoE. Calibrate on a small set.
recipe = GPTQModifier(targets="Linear", scheme="w4a16", ignore=["lm_head"])
oneshot(model=model, recipe=recipe, output_dir=out)
tok.save_pretrained(out)
print("quantized to", out)
PY

echo "==> done: $OUT"
du -sh "$OUT"
echo "Set MAIN_MODEL_DIR=$OUT (and build with that path) to use this."
