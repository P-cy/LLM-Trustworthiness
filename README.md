# LLM Trustworthiness Challenge — Full Pipeline Image

Self-contained GPU image for the **AI Benchmark 2026 LLM Trustworthiness Challenge**.
Two baked-in offline models classify each query by safety, generate a
category-appropriate answer, verify it, and repair if needed — all within a
hard time budget, fully offline at runtime.

## Models (baked in, runtime is offline)

| Role | Model | Size | Runtime |
|---|---|---|---|
| **Main** (answerer + critic) | `RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16` (int4) | ~16GB | vLLM (in-process, no API server) |
| **Guard** (input/output safety) | `Qwen/Qwen3Guard-Gen-4B` (fp16) | ~8GB | HuggingFace transformers |

vLLM **0.24.0**, transformers ≥4.51. Runtime GPU target: **H100 40GB**.

## Pipeline (per query, batched)

```
query
 ├─[prefilter]  regex: base64/hex/leetspeak/homoglyph + jailbreak keywords
 │              (a NUDGE, not a verdict)
 ├─[triage]     guard (3-level: safe/controversial/unsafe) + critic
 │              -> one of: H_clear | H_sensitive_safe | S_harmful | S_jailbreak
 │              safety-first: guard "unsafe" forces stricter; "controversial" does NOT
 ├─[generate]   H_clear/H_sensitive_safe -> single-shot, temp 0
 │              S_harmful/S_jailbreak/ambiguous -> best-of-3 (temp 0.7, distinct seeds)
 │              per-category system prompt; query wrapped as DATA; policy.md attached
 ├─[verify]     guard + critic -> PASS | FIX_SAFETY | FIX_HELPFULNESS
 ├─[repair]     regenerate flagged answers, max 2 rounds (deterministic temp 0)
 └─[finalize]   strip CoT/role markers, collapse newlines, write answer
                FIX_SAFETY -> drop to safe refusal; empty -> category+language fallback
```

### Safety-first choices
- **Guard 3-level.** `unsafe` forces stricter; `controversial` does NOT force
  (prevents over-refusal on risky-but-safe questions).
- **Prefilter is a nudge.** Jailbreak keywords escalate only when guard didn't
  say safe/controversial AND critic didn't pick H_* — so a security-course
  question containing "bypass" is NOT auto-refused.
- **Best-of-3** for S_harmful/S_jailbreak/ambiguous: 3 candidates at temp 0.7
  with distinct seeds, scored by guard+critic, best picked. H_clear stays
  single-shot temp 0.
- **Repair** is a final fallback (max 2 rounds, deterministic).
- **Determinism** via `TRUSTW_SEED=1337`, temp 0 default (except best-of-3).
- **No half-answers.** S_harmful prompt forbids "refuse then give steps";
  verify flags `leaked_harmful`; finalizer drops to refusal if still unsafe.
- **Watchdog.** 27-min budget; under 3 min left -> stop, emit fallbacks.
- **Always calls `/benchmark_lib/progress <n>` in `finally`.**

## Layout

```
testllm/
├── run.py                 # entrypoint: VRAM assert + read -> pipeline -> write -> progress(finally)
├── entrypoint.sh          # applies CONFIG_A runtime env, then exec run.py
├── src/
│   ├── io_csv.py          # postprocess (strip CoT), lang detect, fallbacks
│   ├── prefilter.py       # regex: base64/hex/leetspeak/homoglyph + jb keywords
│   ├── guard.py           # Qwen3Guard (3-level: safe/controversial/unsafe)
│   ├── vllm_engine.py     # vLLM LLM(), prefix caching, seed determinism, best-of-N
│   ├── triage.py          # 4-category (guard+critic, unsafe forces, prefilter nudges)
│   ├── generate.py        # single-shot (H) + best-of-3 (S/ambiguous), per-category prompts
│   ├── verify.py          # guard+critic -> PASS/FIX_SAFETY/FIX_HELPFULNESS + score_candidates
│   ├── repair.py          # regenerate flagged, max 2 rounds, deterministic
│   └── pipeline.py        # orchestrator + watchdog + stage deadlines + logging
├── prompts/               # triage, critic, 4 categories, safe_completion, policy.md
├── Dockerfile             # 2 stages: download weights (HF) + runtime (offline)
├── .dockerignore
├── build_push_linux.sh    # build+push on a Linux host (native amd64)
├── mac_build_push.sh      # (alt) build+push from Mac
├── mac_smoke_test.sh      # 0.6B CPU smoke test (plumbing only)
├── scripts/
│   ├── bake_models.sh     # download weights into ./models (alt to build-time download)
│   └── quantize_w4a16.sh  # OPTIONAL: quantize to INT4 with llm-compressor
└── test/
    ├── test_offline.sh     # --network none, assert ids/order/case/empty/no-leaked-tags
    ├── test_stress.sh      # 2,000 rows, must finish <27min, logs peak VRAM
    ├── test_csv_edge.sh    # case-mixed ids, comma/quote/newline/emoji -> valid CSV
    ├── local_test.sh       # (older) generic local test
    └── progress_stub       # stand-in for /benchmark_lib/progress
```

## Runtime contract (judge mounts these)

```
docker run --network none --gpus all \
  -v /model/test:/model/test:ro -v /result:/result -v /benchmark_lib:/benchmark_lib:ro \
  registry.ai.in.th/2026-trustworthy/cf39c3a7/masterpp.ymjz:<tag>
```
- Reads `/model/test/dataset.csv` (id, query)
- Writes `/result/submission.csv` (id, response), ids verbatim, order, non-empty
- Calls `/benchmark_lib/progress <n>` in `finally`
- Env: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (forced in Dockerfile + run.py)

## Build

### Prerequisites
- A host where `docker build` works (Linux amd64 native is fastest; Mac via QEMU works but slower)
- Internet at **build time** (weights download from HuggingFace)
- No GPU needed to *build*, but needed to *run/test*

### Build (CONFIG_A = int4, default, recommended for H100 40GB)

```bash
cd /path/to/testllm
docker build --platform linux/amd64 -t trustworthy-full:local .
# Optional HF token to avoid rate limits:
# docker build --platform linux/amd64 --build-arg HF_TOKEN=hf_xxx -t trustworthy-full:local .
```
This downloads ~24GB (int4 main ~16GB + guard ~8GB) from HuggingFace and bakes them into the image.

### Build push script (Linux host)
```bash
CONFIG=A bash build_push_linux.sh
#   -> registry.ai.in.th/2026-trustworthy/cf39c3a7/masterpp.ymjz:full-int4
```

### Test before pushing (needs GPU)
```bash
IMAGE=trustworthy-full:local bash test/test_offline.sh
IMAGE=trustworthy-full:local bash test/test_csv_edge.sh
IMAGE=trustworthy-full:local N=2000 bash test/test_stress.sh
```

### Smoke test (plumbing only, on Mac, no GPU)
```bash
bash mac_smoke_test.sh   # builds a 0.6B CPU image, validates end-to-end plumbing
```

## Push to the registry

> Real submission uses `2026-trustworthy` (no `-sample`).
> Sandbox tests use `2026-trustworthy-sample`.

```bash
# sandbox first (test on dashboard)
docker tag trustworthy-full:local registry.ai.in.th/2026-trustworthy-sample/cf39c3a7/masterpp.ymjz:full-int4
docker login registry.ai.in.th          # usercode: masterpp.ymjz
docker push registry.ai.in.th/2026-trustworthy-sample/cf39c3a7/masterpp.ymjz:full-int4
docker logout registry.ai.in.th

# after sandbox passes -> real submission
docker tag trustworthy-full:local registry.ai.in.th/2026-trustworthy/cf39c3a7/masterpp.ymjz:full-int4
docker login registry.ai.in.th
docker push registry.ai.in.th/2026-trustworthy/cf39c3a7/masterpp.ymjz:full-int4
docker logout registry.ai.in.th
```

Pushing 24GB is slow — use `nohup` to avoid losing it on a dropped terminal:
```bash
nohup docker push registry.ai.in.th/2026-trustworthy/cf39c3a7/masterpp.ymjz:full-int4 > push.log 2>&1 &
tail -f push.log
```
On `broken pipe` just re-run the same `docker push` — layers already uploaded are
reused (push resumes).

## Env knobs

| Var | Default | Meaning |
|-----|---------|---------|
| `TRUSTW_BUDGET_S` | 1620 (27min) | hard time cap |
| `TRUSTW_WATCHDOG_S` | 180 (3min) | stop generating below this remaining |
| `TRUSTW_SEED` | 1337 | determinism seed |
| `VLLM_GPU_MEM_UTIL` | 0.68 | vLLM VRAM fraction |
| `VLLM_MAX_MODEL_LEN` | 8192 | context |
| `VLLM_MAX_SEQS` | 32 | batching width |
| `VLLM_DEVICE` | cuda | set `cpu` for smoke test |
| `HF_TOKEN` | (build arg) | optional, faster HF downloads |

## Notes
- Weights are **NOT** in this git repo (`models/` is gitignored). They are
  downloaded from HuggingFace at build time.
- If the pre-quantized int4 checkpoint doesn't load well in vLLM at runtime,
  `scripts/quantize_w4a16.sh` can produce a local INT4 via llm-compressor.
- Runtime must be offline; build must be online.
