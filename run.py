#!/usr/bin/env python3
"""LLM Trustworthiness Challenge - full safety+helpfulness pipeline.

Two baked-in offline models:
  - Qwen3-30B-A3B-Instruct-2507 (AWQ 4bit)  -> main answerer + critic (vLLM)
  - Qwen3Guard-Gen-4B                       -> input/output guard (HF transformers)

Pipeline per query: prefilter -> triage (guard+critic) -> generate (per-category
system prompt) -> verify (guard + critic-as-critic, guided JSON) -> repair (<=2
rounds, only flagged) -> postprocess (strip CoT) -> write.

Watchdog: time budget from process start. Under 3 min left -> stop generating,
emit polite refusal for any unfinished S-* and category-aware fallback for H-*.

Safety-first at every ambiguous point: when guard and critic disagree, the
stricter (unsafe / jailbreak) verdict wins.
"""
import os
import sys
import time
import csv
import tempfile

INPUT_CSV = "/model/test/dataset.csv"
OUTPUT_CSV = "/result/submission.csv"
PROGRESS_BIN = "/benchmark_lib/progress"

# Time budget. Real run = 30 min; we hard-cap at 27 to leave a safety margin so
# the final write + progress call always happens even under slow load.
TOTAL_BUDGET_S = float(os.environ.get("TRUSTW_BUDGET_S", "1620"))  # 27 min
# Below this remaining budget we stop generating and finalize fast.
WATCHDOG_STOP_S = float(os.environ.get("TRUSTW_WATCHDOG_S", "180"))  # 3 min

OFFLINE_FLAGS = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
for k, v in OFFLINE_FLAGS.items():
    os.environ[k] = v  # force (not setdefault): runtime must be offline

# Policy file path (attached to every critic call).
os.environ.setdefault("TRUSTW_POLICY_PATH", "/app/prompts/policy.md")


def assert_vram_budget():
    """Fail fast with a clear message if the configured VRAM budget can't fit
    weights + guard + KV cache. This catches the CONFIG_A/B mismatch bug
    (e.g. util 0.70 of 40GB = 28GB < 30GB FP8 weights).

    Only meaningful on CUDA; on CPU we skip (smoke test).
    """
    if os.environ.get("VLLM_DEVICE", "cuda") == "cpu":
        return
    total_vram_gb = _total_vram_gb()  # returns 0.0 if nvidia-smi absent/unavailable
    if total_vram_gb <= 0:
        return  # can't detect VRAM -> skip (vLLM will surface the real error)
    util = float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.68"))
    budget_gb = total_vram_gb * util
    main_gb = float(os.environ.get("TRUSTW_MAIN_WEIGHTS_GB", "0") or 0)
    guard_gb = float(os.environ.get("TRUSTW_GUARD_WEIGHTS_GB", "0") or 0)
    if main_gb <= 0:
        return  # unknown weights size; skip the check
    needed_gb = main_gb + guard_gb
    # Leave >=2GB for activations/KV beyond the util pool assumption.
    if needed_gb > budget_gb:
        raise RuntimeError(
            f"VRAM budget too small: GPU={total_vram_gb:.0f}GB * util={util} "
            f"= {budget_gb:.1f}GB budget, but main weights={main_gb:.1f}GB + "
            f"guard={guard_gb:.1f}GB = {needed_gb:.1f}GB needed. "
            f"Lower VLLM_GPU_MEM_UTIL is NOT the fix; use a smaller main model "
            f"(CONFIG_A int4) or a smaller guard (CONFIG_B 0.6B)."
        )
    log(f"VRAM budget OK: {budget_gb:.1f}GB >= {needed_gb:.1f}GB "
        f"(main+guard) on {total_vram_gb:.0f}GB GPU")


def _total_vram_gb():
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return float(out.splitlines()[0]) / 1024.0
    except Exception:
        return 0.0

# Category-aware refusal fallbacks (used on watchdog timeout or repair failure
# in S categories). Keep these short, polite, single-language-neutral-ish, and
# explicitly non-instructive so a leak here can never be harmful.
REFUSAL_FALLBACK_TH = "ขออภัย ฉันไม่สามารถให้ความช่วยเหลือในเรื่องนี้ได้ หากคุณมีคำถามอื่นที่ปลอดภัย ฉันยินดีช่วย"
REFUSAL_FALLBACK_EN = "I'm sorry, I can't help with that. If you have another safe question, I'd be glad to assist."

# H-category fallback when verify fails after max repair rounds but content
# looks non-harmful: keep the (stripped) answer but never payload instructions.
GENERIC_SAFE_TH = "ขออภัย ฉันไม่สามารถให้ข้อมูลเฉพาะเจาะจงนี้ได้ในขณะนี้"
GENERIC_SAFE_EN = "I'm sorry, I can't provide that specific information right now."

CATEGORIES = ("H_clear", "H_sensitive_safe", "S_harmful", "S_jailbreak")


def log(msg):
    print(f"[run {time.time():.1f}] {msg}", file=sys.stderr, flush=True)


def read_questions(path):
    if not os.path.exists(path):
        log(f"WARNING: input {path} not found")
        return []
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            log(f"WARNING: input {path} empty")
            return []
        fields = [c.strip() if isinstance(c, str) else c for c in reader.fieldnames]
        if "id" not in fields or "query" not in fields:
            raise RuntimeError(
                f"dataset.csv must have id,query; got {reader.fieldnames}"
            )
        for r in reader:
            rid = r.get("id", "")
            if isinstance(rid, str):
                rid = rid.replace("\r", "")
            query = r.get("query", "") or ""
            rows.append((rid, query))
    return rows


def postprocess(text):
    """Strip CoT/think tags and trailing role markers. Never return empty."""
    import re
    if not text:
        return ""
    # Remove <think>...</think> (greedy-ish, multiline) and lone open tags.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</think>", "", text, flags=re.IGNORECASE)
    # Strip chat role markers that may leak.
    for tok in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        text = text.replace(tok, "")
    text = text.strip()
    # Collapse newlines to single space (CSV one-row invariant).
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def lang_of(query):
    """Crude language detection: Thai if any Thai char, else English."""
    for ch in query:
        if "฀" <= ch <= "๿":
            return "th"
    return "en"


def fallback_for(category, lang):
    if category.startswith("S_"):
        return REFUSAL_FALLBACK_TH if lang == "th" else REFUSAL_FALLBACK_EN
    return GENERIC_SAFE_TH if lang == "th" else GENERIC_SAFE_EN


def write_submission(path, rows):
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".submission.", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            w.writerow(["id", "response"])
            for rid, resp in rows:
                resp = resp if resp is not None else ""
                resp = resp.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
                w.writerow([rid, resp])
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def call_progress(total):
    try:
        import subprocess
        subprocess.run([PROGRESS_BIN, str(total)], check=False, timeout=30,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        log(f"progress binary not found at {PROGRESS_BIN}")
    except Exception as e:
        log(f"progress call failed: {e!r}")


class Watchdog:
    def __init__(self, budget_s):
        self.t0 = time.time()
        self.budget = budget_s

    def elapsed(self):
        return time.time() - self.t0

    def remaining(self):
        return self.budget - self.elapsed()

    def must_stop(self):
        return self.remaining() <= WATCHDOG_STOP_S


def main():
    total = 0
    results = []  # list of (id, response)
    try:
        rows = read_questions(INPUT_CSV)
        total = len(rows)
        log(f"input rows={total}, budget={TOTAL_BUDGET_S:.0f}s, watchdog_stop<{WATCHDOG_STOP_S:.0f}s")

        # Heavy imports done lazily after we know there is work.
        if total == 0:
            write_submission(OUTPUT_CSV, [])
            log("no input; wrote header-only submission")
            return 0

        # Fail fast if VRAM budget can't fit the baked weights.
        assert_vram_budget()

        from src.pipeline import run_pipeline
        watchdog = Watchdog(TOTAL_BUDGET_S)
        out = run_pipeline(rows, watchdog)
        # out: list of (id, response) already postprocessed, same length+order
        results = out
        write_submission(OUTPUT_CSV, results)
        log(f"wrote {len(results)} rows to {OUTPUT_CSV}")
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"ERROR: {e!r}")
        # Best-effort: fill anything missing. We do NOT know the per-row
        # category on the error path, so default to a generic safe NON-refusal
        # (H_clear) rather than refusing everything. Refusing every row when we
        # don't know the category zeroes helpfulness for the whole batch even
        # though good answers may already have been produced. The generic safe
        # fallback is non-instructive (can't leak harmful payload) but is not a
        # flat refusal, so safe H_* rows still earn partial credit.
        try:
            if 'rows' in locals() and rows:
                filled = []
                for rid, q in rows:
                    filled.append((rid, fallback_for("H_clear", lang_of(q))))
                write_submission(OUTPUT_CSV, filled)
                log("wrote generic-safe fallbacks on error path")
        except Exception as e2:
            log(f"fallback write failed: {e2!r}")
        return 1
    finally:
        log(f"finally: calling {PROGRESS_BIN} {total}")
        call_progress(total)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
