"""vLLM offline engine wrapper (LLM() in-process, no API server).

Config A/B via env. Prefix caching ON (system prompts repeat). Determinism via
fixed seed; best-of-3 passes a higher temperature+different seeds.

Offline is asserted: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE set in Dockerfile; we
also confirm the model dir exists locally and fail fast with a clear message.
"""
import os
import sys
import json
import re


def _assert_offline():
    missing = [k for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
              if os.environ.get(k) != "1"]
    if missing:
        print(f"[vllm] WARNING: offline flags not set: {missing} "
              f"(runtime should be offline)", file=sys.stderr, flush=True)


class VLLMEngine:
    MODEL_DIR = os.environ.get("MAIN_MODEL_DIR", "/models/main")

    def __init__(self):
        self.llm = None
        self.ready = False
        self._SamplingParams = None
        self._seed = int(os.environ.get("TRUSTW_SEED", "1337"))

    def load(self):
        _assert_offline()
        try:
            from vllm import LLM, SamplingParams
        except Exception as e:
            print(f"[vllm] import failed: {e!r}", file=sys.stderr, flush=True)
            return False
        if not os.path.isdir(self.MODEL_DIR):
            print(f"[vllm] FAIL FAST: model dir not found: {self.MODEL_DIR}. "
                  f"Image must bake weights (offline).", file=sys.stderr, flush=True)
            return False

        quant = os.environ.get("VLLM_QUANT", "") or None
        device = os.environ.get("VLLM_DEVICE", "cuda")
        try:
            kwargs = dict(
                model=self.MODEL_DIR,
                dtype=os.environ.get("VLLM_DTYPE", "auto"),
                trust_remote_code=True,
                enforce_eager=(os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1"),
                max_num_seqs=int(os.environ.get("VLLM_MAX_SEQS", "32")),
                disable_log_stats=True,
                # Prefix caching: system prompts repeat across the batch.
                enable_prefix_caching=True,
            )
            if device == "cpu":
                kwargs["device"] = "cpu"
                kwargs["max_model_len"] = int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096"))
            else:
                kwargs["gpu_memory_utilization"] = float(
                    os.environ.get("VLLM_GPU_MEM_UTIL", "0.68"))
                kwargs["max_model_len"] = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
            if quant:
                kwargs["quantization"] = quant
            # --- Qwen3.5-35B-A3B-GPTQ-Int4 specific (Path C), env-gated, OFF by default ---
            # Only set when the env opt-in is present, so the existing int4 path
            # (Qwen3-30B-A3B) is unaffected. All three are forwarded by LLM(**kwargs)
            # -> EngineArgs in vLLM 0.24 (verified in the Step 0 smoke test).
            #   - mamba_cache_mode: Qwen3.5 is a hybrid GDN model; raises
            #     NotImplementedError on the default "all" -> must use "align".
            #   - language_model_only: skip the vision tower (dead weight for a
            #     text benchmark; saves VRAM). Qwen3.5 is multimodal.
            #   - limit_mm_per_prompt={}: belt-and-suspenders so a stray image
            #     token can't OOM the vision path.
            mamba_cache_mode = os.environ.get("VLLM_MAMBA_CACHE_MODE", "").strip() or None
            if mamba_cache_mode:
                kwargs["mamba_cache_mode"] = mamba_cache_mode
            if os.environ.get("VLLM_LANGUAGE_MODEL_ONLY", "0") == "1":
                kwargs["language_model_only"] = True
                # Keep image budget at zero by default for a text-only benchmark.
                kwargs.setdefault("limit_mm_per_prompt", {})
            print(f"[vllm] loading model from {self.MODEL_DIR}, device={device}, "
                  f"quant={quant!r}, gpu_mem_util={kwargs.get('gpu_memory_utilization')}, "
                  f"max_model_len={kwargs.get('max_model_len')}, "
                  f"max_seqs={kwargs['max_num_seqs']}, prefix_cache=on, "
                  f"mamba_cache_mode={kwargs.get('mamba_cache_mode')!r}, "
                  f"lang_only={kwargs.get('language_model_only')}",
                  file=sys.stderr, flush=True)
            self.llm = LLM(**kwargs)
            self._SamplingParams = SamplingParams
            self.ready = True
            print(f"[vllm] loaded (device={device})", file=sys.stderr, flush=True)
            return True
        except Exception as e:
            print(f"[vllm] load failed: {e!r}", file=sys.stderr, flush=True)
            self.ready = False
            return False

    def chat_batch(self, messages_list, temperature=0.0, top_p=1.0,
                   max_tokens=1024, seed=None, batch_size=None):
        """Batched chat. Determinism: temperature 0 + fixed seed by default.

        seed=None -> uses the global TRUSTW_SEED. best-of-3 passes explicit
        per-candidate seeds + temperature 0.7 for diversity.

        NOTE: we pass enable_thinking=False via chat_template_kwargs as a
        belt-and-suspenders guard against any model variant that injects a
        thinking block. The currently baked model (Qwen3-30B-A3B-Instruct-2507)
        is the dedicated NON-THINKING variant whose chat template has zero
        thinking logic, so this kwarg is a no-op for it. It is NOT the fix for
        the H=0.000 / S~0.93 result — that requires the H100 runtime logs
        (most likely a vLLM load failure -> ready=False -> all-empty -> refusal
        fallbacks). Set VLLM_ENABLE_THINKING=1 to re-enable if ever needed.
        """
        if not self.ready:
            return [""] * len(messages_list)
        s = self._seed if seed is None else int(seed)
        sp = self._SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=s,
        )
        enable_thinking = os.environ.get("VLLM_ENABLE_THINKING", "0") == "1"
        try:
            outs = self.llm.chat(
                messages_list, sp,
                chat_template_kwargs={"enable_thinking": enable_thinking})
        except TypeError:
            # Older vLLM without chat_template_kwargs in llm.chat(): fall back to
            # the plain call so generation still works (thinking may stay on,
            # but a working-but-truncated answer beats a hard crash to empty).
            outs = self.llm.chat(messages_list, sp)
        except Exception as e:
            print(f"[vllm] chat_batch failed: {e!r}", file=sys.stderr, flush=True)
            return [""] * len(messages_list)
        texts = []
        for o in outs:
            try:
                texts.append(_strip_think(o.outputs[0].text))
            except Exception:
                texts.append("")
        return texts

    def chat_guided_json(self, messages_list, schema, temperature=0.0,
                         top_p=1.0, max_tokens=512, seed=None):
        outs = self.chat_batch(messages_list, temperature=temperature, top_p=top_p,
                               max_tokens=max_tokens, seed=seed)
        return [_extract_json(t) for t in outs]

    def chat_candidates(self, messages_list, n=3, temperature=0.7,
                        top_p=0.9, max_tokens=768, base_seed=None):
        """Best-of-N: generate n candidates per item with distinct seeds.

        Returns list[list[str]]: outer = items, inner = n candidates.
        Implementation: run n passes (one per seed) so prefix caching helps
        across passes and seeds give diversity at temp>0.
        """
        if not self.ready:
            return [["" for _ in range(n)] for _ in range(len(messages_list))]
        b = self._seed if base_seed is None else int(base_seed)
        all_cands = [[] for _ in messages_list]
        for k in range(n):
            outs = self.chat_batch(messages_list, temperature=temperature,
                                   top_p=top_p, max_tokens=max_tokens,
                                   seed=b + k * 101)
            for i, t in enumerate(outs):
                all_cands[i].append(t)
        return all_cands


def _strip_think(text):
    """Defense-in-depth: drop any leaked reasoning/think block from raw model
    output BEFORE it reaches verify/repair (which call chat_guided_json and
    read raw text before the final io_csv.postprocess).

    Qwen3.5's chat template closes the think block before generation when
    enable_thinking=False (the branch we activate via chat_template_kwargs),
    so the model SHOULD write straight to the answer. But vLLM issue #35574
    shows a model can still emit a reasoning trace into the answer block. This
    net catches that case: drop a closed <think>...</think> span, and if an
    unclosed <think> leaked, drop from the open tag to end. The final
    io_csv.postprocess strips markers again, so this is belt-and-suspenders.
    """
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</think>", "", text, flags=re.IGNORECASE)
    return text


def _extract_json(text):
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    cand = m.group(1) if m else None
    if cand is None:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        cand = m.group(0) if m else None
    if cand is None:
        return None
    try:
        return json.loads(cand)
    except Exception:
        return None
