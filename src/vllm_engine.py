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
            print(f"[vllm] loading model from {self.MODEL_DIR}, device={device}, "
                  f"quant={quant!r}, gpu_mem_util={kwargs.get('gpu_memory_utilization')}, "
                  f"max_model_len={kwargs.get('max_model_len')}, "
                  f"max_seqs={kwargs['max_num_seqs']}, prefix_cache=on",
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
        try:
            outs = self.llm.chat(messages_list, sp)
        except Exception as e:
            print(f"[vllm] chat_batch failed: {e!r}", file=sys.stderr, flush=True)
            return [""] * len(messages_list)
        texts = []
        for o in outs:
            try:
                texts.append(o.outputs[0].text)
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
