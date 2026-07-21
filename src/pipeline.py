"""Pipeline orchestrator: triage -> generate (single-shot + best-of-3) ->
verify -> repair (final fallback), with batching + per-stage time budgets.

Safety-first: guard 'unsafe' forces stricter verdicts; 'controversial' does
not. Time budget split so no stage starves the final write.
"""
import time
from .io_csv import postprocess, lang_of, fallback_for
from .prefilter import prefilter
from .guard import GuardClient
from .vllm_engine import VLLMEngine


def _log(msg):
    import sys
    print(f"[pipe {time.time():.0f}] {msg}", file=sys.stderr, flush=True)


def _cat_counts(cats):
    from .triage import CATS
    return ", ".join(f"{c}={cats.count(c)}" for c in CATS)


def run_pipeline(rows, watchdog):
    n = len(rows)
    indexed = [(i, rid, q) for i, (rid, q) in enumerate(rows)]
    answers = [None] * n

    # --- Phase 0: prefilter (regex, no model) ---
    t = time.time()
    pf_results = [prefilter(q) for _, _, q in indexed]
    _log(f"prefilter: {n} rows, jb-suspected={sum(1 for p in pf_results if p.flag=='jailbreak')} "
         f"({time.time()-t:.1f}s)")

    # --- Phase 1: load models (guard first, then vLLM) ---
    t = time.time()
    guard = GuardClient()
    guard_ok = guard.load()
    _log(f"guard load: ok={guard_ok} ({time.time()-t:.1f}s)")
    if watchdog.must_stop():
        _log("watchdog: stop after guard load -> all refusals")
        for idx, rid, q in indexed:
            answers[idx] = fallback_for("S_harmful", lang_of(q))
        return [(rows[i][0], answers[i]) for i in range(n)]

    t = time.time()
    vllm = VLLMEngine()
    vllm_ok = vllm.load()
    _log(f"vLLM load: ok={vllm_ok} ({time.time()-t:.1f}s), elapsed={watchdog.elapsed():.0f}s")
    if not vllm_ok:
        _log("vLLM FAILED -> all refusals")
        for idx, rid, q in indexed:
            answers[idx] = fallback_for("S_harmful", lang_of(q))
        return [(rows[i][0], answers[i]) for i in range(n)]

    # Stage deadlines from remaining budget (reserve write margin).
    remaining = watchdog.remaining()
    WRITE_MARGIN = 60.0
    usable = max(60.0, remaining - WRITE_MARGIN)
    triage_frac, gen_frac, verify_frac, repair_frac = 0.22, 0.50, 0.20, 0.08
    triage_deadline = watchdog.elapsed() + usable * triage_frac
    gen_deadline = triage_deadline + usable * gen_frac
    verify_deadline = gen_deadline + usable * verify_frac
    repair_deadline = verify_deadline + usable * repair_frac
    _log(f"budget remaining={remaining:.0f}s; deadlines triage~{triage_deadline:.0f} "
         f"gen~{gen_deadline:.0f} verify~{verify_deadline:.0f} repair~{repair_deadline:.0f}")

    # --- Phase 2: triage ---
    from .triage import triage_batch
    t = time.time()
    triaged = triage_batch(indexed, pf_results, guard, vllm, watchdog,
                          deadline=triage_deadline)
    cats = [t_[1] for t_ in triaged]
    _log(f"triage: {_cat_counts(cats)} ({time.time()-t:.1f}s)")

    # --- Phase 3: generate (single-shot + best-of-3) ---
    from .generate import generate_batch
    t = time.time()
    raw_answers, gen_meta = generate_batch(indexed, triaged, vllm, guard,
                                            watchdog, deadline=gen_deadline)
    n_bo3 = sum(1 for m in gen_meta if m.get("used_bestof"))
    _log(f"generate: {n} answers, best-of-3 used on {n_bo3} ({time.time()-t:.1f}s)")

    # --- Phase 4: verify ---
    from .verify import verify_batch, FIX_NONE, FIX_SAFETY, FIX_HELPFULNESS
    t = time.time()
    verdicts = verify_batch(indexed, cats, raw_answers, guard, vllm, watchdog,
                             deadline=verify_deadline)
    n_fix = sum(1 for v in verdicts if v != FIX_NONE)
    _log(f"verify: {n_fix} flagged (fix) ({time.time()-t:.1f}s)")

    # --- Phase 5: repair (final fallback, max 2 rounds, hard deadline) ---
    from .repair import repair_round
    pending = [i for i in range(n)
               if verdicts[i] != FIX_NONE and not watchdog.must_stop()]
    for round_n in range(1, 3):
        if not pending or watchdog.must_stop() or watchdog.elapsed() >= repair_deadline:
            break
        t = time.time()
        _log(f"repair round {round_n}: {len(pending)} flagged, elapsed={watchdog.elapsed():.0f}s")
        new_ans, new_verdicts = repair_round(
            indexed, cats, raw_answers, verdicts, pending, vllm, guard, watchdog)
        for i, a in new_ans.items():
            if a is not None:
                raw_answers[i] = a
        for i, v in new_verdicts.items():
            verdicts[i] = v
        pending = [i for i in pending
                   if verdicts[i] != FIX_NONE and not watchdog.must_stop()
                   and watchdog.elapsed() < repair_deadline]
        _log(f"repair round {round_n} done ({time.time()-t:.1f}s), pending={len(pending)}")

    # --- Phase 6: finalize ---
    for i, (idx, rid, q) in enumerate(indexed):
        cat = cats[i]
        lang = lang_of(q)
        ans = postprocess(raw_answers[i])
        if not ans:
            answers[idx] = fallback_for(cat, lang)
            continue
        v = verdicts[i]
        if v == FIX_SAFETY:
            answers[idx] = fallback_for(cat, lang)   # safety-first: drop
        elif v == FIX_HELPFULNESS:
            answers[idx] = ans if ans else fallback_for(cat, lang)
        else:
            answers[idx] = ans
    _log(f"finalize: {sum(1 for a in answers if a)} filled, "
         f"elapsed={watchdog.elapsed():.0f}s")
    return [(rows[i][0], answers[i]) for i in range(n)]
