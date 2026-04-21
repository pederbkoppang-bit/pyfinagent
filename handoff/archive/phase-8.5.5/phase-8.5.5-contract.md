# Sprint Contract — phase-8.5 / 8.5.5 (DSR + PBO blocking gate, CPCV)

**Step id:** 8.5.5 **Cycle:** 1 **Date:** 2026-04-20 **Tier:** simple (closure)

Closure-style brief in `phase-8.5.5-research-brief.md`.

## Hypothesis

Ship `backend/autoresearch/gate.py` with `PromotionGate(min_dsr=0.95, max_pbo=0.2)` + `.evaluate(trial)` returning `{promoted: bool, rejected_reason: str|None}`. Plus `cpcv_folds(n, k)` returning combinatorial purged cross-validation fold pairs. Test script exercises: DSR-below-threshold reject, PBO-above-threshold reject, CPCV applied (fold count matches), rejection+revert regression (a rejected trial does not mutate state).

## Immutable
- `python scripts/harness/autoresearch_gate_test.py` exits 0.

## Plan
Write `backend/autoresearch/gate.py` + `scripts/harness/autoresearch_gate_test.py`. Verify + Q/A + log + flip.
