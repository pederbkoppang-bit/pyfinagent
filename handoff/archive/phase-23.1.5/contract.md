---
step: phase-23.1.5
title: LLM-as-judge meta-scorer (single batched Claude call combines all sub-signals into conviction 1-10)
cycle_date: 2026-04-27
harness_required: true
verification: 'source .venv/bin/activate && python -c "import asyncio; from backend.services.meta_scorer import meta_score_candidates, MetaScoredCandidate; cands = [{\"ticker\": \"AAPL\", \"sector\": \"Information Technology\", \"momentum_1m\": 8.2, \"momentum_3m\": 15.1, \"momentum_6m\": 25.0, \"rsi_14\": 72, \"composite_score\": 14.5}, {\"ticker\": \"NVDA\", \"sector\": \"Information Technology\", \"momentum_1m\": -2.0, \"momentum_3m\": 5.0, \"momentum_6m\": 12.0, \"rsi_14\": 45, \"composite_score\": 8.0}]; out = asyncio.run(meta_score_candidates(cands)); assert isinstance(out, list); assert all(\"conviction_score\" in c for c in out); assert all(1 <= c[\"conviction_score\"] <= 10 for c in out); assert out[0][\"conviction_score\"] >= out[-1][\"conviction_score\"]; print(\"ok n=\" + str(len(out)) + \" top=\" + out[0][\"ticker\"] + \"(\" + str(out[0][\"conviction_score\"]) + \") bottom=\" + out[-1][\"ticker\"] + \"(\" + str(out[-1][\"conviction_score\"]) + \")\")"'
research_brief: handoff/current/phase-23.1.5-research-brief.md
---

# Contract — phase-23.1.5

## Hypothesis

A single batched Claude Haiku 4.5 call over 30 candidates with all sub-signals (momentum + macro regime + PEAD + news + sector) returns a conviction 1-10 per candidate that genuinely re-weighs the multiplicative cascade today's screener uses. The "first state what could go wrong" prompt design + randomized order + explicit regime-momentum interaction rule prevents rubber-stamping. Resulting `conviction_score` becomes the primary ranking key when `meta_scorer_enabled=True`.

## Plan

1. **NEW `backend/services/meta_scorer.py`** mirroring previous service designs:
   - `MetaScoredCandidate` and `MetaScorerBatch` Pydantic models — `ConfigDict(extra="forbid")`, `conviction_score: int`, `conviction_reason: str`, `ticker: str`. ge/le retained on Pydantic (per cycle-1 lesson) but stripped from JSON schema sent to Anthropic.
   - `_build_meta_prompt(candidates, regime)` — prompt with the anti-rubber-stamp structure from research brief (counterargument first, regime-momentum interaction rule, randomized candidate order, calibration anchors)
   - `meta_score_candidates(candidates, regime=None) -> list[dict]` async — shuffle + build prompt + Claude call + parse + clamp `conviction_score` to [1, 10] + sort desc by conviction
   - On any failure (no Anthropic key, Claude error, parse error) → return candidates unchanged with `conviction_score = round(composite_score)` clamped to [1, 10] as fallback
   - Cap at 30 candidates per batch; if more passed in, only score top-30 by composite_score (remaining keep their original ordering)
2. **Wire into `backend/tools/screener.py`** — extend `rank_candidates` with a final pass: if `meta_scorer_enabled=True`, call `meta_score_candidates` on the top-30 by composite_score, attach `conviction_score` field, re-sort by conviction. The composite_score field is preserved alongside conviction_score for audit.
3. **Settings** — `meta_scorer_enabled: bool = False`, `meta_scorer_model: str = "claude-haiku-4-5"`, `meta_scorer_max_batch: int = 30`.
4. **Wire into `backend/services/autonomous_loop.py` Step 1** — pass `meta_scorer_enabled` flag through to `rank_candidates` (or call separately after rank_candidates returns).
5. **Tests** at `tests/services/test_meta_scorer.py`:
   - Schema validation (conviction must be int in [1,10])
   - `_build_meta_prompt` includes counterargument instruction + regime-momentum rule + randomized order disclosure
   - `meta_score_candidates` with mocked Claude returns shuffled-back-to-conviction-sorted list
   - Fallback path on no API key returns candidates with `conviction_score = clamp(composite_score, 1, 10)` and same length
   - Cap at 30 candidates: passing 50 only scores top-30 (remaining unchanged)

## Out of scope

- Per-candidate (non-batched) call mode — batched-only for Phase 1
- LLM-as-judge for SELL decisions (this is buy-side ranking only)
- Replacing `composite_score` field — both kept for audit trail
- UI surface (phase-23.1.6)

## Files modified

- `backend/services/meta_scorer.py` — NEW (~220 LOC)
- `backend/tools/screener.py` — call meta_score_candidates after existing scoring loop when enabled
- `backend/services/autonomous_loop.py` — pass enable flag through (or call directly after rank_candidates)
- `backend/config/settings.py` — 3 new fields
- `tests/services/test_meta_scorer.py` — NEW

## Verification

The front-matter command exercises real Claude on a 2-candidate input. It verifies:
- `meta_score_candidates` returns a list with `conviction_score` field on each
- Conviction is in [1, 10]
- Output is sorted desc by conviction
- Function reachable from clean `python -c`

## References

- `handoff/current/phase-23.1.5-research-brief.md` — full brief (320 lines, 7 sources read in full, gate_passed: true)
- `backend/services/macro_regime.py` — design template (clamp helpers, _strip_unsupported_schema_keys)
- `backend/services/news_screen.py` — batched-prompt pattern (the meta-scorer mirrors it for 30 candidates)
- `backend/tools/screener.py:151` — extension surface
