---
step: phase-23.1.5
cycle_date: 2026-04-27
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python -c "import asyncio; from backend.services.meta_scorer import meta_score_candidates, MetaScoredCandidate; cands = [...]; out = asyncio.run(meta_score_candidates(cands)); ...; print(\"ok n=\" + str(len(out)) + \" top=\" + out[0][\"ticker\"] + \"(\" + str(out[0][\"conviction_score\"]) + \") bottom=\" + out[-1][\"ticker\"] + \"(\" + str(out[-1][\"conviction_score\"]) + \")\")"'
---

# Experiment Results — phase-23.1.5

## What was built

LLM-as-judge meta-scorer — single batched Claude Haiku 4.5 call combines all sub-signals (momentum + macro regime + PEAD + news + sector) into `conviction_score: int` ∈ [1, 10] per candidate. Replaces the multiplicative cascade as the final ranking key when `meta_scorer_enabled=True`. The prompt is structured to defeat rubber-stamping: counterargument-first instruction, explicit regime-momentum interaction rule, randomized candidate order, calibration anchors at 1-2 and 9-10. On any failure, falls back to `conviction_score = clamp(round(composite_score), 1, 10)` so the cycle still runs.

## Files modified

| File | Change |
|---|---|
| `backend/services/meta_scorer.py` | NEW (~225 lines) — `MetaScoredCandidate` + `MetaScorerBatch` Pydantic models, `_format_candidate_block`, `_build_meta_prompt` (anti-rubber-stamp design), `meta_score_candidates` async, `_fallback_conviction` clamp helper |
| `backend/services/autonomous_loop.py` | After `rank_candidates`, calls `meta_score_candidates(candidates, regime=regime)` when `meta_scorer_enabled`; logs top-conviction ticker; populates `summary["meta_scored_top_conviction"]` |
| `backend/config/settings.py` | 3 new fields: `meta_scorer_enabled` (False), `meta_scorer_model`, `meta_scorer_max_batch` (30) |
| `tests/services/test_meta_scorer.py` | NEW (14 tests: schema, prompt content, fallback paths, batching cap, sort order, clamping, parse-error handling, mocked-LLM happy path) |

## Verbatim verification command output

```
$ source .venv/bin/activate && python -c "import asyncio; from backend.services.meta_scorer import meta_score_candidates, MetaScoredCandidate; cands = [{'ticker': 'AAPL', 'sector': 'Information Technology', 'momentum_1m': 8.2, 'momentum_3m': 15.1, 'momentum_6m': 25.0, 'rsi_14': 72, 'composite_score': 14.5}, {'ticker': 'NVDA', 'sector': 'Information Technology', 'momentum_1m': -2.0, 'momentum_3m': 5.0, 'momentum_6m': 12.0, 'rsi_14': 45, 'composite_score': 8.0}]; out = asyncio.run(meta_score_candidates(cands)); assert isinstance(out, list); assert all('conviction_score' in c for c in out); assert all(1 <= c['conviction_score'] <= 10 for c in out); assert out[0]['conviction_score'] >= out[-1]['conviction_score']; print('ok n=' + str(len(out)) + ' top=' + out[0]['ticker'] + '(' + str(out[0]['conviction_score']) + ') bottom=' + out[-1]['ticker'] + '(' + str(out[-1]['conviction_score']) + ')')"
ok n=2 top=AAPL(7) bottom=NVDA(6)
exit=0
```

Real Claude Haiku 4.5 meta-scored 2 candidates: **AAPL → 7** (positive momentum, +RSI=72 on the high side), **NVDA → 6** (negative 1m, mixed signals). The conviction differential matches the input data — AAPL has clearly stronger signals than NVDA across all available fields. Output is sorted desc by conviction (AAPL first).

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/services/ -v --no-header -q
collected 81 items
tests/services/test_macro_regime.py ............  [ 14%]
tests/services/test_meta_scorer.py ..............  [ 32%]
tests/services/test_news_screen.py .....................  [ 58%]
tests/services/test_pead_signal.py ..................  [ 80%]
tests/services/test_sector_calendars.py ................  [100%]
============================== 81 passed in 0.27s ==============================
```

81/81 tests pass (12 macro + 18 PEAD + 21 news + 16 sector + 14 meta_scorer; no regression across all 5 cycles).

Key meta_scorer test coverage:
- `MetaScoredCandidate` rejects out-of-range conviction (0, 11), rejects extras
- `_format_candidate_block` includes ticker/sector/momentum/PEAD/news/sector_event/composite_score_pre_meta fields
- `_build_meta_prompt` includes the anti-rubber-stamp directives ("INDEPENDENTLY", "what could go WRONG", "risk_off", "ordered randomly", "EXACTLY N candidates")
- Regime placeholder substitution works when regime present
- `_fallback_conviction` clamps composite_score to [1, 10] (handles None, negative, >10)
- No-API-key fallback returns sorted-desc by conviction (test_meta_score_no_anthropic_key_returns_fallback)
- 50-candidate batch caps at MAX_BATCH=30 (top-30 LLM-scored, bottom-20 fallback)
- Mocked-LLM happy path returns parsed batch sorted by conviction
- LLM out-of-range output (`11`) clamped to 10
- LLM parse error → fallback (no exception leaks)

## Anti-rubber-stamp design summary (per research brief)

The prompt is built in `_build_meta_prompt`. Six mitigations baked in:

1. **Counterargument-first:** "First state what could go WRONG with this pick (one clause), then state why you are still bullish or bearish."
2. **Explicit regime interaction rule:** "If momentum is strong but regime is risk_off: this is a warning sign, not a green light."
3. **`composite_score_pre_meta` labeling:** signals input data, not a target to justify.
4. **Calibration anchors:** "Score 9-10 only when momentum, PEAD, regime, AND news all align positively."
5. **Randomized candidate order:** `random.Random(0xC0FFEE).shuffle()` — fixed seed for reproducibility, but order is decoupled from composite_score ranking, removing position bias.
6. **Independence directive:** "Score each candidate INDEPENDENTLY. Do not let one ticker's data influence another's."

## Cost / cycle posture

- ONE Claude Haiku 4.5 call per cycle (batched ≤30 candidates)
- Estimated cost: ~$0.025/cycle (~15K input + 3K output tokens at Haiku pricing)
- Cumulative cost across all 5 cycles when all flags ON: ~$0.10/day
- Default OFF (`meta_scorer_enabled = False`)
- Fallback (no LLM) returns clamped composite_score → cycle still runs

## Architecture choice (Option B from brief)

Per the research brief's "Application to pyfinagent" section, Option B was chosen: **`conviction_score` is added alongside `composite_score`, not as a replacement.** This:
- Preserves audit trail (downstream code can still inspect the multiplicative-cascade score)
- Matches the existing additive overlay pattern (regime, PEAD, news, sector all add fields rather than overwrite)
- Provides graceful fallback (if conviction_score missing, downstream sorts by composite_score)

## Out of scope (per contract)

- Per-candidate (non-batched) call mode — batched-only for Phase 1
- LLM-as-judge for SELL decisions — buy-side ranking only
- BQ persistence of meta scores — file/log only for Phase 1
- UI surface (phase-23.1.6)

## What's next

1. Spawn fresh Q/A
2. On PASS: log → flip → archive → commit → move to phase-23.1.6 (Settings + UI surface — final cycle)
