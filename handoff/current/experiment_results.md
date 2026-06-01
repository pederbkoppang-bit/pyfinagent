# experiment_results -- phase-52.2: wire the 52wh tilt LIVE (config-gated, default OFF)

**Step:** 52.2 | **Date:** 2026-06-01 | **$0 LLM** | no pip | **flag DEFAULT OFF -> byte-identical; NO enable** | GENERATE complete

## What was changed (the 52.1-measured edge -> production-ready, gated, reversible)

| File | Change |
|------|--------|
| `backend/tools/screener.py` | NEW `_apply_52wh_tilt(scored, k)` helper (centered multiplicative 52wh tilt on composite_score, mean over non-None pct_to_52w_high, missing->1.0, writes composite_score_raw) -- mirrors the 52.1 `hi52_tilt_basket` EXACTLY. Added kwargs `momentum_52wh_tilt=False, momentum_52wh_tilt_k=0.5` to `rank_candidates`; gated post-pass `if momentum_52wh_tilt and scored: _apply_52wh_tilt(...)` inserted after the sector_neutral block, before the final sort (:473). |
| `backend/config/settings.py` | NEW `momentum_52wh_tilt_enabled: bool = Field(False)` + `momentum_52wh_tilt_k: float = Field(0.5)` (beside the multidim flags). |
| `backend/services/autonomous_loop.py:655` | the (only) live `rank_candidates` caller now passes `momentum_52wh_tilt`/`_k` from settings (default OFF). |
| `backend/tests/test_phase_52_2_live_tilt.py` | NEW 5 tests: OFF byte-identity (+ no composite_score_raw witness); ON tilts toward 52wh; LIVE basket == 52.1 hi52_tilt_basket; missing-pct no-op. |
| `backend/tests/test_phase_50_3_universe.py:46` | FIXED a go-live regression: assert the CODE DEFAULT (`Settings.model_fields['paper_markets'].default_factory()==['US']`) instead of `get_settings()` (which the go-live `.env` override correctly made ['US','EU','KR']). |

## pct_to_52w_high needed no threading
Already computed in `screen_universe`'s per-ticker loop (screener.py:210-214) + set on every row (:228) for all screened names -> flows to rank_candidates at rank time. Confirmed by the researcher.

## Verification command output (verbatim)
```
=== syntax === OK screener.py / settings.py / autonomous_loop.py / test
$ python -m pytest backend/tests/test_phase_52_2_live_tilt.py -q
.....                                                                    [100%]
5 passed in 0.30s
regression sweep (52.2 + 52.1 + 51.2 + 50.3-universe): 20 passed
```
Live proof (-> live_check_52.2.md): settings default OFF; OFF order == explicit-False (byte-identical); OFF writes no composite_score_raw; ON (k=0.5) tilts toward 52wh (order changes).

## Byte-identity / safety (criterion #2/#3)
- Flag DEFAULT OFF -> the gated post-pass is skipped -> rank_candidates is BYTE-IDENTICAL -> the live +20% engine is UNCHANGED after this step.
- The OFF path writes no `composite_score_raw` (the witness that the pass never ran).
- The LIVE tilt logic == the 52.1-measured `hi52_tilt_basket` (test-proven) -> an enable would deliver the measured +0.05.

## The go-live regression fix (transparent, not rigging)
`test_paper_markets_default_is_us_only` asserted the EFFECTIVE `get_settings().paper_markets`, which the operator-authorized go-live `.env` override (PAPER_MARKETS=['US','EU','KR']) correctly changed. Fixed it to assert the CODE DEFAULT (still ['US']) -- the faithful byte-identity invariant (default US-only; multi-market is an explicit opt-in). The invariant holds; this corrects a test that conflated default with effective.

## Artifact shape
- `_apply_52wh_tilt(scored, k) -> None` (in-place centered tilt)
- `rank_candidates(..., momentum_52wh_tilt=False, momentum_52wh_tilt_k=0.5)`
- settings: `momentum_52wh_tilt_enabled` (False), `momentum_52wh_tilt_k` (0.5)

## DEFERRED enable (NOT done; operator-gated, post-Monday)
Enable = `MOMENTUM_52WH_TILT_ENABLED=true` in backend/.env + backend restart. Caveats: DSR-deflate the +0.05 (1-of-5 configs); confirm OOS post-Monday; k=0.5 (milder/plateau). Reversible.

## Session position
phase-52 (element-2 redirect / north-star #4): 52.1 measured the edge, 52.2 productionized it (gated). The measured momentum edge is now a one-flag-reversible live lever, ready to enable after Monday's multi-market baseline. Remaining: enable decision (operator), 52.3 residual momentum (bigger edge, optional), calendar_events, 50.6 UI, MEASURE Monday.
