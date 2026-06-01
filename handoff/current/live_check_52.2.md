# live_check -- phase-52.2: wire the 52wh tilt LIVE (config-gated, default OFF)

**Step:** 52.2 | **Date:** 2026-06-01 | **Result shape:** the 52.1-measured 52wh tilt is now a
config-gated overlay in the LIVE `screener.rank_candidates`; with the flag OFF (default) the engine
is BYTE-IDENTICAL, with the flag ON the ranking tilts toward 52w-high proximity (== the 52.1 logic).
**NO flag flip in this step** -- enable is a deferred, operator-gated, post-Monday-baseline action.

## Verbatim proof
```
=== phase-52.2 proof ===
1. settings default OFF: False | k: 0.5
2. OFF (default) order:         ['T0', 'T1', 'T2', 'T3', 'T4']
3. OFF==explicit-False (byte-identical): True
4. OFF path writes no composite_score_raw (witness untouched): True
5. ON (k=0.5) order (tilted toward 52wh): ['T0', 'T2', 'T4', 'T1', 'T3'] | changed vs OFF: True
```
(T2 pct=0.95 + T4 pct=0.98 tilt UP above T1 pct=0.70 + T3 pct=0.60, despite T1/T3's higher raw composite -- the centered tilt working.)

## pytest
```
$ python -m pytest backend/tests/test_phase_52_2_live_tilt.py -q
.....                                                                    [100%]
5 passed in 0.30s
```
- `test_flag_off_is_byte_identical_default` + `test_flag_off_writes_no_raw_field` (criterion #2).
- `test_flag_on_tilts_toward_52w_high` + `test_live_tilt_matches_52_1_replay_logic` (criterion #1 -- the LIVE basket == the 52.1 `hi52_tilt_basket` basket on the same data).
- `test_missing_pct_is_noop_for_that_name`.

Regression sweep: `20 passed` (52.2 + 52.1 + 51.2 + 50.3-universe).

## Criterion-by-criterion

| # | Criterion | Evidence | Verdict |
|---|-----------|----------|---------|
| 1 | rank_candidates gains a config-gated 52wh-tilt post-pass reproducing the 52.1 ranking logic; composite otherwise unchanged | `_apply_52wh_tilt` (screener.py) inserted after sector_neutral, before the sort; `test_live_tilt_matches_52_1_replay_logic` proves LIVE basket == 52.1 hi52_tilt_basket | PASS |
| 2 | flag OFF (default) -> BYTE-IDENTICAL; working US engine not regressed | proof #2/#3/#4 (OFF==explicit-False, no composite_score_raw written); `test_flag_off_*` | PASS |
| 3 | flag plumbed from settings (momentum_52wh_tilt_enabled default False) through autonomous_loop; NO flag flip in this step | settings.py momentum_52wh_tilt_enabled=False + autonomous_loop.py:655 passes it; the flag is OFF -- no enable | PASS |
| 4 | live_check records OFF byte-identity + ON behavior + the deferred-enable plan | this file | PASS |

## Regression fixed (from the go-live flip, surfaced here)
`test_phase_50_3_universe.py::test_paper_markets_default_is_us_only` was asserting
`get_settings().paper_markets == ['US']` -- which the operator's 2026-06-01 go-live `.env` override
(`PAPER_MARKETS=['US','EU','KR']`) correctly broke. Fixed the test to assert the CODE DEFAULT
(`Settings.model_fields['paper_markets'].default_factory() == ['US']`) -- the faithful byte-identity
invariant (default is US-only; multi-market is an explicit opt-in). This is a legitimate fix of a
test that conflated "code default" with "effective .env-overridden value", NOT rigging -- the
invariant (code default ['US']) still holds; get_settings() correctly reflects the live opt-in.

## The DEFERRED enable plan (NOT done in this step)
To enable the 52wh tilt live (a SEPARATE, operator-gated action, recommended AFTER Monday's
multi-market baseline is measured):
1. Set `MOMENTUM_52WH_TILT_ENABLED=true` in `backend/.env` (k stays 0.5).
2. Restart the backend (`launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend`).
3. Verify the next cycle's ranking reflects the tilt; monitor paper_* Sharpe over multiple cycles.
4. Enable-decision caveats: the +0.05 was 1-of-5 configs -> DSR-deflate (Bailey-LdP) before trusting; confirm OOS post-Monday; k=0.5 is the milder/plateau choice. Reversible (flag back to false + restart).

## Scope / safety
- **Flag DEFAULT OFF -> the live engine is BYTE-IDENTICAL** after this step (no behavior change; the +20% engine untouched). The tilt is dormant until the operator flips the flag.
- The LIVE tilt logic is proven to MATCH the 52.1-measured `hi52_tilt_basket` (so the measured +0.05 is what an enable would deliver).
- $0 LLM; no pip; no flag flip; reversible.
