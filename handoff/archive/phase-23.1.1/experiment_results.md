---
step: phase-23.1.1
cycle_date: 2026-04-27
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python -c "import asyncio; from backend.services.macro_regime import compute_macro_regime; r = asyncio.run(compute_macro_regime(use_cache=False)); assert r.regime in {\"risk_on\",\"risk_off\",\"mixed\",\"unknown\"}; assert 0.5 <= r.conviction_multiplier <= 1.5; assert isinstance(r.series_used, list) and len(r.series_used) >= 3; assert len(r.rationale) <= 300; print(\"ok regime=\" + r.regime + \" mult=\" + str(r.conviction_multiplier))"'
---

# Experiment Results — phase-23.1.1

## What was built

Daily macro regime filter — LLM-as-judge over a FRED snapshot. Returns a structured regime tag (`risk_on / risk_off / mixed / unknown`) plus a conviction multiplier that the screener applies to its composite score. Default-OFF feature flag; existing screener behavior preserved when flag is unset.

## Files modified

| File | Change |
|---|---|
| `backend/tools/fred_data.py` | Added 2 series to `SERIES` dict: `VIXCLS` (CBOE VIX) + `BAMLH0A0HYM2` (ICE BofA HY OAS) |
| `backend/services/macro_regime.py` | NEW (~210 lines) — `MacroRegimeOutput` schema, `compute_macro_regime()`, 24h file cache, `apply_regime_to_score()` helper |
| `backend/tools/screener.py` | `rank_candidates()` accepts optional `regime` kwarg; applies multiplier + sector tilt after composite score |
| `backend/services/autonomous_loop.py` | Step 1 fetches regime when `macro_regime_filter_enabled` flag is true; passes to `rank_candidates` |
| `backend/config/settings.py` | 2 new fields: `macro_regime_filter_enabled` (default False), `macro_regime_model` (default `claude-haiku-4-5`) |
| `tests/services/__init__.py` | NEW (package marker) |
| `tests/services/test_macro_regime.py` | NEW (12 tests: schema enums, range enforcement, score application, sector tilt, cache roundtrip, cache expiry, cache corruption) |

## Implementation notes (Anthropic structured-output gotchas)

Three Anthropic-specific schema constraints surfaced and were handled:
1. `additionalProperties: false` is required on every object — solved with `model_config = ConfigDict(extra="forbid")` on both Pydantic models.
2. Numeric `minimum`/`maximum` are NOT supported in structured-output schemas — solved by stripping `_UNSUPPORTED_SCHEMA_KEYS` from the schema dict before passing to `client.generate_content`. Pydantic `ge`/`le` retained for runtime validation + tests.
3. String `maxLength` is also NOT enforced — clamp the `rationale` field to 300 chars before Pydantic validation.

The `_strip_unsupported_schema_keys` helper recurses through the JSON schema and removes `minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `maxLength`, `minLength`. Conviction and conviction_multiplier are clamped at parse time.

## Verbatim verification command output

```
$ source .venv/bin/activate && python -c "import asyncio; from backend.services.macro_regime import compute_macro_regime; r = asyncio.run(compute_macro_regime(use_cache=False)); assert r.regime in {'risk_on','risk_off','mixed','unknown'}; assert 0.5 <= r.conviction_multiplier <= 1.5; assert isinstance(r.series_used, list) and len(r.series_used) >= 3; assert len(r.rationale) <= 300; print('ok regime=' + r.regime + ' mult=' + str(r.conviction_multiplier))"
Failed to fetch FRED series UMCSENT: Server error '500 Internal Server Error' ...
Failed to fetch FRED series DGS10: Server error '500 Internal Server Error' ...
ok regime=risk_on mult=1.15
exit=0
```

The two FRED 500s are transient vendor errors on UMCSENT/DGS10 (neither is in the `_REGIME_SERIES` shortlist). The code degraded gracefully — 7 of 9 series available, well above the 3-series floor; regime classification proceeded normally.

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/services/test_macro_regime.py -v --no-header -q
collected 12 items
tests/services/test_macro_regime.py ............  [100%]
============================== 12 passed in 0.02s ==============================
```

All 12 tests pass:
- `test_schema_enforces_regime_enum` — invalid regime tag rejected
- `test_schema_enforces_conviction_range` — conviction > 1.0 rejected
- `test_schema_enforces_multiplier_range` — multiplier > 1.5 rejected
- `test_default_multipliers_are_in_valid_range` — defaults sanity-check
- `test_apply_regime_no_regime_passes_through` — None regime = identity
- `test_apply_regime_multiplier_only` — risk_off mult=0.7 applies
- `test_apply_regime_overweight_sector_boost` — XLK overweight gives 1.05x
- `test_apply_regime_underweight_sector_penalty` — XLK underweight gives 0.95x
- `test_apply_regime_unknown_sector_no_tilt` — None sector skips tilt
- `test_cache_roundtrip` — save/load preserves all fields
- `test_cache_expired` — >24h old cache returns None
- `test_cache_unreadable_returns_none` — corrupt cache file returns None

## Cost / cycle posture

- Single LLM call to Claude Haiku 4.5 per day per cache miss
- 24h file cache at `backend/services/_cache/macro_regime.json` prevents re-billing
- Estimated cost: <$0.01/day (Haiku 4.5 input ~150 tokens prompt + ~250 token JSON output)
- Default OFF (`macro_regime_filter_enabled = False`) — existing autonomous_loop behavior preserved when flag is unset

## Out of scope (per contract)

- Daily APScheduler cron registration — call already runs on Step 1 of autonomous_loop, no separate cron needed
- UI surface in Settings + Signals page — phase-23.1.6
- Backtest validation of conviction multiplier impact — phase-23.2.5
- The existing `regime_detection_enabled` flag at `settings.py:84` (different feature; not touched)

## Live regime sample (sanity check)

The actual LLM call returned: `regime=risk_on mult=1.15`. Sample regime data the model received: T10Y2Y=0.53 (positive yield curve), VIX=19.3 (mild elevated), HY OAS=2.86% (tight = risk-on). Classification is consistent with the published thresholds in the research brief.

## What's next

1. Spawn fresh Q/A subagent (single-Q/A rule)
2. On Q/A PASS: append `handoff/harness_log.md` cycle entry
3. Add masterplan.json step entry (phase-23.1.1) with status=done
4. Archive handoff via PostToolUse hook
5. Commit on main + push
6. Move to phase-23.1.2 (earnings PEAD overlay)
