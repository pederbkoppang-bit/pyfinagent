# Contract — step 64.3 (Backend gap tests)

**Phase:** phase-64 | **Step:** 64.3 | **Priority:** P1 | harness_required: true | depends_on: none
**Cycle:** 1 | Date: 2026-07-17 | **Type:** test-infra (4 pure pytest files). $0; local-only; NO production/live-loop
change; historical_macro FROZEN; live book untouched. Pure unit tests — NO live network/BQ.

## Research-gate summary (gate PASSED)

Researcher subagent (Agent tool, Opus 4.8 effort:max, $0), brief `research_brief_64.3.md`. Envelope:
**gate_passed=true**, tier=moderate, **5 external sources read in full**, 9 snippet-only, 14 URLs, recency scan, 10
internal files. KEY: all 4 gap areas map to PURE testable seams (no live net/BQ) → `requires_live` quarantine stays at
6 (criterion 1). Mocking patterns identified from existing harnesses.

## Plan — 4 test files (each name carries a `-k` keyword so the immutable command selects it)

### 1. `backend/tests/test_64_3_kill_switch_machine.py` [criterion 2]
Target `backend/services/kill_switch.py`: `evaluate_breach` (:281), `check_auto_resume(..., enabled=False)` (:340).
Isolate via a `_fresh_state` helper (mirror `test_phase_38_1_kill_switch_auto_resume.py:26` — monkeypatch
`kill_switch._AUDIT_PATH`→tmp_path AND `kill_switch._state`→fresh `KillSwitchState()`). ASSERT (rail-5 stays-paused,
away-ops-rules.md:17-18): pause("test")→is_paused True; `check_auto_resume(healthy_nav,4,10,enabled=False)`→
action=="no_op" AND "auto_resume_disabled" in reason AND is_paused() STILL True; enabled=True + active breach →
"breach_still_active" still paused; `evaluate_breach(nav<=0)` → nav_invalid True / any_breached False.

### 2. `backend/tests/test_64_3_currency_path.py` [criterion 3]
Target `backend/services/paper_trader.py` `execute_buy` add-on avg_entry math (:332 fix ON vs :334 legacy), flag
`settings.paper_avg_entry_fx_fix_enabled` (settings.py:455, default False). Mirror the proven KR harness
(`test_phase_70_3_atomic_swap.py:192` `_kr_trader`: MagicMock bq with the get_* return_values, `save_paper_position`
side_effect captures the row, patch `fx_rates.get_fx_rate`, patch `ExecutionRouter`, `_maybe_notify_trade`=noop,
`get_settings().model_copy(update={flag})`). ASSERT: KR add-on avg_entry stays KRW-scale (~70000, tolerance) ON vs
tiny OFF; **ADD EU (.DE)**: avg_entry stays EUR-scale (== EUR-weighted avg) ON vs USD-inflated OFF; US byte-identical;
fx unavailable (`get_fx_rate`→None) → execute_buy returns None (skips buy). Note: fix is **phase-70.3** (61.3 was
display-only) — assert 70.3 behavior in the SHAPE of the 61.3 criteria. Tolerance asserts (`abs<eps`), not bit-exact.

### 3. `backend/tests/test_64_3_screener_market.py`
Target `backend/tools/price_quality.py` `validate_ohlcv(df, market, ticker)` (:48, pure pandas) + `market_for_symbol`
(`backend/backtest/markets.py:142`). ASSERT: market="US" → df UNCHANGED (dropped==0, fast-path); EU/KR impossible bar
(R1) dropped; R2 identical-OHLC zero-vol dropped vs vol>0 flagged-not-dropped; R3 >50% move dropped; market_for_symbol
`.KS`→KR / `.DE`→EU / bare→US. Pure pandas DataFrames, no mock.

### 4. `backend/tests/test_64_3_learnings_reader.py`
Target `backend/db/bigquery_client.py` `get_paper_trades_in_window` (:948) — the clean error≠empty seam. Construct via
`BigQueryClient.__new__(BigQueryClient)`; `bq.client=MagicMock()`; `bq._pt_table=lambda t:"p.d.t"` (skips
__init__/ADC). ASSERT: `bq.client.query.side_effect=RuntimeError` → `pytest.raises` (error SURFACES); `.return_value=[]`
→ returns [] (empty); `pair_round_trips([])==[]`. Pins error != empty. (Do NOT touch `_compute_learnings`'s swallow —
behavior change, out of scope; flag it in the brief only.)

## Boundaries (binding)
$0; local-only; test-infra ONLY (4 new pure pytest files; NO production code change). All tests are PURE (MagicMock
bq, patched fx/ExecutionRouter, monkeypatched _AUDIT_PATH/_state, pandas frames); `conftest.py` already sets
`PYFINAGENT_TEST_NO_BQ=1`. NONE use `@pytest.mark.requires_live` → the quarantine list STAYS at 6 (criterion 1). NO
trade/risk/money touch; kill-switch/stops/caps/DSR/PBO byte-untouched; historical_macro FROZEN; live book untouched.
The tests only READ the code-under-test with mocked IO. Note: the 70.3 KR test already exists + passes; 64.3 adds the
EU case + the `currency_path` name for `-k` selection.

## Immutable success criteria (verbatim from masterplan.json 64.3)
1. "new test files cover all four gap areas and pass; the requires_live quarantine list does not grow"
2. "the kill-switch tests assert the stays-paused policy (auto-resume OFF) that rail 5 depends on"
3. "currency tests assert KR avg_entry stays KRW-scale on add-on buys and EU rows stay EUR-scale (mirrors 61.3 criteria)"

**Verification command (immutable):**
`cd /Users/ford/.openclaw/workspace/pyfinagent && source .venv/bin/activate && python -m pytest backend/tests -k '64_3 or kill_switch_machine or currency_path or screener_market or learnings_reader' -q`

## References
research_brief_64.3.md; backend/services/kill_switch.py:281,340; docs/runbooks/away-ops-rules.md:17-18 (rail 5);
backend/services/paper_trader.py:231-233,332,334; backend/config/settings.py:455 (paper_avg_entry_fx_fix_enabled);
backend/tools/price_quality.py:48; backend/backtest/markets.py:142; backend/db/bigquery_client.py:948;
test_phase_38_1_kill_switch_auto_resume.py:26 (_fresh_state); test_phase_70_3_atomic_swap.py:192 (_kr_trader harness);
conftest.py (PYFINAGENT_TEST_NO_BQ). pytest docs (monkeypatch, parametrize).
