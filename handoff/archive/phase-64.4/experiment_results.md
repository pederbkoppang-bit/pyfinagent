# Experiment results — step 64.4 (Multi-market fixture-replay e2e)

**Step:** 64.4 (P1, phase-64, depends_on 66.2=done). $0; local-only; test-infra (1 e2e test file). Research gate
PASSED (research_brief_64.4.md, gate_passed=true, 7 external sources read in full). historical_macro FROZEN; live book
untouched; NO network in the default run (synthetic fixtures).

## What was built (1 pure test file — NO production code changed)

`backend/tests/test_64_4_multi_market_e2e.py` (6 tests: 5 default + 1 requires_live smoke):
- **Synthetic fixtures** (`_bars`/`_ohlcv`): a `yf.download group_by="ticker"` MultiIndex-shaped OHLCV frame (60 bars,
  gentle drift, survives `validate_ohlcv` R1-R3). No network EVER.
- **Per-market funnel** [criterion 1]: drives the PURE seam once per market — `screener.screen_universe` (patching
  `screener.yf.download` → the synthetic frame) → `screener.rank_candidates` → `decide_trades` → `list[TradeOrder]`.
  Asserts screened>0, ranked>0, order-intent>0 for US/KR/EU, and the TradeOrder `.market` matches. Deliberately does
  NOT drive the full autonomous loop (its phase-50.4 calendar gate calls `datetime.now()` and drops intl on weekends).
- **EU under lowered thresholds** [criterion 1, "via test flag"]: EU tickers with sub-default volume (50k) →
  `screen_universe` with DEFAULT thresholds returns **0** (asserted); with the lowered `min_avg_volume=10_000`/
  `min_price=1.0` KWARGS returns >0 → proves the lowered-threshold override is LOAD-BEARING (anti-rig).
- **Currency invariants** [criterion 2]: reuses the 50.2/64.3 fx-mock pattern (`paper_avg_entry_fx_fix_enabled` +
  patched `fx_rates.get_fx_rate` + ExecutionRouter). KR add-on avg_entry stays KRW-scale (~70000); EU stays EUR-scale
  (~150).
- **requires_live variant** [criterion 3]: one `@pytest.mark.requires_live` smoke hitting REAL `yf.download`,
  EXCLUDED by `-m 'not requires_live'` (marker registered pytest.ini:8-9).

## Criterion-1 interpretation (documented; flagged for Q/A)

"EU under the 65.2 thresholds via test flag" — step 65.2 (per-market threshold PRODUCTION flag) is `pending` and its
code does NOT exist (grep = 0 hits). Read "via **test** flag" as a TEST-ONLY override: lowered `min_avg_volume`/
`min_price` kwargs to `screen_universe` (already accepted; screener.py:93-94). Justification: 64.4's DAG `depends_on`
is **66.2 (done), NOT 65.2**; "via test flag" ≠ "via the production flag"; 65.2 will productionize the same concept
later. Mirrors the accepted 64.2 "(testid)" interpretation pattern. **Flagged for the Q/A to adjudicate.**

## Verification (verbatim)

- IMMUTABLE cmd `source .venv/bin/activate && python -m pytest backend/tests -k 'multi_market_e2e' -q -m 'not
  requires_live'` → **5 passed, 1144 deselected** (exit 0).
- **Anti-vacuous funnel counts** (printed): US `universe=2 screened=2 ranked=2 order_intent=2 market=['US','US']`; KR
  `2/2/2/2 market=['KR','KR']`; EU `2/2/2/2 market=['EU','EU']`. All stages >0 for all 3 markets; per-market .market
  correct.
- **requires_live variant** (criterion 3): the 64.4 file has exactly **1** `@pytest.mark.requires_live` mark; `pytest
  -m requires_live --co` → **12** (was 11; +1 = the intentional live smoke, EXCLUDED from the default run). This is
  the criterion-3 deliverable, NOT a quarantine of a flaky default.
- `uvx ruff check` → **All checks passed**. All default tests PURE (no network; `yf.download` mocked; conftest sets
  PYFINAGENT_TEST_NO_BQ=1).

## Do-no-harm / boundaries

$0; local-only; test-infra ONLY (1 new test file; NO production code change — git status = only the test file +
handoff). NO network in the default run (synthetic fixtures; the requires_live smoke is EXCLUDED). NO trade/risk/money
touch; kill-switch/stops/caps/DSR/PBO byte-untouched; historical_macro FROZEN; live book untouched. Drove the PURE
seam only (calendar-gate weekend-flake pitfall avoided). Scope honesty: `git status` may also show the incidental
live autonomous-loop runtime artifacts (cycle_heartbeat/cycle_history/auth_probe/.autonomous_loop.lock) touched by the
running :8000 backend — runtime state, NOT 64.4 code.

## Artifact shape
`backend/tests/test_64_4_multi_market_e2e.py` (6 tests). Re-runnable green: the immutable command above (5 passed).
live_check_64.4.md holds the green run + the per-market funnel counts.
