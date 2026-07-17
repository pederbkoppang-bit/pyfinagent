# Experiment results — step 64.3 (Backend gap tests)

**Step:** 64.3 (P1, phase-64, depends_on none). $0; local-only; test-infra (4 pure pytest files). Research gate PASSED
(research_brief_64.3.md, gate_passed=true, 5 external sources read in full). historical_macro FROZEN; live book
untouched; NO production/live-loop change.

## What was built (4 pure test files — NO production code changed; NO live net/BQ)

1. **`backend/tests/test_64_3_kill_switch_machine.py`** (5 tests) [criterion 2] — `evaluate_breach` +
   `check_auto_resume(enabled=False)` over a fresh tmp-backed `KillSwitchState` (`_fresh_state` mirrors
   test_phase_38_1:26). Asserts the RAIL-5 stays-paused policy (away-ops-rules.md:17-18): pause→is_paused;
   `check_auto_resume(healthy, enabled=False)`→action "no_op" + "auto_resume_disabled" + **is_paused STILL True**;
   enabled=True + active breach → action != "resume" + still paused; `evaluate_breach(nav<=0)` → nav_invalid True /
   any_breached False (phase-69.1 fail-safe); clean NAV → any_breached False.
2. **`backend/tests/test_64_3_currency_path.py`** (4 tests) [criterion 3] — mirrors the proven KR harness
   (test_phase_70_3:192) generalized to KR/EU/US + fx-none. Asserts: **KR** add-on avg_entry stays KRW-scale (~70000)
   ON vs tiny OFF; **EU (.DE)** avg_entry stays EUR-scale (~150) ON vs USD-inflated (~162) OFF; **US** byte-identical
   (ON==OFF); **fx unavailable** → execute_buy returns None (no position saved). Tolerance asserts (`abs<eps`), not
   bit-exact. (The fix is phase-70.3; 61.3 was display-only — asserts 70.3 behavior in the SHAPE of the 61.3 criteria.)
3. **`backend/tests/test_64_3_screener_market.py`** (5 tests) — `validate_ohlcv` (pure pandas) + `market_for_symbol`.
   Asserts: US → dropped==0 + SAME object (byte-identical fast-path); R1 impossible bar dropped; R2 identical-OHLC +
   zero-vol dropped vs vol>0 flagged-not-dropped; R3 >50% single-day move dropped; market_for_symbol .KS→KR / .DE→EU /
   bare→US.
4. **`backend/tests/test_64_3_learnings_reader.py`** (4 tests) — `get_paper_trades_in_window` (BigQueryClient with a
   MagicMock client, skips __init__/ADC). Asserts error != empty: `query.side_effect=RuntimeError` → **raises**
   (surfaces); empty result → **[]**; `pair_round_trips([])==[]`. (Does NOT touch `_compute_learnings`'s swallow — a
   known gap, behavior change, out of scope for a test-only step.)

## Verification (verbatim)

- IMMUTABLE cmd `source .venv/bin/activate && python -m pytest backend/tests -k '64_3 or kill_switch_machine or
  currency_path or screener_market or learnings_reader' -q` → **18 passed, 1125 deselected** (exit 0). Per-area:
  kill_switch_machine 5, currency_path 4, screener_market 5, learnings_reader 4 = all 4 gap areas covered (criterion 1).
- **requires_live quarantine does NOT grow**: `grep requires_live` on the 4 files → NONE. `pytest -m requires_live
  --co` → 11 collected (unchanged; my files add 0). NOTE: the actual `requires_live` marker count is **11**, not the
  research's estimate of 6 — but criterion 1 is "does not grow", and 64.3 contributes **+0** (all 4 files are pure).
- `uvx ruff check` (the 4 files) → **All checks passed**.
- All tests PURE: MagicMock bq, patched `fx_rates.get_fx_rate` + `pt.ExecutionRouter`, monkeypatched
  `kill_switch._AUDIT_PATH`/`_state`, pandas frames; `conftest.py` sets `PYFINAGENT_TEST_NO_BQ=1`. No network/BQ.

## Do-no-harm / boundaries

$0; local-only; test-infra ONLY (4 new pure pytest files; NO production code change — the tests only READ the
code-under-test with mocked IO). NO trade/risk/money touch; kill-switch/stops/caps/DSR/PBO byte-untouched;
historical_macro FROZEN; live book untouched. Scope honesty: `git status` may also show the incidental live
autonomous-loop runtime artifacts (cycle_heartbeat/cycle_history/auth_probe/.autonomous_loop.lock) touched by the
running :8000 backend — runtime state, NOT 64.3 code (same as prior steps this session). My 64.3 deliverable is
exactly the 4 test files.

## Artifact shape
4 pytest files under `backend/tests/test_64_3_*.py`. Re-runnable green: the immutable command above (18 passed).
live_check_64.3.md holds the green run transcript + per-area counts.
