# Sprint Contract — phase-32.4 Backfill Company Names on Legacy paper_positions

**Step ID:** `phase-32.4`
**Date:** 2026-05-21
**Cycle type:** Implementation. Migration + backfill helper + autonomous-loop wiring. P2 priority (cosmetic/dashboard fix, not safety-critical).

---

## Research-Gate Summary

- **Tier:** simple. **gate_passed:** true (per researcher).
- **Brief:** `handoff/current/research_brief.md`.
- **Key findings:**
  - `_fetch_ticker_meta` lives at `backend/api/paper_trading.py:971`. Helper `_yfinance_ticker_info` at line 958-968 — uses `info.get("shortName") or info.get("longName") or ticker` (the masterplan's spec was inverted; the actual canonical chain in the codebase is `shortName` THEN `longName`). Phase-32.4 mirrors this exact chain for consistency.
  - `paper_positions` schema (verified via `mcp__claude_ai_Google_Cloud_BigQuery__get_table_info` after 32.1+32.2+32.3): 21 fields. No `company_name`. Migration required.
  - `backfill_missing_stops` template at `paper_trader.py:495-562` is the canonical idempotency pattern (filter NULL/empty, mutate via `_safe_save_position`, return `{backfilled, skipped, count_backfilled, count_skipped}`).
- **Dashboard wiring gap (out-of-band finding):** the `/api/paper-trading/portfolio` endpoint already pulls `company_name` from `analysis_results` via `_fetch_ticker_meta`, NOT from `paper_positions`. The frontend at `paper-trading/page.tsx:845` consumes `tickerMeta[pos.ticker]?.company_name`. Per the masterplan's strict scope, phase-32.4 backfills `paper_positions.company_name` only; the API endpoint wiring to make the dashboard READ from there is OUT OF SCOPE and recorded as a phase-32.5 followup. The audit's user-facing observation (9 of 11 positions showing ticker-as-company) will be ADDRESSED at the data layer but not yet at the display layer.

---

## Hypothesis

Add `backfill_missing_company_names()` helper to PaperTrader, modelled on `backfill_missing_stops()`. Wire into autonomous_loop.py Step 5.6 alongside `backfill_missing_stops()`. Result: `paper_positions.company_name` will be populated for all 11 current positions with real yfinance names. The dashboard will still show ticker-as-company until phase-32.5 fixes the API wiring — but the data foundation is in place.

---

## Success Criteria (IMMUTABLE — from `.claude/masterplan.json::phase-32.4.verification.success_criteria`)

1. `backfill_missing_company_names_helper_added_to_paper_trader`
2. `called_from_autonomous_loop_alongside_backfill_missing_stops`
3. `uses_same_yfinance_longName_path_as_fetch_ticker_meta`
4. `idempotent_returns_zero_on_repeat_run`
5. `skips_when_company_name_is_already_a_real_name_not_just_ticker`
6. `fail_open_logs_warning_on_yfinance_error`
7. `unit_test_4_cases_pass` (we will ship ≥5)

Verification command (must pass):
```bash
python -m pytest backend/tests/test_phase_32_4_backfill_company_names.py -v && \
grep -n 'backfill_missing_company_names' backend/services/paper_trader.py backend/services/autonomous_loop.py
```

Live check requirement: `handoff/current/live_check_32.4.md` shows a BQ row from `paper_positions` confirming at least 8 of 9 affected tickers (MU, KEYS, GEV, COHR, ON, DELL, GLW, LITE, WDC) now have `company_name != ticker`.

---

## Immutable Hard Guardrails (verbatim from `implementation_plan.hard_guardrails`)

1. Cosmetic-only change — MUST NOT affect any trading decision.
2. NO change to `risk_judge.md` or any agent skill.
3. NO change to `decide_trades`, `check_stop_losses`, `mark_to_market` exit logic.
4. Fail-open ALWAYS: yfinance fetch failure must NOT block the cycle.
5. Idempotent: re-running must be a no-op when all names are real.

Plus global overnight goal guardrails (NO `AskUserQuestion`, NO mutating Alpaca, scope honesty).

---

## Plan Steps

1. **RESEARCH** ✅ done.
2. **PLAN** ✅ this file.
3. **MIGRATION** — `scripts/migrations/phase_32_4_add_company_name.py`. Idempotent ALTER TABLE ADD COLUMN IF NOT EXISTS company_name STRING. Backfill is handled by the helper, not the migration.
4. **GENERATE** — `backfill_missing_company_names(self, force: bool = False) -> dict` on PaperTrader near `backfill_missing_stops` at `paper_trader.py:495-562`. Iterates `self.get_positions()`. For each pos where `company_name in (None, "", ticker)` (the sentinel set), fetch yfinance `shortName` or `longName` (mirroring `_yfinance_ticker_info` at `paper_trading.py:958-968`). Persist via `_safe_save_position`. Returns `{backfilled, skipped, count_backfilled, count_skipped}`. Try/except around yfinance — fail-open. Wire into `autonomous_loop.py` Step 5.6 region AFTER `backfill_missing_stops`. Update `_POSITION_RT_FIELDS` to include `company_name`.
5. **TESTS** — `backend/tests/test_phase_32_4_backfill_company_names.py`. 5+ cases per `test_specs`.
6. **VERBATIM RESULTS** — `handoff/current/experiment_results.md`.
7. **EVALUATE** — `qa` ONCE.
8. **LIVE CHECK** — run the migration, run the helper against production paper_positions, quote BQ rows.
9. **LOG** — append cycle block.
10. **FLIP** — `phase-32.4.status: pending → done`. Commit `phase-32.4:`. **END of overnight run** — phase-32.4 is the last pending step.

---

## References

- Researcher brief at `handoff/current/research_brief.md`.
- Phase-32.1/2/3 commits.
- `_yfinance_ticker_info` at `backend/api/paper_trading.py:958-968` (canonical chain).
- `backfill_missing_stops` at `backend/services/paper_trader.py:495-562` (template).
- Phase-32.5 candidate documented in experiment_results.md "Followups" section: API wiring change so dashboard reads `paper_positions.company_name` directly.
