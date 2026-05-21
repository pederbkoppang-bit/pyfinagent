# Sprint Contract — phase-32.5 Dashboard Wiring (Prefer paper_positions in _fetch_ticker_meta)

**Step ID:** `phase-32.5`
**Date:** 2026-05-21
**Cycle type:** Hot-fix follow-up to phase-32.4. ~50 LOC change to `_fetch_ticker_meta` that closes the operator-visible dashboard gap surfaced after phase-32.4 closed the data layer.

---

## Research-Gate Summary (lightweight)

Research already done in phase-32.4's brief (`handoff/archive/phase-32.4/research_brief.md`) which explicitly recommended this fix: "modify `_fetch_ticker_meta` Step 1 BQ query to consult `paper_positions.company_name` with higher priority than `analysis_results.company_name`." The gap mechanics + 11 affected tickers + canonical yfinance chain are all documented there. No new external research needed for this 50-LOC change.

---

## Hypothesis

Replace the single-table `analysis_results`-only BQ query at `_fetch_ticker_meta` Step 1 with a `UNION ALL` over both `paper_positions` (priority 1) and `analysis_results` (priority 2), `ROW_NUMBER()`-rank per ticker, return the highest-priority non-null name. Filter ticker-as-name sentinel rows at the SQL `WHERE` clause so stale `analysis_results.company_name == ticker` rows cannot outrank a real `paper_positions` value.

After this change, the dashboard COMPANY column will show real names for all 11 positions (matching the `paper_positions.company_name` values backfilled in phase-32.4) AT MOST 24 hours later (the existing `ticker-meta` route cache TTL), or immediately on cache-bust.

---

## Success Criteria (IMMUTABLE — from masterplan)

1. `fetch_ticker_meta_paper_positions_primary_source`
2. `analysis_results_fallback_preserved`
3. `yfinance_fallback_preserved`
4. `ticker_as_name_sentinel_filtered_at_sql`
5. `no_regression_full_sweep_285`

Verification command (must pass):
```bash
python -m pytest backend/tests/ -q --tb=line && \
grep -n 'paper_positions' backend/api/paper_trading.py | head -5
```

Live check requirement: invoke `_fetch_ticker_meta` against production with the 11 current tickers. All 11 must return `source: 'paper_positions'` with real `company_name` values.

---

## Immutable Hard Guardrails

1. NO change to `/portfolio` endpoint shape — only the underlying source-resolution logic in `_fetch_ticker_meta`.
2. NO change to `paper_trader.py`, `autonomous_loop.py`, or `risk_judge.md`.
3. NO mutating BQ writes — read-only query.
4. Preserve graceful fallback (try/except).

---

## Files Touched

- `backend/api/paper_trading.py` — single function `_fetch_ticker_meta` modified. ~50 LOC change.
- `.claude/masterplan.json` — new phase-32.5 entry + flip phase-32 umbrella back to in_progress.
- `handoff/current/contract.md`, `experiment_results.md`, `live_check_32.5.md`, `evaluator_critique.md`.
- `handoff/harness_log.md` — appended cycle block.

No new test file (this is a SQL change that's deeply mocked-out at the test layer; the live invocation against production is the definitive verification).

NO out-of-scope edits anywhere.
