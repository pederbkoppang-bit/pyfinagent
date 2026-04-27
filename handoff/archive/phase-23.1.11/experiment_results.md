---
step: phase-23.1.11
cycle_date: 2026-04-27
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_11.py'
---

# Experiment Results — phase-23.1.11

## What was built

The lite Claude analyzer now writes a row to `analysis_results` after every successful analysis. The Reports page History tab will surface paper-trading-cycle candidates (COHR, KEYS, GEV, etc.) alongside manually-triggered analyses starting from tomorrow's first cycle. **Path A** confirmed (extend existing table; no migration).

## Files modified

| File | Change |
|---|---|
| `backend/services/autonomous_loop.py` | (1) Fixed hardcoded `"source": "claude-sonnet-4"` → `"source": model_name` so audit trail reflects the actual model. (2) Enriched `full_report.market_data` with `name` + `industry` fields (already-computed local vars). (3) NEW `_persist_lite_analysis(analysis, bq)` async helper (~50 LOC) — calls `bq.save_report(...)` with the 14 fields the lite path actually has; non-fatal try/except so BQ outage doesn't kill the cycle. (4) Two call sites in Steps 3 + 4 that invoke the helper after a successful `_run_single_analysis`, guarded by `settings.lite_mode` to avoid double-write when the full Gemini fallback path runs. |
| `tests/services/test_persist_lite_analysis.py` | NEW (8 tests covering field mapping, missing market_data graceful fallback, BQ exception graceful, missing ticker skipped, missing full_report safe defaults, full_report passthrough, recommendation defaults to HOLD). |
| `tests/verify_phase_23_1_11.py` | NEW immutable verification script. |

## Verbatim verification command output

```
$ source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_11.py
Failed to persist lite analysis for COHR: BQ down
ok _persist_lite_analysis async + signature + save_report invocation + graceful BQ failure
exit=0
```

The "Failed to persist… BQ down" line is intentional — proves the graceful-on-error path works (BQ outage simulated; cycle survives). Verification asserts:
1. Function is importable + async
2. Signature is `(analysis, bq)`
3. Calls `bq.save_report` exactly once with `ticker=COHR, recommendation=BUY, final_score=7.0, summary="lite reason", company_name="Coherent Corp."`
4. BQ exceptions do NOT propagate

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/api/ tests/services/ -v --no-header -q
collected 154 items
tests/api/test_paper_trading_deposit.py ............         [ 7%]
tests/api/test_settings_api_signal_stack.py ..............    [16%]
tests/api/test_ticker_meta.py .........                       [22%]
tests/services/test_extract_stop_loss.py ..........            [29%]
tests/services/test_macro_regime.py ............              [37%]
tests/services/test_meta_scorer.py ..............             [46%]
tests/services/test_news_screen.py .....................      [59%]
tests/services/test_pead_signal.py ..................         [71%]
tests/services/test_persist_lite_analysis.py ........         [76%]
tests/services/test_sector_calendars.py ................      [87%]
tests/services/test_signal_attribution.py ....................[100%]
============================== 154 passed in 3.31s ==============================
```

8 new + 146 prior = 154/154 tests pass. Zero regression across all 11 phase-23.1 cycles.

## Field mapping (verified by tests)

| `analysis_results` column | Source from lite analysis dict |
|---|---|
| `ticker` | REQUIRED — `analysis["ticker"]` |
| `analysis_date` | REQUIRED — set by `bq.save_report` to `now()` |
| `recommendation` | REQUIRED — `analysis["recommendation"]` (BUY/SELL/HOLD) |
| `company_name` | `full_report.market_data.name` (from yfinance shortName) |
| `final_score` | `analysis["final_score"]` (1-10) |
| `summary` | `analysis["risk_assessment"]["reason"]` (Claude's reasoning) |
| `price_at_analysis` | `analysis["price_at_analysis"]` |
| `market_cap` | `full_report.market_data.market_cap` |
| `pe_ratio` | `full_report.market_data.pe_ratio` |
| `sector` | `full_report.market_data.sector` |
| `industry` | `full_report.market_data.industry` |
| `recommendation_confidence` | `full_report.analysis.confidence` (0-100) |
| `total_cost_usd` | `analysis["total_cost_usd"]` (0.01 for lite) |
| `standard_model` | `full_report.source` (= model_name; was hardcoded "claude-sonnet-4") |
| `full_report_json` | full `full_report` dict serialized as JSON for the report detail view |
| **All other ~74 columns** | NULL (lite path doesn't run debate / risk / enrichment / bias steps) |

## Why Path A over Path B (per research brief)

| Factor | Path A (extend `analysis_results`) | Path B (new `paper_trading_analyses`) |
|---|---|---|
| BQ migration | None | Required (operator --apply) |
| Reports History tab | Works immediately | New endpoint + UNION query |
| outcome_tracker | Picks up rows automatically | Needs UNION |
| Frontend | Zero changes (defensive null-checks already exist) | New "lite" tab or merge logic |
| Storage cost | Free (NULLs are 0 bytes in BQ Capacitor columnar) | Same, but extra table maintenance |
| Code change | 1 helper + 2 call sites | New table + new helper + new endpoint + UNION everywhere |

Path A is the standard data warehouse design pattern: single grain (ticker × date × recommendation), single table, NULL columns honestly indicate which pipeline ran.

## What changes for the operator at 09:30 ET tomorrow

1. **First scheduled cycle of the day fires:** N candidates analyzed → N rows added to `analysis_results`
2. **Reports page History tab:** previously stuck at older manual analyses (SNDK, GOOGL, AAPL, etc.) → now shows **today's autonomous picks** (COHR, KEYS, GLW, INTC, etc.) at the top, with company names, BUY/HOLD/SELL recommendations, and Claude's reasoning sentence.
3. **Click-through to report detail:** lite-path rows render the basics (ticker, score, summary, market data, full_report JSON) but Debate/Bull/Bear/Risk Assessment sections will be empty — that's correct because the lite analyzer doesn't run those steps. The frontend's existing null-checks should hide those sections gracefully.
4. **outcome_tracker** can now retrieve lite-row context for reflection generation in 7/30/90-day windows.

## Out of scope (per contract; Phase-2 follow-ups)

- New `analysis_source` discriminator column (operator --apply needed; `full_report.source` already carries the model name as a workaround)
- Frontend "Lite Analysis" badge on the report detail page
- New `paper_trading_analyses` table (Path B — explicitly rejected)
- `outcome_tracker` enhancements specific to lite rows

## Honest disclosure

- **Existing 11 paper-trading trades from earlier cycles** (booked before this fix) won't retroactively get analysis_results rows. Tomorrow's first cycle will be the first to populate them.
- **Report detail page for lite rows** will have many empty sections. Frontend currently returns the row as-is; sections that consume `bull_thesis` / `bear_thesis` / `analyst_summary` etc. will receive empty strings. If the operator clicks through and finds the page sparse, that's because the lite path produced ~14 fields not the 88 the full-Gemini path produces. A Phase-2 polish item would be a "Lite Analysis" badge to set expectations.
- **double-write guard** — when the full Gemini fallback runs (Claude API down or selected model is non-Claude), it ALREADY calls `bq.save_report` itself via `backend/api/analysis.py`. The `if settings.lite_mode` guard prevents us from writing twice. Lite mode is forced ON for paper trading (see line 215 in autonomous_loop), so this guard is the correct discriminator.

## What's next

1. Spawn fresh Q/A
2. On PASS: log → flip → archive → commit → restart backend so the new code ships
3. **Phase-23.1 plan now 11/11 cycles complete.**
