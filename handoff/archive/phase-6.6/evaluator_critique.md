# Q/A Critique -- phase-6.6 FOMC + Earnings Calendar Watcher

**qa_id:** qa_66_v1
**Timestamp:** 2026-04-19 UTC
**Cycle:** 1
**Verdict:** PASS

## 5-item harness-compliance audit (MANDATORY FIRST)

| # | Check | Result |
|---|-------|--------|
| 1 | Researcher spawned before contract, gate_passed=true, Recency scan present | PASS -- `phase-6.6-research-brief.md` (17825 bytes, mtime 10:30 today) envelope: `tier=moderate, external_sources_read_in_full=8, snippet_only_sources=10, urls_collected=18, recency_scan_performed=true, internal_files_inspected=8, gate_passed=true`. Recency scan section reports 5 new 2024-2026 findings. |
| 2 | Contract PRE-committed before generate | PASS -- `phase-6.6-contract.md` mtime epoch 1776587484; earliest calendar file mtime 1776587534 (50s later). Contract preamble explicitly explains phase-specific filename collision avoidance with the autonomous harness. |
| 3 | `phase-6.6-experiment-results.md` present + accurate | PASS -- 10330 bytes, 10:37; 11-file list matches filesystem; commands match outputs. |
| 4 | `harness_log.md` NOT yet appended for phase-6.6 | PASS -- most recent entry is `Cycle N+40 phase=6.5 result=PASS` (10:20 UTC). No phase-6.6 block. Autonomous-harness cycle-1 entry for phase-2.12 is separate context and not a protocol breach. |
| 5 | No verdict-shopping | PASS -- cycle-1, no prior Q/A. |

All 5 PASS.

## Deterministic checks

### A. Syntax (11 files)
All 11 files pass `ast.parse`:
```
OK backend/calendar/__init__.py
OK backend/calendar/registry.py
OK backend/calendar/normalize.py
OK backend/calendar/blackout.py
OK backend/calendar/watcher.py
OK backend/calendar/sources/__init__.py
OK backend/calendar/sources/finnhub_earnings.py
OK backend/calendar/sources/fed_scrape.py
OK backend/calendar/sources/fred_releases.py
OK scripts/migrations/add_calendar_events_schema.py
OK backend/tests/test_calendar_watcher.py
```

### B. Public-API import
`from backend.calendar import CalendarEvent, CalendarSource, CalendarFetchReport, register, get_sources, run_once, compute_fomc_blackout, compute_event_id; print('ok')` -> `ok`. All 8 names exported.

### C. Blackout
`compute_fomc_blackout(2025-03-18T19Z, 2025-03-19T19Z)` -> `2025-03-08T00:00:00+00:00 2025-03-20T00:00:00+00:00`. Exact match.

### D. BQ migration dry-run
Exit 0. DDL has all 14 columns (event_id, event_type, ticker, scheduled_at, window, fiscal_period_end, source, confidence, blackout_start, blackout_end, eps_estimate, revenue_estimate, fetched_at, metadata), PARTITION BY DATE(scheduled_at), CLUSTER BY event_type, ticker. "dry-run: no BigQuery writes executed." printed.

### E. Pytest
`pytest backend/tests/test_calendar_watcher.py -x -q` -> `10 passed in 0.01s`. (Contract minimum was >=7; delivered 10.)

### F. Registry independence
`calendar: 3 news: 4`. Three calendar sources register (finnhub_earnings, fed_scrape, fred_releases); four news sources register; no cross-leak. Separate module-level registry lists confirmed in `backend/calendar/registry.py` vs `backend/news/registry.py`.

## LLM judgment

### Contract alignment (15 criteria)
All 15 match: package structure (1), TypedDict 14 fields (2), sha256 event_id + case-normalized (3, normalize.py), CalendarSource Protocol (4), register/get_sources + clear helper (5), FinnhubEarningsSource bmo/amc/dmh mapping (6, finnhub_earnings.py:60-73), FedScrapeSource HTML scrape (7, fed_scrape.py), FredReleasesSource with `_RELEASE_WHITELIST` mapping CPI=10/PPI=20/NFP=50/GDP=53/retail=82/unemp=91 (8, fred_releases.py), compute_fomc_blackout correctness verified numerically (9), run_once orchestration+dedup+blackout+fail-open (10, watcher.py:135-202), CalendarFetchReport dataclass (11, watcher.py:97-104), migration script idempotent with --dry-run (12), __init__.py exports 8 names (13), fail-open per source and per normalize (14, watcher.py:160-173), non-goals honored -- no bigquery/paper_trader/FMP/EDGAR references (15).

### Research-gate tracing
Separate module tree: brief section "Architecture decision" + contract line 13. Finnhub primary: brief + contract line 14. AV secondary: contract line 15 (stubbed-not-implemented is honest). FRED for macro: contract line 17. Fed HTML scrape: contract line 16. Blackout formula: contract line 20 + blackout.py docstring citing ITC Markets. Dedup key: contract line 21. All traceable.

### Mutation-resistance
- (a) Swap bmo/amc mapping: `test_normalize_window_finnhub_hour_mapping` asserts each mapping explicitly (bmo->pre_open, amc->post_close, dmh->intraday) -- WOULD FAIL on swap.
- (b) Break event_id hashing: `test_event_id_deterministic_across_recompute` + `test_event_id_stable_across_ticker_casing` + FOMC-date-anchor test -- any change to the hashing function breaks all three.
- (c) Change Saturday offset: `test_blackout_tue_wed_fomc_march_2025` asserts `bs.date() == date(2025, 3, 8)` AND `bs.weekday() == 5`. Off-by-7-days mutation -> fails the date assertion; off-by-day mutation -> fails weekday assertion.

All three mutation classes covered.

### Fail-open completeness
- `finnhub_earnings.fetch`: try-block wraps GET + raise_for_status + json() (finnhub_earnings.py:48-54). Empty token short-circuits before network.
- `fed_scrape.fetch`: try-blocks on HTTP fetch (fed_scrape.py:60-62), HTML parse (fed_scrape.py:65-68), and per-row yield (fed_scrape.py:82-84). Three-layer fail-open.
- `fred_releases.fetch`: try wraps GET + raise_for_status + json (fred_releases.py:56-64). Empty API key short-circuits.
- `watcher.run_once`: try-blocks per-source fetch (watcher.py:160-165) and per-event normalize (watcher.py:167-173); errors accumulate in report.errors[] and iteration continues.
- `_apply_blackouts`: try wraps datetime.fromisoformat + compute_fomc_blackout (watcher.py:116-132). Defensive.

Verified: a single broken source cannot propagate to the others. Logger calls all ASCII-safe (`%s`, `%r`, no unicode) per `security.md`.

### Scope honesty
No BigQuery client imports in `backend/calendar/`. No `backend.paper_trader` imports. No FMP references. EDGAR not stubbed (contract says "for now -- wire in phase-6.8 or later"; Main simply omitted it, which is acceptable because the contract non-goal was "NOT implementing EDGAR EFTS backfill in this cycle"). Caveat #1 explicitly flags no live-API exercise -- honest.

### Name-collision risk
`grep -RE "^import calendar|^from calendar " backend/` -> 0 matches. `from backend.calendar import ...` is unambiguous because the import path is dotted. Risk is theoretical and manageable.

## Summary

- **violated_criteria:** []
- **violation_details:** []
- **certified_fallback:** false
- **checks_run:** ["5-item protocol audit", "syntax x11", "public-api import smoke", "blackout numeric check", "BQ migration dry-run", "pytest 10/10", "registry independence", "contract alignment", "research-gate trace", "mutation-resistance", "fail-open completeness", "scope honesty", "name-collision grep", "ASCII-logger audit"]
- **verdict:** PASS

No follow-up required. Main may proceed to append the harness_log entry, then flip masterplan status.
