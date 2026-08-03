# Contract -- step 82.0

**Step id:** 82.0 (phase-82)
**Priority:** P0
**Written:** 2026-07-31 (PLAN phase, before GENERATE)

## Research gate

`handoff/current/research_brief_82.0.md` -- **gate_passed: true**

```json
{"tier":"moderate","external_sources_read_in_full":8,"snippet_only_sources":14,
 "urls_collected":22,"recency_scan_performed":true,"internal_files_inspected":19,
 "coverage":{"audit_class":false,"rounds":1,"dry_rounds":0,"K_required":2,
 "new_findings_last_round":0,"dry":false},"gate_passed":true}
```

Audit-class is false, so `coverage.dry` is informational and does not gate.

Main independently re-verified the two load-bearing claims before acting on
them (not taken at face value):

- `grep -rn "ingest_macro"` over the repo excluding `.venv` and
  `handoff/archive` returns exactly the definition (`data_ingestion.py:297`)
  and one call site (`:373`). No scheduler.
- `settings.py:244` pins `backtest_end_date = "2025-12-31"`, and
  `data_ingestion.py:313` interpolates the passed `end_date` into the FRED
  `&observation_end=` parameter. Confirmed.
- `macro_regime.py:23` imports `get_macro_indicators` from
  `backend/tools/fred_data.py`, which calls the FRED HTTP API directly ->
  **live trading does not read the frozen table.** This REFUTES Main's
  earlier statement that the dead feed degrades live signal quality.
- `_get_existing_macro` (`data_ingestion.py:288-295`) catches bare
  `Exception` and returns `set()` (fail-OPEN); `_get_existing_price_dates`
  (`:98-103`) logs and re-raises (fail-CLOSED). Asymmetry confirmed.

## Hypothesis

The macro "freeze" is not a broken job. It is (a) a job that was never
scheduled, and (b) an end-date constant that caps every ingest at
2025-12-31. Therefore the repair is not "re-run the ingest" -- that inserts
zero rows and reports success. The repair is: sever the macro end-date from
`backtest_end_date`, create the scheduled caller that never existed, make
the staleness guard per-series (a global MAX cannot see a dead GDP behind a
live DGS10), record a run-receipt (append+dedupe makes a healthy no-op
indistinguishable from a dead job), harden the dedupe to fail closed before
running a large backfill, and land the vintage column WITH the backfill
because rows written without it can never be retro-attributed.

Predicted post-fix observable: `cache.preload_macro()` returns > 0 where it
returns 0 today.

> **ANNOTATION 2026-08-03 -- the prediction above was based on a FALSE
> premise.** `preload_macro` did not return 0: `historical_macro.date` is a
> STRING column and the staleness gate tested `isinstance(rd, datetime.date)`,
> so the refusal branch never ran. Pre-step it returned **4412**; post-fix
> **4729**. Nothing was hanging -- backtests were silently fed 212-day-old
> macro. Annotated, not rewritten: this is the dated PLAN-phase record.
> The `live_check` quoted below is IMMUTABLE and carries the same false
> premise; it is overturned in `handoff/current/live_check_82.0.md` rather
> than edited.

## Immutable success criteria (verbatim from .claude/masterplan.json)

1. macro ingestion end-date is severed from settings.backtest_end_date: a test that pins backtest_end_date to a past constant asserts the FRED observation_end used by ingest_macro is strictly later than that constant
2. a scheduled caller for macro ingestion is registered and a test asserts the registration exists by job id (the pre-fix repo has zero scheduled callers, so this test must fail against unmodified code)
3. _get_existing_macro fails CLOSED: a test injecting a BQ query exception asserts the exception propagates rather than returning an empty set
4. macro freshness is evaluated PER SERIES against a per-series SLA table rather than a single global MAX(date): a fixture with fresh DGS10/T10Y2Y and a stale GDP is reported UNHEALTHY (the pre-fix global-max logic reports it healthy, so this test must fail against unmodified code)
5. an ingestion run-receipt records each attempt and its outcome so a healthy zero-insert run is distinguishable from a job that never ran; a test asserts a receipt is written on a zero-row run
6. rows written by the backfill carry a populated vintage/realtime_start value; a test asserts newly-built macro rows include it

**Verification command:** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_0_macro_ingestion.py -q`

**live_check:** verbatim terminal output of a python -c call to backend.backtest.cache.preload_macro() against the live historical_macro table showing a return value > 0 (it returns 0 today), plus the BQ row showing MAX(date) advanced past 2025-12-31

## Plan

1. Add a macro-specific end date decoupled from `backtest_end_date`;
   default to today. Leave `backtest_end_date` itself untouched (backtests
   read it).
2. Harden `_get_existing_macro` to fail closed, mirroring
   `_get_existing_price_dates:98-103`. Do this BEFORE any backfill runs.
3. Add the vintage/`realtime_start` column and populate it on write.
4. Replace the global-max staleness gate with a per-series SLA table
   (DGS10/T10Y2Y 5d, FEDFUNDS/UMCSENT 70d, UNRATE 75d, CPIAUCSL 80d,
   GDP 225d -- per the brief; FRED dates monthly to month-START and
   quarterly to quarter-START).
5. Add an ingestion run-receipt so a zero-insert healthy run is
   distinguishable from a job that never ran.
6. Register a scheduled caller.
7. Run the backfill; capture the live_check evidence.
8. Spawn a fresh Q/A on the Workflow rail.

## Out of scope (queue as their own steps)

- `sortino.py:108` queries `pyfinagent_data.historical_macro` (wrong
  dataset -- the table is in `financial_reports`) for `DGS3MO`/`DTB3`
  (not in `FRED_SERIES`); its tier-1 lookup has always been dead,
  independent of staleness.
- `data_server.py:185` serves 7-month-old rows stamped `as_of: today`.
- The browser-driven alarm: `compute_freshness` is only reachable from HTTP
  handlers the frontend calls, so nothing pages when the feed dies.

## References

- `handoff/current/research_brief_82.0.md` (8 sources read in full)
- CLAUDE.md -- "Always call `cache.preload_macro()` or backtests hang"
- `backend/backtest/cache.py:26,203,232-251,380,387,399-417`
- `backend/backtest/data_ingestion.py:21-22,288-295,297,313,373`
- `backend/config/settings.py:244`
- `backend/services/cycle_health.py:51,507,565`
- `backend/services/macro_regime.py:23`; `backend/tools/fred_data.py:13`
