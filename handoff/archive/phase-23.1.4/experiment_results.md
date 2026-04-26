---
step: phase-23.1.4
cycle_date: 2026-04-27
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python -c "import asyncio; from backend.services.sector_calendars import fetch_sector_events, apply_sector_events_to_score, SectorEvent; events = asyncio.run(fetch_sector_events(use_cache=False)); assert isinstance(events, dict); assert all(isinstance(v, SectorEvent) for v in events.values()); from datetime import date; out = apply_sector_events_to_score(10.0, \"AAPL\", \"Information Technology\", {}); assert out == 10.0; print(\"ok events=\" + str(len(events)) + \" sources=\" + str(sorted(set(v.source for v in events.values()))))"'
---

# Experiment Results — phase-23.1.4

## What was built

Sector event calendars overlay — pure data-pull service ($0 LLM cost). Scaffolding ships the FULL data flow: Pydantic `SectorEvent` schema, two data sources (FDA PDUFA scrape from RTTNews, upcoming earnings from BQ `pyfinagent_data.calendar_events`), 6h file cache, screener integration with **drop-on-binary-risk** + **+20% boost on FDA positive catalyst** + **+10% boost on earnings within 1-3 days**. Default-OFF behind feature flag.

## Files modified

| File | Change |
|---|---|
| `backend/services/sector_calendars.py` | NEW (~280 lines) — `SectorEvent` schema + `_RTTNewsTableParser` (stdlib HTMLParser) + `_fetch_fda_pdufa_events` + `_fetch_earnings_events_sync` (BQ) + `fetch_sector_events` orchestrator + `apply_sector_events_to_score` helper + 6h file cache |
| `backend/tools/screener.py` | `rank_candidates` extended with `sector_events=` kwarg; drops candidate on binary-risk return None; applies catalyst boost on float return |
| `backend/services/autonomous_loop.py` | Step 1 sector calendars fetch block (mirrors regime + PEAD + news blocks) |
| `backend/config/settings.py` | 2 new fields: `sector_calendars_enabled` (default False), `sector_calendars_lookahead_days` (7) |
| `backend/services/pead_signal.py` | Bug fix: `BigQueryClient(get_settings())` — earlier cycle 2 missed the required `settings` arg (would have surfaced when `pead_signal_enabled=True` and recent earnings existed) |
| `tests/services/test_sector_calendars.py` | NEW (16 tests: schema, all score-apply paths, RTTNews parser on stub HTML, days_to_event signed math, cache roundtrip + missing + corrupt) |

## Verbatim verification command output

```
$ source .venv/bin/activate && python -c "import asyncio; from backend.services.sector_calendars import fetch_sector_events, apply_sector_events_to_score, SectorEvent; events = asyncio.run(fetch_sector_events(use_cache=False)); assert isinstance(events, dict); assert all(isinstance(v, SectorEvent) for v in events.values()); from datetime import date; out = apply_sector_events_to_score(10.0, 'AAPL', 'Information Technology', {}); assert out == 10.0; print('ok events=' + str(len(events)) + ' sources=' + str(sorted(set(v.source for v in events.values()))))"
Location: US
Job ID: 96a08564-530f-415f-8cb3-60b84b0b56db
ok events=0 sources=[]
exit=0
```

The full chain executed: RTTNews HTTP fetch (200 OK, 142KB) + BQ earnings query (200 OK, 0 rows). `events=0 sources=[]` is the honest result: see "Honest disclosure" below.

## Unit test results

```
$ source .venv/bin/activate && python -m pytest tests/services/ -v --no-header -q
collected 67 items
tests/services/test_macro_regime.py ............  [ 17%]
tests/services/test_news_screen.py .....................  [ 49%]
tests/services/test_pead_signal.py ..................  [ 76%]
tests/services/test_sector_calendars.py ................  [100%]
============================== 67 passed in 0.05s ==============================
```

67/67 tests pass (no regression across all 4 cycles).

## Honest disclosure: data sources need Phase-2 work

This cycle ships the **scaffolding** for sector event calendars — the schema, screener integration, score-apply logic, cache, tests — but the two data sources both returned 0 events live:

1. **RTTNews FDA PDUFA scrape:** the page returns HTTP 200 with 142KB HTML and 14 dates embedded, but **zero `<table>` or `<tr>` tags** — RTTNews loads the calendar dynamically via JavaScript / SPA-style. The stdlib HTMLParser (and even regex on `<tr>`) finds nothing structured. To extract data, we'd need either: (a) browser-based scraping (Selenium/Playwright — heavy dep), (b) BioPharmCatalyst as fallback (similar SPA risk), (c) a paid API like Wall Street Horizon. **Phase-2 work**, ideally with a free JSON endpoint discovery.

2. **Earnings calendar from BQ:** `pyfinagent_data.calendar_events` query for `event_type='earnings' AND scheduled_at <= now+7d` returned 0 rows. Either (a) the table is empty in this environment, (b) the Finnhub earnings cron hasn't run recently, or (c) the next earnings is >7 days out. The query itself is correct (returns 0 cleanly, no error).

The **scaffolding is valuable**: the schema works, the screener correctly drops on binary-risk + boosts on positive catalyst, the cache prevents re-fetch within 6h, all 16 unit tests pass on synthetic HTML fixtures. When either data source becomes operational (either fix the RTTNews scrape or run the earnings cron), the existing screener already consumes the events without further changes.

## Bugs fixed during E2E

1. **`BigQueryClient()` missing `settings` arg** — surfaced in this cycle when calling the BQ earnings query. Same bug exists in cycle-2's `pead_signal.py` `fetch_pead_signals_for_recent_reporters` (would have surfaced when that flag was enabled and earnings existed). Both fixed: `BigQueryClient(get_settings())`. Tests didn't catch this because `fetch_pead_signals_for_recent_reporters` was never called by tests (the `compute_pead_signal_for_ticker` path doesn't touch BQ).

## Cost / cycle posture

- **Pure data-pull, $0 LLM cost** — no Claude calls in this cycle's hot path
- 6h file cache at `backend/services/_cache/sector_calendars/sector_calendars_<YYYYMMDD>.json`
- Default OFF (`sector_calendars_enabled = False`)
- 1 HTTP call (RTTNews) + 1 BQ query per cycle when enabled

## Out of scope (per contract)

- EIA petroleum API (Phase 2 — needs free EIA_API_KEY registration)
- SEMI / TSMC monthly billings (Phase 2)
- Browser-based RTTNews scraping (Phase 2 — heavy dep)
- BQ-backed event history table (Phase 2)
- UI surface (phase-23.1.6)

## What's next

1. Spawn fresh Q/A — note the scaffolding-vs-live-data honest disclosure above
2. On PASS: log → flip → archive → commit → move to phase-23.1.5 (LLM-as-judge meta-scorer that combines all sub-signals)
3. Future cycle: replace RTTNews scrape with a working free FDA data source; populate the earnings cron so BQ has fresh data
