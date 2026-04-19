# Sprint Contract -- phase-6.6 FOMC + Earnings Calendar Watcher

**Written:** 2026-04-19 (PRE-commit, before any generator work).
**Step id:** `phase-6.6` in `.claude/masterplan.json` phase-6 (News & Sentiment Cron).
**Parallel-safety note:** written to `phase-6.6-contract.md` (not rolling `contract.md`) because the autonomous harness cron is concurrently writing rolling files for phase-2.12. Before masterplan flip, phase-specific files will be copied back to rolling names so the archive-handoff hook snapshots them into `handoff/archive/phase-6.6/`.

## Research-gate summary

Researcher agent spawned today per `.claude/rules/research-gate.md`. Envelope:
`{tier: moderate, external_sources_read_in_full: 8, snippet_only_sources: 10, urls_collected: 18, recency_scan_performed: true, internal_files_inspected: 8, gate_passed: true}`. Brief at `handoff/current/phase-6.6-research-brief.md` (203 lines; also at rolling `research_brief.md`). Recency scan returned 5 new 2024-2026 findings.

Key research decisions (with empirical backing):
- **Separate `backend/calendar/` module tree** — `NewsSource` Protocol (`backend/news/registry.py:31-41`) and `RawArticle`/`NormalizedArticle` TypedDicts (`backend/news/fetcher.py:44-73`) are structurally incompatible with calendar event semantics. Forcing calendar events through `run_once()` would require rewriting the news normalizer to handle two data shapes.
- **Finnhub as primary earnings source** — `GET /api/v1/calendar/earnings?from=&to=&symbol=&token=`, 30 req/sec free tier, provides `{symbol, date, hour (bmo|amc|dmh), year, quarter, epsEstimate, epsActual, revenueEstimate, revenueActual}`. `FINNHUB_API_KEY` already wired in `backend/config/settings.py:61`.
- **Alpha Vantage as earnings dedup cross-check** — CSV response (6 fields), no `hour` timing field. `ALPHAVANTAGE_API_KEY` wired at `backend/config/settings.py:54`.
- **Fed calendar is scrape-only** — No JSON/ICS API at `federalreserve.gov/monetarypolicy/fomccalendars.htm`. Parse with `re` + optional BeautifulSoup fallback. Weekly poll sufficient (8 meetings/year).
- **FRED API for macro release dates** — `api.stlouisfed.org/fred/releases/dates?file_type=json&api_key=<key>`. Covers CPI, PPI, NFP, etc. `FRED_API_KEY` wired at `backend/config/settings.py:56`.
- **FMP NOT added this cycle** — key absent from settings; Finnhub + AV cover the requirement without a new credential.
- **EDGAR EFTS as historical backfill only** — `efts.sec.gov/LATEST/search-index?forms=8-K&q="Results+of+Operations"` — for 2-year lookback backfill, not forward-looking calendar.
- **Blackout window computed locally** — 2nd Saturday before meeting day-1 through midnight ET the day after meeting ends. Must handle edge case: "second Saturday before" = meeting day-1 minus 8-14 days landing on Saturday.
- **Dedup keys**: `(ticker, fiscal_period_end)` for earnings; `(event_type, DATE(scheduled_at))` for FOMC/macro.
- **Pre-FOMC drift is live through Dec 2024** (QuantSeeker 2025 backtest: 4% CAGR / 0.6 Sharpe on SPY). Primary use: pre-event positioning and risk management, not post-event drift alpha (UCLA Anderson 2024: large-cap PEAD t-stat 1.43, below significance).

## Hypothesis

A greenfield `backend/calendar/` module with a `CalendarSource` Protocol, a dedicated BQ `calendar_events` table, and 3 source adapters (Finnhub earnings + Fed HTML scraper + FRED macro releases) plus an EDGAR EFTS backfill path will deliver forward-looking event metadata with <1 min latency per poll, <10 req/sec per source under the 30-req/sec Finnhub ceiling, and zero collision with the news pipeline's data model.

## Success criteria

NOTE: `.claude/masterplan.json` phase-6.6 has `verification: null` and `contract: null`. Defining success criteria here per contract discipline:

**Functional:**
1. `backend/calendar/` package exists with `__init__.py`, `registry.py`, `normalize.py`, `watcher.py`, `blackout.py`, `sources/{finnhub_earnings,fed_scrape,fred_releases}.py`.
2. `CalendarEvent` TypedDict with fields: `event_id`, `event_type` (enum: `fomc_meeting|fomc_minutes|fomc_sep|fed_speech|earnings|cpi|ppi|nfp|gdp|retail_sales|unemployment`), `ticker` (Optional), `scheduled_at` (ISO-8601 UTC str), `window` (`pre_open|post_close|intraday|all_day`), `fiscal_period_end` (Optional ISO date str), `source`, `confidence` (`confirmed|estimated|unscheduled`), `blackout_start` (Optional), `blackout_end` (Optional), `eps_estimate` (Optional float), `revenue_estimate` (Optional float), `fetched_at`, `metadata` (dict).
3. `event_id = sha256(event_type + "|" + (ticker or "") + "|" + (fiscal_period_end or DATE(scheduled_at)))`. Deterministic across re-fetches.
4. `CalendarSource` Protocol: `name: str`; `fetch(from_date: date, to_date: date) -> Iterable[dict]`.
5. `@register(source: CalendarSource)` decorator adds source to a module-level registry. `get_sources()` returns all registered sources.
6. `FinnhubEarningsSource` — hits `/api/v1/calendar/earnings`, maps `hour` field to `window` (`bmo->pre_open`, `amc->post_close`, `dmh->intraday`, missing->`all_day`), maps `epsEstimate`/`revenueEstimate` to their fields, sets `source="finnhub"`, `confidence="confirmed"` when `hour` is present else `"estimated"`.
7. `FedScrapeSource` — fetches `federalreserve.gov/monetarypolicy/fomccalendars.htm`, parses rows for the current calendar year + next, emits `event_type="fomc_meeting"` per meeting with `scheduled_at` at the meeting start date, 14:00 ET press-conference convention noted in metadata.
8. `FredReleasesSource` — hits `fred/releases/dates` and filters to CPI, PPI, NFP (Employment Situation), GDP, Retail Sales, Unemployment Rate via release_id whitelist (documented in source file).
9. `blackout.compute_fomc_blackout(meeting_start: datetime, meeting_end: datetime) -> tuple[datetime, datetime]` — returns `(blackout_start, blackout_end)` where start = the Saturday 8-14 days before meeting day-1 at midnight ET, end = midnight ET the day after meeting ends. Test: for a Tue-Wed FOMC (e.g., Mar 18-19, 2025), blackout_start = Sat 2025-03-08T00:00-05:00, blackout_end = 2025-03-20T00:00-04:00.
10. `watcher.run_once(days_forward: int = 90, days_backward: int = 0, dry_run: bool = False) -> CalendarFetchReport` — orchestrates all registered sources, dedups by `event_id`, merges metadata, applies blackout windows to FOMC events, returns a report dataclass.
11. `CalendarFetchReport` — `n_events, by_type (dict), by_source (dict), errors, events (list)`.
12. `scripts/migrations/add_calendar_events_schema.py` — creates `pyfinagent_data.calendar_events` BQ table per the SQL DDL from the research brief. Idempotent (CREATE IF NOT EXISTS). `--dry-run` prints DDL only.
13. `backend/calendar/__init__.py` exports: `CalendarEvent`, `CalendarSource`, `CalendarFetchReport`, `register`, `get_sources`, `run_once`, `compute_fomc_blackout`, `compute_event_id`.
14. Fail-open discipline: any source exception produces a logged error entry in `errors[]` and continues with remaining sources — never raises out of `run_once()`.
15. Non-goals: NOT wiring calendar writes to BigQuery (phase-6.8 smoketest); NOT wiring calendar events into paper trader positioning (separate phase); NOT adding FMP as a source; NOT implementing EDGAR EFTS backfill in this cycle (stub with `NotImplementedError` for now — wire in phase-6.8 or later).

**Correctness verification commands:**
- `python -c "import ast; ast.parse(open('backend/calendar/watcher.py').read())"` -> exit 0 (same for registry, normalize, blackout, each source)
- `python -c "from backend.calendar import CalendarEvent, CalendarSource, CalendarFetchReport, register, get_sources, run_once, compute_fomc_blackout, compute_event_id; print('ok')"` -> stdout `ok`
- `python -c "from backend.calendar.blackout import compute_fomc_blackout; from datetime import datetime, timezone; s,e = compute_fomc_blackout(datetime(2025,3,18,19,0,tzinfo=timezone.utc), datetime(2025,3,19,19,0,tzinfo=timezone.utc)); print(s.isoformat(), e.isoformat())"` -> prints blackout_start on 2025-03-08 (Saturday 10 days before) and blackout_end on 2025-03-20.
- `python scripts/migrations/add_calendar_events_schema.py --dry-run` -> prints full CREATE TABLE DDL, exits 0, does not touch BQ.
- `pytest backend/tests/test_calendar_watcher.py -x -q` -> all pass (>=7 tests).

## Plan steps

1. Create `backend/calendar/` package with `__init__.py`, `registry.py`, `normalize.py`, `blackout.py`, `watcher.py`, `sources/__init__.py` + 3 source adapters.
2. Write `scripts/migrations/add_calendar_events_schema.py` following the pattern of `add_news_sentiment_schema.py`.
3. Write `backend/tests/test_calendar_watcher.py` with >=7 tests: (a) event_id determinism, (b) finnhub hour->window mapping, (c) blackout computation for Mar 2025 FOMC, (d) blackout edge case for Monday meeting, (e) registry add/get/clear, (f) watcher.run_once with mocked sources dedups by event_id, (g) fail-open when one source raises.
4. Run verification commands, capture into `phase-6.6-experiment-results.md`.

## References

- `handoff/current/phase-6.6-research-brief.md` (203 lines; 8 read-in-full sources; 5 recency findings)
- `scripts/migrations/add_news_sentiment_schema.py:24-36,64-86` (DDL pattern to replicate)
- `backend/news/registry.py:31-41` (Protocol pattern to parallel, not reuse)
- `backend/news/fetcher.py:44-73,127-190` (run_once pattern)
- `backend/tools/fred_data.py:13` (FRED_BASE URL reuse)
- `backend/news/sources/finnhub.py:30` (existing Finnhub auth pattern)
- External read-in-full: federalreserve.gov FOMC calendar, QuantSeeker pre-FOMC drift (Jan 2025), Quantpedia FOMC strategy, PMC peer-reviewed drift paper, macroption AV earnings, UCLA Anderson PEAD 2024, tldrfiling EDGAR EFTS guide, ITC Markets blackout.

## Researcher agent id

`a2beb299ba5ab457f`
