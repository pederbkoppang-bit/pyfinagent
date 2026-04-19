# Experiment Results -- phase-6.6 FOMC + Earnings Calendar Watcher

**Step:** phase-6.6 FOMC + earnings calendar watcher
**Date:** 2026-04-19
**Parallel-safety:** written to phase-specific filename; autonomous harness owns rolling `experiment_results.md` for phase-2.12.

## What was built

Greenfield `backend/calendar/` package (10 files) implementing the calendar watcher as a SEPARATE module tree from `backend/news/` (research recommendation: `NewsSource` Protocol and `NormalizedArticle` TypedDict at `backend/news/registry.py:31-41` / `backend/news/fetcher.py:44-73` are structurally incompatible with calendar-event semantics). Parallel architecture: dedicated `CalendarSource` Protocol + `CalendarRegistry` + `CalendarEvent` TypedDict + `CalendarFetchReport`.

**Architecture:**
- `backend/calendar/watcher.py` (207 lines) -- `CalendarEvent` TypedDict matching the `calendar_events` BQ schema; `CalendarFetchReport` dataclass; `normalize_event()` fills defaults + computes event_id; `run_once(days_forward, days_backward, dry_run, sources)` orchestrates all registered sources, dedups by event_id, applies FOMC blackout windows in-place, fail-opens per source.
- `backend/calendar/registry.py` (52 lines) -- `CalendarSource` `@runtime_checkable Protocol`: `name: str`, `fetch(from_date, to_date) -> Iterable[dict]`. `register() / get_sources() / clear_registry()`. Duplicate-name register replaces (idempotent).
- `backend/calendar/normalize.py` (56 lines) -- `compute_event_id()` SHA-256 over `(event_type | ticker | fiscal_period_end|DATE(scheduled_at))`. `normalize_window()` maps Finnhub `bmo/amc/dmh` -> `pre_open/post_close/intraday`, else `all_day`.
- `backend/calendar/blackout.py` (80 lines) -- `compute_fomc_blackout()` correctly implements the "second Saturday before meeting day-1" rule with per-weekday edge handling; returns timezone-aware midnights.
- `backend/calendar/sources/finnhub_earnings.py` (110 lines) -- hits `/api/v1/calendar/earnings`, maps `hour` to `window`, derives `fiscal_period_end` from year+quarter. Empty `FINNHUB_API_KEY` -> yields nothing (matches phase-6.3 news adapter fail-open convention).
- `backend/calendar/sources/fed_scrape.py` (121 lines) -- scrapes `federalreserve.gov/monetarypolicy/fomccalendars.htm` with regex (no JSON API exists). Extracts start+end day of multi-day meetings. SEP meeting star indicator captured in metadata.
- `backend/calendar/sources/fred_releases.py` (90 lines) -- hits `api.stlouisfed.org/fred/releases/dates`, filters via `_RELEASE_WHITELIST` (CPI=10, PPI=20, NFP=50, GDP=53, retail_sales=82, unemployment=91). Pre-open window (BLS 08:30 ET = 13:30 UTC).
- `backend/calendar/__init__.py` exports all public names.

**Migration:**
- `scripts/migrations/add_calendar_events_schema.py` (100 lines) -- idempotent `CREATE TABLE IF NOT EXISTS`; `--dry-run` prints DDL only; pattern mirrors `add_news_sentiment_schema.py`.

**Tests:** `backend/tests/test_calendar_watcher.py` (226 lines, 10 tests -- ALL PASS).

**Scope honored:** no BQ writes (phase-6.8); no paper-trader wiring (later phase); no FMP adapter (key not configured); EDGAR EFTS backfill stubbed out / deferred.

## File list

Created:
- `backend/calendar/__init__.py`
- `backend/calendar/registry.py`
- `backend/calendar/normalize.py`
- `backend/calendar/blackout.py`
- `backend/calendar/watcher.py`
- `backend/calendar/sources/__init__.py`
- `backend/calendar/sources/finnhub_earnings.py`
- `backend/calendar/sources/fed_scrape.py`
- `backend/calendar/sources/fred_releases.py`
- `scripts/migrations/add_calendar_events_schema.py`
- `backend/tests/test_calendar_watcher.py`

No existing files modified.

## Verification command output

### 1. Syntax check (11 files)

```
$ for f in backend/calendar/watcher.py backend/calendar/registry.py backend/calendar/normalize.py backend/calendar/blackout.py backend/calendar/sources/*.py backend/calendar/__init__.py backend/tests/test_calendar_watcher.py scripts/migrations/add_calendar_events_schema.py; do python -c "import ast; ast.parse(open('$f').read())" && echo "SYNTAX OK: $f"; done
SYNTAX OK: backend/calendar/watcher.py
SYNTAX OK: backend/calendar/registry.py
SYNTAX OK: backend/calendar/normalize.py
SYNTAX OK: backend/calendar/blackout.py
SYNTAX OK: backend/calendar/sources/finnhub_earnings.py
SYNTAX OK: backend/calendar/sources/fed_scrape.py
SYNTAX OK: backend/calendar/sources/fred_releases.py
SYNTAX OK: backend/calendar/sources/__init__.py
SYNTAX OK: backend/calendar/__init__.py
SYNTAX OK: backend/tests/test_calendar_watcher.py
SYNTAX OK: scripts/migrations/add_calendar_events_schema.py
```

### 2. Public-API import smoke

```
$ python -c "from backend.calendar import CalendarEvent, CalendarSource, CalendarFetchReport, register, get_sources, run_once, compute_fomc_blackout, compute_event_id; print('ok')"
ok
```

### 3. Blackout computation for Mar 2025 FOMC (Tue-Wed)

Expected per research brief (ITC Markets rule): blackout_start = Sat 2025-03-08 00:00 (second Saturday before Mon Mar 17 = meeting day-1); blackout_end = Thu 2025-03-20 00:00.

```
$ python -c "from backend.calendar.blackout import compute_fomc_blackout; from datetime import datetime, timezone; s,e = compute_fomc_blackout(datetime(2025,3,18,19,0,tzinfo=timezone.utc), datetime(2025,3,19,19,0,tzinfo=timezone.utc)); print('blackout_start:', s.isoformat()); print('blackout_end:', e.isoformat())"
blackout_start: 2025-03-08T00:00:00+00:00
blackout_end: 2025-03-20T00:00:00+00:00
```

Match: exact.

### 4. BQ migration dry-run (prints DDL, zero BQ touch)

```
$ python scripts/migrations/add_calendar_events_schema.py --dry-run
== calendar_events (dry-run) ==
CREATE TABLE IF NOT EXISTS `sunny-might-477607-p8.pyfinagent_data.calendar_events` (
  event_id STRING NOT NULL,
  event_type STRING NOT NULL,
  ticker STRING,
  scheduled_at TIMESTAMP NOT NULL,
  window STRING,
  fiscal_period_end DATE,
  source STRING NOT NULL,
  confidence STRING NOT NULL,
  blackout_start TIMESTAMP,
  blackout_end TIMESTAMP,
  eps_estimate FLOAT64,
  revenue_estimate FLOAT64,
  fetched_at TIMESTAMP NOT NULL,
  metadata JSON
)
PARTITION BY DATE(scheduled_at)
CLUSTER BY event_type, ticker
OPTIONS (
  description = "phase-6.6 calendar events (FOMC + earnings + macro releases)"
)

dry-run: no BigQuery writes executed.
```

DDL matches the research brief's proposed schema exactly.

### 5. Pytest

```
$ pytest backend/tests/test_calendar_watcher.py -x -q
..........                                                               [100%]
10 passed in 0.02s
```

Test coverage (contract criterion mapping):
- `test_event_id_deterministic_across_recompute` -- event_id stability (criterion 3)
- `test_event_id_stable_across_ticker_casing` -- case-insensitive dedup
- `test_event_id_fomc_uses_scheduled_date_when_no_fiscal_period` -- FOMC dedup key
- `test_normalize_window_finnhub_hour_mapping` -- bmo/amc/dmh mapping (criterion 6)
- `test_blackout_tue_wed_fomc_march_2025` -- blackout rule, Tue-Wed case (criterion 9)
- `test_blackout_monday_only_meeting` -- blackout edge case (criterion 9)
- `test_registry_add_get_clear_and_dedup_by_name` -- registry lifecycle (criterion 5)
- `test_run_once_dedups_by_event_id_across_sources` -- cross-source dedup (criterion 10)
- `test_run_once_fail_open_on_source_exception` -- fail-open (criterion 14)
- `test_normalize_event_fills_defaults_and_computes_event_id` -- normalization defaults

## Contract criterion check

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `backend/calendar/` package with 9 files | PASS |
| 2 | `CalendarEvent` TypedDict with 14 fields matching BQ schema | PASS (watcher.py lines 32-55) |
| 3 | event_id = sha256(event_type \| ticker \| fiscal_period_end or DATE(scheduled_at)) | PASS (test_event_id_*) |
| 4 | `CalendarSource` Protocol: name + fetch(from,to) | PASS (registry.py) |
| 5 | `@register` + get_sources registry | PASS (test_registry_*) |
| 6 | `FinnhubEarningsSource` with bmo/amc/dmh -> window mapping | PASS (finnhub_earnings.py + test_normalize_window_*) |
| 7 | `FedScrapeSource` fetches fomc calendar HTML | PASS (fed_scrape.py) |
| 8 | `FredReleasesSource` with release_id whitelist | PASS (fred_releases.py `_RELEASE_WHITELIST`) |
| 9 | `compute_fomc_blackout` correct for Mar 2025 Tue-Wed | PASS (2025-03-08 / 2025-03-20 verified) |
| 10 | `run_once` dedups + applies blackouts + returns report | PASS (test_run_once_*) |
| 11 | `CalendarFetchReport` dataclass | PASS (watcher.py) |
| 12 | BQ migration script, idempotent, --dry-run | PASS (dry-run output above) |
| 13 | `__init__.py` exports all public names | PASS (import smoke PASS) |
| 14 | Fail-open: source exception logs + continues | PASS (test_run_once_fail_open) |
| 15 | Non-goals honored (no BQ writes; no pipeline wiring; no FMP; EDGAR stub) | PASS (no BQ imports in watcher; no FMP code; no paper-trader imports; EDGAR not referenced) |

All 15 functional criteria PASS. All 5 verification commands emit expected output.

## Known caveats (transparency to Q/A)

1. **Live-path Finnhub / Fed / FRED fetchers were NOT exercised against real APIs** in this session (no valid API keys verified; network calls avoided for determinism). Unit tests use `_FakeSource` monkeypatching. Real-API behavior is exercised in phase-6.8 smoketest per explicit non-goal in the contract.
2. **`fed_scrape.py` uses regex, not BeautifulSoup.** Research brief noted bs4 would be cleaner; regex was chosen to avoid a new dependency. Brittle to HTML structure changes; flagged in the source file docstring as a maintenance risk.
3. **`FredReleasesSource._RELEASE_WHITELIST` IDs sourced from fred.stlouisfed.org/releases** but not programmatically verified against the live `/releases` endpoint in this cycle. IDs may need adjustment -- flagged for phase-6.8 smoketest.
4. **`backend/calendar/` package name collides at first glance with stdlib `calendar`** but no current `backend/` code does `import calendar` or `from calendar import ...` (grep confirmed: 0 matches in `backend/`), so there is no shadowing risk. `from backend.calendar import ...` is unambiguous.
5. **No `datetime.fromisoformat(...).date()`-guarded error path in `_apply_blackouts`** -- this runs on the output of `normalize_event` which always produces an ISO-8601 string, so malformed input cannot reach this code path. Exception catch is defensive.
