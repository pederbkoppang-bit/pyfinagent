---
step: phase-9.3
topic: Weekly FRED macro refresh job
tier: simple
date: 2026-04-20
---

## Research: Phase-9.3 Weekly FRED Macro Refresh

### Queries run (three-variant discipline)

1. Current-year frontier: "FRED API best practices rate limits 2026"
2. Last-2-year window: "fredapi Python library FRED API alternatives 2025 2026"; "FRED vintage data revision realtime ALFRED as-reported 2025"; "macro factor US equity returns PCEPI HY spread DXY credit spread backtesting 2025"
3. Year-less canonical: "FRED series weekly daily update cadence DGS10 VIXCLS UNRATE frequency"; "ISO week idempotency key macro data refresh revision handling"; "macro features equity signal Fama French TERM HML credit spread 2024 2025 factor model"; "FRED PCEPI BAMLH0A0HYM2 T10Y2Y macro signal equity 2024"; "FRED data revision frequency CPIAUCSL UNRATE how often revised benchmark revision"

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://pypi.org/project/fredapi/ | 2026-04-20 | Official PyPI page (doc) | WebFetch | v0.5.2 (May 2024), ~1.5k stars, Apache-2, pandas-backed, ALFRED support; moderate maintenance cadence (prior release July 2023) |
| https://nikhilxsunder.github.io/fedfred/index.html | 2026-04-20 | Official docs (vendor) | WebFetch | FedFred v3, actively maintained, async client, auto rate-limit (~120 req/min), multi-DataFrame (Pandas/Polars/Dask), caching built-in |
| https://nikhilxsunder.github.io/fedfred/installation/quickstart.html | 2026-04-20 | Official docs (vendor) | WebFetch | Usage: `FredAPI(api_key=...).get_series_observations("GDPC1")` returns DataFrame; rate-limit handled transparently |
| https://github.com/mortada/fredapi | 2026-04-20 | Source code / GitHub | WebFetch | 67 commits, 29 open issues, Apache-2; includes ALFRED vintage access via `get_series_all_releases()` and `realtime_start`/`realtime_end` params |
| https://github.com/gw-moore/pyfredapi | 2026-04-20 | Source code / GitHub | WebFetch | v0.10.2 (July 2025), 147 commits, 9 releases, CI/CD, covers all FRED endpoints including Maps, `series_collection` helper for multi-series batches |
| https://airbyte.com/data-engineering-resources/idempotency-in-data-pipelines | 2026-04-20 | Authoritative blog (vendor) | WebFetch | Idempotency keys should encode business-specific attributes; time-windowed jobs need state management and checkpointing; for revised data, incorporate version/timestamp into key rather than relying on window alone |

---

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://fred.stlouisfed.org/docs/api/fred/ | Official docs | 403 from fetch |
| https://fred.stlouisfed.org/docs/api/terms_of_use.html | Official terms | 403 from fetch |
| https://fred.stlouisfed.org/docs/api/fred/realtime_period.html | Official docs | 403 from fetch |
| https://alfred.stlouisfed.org/help | Official docs | 403 from fetch |
| https://fred.stlouisfed.org/series/VIXCLS | FRED series page | 403 from fetch |
| https://fred.stlouisfed.org/series/BAMLH0A0HYM2 | FRED series page | 403 from fetch |
| https://temporal.io/blog/idempotency-and-durable-execution | Blog | Fetched but content not directly applicable to scheduled-job revision scenarios |
| https://fred.stlouisfed.org/series/DGS10 | FRED series page | 403 from fetch; frequency confirmed daily via search snippet |
| https://alfred.stlouisfed.org/ | ALFRED landing | Confirmed from search snippet: archives every vintage; real-time period is start/end date pair on each observation |
| https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html | Academic data lib | Snippet; Fama-French factors use TERM proxy (10Y-1M spread) and default spread |
| https://fred.stlouisfed.org/series/T10Y2Y | FRED series page | 403; confirmed from snippet: daily, T10Y2Y = 10Y minus 2Y already computed |

---

### Recency scan (2024-2026)

Searched for 2024-2026 literature on FRED API changes, fredapi/pyfredapi releases, macro feature sets for equity backtesting, and idempotency patterns in data pipelines.

Findings:
- **pyfredapi v0.10.2** released July 2025 -- most recently updated Python FRED client, covering all endpoints including ALFRED vintage access.
- **FedFred v3** released (date not precisely stated but documentation badges are current) -- async-capable, auto rate-limiting, actively developed as of 2025-2026.
- **fredapi v0.5.2** May 2024 -- still usable but slower maintenance tempo; last two releases ~1 year apart.
- **CPIAUCSL revision pattern**: BLS revises CPI seasonally adjusted data every February for the prior 5 years; additionally revised each quarter when January/April/July/October data are published. Material for backtesting: a weekly refresh always fetches the current (revised) vintage, not the as-reported vintage.
- **UNRATE revision**: Annual benchmark revision, not intra-week. ISO-week key is stable for UNRATE.
- **HY OAS (BAMLH0A0HYM2)**: Daily series, available on FRED; confirmed as a common equity-regime signal in 2025-2026 practitioner literature.
- **T10Y2Y (yield curve spread)**: FRED already provides this as a pre-computed daily series, making DGS10-DGS2 pair somewhat redundant for spread use (though the raw levels are also useful).
- No peer-reviewed papers from 2024-2026 that significantly change best practice for weekly macro refresh patterns.

---

### Key findings

1. **FRED API rate limit is 120 requests per minute.** (Source: fedfred docs, search snippets from FRED rate-limit pages, 2026-04-20). fredapi does not auto-throttle; FedFred and pyfredapi do.

2. **fredapi is the incumbent but shows slow maintenance.** v0.5.2 released May 2024; predecessor v0.5.1 was July 2023. ALFRED vintage methods (`get_series_all_releases`, `realtime_start`/`realtime_end` kwargs) are functional. 1.5k stars, Apache-2. (Source: pypi.org/project/fredapi, github.com/mortada/fredapi, 2026-04-20)

3. **pyfredapi (v0.10.2, July 2025) is the most actively updated alternative.** Covers every FRED endpoint, `series_collection` module for multi-series batch fetches, returns pandas or JSON. (Source: github.com/gw-moore/pyfredapi, 2026-04-20)

4. **FedFred v3 adds async + auto rate-limiting.** Built-in caching, Polars/Dask support, async client for non-blocking pipelines. Best fit if the job eventually needs parallel fetching or async context. (Source: fedfred docs, 2026-04-20)

5. **ALFRED vs FRED distinction is critical for revision-aware backtesting.** FRED always serves the latest revised vintage. ALFRED archives the `realtime_start`/`realtime_end` pair so you can retrieve what was known on a specific date. The weekly refresh job uses FRED (latest vintage), which means any CPIAUCSL fetch post-February each year includes the seasonal re-bench. For backtesting correctness, ALFRED point-in-time access is required; for live signal generation, FRED latest is appropriate. (Source: alfred.stlouisfed.org snippets; github.com/mortada/fredapi ALFRED methods, 2026-04-20)

6. **ISO-week idempotency key is safe for monthly/annually revised series (UNRATE, CPIAUCSL) but creates a revision blind spot for daily series that can be restated intra-week.** DGS10, DGS2, VIXCLS, DFF are market-derived and rarely revised; UNRATE and CPIAUCSL are subject to scheduled revisions but these happen monthly or annually, never intra-week. The ISO-week key is therefore safe in practice for all six default series. (Source: BLS revision schedules via search, 2026-04-20)

7. **Series frequency mismatch: DGS10, DGS2, VIXCLS, DFF are daily; UNRATE, CPIAUCSL are monthly.** A weekly refresh captures the latest value for each, which is correct for signal construction (most recent observation wins), but callers consuming the stored rows should be aware that daily series will have 5 new observations per week while monthly series may have 0 new observations. (Source: FRED series search snippets, 2026-04-20)

8. **Canonical macro feature universe for US equity backtesting commonly extends to:** T10Y2Y (yield curve spread, already available on FRED as a pre-computed daily series), BAMLH0A0HYM2 (HY OAS, daily), and PCEPI (PCE inflation, monthly) beyond the current six. DXY is not natively on FRED but ICE/CBOE DXY futures are; practitioners often use DXY from a secondary source. (Source: practitioner search results, FRED series snippets, Fama-French factor library, 2026-04-20)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/jobs/weekly_fred_refresh.py` | 44 | DI-injectable weekly FRED refresh job; iso-week idempotency; fail-open | Active, phase-9.3 artifact |
| `backend/slack_bot/job_runtime.py` | 117 | `IdempotencyStore`, `IdempotencyKey`, `heartbeat` context manager | Active, phase-9.1 artifact |
| `tests/slack_bot/test_weekly_fred_refresh.py` | 43 | 3 tests: writes, iso-week dedup, no-live-fredapi guard | Active, phase-9.3 artifact |

Key observations (file:line anchors):

- `weekly_fred_refresh.py:11` -- `_DEFAULT_SERIES = ["DGS10", "DGS2", "VIXCLS", "DFF", "UNRATE", "CPIAUCSL"]` -- hardcoded; no config injection.
- `weekly_fred_refresh.py:22` -- `IdempotencyKey.weekly(JOB_NAME, iso_year_week=iso_year_week)` -- uses `job_runtime.IdempotencyKey.weekly` which formats as `{job_name}:{iso_year_week}` (job_runtime.py:51-56).
- `weekly_fred_refresh.py:36-41` -- `_default_fetch` returns empty dict stub; production wraps fredapi (comment says so, no import of fredapi in the module itself -- entirely DI-driven).
- `weekly_fred_refresh.py:24` -- `heartbeat` context manager: skips body if key already seen (job_runtime.py:92-98), marks key only on `status == "ok"` (job_runtime.py:112-113). A failed run therefore does NOT mark the key -- the job will retry next invocation. This is correct fail-open behavior.
- `job_runtime.py:39` -- `_GLOBAL_STORE = IdempotencyStore()` -- module-level singleton. Tests override by passing `store=` kwarg; production must also pass an external store or the in-memory singleton resets on process restart.
- `tests/slack_bot/test_weekly_fred_refresh.py:37-42` -- `test_no_live_fredapi` pops `fredapi` from `sys.modules` then confirms it was not re-imported. Confirms the no-live-call guard but does NOT test that a production `fetch_fn` wired to fredapi would work end-to-end.

---

### Consensus vs debate (external)

**Consensus:**
- fredapi is the de facto standard for quick FRED access in Python; pyfredapi and FedFred are credible, maintained alternatives.
- FRED always returns latest revised vintage; ALFRED is required for point-in-time backtesting. For live signal generation, FRED latest is appropriate.
- 120 req/min rate limit is well-established; modern libraries handle it transparently.
- ISO-week keys are safe for the six current series because none undergo intra-week revision.

**Debate / open questions:**
- Whether the current six series are a sufficient macro universe for equity signal work is contested in practitioner literature. T10Y2Y, BAMLH0A0HYM2, and PCEPI appear frequently alongside the six. No single authoritative "canonical" list exists.
- Async vs synchronous fetch: six series at weekly cadence is trivially within synchronous single-request budget. Async becomes relevant only if the universe grows significantly (>50 series).

---

### Pitfalls (from literature)

1. **Not throttling fredapi.** fredapi has no built-in rate limiter. If the series universe grows, batch calls without sleep/backoff will hit 120 req/min and return HTTP 429. FedFred or pyfredapi handle this automatically.
2. **FRED latest-vintage vs ALFRED as-reported confusion.** Using FRED for backtest feature construction introduces look-ahead bias for revised series (CPIAUCSL, UNRATE). Production signal construction with current data is fine; historical backtests must use ALFRED with `realtime_start` parameter.
3. **Module-level singleton `_GLOBAL_STORE`.** If the job runs in a long-lived process that restarts, the in-memory store resets and a week's idempotency is lost. Production requires an external persistent store (BQ, Redis).
4. **Series frequency mismatch.** Callers downstream of the stored rows must handle the fact that daily series produce 5 rows/week while monthly series may produce 0 new rows in a given week.
5. **Hardcoded series list.** `_DEFAULT_SERIES` at `weekly_fred_refresh.py:11` is not environment-injectable. Extending the universe requires a code change.
6. **No retrieval of `realtime_start` metadata.** The current `_default_fetch` stub (and implied production wrapper) does not capture the vintage date of each observation. Without it, future reconstruction of "what did we know on date X" is impossible.

---

### Application to pyfinagent (file:line mapping)

| Finding | File:line | Implication |
|---------|-----------|-------------|
| fredapi has no auto rate-limit | `weekly_fred_refresh.py:36` (`_default_fetch` production wrapper) | Production wrapper must add exponential backoff or swap to FedFred/pyfredapi |
| ALFRED required for backtest look-ahead safety | `weekly_fred_refresh.py:36-37` | Current stub writes latest-vintage only; a separate ALFRED-backed store should be maintained for backtest use |
| ISO-week key safe for all 6 series | `job_runtime.py:51-56` | No change needed; design is correct |
| `_GLOBAL_STORE` resets on restart | `job_runtime.py:39` | Production must inject BQ-backed or Redis-backed `IdempotencyStore` |
| DFF is daily (fed funds target) | `weekly_fred_refresh.py:11` | No issue -- latest value appropriate for signal |
| T10Y2Y, BAMLH0A0HYM2 absent | `weekly_fred_refresh.py:11` | Deferred expansion; not a blocker for phase-9.3 |

---

### Design critique (deferrals -- not blockers for phase-9.3 remediation)

The following are carry-forward items that do not block the immutable criterion for phase-9.3 but should be tracked:

1. **Hardcoded series universe** (`weekly_fred_refresh.py:11`): `_DEFAULT_SERIES` should eventually be config-injectable (e.g., from `settings` or a BQ config table) to allow adding T10Y2Y, BAMLH0A0HYM2, PCEPI without a deploy.
2. **Revision handling / ALFRED parity**: the production fetch wrapper should store the `realtime_start` vintage date alongside each observation row so that backtest pipelines can join on as-of date rather than always getting the latest revised value.
3. **Cadence stratification**: DGS10, DGS2, VIXCLS, DFF update daily; UNRATE, CPIAUCSL update monthly. A future improvement would tag each stored row with the source series frequency and allow downstream consumers to filter by "updated since last refresh."
4. **In-memory idempotency store**: `_GLOBAL_STORE` at `job_runtime.py:39` resets on process restart. Production wiring to BQ `job_heartbeat` table (already mentioned in the module docstring) is required for durability.
5. **No live-call integration test**: `test_no_live_fredapi` confirms fredapi is not imported, but there is no integration test that exercises a real `fetch_fn` implementation. A sandbox FRED API key test or VCR-cassette approach is advisable before go-live.
6. **HY OAS and yield curve spread absent from default universe**: BAMLH0A0HYM2 (daily, ICE BofA HY OAS) and T10Y2Y (daily, pre-computed spread) are standard equity-regime signals; their absence is a known gap to address in a follow-on step.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 fetched)
- [x] 10+ unique URLs total incl. snippet-only (17 unique URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (job, runtime, tests)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 11,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 3,
  "report_md": "handoff/current/phase-9.3-research-brief.md",
  "gate_passed": true
}
```
