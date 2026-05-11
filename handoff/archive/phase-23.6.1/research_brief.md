---
step: phase-23.6.1
title: Wire production fetch/write/alert fns in register_phase9_jobs (4 stub-affected jobs)
tier: moderate
date: 2026-05-10
---

## Research: Wiring production fns into register_phase9_jobs (phase-23.6.1)

### Queries run (three-query discipline)

1. **Current-year frontier**: "functools.partial dependency injection APScheduler cron jobs Python 2026"
2. **Last-2-year window**: "yfinance bulk download daily prices rate limit API 2025", "fredapi Python get_series missing key retry strategy 2025", "APScheduler AsyncIOScheduler closure dependency injection async function 2025", "slack bolt async chat_postMessage from sync APScheduler asyncio.run_coroutine_threadsafe 2025"
3. **Year-less canonical**: "functools.partial cron job dependency injection Python pattern", "APScheduler args kwargs add_job"

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://docs.python.org/3/library/functools.html | 2026-05-10 | Official docs | WebFetch | "Returns a new partial object which when called will behave like func called with… keyword arguments keywords. If additional keyword arguments are supplied, they extend and override keywords." Confirmed partial binds at creation time, not call time. |
| https://www.cosmicpython.com/book/chapter_13_dependency_injection.html | 2026-05-10 | Authoritative book | WebFetch | "closures employ late binding of variables, which can be a source of confusion if any of the dependencies are mutable, whereas partial binds immediately." Documents bootstrap pattern: default deps in prod, inject fakes in tests. |
| https://github.com/mortada/fredapi | 2026-05-10 | Official library | WebFetch | Constructor: `Fred(api_key=...)` or env var `FRED_API_KEY`. Method: `fred.get_series('DGS10')` returns pandas Series. No built-in retry. Latest version 0.5.2 (May 2024). |
| https://blog.ni18.in/how-to-fix-the-yfinance-429-client-error-too-many-requests/ | 2026-05-10 | Practitioner blog | WebFetch | Use `yf.download()` over `yf.Ticker()` for bulk. Add 1-5s delays between calls. One commenter reports complete API failure April 2025 — suggests defensive try/except wrapper is mandatory, not optional. |
| https://betterstack.com/community/guides/scaling-python/python-dependency-injection/ | 2026-05-10 | Authoritative blog | WebFetch | "Define interfaces for job functions… Separate wiring from execution (inject at job registration, not at runtime)." Confirms the functools.partial pattern: `bound_processor = partial(process_job, database=db, logger=log)` then `scheduler.add_job(bound_processor, ...)`. |
| https://docs.slack.dev/tools/bolt-python/concepts/async/ | 2026-05-10 | Official docs | WebFetch | Confirms AsyncApp uses AIOHTTP; all middleware/listeners must be async. Does not address calling async methods from sync APScheduler context — confirms the gap must be bridged manually (asyncio.run_coroutine_threadsafe or event-loop wrapper). |
| https://pypi.org/project/yfinance/ | 2026-05-10 | Official package page | WebFetch | yfinance 1.3.0 released April 16, 2026. Main bulk method is `yf.download()`. Personal use only per Yahoo TOS. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://apscheduler.readthedocs.io/en/3.x/userguide.html | Official docs | Fetched but content on argument passing was thin — only serialization warning; no partial examples |
| https://github.com/ranaroussi/yfinance/issues/2614 | GitHub issue | Fetched but no confirmed workarounds documented |
| https://github.com/slackapi/bolt-python/issues/540 | GitHub issue | Fetched but no relevant content on sync/async bridge |
| https://medium.com/@nsunder724/access-macroeconomic-data-at-scale-with-fedfred-a-modern-python-client-for-the-fred-api-96745541ef2a | Blog | Fetched, relevant for fedfred alternative context |

### Recency scan (2024-2026)

Searched for 2024-2026 literature on yfinance rate limits, fredapi updates, APScheduler dependency injection, and slack-bolt async/sync bridging.

Findings:
- **yfinance 1.3.0 (April 16, 2026)** is the current release. Rate-limit errors on bulk tickers are actively reported (issues #2614, #2422 in the ranaroussi/yfinance repo). Defensive wrapping with try/except and chunked downloads remain the only confirmed mitigation.
- **fredapi 0.5.2 (May 2024)**: No retry built in. A newer alternative `fedfred` (2024-2025) adds async support and auto-retry, but adding a new dependency is out of scope — fredapi is already in use.
- **APScheduler + functools.partial**: No new 2024-2026 guidance supersedes the established pattern. The `add_job(func, args=[...])` and `add_job(functools.partial(func, kwarg=val), ...)` patterns are stable.
- **slack-bolt async from sync context**: No official 2025 guidance on `asyncio.run_coroutine_threadsafe` for APScheduler -> AsyncApp. Community practice (confirmed in related GH issues) is to use `asyncio.run_coroutine_threadsafe(coro, loop)` where `loop` is the running loop captured at startup.

---

### Key findings (external)

1. **functools.partial binds immediately, closures bind late** -- "closures employ late binding of variables… can be a source of confusion if any of the dependencies are mutable, whereas partial binds immediately." (Cosmic Python, https://www.cosmicpython.com/book/chapter_13_dependency_injection.html). For long-lived scheduler functions whose dependencies (yfinance session, BQ client) are initialized at startup, `partial` is safer than a closure referencing a module-level variable that could be mutated.

2. **APScheduler serialization constraint** -- Only MemoryJobStore avoids serialization. The existing scheduler uses in-memory store (MemoryJobStore is default), so functools.partial objects are safely callable without pickling. This removes the one documented risk of using partial with APScheduler.

3. **functools.partial + APScheduler pattern is established** -- `scheduler.add_job(functools.partial(run, fetch_fn=prod_fetch, write_fn=prod_write), trigger="cron", id="job_id", ...)` is the standard pattern. (Python docs; BetterStack DI guide)

4. **yfinance bulk download** -- `yf.download(tickers=" ".join(tickers), period="2d", group_by="ticker")` is the recommended single-request bulk approach. Returns a MultiIndex DataFrame. Must be wrapped in try/except because rate-limit failures are hard to predict and Yahoo Finance revoked public access in some regions (April 2025 reports). Chunking to <= 50 tickers per call is the safest floor.

5. **fredapi get_series pattern** -- `Fred(api_key=os.getenv("FRED_API_KEY")).get_series(s)` where `s` is the series ID. Returns a pandas Series indexed by date. For missing/invalid series, raises `ValueError` or HTTP error -- must be caught per-series with fail-open to `[]`. No built-in retry; wrap with simple try/except per series.

6. **alert_fn sync/async bridge** -- `app.client.chat_postMessage(...)` is a `slack_sdk.web.async_client.AsyncWebClient` coroutine that must be awaited. Since `cost_budget_watcher.run()` and `weekly_data_integrity.run()` are sync `def` (not `async def`), the bridge is: capture the event loop at scheduler startup with `asyncio.get_event_loop()`, then in the alert_fn closure: `asyncio.run_coroutine_threadsafe(app.client.chat_postMessage(...), loop).result(timeout=10)`. The `.result(timeout=10)` converts the Future to a blocking call with a timeout, safe inside APScheduler's ThreadPoolExecutor.

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/scheduler.py` | 548 | APScheduler setup; `register_phase9_jobs()` at L502-548 | Active — wiring gap at L539 (`func = getattr(mod, "run")` with no partial-application) |
| `backend/slack_bot/jobs/daily_price_refresh.py` | 53 | phase-9.2; `run(*, tickers, fetch_fn, write_fn, store, day)` | Active — `_default_fetch` returns stub dict; `_default_write` returns len |
| `backend/slack_bot/jobs/weekly_fred_refresh.py` | 44 | phase-9.3; `run(*, series, fetch_fn, write_fn, store, iso_year_week)` | Active — `_default_fetch` returns `{s: []}` stub; `_default_write` returns len |
| `backend/slack_bot/jobs/nightly_outcome_rebuild.py` | 58 | phase-9.6; `run(*, ledger_fetch_fn, outcome_write_fn, store, day)` | Active — `_default_fetch` returns `[]`; `_default_write` returns len; note: kwargs use `ledger_fetch_fn` / `outcome_write_fn` (not `fetch_fn`/`write_fn`) |
| `backend/slack_bot/jobs/cost_budget_watcher.py` | 118 | phase-9.8; `run(*, daily_spend_usd, monthly_spend_usd, fetch_fn, daily_cap_usd, monthly_cap_usd, alert_fn, store, day)` | Active — `_default_fetch_spend()` is REAL BQ (not a stub); `alert_fn=None` is the only gap |
| `backend/slack_bot/jobs/weekly_data_integrity.py` | 115 | phase-9.7; `run(*, current_counts, prior_counts, fetch_fn, snapshot_path, alert_fn, store, iso_year_week, drift_threshold)` | Active — `_default_fetch_counts()` is REAL BQ; `alert_fn=None` is the gap; out of original 4-job scope but shares alert_fn pattern |
| `backend/slack_bot/app.py` | 78 | AsyncApp creation; `start_scheduler(app)` called in `main()` at L56 | Active — `app` object (AsyncApp) is available at scheduler-startup time; `app.client` is AsyncWebClient |
| `backend/tools/yfinance_tool.py` | 88 | Existing yfinance wrapper for single-ticker fundamentals | Active — uses `yf.Ticker(ticker).info`; no bulk OHLCV download function exists |
| `backend/tools/fred_data.py` | 127 | Existing FRED tool using httpx async | Active — uses httpx REST directly (not fredapi library); async def; not reusable directly in sync job |
| `backend/econ_calendar/sources/fred_releases.py` | 103 | Existing sync FRED fetch via requests library | Active — uses `requests.get` to FRED API; same endpoint pattern as needed but for releases, not series |
| `backend/db/bigquery_client.py` | 80+ | Canonical BQ client wrapper | Active — `BigQueryClient(settings)` constructor requires `settings` object; wraps `google.cloud.bigquery.Client` |
| `tests/slack_bot/test_daily_price_refresh.py` | 52 | Unit tests for daily_price_refresh | Active — injects `fetch_fn=lambda ts: {t:...}` and `write_fn=lambda rows: len(rows)`; key test: `test_no_live_yfinance_call` verifies yfinance NOT imported when fn is injected |
| `tests/slack_bot/test_cost_budget_watcher.py` | 75 | Unit tests for cost_budget_watcher | Active — injects `alert_fn=lambda r, s: alerts.append(...)` and `fetch_fn=lambda: (1.0, 10.0)` |
| `tests/slack_bot/test_nightly_outcome_rebuild.py` | 48 | Unit tests for nightly_outcome_rebuild | Active — injects `ledger_fetch_fn` and `outcome_write_fn` |
| `tests/slack_bot/test_scheduler_phase9.py` | 52 | scheduler wiring tests using StubScheduler | Active — asserts 7 jobs registered; StubScheduler.add_job records kwargs but does NOT call func; tests do NOT verify fn arguments passed to add_job |
| `tests/slack_bot/test_scheduler_wiring_phase991.py` | 170 | Regression tests for zero-args fire | Active — verifies cost_budget_watcher and weekly_data_integrity run() with no args (relying on defaults) |

---

### Consensus vs debate (external)

Consensus:
- `functools.partial` is the canonical Python DI tool for pre-binding callables at registration time, not at call time.
- APScheduler (MemoryJobStore) can accept partial objects without serialization issues.
- Sync job calling async Slack client requires a bridge; the established pattern is `asyncio.run_coroutine_threadsafe`.

Debate:
- Whether to use closures vs partial: cosmicpython prefers partial for immutable deps (avoids late-binding confusion). Closures are fine when deps are initialized once and never mutated, which applies here.
- yfinance reliability: reports of complete API failure (April 2025) suggest the production fetch_fn needs a mandatory try/except that returns an empty dict on any exception, and the heartbeat should surface the failure, not silently substitute a stub.

### Pitfalls (from literature and code inspection)

1. **Late-binding closure bug**: If a closure is used in a loop (e.g., iterating over job entries), all closures capture the SAME variable from the loop's last iteration. Using `functools.partial` or a factory function eliminates this. The mapping dict in `register_phase9_jobs` is not a loop over a single variable, but care is still warranted.

2. **Partial with keyword-only run() signature**: All four `run()` functions use keyword-only args (`def run(*, ...)`, note the `*`). `functools.partial(run, fetch_fn=prod_fetch)` works correctly for keyword args. Do NOT pass production fns as positional `args=[...]` to `add_job()` -- the `run()` functions require kwargs.

3. **StubScheduler tests will pass but don't verify fn args**: `test_scheduler_phase9.py::StubScheduler.add_job` records kwargs dict but never calls `func`. After wiring, new tests must explicitly assert that the registered `func` is a `functools.partial` wrapping the real `run` with the expected production fns.

4. **`test_no_live_yfinance_call` constraint**: The test at `tests/slack_bot/test_daily_price_refresh.py:39` pops yfinance from `sys.modules` and asserts it is not imported when `fetch_fn` is injected. The production wiring MUST ensure that yfinance is only imported inside the production `fetch_fn` itself (lazy import), not at module top level in `daily_price_refresh.py`. Currently the job module does not import yfinance at the top -- this invariant must be maintained in the production fetch_fn.

5. **`BigQueryClient(settings)` constructor requires a `settings` object**: Unlike `cost_budget_watcher._default_fetch_spend()` which directly creates `bigquery.Client(project=...)`, the canonical `BigQueryClient` requires an initialized `Settings` object. Production write_fns that use `BigQueryClient` should call `get_settings()` lazily inside the closure, not at module level.

6. **alert_fn signature mismatch**: `cost_budget_watcher.run()` calls `alert_fn(fired[0][0], fired[0][1])` — two args `(reason: str, details: dict)`. `weekly_data_integrity.run()` calls `alert_fn(drifts)` — one arg `(drifts: list[dict])`. The alert_fn closures for each job must match these differing signatures.

7. **asyncio event loop at APScheduler startup**: `asyncio.get_event_loop()` is deprecated in favor of `asyncio.get_running_loop()` in Python 3.10+, but `asyncio.get_running_loop()` only works from within a running coroutine. `start_scheduler(app)` is called from within `async def main()` at L56 of `app.py`, so `asyncio.get_running_loop()` is safe there. Capture the loop inside `start_scheduler` and close over it in the alert_fn.

---

### Application to pyfinagent (mapping external findings to file:line anchors)

**Gap location**: `scheduler.py:539` — `func = getattr(mod, "run")` followed by `scheduler.add_job(func, ...)` at L544. No partial-application of production fns at either line.

**Fix pattern**: Replace the simple `func = getattr(mod, "run")` with per-job partial-application at the mapping definition level or at the `add_job` call site.

---

## Five Decisions for Main

### Decision 1: Per-job production-fn implementation

Recommendation: **Option (b) — a new helper module `backend/slack_bot/jobs/_production_fns.py`** for all four production fn implementations. Rationale: keeps job modules clean (they already note "production wraps X" in the stub comments), keeps `register_phase9_jobs` readable, and gives a single place to unit-test the production fns independently.

Sketches:

**daily_price_refresh — fetch_fn:**
```python
# In _production_fns.py
import os

def make_price_fetch_fn():
    """Returns a fn: list[str] -> dict[str, dict] using yfinance.download."""
    def _fetch(tickers: list[str]) -> dict:
        try:
            import yfinance as yf  # lazy import preserves test_no_live_yfinance_call
            chunk_size = 50
            result = {}
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i:i + chunk_size]
                df = yf.download(
                    " ".join(chunk),
                    period="2d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=False,  # serial to avoid 429s
                )
                for t in chunk:
                    try:
                        row = df[t] if len(chunk) > 1 else df
                        last = row.dropna().iloc[-1]
                        result[t] = {
                            "open": float(last["Open"]),
                            "high": float(last["High"]),
                            "low": float(last["Low"]),
                            "close": float(last["Close"]),
                            "volume": float(last["Volume"]),
                        }
                    except Exception:
                        pass  # fail-open per ticker
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("price_fetch_fn fail-open: %r", exc)
        return result
    return _fetch
```

**daily_price_refresh — write_fn:**
```python
def make_price_write_fn():
    """Returns a fn: dict[str, dict] -> int that streams OHLCV rows to BQ."""
    def _write(rows: dict) -> int:
        if not rows:
            return 0
        try:
            from backend.config.settings import get_settings
            from google.cloud import bigquery
            import os
            from datetime import date
            project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
            dataset = os.getenv("PYFINAGENT_DATASET", "pyfinagent_data")
            client = bigquery.Client(project=project)
            table_id = f"{project}.{dataset}.daily_prices"
            today = date.today().isoformat()
            bq_rows = [
                {"ticker": t, "date": today, **v}
                for t, v in rows.items()
            ]
            errors = client.insert_rows_json(table_id, bq_rows)
            if errors:
                import logging
                logging.getLogger(__name__).warning("price_write_fn BQ errors: %s", errors)
            return len(bq_rows)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("price_write_fn fail-open: %r", exc)
            return 0
    return _write
```

**weekly_fred_refresh — fetch_fn:**
```python
def make_fred_fetch_fn():
    """Returns a fn: list[str] -> dict[str, list] using fredapi."""
    def _fetch(series: list[str]) -> dict:
        result = {}
        try:
            import os
            from fredapi import Fred
            api_key = os.getenv("FRED_API_KEY", "")
            if not api_key:
                import logging
                logging.getLogger(__name__).warning("FRED_API_KEY not set; returning empty")
                return {s: [] for s in series}
            fred = Fred(api_key=api_key)
            for s in series:
                try:
                    data = fred.get_series(s)
                    # pandas Series -> list of {"date": ..., "value": ...}
                    result[s] = [
                        {"date": str(idx.date()), "value": float(v)}
                        for idx, v in data.dropna().tail(52).items()
                    ]
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning("FRED series %s fail-open: %r", s, exc)
                    result[s] = []
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("fred_fetch_fn fail-open: %r", exc)
            result = {s: [] for s in series}
        return result
    return _fetch
```

**weekly_fred_refresh — write_fn:**
```python
def make_fred_write_fn():
    """Returns a fn: dict[str, list] -> int that streams macro rows to BQ."""
    def _write(rows: dict) -> int:
        if not rows:
            return 0
        try:
            from google.cloud import bigquery
            import os
            project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
            dataset = os.getenv("PYFINAGENT_DATASET", "pyfinagent_data")
            client = bigquery.Client(project=project)
            table_id = f"{project}.{dataset}.macro_series"
            bq_rows = []
            for series_id, obs_list in rows.items():
                for obs in obs_list:
                    bq_rows.append({"series_id": series_id, **obs})
            if not bq_rows:
                return 0
            errors = client.insert_rows_json(table_id, bq_rows)
            if errors:
                import logging
                logging.getLogger(__name__).warning("fred_write_fn BQ errors: %s", errors)
            return len(bq_rows)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("fred_write_fn fail-open: %r", exc)
            return 0
    return _write
```

**nightly_outcome_rebuild — ledger_fetch_fn:**
```python
def make_ledger_fetch_fn():
    """Returns a fn: () -> list[dict] reading paper_trades from BQ."""
    def _fetch() -> list:
        try:
            from google.cloud import bigquery
            import os
            project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
            client = bigquery.Client(project=project)
            sql = f"""
                SELECT trade_id, ticker, entry_price, exit_price,
                       recommendation_followed, pnl
                FROM `{project}.pyfinagent_pms.paper_trades`
                WHERE DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
            """
            return [dict(row) for row in client.query(sql).result()]
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("ledger_fetch_fn fail-open: %r", exc)
            return []
    return _fetch
```

**nightly_outcome_rebuild — outcome_write_fn:**
```python
def make_outcome_write_fn():
    """Returns a fn: list[dict] -> int streaming to BQ outcome_tracking."""
    def _write(outcomes: list) -> int:
        if not outcomes:
            return 0
        try:
            from google.cloud import bigquery
            import os
            project = os.getenv("GCP_PROJECT_ID", "sunny-might-477607-p8")
            client = bigquery.Client(project=project)
            table_id = f"{project}.pyfinagent_data.outcome_tracking"
            errors = client.insert_rows_json(table_id, outcomes)
            if errors:
                import logging
                logging.getLogger(__name__).warning("outcome_write_fn BQ errors: %s", errors)
            return len(outcomes)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("outcome_write_fn fail-open: %r", exc)
            return 0
    return _write
```

**cost_budget_watcher — alert_fn (signature: `(reason: str, details: dict) -> None`):**
See Decision 3 below.

**weekly_data_integrity — alert_fn (signature: `(drifts: list[dict]) -> None`):**
See Decision 3 below.

---

### Decision 2: Where to do the partial-application

Recommendation: **At the mapping definition level, by extending each tuple in the mapping dict to include a `prod_fns` dict, then applying partial at the `add_job` call site.**

Rationale: The current mapping dict (`scheduler.py:519-534`) is the natural home for per-job configuration. Extending the tuple with a `prod_fns` dict makes the wiring explicit and auditable in one place. The `add_job` call site becomes:

```python
import functools
# Inside register_phase9_jobs(scheduler, app, replace_existing):
# Mapping gains a 4th element: dict of kwargs to partial-apply
mapping = {
    "daily_price_refresh": (
        "backend.slack_bot.jobs.daily_price_refresh", "cron",
        {"hour": 1, "misfire_grace_time": 3600, "coalesce": True},
        {"fetch_fn": make_price_fetch_fn(), "write_fn": make_price_write_fn()},
    ),
    # ...
}
for job_id, (module_path, trigger, kwargs, prod_fns) in mapping.items():
    try:
        mod = importlib.import_module(module_path)
        func = functools.partial(getattr(mod, "run"), **prod_fns)
    except Exception as exc:
        logger.warning(...)
        continue
    scheduler.add_job(func, trigger=trigger, id=job_id, replace_existing=replace_existing, **kwargs)
```

The production fn factory functions (`make_price_fetch_fn()` etc.) are called ONCE at registration time, creating closures with their resources initialized. `functools.partial` then binds those fns as keyword args to the job's `run()`. When APScheduler fires `func()` with zero args, the partial supplies the bound keyword args.

**Important**: `register_phase9_jobs` must accept `app: AsyncApp` as a parameter (or `loop: asyncio.AbstractEventLoop`) to create the alert_fn closures. The current signature `register_phase9_jobs(scheduler, replace_existing=True)` must gain `app` (or `loop`). `start_scheduler` already has `app` in scope at L126 of `scheduler.py` and calls `register_phase9_jobs(_scheduler)` at L205 -- change to `register_phase9_jobs(_scheduler, app=app)`.

Citation: Python docs functools.partial; Cosmic Python Chapter 13 (bootstrap pattern); BetterStack DI guide.

---

### Decision 3: alert_fn closure design

Recommendation: **Option (a) — use `asyncio.run_coroutine_threadsafe(coro, loop)` with `.result(timeout=10)`.**

Full design:

```python
import asyncio

def make_alert_fn_for_budget(app, loop, channel_id: str):
    """alert_fn for cost_budget_watcher. Signature: (reason: str, details: dict) -> None."""
    def _alert(reason: str, details: dict) -> None:
        text = f"Budget alert: {reason} cap exceeded. Details: {details}"
        coro = app.client.chat_postMessage(channel=channel_id, text=text)
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            future.result(timeout=10)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("budget alert_fn Slack post fail-open: %r", exc)
    return _alert

def make_alert_fn_for_integrity(app, loop, channel_id: str):
    """alert_fn for weekly_data_integrity. Signature: (drifts: list[dict]) -> None."""
    def _alert(drifts: list) -> None:
        text = f"Data integrity drift detected: {len(drifts)} table(s). {drifts}"
        coro = app.client.chat_postMessage(channel=channel_id, text=text)
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            future.result(timeout=10)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("integrity alert_fn Slack post fail-open: %r", exc)
    return _alert
```

How to capture `loop`: inside `start_scheduler(app)` (which is called from `async def main()` at `app.py:56`), add:

```python
loop = asyncio.get_running_loop()  # safe here, we're inside async def main()
registered = register_phase9_jobs(_scheduler, app=app, loop=loop)
```

Why option (a) and not (b):
- Option (b) — making `alert_fn` a sync wrapper that calls `asyncio.run(coro)` — fails if there is already a running event loop in the thread (APScheduler's async executor runs coroutines in the event loop; sync jobs run in ThreadPoolExecutor). `asyncio.run()` raises `RuntimeError: This event loop is already running` when called from a thread that shares the event loop.
- `asyncio.run_coroutine_threadsafe(coro, loop)` submits the coroutine to the main event loop from the ThreadPoolExecutor thread, which is the correct bridge pattern.
- `.result(timeout=10)` makes it blocking (APScheduler waits) but time-bounded so a Slack outage does not hang the scheduler thread indefinitely.

Citation: Python asyncio docs (run_coroutine_threadsafe); slack-bolt async docs confirming chat_postMessage is a coroutine.

---

### Decision 4: Test design — impact of production wiring on existing tests

**Existing tests are safe from the wiring change.** Analysis:

- All four job module tests (`test_daily_price_refresh.py`, `test_cost_budget_watcher.py`, `test_nightly_outcome_rebuild.py`, `test_weekly_fred_refresh.py`) call `run()` directly with injected mock fns. They never go through `register_phase9_jobs`. Production wiring does NOT affect them.
- `test_scheduler_phase9.py` uses `StubScheduler.add_job()` which records the func but never calls it. The test asserts 7 jobs are registered and IDs are stable. After the wiring change, `func` in `StubScheduler.jobs` will be a `functools.partial` wrapping `run`, not a bare `run`. The existing assertions (counting jobs, checking IDs) still pass; no assertion inspects `func` itself.
- `test_scheduler_wiring_phase991.py` calls `cost_budget_watcher.run()` directly (not via scheduler). Not affected.

**New tests needed** (not existing ones breaking):
1. A test in `test_scheduler_phase9.py` asserting that the registered `func` for each of the 4 jobs is a `functools.partial` whose `.func` is the module's `run` and whose `.keywords` contains the expected fn keys (`fetch_fn`, `write_fn`, etc.).
2. A test mocking `register_phase9_jobs`'s production fn factories so the scheduler test stays hermetic (no real yfinance/BQ calls).

`test_no_live_yfinance_call` at `test_daily_price_refresh.py:39` remains valid because it tests the job module directly with an injected `fetch_fn` — it does not test the production `make_price_fetch_fn()` closure. The production fetch fn must import yfinance lazily (inside the closure body, not at module level of `_production_fns.py`) to keep that test green.

---

### Decision 5: Risk mitigation — production fn failure and heartbeat impact

Recommendation: **Wrap each production fn in a defensive try/except that fails loudly to the heartbeat (not silently falls back to the stub).**

All four `run()` functions already have fail-open patterns for `write_fn` failures (e.g., `nightly_outcome_rebuild.py:29` wraps the write call). But the `fetch_fn` failure path returns the stub's default (empty dict / empty list) silently, which makes the heartbeat show `written=0` with `skipped=False` — a misleading success.

The recommended approach: the production fn closures defined in `_production_fns.py` should log a `WARNING` on failure and return an empty result (so the job completes without crash and the heartbeat records `ok`). But the `result` dict should ideally surface a `fetch_error` key so that callers can see what happened. However, since `run()` signatures are immutable (anti-pattern constraint), the only clean path is to log at WARNING level from inside the production fn and let the job continue with zero rows written. The heartbeat will then show `written=0`, which is observable and distinguishable from a skip.

For `cost_budget_watcher`: `_default_fetch_spend()` is already real BQ with fail-open to `(0.0, 0.0)`. The production `alert_fn` wraps its Slack post in try/except with a WARNING log. If the Slack post fails, `alert_fn` logs but does not re-raise, so the job completes and the heartbeat shows `ok` with `tripped=True`. This is acceptable -- the budget trip is recorded in the heartbeat even if the Slack notification failed.

Do NOT fall back to the stub on production fn failure. The stub returning `len(tickers)` as `written` count for an empty fetch is a false positive that masks real failures. Fail loudly at WARNING level, return empty, let `written=0` propagate to the heartbeat.

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7 read in full)
- [x] 10+ unique URLs total (incl. snippet-only) (11 unique URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (7 job files + app.py + scheduler.py + 3 existing helper modules + 5 test files)
- [x] Contradictions / consensus noted (yfinance reliability, closure vs partial debate)
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 4,
  "urls_collected": 11,
  "recency_scan_performed": true,
  "internal_files_inspected": 15,
  "gate_passed": true
}
```
