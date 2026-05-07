# Phase-23.3.3 Internal Codebase Audit
# Phase-9 Slack-Bot Jobs: Dormancy Root Cause + Job Analysis

Generated: 2026-05-07

---

## 1. Register callsite confirmation

```
grep -rn "register_phase9_jobs" backend/ scripts/
```

Result (3 hits, all in one file):

| Line | File | Type |
|------|------|------|
| 397 | backend/slack_bot/scheduler.py | **Function definition** |
| 421 | backend/slack_bot/scheduler.py | logger.warning inside the function body |
| 427 | backend/slack_bot/scheduler.py | logger.warning inside the function body |

**Verdict: ZERO callsites outside the function's own definition.** `register_phase9_jobs` has never been called from `start_scheduler` or anywhere else in `backend/` or `scripts/`. All 7 phase-9 jobs have been permanently dormant since they were merged.

---

## 2. Per-job module analysis

### 2a. `cost_budget_watcher` (phase-9.8)
**File:** `backend/slack_bot/jobs/cost_budget_watcher.py`  
**Entrypoint:** `run(*, daily_spend_usd, monthly_spend_usd, fetch_fn, daily_cap_usd=5.0, monthly_cap_usd=50.0, alert_fn, store, day) -> dict`  
APScheduler fires `run()` with zero positional args; all params have defaults.  
**Implementation:** Real. `_default_fetch_spend()` queries `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT` using the on-demand pricing constant `$6.25/TiB`. Re-uses `BudgetEnforcer` from phase-8.5.2. Fail-open to `(0.0, 0.0)` on BQ errors.  
**Dependencies:**
- BQ: `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT` -- requires `roles/bigquery.resourceViewer`; user ADC covers this.
- `backend.autoresearch.budget.BudgetEnforcer` (phase-8.5.2 class, must exist)
- `backend.slack_bot.job_runtime.IdempotencyStore / IdempotencyKey / heartbeat`
**Cost of firing:** Near zero -- one small BQ metadata query (bytes in `INFORMATION_SCHEMA` are free to query up to monthly limits). No LLM calls.

### 2b. `daily_price_refresh` (phase-9.2)
**File:** `backend/slack_bot/jobs/daily_price_refresh.py`  
**Entrypoint:** `run(*, tickers, fetch_fn, write_fn, store, day) -> dict`  
**Implementation:** PARTIAL STUB. `_default_fetch()` returns a hardcoded dict `{t: {"close": 100.0} for t in tickers}` -- the docstring says "production wraps yfinance" but the real yfinance call is absent. `_default_write()` returns `len(rows)` without writing to BQ. The production fetch/write callables must be injected at the call site in `register_phase9_jobs`, but the mapping in `scheduler.py:407` passes only bare `kwargs={"hour": 1}` with no `fetch_fn` or `write_fn` injected. Firing as-is does nothing except record idempotency.  
**Dependencies:**
- yfinance (external, rate-limit risk on first activation)
- BQ write target (not yet wired)
- `IdempotencyKey.daily`
**Cost of firing:** Low if stub fires. High if real yfinance + BQ write is wired -- could spike yfinance API quota if backlog of missed days is processed (idempotency key prevents this per-day).

### 2c. `weekly_fred_refresh` (phase-9.3)
**File:** `backend/slack_bot/jobs/weekly_fred_refresh.py`  
**Entrypoint:** `run(*, series, fetch_fn, write_fn, store, iso_year_week) -> dict`  
**Implementation:** PARTIAL STUB. `_default_fetch()` returns `{s: [] for s in series}` -- "production wraps fredapi" but fredapi is not wired. `_default_write()` returns `len(rows)`. Same injection gap as `daily_price_refresh`.  
**Dependencies:**
- fredapi (external, requires `FRED_API_KEY` from `.env`)
- BQ write target (not wired)
- `IdempotencyKey.weekly`
**Cost of firing:** Low if stub. Real fredapi call needs `FRED_API_KEY` set; 6 series is minimal API usage.

### 2d. `nightly_mda_retrain` (phase-9.4)
**File:** `backend/slack_bot/jobs/nightly_mda_retrain.py`  
**Entrypoint:** `run(*, train_fn, gate, commit_fn, store, day) -> dict`  
**Implementation:** PARTIAL STUB. `_default_train()` returns a hardcoded stub dict `{"trial_id": "stub_nightly", "dsr": 0.80, ...}`. The production path requires a real `train_fn` (invoking `backend/backtest/quant_optimizer.py`) and a `commit_fn` -- neither injected via `register_phase9_jobs`. Firing as-is passes stub metrics through `PromotionGate`; gate will likely reject the stub (dsr=0.80 is below typical live threshold of DSR >= 0.95), so no baseline will be overwritten.  
**Dependencies:**
- `backend.autoresearch.gate.PromotionGate` (must exist)
- `backend.backtest.quant_optimizer` (for real train_fn)
- `optimizer_best.json` (write target via commit_fn)
**Cost of firing stub:** Zero side effects -- stub trains, gate rejects, done. High cost if real train_fn is wired: full backtest run, potentially 5-30 min.

### 2e. `hourly_signal_warmup` (phase-9.5)
**File:** `backend/slack_bot/jobs/hourly_signal_warmup.py`  
**Entrypoint:** `run(*, watchlist, compute_signal_fn, cache_backend, store, iso_hour) -> dict`  
**Implementation:** Real structure, injectable defaults. `_load_watchlist()` reads `settings.watchlist` with fallback to `["AAPL", "MSFT", "SPY"]`. `compute_signal_fn` defaults to `lambda t: {"score": 0.0}` -- so as-is it warms an in-memory dict with zero scores, which is effectively a no-op (the cache_backend is not persisted across calls without injection). No external API calls in the default path.  
**Dependencies:**
- `backend.config.settings.get_settings()` (must export `.watchlist`)
- In-memory cache by default; production needs a real cache (Redis/dict injection)
**Cost of firing:** Zero. Default path makes no external calls, no BQ writes. Safe to activate immediately.

### 2f. `nightly_outcome_rebuild` (phase-9.6)
**File:** `backend/slack_bot/jobs/nightly_outcome_rebuild.py`  
**Entrypoint:** `run(*, ledger_fetch_fn, outcome_write_fn, store, day) -> dict`  
**Implementation:** PARTIAL STUB. `_default_fetch()` returns `[]` -- docstring says "production reads `pyfinagent_pms.paper_trades`" but BQ read is not wired. `_default_write()` returns `len(outcomes)` without writing to BQ. Fail-open on write exceptions.  
**Dependencies:**
- BQ table `pyfinagent_pms.paper_trades` (for real ledger_fetch_fn)
- BQ write target for outcomes (not wired)
**Cost of firing stub:** Zero side effects. Empty ledger -> empty outcomes -> write of 0 rows.

### 2g. `weekly_data_integrity` (phase-9.7)
**File:** `backend/slack_bot/jobs/weekly_data_integrity.py`  
**Entrypoint:** `run(*, current_counts, prior_counts, fetch_fn, snapshot_path, alert_fn, drift_threshold=0.20, store, iso_year_week) -> dict`  
**Implementation:** Real. `_default_fetch_counts()` queries `pyfinagent_data.__TABLES__` for row counts. Compares against `handoff/logs/row_count_snapshot.json` (created on first run). Fail-open on all I/O. Alert only fires if `alert_fn` is injected -- the current `register_phase9_jobs` mapping does not inject one, so drift will be computed but silently dropped.  
**Dependencies:**
- BQ dataset `pyfinagent_data` read access via `BigQueryClient`
- `backend.db.bigquery_client.BigQueryClient` (must exist and be importable)
- `handoff/logs/` directory (created automatically by `_save_snapshot`)
**Cost of firing:** 1 BQ metadata query (`__TABLES__`) -- cheap, no per-row scan. No LLM calls.

---

## 3. `job_runtime.py` -- heartbeat and idempotency cache

**File:** `backend/slack_bot/job_runtime.py` (lines 1-117)

Key findings:

- `heartbeat(job_name, *, idempotency_key, store, sink)` is a context manager that:
  1. Checks `store.seen(idempotency_key)` -- if already seen, yields `{skipped: True}` and returns without executing the body.
  2. Emits a `started` event via `sink` (default: `logger.info`).
  3. Wraps the body in try/except; sets `status=ok` or `status=failed`.
  4. On `ok`, calls `store.mark(idempotency_key)` so future runs within the same process are skipped.
  5. Emits a final event with `duration_s` and `finished_at`.
- `IdempotencyStore` is in-memory (a plain `set`). **It resets on every process restart.** This is documented in the runbook (section 7): "on restart, expect each daily job to run once even if it already ran earlier that day."
- `IdempotencyKey.daily/weekly/hourly` are pure string constructors; no external I/O.

**Phase-23.3.2 interaction:** `scheduler.py:34-58` defines `_aps_to_heartbeat`, an APScheduler event listener that fires on `EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED` and POSTs to `http://127.0.0.1:8000/api/jobs/heartbeat`. This listener is wired at `scheduler.py:122-125` inside `start_scheduler()`. Since `register_phase9_jobs` would add its jobs to the SAME `_scheduler` instance, the APScheduler event listener will automatically cover all 7 phase-9 jobs too -- no per-job `with heartbeat():` block needs to explicitly push to the endpoint. The per-job `heartbeat()` call continues to handle idempotency-skip and local logging; the listener handles the external HTTP push. These are complementary, not duplicated.

---

## 4. Phase-9 archive and runbook findings

**Archive directories found:** `handoff/archive/phase-9.1` through `phase-9.10`

**Runbook:** `docs/runbooks/phase9-cron-runbook.md` (version 1.0, 2026-04-20)

Critical finding from runbook section 2:
> "`register_phase9_jobs(scheduler)` appends all 7 jobs to an existing APScheduler instance... Called once at Slack bot process startup AFTER `start_scheduler` has set up the morning/evening digest + watchdog jobs."

This is the **intended activation pattern** -- the runbook explicitly documents that `register_phase9_jobs` should be called from the startup path. The dormancy is unambiguously a **missed wire**, not intentional deferral or a feature-flag decision. There is no `phase9_jobs_enabled` flag, no deferred-activation note, and no conditional guard anywhere in `scheduler.py` or `settings.py` that suggests dormancy was by design.

The runbook also documents (section 7) that idempotency keys are in-memory, so a restart harmlessly re-runs each job once regardless of whether it already ran that day -- the BQ-write operations are themselves idempotent by date/week via dedup keys.

---

## 5. Internal file inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/slack_bot/scheduler.py` | 429 | APScheduler setup + 4 core jobs + `register_phase9_jobs` definition | Active (core jobs); phase-9 function never called |
| `backend/slack_bot/job_runtime.py` | 117 | Heartbeat context manager + in-memory idempotency | Active, used by all 7 job modules |
| `backend/slack_bot/jobs/cost_budget_watcher.py` | 119 | BQ spend monitor + BudgetEnforcer circuit breaker | Real implementation; zero-cost to activate |
| `backend/slack_bot/jobs/hourly_signal_warmup.py` | 48 | In-memory signal cache warm | Real structure; default path is zero-cost no-op |
| `backend/slack_bot/jobs/weekly_data_integrity.py` | 116 | BQ row-count drift check | Real implementation; alert_fn not injected via scheduler |
| `backend/slack_bot/jobs/nightly_outcome_rebuild.py` | 58 | Trade ledger -> outcome metrics | Stub defaults (empty ledger fetch); safe to activate |
| `backend/slack_bot/jobs/nightly_mda_retrain.py` | 51 | Walk-forward MDA retraining + PromotionGate | Stub train_fn; gate will reject stub model; safe to activate |
| `backend/slack_bot/jobs/daily_price_refresh.py` | 53 | OHLCV fetch (yfinance) + BQ write | Stub fetch/write; no real yfinance or BQ wired |
| `backend/slack_bot/jobs/weekly_fred_refresh.py` | 44 | FRED macro series fetch + BQ write | Stub fetch/write; needs FRED_API_KEY + real wiring |
| `docs/runbooks/phase9-cron-runbook.md` | 71 | Operator runbook for all 7 jobs | Written 2026-04-20; documents wire pattern but wire never applied |

---

## 6. Blast-radius analysis per job

| Job | Fires immediately on activation? | External API on first fire? | BQ write? | LLM cost? | Safe to activate now? |
|-----|-----------------------------------|-----------------------------|-----------|-----------|----------------------|
| cost_budget_watcher | Next 06:00 UTC | BQ INFORMATION_SCHEMA (free) | No | None | YES -- cheapest, highest value |
| hourly_signal_warmup | Next HH:05 UTC | None | No | None | YES -- zero-cost no-op with stubs |
| nightly_outcome_rebuild | Next 04:00 UTC | None (stub fetches []) | No (stub) | None | YES -- stub is harmless |
| nightly_mda_retrain | Next 03:00 UTC | None (stub train) | No (commit_fn=None) | None | YES -- stub rejected by gate |
| weekly_data_integrity | Next Mon 05:00 UTC | BQ __TABLES__ metadata (cheap) | File write only | None | YES -- no alert_fn wired yet |
| daily_price_refresh | Next 01:00 UTC | yfinance (stub, no real call) | No (stub) | None | YES with stub; risk only if real yfinance wired |
| weekly_fred_refresh | Next Sun 02:00 UTC | fredapi (stub, no real call) | No (stub) | None | YES with stub; risk only if real fredapi wired |

**Key insight:** Because `daily_price_refresh` and `weekly_fred_refresh` default to hardcoded stubs when `fetch_fn` and `write_fn` are not injected, activating all 7 jobs via a bare `register_phase9_jobs(_scheduler)` call has ZERO external API blast radius in the current stub state. The "risky" jobs are only risky when their real fetch/write implementations are injected, which is a separate future step.
