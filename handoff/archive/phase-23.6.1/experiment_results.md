---
step: phase-23.6.1
title: Wire production fetch/write/alert fns in register_phase9_jobs (4 stub-affected jobs) — experiment results
date: 2026-05-10
verdict_class: PASS_PENDING_QA
verification_command: 'python3 tests/verify_phase_23_6_1.py'
---

# Experiment Results — phase-23.6.1

## What was done

3 source edits + 2 new files. Per researcher's 5 decisions
verbatim.

### File 1 (NEW) — `backend/slack_bot/jobs/_production_fns.py` (~300 lines)

8 factory closures, all with **lazy imports** so the module loads
without requiring yfinance / fredapi / google.cloud.bigquery to be
present at module-init time:

| Factory | Returns | External dep |
|---------|---------|--------------|
| `make_price_fetch_fn()` | `(tickers) -> dict[ticker, {"close": ...}]` | yfinance |
| `make_price_write_fn()` | `(rows) -> int` (rows written to `pyfinagent_data.price_snapshots`) | google.cloud.bigquery |
| `make_fred_fetch_fn()` | `(series) -> dict[series, [{date, value}, ...]]` | fredapi + `FRED_API_KEY` env |
| `make_fred_write_fn()` | `(rows) -> int` (rows written to `pyfinagent_data.fred_observations`) | google.cloud.bigquery |
| `make_ledger_fetch_fn()` | `() -> list[dict]` (last 30d paper_trades) | google.cloud.bigquery |
| `make_outcome_write_fn()` | `(outcomes) -> int` (writes to `financial_reports.outcome_tracking`) | google.cloud.bigquery |
| `make_alert_fn_for_budget(app, loop, channel)` | `(reason, state) -> None` (Slack post) | slack-bolt async |
| `make_alert_fn_for_integrity(app, loop, channel)` | `(drifts) -> None` (Slack post) | slack-bolt async |

**Failure mode:** every closure logs WARNING + returns empty (`{}`,
`[]`, `0`) on exception. Heartbeat surfaces `written=0` honestly so
the operator sees the real problem on the dashboard. NOT a silent
stub-fallback.

**Sync→async Slack bridge** (alert_fn closures): inside
`_post_slack_sync` use `asyncio.run_coroutine_threadsafe(coro,
loop).result(timeout=10)`. The slack-bot's main asyncio loop is
captured at scheduler-startup time; APScheduler executor threads
dispatch coros onto it safely.

### File 2 (edited) — `backend/slack_bot/scheduler.py`

- `import asyncio` added at the top.
- `register_phase9_jobs(scheduler, replace_existing=True, *, app=None, loop=None)` —
  new keyword-only `app` and `loop` parameters with `None` defaults.
  When BOTH are present, builds a `prod_fns_per_job` dict by
  invoking the 8 factories from `_production_fns`. Otherwise leaves
  the dict empty (preserves existing test-injection paths).
- `add_job` call site changed from `scheduler.add_job(func, ...)` to
  `func = functools.partial(run_fn, **prod_fns) if prod_fns else
  run_fn`. So when prod-wired, the registered callable is a
  `functools.partial` wrapping `run` with the expected fn keys.
- `start_scheduler` captures `running_loop = asyncio.get_running_loop()`
  (the slack-bot's `async def main` provides one) and passes
  `app=app, loop=running_loop` to `register_phase9_jobs`.

### File 3 (NEW) — `tests/slack_bot/test_phase9_production_wiring.py` (14 tests)

| Test | Asserts |
|------|---------|
| `test_register_accepts_app_and_loop_kwargs` | signature + None defaults |
| `test_register_without_app_returns_bare_run` | None-app falls back to bare `run` (test injection still works) |
| `test_register_with_app_uses_functools_partial` | with app+loop, all 5 prod-wired jobs are `functools.partial` with expected `prod_fns` keys |
| `test_register_with_app_but_none_loop_returns_bare_run` | defensive — partial requires BOTH app + loop |
| `test_make_*_returns_callable` (×8) | each factory returns a callable; lazy-import discipline |
| `test_alert_fn_for_budget_swallows_post_failure` | Slack post exception inside closure does NOT raise |
| `test_alert_fn_for_integrity_handles_empty_drifts` | empty drifts → no Slack post |

### File 4 (NEW) — `tests/verify_phase_23_6_1.py`

6-check verifier:
1. Factories present + lazy imports preserved.
2. `register_phase9_jobs` signature has `app` + `loop` with `None` defaults.
3. `start_scheduler` captures running loop + passes app+loop.
4. New wiring unit tests pass (14).
5. ALL slack_bot tests pass (no regression — 72 total).
6. Live `/api/jobs/all` shows 11/11 slack_bot non-manifest (bridge intact).

## Verbatim verifier result

```
$ python3 tests/verify_phase_23_6_1.py
=== phase-23.6.1 verifier ===
  [PASS] factories present + lazy imports: all 8 factories present + lazy imports
  [PASS] register_phase9_jobs signature: signature accepts app+loop with None defaults (venv-import check)
  [PASS] start_scheduler passes loop: start_scheduler captures loop and passes app+loop
  [PASS] wiring unit tests pass: 14 passed in 0.10s
  [PASS] all slack_bot tests pass: 72 passed in 0.86s
  [PASS] live /api/jobs/all unchanged: 11/11 slack_bot non-manifest

PASS (6/6)
EXIT=0
```

## Operational deploy

Restarted slack-bot daemon (`pkill -f slack_bot.app` + `nohup
.venv/bin/python -m backend.slack_bot.app`). New PID 49858.
Startup log:

```
2026-05-10 10:09:48,285 INFO apscheduler.scheduler: Scheduler started
2026-05-10 10:09:48,288 INFO backend.slack_bot.scheduler: phase-9 jobs registered: ['daily_price_refresh', 'weekly_fred_refresh', 'nightly_mda_retrain', 'hourly_signal_warmup', 'nightly_outcome_rebuild', 'weekly_data_integrity', 'cost_budget_watcher']
2026-05-10 10:09:48,927 INFO slack_bolt.AsyncApp: ⚡️ Bolt app is running!
```

No fail-open warnings during phase-9 wiring — the production-fn
construction succeeded for all 5 affected jobs.

## Sibling verifier sweep — 27/27 green

```
=== Sibling verifier sweep ===
PASS: 27 / FAIL: 0
```

(All phase-23.5.* + phase-23.6.* verifiers exit 0.)

## Operator-visible behavior changes

After this deploy, the next fires of these jobs will perform real
work instead of running stubs:

- **daily_price_refresh** (next fire: 01:00 CEST tomorrow) — yfinance
  bulk download for `["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]` →
  `pyfinagent_data.price_snapshots` BQ insert.
- **weekly_fred_refresh** (next fire: Sunday 02:00 ET) — fredapi
  fetch of `["DGS10", "DGS2", "VIXCLS", "DFF", "UNRATE", "CPIAUCSL"]`
  → `pyfinagent_data.fred_observations` BQ insert. Requires
  `FRED_API_KEY` env (already in operator's `backend/.env`).
- **nightly_outcome_rebuild** (next fire: 04:00 CEST) — BQ read
  paper_trades 30d → outcome compute → BQ write
  `financial_reports.outcome_tracking`.
- **cost_budget_watcher** (next fire: 06:00 CEST) — already did
  real BQ read; now ALSO posts a Slack alert when budget tripped.
- **weekly_data_integrity** (next fire: Monday 05:00 ET) — already
  did real BQ read; now ALSO posts a Slack alert when drift
  detected.

If a production fn fails (yfinance 429, BQ quota, missing FRED key,
table not yet created), the closure logs WARNING and returns empty
— the heartbeat shows `written=0` so the dashboard surfaces the
real failure rather than masking it.

## Findings to surface to the operator

1. **Two new BQ tables may need to be created** before the writes
   succeed cleanly:
   - `pyfinagent_data.price_snapshots(ticker STRING, date STRING, close FLOAT, recorded_at TIMESTAMP)`
   - `pyfinagent_data.fred_observations(series STRING, date STRING, value FLOAT, recorded_at TIMESTAMP)`
   The closures log WARNING + return 0 on insert failure. Operator
   can create via standard BQ migration scripts when ready.
2. **`outcome_tracking` is in `financial_reports`** (not pyfinagent_data
   per CLAUDE.md). Confirmed via prior phase-23 audits.
3. **alert_fn posts go to `settings.slack_channel_id`** — same
   channel as digests + watchdog.
4. **First-fire timing** — most affected jobs are nightly/weekly cron;
   real-world impact won't be visible until tomorrow morning's
   `/api/jobs/all` post-fire status.

## What this step does NOT do

- Create the two new BQ tables (operator-action; standard migration).
- Add yfinance retry/backoff (basic try/except on first failure;
  closure logs WARNING + returns empty).
- Touch hourly_signal_warmup compute_signal_fn (different beast;
  per researcher in 23.5.10).
- Touch nightly_mda_retrain (already runs real `train_fn`; only
  PromotionGate threshold is a separate concern).
- Modify the existing per-job tests' mock-injection paths.

## Artifact files

- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/phase-23.6.1-research-brief.md`
- `tests/verify_phase_23_6_1.py` (NEW)
- `tests/slack_bot/test_phase9_production_wiring.py` (NEW; 14 tests)
- `backend/slack_bot/jobs/_production_fns.py` (NEW; ~300 lines, 8 factories)
- `backend/slack_bot/scheduler.py` (extended `register_phase9_jobs` +
  `start_scheduler`; +1 `import asyncio` + ~50 lines)

## How to re-run

```bash
cd /Users/ford/.openclaw/workspace/pyfinagent
python tests/verify_phase_23_6_1.py
```
