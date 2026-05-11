---
step: phase-23.6.1
title: Wire production fetch/write/alert fns in register_phase9_jobs (4 stub-affected jobs)
cycle_date: 2026-05-10
harness_required: true
verification: 'python3 tests/verify_phase_23_6_1.py'
research_brief: handoff/current/phase-23.6.1-research-brief.md
---

# Contract — phase-23.6.1

## Hypothesis

A new `backend/slack_bot/jobs/_production_fns.py` module exposes
factory closures that produce real `fetch_fn` / `write_fn` /
`alert_fn` injections for the 4 stub-affected phase-9 jobs.
`register_phase9_jobs()` extends its mapping with a `prod_fns`
dict per job and applies `functools.partial(run, **prod_fns)` at
the `add_job` call site. After the edit + slack-bot daemon
restart, the 4 phase-9 cron fires perform real BQ /
yfinance / FRED I/O AND budget breaches surface as Slack alerts.

## Research-gate summary

`researcher` agent `aa4d22c122d48043b` (tier=moderate) returned
`gate_passed: true`:
- 7 external sources read in full (Python functools.partial,
  Cosmic Python ch.13 DI, fredapi GitHub, yfinance 429 blog,
  BetterStack DI guide, Slack Bolt async docs, yfinance PyPI).
- 4 snippet-only + 7 read-in-full = 11 URLs (≥10 floor).
- Recency scan 2024-2026 performed.
- Three-query discipline followed.
- 15 internal files inspected.

Brief: `handoff/current/phase-23.6.1-research-brief.md`.

## Five researcher decisions (load-bearing)

1. **Production-fn location:** new module
   `backend/slack_bot/jobs/_production_fns.py` with factory
   closures (`make_price_fetch_fn`, `make_price_write_fn`,
   `make_fred_fetch_fn`, `make_fred_write_fn`,
   `make_ledger_fetch_fn`, `make_outcome_write_fn`,
   `make_alert_fn_for_budget`, `make_alert_fn_for_integrity`).
   Lazy imports (yfinance inside closure body, not at module top)
   preserve the existing `test_no_live_yfinance_call` constraint.

2. **Partial-application site:** extend each mapping tuple in
   `register_phase9_jobs()` (`scheduler.py:519-534`) with a 4th
   element (`prod_fns` dict). Apply
   `functools.partial(getattr(mod, "run"), **prod_fns)` at the
   `add_job` call site. `register_phase9_jobs` gains `app:
   AsyncApp` and `loop: asyncio.AbstractEventLoop` parameters;
   `start_scheduler` passes both at line 205.

3. **alert_fn closure (sync→async bridge):** use
   `asyncio.run_coroutine_threadsafe(app.client.chat_postMessage(...),
   loop).result(timeout=10)` inside a sync closure. Capture
   `loop = asyncio.get_running_loop()` inside `start_scheduler`
   (which IS called from `async def main()`). Do NOT use
   `asyncio.run()` — raises `RuntimeError` when loop already
   running.

4. **Test impact:** zero existing tests break. `StubScheduler`
   in `test_scheduler_phase9.py` records but never calls `func`,
   so partial vs bare `run` is invisible to current assertions.
   Add new tests: assert `func.func is run` and `func.keywords`
   contains expected fn keys.

5. **Failure mode:** production fn closures log WARNING and
   return empty results on failure (NOT silent stub-fallback —
   that would defeat the purpose). Heartbeat surfaces
   `written=0` honestly. `alert_fn` wraps Slack post in
   try/except + logs WARNING on failure — budget trip is still
   recorded in the heartbeat dict.

## Immutable success criteria (verbatim — DO NOT EDIT)

```
python3 tests/verify_phase_23_6_1.py
```

The verifier exits 0 only when:

1. `backend/slack_bot/jobs/_production_fns.py` exists with all 8
   factory functions + lazy imports of yfinance / fredapi.
2. `backend/slack_bot/scheduler.py:register_phase9_jobs()` accepts
   `app` and `loop` parameters AND uses `functools.partial` at
   the `add_job` call site.
3. `start_scheduler` captures the running loop and passes it to
   `register_phase9_jobs`.
4. New unit tests in `tests/slack_bot/test_phase9_production_wiring.py`
   pass — each asserts the registered job is a `functools.partial`
   wrapping `run` with expected `prod_fns` keys.
5. Existing tests still pass:
   `tests/slack_bot/test_daily_price_refresh.py`,
   `test_weekly_fred_refresh.py`, `test_nightly_outcome_rebuild.py`,
   `test_cost_budget_watcher.py`,
   `test_weekly_data_integrity.py`, `test_hourly_signal_warmup.py`,
   `test_nightly_mda_retrain.py`,
   `test_digest_url_semantics.py`,
   `test_evening_digest_envelope_coerce.py`,
   `test_watchdog_alert_semantics.py`.
6. Live `/api/jobs/all` shows all 11 slack_bot jobs still
   `status != "manifest"` (no regression of the 23.5.2.5 bridge).

## Plan steps

1. (DONE — RESEARCH) `gate_passed: true`.
2. (DONE — PLAN) This contract.
3. **GENERATE phase:**
   a. Create `backend/slack_bot/jobs/_production_fns.py` with 8
      factory closures per researcher's decision 1. Each factory:
      - Lazy-imports its dep inside the closure body.
      - Returns a sync closure matching the job's expected
        signature (`fetch_fn(args) -> dict|list`, etc.).
      - Wraps the underlying call in try/except; on failure logs
        WARNING and returns empty (`{}`, `[]`, etc.).
   b. Edit `backend/slack_bot/scheduler.py`:
      - Import `functools` (if not already).
      - Extend `register_phase9_jobs(scheduler, replace_existing,
        app=None, loop=None) -> list[str]`.
      - Build `prod_fns` per job inside the function body (using
        the factories from `_production_fns`; only when `app` and
        `loop` are not None).
      - Use `functools.partial(run, **prod_fns)` at the `add_job`
        call site.
      - In `start_scheduler`, capture
        `loop = asyncio.get_running_loop()` and pass `app, loop`
        to `register_phase9_jobs`.
   c. Add `tests/slack_bot/test_phase9_production_wiring.py`:
      - Assert `register_phase9_jobs` accepts `app` and `loop`
        kwargs.
      - Inject a `StubScheduler` and assert the registered `func`
        is a `functools.partial` wrapping `run` with expected
        keys per job.
      - Assert factories return callables with proper signature
        (lazy import only fires when called).
      - Assert when `app=None`, the partial-application gracefully
        falls back to the bare `run` (so unit tests that don't
        provide `app` still work).
   d. Add `tests/verify_phase_23_6_1.py` — 6-check verifier.
   e. Restart slack-bot daemon to deploy (`pkill -f slack_bot.app`
      + `nohup`).
   f. Run sibling verifier sweep — 18 prior phase-23.5 verifiers +
      23.6.0 must all stay green.
4. **EVALUATE phase:** spawn fresh `qa` agent with heightened
   scrutiny on real-vs-stub injection.
5. **LOG phase:** append `harness_log.md` AFTER Q/A. Flip status.

## Anti-patterns guarded (≥4)

1. **Silent stub-fallback on production-fn failure** — researcher
   ruled out: fall back to empty result + log WARNING; heartbeat
   shows `written=0` honestly so the operator sees the problem.
2. **Eager import of yfinance / fredapi at module top** — would
   break the existing `test_no_live_yfinance_call` constraint.
   Lazy imports inside closure body.
3. **Using `asyncio.run()` for the sync→async bridge** — raises
   `RuntimeError` when loop is already running. Use
   `run_coroutine_threadsafe`.
4. **Changing `run()` signatures** — would break tests. We only
   add `prod_fns` injection at the registration layer.
5. **Self-evaluation by Main** — Q/A is mandatory.

## Out of scope

- `hourly_signal_warmup` — `compute_signal_fn` is a per-ticker
  signal calculator; researcher confirmed deferred.
- `nightly_mda_retrain` — already runs real `train_fn` (researcher
  in 23.5.9 confirmed). PromotionGate threshold tuning is separate.
- New BQ tables.
- yfinance retry / backoff logic (the closure logs WARNING and
  returns empty; backoff is a future enhancement).
- Modifying `app.client.chat_postMessage` call shape.

## Backwards compatibility

- `register_phase9_jobs(scheduler, replace_existing=True)` →
  `register_phase9_jobs(scheduler, replace_existing=True, app=None,
  loop=None)`. New params have defaults; existing callers (if any)
  unchanged. When `app/loop` are None, no partial-application
  happens — the bare `run` is registered, identical to current
  behavior.
- All existing job tests inject mocks via `run(fetch_fn=mock,
  write_fn=mock)`; the production wiring is at registration time,
  not run() time. No mock interference.
- Lazy imports preserve `test_no_live_yfinance_call`.
- Existing failure modes preserved: heartbeat path on every fire,
  fail-open on subprocess errors.

## Risk

- **yfinance API revocation** (April 2025 reports per researcher's
  brief). Mitigation: production closure has try/except + logs
  WARNING + returns empty. Operator sees `written=0` in the
  dashboard and can take action.
- **fredapi missing API key** (`FRED_API_KEY` env var). Mitigation:
  same — try/except + log + empty.
- **Slack post fails** (rate-limit / network). Mitigation:
  `alert_fn` wraps in try/except. Budget trip is still in the
  registry; the post-failure is logged.
- **`asyncio.get_running_loop()` raises** if `start_scheduler` is
  ever called from sync context. Mitigation: assert pattern at
  the top of `start_scheduler` to catch this early; fail loudly.

## References

- Research brief: `handoff/current/phase-23.6.1-research-brief.md`.
- Files to edit:
  - `backend/slack_bot/scheduler.py` (extend
    `register_phase9_jobs` + `start_scheduler`).
- New files:
  - `backend/slack_bot/jobs/_production_fns.py` (factory module).
  - `tests/slack_bot/test_phase9_production_wiring.py`.
  - `tests/verify_phase_23_6_1.py`.
- Phase-23.5.13 archive (final stub-tally documentation):
  `handoff/archive/phase-23.5.13/`.
- Python functools.partial:
  https://docs.python.org/3/library/functools.html#functools.partial
- Cosmic Python ch. 13: https://www.cosmicpython.com/book/chapter_13_dependency_injection.html
- BetterStack DI guide:
  https://betterstack.com/community/guides/scaling-python/dependency-injection-python/
- Slack Bolt async:
  https://slack.dev/bolt-python/api-docs/slack_bolt/
