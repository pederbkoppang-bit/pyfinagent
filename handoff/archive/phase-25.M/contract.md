---
step: 25.M
slug: cost-budget-alert-wire-repair
status: in_progress
cycle_date: 2026-05-13
parent_research_brief: handoff/current/research_brief.md
---

# Contract -- phase-25.M

## Step ID + masterplan reference

`25.M` -- "Cost-budget Slack alert wire repair (no silent fail-open)"
(P2, harness_required, depends on `25.A9` done).

## Research-gate summary

Tier=simple. Main authored brief from internal inspection of
`_production_fns.py` + `scheduler.py`. JSON envelope shows
`gate_passed=true`. The fix is mechanical: fail-fast at factory time +
ERROR-level logging at execution time.

## Hypothesis

The cost_budget_watcher job has TWO silent-failure paths today:
1. `make_alert_fn_for_budget(app=None, loop=None, channel="")` returns
   a non-throwing closure that just posts to `""` Slack channel and the
   API 400s into a `logger.warning` -- invisible to operators.
2. `register_phase9_jobs` wraps the production-fn wiring in a try/except
   that logs at WARNING and falls back to bare run with NO alert plumbing
   -- so the watcher runs, finds a breach, calls a no-op alert_fn, and
   nobody sees it.

The fix: raise `ValueError` in the factory if any of `app`/`loop`/`channel`
are missing; promote the WARNING in `register_phase9_jobs` to ERROR with
traceback; promote the runtime fail-open in `_post_slack_sync` to ERROR.

## Success criteria (verbatim from masterplan.json)

> `make_alert_fn_for_budget_raises_loudly_on_wiring_error`
>
> `scheduler_register_phase9_jobs_logs_error_visibly`

## Plan steps

1. **`backend/slack_bot/jobs/_production_fns.py::make_alert_fn_for_budget`**:
   add input validation at the top of the factory -- raise `ValueError`
   if `app is None or loop is None or not channel`.
2. **`backend/slack_bot/jobs/_production_fns.py::_post_slack_sync`**:
   promote the except clause from `logger.warning` to `logger.error(..., exc_info=True)`
   so tracebacks land in logs. Keep the try/except so APScheduler doesn't
   crash.
3. **`backend/slack_bot/scheduler.py::register_phase9_jobs`**:
   promote the outer production-fn-wiring `logger.warning` at line 670
   to `logger.error(..., exc_info=True)` so a factory failure surfaces
   visibly rather than being swallowed.
4. **Verifier** -- `tests/verify_phase_25_M.py` with 4 claims:
   - Claim 1: `make_alert_fn_for_budget` raises `ValueError` when called
     with `channel=""`.
   - Claim 2: `make_alert_fn_for_budget` raises `ValueError` when called
     with `loop=None`.
   - Claim 3: `register_phase9_jobs` uses `logger.error` (not `.warning`)
     on the production-fn wiring exception.
   - Claim 4: `_post_slack_sync` uses `logger.error` (not `.warning`) when
     the chat_postMessage call fails.

## Files

| File | Action |
|------|--------|
| `backend/slack_bot/jobs/_production_fns.py` | Edit `make_alert_fn_for_budget` + `_post_slack_sync` |
| `backend/slack_bot/scheduler.py` | Promote WARNING -> ERROR at line 670 |
| `tests/verify_phase_25_M.py` | NEW verifier |

## Verification command (immutable)

```
source .venv/bin/activate && python3 tests/verify_phase_25_M.py
```

## Live-check

`Inject wiring failure; confirm error logged at WARNING+ not fail-open silent`.
Will write `handoff/current/live_check_25.M.md` summarising injection
output + observed log level.

## Risks + mitigations

- **Risk**: Factory now raises in callers that pass `channel=""` legitimately
  (e.g., tests that just want a no-op alert_fn).
  **Mitigation**: Audit call sites -- production caller in `scheduler.py:663`
  always passes a real `app/loop/channel`; the test path passes `StubScheduler`
  and never reaches the factory (only bare `run` registered).
- **Risk**: Promoting WARNING -> ERROR floods logs.
  **Mitigation**: Only the failure path changes level. Healthy ops stay silent.

## References

- `handoff/current/research_brief.md` (this cycle)
- `backend/slack_bot/jobs/_production_fns.py:282-301, 260-279`
- `backend/slack_bot/scheduler.py:644-671`
- `.claude/masterplan.json::25.M`
