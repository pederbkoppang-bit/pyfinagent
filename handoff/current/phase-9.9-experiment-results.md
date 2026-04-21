# Experiment Results — phase-9 / 9.9 (scheduler wiring) — REMEDIATION v1

**Step:** 9.9 **Remediation cycle:** 1 **Date:** 2026-04-20

## What was done

1. Fresh researcher (moderate tier, 10 internal files inspected): 8 sources in full; `handoff/current/phase-9.9-research-brief.md`; gate passed.
2. Contract authored disclosing 2 runtime bugs (CRITICAL: cost_budget_watcher TypeError; MEDIUM: weekly_data_integrity empty-dict inert).
3. Re-verified immutable criterion.
4. No code changes (remediation scope is harness-compliance; runtime bugs are carry-forwards to hardening).

## Verification (verbatim)

```
$ python -c "import ast; ast.parse(open('backend/slack_bot/scheduler.py').read())" && pytest tests/slack_bot/test_scheduler_phase9.py -q
....                                                                     [100%]
4 passed in 0.14s
(exit 0)
```

Tests: test_all_seven_jobs_registered, test_no_double_registration_on_reload, test_phase9_ids_stable, test_runbook_exists.

## Artifact shape

- `scheduler.py:347` `register_phase9_jobs()` — 31-line function with 7-entry mapping, fail-open try/except per job, `replace_existing=True`.
- `_PHASE9_JOB_IDS` tuple of 7 ids.
- Cron-only triggers (no interval/date triggers); single-process scheduler in Slack bot.

## Runtime bugs disclosed (NOT fixed this cycle)

| # | Severity | Bug | Fix location |
|---|---|---|---|
| 1 | **CRITICAL** | `cost_budget_watcher.run()` raises TypeError on every fire (APScheduler passes no args; required kw-only params `daily_spend_usd`/`monthly_spend_usd` unsatisfied) | Hardening phase (add defaults + side-channel fetch, OR pass args in register) |
| 2 | MEDIUM | `weekly_data_integrity.run()` inert (empty dicts → zero drifts) | Hardening phase (wire BQ INFORMATION_SCHEMA query + prior-week persistence) |
| 3 | LOW | No explicit UTC timezone | `scheduler.configure(timezone="UTC")` |
| 4 | LOW | Single-process scheduler (no multi-worker lock) | Redis lock or dedicated scheduler service for scale |

## Success criteria

| # | Criterion | Status |
|---|---|---|
| 1 | ast.parse OK | PASS |
| 2 | pytest 4/4 | PASS |

**Note:** Immutable criterion passes because tests verify registration only, not invocation. The CRITICAL carry-forward #1 MUST be closed before production go-live.
