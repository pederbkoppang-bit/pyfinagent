---
step: phase-10.7.6
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - backend/meta_evolution/cron.py (NEW, ~155 LOC)
  - tests/scheduler/__init__.py (NEW, empty marker)
  - tests/scheduler/test_meta_cron.py (NEW, ~210 LOC, 11 tests)
---

# Experiment Results -- phase-10.7.6

## What was done

Wired the weekly meta-evolution loop onto APScheduler-shaped objects via
a new pure module `backend/meta_evolution/cron.py`. Defaults to Sunday
02:00 America/New_York. Pattern mirrors `backend/autoresearch/cron.py`
(separate cron module shim) and `backend/slack_bot/scheduler.py:351-382`
(register signature + per-sub-call fail-open).

## Deliverables

### `backend/meta_evolution/cron.py` (NEW, ~155 LOC)

Module exports:
- `JOB_ID = "meta_evolution_weekly"` (canonical id; matches slot 14 in `.claude/cron_budget.yaml`)
- `TIMEZONE = ZoneInfo("America/New_York")`
- `register_meta_evolution_cron(scheduler, *, replace_existing=True, day_of_week="sun", hour=2, minute=0) -> str | None`
  -- adds the job; returns `JOB_ID` on success, `None` on add_job exception (fail-open)
- `run_meta_evolution_cycle(*, cron_budget_yaml=None, provider_budget_yaml=None, bq_client=None, now=None) -> dict[str, Any]`
  -- pure orchestration; each sub-call wrapped in its own try/except with
  warning-log fail-open per Google SRE monitoring-tier discipline

Sub-calls invoked: `cron_allocator.allocate(yaml)` (10.7.4),
`provider_rebalancer.allocate(yaml)` (10.7.5), `len(archetype_library.ARCHETYPES)`
(10.7.3). `bq_client` parameter accepted for forward compatibility
(unused this cycle; no BQ writes added).

ASCII-only logger messages (per `.claude/rules/security.md`). Pure stdlib
imports: `logging`, `datetime`, `pathlib`, `typing`, `zoneinfo`. No new
dependencies added; APScheduler `CronTrigger` not imported here -- the
scheduler-passed `add_job(..., trigger="cron", ...)` form keeps this
module agnostic to APScheduler's class hierarchy.

### `tests/scheduler/__init__.py` (NEW, empty)

New directory marker so `python -m pytest tests/scheduler/...` works.

### `tests/scheduler/test_meta_cron.py` (NEW, ~210 LOC, 11 tests)

8 tests planned in contract + 3 additional defensive:

1. `test_register_adds_job_to_scheduler`
2. `test_job_id_is_meta_evolution_weekly`
3. `test_register_passes_replace_existing_true`
4. `test_register_uses_cron_trigger_with_sunday_2am_kwargs` (defensive: trigger kwargs assertion)
5. `test_timezone_is_explicitly_new_york`
6. `test_register_fail_open_returns_none` (defensive: add_job raises -> None, no propagation)
7. `test_trigger_fires_sunday_2am_et` (instantiate `CronTrigger` directly + `get_next_fire_time(None, monday_2026_01_05)` -> `2026-01-11 02:00 ET`)
8. `test_run_cycle_calls_cron_allocator` (monkeypatch spy)
9. `test_run_cycle_calls_provider_rebalancer` (monkeypatch spy)
10. `test_run_cycle_handles_sub_failures_fail_open` (both sub-calls raise; cycle still returns; errors recorded)
11. `test_run_cycle_returns_well_formed_dict` (defensive: top-level keys + duration_seconds >= 0 + empty errors list when sub-calls succeed)

Uses StubScheduler pattern from `tests/slack_bot/test_scheduler_phase9.py:14-21`.

## Verification (verbatim, immutable from masterplan)

```
$ python -m pytest tests/scheduler/test_meta_cron.py -v
============================== 11 passed in 0.03s ==============================
```

## Files touched

| Path | Action | Note |
|------|--------|------|
| `backend/meta_evolution/cron.py` | CREATED | ~155 LOC pure module |
| `tests/scheduler/__init__.py` | CREATED | empty marker |
| `tests/scheduler/test_meta_cron.py` | CREATED | ~210 LOC, 11 tests |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-10.7.6-research-brief.md` | created (researcher) | -- |

NO live wiring into `backend/slack_bot/scheduler.py` `start_scheduler()`
this cycle (out-of-scope per contract). Module is import-safe and
ready to be wired in a follow-up one-liner.

NO new dependencies (APScheduler is already pinned `>=3.10.0`).

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | `tests/scheduler/test_meta_cron.py` exists | PASS |
| 2 | All tests pass via `python -m pytest tests/scheduler/test_meta_cron.py -v` | PASS (11/11) |
| 3 | `register_meta_evolution_cron(scheduler, *, replace_existing=True)` signature matches contract | PASS |
| 4 | Default schedule = Sunday 02:00 America/New_York | PASS (kwargs day_of_week="sun", hour=2, minute=0; timezone=ZoneInfo("America/New_York")) |
| 5 | `replace_existing=True` propagated to scheduler.add_job | PASS |
| 6 | Fail-open per sub-call (one raises, others continue) | PASS (test_run_cycle_handles_sub_failures_fail_open) |
| 7 | ASCII-only logger messages | PASS (no Unicode in logger.* calls) |
| 8 | Pattern parity with autoresearch/cron.py + slack_bot/scheduler.py | PASS |

## Honest disclosures

1. **11 tests vs 8 in contract** -- added 3 defensive: trigger kwargs
   assertion, register fail-open path, well-formed dict shape. Floor
   exceeded; not a violation.

2. **No cycle-2 fix needed.** First pytest run was 11/11 PASS -- contract
   was specific enough that the implementation matched on first try.

3. **`bq_client` parameter unused this cycle.** Kept in the
   `run_meta_evolution_cycle` signature for forward compatibility with
   10.7.7 / 10.7.8 follow-ups (alpha_velocity persistence, evaluator
   review gate). Not a leak -- documented in module docstring.

4. **No live wiring.** The new module is not yet called from
   `backend/slack_bot/scheduler.py` `start_scheduler()`. That is a
   one-line follow-up explicitly out of scope per contract section
   "Out of scope". Module is import-safe and ready.

5. **Pure orchestration.** `run_meta_evolution_cycle` does not itself
   call any LLM, BQ, or HTTP -- it dispatches to already-existing pure
   modules (cron_allocator, provider_rebalancer, archetype_library)
   and aggregates their return shapes. Cost-safe to fire weekly even
   without budget.

6. **No BQ table needed.** Cycle returns telemetry as a dict; no schema
   migration this cycle. Future telemetry sink is a 10.7.x follow-up.

## Closes

Task list item #75. Masterplan step phase-10.7.6.

## Next

Spawn Q/A.
