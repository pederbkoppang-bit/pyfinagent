# Sprint Contract — phase-9.9.1 (cost_budget_watcher + weekly_data_integrity wiring fix)

**Step id:** 9.9.1 **Date:** 2026-04-20 **Tier:** moderate
**Predecessor:** phase-9.9 (qa_99_remediation_v1 surfaced the bugs)

## Why

Q/A qa_99_remediation_v1 independent-reproduced two runtime bugs:
1. **CRITICAL:** `cost_budget_watcher.run()` raises `TypeError` on every APScheduler fire (kw-only args unsatisfied; fail-open wrapper swallows; heartbeat "ok"; watcher has never actually checked a budget).
2. **MEDIUM:** `weekly_data_integrity.run()` runs with empty dicts → zero drifts ever detected.

Blocks phase-9 go-live.

## Research-gate summary

Fresh researcher (moderate tier): `handoff/current/phase-9.9.1-research-brief.md` — 5 sources in full + 10 snippet-only, 15 URLs, three-variant queries, recency, internal inventory of 10 files, gate_passed=true.

**Decision: Option B (callable injection, `fetch_fn or _default_fetch_*`).** Matches the codebase's established idiom at `daily_price_refresh.py:36` and `nightly_mda_retrain.py:37`. Zero changes to `scheduler.py`. Keeps job module self-testable with `fetch_fn=stub`.

## Hypothesis

Changing `cost_budget_watcher.run()` signature to make `daily_spend_usd`/`monthly_spend_usd` optional (`| None = None`) and adding `fetch_fn: Callable[[], tuple[float, float]] | None = None` + `_default_fetch_spend()` (Anthropic Cost API, fail-open) will make zero-arg APScheduler invocation succeed. Same pattern for `weekly_data_integrity.run()` with `fetch_fn` (BQ `__TABLES__`) + `snapshot_path` (JSON file persistence for prior-week counts).

## Immutable success criteria

1. `python -c "import ast; ast.parse(open('backend/slack_bot/jobs/cost_budget_watcher.py').read())"` exit 0
2. `python -c "import ast; ast.parse(open('backend/slack_bot/jobs/weekly_data_integrity.py').read())"` exit 0
3. `pytest tests/slack_bot/test_cost_budget_watcher.py -q` exit 0 (existing 4 tests still pass)
4. `pytest tests/slack_bot/test_weekly_data_integrity.py -q` exit 0 (existing 3 tests still pass)
5. `pytest tests/slack_bot/test_scheduler_phase9.py -q` exit 0 (existing 4 tests still pass)
6. **NEW regression test** `tests/slack_bot/test_scheduler_wiring_phase991.py` — at least 3 tests: (a) `cost_budget_watcher.run()` called with zero spend args invokes `fetch_fn` stub and returns a non-None `daily`; (b) `weekly_data_integrity.run()` with zero count args invokes `fetch_fn` stub + `snapshot_path`; (c) Anthropic Cost API fail-open path (no `ANTHROPIC_ADMIN_API_KEY`) returns `(0.0, 0.0)` without raising.
7. **Integration test proving the scheduler wiring**: `StubScheduler.fire_job("cost_budget_watcher")` (new helper) executes without TypeError.

## Plan

1. Amend `backend/slack_bot/jobs/cost_budget_watcher.py`:
   - Make `daily_spend_usd` / `monthly_spend_usd` optional with `None` default
   - Add `fetch_fn: Callable[[], tuple[float, float]] | None = None` param
   - Add `_default_fetch_spend()` — Anthropic Cost API with httpx, fail-open on missing `ANTHROPIC_ADMIN_API_KEY` or any exception
   - Inside `run()`: resolve spend values via `fetch_fn or _default_fetch_spend` if either is None
2. Amend `backend/slack_bot/jobs/weekly_data_integrity.py`:
   - Add `fetch_fn: Callable[[], dict[str, int]] | None = None` param
   - Add `snapshot_path: str | None = None` param
   - Add `_default_fetch_counts()` — BQ `__TABLES__` via existing client, fail-open returning `{}`
   - Add `_load_snapshot()` / `_save_snapshot()` — JSON file at `handoff/logs/row_count_snapshot.json` (create parent dir if missing)
   - Inside `run()`: resolve counts + load snapshot if not injected; save snapshot after drift computation
3. Write `tests/slack_bot/test_scheduler_wiring_phase991.py` with the 3+ tests from criterion #6.
4. Enhance `StubScheduler` in `tests/slack_bot/test_scheduler_phase9.py` with a `fire_job(job_id)` helper (or add the test in the new file using the existing StubScheduler pattern).
5. Re-run all 5 existing test files to confirm no regression.
6. Spawn fresh Q/A.
7. Log and mark task complete.

## References

- `handoff/current/phase-9.9.1-research-brief.md` (5 in full, 15 URLs, gate_passed=true)
- `backend/slack_bot/jobs/cost_budget_watcher.py` (CRITICAL fix target)
- `backend/slack_bot/jobs/weekly_data_integrity.py` (MEDIUM fix target)
- `backend/slack_bot/jobs/daily_price_refresh.py:36` (reference pattern)
- `backend/slack_bot/jobs/nightly_mda_retrain.py:37` (reference pattern)
- `backend/slack_bot/scheduler.py:374` (no change; confirms `**kwargs` is trigger-config)
- Anthropic Cost API: `POST /v1/organizations/cost_report` (researcher-validated)
- BQ `pyfinagent_data.__TABLES__` (researcher-validated cheapest row-count source)

## Carry-forwards (NOT in 9.9.1 scope)

- `ANTHROPIC_ADMIN_API_KEY` env var must be provisioned in production `.env` before the watcher reports real numbers (expect 0.0/0.0 with warning log until then — safe fail-open)
- BQ `roles/bigquery.resourceViewer` for `__TABLES__` access — confirm service account IAM before go-live
- Runbook update (phase-9.10) to add silent-no-op failure rows — follows this fix
