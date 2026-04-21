# Sprint Contract — phase-9.9.2 (cost watcher source swap: Anthropic Cost API → BQ INFORMATION_SCHEMA)

**Step id:** 9.9.2 **Date:** 2026-04-20 **Tier:** simple
**Predecessor:** phase-9.9.1 (Anthropic Cost API wiring — now deprecated for this org)

## Why

User runs pyfinagent on a **Claude Max subscription** (flat-fee, no per-token billing). The Anthropic `/v1/organizations/cost_report` endpoint is a billable-API-customer-only feature; under Max it either 401s or returns zero. The only remaining variable cost under Max is **BigQuery query bytes**. Swap the spend signal.

## Research-gate summary

Fresh researcher: `handoff/current/phase-9.9.2-research-brief.md` — 7 sources in full, 14 URLs, three-variant queries, recency scan, gate_passed=true.

**Corrections the brief forced on the initial design sketch:**
1. Price is **$6.25/TiB** (not $5/TB) — stable since 2023-07-05
2. Use `total_bytes_billed` (not `total_bytes_processed`) — reflects the 10 MB minimum-billing floor
3. `region-us` qualifier is required on the `JOBS_BY_PROJECT` view (no global view)
4. `roles/bigquery.resourceViewer` grants the needed `bigquery.jobs.listAll` — already planned for phase-9.9.1 go-live
5. Use `google.cloud.bigquery.Client` directly, NOT the `BigQueryClient` wrapper (which requires a Settings object)

## Immutable success criteria

1. `python -c "import ast; ast.parse(open('backend/slack_bot/jobs/cost_budget_watcher.py').read())"` exit 0
2. **No reference to `ANTHROPIC_ADMIN_API_KEY` remains** in `cost_budget_watcher.py` — verified by `grep -c "ANTHROPIC_ADMIN" backend/slack_bot/jobs/cost_budget_watcher.py | python -c "import sys; sys.exit(0 if int(sys.stdin.read()) == 0 else 1)"`
3. `pytest tests/slack_bot/test_cost_budget_watcher.py -q` — existing 4 tests still pass
4. `pytest tests/slack_bot/test_scheduler_wiring_phase991.py -q` — regression suite updated + still pass
5. `pytest tests/slack_bot/ -q` — full suite 44/44 pass (no regression; test renames net-zero count)
6. Bare `cost_budget_watcher.run()` call returns `{daily: 0.0, tripped: False}` in CI (BQ unreachable path)
7. `_default_fetch_spend()` source code contains literal `"INFORMATION_SCHEMA.JOBS_BY_PROJECT"` and `6.25` (the correct per-TiB rate) — guards against regression to $5/TB or wrong view

## Plan

1. Edit `backend/slack_bot/jobs/cost_budget_watcher.py`:
   - Remove `ANTHROPIC_ADMIN_API_KEY` env var handling + the `httpx` Anthropic call
   - Replace `_default_fetch_spend()` body with the researcher's BQ query pattern (using `google.cloud.bigquery.Client` directly)
   - Price constant: `6.25` USD per TiB (`bytes / 1e12 * 6.25`)
   - `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT` with `creation_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH)` partition filter
   - Fail-open to `(0.0, 0.0)` on any exception (unchanged semantic)
   - Keep module-level docstring updated to mention Max-subscription context
2. Edit `tests/slack_bot/test_scheduler_wiring_phase991.py`:
   - Replace `test_cost_budget_watcher_no_admin_key_fail_open` with `test_cost_budget_watcher_bq_unreachable_fail_open` that monkeypatches `google.cloud.bigquery.Client` to raise and asserts `(0.0, 0.0)` is returned
   - Other tests remain as-is (end-to-end bare-call test still validates fail-open path)
3. Re-verify all 7 criteria.
4. Spawn fresh Q/A.
5. Log; mark task #93 complete.

## References

- `handoff/current/phase-9.9.2-research-brief.md` (7 in full, 14 URLs, gate_passed=true)
- `backend/slack_bot/jobs/cost_budget_watcher.py` (edit target)
- `tests/slack_bot/test_scheduler_wiring_phase991.py` (1 test replaced, others kept)
- `backend/slack_bot/jobs/weekly_data_integrity.py` (reference pattern: `google.cloud.bigquery.Client` + fail-open — already follows this idiom at `_default_fetch_counts`)
- BQ pricing doc: `$6.25/TiB` on-demand (stable 2023-07-05 → 2026)

## Carry-forwards (out of scope)

- **Dashboards/alerts:** once BQ spend numbers are flowing, consider adding a weekly summary line in the morning digest (phase-10.x)
- **Per-dataset attribution:** the current query aggregates project-wide; to split by dataset (e.g., `pyfinagent_data` vs `pyfinagent_hdw`), add `job.referenced_tables` join
- **Cost caps tuning:** current defaults `daily=$5, monthly=$50` were chosen for LLM spend; for BQ-only, Peder may want lower caps since most pyfinagent BQ queries are partitioned and should be <$1/day
