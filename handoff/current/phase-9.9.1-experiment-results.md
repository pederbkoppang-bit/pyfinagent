# Experiment Results — phase-9.9.1 (cost_budget_watcher + weekly_data_integrity wiring fix)

**Step:** 9.9.1 **Date:** 2026-04-20

## What was done

1. Fresh researcher: 5 sources in full + 10 snippet-only, 15 URLs; brief at `handoff/current/phase-9.9.1-research-brief.md`; gate passed; Option B (callable injection) recommended.
2. Contract authored at `handoff/current/phase-9.9.1-contract.md`.
3. Amended `backend/slack_bot/jobs/cost_budget_watcher.py`:
   - `daily_spend_usd` / `monthly_spend_usd` now `float | None = None`
   - Added `fetch_fn: Callable[[], tuple[float, float]] | None = None` param
   - Added `_default_fetch_spend()` — Anthropic Cost API via `httpx`, fail-open on missing `ANTHROPIC_ADMIN_API_KEY` or any Exception (returns `(0.0, 0.0)`)
   - Inside `run()`: resolve spend values via `fetch_fn or _default_fetch_spend` when either is None
   - File: 113 lines (was 64)
4. Amended `backend/slack_bot/jobs/weekly_data_integrity.py`:
   - Added `fetch_fn: Callable[[], dict[str, int]] | None = None` + `snapshot_path: str | None = None`
   - Added `_default_fetch_counts()` — BQ `__TABLES__` query, fail-open to `{}`
   - Added `_load_snapshot()` / `_save_snapshot()` — JSON file at `handoff/logs/row_count_snapshot.json` (default), parent dir auto-created
   - Snapshot save inside `heartbeat` block after drift compute
   - File: 114 lines (was 57)
5. Wrote `tests/slack_bot/test_scheduler_wiring_phase991.py` — 9 regression tests covering:
   - cost_budget_watcher zero-args fetch_fn wiring
   - fetch_fn over-cap trips
   - no-admin-key fail-open path
   - explicit values beat fetch_fn
   - weekly_data_integrity zero-args fetch + snapshot roundtrip
   - first-run missing-snapshot baseline write
   - BQ fetch fail-open
   - **End-to-end: `cost_budget_watcher.run()` bare call succeeds** (regression guard for original TypeError)
   - **End-to-end: `weekly_data_integrity.run()` bare call succeeds**
6. Scheduler.py unchanged (Option B keeps wiring idiomatic; no args=/kwargs= needed).

## Verification (verbatim)

```
$ python -c "import ast; ast.parse(open('backend/slack_bot/jobs/cost_budget_watcher.py').read()); ast.parse(open('backend/slack_bot/jobs/weekly_data_integrity.py').read()); print('AST OK')"
AST OK

$ pytest tests/slack_bot/test_cost_budget_watcher.py tests/slack_bot/test_weekly_data_integrity.py tests/slack_bot/test_scheduler_phase9.py tests/slack_bot/test_scheduler_wiring_phase991.py -q
....................                                                     [100%]
20 passed in 1.07s

$ pytest tests/slack_bot/ -q
............................................                             [100%]
44 passed in 0.83s
(exit 0)
```

## Success criteria

| # | Criterion | Status |
|---|---|---|
| 1 | ast.parse cost_budget_watcher.py | PASS |
| 2 | ast.parse weekly_data_integrity.py | PASS |
| 3 | test_cost_budget_watcher.py 4/4 (no regression) | PASS |
| 4 | test_weekly_data_integrity.py 3/3 (no regression) | PASS |
| 5 | test_scheduler_phase9.py 4/4 (no regression) | PASS |
| 6 | test_scheduler_wiring_phase991.py ≥3 tests | PASS (9 tests) |
| 7 | Zero-arg `cost_budget_watcher.run()` executes without TypeError | PASS (test `test_scheduler_wiring_cost_budget_watcher_fires_zero_args`) |
| | Full slack_bot suite | PASS (44/44) |

## Before vs after

**Before:** `cost_budget_watcher.run()` required `daily_spend_usd` + `monthly_spend_usd` with no defaults → `TypeError` on every APScheduler fire. `weekly_data_integrity.run()` defaulted to `{}` dicts → `_compute_drifts({}, {})` → always `[]`.

**After:** Both jobs fire successfully with zero args. `cost_budget_watcher` calls Anthropic Cost API (or fails open to 0.0/0.0 with warning log). `weekly_data_integrity` queries BQ `__TABLES__` (or fails open to `{}`) + loads/saves snapshot JSON for prior-week comparison.

## Carry-forwards (out of scope)

- Provision `ANTHROPIC_ADMIN_API_KEY` in production `.env` — until then, cost watcher reports 0.0/0.0 (safe fail-open)
- Confirm service-account IAM has `roles/bigquery.resourceViewer` for `__TABLES__` access
- Update `docs/runbooks/phase9-cron-runbook.md` §5 with silent-no-op rows (addressed in next phase-9.10 revision)
