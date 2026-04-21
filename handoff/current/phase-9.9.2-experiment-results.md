# Experiment Results — phase-9.9.2 (cost watcher source swap: BQ spend)

**Step:** 9.9.2 **Date:** 2026-04-20

## What was done

1. Fresh researcher: 7 sources in full, 14 URLs, gate_passed=true. Brief at `handoff/current/phase-9.9.2-research-brief.md`. Corrected pricing to $6.25/TiB, specified `region-us` qualifier, confirmed `resourceViewer` IAM sufficient, flagged to use `google.cloud.bigquery.Client` directly (not the wrapper).
2. Contract authored at `handoff/current/phase-9.9.2-contract.md`.
3. Edited `backend/slack_bot/jobs/cost_budget_watcher.py`:
   - Dropped `datetime` import and all `ANTHROPIC_ADMIN_API_KEY` logic
   - Added `_BQ_USD_PER_TIB = 6.25` module constant
   - Rewrote `_default_fetch_spend()` body: uses `google.cloud.bigquery.Client(project=...)`, runs a month-partition-filtered query against `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`, multiplies `total_bytes_billed / 1e12 * 6.25`
   - Updated module docstring to mention Max-subscription context
   - File is now 112 lines (was 113; net −1 because the Anthropic HTTP call was longer)
4. Edited `tests/slack_bot/test_scheduler_wiring_phase991.py`:
   - Renamed `test_cost_budget_watcher_no_admin_key_fail_open` → `test_cost_budget_watcher_bq_unreachable_fail_open`; now monkeypatches `google.cloud.bigquery.Client` to raise
   - Fixed `test_scheduler_wiring_cost_budget_watcher_fires_zero_args` to monkeypatch `_default_fetch_spend` (hermetic — was hitting real BQ on dev machine and failing the old `== 0.0` assumption)

## Verification (verbatim)

### Deterministic checks

```
$ python -c "import ast; ast.parse(open('backend/slack_bot/jobs/cost_budget_watcher.py').read()); print('AST OK')"
AST OK

$ grep -c "ANTHROPIC_ADMIN" backend/slack_bot/jobs/cost_budget_watcher.py
0

$ grep -E "INFORMATION_SCHEMA\.JOBS_BY_PROJECT|6\.25" backend/slack_bot/jobs/cost_budget_watcher.py
`region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT * $6.25/TiB`.
_BQ_USD_PER_TIB = 6.25  # on-demand pricing, stable 2023-07-05 -> 2026
    """Fetch today + this-month BQ spend from INFORMATION_SCHEMA.JOBS_BY_PROJECT.
    Price: $6.25/TiB on-demand (stable 2023-07-05 -> 2026). Uses
            FROM `{project}.region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`

$ pytest tests/slack_bot/test_cost_budget_watcher.py tests/slack_bot/test_scheduler_wiring_phase991.py -q
.............                                                            [100%]
13 passed in 0.76s

$ pytest tests/slack_bot/ -q
............................................                             [100%]
44 passed in 0.84s
```

### Live BQ smoke test (dev machine — user ADC credentials)

```
$ python -c "
from backend.slack_bot.jobs.cost_budget_watcher import _default_fetch_spend, run
daily, monthly = _default_fetch_spend()
print(f'BQ spend (live): daily=\${daily:.4f}  monthly=\${monthly:.4f}')
result = run()
print(f'run() result: tripped={result[\"tripped\"]}  daily=\${result[\"daily\"]:.4f}  monthly=\${result[\"monthly\"]:.4f}')
"
BQ spend (live): daily=$0.0003  monthly=$0.3820
run() result: tripped=False  daily=$0.0004  monthly=$0.3821
```

**Interpretation:** the project's YTD BQ cost through April 2026 is ~$0.38, well under the $50 monthly cap and the $5 daily cap. Today's marginal cost is ~$0.0003. The numbers line up with the expected lightweight footprint of pyfinagent's partitioned queries.

## Success criteria

| # | Criterion | Status |
|---|---|---|
| 1 | ast.parse exit 0 | PASS |
| 2 | `grep -c ANTHROPIC_ADMIN` == 0 | PASS |
| 3 | test_cost_budget_watcher.py 4/4 | PASS |
| 4 | test_scheduler_wiring_phase991.py 9/9 | PASS |
| 5 | Full slack_bot/ 44/44 | PASS |
| 6 | Bare `run()` returns `{tripped: False}` in CI | PASS |
| 7 | Code contains `INFORMATION_SCHEMA.JOBS_BY_PROJECT` + `6.25` | PASS |

## Before vs after

**Before (phase-9.9.1):** `_default_fetch_spend` hit `/v1/organizations/cost_report` with `ANTHROPIC_ADMIN_API_KEY`. Under a Claude Max subscription the endpoint is 401/empty — the watcher would have permanently reported 0.0/0.0 even with a valid admin key provisioned.

**After (phase-9.9.2):** `_default_fetch_spend` queries `region-us.INFORMATION_SCHEMA.JOBS_BY_PROJECT`, sums `total_bytes_billed` for (a) today and (b) current month, multiplies by $6.25/TiB. Live dev-machine call returns real numbers (~$0.38 month-to-date). Fail-open semantics unchanged.

## Carry-forwards (out of scope)

- **Cap tuning:** current `$5/day, $50/month` was chosen for LLM spend. Under BQ-only on Max, typical daily cost is <$0.01; Peder may want to lower caps (e.g., `$1/day, $10/month`) to make trips meaningful.
- **Dashboards:** morning digest line summarizing yesterday's BQ spend — phase-10.x
- **Per-dataset attribution:** join `job.referenced_tables` to split spend by dataset — phase-10.x
