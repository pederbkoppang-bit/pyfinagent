# BLOCKER-3 HITL C/C gate end-to-end -- Experiment results

## What was built

Added the missing BQ audit emission to the HITL Champion/Challenger
approval path, then built a hermetic end-to-end drill that exercises
gate-fire -> pending-state -> Slack-ping -> approval -> BQ-log in one
run, without touching real production state, real Slack, or real BQ.

## Root cause

Research-gate + Main audit confirmed:
- `backend/autoresearch/monthly_champion_challenger.py::record_approval`
  wrote ONLY to `handoff/logs/monthly_approval_state.json`. No BQ
  write anywhere in the gate pipeline, so the task-#42 contract
  requirement "state visible in monthly_approval_state.json + BQ log
  row" was unsatisfiable with the shipped code.
- No end-to-end drill existed. Unit tests exercised each step in
  isolation, but the full pipeline (gate fire -> pending -> approval
  -> BQ) had never been run against a single synthetic promotion.

## Fix

### 1. `backend/autoresearch/monthly_champion_challenger.py`

Added optional `bq_fn: Callable[[dict], None] | None = None` kwarg to
`record_approval`. On every terminal state transition (approved,
rejected, or lazily-expired from-pending), a new helper
`_emit_deployment_log_row` builds a `strategy_deployments_log`-shaped
dict and calls `bq_fn(log_row)` behind a fail-open try/except. Zero
behavior change when `bq_fn` is `None` (backward-compatible with every
existing caller).

Dict shape matches BQ schema: strategy_id, status, sharpe, dsr, pbo,
max_dd, deployed_at, allocation_pct, notes.

### 2. `backend/api/monthly_approval_api.py`

Added `_default_bq_logger(log_row)` -- fail-open production writer
using `google.cloud.bigquery` to INSERT into
`<project>.pyfinagent_pms.strategy_deployments_log`. Wired into
`post_monthly_approval` via `bq_fn=_default_bq_logger`. Imports
Settings for project id; local-import of bigquery to avoid eager GCP
auth cost on cold start.

### 3. `scripts/go_live_drills/hitl_gate_drill.py` (new)

Hermetic drill that:
- Picks a past last-trading Friday (2026-03-27) so
  `is_last_trading_friday` evaluates deterministically regardless of
  today.
- Synthesizes champion vs challenger return series with a large mean
  gap and low noise (seeded) that clear all three gates: Sortino
  delta 119 (>= 0.3), DD ratio 0.80 (<= 1.2), PBO 0.10 (< 0.2).
- Uses `state_path=<tempdir>/drill_state.json` so real
  `handoff/logs/monthly_approval_state.json` is never touched.
- Captures slack_fn + bq_fn calls into in-memory lists -- no real
  Slack, no real BQ.
- Step 1: runs the gate, asserts `fired=True`, `gate_pass=True`,
  `approval_pending=True`; captures the slack ping.
- Step 2: reads the temp state file, asserts `status=pending` and
  challenger_id/expires_at_iso are present.
- Step 3: `record_approval(month_key, status="approved", bq_fn=capture)`
  asserts JSON transitions to `approved` with a `resolved_at_iso`.
- Step 4: asserts exactly 1 bq_fn call with the right strategy_id and
  status.
- Cleans up the temp dir on exit.

## Files changed

1. `backend/autoresearch/monthly_champion_challenger.py` (signature
   + `_emit_deployment_log_row` helper).
2. `backend/api/monthly_approval_api.py` (`_default_bq_logger` + wire
   into `post_monthly_approval`).
3. `scripts/go_live_drills/hitl_gate_drill.py` (new).

## Verification command output (verbatim)

```
$ grep -c "bq_fn" backend/autoresearch/monthly_champion_challenger.py
9
$ grep -c "_default_bq_logger" backend/api/monthly_approval_api.py
3
$ python -c "import ast; ast.parse(open('backend/autoresearch/monthly_champion_challenger.py').read())" && echo SYNTAX_MCC_OK
SYNTAX_MCC_OK
$ python -c "import ast; ast.parse(open('backend/api/monthly_approval_api.py').read())" && echo SYNTAX_API_OK
SYNTAX_API_OK
$ python -c "import ast; ast.parse(open('scripts/go_live_drills/hitl_gate_drill.py').read())" && echo SYNTAX_DRILL_OK
SYNTAX_DRILL_OK
$ python -c "from backend.autoresearch import monthly_champion_challenger; print('IMPORT_MCC_OK')"
IMPORT_MCC_OK
$ python -c "from backend.api import monthly_approval_api; print('IMPORT_API_OK')"
IMPORT_API_OK

$ python scripts/go_live_drills/hitl_gate_drill.py
step1_gate_fired: sortino_delta=119.281 dd_ratio=0.800 pbo=0.100
step2_pending: status=pending challenger_id=DRILL-2026-03 expires_at_iso=2026-03-29T01:00:00+00:00
step3_approved: status=approved resolved_at_iso=2026-03-28T10:00:00+00:00
step4_bq_row_written: strategy_id=DRILL-2026-03 status=approved pbo=0.1 deployed_at=2026-03-28T10:00:00+00:00
PASS
```

All 10 contract criteria green.

## Success-criteria coverage

| # | Criterion | Evidence |
|---|---|---|
| 1 | record_approval signature contains `bq_fn` | PASS (grep count 9) |
| 2 | record_approval calls bq_fn on terminal transitions inside try/except | PASS (via `_emit_deployment_log_row` helper) |
| 3 | `_default_bq_logger` exists + wired into post_monthly_approval | PASS (grep count 3) |
| 4 | hitl_gate_drill.py exists, ast.parse ok | PASS |
| 5 | drill exits 0 prints PASS | PASS |
| 6 | drill output contains all 4 step evidence lines | PASS |
| 7 | monthly_champion_challenger.py syntax valid | PASS |
| 8 | monthly_approval_api.py syntax valid | PASS |
| 9 | monthly_champion_challenger imports clean | PASS |
| 10 | monthly_approval_api imports clean | PASS |

## Scope discipline

- **Did NOT** post to real Slack. Drill uses an in-memory capture list
  so no message is sent. Peder-interactive real-Slack test remains a
  follow-up cycle (not a pre-prod blocker).
- **Did NOT** write to real BQ from the drill. The drill uses an
  in-memory bq_fn capture. Production API endpoint DOES now write to
  real BQ via `_default_bq_logger`, but that's exercised through live
  traffic / a separate manual verification, not this drill.
- **Did NOT** lift the `actual_replacement=False` invariant (SR 11-7
  compliance pass, out of scope for pre-prod).
- **Did NOT** change gate thresholds (Sortino delta 0.3 / PBO 0.2 / DD
  ratio 1.2).
- **Did NOT** touch the existing unit tests (backward-compatible).

## Notes / follow-ups

- **Peder-interactive real-Slack drill**: once Peder is ready for a
  true human-in-the-loop rehearsal, the same infrastructure supports
  it. Call `run_monthly_sortino_gate` with a TEST-labelled
  `challenger_id` and `slack_fn=<real>`. Then Peder clicks or posts to
  `POST /api/harness/monthly-approval/<month_key>`. The API endpoint
  will write a real BQ row via `_default_bq_logger`. This is NOT a
  pre-prod blocker -- the internal pipeline is proven -- but a
  recommended rehearsal before BLOCKER-4 (paper->live transition).
- **BQ integration sanity**: to verify `_default_bq_logger` against
  real BQ, run the drill then issue a real POST to a test month:
  `curl -X POST http://localhost:8000/api/harness/monthly-approval/2026-XX -d '{"action":"approved"}' -H 'Content-Type: application/json'`
  and SELECT from `pyfinagent_pms.strategy_deployments_log`. Not a
  blocker because the fail-open pattern means a BQ write-error won't
  block approvals.
