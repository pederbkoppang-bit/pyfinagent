# Live-check placeholder -- phase-25.C3

**Step:** 25.C3 -- Strategy registry status field; flip actual_replacement
**Date:** 2026-05-12

## Live-check field (per masterplan)
> "Monthly HITL approval test flips registry status atomically"

## Pre-deployment evidence
- 12/12 verifier PASS (`source .venv/bin/activate && python3 tests/verify_phase_25_C3.py`)
- 4 behavioral round-trips:
  - **Approval flip:** `record_approval(approved)` invokes `status_update_fn(challenger_id, "active")` exactly once.
  - **Rejection no-flip:** `record_approval(rejected)` does NOT invoke `status_update_fn`.
  - **Derived actual_replacement:** with `real_capital_enabled=True` -> `actual_replacement=True`; with `=False` (default) -> `actual_replacement=False`.
  - **BQ UPDATE round-trip:** `update_promoted_strategy_status` builds parameterized UPDATE + calls `result(timeout=30)`.
- Backend AST clean for all 3 touched files.

## Post-deployment operator workflow
1. (Prereq) 25.A3 migration applied (`--apply`).
2. Trigger the monthly gate with a non-None bq_client + the `real_capital_enabled` flag (default False):
   ```python
   from backend.autoresearch.monthly_champion_challenger import run_monthly_sortino_gate, record_approval
   from backend.db.bigquery_client import BigQueryClient
   from backend.config.settings import get_settings
   settings = get_settings()
   bq = BigQueryClient(settings)
   result = run_monthly_sortino_gate(
       eval_date=<last trading Friday>,
       champion_returns=...,
       challenger_returns=...,
       champion_max_dd=..., challenger_max_dd=...,
       challenger_pbo=...,
       challenger_id="trial_42",
       real_capital_enabled=settings.real_capital_enabled,  # False by default
   )
   ```
3. After the 48-hour HITL window, record the approval AND flip BQ status:
   ```python
   record_approval(
       month_key="2026-05",
       status="approved",
       status_update_fn=bq.update_promoted_strategy_status,
   )
   ```
4. Verify the BQ row is now active:
   ```sql
   SELECT strategy_id, status, promoted_at
   FROM `sunny-might-477607-p8.pyfinagent_data.promoted_strategies`
   WHERE strategy_id = 'trial_42'
   ORDER BY promoted_at DESC
   LIMIT 1;
   ```
   Expected: `status = 'active'`.

## SR 11-7 invariant preserved
`Settings.real_capital_enabled` defaults to False. The derived `actual_replacement`
flag is False unless an operator explicitly toggles the Setting after a
compliance review. The audit-log notes line now reads
`actual_replacement=False` (driven by the flag) rather than being hardcoded;
the value is materially identical for now but the indirection is the
substance of the audit fix.

## Downstream
Unblocks **25.R** (auto-switching policy / red-line goal-c).

**Audit anchor for next bucket:** 25.R (auto-switching policy) or 25.B (P2 cosmetic-patch cleanup).
