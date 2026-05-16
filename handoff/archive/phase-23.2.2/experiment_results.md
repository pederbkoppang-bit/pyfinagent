---
step: 23.2.2
slug: trade-position-reconciliation-regression
cycle: post-phase-26-cleanup
date: 2026-05-16
researcher_id: aeec30a118f1fe213
research_gate_passed: true
research_tier: complex
max_effort_directive: applied  # per user 2026-05-16 "mas agents all running max effort"
verdict_by_main: PASS  # Q/A is authoritative
---

# Experiment Results -- phase-23.2.2 Verify zero phantom trades / cash-leak regressions

## File list

No source code modified -- this is a PURE VERIFICATION step.

Files written this step:
- `handoff/current/research_brief.md` (Main internal + researcher_aeec30a118f1fe213 external; composed-brief pattern)
- `handoff/current/contract.md`
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/live_check_23.2.2.md` (verbatim BQ query stdout for the 3 reconciliation queries)

BQ tables read (no writes):
- `sunny-might-477607-p8.financial_reports.paper_trades` (15 rows, 18 cols)
- `sunny-might-477607-p8.financial_reports.paper_positions` (13 rows, 19 cols)
- `sunny-might-477607-p8.financial_reports.paper_portfolio` (1 row, 10 cols)

## Plan-step 1: Exploratory schema check

Confirmed paper_trades + paper_positions + paper_portfolio all exist in `financial_reports` (us-central1). `bigquery_client._pt_table` resolves to `{project}.{bq_dataset_reports}.{name}` where `bq_dataset_reports = financial_reports`. The pyfinagent_pms dataset does NOT have the paper_* tables (operator note: the masterplan / CLAUDE.md may refer to "pyfinagent_pms" generically but the actual tables live in financial_reports).

## Plan-step 2-3: FULL OUTER JOIN + cash invariant

See `handoff/current/live_check_23.2.2.md` for verbatim stdout.

**Reconciliation summary (14 ticker rows):**
- MATCHED: 13 (CIEN, COHR, DELL, FIX, GEV, GLW, INTC, KEYS, LITE, MU, ON, SNDK, WDC)
- CLOSED_OK: 1 (TER -- properly-closed round-trip; 1 BUY + 1 SELL = net qty 0, no position row)
- ORPHAN_BUY: 0
- PHANTOM_POSITION: 0
- QTY_BREAK: 0

**Cash invariant:**
- starting_capital: $20,000.00
- current_cash: $7,587.44
- total_open_value: $15,314.36
- nav_recomputed: $22,901.80
- nav_stored: $22,901.81
- **nav_break: -$0.01** (sub-cent float-rounding; well within $1.00 portfolio-level tolerance from phase-23.1.15 precedent)
- realized_pnl_since_inception: -$798.60 at cost basis (== current_cash - (starting_capital - cost_basis) -- TER round-trip P&L)
- Total P&L (open + closed): +14.51% -> +$2,901.81 / $20,000

## Plan-step 4: Capture + classify

All 14 ticker rows in MATCHED or CLOSED_OK status. NO regression detected.

The phase-23.1.15 fix (which addressed a phantom-trade / cash-leak bug class) is HOLDING. The invariants:
- "every BUY trade has a matching position (or matching SELL closing it)"
- "current_cash + open_position_value == stored_nav within $1 tolerance"
both hold across the current portfolio.

## Sub-criteria self-summary (NOT a verdict)

The masterplan verification is a SINGLE STRING (no sub-criteria list, no live_check field). Operationalized:
- ✓ `orphan_trades = 0` (Evidence A)
- ✓ `leak_dollars ≤ $0.01` rounding (Evidence B; well within $1.00 tolerance)

PASS.

## Scope honesty

In scope, completed:
- 3 BQ reconciliation queries against live paper_trades / paper_positions / paper_portfolio ✓
- FULL OUTER JOIN exactly per the brief's canonical pattern ✓
- Cash invariant double-check (cash + open_value vs stored NAV) ✓
- Per-action sanity cross-check (14 BUYs + 1 SELL; cash trail balances) ✓
- Honest disclosure of dataset location (financial_reports, not pyfinagent_pms; CLAUDE.md may be slightly misleading on this) ✓

Out of scope:
- No code changes; this is a verification step.
- No re-running of the prior phase-23.1.15 fix (that already happened; this step confirms it hasn't regressed).
- Sub-cent ($0.01) nav_break is documented as float-rounding, NOT regression. If operator wants tighter (zero) tolerance, the fix is in the paper_trader cash-update path (use Decimal not float for cash arithmetic) -- separate scope.
- No examination of historical reconciliation breaks (the query is over current state only). Backfilling historical break detection across all snapshots is a phase-27 affordance.

Honest note on dataset path: CLAUDE.md references `pyfinagent_pms` in its BQ section, but the actual paper_* tables live in `financial_reports` (per `bq_dataset_reports = financial_reports` in settings, and `_pt_table()` in bigquery_client.py). This inconsistency was surfaced during 23.2.2 schema check; operator follow-on: clarify CLAUDE.md or move tables to align with documentation.

## Verdict-by-Main (self-summary, NOT authoritative)

All success criteria literal-satisfied. 0 orphan trades, 0 phantom positions, 0 quantity breaks. Cash invariant within sub-cent rounding ($0.01 << $1.00 tolerance). The phase-23.1.15 bug class has NOT regressed. The portfolio is up +14.51% with the only realized P&L coming from one closed round-trip (TER).

Step 23.2.2 is ready for Q/A evaluation.
