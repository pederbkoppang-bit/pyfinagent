---
step: phase-23.1.12
title: Honor operator's model choice (remove hardcoded lite_mode override) + cycle pill amber-on-unknown
cycle_date: 2026-04-27
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_12.py'
research_brief: handoff/current/phase-23.1.12-research-brief.md
---

# Contract — phase-23.1.12

## Hypothesis

Two bug fixes:

**Bug 1 — Model choice ignored.** Removing the hardcoded `settings.lite_mode = True` in `autonomous_loop.py` Step 3 lets `_run_single_analysis` branch on the operator's actual `lite_mode` setting. Existing `paper_max_daily_cost_usd` cap continues to bound runaway costs. Operators who picked Sonnet/Opus get the full Gemini orchestrator pipeline (which respects their model choice via `gemini_model` + `deep_think_model`).

**Bug 2 — Cycle pill green-on-unknown.** `OpsStatusBar.tsx` aggregation treats `"unknown"` as not-amber. Fix: collapse `unknown ⇒ amber` so the pill reflects the worst-of-N component state per Google SRE / Azure WAF convention.

## Plan

1. **Backend `backend/services/autonomous_loop.py`** Step 3:
   - Remove the `settings.lite_mode = True` / `original_lite = ...` / `settings.lite_mode = original_lite` lines (currently lines ~213-216 and ~248).
   - Refactor `_run_single_analysis` to branch on `settings.lite_mode`:
     - `lite_mode == True` → call `_run_claude_analysis` (current lite path), no Gemini fallback
     - `lite_mode == False` → call the full Gemini orchestrator first, fall back to lite Claude if orchestrator fails
   - Default `settings.lite_mode = False` already exists in `backend/config/settings.py`. Operator can still flip it via `Manage tab → Trading settings` (already shipped phase-23.1.9 — but `lite_mode` isn't in that list yet; add it).
   - Update the `_persist_lite_analysis` guard from `if settings.lite_mode` to also persist when the full path was used (the orchestrator already calls `bq.save_report`, so this stays as-is — guard correctly prevents double-write).

2. **Frontend `frontend/src/components/OpsStatusBar.tsx`** lines 273-279:
   - Change the `worst` aggregation so `unknown` collapses to `amber` (degraded), not silently to green/gray.

3. **Backend `backend/api/settings_api.py`** + `backend/api/paper_trading.py`:
   - Add `lite_mode` to the writable subset for the Manage tab so the operator can opt back into lite mode if they want speed/cost over depth. (lite_mode already in FullSettings + SettingsUpdate from phase-23.1.6 — verify and add to the Manage tab UI if missing.)

4. **Frontend `frontend/src/app/paper-trading/page.tsx`** Manage tab:
   - Add `lite_mode` toggle to the Trading Settings card (between existing inputs).

5. **Tests** at `tests/services/test_run_single_analysis_branch.py`:
   - `lite_mode=True` → `_run_claude_analysis` called once, Gemini orchestrator NOT called
   - `lite_mode=False` → `AnalysisOrchestrator.run_full_analysis` called, `_run_claude_analysis` only as fallback
   - Step 3 cycle no longer mutates `settings.lite_mode` (settings are read-only during cycle execution)

6. **Tests** at `tests/frontend/test_ops_status_bar_aggregation.ts` OR (if Vitest not set up) document the rule + verify via TypeScript that the new logic compiles.

7. **Verification script** `tests/verify_phase_23_1_12.py` — asserts:
   - `settings.lite_mode = True` literal absent from `_run_single_analysis` and Step 3 of `run_paper_trading_cycle`
   - `_run_single_analysis` branches on `settings.lite_mode` (mocked test)
   - The OpsStatusBar aggregation rule is `unknown ⇒ amber` (read the source file and grep for the rule)

## Out of scope

- Cost-aware model fallback (Phase 2 — e.g., "if Opus exceeds cap, downgrade to Sonnet")
- Async parallel analysis (currently 1 ticker at a time; Phase 2)
- Per-ticker model selection (Phase 2)
- New `cycle_health` BQ table to populate `paper_trades` and `paper_snapshots` statuses (Phase 2 — currently both return `unknown` because the `compute_freshness` helper just doesn't read those rows yet; that's a deeper bug for a future cycle)

## Files modified

- `backend/services/autonomous_loop.py` — remove lite_mode override + refactor `_run_single_analysis` branch
- `frontend/src/components/OpsStatusBar.tsx` — fix unknown ⇒ amber aggregation
- `frontend/src/app/paper-trading/page.tsx` — add `lite_mode` toggle to Manage tab Trading Settings
- `tests/services/test_run_single_analysis_branch.py` — NEW (4 tests)
- `tests/verify_phase_23_1_12.py` — NEW immutable verification script
- `handoff/current/{contract,experiment_results,evaluator_critique}.md` — rolling

## Verification

The verification script grep-asserts that:
1. `settings.lite_mode = True` does NOT appear in `_run_single_analysis` or in Step 3 of `run_paper_trading_cycle`
2. `_run_single_analysis` has an `if settings.lite_mode` branch
3. `OpsStatusBar.tsx` contains the `unknown` substring in the amber clause of the worst-of-N aggregator

## What this fixes

| Before | After |
|---|---|
| Operator picks Sonnet 4.6 + Opus 4.6 → cycle silently uses lite 4-field Claude prompt | Operator's choice respected; full orchestrator runs; debate/bull/bear/risk persisted to `analysis_results` |
| `settings.lite_mode = True` hardcoded mutates Settings during cycle execution | Settings read-only during cycle; operator-configured value preserved |
| Cycle pill GREEN despite `paper_trades: unknown` and `paper_snapshots: unknown` | Cycle pill AMBER when any component status is unknown — worst-of-N convention |
| Hidden cost-control behavior surprises operator | Operator-visible `lite_mode` toggle in Manage tab; cost cap still active as circuit breaker |

## References

- `handoff/current/phase-23.1.12-research-brief.md` — full brief (344 lines, 5 sources read in full, gate_passed: true)
- `backend/services/autonomous_loop.py:213-216` — the override to remove
- `backend/services/autonomous_loop.py:420-463` — `_run_single_analysis` to refactor
- `frontend/src/components/OpsStatusBar.tsx:273-279` — the aggregation to fix
- TradingAgents / FinCon literature on cost-as-circuit-breaker rather than silent degradation
- Google SRE / Azure WAF convention on health-status aggregation (worst-of-N)
