---
step: phase-23.1.11
title: Persist lite-Claude analyzer rows to analysis_results so Reports History tab shows paper-trading candidates
cycle_date: 2026-04-27
harness_required: true
verification: 'source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_11.py'
research_brief: handoff/current/phase-23.1.11-research-brief.md
---

# Contract — phase-23.1.11

## Hypothesis

A new `_persist_lite_analysis(analysis, bq)` helper called after every successful `_run_single_analysis()` in the autonomous-loop Steps 3+4 writes a row to `analysis_results` with the ~14 fields the lite path actually has (NULLs on the other ~74). Reports page History tab surfaces paper-trading candidates immediately; existing manual analyses are unchanged. **Path A** (extend existing table; no new table, no migration).

## Plan

1. **Backend `backend/services/autonomous_loop.py`** — three changes:
   - Fix the hardcoded `"source": "claude-sonnet-4"` in `_run_claude_analysis`'s return dict (line ~571) → use the actual `model_name` variable so the audit trail reflects the real model.
   - Enrich `full_report.market_data` with `name` and `industry` fields (already-computed local vars).
   - NEW `_persist_lite_analysis(analysis, bq)` helper: calls `bq.save_report(...)` with the 14 fields available from the lite path. Wraps in try/except — failures log but don't fail the cycle.
   - Call it after every successful `_run_single_analysis` in Step 3 (candidate analysis loop) AND Step 4 (re-eval loop). Guarded by `settings.lite_mode` so we don't double-write when the full Gemini fallback path runs (which already calls `bq.save_report` itself).

2. **Tests** at `tests/services/test_persist_lite_analysis.py`:
   - `_persist_lite_analysis` calls `bq.save_report` with the right field mapping
   - Missing optional fields (no PE ratio, no industry) → still writes, NULLs underneath
   - `bq.save_report` raising → logs but does NOT propagate (cycle continues)
   - The `model_name` flows into `full_report.source` (no more hardcoded "claude-sonnet-4")
   - `name` and `industry` flow into `full_report.market_data`

3. **Verification script** `tests/verify_phase_23_1_11.py` — imports the helper + autonomous_loop, asserts:
   - `_persist_lite_analysis` is importable + async-callable
   - Helper signature `(analysis: dict, bq)`
   - When called with a synthetic lite-shape analysis dict and a stub BQ client, it invokes `bq.save_report(...)` exactly once with `ticker`, `recommendation`, `final_score`, `summary`, `company_name` populated correctly

## Out of scope

- New `analysis_source` discriminator column (Phase 2 — needs migration --apply; the brief notes it's a "follow-up, no blocking" — `full_report_json.source` already carries the model name)
- Frontend "Lite Analysis" badge on the report detail page (Phase 2 — defensive null-checks on the frontend should already gracefully hide debate/bull/bear sections when missing)
- New `paper_trading_analyses` table (Path B — explicitly rejected per brief)
- `outcome_tracker` enhancements to use lite rows (the existing path-agnostic reader picks them up automatically; only `price_at_rec` extraction may silently fall back)

## Files modified

- `backend/services/autonomous_loop.py` — fix model_name + enrich market_data + NEW `_persist_lite_analysis` + 2 call sites in Step 3 + Step 4
- `tests/services/test_persist_lite_analysis.py` — NEW (~5 tests)
- `tests/verify_phase_23_1_11.py` — NEW immutable verification script
- `handoff/current/{contract,experiment_results,evaluator_critique}.md` — rolling

## Verification

The verification script asserts the helper:
- Is importable + async
- Has the documented signature
- Calls `bq.save_report` exactly once with the documented field mapping
- Doesn't propagate exceptions (cycle survives BQ outage)

## What this fixes

| Before | After |
|---|---|
| `analysis_results WHERE analysis_date >= '2026-04-26'` returns 0 rows | Each scheduled cycle adds N rows (one per analyzed candidate) |
| Reports page History tab shows only old manual analyses | Paper-trading candidates (COHR, KEYS, GEV, etc.) appear in History |
| `outcome_tracker.evaluate_recommendation` has no analysis context | Has the lite analysis row to retrieve (limited but better than nothing) |
| Hardcoded `"source": "claude-sonnet-4"` lies about which model ran | `"source": model_name` reflects the actual model used |

## References

- `handoff/current/phase-23.1.11-research-brief.md` — full brief (323 lines, 5 sources read in full, gate_passed: true)
- `backend/services/autonomous_loop.py:466-583` — `_run_claude_analysis` (the return-dict shape we persist)
- `backend/db/bigquery_client.py:41-310` — `save_report` (the canonical INSERT path)
- `backend/api/reports.py` — Reports endpoint (no change needed; works automatically once rows land)
- Phase-23.1.7 experiment_results — original "Gap 3" deferral note
