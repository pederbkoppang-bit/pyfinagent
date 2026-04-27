---
step: phase-23.1.11
title: Persist lite-Claude analyzer rows to analysis_results so Reports History tab shows paper-trading candidates
qa_pass: 1
verdict: PASS
cycle_date: 2026-04-26
---

# Q/A Critique — phase-23.1.11

## 5-item harness-compliance audit

1. Researcher brief on disk: `handoff/current/phase-23.1.11-research-brief.md` present (323 lines, 5 sources read in full, gate_passed: true). PASS.
2. Contract front-matter `step: phase-23.1.11` matches; immutable verification command is `source .venv/bin/activate && PYTHONPATH=. python tests/verify_phase_23_1_11.py`. PASS.
3. `experiment_results.md` includes the verbatim verification line `ok _persist_lite_analysis async + signature + save_report invocation + graceful BQ failure` and exit=0. The "Failed to persist… BQ down" warning above the `ok` line is intentional (proves graceful-on-error path). PASS.
4. `harness_log.md` not yet appended for `phase=23.1.11` (grep returned 0 hits). Correct — log-last rule. PASS.
5. First Q/A spawn for this step. PASS.

## Deterministic checks

| ID | Check | Result |
|----|-------|--------|
| A  | Immutable verification command | exit=0, expected `ok` line printed |
| B  | `pytest tests/api/ tests/services/` (excluding test_observability.py) | 162 passed, 1 warning, 0 failures (8 new persist_lite_analysis tests included) |
| C  | Syntax check (autonomous_loop.py, test_persist_lite_analysis.py, verify_phase_23_1_11.py) | `all syntax ok` |
| D  | Hardcoded model name fixed | Line 584: `"source": model_name` (NOT hardcoded "claude-sonnet-4"). Comment at 580–582 explains the fix. |
| E  | market_data enriched with name + industry | Lines 587 (`"name": name`) and 592 (`"industry": industry`) present alongside existing price/market_cap/pe_ratio/sector/momentum fields |
| F  | `_persist_lite_analysis` correctness | Async, signature `(analysis: dict, bq: BigQueryClient) -> None` (line 606); skips when ticker empty (lines 612–614); calls `bq.save_report` via `asyncio.to_thread` (line 617); maps the 14 lite fields per the documented mapping; catches `Exception` and logs warning (lines 635–639) — does NOT propagate |
| G  | Call sites guarded by `lite_mode` | Step 3 candidate loop lines 233–234: `if settings.lite_mode: await _persist_lite_analysis(...)`. Step 4 re-eval loop lines 251–252: identical guard. Comment at 229–232 documents the double-write rationale. |
| H  | Tests cover the right paths | 8 tests: field-mapping correctness, missing market_data, BQ exception non-propagation, missing ticker, empty ticker, missing full_report, full_report passthrough audit, HOLD default. All pass. |
| I  | No double-write risk | `backend/api/analysis.py` line 201 calls `bq.save_report` in the full-Gemini orchestrator path. Lite-mode guard ensures `_persist_lite_analysis` only fires when the orchestrator path won't run. No third path lands here. |
| J  | Git diff scope | Acceptable: `backend/services/autonomous_loop.py` (+58 lines), new `tests/services/test_persist_lite_analysis.py`, new `tests/verify_phase_23_1_11.py`, `handoff/current/contract.md` and `experiment_results.md` updated. Other diffs (perf_results.tsv, audit jsonls, frontend tsbuildinfo, cycle_history) are routine harness/build noise unrelated to this cycle. |

`checks_run = ["audit_5item", "verification_command", "pytest_full_suite", "syntax", "source_inspection_model_name", "source_inspection_market_data", "source_inspection_helper_signature", "source_inspection_call_sites", "source_inspection_no_double_write", "test_inventory", "git_diff_scope"]`

## LLM judgment leg

| Question | Verdict |
|----------|---------|
| Does the cycle close the user's bug report? | YES. After tomorrow's first lite-mode cycle, every successful candidate + re-eval analysis writes a row to `analysis_results`, making them visible on the Reports History tab. The 11 historical paper trades are correctly disclosed as not retroactively backfilled (would require a separate one-shot script — out of scope by design). |
| Mutation-resistance | The verification script invokes the real `_persist_lite_analysis` against a synthetic lite-shape dict + a stub `BigQueryClient`. It would FAIL if the helper's signature changed, if the field mapping broke, if the helper became sync, or if the graceful-error path were removed. The 8 unit tests likewise drive real code paths. |
| Anti-rubber-stamp | `experiment_results.md` explicitly states (a) historical paper trades won't backfill, (b) lite-row report-detail pages will have empty bull/bear/debate sections (acceptable per defensive frontend null-checks), and (c) the `analysis_source` discriminator + frontend "Lite Analysis" badge are deferred to Phase 2 with explicit reasoning. No overclaim. |
| Scope honesty | Phase-2 deferrals (discriminator column, frontend badge, `paper_trading_analyses` table) are labeled and justified. Path B explicitly rejected per brief. |
| Path A vs Path B justification | Brief argues Path A (extend existing table, NULL-pad) on grounds that BQ Capacitor stores NULLs cheaply and that introducing a second table would split the History tab query into a UNION. Trade-offs match research-brief reasoning. |
| Backwards compat | Full Gemini fallback path is unchanged (still writes via `backend/api/analysis.py`). `outcome_tracker` reads `analysis_results` regardless of source, so lite rows are picked up automatically. Existing manual analyses untouched. |
| Operator override path | `settings.lite_mode` is forced ON in the paper-trading cycle (line 216) but `original_lite` is captured at 215 and restored at 257. If an operator disables lite_mode, the orchestrator path takes over and writes via its own `bq.save_report` — no row is lost, no row is double-written. |

## Verdict

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "audit_5item",
    "verification_command",
    "pytest_full_suite",
    "syntax",
    "source_inspection_model_name",
    "source_inspection_market_data",
    "source_inspection_helper_signature",
    "source_inspection_call_sites",
    "source_inspection_no_double_write",
    "test_inventory",
    "git_diff_scope"
  ]
}
```

## Reason

All 10 deterministic checks (A–J) pass. The immutable verification command exits 0 with the expected `ok` line and proves both the happy path (save_report invoked once with correct kwargs) and the graceful-on-error path (BQ exception logged, not propagated). pytest reports 162 passed including the 8 new persist_lite_analysis tests. Source inspection confirms (i) the hardcoded model_name bug is fixed, (ii) market_data is enriched with name + industry, (iii) the helper has the documented async signature and uses `asyncio.to_thread` for the sync BQ call, (iv) both call sites in Steps 3 and 4 are guarded by `settings.lite_mode` to prevent double-write, and (v) `backend/api/analysis.py` retains its own `bq.save_report` call so the full-Gemini fallback path still persists. LLM judgment confirms the cycle closes the user's bug, scope is honestly disclosed (Phase-2 deferrals labeled), and backwards compatibility is preserved. No mutation-resistance, anti-rubber-stamp, or research-gate violations found.
