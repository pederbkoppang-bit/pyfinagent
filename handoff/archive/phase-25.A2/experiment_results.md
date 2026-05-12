---
step: phase-25.A2
cycle: 66
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_25_A2.py'
title: Wire bq.save_report into full pipeline persistence (P1; first P1 in sprint)
---

# Experiment Results — phase-25.A2

## Code changes (single file: `backend/services/autonomous_loop.py`)

1. **Full-pipeline return dict (L649)**: added `"_path": "full"` marker
2. **Persist guards (L277, L295)**: `if analysis.get("_path") == "lite":` → `if analysis.get("_path") in ("lite", "full"):`
3. **Function rename (L795)**: `_persist_lite_analysis` → `_persist_analysis`; docstring corrected
4. **Stale comment fix (L272-275)**: old comment claimed full path self-persisted via run_full_analysis; corrected to acknowledge it did NOT (orchestrator.py had zero save_report calls per audit)
5. **All 2 callsites updated** to use renamed function

## Verbatim verifier output

```
=== phase-25.A2 (full-pipeline persistence) verifier ===
  [PASS] full_pipeline_return_dict_includes_path_full_marker
  [PASS] persist_guards_accept_both_lite_and_full_paths
  [PASS] persist_analysis_function_defined
  [PASS] all_persist_callsites_use_renamed_function_no_legacy_left
  [PASS] phase_25_A2_attribution_comment_present
  [PASS] stale_comment_about_run_full_analysis_self_persisting_corrected_or_removed
  [PASS] autonomous_loop_py_syntax_clean
  [PASS] persist_analysis_calls_bq_save_report
PASS (8/8) EXIT=0
```

8/8 PASS.

## Hypothesis verdict
CONFIRMED. Full-pipeline runs now persist to `analysis_results` via the same `_persist_analysis` helper used for lite path. `/reports` page will populate with both lite and full rows (operator can see which produced what via `_path` field).

## Live-check
Per masterplan: "Frontend /reports page shows non-zero recent rows after next full-pipeline cycle". Operator confirms via `/reports` after next cycle with `lite_mode=False`.

## Next phase
Q/A pending.
