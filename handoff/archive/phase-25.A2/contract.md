# Sprint Contract — phase-25.A2 — Wire bq.save_report into full pipeline

**Cycle:** phase-25 cycle 10 (FIRST P1 in sprint)
**Date:** 2026-05-12
**Step ID:** 25.A2
**Priority:** P1

## Research-gate
Reuses phase-24.2 cycle 5 researcher gate.

## Hypothesis
Adding `_path: "full"` to the full-pipeline return dict + updating persist guards to accept both lite and full paths + renaming `_persist_lite_analysis` to `_persist_analysis` will close the empty /reports bug.

## Success criteria (verbatim)
1. grep_save_report_in_autonomous_loop_returns_match
2. reports_table_grows_per_full_pipeline_run
3. stale_comment_at_autonomous_loop_py_273_corrected

## Plan
1. Add `"_path": "full"` to full-pipeline return dict
2. Generalize persist guards to accept both paths
3. Rename `_persist_lite_analysis` → `_persist_analysis` with corrected docstring
4. Update stale comment that claimed full-path self-persisted
5. Verifier `tests/verify_phase_25_A2.py` (8 claims)
6. Q/A
7. Cycle 66 log
8. Flip 25.A2

## References
- `docs/audits/phase-24-2026-05-12/24.2-pipeline-routing-findings.md` F-2
- `backend/services/autonomous_loop.py:646,649,277,295,795`
