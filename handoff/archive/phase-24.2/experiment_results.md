---
step: phase-24.2
cycle: 5
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_24_2.py'
title: Pipeline routing + report persistence audit (P1)
---

# Experiment Results — phase-24.2

**Action:** READ-ONLY. Produced findings + brief + contract. No code changes.

## Verbatim verifier output

```
=== phase-24.2 (pipeline-routing) verifier ===
  [PASS] findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_2_pipeline_routing_findings_md
  [PASS] research_gate_envelope_present_with_gate_passed_true
  [PASS] external_sources_count_at_least_5
  [PASS] canonical_url_cited_verbatim_autonomous_loop_py
  [PASS] recency_scan_2024_2026_section_present
  [PASS] at_least_three_phase_25_candidate_steps_proposed
  [PASS] each_candidate_step_has_files_list_with_absolute_paths
  [PASS] each_candidate_step_has_draft_verification_command
  [FAIL] harness_log_has_phase_24_24_2_cycle_entry
         -> harness_log.md must contain `## Cycle N -- ... phase=24.2 result=...` header
  [PASS] executive_summary_section_present
  [PASS] findings_cites_autonomous_loop_branching_at_564_615
  [PASS] findings_documents_why_full_28_skill_pipeline_is_unreached
  [PASS] findings_traces_report_persistence_lite_vs_full_gap
FAIL (12/13) EXIT=1
```

12/13 PASS. Log-last gap is the only FAIL — expected.

## Hypothesis verdict
CONFIRMED with correction: branch is at `autonomous_loop.py:575` (`if settings.lite_mode:`). BOTH lite AND full paths fail to populate `/reports` from paper-trading. `orchestrator.py` has ZERO `save_report` calls. Only `/api/analysis` (`analysis.py:201`) writes to the reports table. Lite path is correctly guarded but full path returns no `_path` marker, so `_persist_lite_analysis` is bypassed for full runs and no `_persist_full_analysis` exists. Default `settings.lite_mode = False` (settings.py:119) — full pipeline is the default but its outputs evaporate.

Skills count: 31 (not 28 per master prompt). Cost ratio: lite ~$0.01/ticker; full ~$0.10-0.20 (10-20x).

## Phase-25 candidates (5)
- 25.A2 (P0) — Wire `bq.save_report` into full pipeline (fixes empty /reports)
- 25.B2 (P1) — Unify persistence under `_persist_analysis` with `_path` marker
- 25.C2 (P1) — Per-ticker `full_pipeline_tickers` cost-bounded routing
- 25.D2 (P2) — A/B quality lift measurement (lite vs full)
- 25.E2 (P2) — `/reports` lite-vs-full badge

## Next phase
EVALUATE — Q/A spawn pending.
