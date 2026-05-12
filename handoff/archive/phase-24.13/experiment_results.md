---
step: phase-24.13
cycle: 14
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_24_13.py'
title: Profit-maximization red-line alignment synthesis (P1)
---

# Experiment Results — phase-24.13

**Action:** READ-ONLY. Synthesis across 24.1-24.9. No code changes.

## Verbatim verifier output

```
=== phase-24.13 (redline-synthesis) verifier ===
  [PASS] findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_13_redline_synthesis_findings_md
  [PASS] research_gate_envelope_present_with_gate_passed_true
  [PASS] external_sources_count_at_least_5
  [PASS] canonical_url_cited_verbatim_project_system_goal_md
  [PASS] recency_scan_2024_2026_section_present
  [PASS] at_least_three_phase_25_candidate_steps_proposed
  [PASS] each_candidate_step_has_files_list_with_absolute_paths
  [PASS] each_candidate_step_has_draft_verification_command
  [FAIL] harness_log_has_phase_24_24_13_cycle_entry
  [PASS] executive_summary_section_present
  [PASS] synthesis_references_all_prior_buckets_24_1_through_24_9
  [PASS] synthesis_quantifies_cost_vs_pnl_ratio
  [PASS] synthesis_audits_strategy_switching_mechanism
  [PASS] synthesis_audits_cost_budget_enforcement_path
FAIL (13/14) EXIT=1
```

13/14 PASS. Log-last only FAIL.

## Hypothesis verdict
CONFIRMED across 4 compounding misalignments (a-d goals of red-line). Dollar estimates: TER -$1,107 accruing; ~$45/month full-pipeline waste; ~$1.80/month cost under-count. arxiv 2503.21422 confirms NO published trading system has profit-per-LLM-dollar metric — pyfinagent first-mover.

## Phase-25 candidates (5)
- 25.Q (P1) — Real-time `profit_per_llm_dollar` metric (FIRST-MOVER)
- 25.R (P1) — Strategy auto-switching policy (depends on 24.3 25.A3-25.C3)
- 25.S (P2) — Daily P&L attribution report (per SHARP)
- 25.T (cross-link 24.8 25.A8) — Cost-budget HARD-BLOCK (P0)
- 25.U (cross-link 24.6 25.D6) — Plateau-detection enforced rotation (P1)

## Next: Q/A
