---
step: phase-24.8
cycle: 8
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_24_8.py'
title: Observability + monitoring + safety rails audit (P1)
---

# Experiment Results — phase-24.8

**Action:** READ-ONLY. Findings + brief + contract. No code changes.

## Verbatim verifier output

```
=== phase-24.8 (observability) verifier ===
  [PASS] findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_8_observability_findings_md
  [PASS] research_gate_envelope_present_with_gate_passed_true
  [PASS] external_sources_count_at_least_5
  [PASS] canonical_url_cited_verbatim_observability_api_py
  [PASS] recency_scan_2024_2026_section_present
  [PASS] at_least_three_phase_25_candidate_steps_proposed
  [PASS] each_candidate_step_has_files_list_with_absolute_paths
  [PASS] each_candidate_step_has_draft_verification_command
  [FAIL] harness_log_has_phase_24_24_8_cycle_entry
         -> harness_log.md must contain `## Cycle N -- ... phase=24.8 result=...` header
  [PASS] executive_summary_section_present
  [PASS] findings_audits_watchdog_wiring
  [PASS] findings_audits_killswitch_state_machine
  [PASS] findings_audits_sla_monitor_paths
  [PASS] findings_audits_cost_budget_enforcement
  [PASS] findings_audits_governance_limits_hot_reload
FAIL (14/15) EXIT=1
```

14/15 PASS. Log-last only FAIL.

## Hypothesis verdict
CONFIRMED with partial good news: kill-switch UI works (OpsStatusBar.tsx:96-130 + kill_switch.py:144-156 auto-pause + Slack). Watchdog works (3 restarts in 12 days verified). Critical gaps: cost budget honor-system only (llm_client.py no check), SLA imsg-only no Slack fallback, governance watcher os._exit(2) with no pre-exit Slack.

## Phase-25 candidates (5)
- 25.A8 (P0) — Cost-budget HARD-BLOCK in llm_client
- 25.B8 (P1) — SLA Slack fallback (replace imsg-only)
- 25.C8 (P1) — Governance watcher pre-exit Slack alert
- 25.D8 (P1) — Kill-switch hot-key in Slack (third independent trigger)
- 25.E8 (P2) — /observability/status aggregator endpoint

## Next: Q/A
