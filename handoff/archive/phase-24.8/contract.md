# Sprint Contract — phase-24.8 — Observability + Safety Rails Audit

**Cycle:** phase-24 cycle 8
**Date:** 2026-05-12
**Step ID:** 24.8
**Priority:** P1

## Research-gate
`gate_passed: true` (tier=complex). 6 sources: harness-design, Google SRE Book monitoring, arxiv AutoGuard kill-switch, MS Agent Governance Toolkit April 2026, Sakura Sky kill-switch primitives, fahimulhaq practitioner post.

```json
{"tier":"complex","external_sources_read_in_full":6,"snippet_only_sources":12,"urls_collected":18,"recency_scan_performed":true,"internal_files_inspected":15,"gate_passed":true}
```

## Hypothesis
Safety rails exist but may not be wired end-to-end. Kill-switch UI reachability, watchdog action, cost-budget hard-block are all suspect.

**Researcher verdict: CONFIRMED with partial good news.**

**Working:**
- Kill-switch operator-reachable from `OpsStatusBar.tsx:96-130` (PAUSE/RESUME/FLATTEN_ALL buttons)
- Auto-pause fires Slack alert (`kill_switch.py:144-156`)
- Watchdog fires `kickstart -k` + Slack (verified — 3 restarts in 12 days)

**Critical gaps:**
- Cost budget `tripped=True` is honor-system only — `llm_client.py` has zero budget checks. New cycle after breach makes LLM calls normally.
- SLA escalation uses `imsg` (non-standard binary) with no Slack fallback
- Governance watcher `os._exit(2)` sends no Slack alert before terminating

## Success criteria (verbatim)
1. findings_md_exists
2. research_gate_envelope_present_with_gate_passed_true
3. external_sources_count_at_least_5
4. canonical_url_cited_verbatim_observability_api_py
5. recency_scan_2024_2026_section_present
6. at_least_three_phase_25_candidate_steps_proposed
7. each_candidate_step_has_files_list_with_absolute_paths
8. each_candidate_step_has_draft_verification_command
9. harness_log_has_phase_24_24_8_cycle_entry
10. executive_summary_section_present
11. findings_audits_watchdog_wiring
12. findings_audits_killswitch_state_machine
13. findings_audits_sla_monitor_paths
14. findings_audits_cost_budget_enforcement
15. findings_audits_governance_limits_hot_reload

**Verifier:** `python3 tests/verify_phase_24_8.py`

## Plan
1. Findings doc
2. experiment_results.md
3. Q/A
4. Cycle 49 log append
5. live_check_24.8.md
6. Flip 24.8 to done
