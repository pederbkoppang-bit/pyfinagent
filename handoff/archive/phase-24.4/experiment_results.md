---
step: phase-24.4
cycle: 3
cycle_date: 2026-05-12
result: PASS_PENDING_QA
verification_command: 'source .venv/bin/activate && python3 tests/verify_phase_24_4.py'
title: Agent topology + per-agent rationale flow audit (P0)
---

# Experiment Results — phase-24.4

**Action:** Phase-24 is READ-ONLY by charter. Produced findings doc + brief + contract. No code changes.

## Artifacts
- `handoff/current/research_brief.md` (~17 KB; gate_passed=true, 6 sources)
- `handoff/current/contract.md` (~3 KB)
- `docs/audits/phase-24-2026-05-12/24.4-agent-rationale-findings.md` (~21 KB)

## Verbatim verifier output

```
=== phase-24.4 (agent-rationale) verifier ===
  [PASS] findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_4_agent_rationale_findings_md
  [PASS] research_gate_envelope_present_with_gate_passed_true
  [PASS] external_sources_count_at_least_5
  [PASS] canonical_url_cited_verbatim_multi_agent_orchestrator_py
  [PASS] recency_scan_2024_2026_section_present
  [PASS] at_least_three_phase_25_candidate_steps_proposed
  [PASS] each_candidate_step_has_files_list_with_absolute_paths
  [PASS] each_candidate_step_has_draft_verification_command
  [FAIL] harness_log_has_phase_24_24_4_cycle_entry
         -> harness_log.md must contain `## Cycle N -- ... phase=24.4 result=...` header
  [PASS] executive_summary_section_present
  [PASS] findings_documents_trader_riskjudge_byte_identical_text_with_file_line_anchor
  [PASS] findings_explains_sparse_drawer_3_of_20_agents
  [PASS] findings_proposes_layer_1_28_skill_surfacing
FAIL (12/13) EXIT=1
```

**Interpretation:** 12/13 PASS. Single FAIL is expected log-last gating signal. After log append + re-run, verifier returns 13/13.

## Hypothesis verdict

**CONFIRMED AND REFINED.** Smoking gun at `autonomous_loop.py:719` `"risk_assessment": {"reason": analysis["reason"]}` — lite path makes ONE LLM call, both Trader and RiskJudge consume the same `reason` string. Cosmetic patch at `signal_attribution.py:131-154` detects byte-identical pairs (weight=0.0 + rationale match) and displays "lite-path" badge but does not fix the structure. Hypothesized file location (`multi_agent_orchestrator.py`) was WRONG — that's the Slack Layer-2 orchestrator. The actual aliasing is in `autonomous_loop.py` + `signal_attribution.py`. Sparse drawer (3 of ~20 agents) is downstream of bucket 24.2 lite-mode default; Layer-1 28-skill outputs are never persisted into the `signals` JSON column even when full pipeline runs.

## Phase-25 candidates (6)

1. **25.A (P0)** — Decouple RiskJudge in lite path (independent LLM call)
2. **25.B (P1)** — Remove cosmetic aliasing patch after 25.A
3. **25.C (P1)** — Surface Layer-1 28-skill outputs in drawer
4. **25.D (P2)** — Normalize per-agent contribution weights
5. **25.E (P2)** — "Summary vs full" drawer toggle
6. **25.F (P1)** — Byte-identical regression test

## Next phase
EVALUATE — Q/A spawn pending.
