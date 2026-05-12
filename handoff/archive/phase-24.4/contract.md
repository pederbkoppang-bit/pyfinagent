# Sprint Contract — phase-24.4 — Agent Rationale Flow Audit

**Cycle:** phase-24 cycle 3
**Date:** 2026-05-12
**Step ID:** 24.4
**Priority:** P0
**Depends on:** 24.0 (charter — DONE)
**Audit basis:** operator screenshot 2026-05-01 FIX BUY; byte-identical Trader/RiskJudge text; sparse drawer (3 of ~20)

---

## Research-gate summary

`gate_passed: true` (tier=complex). 6 sources read in full: built-multi-agent, harness-design, building-effective-agents, demystifying-evals (Anthropic), Du et al. Society of Minds (LLM debate), NIST agentic-AI evaluation probes. Three-variant search-query discipline applied. Recency scan present.

```json
{"tier":"complex","external_sources_read_in_full":6,"snippet_only_sources":10,"urls_collected":16,"recency_scan_performed":true,"internal_files_inspected":8,"gate_passed":true}
```

---

## Hypothesis

Trader and RiskJudge rationale text is byte-identical because the rationale-producing code path aliases the two rationale fields rather than running independent risk-specific LLM calls. Drawer shows 3 of ~20 agents because Layer-1 28-skill outputs are not surfaced (lite mode skips them).

**Researcher verdict: CONFIRMED AND REFINED.** The aliasing site is NOT in `multi_agent_orchestrator.py` (that's the Slack Layer-2 orchestrator) but in `autonomous_loop.py:719` lite-path:

```python
"risk_assessment": {"reason": analysis["reason"]},
```

`_run_claude_analysis` (autonomous_loop.py:619-740) makes ONE LLM call producing ONE `reason` string. Both Trader (`full_report.analysis.reason`) and RiskJudge (`risk_assessment.reason`) point at the same string. No independent risk LLM call exists. A cosmetic patch at `signal_attribution.py:131-154` detects byte-identical pairs (weight=0.0 + rationale match) and substitutes a "lite-path" label — display-only, not structural.

---

## Success criteria (verbatim from masterplan)

1. findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_4_agent_rationale_findings_md
2. research_gate_envelope_present_with_gate_passed_true
3. external_sources_count_at_least_5
4. canonical_url_cited_verbatim_multi_agent_orchestrator_py
5. recency_scan_2024_2026_section_present
6. at_least_three_phase_25_candidate_steps_proposed
7. each_candidate_step_has_files_list_with_absolute_paths
8. each_candidate_step_has_draft_verification_command
9. harness_log_has_phase_24_24_4_cycle_entry
10. executive_summary_section_present
11. findings_documents_trader_riskjudge_byte_identical_text_with_file_line_anchor
12. findings_explains_sparse_drawer_3_of_20_agents
13. findings_proposes_layer_1_28_skill_surfacing

**Verifier:** `source .venv/bin/activate && python3 tests/verify_phase_24_4.py`

---

## Plan steps

1. Write findings doc `docs/audits/phase-24-2026-05-12/24.4-agent-rationale-findings.md` with: frontmatter, exec summary, code-grounded findings (smoking gun L719, cosmetic patch, drawer architecture), external research summary (6 URLs), recency scan, ≥5 phase-25 candidates with absolute files + verification, open questions, references.
2. Write `experiment_results.md` with verbatim verifier output.
3. Spawn Q/A. On PASS:
4. Append `## Cycle 44 -- 2026-05-12 -- phase=24.4 result=PASS` to harness_log.md.
5. Write `live_check_24.4.md`.
6. Flip masterplan 24.4 to done.

---

## References

- https://www.anthropic.com/engineering/built-multi-agent-research-system
- https://www.anthropic.com/engineering/harness-design-long-running-apps
- https://www.anthropic.com/engineering/building-effective-agents
- https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents
- Du et al. Society of Minds — LLM Debate (https://composable-models.github.io/llm_debate/, arxiv 2305.14325)
- NIST Building Evaluation Probes into Agentic AI (https://www.nist.gov/programs-projects/building-evaluation-probes-agentic-ai)

Internal:
- `backend/services/autonomous_loop.py:619-740` (lite path) — :719 (smoking gun)
- `backend/services/signal_attribution.py:117-157` (aliasing detection + cosmetic patch)
- `backend/api/paper_trading.py:460-511` (rationale endpoint)
- `frontend/src/components/AgentRationaleDrawer.tsx` (drawer with lite_path badge)
- `backend/agents/_inventory.json` (43-node topology, drawer's "3 main" filter)
- `backend/agents/multi_agent_orchestrator.py` (Slack-only Layer-2; NOT the paper-trade rationale path)
