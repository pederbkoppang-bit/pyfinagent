# Sprint Contract — phase-24.2 — Pipeline Routing + Report Persistence

**Cycle:** phase-24 cycle 5
**Date:** 2026-05-12
**Step ID:** 24.2
**Priority:** P1
**Depends on:** 24.0 (charter — DONE)

## Research-gate summary
`gate_passed: true` (tier=complex). 5 sources: harness-design, built-multi-agent, building-effective-agents, Gemini structured output, arxiv hybrid cost-quality routing.

```json
{"tier":"complex","external_sources_read_in_full":5,"snippet_only_sources":10,"urls_collected":16,"recency_scan_performed":true,"internal_files_inspected":7,"gate_passed":true}
```

## Hypothesis

`autonomous_loop.py:564-615` branches on `settings.lite_mode`. Lite path skips `AnalysisOrchestrator`. `_persist_lite_analysis` is the only persistence path. `/reports` empty because only lite-path writes.

**Researcher verdict: CONFIRMED with correction.** Actual line: `autonomous_loop.py:575` `if settings.lite_mode:`. **The full pipeline path does NOT call `bq.save_report` either** — `orchestrator.py` contains zero `save_report` calls. So `/reports` is empty REGARDLESS of `lite_mode` setting from the paper-trading path. Only the manual `/api/analysis` endpoint (`analysis.py:201`) writes reports. Lite-path persistence is guarded by `_path == "lite"` at autonomous_loop.py:276/294 — full path returns no `_path` marker. Default of `settings.lite_mode = False` (`settings.py:119`) means full pipeline is currently the route in default config — but neither path persists from paper-trading flow.

Skills count: 31 active `.md` skills + 1 template + experiments subdir.
Cost ratio: lite ≈$0.01/ticker; full ≈$0.10-0.20/ticker (10-20x).

## Success criteria (verbatim from masterplan)

1. findings_md_exists_at_docs_audits_phase_24_2026_05_12_24_2_pipeline_routing_findings_md
2. research_gate_envelope_present_with_gate_passed_true
3. external_sources_count_at_least_5
4. canonical_url_cited_verbatim_autonomous_loop_py
5. recency_scan_2024_2026_section_present
6. at_least_three_phase_25_candidate_steps_proposed
7. each_candidate_step_has_files_list_with_absolute_paths
8. each_candidate_step_has_draft_verification_command
9. harness_log_has_phase_24_24_2_cycle_entry
10. executive_summary_section_present
11. findings_cites_autonomous_loop_branching_at_564_615
12. findings_documents_why_full_28_skill_pipeline_is_unreached
13. findings_traces_report_persistence_lite_vs_full_gap

**Verifier:** `source .venv/bin/activate && python3 tests/verify_phase_24_2.py`

## Plan
1. Write findings doc
2. experiment_results.md
3. Q/A spawn
4. harness_log append Cycle 46
5. live_check_24.2.md
6. Flip masterplan 24.2 to done

## References
External: 5 read-in-full above + arxiv 2404.14618v1.
Internal: `backend/services/autonomous_loop.py:273,276,294,575,595,722`; `backend/agents/orchestrator.py:444`; `backend/api/analysis.py:201`; `backend/api/reports.py`; `backend/db/bigquery_client.py`; `backend/config/settings.py:119`; `backend/agents/skills/*.md` (31 files).
