# Sprint Contract — phase-24.3 — Autoresearch ↔ Daily-Loop Wiring

**Cycle:** phase-24 cycle 6
**Date:** 2026-05-12
**Step ID:** 24.3
**Priority:** P1

## Research-gate
`gate_passed: true` (tier=complex). 6 sources: built-multi-agent, harness-design, Snowflake champion-challenger MLOps, MLOps Feb 2026, DataRobot, arxiv 2512.02227 financial orchestration.

```json
{"tier":"complex","external_sources_read_in_full":6,"snippet_only_sources":8,"urls_collected":14,"recency_scan_performed":true,"internal_files_inspected":20,"gate_passed":true}
```

## Hypothesis
`meta_evolution/cron.py:87-125` runs Sunday-only. `backend/autoresearch/` produces proposals → gates → promotions weekly. Zero cross-module imports = daily cycle never sees evolution outputs.

**Researcher verdict: CONFIRMED with severe corollaries:**
- `autonomous_loop.py` has zero `autoresearch|meta_evolution` imports (grep verified)
- `meta_evolution/cron.py:87-152` weekly cron touches only YAML budget allocation — never reads promoted IDs
- `friday_promotion.py:121-131` writes promoted IDs to flat TSV `weekly_ledger.tsv` — no listener
- `monthly_champion_challenger.py:76` hard-codes `actual_replacement: False` — even HITL approval produces no downstream action
- `autoresearch/cron.py:29-38` passes `func=lambda: None` — nightly autoresearch cron is a stub, runs zero experiments
- `slot_accounting.py:26` writes to BQ `harness_learning_log` but UI-only (`harness_autoresearch.py` reads it; daily loop does not)
- `autonomous_loop.py:33-43` reads `optimizer_best.json` at startup, which is in `Proposer` whitelist — but no scheduled path writes that file

## Success criteria (verbatim)

1. findings_md_exists
2. research_gate_envelope_present_with_gate_passed_true
3. external_sources_count_at_least_5
4. canonical_url_cited_verbatim_anthropic_harness_design
5. recency_scan_2024_2026_section_present
6. at_least_three_phase_25_candidate_steps_proposed
7. each_candidate_step_has_files_list_with_absolute_paths
8. each_candidate_step_has_draft_verification_command
9. harness_log_has_phase_24_24_3_cycle_entry
10. executive_summary_section_present
11. findings_documents_meta_evolution_cron_decoupling
12. findings_documents_skill_optimizer_active_strategy_gap

**Verifier:** `python3 tests/verify_phase_24_3.py`

## Plan
1. Findings doc
2. experiment_results.md
3. Q/A
4. Cycle 47 log append
5. live_check_24.3.md
6. Flip 24.3 to done
