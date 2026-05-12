# Sprint Contract — phase-24.13 — Profit-Maximization Red-Line Synthesis

**Cycle:** phase-24 cycle 14
**Date:** 2026-05-12
**Step ID:** 24.13
**Priority:** P1
**Depends on:** 24.1-24.9 (all complete)

## Research-gate
`gate_passed: true` (tier=complex). 6 sources: Anthropic built-multi-agent, Markowitz MPT, arxiv SHARP 2026, ATLAS 2025, AI-Trader 2025, survey quantitative AI 2025.

```json
{"tier":"complex","external_sources_read_in_full":6,"snippet_only_sources":12,"urls_collected":18,"recency_scan_performed":true,"internal_files_inspected":14,"gate_passed":true}
```

## Hypothesis
Current behavior mis-aligned: stops orphan (anti-profit), full pipeline unused (missing alpha), autoresearch isolated (no switching), cost budget honor-system (anti-cost).

**Researcher verdict: CONFIRMED across 4 compounding misalignments:**
1. **Anti-profit**: TER -12.30% (~-$1,107 accruing) — stop orphan + None-bypass; 6 stop-less positions = 55% of portfolio
2. **Anti-cost**: `llm_client.py` no budget check; `cost_tracker.py:147` 1.25x vs 2.0x actual cache write (60% undercount); system prompt below 4096 cache threshold; full-pipeline runs without persistence (~$1.50/day waste)
3. **Anti-switching**: zero `autoresearch|meta_evolution` imports in `autonomous_loop.py`; `monthly_champion_challenger.py:76` hard-codes `actual_replacement: False`; `autoresearch/cron.py:29-38` lambda stub
4. **Unobservable**: `sovereign_api.py:394-395` hardcodes `anthropic: 0.0, vertex: 0.0, openai: 0.0`; BQ leaderboard view missing; no `profit_per_llm_dollar` metric

**Industry context**: arxiv 2503.21422 confirms NO published trading system has profit-per-LLM-dollar metric — pyfinagent would be first mover.

## Success criteria (verbatim)
1. findings_md_exists
2-10. common pack
11. synthesis_references_all_prior_buckets_24_1_through_24_9
12. synthesis_quantifies_cost_vs_pnl_ratio
13. synthesis_audits_strategy_switching_mechanism
14. synthesis_audits_cost_budget_enforcement_path

**Verifier:** `python3 tests/verify_phase_24_13.py`

## Plan
1. Findings
2. Results
3. Q/A
4. Cycle 55 log
5. live_check_24.13.md
6. Flip
