# Researcher Agent Memory Index

- [PSR/DSR formulas researched](project_psr_dsr_formulas.md) — Bailey & Lopez de Prado PSR/DSR exact formulas, kurtosis convention, Python pseudocode, anti-patterns for paper_metrics_v2.py
- [Anthropic agent patterns — retry loops and research-on-demand](project_anthropic_agent_patterns.md) — Exact mechanics for evaluator→generator retry loop and lead-spawns-researcher pattern, mapped to run_harness.py variable names
- [Alpaca paper execution — phase-3.7.5 research](project_alpaca_paper_execution.md) — SDK choice, ops-toggle flag shape, shadow-mode drift threshold, paper gotchas (fractional/settlement/coverage), rollback pattern
- [phase-4.7.4 vitest + leaderboard conventions](project_474_vitest_leaderboard.md) — vitest Next.js 15 setup, --filter positional-arg fix, DSR/PBO colour thresholds, fake-timers polling pattern
- [Research gate discipline — >=5 sources floor and stale file locations](project_research_gate_discipline.md) — phase-4.16.1 raised floor to >=5; stale context/research-gate.md and mas-architecture.md identified
- [BigQuery dataset locations](project_bq_dataset_locations.md) — financial_reports lives in us-central1 (NOT US); outcome_tracking + agent_memories + paper_* tables live there; pyfinagent_data and pyfinagent_pms are US; timestamp col conventions per table
- [LLM cost/pricing tables inventory](project_cost_pricing_tables_inventory.md) — 3+ independent pricing tables (cost_tracker MODEL_PRICING canonical, settings_api display list, governance rough estimate); patch ALL on model add; Opus 4.8=$5/$25
- [Metric source paths + DESC-order trap](project_metric_source_paths.md) — THREE distinct Sharpe/maxDD paths (cockpit-backend, cockpit-TS, gate); get_paper_snapshots returns DESC -> re-sort to chronological or Sharpe sign flips + maxDD reads growth as crash (phase-47.4)
- [Strategy-rotation infra already exists](project_strategy_rotation_infra.md) — promoter/friday_promotion/promoted_strategies/load_promoted_params + optimizer categorical strategy all BUILT (phase-25/26/30); phase-47.6 only adds per-strategy DSR bake-off selector; per-strategy DSR NOT in existing artifacts (needs 5 $0-LLM backtests)
