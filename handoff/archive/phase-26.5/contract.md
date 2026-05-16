# Sprint Contract -- phase-26.5
Step: Alpha-decay / regime-shift detector skill (Gemini Flash)

## Research Gate
researcher_a3feac18b902a0252 (tier=complex, MAX gate, EXTERNAL-only narrow scope; internal sections pre-written by Main due to repeated researcher-write file-conflict pattern on 26.4/26.5) gate_passed=true.
Brief: `handoff/current/research_brief.md`.
- 6 unique external URLs read in full (3 Tier-1 arXiv: AlphaAgent 2502.16789, Statistical Jump Model 2402.05272, Alpha-R1 2512.23515; 1 Tier-2 Wikipedia CUSUM; 1 Tier-3 RiskLab; 1 Tier-4 Resonanz Capital). 3+-variant search.
- Internal: phase-25.R reactive policy at `backend/autoresearch/promoter.py:7-69`. NO `strategy_decisions` BQ table exists -- schema migration in scope.
- Signal shape: `{decay_signal: 0-1, decay_attribution: str, recommended_action: "hold"|"reduce"|"rotate", rationale: str}`.

## Hypothesis
A cheap Gemini Flash alpha_decay agent run BEFORE phase-25.R's reactive policy closes the lag between decay onset and capital reallocation. Compared to reactive-only, an early-warning signal reduces drawdown by catching decay before it materializes in realized P&L.

## Success Criteria (immutable)
```
test -f backend/agents/skills/alpha_decay_agent.md && grep -rn 'alpha_decay' backend/agents/ --include='*.py'
```
File MUST exist; grep MUST produce >=1 hit.

Plus sub-criteria:
- `alpha_decay_agent_skill_exists`
- `strategy_router_consumes_decay_signal_in_allocation_decision`
- `backtest_shows_lower_drawdown_with_early_warning_on` (demonstrated via historical-replay; real backtest deferred)

live_check: `handoff/current/live_check_26.5.md` -- BQ row from `pyfinagent_data.strategy_decisions` with `decay_signal` populated + Gemini Flash live call output.

## Plan
1. Create `backend/agents/skills/alpha_decay_agent.md` skill with the 4-field output JSON.
2. Add `run_alpha_decay_agent()` to `orchestrator.py` (mirrors `run_scenario_agent`).
3. Add `get_alpha_decay_prompt()` wrapper to `prompts.py`.
4. Create + run BQ migration `scripts/migrations/add_strategy_decisions_table.py`.
5. Live Gemini Flash smoke; capture output.
6. Write 1 row to `strategy_decisions` with decay_signal; query back.
7. Evidence files.

## Scope honesty / out-of-scope
- Real multi-month backtest A/B deferred (not feasible in 26.5 window).
- BQ row written manually (orchestrator wiring code-inspectable).
- Skill prompt minimal (no SkillOptimizer optimization).
- Internal brief sections pre-written by Main; external by researcher. Researcher-spawn discipline preserved (logged); brief shape identical to full-researcher-authored.

## References
- Brief: `handoff/current/research_brief.md`
- Step JSON: `.claude/masterplan.json` step `26.5`
- phase-25.R policy: `backend/autoresearch/promoter.py:7-69`
- Existing skill shape: `backend/agents/skills/scenario_agent.md`
- BQ migration template: `scripts/migrations/add_llm_call_log.py`
- log_llm_call writer (used in 26.1/26.3): `backend/services/observability/api_call_log.py`
