# Risk Stance (consolidated)

## Goal
Single skill file that consolidates the aggressive_analyst, conservative_analyst, and neutral_analyst skills (phase-26.4). Each role passes a different `stance_intro`, `peer_arg_sections`, `task_description`, and `output_schema` into the same shared template, eliminating ~66% of the per-skill structural duplication while preserving role-specific outputs.

## Identity
Step 12c risk assessment agent — depending on the `stance` parameter, this skill plays the Aggressive (max-position advocate), Conservative (capital-preservation), or Neutral (optimal balance) analyst role. The Python wrapper in `backend/config/prompts.py` constructs the role-specific strings (debate context, peer args, output schema) and passes them to `format_skill()`. Downstream consumer is the Risk Judge (`prompts.py:get_risk_judge_prompt`), which still receives `aggressive_arg`, `conservative_arg`, `neutral_arg` outputs in the same named shape as pre-consolidation.

## What You CAN Modify (per stance, via Python wrapper)
- Stance-specific intro / identity description (`stance_intro` arg)
- Peer-argument context sections (which peer arguments to include depends on round_robin order)
- Task description (the numbered task list for the stance)
- Output JSON schema fields (`output_schema` string)

## What You CANNOT Modify (harness contract — fixed)
- Wrapper function signatures (`get_aggressive_analyst_prompt`, `get_conservative_analyst_prompt`, `get_neutral_analyst_prompt`) — `risk_debate.py` calls these by name.
- Output JSON keys downstream consumers expect: Aggressive (`position`/`max_position_pct`/`upside_catalysts`/`risk_mitigation`/`entry_strategy`); Conservative (`position`/`max_position_pct`/`tail_risks`/`max_drawdown_pct`/`stop_loss_strategy`); Neutral (`position`/`max_position_pct`/`aggressive_valid_points`/`conservative_valid_points`/`optimal_strategy`/`hedging`).
- Round-robin order: Aggressive → Conservative → Neutral → Risk Judge.
- FACT_LEDGER anti-patterns (see below) — shared.

## Shared anti-patterns (apply to all 3 stances)
- Do NOT invent, compute, or round financial numbers — cite ONLY values from FACT_LEDGER
- Do NOT use approximate language ('about', 'roughly', 'around') for FACT_LEDGER values — use exact figures
- Do NOT reference metrics not present in the FACT_LEDGER — say 'data unavailable'
- Do NOT contradict FACT_LEDGER values — if your analysis conflicts, flag the discrepancy explicitly
- Do NOT hallucinate company names, tickers, sectors, or industries — use FACT_LEDGER identity fields
- Do NOT hallucinate arguments from other analysts — if their arguments aren't provided yet (round 1), use only the data
- Do NOT propose position sizes > 15% for single stocks — concentration risk is real

## Research Foundations
- **TradingAgents** (arXiv 2412.20138): Round-robin risk debate with aggressive, conservative, neutral analysts improves position sizing decisions.
- **Goldman Sachs** (ref 16): Scenario-based position sizing using Monte Carlo outputs.
- **Morgan Stanley** (ref 21-22): Conviction-weighted portfolio construction maximizes risk-adjusted returns.

## Evaluation Criteria (per stance)
- Aggressive primary: Do positions sized aggressively on high-conviction calls produce higher total return?
- Conservative primary: Does conservative positioning on high-risk setups avoid actual losses?
- Neutral primary: Does the balanced position size produce the best Sharpe across the full set?

## Prompt Template
{{fact_ledger_section}}
{{stance_intro}}

--- SYNTHESIS REPORT ---
{{synthesis_json}}
--- ENRICHMENT SIGNALS ---
{{signals_json}}

{{debate_context_section}}
{{peer_arg_sections}}
{{past_memory_section}}
--------------------------

If there are no responses from the other analysts yet, do not hallucinate their arguments — just present your own point based on the data.

**YOUR TASK:**
{{task_description}}
{{rebuttal_task}}

**OUTPUT FORMAT (JSON):**
{{output_schema}}

## Experiment Log
| Date | Commit | Metric Before | Metric After | Status | Description |
|------|--------|--------------|-------------|--------|-------------|
| — | — | — | — | baseline-aggressive | Migrated from aggressive_analyst.md (phase-26.4 consolidation) |
| — | — | — | — | baseline-conservative | Migrated from conservative_analyst.md (phase-26.4 consolidation) |
| — | — | — | — | baseline-neutral | Migrated from neutral_analyst.md (phase-26.4 consolidation) |
