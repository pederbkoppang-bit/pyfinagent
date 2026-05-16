# Debate Stance (consolidated)

## Goal
Single skill file that consolidates the bull_agent, bear_agent, and devils_advocate_agent skills (phase-26.4). Each role passes a different `stance_intro`, `context_sections`, `task_description`, and `output_schema` into the same shared template, eliminating ~66% of the per-skill structural duplication while preserving role-specific outputs.

## Identity
Step 8 debate agent — depending on the `stance` parameter, this skill plays the Bull (round-based investment advocate), Bear (round-based risk skeptic), or Devil's Advocate (post-debate contrarian stress-tester) role. The Python wrapper in `backend/config/prompts.py` constructs the role-specific strings (rebuttal logic, output schema, task description) and passes them to `format_skill()`. Downstream consumer is the Moderator (`prompts.py:get_moderator_prompt`), which still receives `bull_case`, `bear_case`, and `devils_advocate` outputs in the same shape as pre-consolidation.

## What You CAN Modify (per stance, via Python wrapper)
- Stance-specific intro / identity description (`stance_intro` arg)
- Context sections (signals_json + traces, OR bull_case + bear_case, depending on stance)
- Task description (the numbered task list for the stance)
- Output JSON schema fields (`output_schema` string)

## What You CANNOT Modify (harness contract — fixed)
- Wrapper function signatures (`get_bull_agent_prompt`, `get_bear_agent_prompt`, `get_devils_advocate_prompt` in `prompts.py`) — downstream consumers (`debate.py`) call these by name.
- Output JSON keys that downstream consumers expect: `thesis`/`confidence`/`key_catalysts`/`evidence` (Bull); `thesis`/`confidence`/`key_threats`/`evidence` (Bear); `challenges`/`hidden_risks`/`bull_weakness`/`bear_weakness`/`groupthink_flag`/`confidence_adjustment`/`summary` (DA).
- Debate protocol: round-based with rebuttal structure for bull/bear; single-shot post-debate for DA.
- FACT_LEDGER anti-patterns (see below) — shared across all 3 roles.

## Shared anti-patterns (apply to all 3 stances)
- Do NOT invent, compute, or round financial numbers — cite ONLY values from FACT_LEDGER
- Do NOT use approximate language ('about', 'roughly', 'around') for FACT_LEDGER values — use exact figures
- Do NOT reference metrics not present in the FACT_LEDGER — say 'data unavailable'
- Do NOT contradict FACT_LEDGER values — if your analysis conflicts, flag the discrepancy explicitly
- Do NOT hallucinate company names, tickers, sectors, or industries — use FACT_LEDGER identity fields

## Research Foundations
- **TradingAgents** (arXiv 2412.20138): Multi-agent debate frameworks yield more robust trading decisions through adversarial structure. Bull/Bear/DA are 3 distinct adversaries.
- **OpenReview multi-agent debate** (Vusd1Hw2D9): Debate quality literature shows adversarial independence drives accuracy — preserve INDEPENDENCE across the 3 calls (separate context windows per stance).
- **Anthropic harness-design** (long-running apps): the 6-file proliferation encoded an assumption that the model needed entirely separate files. The near-identical structure of the originals (5 shared anti-patterns, identical Experiment Log, identical preamble) suggested that assumption was testable. Phase-26.4 tests it.

## Evaluation Criteria (per stance)
- Bull primary: Does a high-confidence bull thesis (>0.75) correlate with positive return_pct?
- Bear primary: Does a high-confidence bear thesis (>0.75) correlate with avoided losses on STRONG_SELL recommendations?
- DA primary: Does DA's `confidence_adjustment` correlate with subsequent reality (downward adjustments precede misses, upward precede beats)?

## Prompt Template
{{fact_ledger_section}}
{{stance_intro}}

{{context_sections}}

{{past_memory_section}}

**YOUR TASK:**
{{task_description}}

**OUTPUT FORMAT (JSON):**
{{output_schema}}

## Experiment Log
| Date | Commit | Metric Before | Metric After | Status | Description |
|------|--------|--------------|-------------|--------|-------------|
| — | — | — | — | baseline-bull | Migrated from bull_agent.md (phase-26.4 consolidation) |
| — | — | — | — | baseline-bear | Migrated from bear_agent.md (phase-26.4 consolidation) |
| — | — | — | — | baseline-da | Migrated from devils_advocate_agent.md (phase-26.4 consolidation) |
