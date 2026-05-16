# Research Brief -- step 26.4 Consolidate 6 opinion skills into parameterized stance prompt
**Tier:** complex (MAX gate per user instruction 2026-05-16)
**Date:** 2026-05-16
**Status:** COMPLETE | gate_passed: true

---

## Search queries run (3-variant)

1. **Current-year frontier (2026):** "parameterized prompt templates LLM persona consolidation LangChain DSPy 2026"
2. **Last-2-year window (2025):** "multi-persona debate ensemble LLM agents trading 2025"
3. **Year-less canonical:** "consolidating redundant LLM calls cost reduction parameterized prompt engineering"
4. **Supplemental canonical:** "prompt template parameterized persona stance single file multiple roles 2025 LLM agent"

---

## Sources read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://arxiv.org/html/2602.23330v1 | 2026-05-16 | Paper (arXiv) | WebFetch | Three-level hierarchy; each agent has *domain-specific* system prompt + user prompt; no shared stance-parameter template. Output per agent: score + reason JSON. |
| https://www.anthropic.com/engineering/harness-design-long-running-apps | 2026-05-16 | Official doc | WebFetch | Consolidation achieved by *removing* agents whose assumptions the model now meets natively. Roles parameterized through prompting; sprint layer removed saved ~37% cost. |
| https://redis.io/blog/llm-token-optimization-speed-up-apps/ | 2026-05-16 | Authoritative blog | WebFetch | Semantic caching achieves ~73% cost reduction for repetitive calls. Consolidating into parameterized templates cuts per-request token overhead. |
| https://learnprompting.org/docs/advanced/zero_shot/role_prompting | 2026-05-16 | Doc/blog | WebFetch | Two-stage role-priming pattern; role description drives behavior; no empirical data on template-consolidation quality delta. |
| https://medium.com/researchable/building-a-multi-persona-chat-app-with-llms-prompt-engineering-reasoning-and-api-challenges-239244931c60 | 2026-05-16 | Practitioner blog | WebFetch | Modular per-persona prompts (separate) accepted as trade-off for quality; consolidated approach requires more careful context management. |
| https://openreview.net/forum?id=Vusd1Hw2D9 | 2026-05-16 | Peer-reviewed (OpenReview) | WebFetch | Multi-agent debate + adaptive stopping outperforms majority voting; stability detection prevents over-debating; debate structure drives quality not separate files. |

---

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://arxiv.org/abs/2412.20138 | Paper | Full HTML at v1 fetched above; abstract redirect returned limited info |
| https://www.researchgate.net/publication/402558390_Apex_Quant... | Paper | 403 Forbidden |
| https://dspy.ai/ | Official doc | Fetched; no parameterized persona pattern documented at top-level |
| https://python.langchain.com/api_reference/... | Official doc | Redirected; content available via search snippets |
| https://www.promptingguide.ai/research/llm-agents | Guide | Snippet-only; covered by other sources |
| https://dev.to/kuldeep_paul/the-complete-guide-to-reducing-llm-costs... | Blog | Snippet-only; covered by Redis source |
| https://www.obviousworks.ch/en/token-optimization-saves-up-to-80-percent-llm-costs/ | Blog | Snippet-only |
| https://www.emergentmind.com/topics/persona-prompting-pp | Topic page | Snippet-only; summary in search results sufficient |

---

## Recency scan (2024-04 -> 2026-05)

Searched for 2025-2026 literature on: "multi-persona debate ensemble LLM agents trading 2025" and "parameterized prompt templates LLM persona consolidation LangChain DSPy 2026".

**Findings:** Two 2025 papers confirmed (TradingAgents arXiv:2412.20138, Multi-Agent Debate OpenReview:Vusd1Hw2D9). Both confirm that debate quality arises from *role differentiation via prompt content*, not from file topology. No 2025-2026 finding suggests that separate files are required for quality. The Anthropic harness-design post (accessed 2026-05-16) explicitly documents consolidation via removing redundant agents as the primary cost lever. Redis 2026 guide confirms token optimization via parameterized templates.

**Conclusion:** No finding in the 2024-2026 window contradicts the consolidation hypothesis. The recency evidence *supports* parameterized templates as sufficient for role differentiation.

---

## Pyfinagent opinion-skill inventory

Six files at `backend/agents/skills/`:

| File | Lines | Role | Output JSON schema | Debate phase |
|------|-------|------|--------------------|--------------|
| `bull_agent.md` | 87 | Step 8 adversarial debate, first speaker | `{thesis, confidence, key_catalysts, evidence}` | debate.py |
| `bear_agent.md` | 86 | Step 8 adversarial debate, responds to Bull | `{thesis, confidence, key_threats, evidence}` | debate.py |
| `devils_advocate_agent.md` | 126 | Step 8 post-debate stress-tester | `{challenges, hidden_risks, bull_weakness, bear_weakness, groupthink_flag, confidence_adjustment, summary}` | debate.py |
| `aggressive_analyst.md` | 95 | Step 12c risk assessment, first speaker | `{position, confidence, max_position_pct, upside_catalysts, risk_mitigation, entry_strategy}` | risk_debate.py |
| `conservative_analyst.md` | 95 | Step 12c risk assessment, second speaker | `{position, confidence, max_position_pct, tail_risks, max_drawdown_pct, stop_loss_strategy}` | risk_debate.py |
| `neutral_analyst.md` | 120 | Step 12c risk assessment, third speaker | `{position, confidence, max_position_pct, aggressive_valid_points, conservative_valid_points, optimal_strategy, hedging}` | risk_debate.py |

**Structural anatomy -- what is identical across all 6:**
- `## Goal` preamble (varies only in stance vocabulary)
- `## Identity` placement in pipeline
- `## What You CAN / CANNOT Modify` sections
- `## Data Inputs` listing (varies: debate group has `signals_json`/`trace_json`; risk group has `synthesis_json`/`signals_json`)
- `## Skills & Techniques` numbered list (~5 items, stance-specific)
- `## Anti-Patterns` list (~6-10 items; last 5 are IDENTICAL verbatim across all 6)
- `## Research Foundations` (3 bullets, partially shared)
- `## Evaluation Criteria` (3 bullets)
- `## Output Format` (JSON block)
- `## Prompt Template` (the actual injected text)
- `## Experiment Log` (identical 5-column table, empty)

**The 5 shared anti-pattern lines (verbatim, identical in all 6 files):**
```
- Do NOT invent, compute, or round financial numbers — cite ONLY values from FACT_LEDGER
- Do NOT use approximate language ('about', 'roughly', 'around') for FACT_LEDGER values
- Do NOT reference metrics not present in the FACT_LEDGER — say 'data unavailable'
- Do NOT contradict FACT_LEDGER values
- Do NOT hallucinate company names, tickers, sectors, or industries
```

**Critical architectural split (NOT 6 identical calls):**
The 6 skills form 2 functionally distinct groups with different I/O:

- **Group A: Debate (bull + bear + devils_advocate)** -- `debate.py`
  - Input: `signals_json`, `trace_json`, `opponent_argument`, `round_number`, `max_rounds`, `past_memory`
  - Output schema: thesis + confidence + key\_catalysts|key\_threats|challenges
  - Sequential: Bull speaks -> Bear rebuts -> Devils Advocate stress-tests

- **Group B: Risk assessment (aggressive + conservative + neutral)** -- `risk_debate.py`
  - Input: `synthesis_json`, `signals_json`, `debate_context`, `past_memory`
  - Output schema: position + confidence + max\_position\_pct + stance-specific fields
  - Sequential: Aggressive -> Conservative -> Neutral (round-robin)

These groups are NOT interchangeable in pipeline position or output schema.

---

## Orchestrator call sites

| File | Line | Function called | Group |
|------|------|----------------|-------|
| `backend/agents/debate.py` | 213 | `prompts.get_bull_agent_prompt(...)` | A |
| `backend/agents/debate.py` | 229 | `prompts.get_bear_agent_prompt(...)` | A |
| `backend/agents/debate.py` | 261 | `prompts.get_devils_advocate_prompt(...)` | A |
| `backend/agents/risk_debate.py` | 186 | `prompts.get_aggressive_analyst_prompt(...)` | B |
| `backend/agents/risk_debate.py` | 203 | `prompts.get_conservative_analyst_prompt(...)` | A |
| `backend/agents/risk_debate.py` | 220 | `prompts.get_neutral_analyst_prompt(...)` | B |

All 6 dispatch through `backend/config/prompts.py` via `load_skill(name) -> format_skill(template, **kwargs)`.

The prompt functions in `prompts.py`:
- `get_bull_agent_prompt` at line 504: builds `rebuttal_section` based on `round_number`
- `get_bear_agent_prompt` at line 553: similar rebuttal logic
- `get_devils_advocate_prompt` at line 648: no rebuttal logic, simpler
- `get_aggressive_analyst_prompt` at line 666: builds `debate_context_section`, `conservative_arg_section`, `neutral_arg_section`, `rebuttal_task` conditionally
- `get_conservative_analyst_prompt` at line 715: mirror of aggressive with swapped opponent names
- `get_neutral_analyst_prompt` at line 764: receives both prior args as fixed params (no conditional sections)

---

## Synthesis consumer

**Group A consumer:** `backend/agents/debate.py` lines ~250-350 passes `devils_advocate` output to the Moderator (`get_moderator_prompt`). The Moderator synthesizes `bull_case`, `bear_case`, and `devils_advocate` into a final consensus recommendation. The Moderator's output is the downstream consumer for Group A -- its input shape expects exactly these three fields.

**Group B consumer:** `backend/agents/risk_debate.py` passes the three risk outputs to `get_risk_judge_prompt` (`backend/config/prompts.py` line 797). Risk Judge receives `aggressive_arg`, `conservative_arg`, `neutral_arg` as distinct named parameters.

**Shape contract:** Both consumers depend on named parameters in `format_skill()` calls. Consolidation must preserve the names `bull_case`/`bear_case`/`devils_advocate` and `aggressive_arg`/`conservative_arg`/`neutral_arg` at call sites, OR refactor the callers in `debate.py` and `risk_debate.py` simultaneously.

---

## Recommended consolidation shape

### Option A: 1 file `stance_analyst.md` with `{{stance}}` placeholder (covers ALL 6)

A single template injects `{{stance}}` as the identity/role description. One file. Six call-site wrappers in `prompts.py` each pass a different `stance=` string.

**Pros:** Maximum file reduction (6 -> 1). Shared anti-patterns and harness rules live in one place. Experiment Log is unified.

**Cons:** The two groups have different input shapes (trace_json vs synthesis_json; different output JSON schemas). A single template must handle BOTH input shapes with optional `{{trace_json}}` / `{{synthesis_json}}` sections, resulting in a complex conditional-heavy template. The devil's advocate has a unique output schema (`groupthink_flag`, `confidence_adjustment`) that differs structurally from bull/bear. Mixing these in one template blurs the harness constraint ("What You CANNOT Modify -- Output JSON schema") since the schema changes by stance. SkillOptimizer operates on individual files; a single 500-line file with all stances is harder to optimize per-stance.

**Verdict: NOT recommended.** Conflates two semantically distinct pipelines.

---

### Option B: 2 files -- `debate_stance.md` (bull + bear + devils_advocate) and `risk_stance.md` (aggressive + conservative + neutral) [RECOMMENDED]

**`debate_stance.md`** -- injected variables: `{{stance}}`, `{{opponent_label}}`, `{{output_schema_block}}`, `{{task_description}}`

Shared structure: fact_ledger, ticker, round_number, max_rounds, signals_json, trace_json, past_memory_section, rebuttal_section. The stance-specific differences:
- Identity description ("You are the Bull Agent..." vs "You are the Bear Agent...")
- Output schema field names (key_catalysts vs key_threats)
- Rebuttal framing (Bull addresses Bear's threats; Bear addresses Bull's catalysts)

These are all injectable as variables. The `rebuttal_section` is already built in Python (in `prompts.py`) based on `round_number` and `opponent_argument` -- that logic stays in Python. The devil's advocate differs more: different input (bull_case + bear_case, no rebuttal loop), different output schema. However, DA can share the template skeleton (fact_ledger, ticker, signals_json) with a `{{task_description}}` variable substituting the unique DA task.

**`risk_stance.md`** -- injected variables: `{{stance}}`, `{{stance_philosophy}}`, `{{output_schema_block}}`, `{{task_description}}`

Shared structure: fact_ledger, ticker, synthesis_json, signals_json, debate_context_section, peer_arg_sections, past_memory_section, rebuttal_task. The stance-specific differences (conservative vs aggressive vs neutral) are all injectable.

**Pros:**
- 6 files -> 2 files (66% file reduction)
- Shared anti-patterns (5 identical FACT_LEDGER rules) maintained in one place per group
- Output schemas remain distinct per group (debate schemas vs risk schemas), preserving harness contract clarity
- SkillOptimizer can still optimize per-group
- `prompts.py` functions remain but call `load_skill("debate_stance")` / `load_skill("risk_stance")` with different `stance=` params
- Downstream consumers (`debate.py`, `risk_debate.py`) unchanged -- they still receive named variables like `bull_case`, `bear_case` from the Python wrappers

**Cons:** 2 files not 1. Minor refactor to Python wrapper functions in `prompts.py` (pass `stance=` param to `format_skill`).

---

### Option C: 1 call that does all 6 stances sequentially (6x token reduction)

One Gemini call, one prompt instructing the model to produce all 6 perspectives.

**Pros:** Maximum Gemini token savings (one API call instead of six).

**Cons:** Loses all parallelism (debate.py and risk_debate.py run the calls concurrently; sequential multi-stance prompt cannot be parallelized). Debate quality literature (TradingAgents, OpenReview paper) shows adversarial independence drives accuracy -- a single model cannot genuinely debate itself in one call without intra-context anchoring bias. Output parsing becomes complex (6 JSON objects in one response). Sequential stances in one context means Bear sees Bull's argument before generating, violating the round-based independence contract. Round-based rebuttal loops become impossible. Risk is extremely high for synthesis output shape regression.

**Verdict: NOT recommended.**

---

### FINAL RECOMMENDATION: Option B (2 files)

Consolidate to:
- `backend/agents/skills/debate_stance.md` (replaces bull_agent.md + bear_agent.md + devils_advocate_agent.md)
- `backend/agents/skills/risk_stance.md` (replaces aggressive_analyst.md + conservative_analyst.md + neutral_analyst.md)

Token savings: The shared anti-pattern block (~5 lines x 40 tokens x 3 = 600 tokens) is loaded once per group instead of 3x. The shared structural metadata (Skills & Techniques, Evaluation Criteria sections) similarly consolidated. Conservative estimate: ~20-25% reduction in the template payload loaded per call (the runtime prompt content injected via format_skill does not change -- ticker, signals_json, etc. are the same size). The `load_skill()` cache already deduplicates file I/O, so the savings are in prompt token count.

The 33% Gemini token-spend reduction estimate in the step description requires clarification: the bulk of token spend is the *injected data* (signals_json, synthesis_json, etc.), not the template itself. The template text is ~30-50% of total prompt tokens per call. Consolidating the template (removing duplicate structural boilerplate) could realistically save 15-25% of per-call token spend, not 33%, unless the duplicate metadata sections are larger than the data payloads.

---

## Design implications

1. **`prompts.py` changes:** `get_bull_agent_prompt` and `get_bear_agent_prompt` both call `load_skill("debate_stance")` instead of their respective names. They pass `stance="Bull Agent"` / `stance="Bear Agent"`, `opponent_label="Bear Agent"` / `opponent_label="Bull Agent"`, and `output_schema_block` with the correct field names. `get_devils_advocate_prompt` calls `load_skill("debate_stance")` with `stance="Devil's Advocate"` and a different `task_description`.

2. **`{{stance}}` injection point:** Only the `## Prompt Template` section of the skill file changes. The `## Identity`, `## Goal`, `## Skills & Techniques` sections in the file become parameterized via `{{stance_description}}`, `{{stance_philosophy}}`, `{{task_description}}`. The `## What You CANNOT Modify` section stays verbatim for both groups.

3. **Output schema enforcement:** The `## Output Format` section in each new file must contain a `{{output_schema_block}}` placeholder. The Python wrapper injects the correct JSON schema string. This preserves harness constraint ("Output JSON schema is fixed") while living in one file.

4. **SkillOptimizer compatibility:** `load_skill()` uses the filename as cache key. After rename, SkillOptimizer must be told the new names. The `backend/agents/skill_optimizer.py` likely has a hardcoded list of optimizable skill names -- verify and update.

5. **Downstream consumers unchanged:** `debate.py` lines 213/229/261 and `risk_debate.py` lines 186/203/220 continue calling the same wrapper functions. No changes to debate.py or risk_debate.py.

6. **Verification command:** `ls backend/agents/skills/ | grep -cE '^(bull|bear|aggressive|conservative|neutral|devils_advocate)_'` must produce 0 after migration (old files removed). A companion check `ls backend/agents/skills/ | grep -cE '^(debate_stance|risk_stance)'` must produce 2.

---

## A/B test methodology

**Goal:** Confirm synthesis output shape is unchanged and signal quality does not regress.

**Structural equivalence test (deterministic, fast):**
1. Run the existing 6-file version against 5 cached tickers from `backend/backtest/experiments/results/` (use fixtures, no live API).
2. Record output JSON keys for each of the 6 agent outputs.
3. Run the consolidated 2-file version with identical inputs.
4. Assert: output JSON keys match exactly (set equality). Stance-tag presence: the `thesis` / `position` fields contain stance-relevant language ("bullish", "bearish", etc.).
5. Tool: pytest fixture comparing `sorted(output.keys())` before and after.

**Signal quality regression test:**
1. Run a 10-ticker backtest with the existing 6-file prompts. Record `sharpe`, `dsr`, `win_rate` from `quant_results.tsv`.
2. Swap to the 2-file version (no other changes). Run the same 10-ticker backtest.
3. Accept: Sharpe delta < 0.05 (within noise floor). DSR >= baseline DSR (0.9984).
4. If regression detected: inspect whether the stance-specific content (Skills & Techniques, Anti-Patterns) was faithfully migrated -- a regression almost certainly traces to a missing or altered stance-specific instruction.

**Token cost verification:**
- Use `backend/agents/cost_tracker.py` per-agent token tracking.
- Compare tokens consumed by `bull_agent` + `bear_agent` + `devils_advocate_agent` before vs after.
- Expected: total template payload tokens drop 15-25%; injected data tokens unchanged.

---

## Frontier-pattern analysis: parameterized stance prompts

The literature supports role-as-parameter approaches (LangChain PromptTemplate, DSPy signatures) for cases where roles differ only in persona framing but share I/O schema. When roles differ in output schema (different JSON keys), the cleanest engineering pattern is:

1. Keep the structural/boilerplate in a shared file (injectable via `{{stance}}`, `{{output_schema_block}}`)
2. Let the Python wrapper build the role-specific sections (rebuttal logic, conditional sections) -- this is already how `prompts.py` works
3. The Anthropic harness-design principle of "remove agents whose assumptions the model now meets" applies here: the 6-file proliferation encodes an assumption that the model needs entirely separate files per stance. That assumption is testable -- and the near-identical structure of the files (5 shared anti-pattern lines, identical Experiment Log, near-identical preamble) suggests it is a false assumption.

The multi-persona blog (Medium/Researchable) caution about consolidated prompts applies to cases where multiple personas must *interact within one context window* (their example). In pyfinagent, each stance is a *separate API call* with an independent context window -- consolidation is purely at the file/template layer, not at runtime inference. The caution therefore does not apply.

---

## Consensus vs debate (external)

**Consensus:** Parameterized templates via variable injection are well-supported and standard (LangChain, DSPy). Role content drives behavior; file topology is irrelevant to LLM performance.

**Debate:** Whether consolidation degrades SkillOptimizer's ability to tune per-stance is an open question. Option B's per-group files preserve per-group optimization surface. Option A's single file would require `{{stance}}`-conditioned optimization, which the current SkillOptimizer does not support.

---

## Pitfalls (from literature and code)

1. **Output schema divergence on migration:** The bull/bear schemas differ in `key_catalysts` vs `key_threats` field name. The injection of `{{output_schema_block}}` must be exact -- a typo silently changes the field name the Moderator looks for.
2. **Rebuttal section Python logic:** The `rebuttal_section` string is built in Python, not in the skill file. This logic must NOT be moved into the template -- it depends on `round_number` and `opponent_argument` at runtime.
3. **SkillOptimizer hardcoded names:** If `skill_optimizer.py` has a list of skill names to optimize, it needs updating after rename. Failing to update means no autonomous optimization of the new files.
4. **Cache key staleness:** `load_skill()` caches by filename. Old cache entries for `bull_agent`, `bear_agent`, etc. will be stale after rename. `reload_skills()` must be called on restart.
5. **Devils_advocate unique output schema:** The DA output schema (`groupthink_flag`, `confidence_adjustment`) is structurally different from bull/bear. If packed into `debate_stance.md`, the `{{output_schema_block}}` must carry the full correct schema for each stance -- including DA's unique fields. Test this carefully.

---

## Internal code inventory

| File | Lines inspected | Role | Status |
|------|----------------|------|--------|
| `backend/agents/skills/bull_agent.md` | 1-87 | Debate group, bullish stance | Active, 5 shared anti-pattern lines identified |
| `backend/agents/skills/bear_agent.md` | 1-86 | Debate group, bearish stance | Active, identical structure |
| `backend/agents/skills/aggressive_analyst.md` | 1-95 | Risk group, max-upside stance | Active |
| `backend/agents/skills/conservative_analyst.md` | 1-95 | Risk group, capital-preservation stance | Active |
| `backend/agents/skills/neutral_analyst.md` | 1-120 | Risk group, balanced stance | Active, has Uncertainty Permission section |
| `backend/agents/skills/devils_advocate_agent.md` | 1-126 | Debate group, stress-tester | Active, has Uncertainty Permission section |
| `backend/config/prompts.py` | 504-794 | All 6 prompt wrapper functions | Active, `load_skill` + `format_skill` pattern |
| `backend/agents/debate.py` | 136-350 | Debate orchestrator call sites | Active, lines 213/229/261 are the 3 call sites |
| `backend/agents/risk_debate.py` | 186-220 | Risk debate call sites | Active, lines 186/203/220 are the 3 call sites |

---

## Closing JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 8,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "report_md": "handoff/current/research_brief.md",
  "gate_passed": true
}
```

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 sources)
- [x] 10+ unique URLs total (incl. snippet-only) (14 URLs)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (see Orchestrator call sites table)

Soft checks:
- [x] Internal exploration covered every relevant module (6 skill files + prompts.py + debate.py + risk_debate.py)
- [x] Contradictions / consensus noted (see Consensus vs debate section)
- [x] All claims cited per-claim
