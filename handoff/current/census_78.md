# Census 78.0 — Anthropic call-site routing decision table

Generated 2026-07-25 | **28 roles** | max_rail_cli=19, max_rail_proxy=1, stay_metered=8

Machine-readable twin: `handoff/current/census_78.json` (both files are generated from ONE source of truth, so they cannot drift).

## How to read the volume column

Volumes are a VERBATIM 30d GROUP BY on pyfinagent_data.llm_call_log run 2026-07-25. 'UNMEASURABLE' means the call site writes NO llm_call_log row -- absence of rows there is not evidence of absence of calls. IMPORTANT: because the wrapper clients log ok=True only, a zero row count is evidence of 'no SUCCESSFUL calls', never of 'no calls'. No row in this census may be read as proving a code path was dormant.

## Instrumentation finding (measured this cycle — NOT in the research brief)

11 of the 12 raw-SDK Anthropic call sites write no llm_call_log row at all: A5, A6, D1, D2, D3, D4, F1, F2, F3, G1, H1 (only A4 advisor_call is instrumented). DERIVED from this file's own `instrumented` field over the raw-SDK denominator ['A4', 'A5', 'A6', 'D1', 'D2', 'D3', 'D4', 'F1', 'F2', 'F3', 'G1', 'H1'], not asserted -- an earlier revision of this sentence said '9 of 12', which did not reproduce. Instrumentation follows the CC RAIL (claude_code_client.py:498) and the wrapper clients (ClaudeClient llm_client.py:1887, GeminiClient :1123, OpenAIClient :1275, advisor_call :2321), NOT the raw-SDK metered path. Consequence: any spend metric built on llm_call_log (75.5.1 fetch_llm_spend / the $25/day breaker) is structurally blind to those sites. SECOND, DISTINCT HOLE (found by the 78.0 Q/A): even the INSTRUMENTED wrapper clients log successes only -- ClaudeClient hardcodes ok=True and has no ok=False writer -- so a wrapper-client site that runs and fails leaves NO row either. The CC rail does log failures (claude_code_client.py:607, autonomous_loop.py:2469/:2545), which is precisely why the rail shows 1,547 failures and the wrapper clients show none. Both holes are owned by 78.8.

## Decision table

### A. Layer-1 analysis pipeline

| # | Role | Anchor (re-derived) | Model today | Structured output | 30d volume | Instrumented | Decision | Owner step |
|---|------|--------------------|-------------|-------------------|-----------|--------------|----------|------------|
| A1 | Layer-1 enrichment/debate tier | `backend/agents/orchestrator.py:652 make_client` | claude-sonnet-4-6 | pydantic-schema (9 pipeline schemas) | 2,192 calls / 4,370,458 tok agent='cc_rail' model=claude-sonnet-4-6 (measured 30d llm_call_log); 1,547 of those FAILED (ok=false) | yes | **max_rail_cli** | 78.7 (sonnet-5 swap only) |
| A2 | Layer-1 quant-executor tier | `backend/agents/orchestrator.py:659 make_client` | claude-sonnet-4-6 | pydantic-schema | included in the A1 cc_rail aggregate (same agent tag) | yes | **max_rail_cli** | 78.7 |
| A3 | deep_think / synthesis clients | `backend/agents/orchestrator.py:653-654 make_client` | GEMINI_DEEP_THINK (latent claude only under operator override) | n/a | 0 anthropic rows; Gemini default carries 226 calls / 232,090 tok | yes | stay_metered | none (watch item) |
| A4 | advisor_call synthesis | `backend/agents/llm_client.py:2191 (beta.messages.create :2273)` | executor claude-sonnet-4-6 + advisor claude-opus-4-8 | tool-use (advisor_20260301 beta) | 0 rows in 30d -- DARK (enable_advisor_tool=False, settings.py:391) | yes | stay_metered | none (justified) |
| A5 | BatchClient enrichment batch | `backend/agents/llm_client.py:1931 (batches.create :1978), caller orchestrator.py:1043` | per-request model param | n/a (passthrough) | 0 rows -- DOUBLY DARK (backtest_batch_mode has zero consumers; _run_enrichment_batch has no caller) | **NO** | stay_metered | 78.5 fix-or-retire |
| A6 | HaikuScorer sentiment tier-3 | `backend/news/sentiment.py:798 (tool_choice :809)` | claude-haiku-4-5-20251001 (dated pin) | forced tool-use (classify_sentiment) | 3 calls / 3,150 tok on claude-haiku-4-5-20251001, last 2026-07-07 (measured 30d llm_call_log); agent=NULL | **NO** | stay_metered | 78.8 (instrumentation only) |

- **A1 — why:** ALREADY railed today via make_client under PAPER_USE_CLAUDE_CODE_ROUTE=true; CC rail serves pydantic+dict schemas since 75.5. KEEP -- do not double-route.
- **A2 — why:** Same make_client rail as A1. KEEP.
- **A3 — why:** NOT an Anthropic site today -- Gemini default. Recorded as a LATENT claude path so a future DEEP_THINK_MODEL override does not silently create an unrouted metered site.
- **A4 — why:** No CC-rail equivalent: the advisor beta tool is a server-side tool-use feature the `claude -p` CLI does not expose. Already HARD-RAISES under the route flag (llm_client.py:2233-2240) -- fails loud, never silently metered.
- **A5 — why:** Batches API = 50% discount + 24h window; the CLI rail has NO batch equivalent, so railing it would COST money. Also carries a latent no-args TypeError (__init__(model_name, api_key) has no defaults).
- **A6 — why:** Forced tool_choice tool-use -- the CC rail (--json-schema) cannot express forced tool selection. Uninstrumented: writes no llm_call_log row, so the 3 logged rows come from elsewhere; its true volume is unmeasurable.

### B. Lite paper-trading path

| # | Role | Anchor (re-derived) | Model today | Structured output | 30d volume | Instrumented | Decision | Owner step |
|---|------|--------------------|-------------|-------------------|-----------|--------------|----------|------------|
| B1 | lite paper trader | `backend/services/autonomous_loop.py:2454 rail / :2472 direct` | rail: CLI session default (NO --model passed); direct: claude-sonnet-4-6 | json-prompt (regex extract, degraded-HOLD on parse fail) | 9 calls / 20,997 tok agent='lite_trader' provider='claude-code' ({M}); 4 failed | yes | **max_rail_cli** | 78.2 rail model-args |
| B2 | lite risk judge | `backend/services/autonomous_loop.py:2530 rail / :2548 direct` | same as B1 | json-prompt | 9 calls / 28,840 tok agent='lite_risk_judge' ({M}); 3 failed | yes | **max_rail_cli** | 78.2 |

- **B1 — why:** ALREADY railed. DEFECT: the rail branch passes no --model, so the model is whatever the CLI session default happens to be (silent drift).
- **B2 — why:** ALREADY railed; same no---model defect as B1.

### C. Signal-overlay services (the rail-BYPASS block -- top money)

| # | Role | Anchor (re-derived) | Model today | Structured output | 30d volume | Instrumented | Decision | Owner step |
|---|------|--------------------|-------------|-------------------|-----------|--------------|----------|------------|
| C1 | meta_scorer conviction | `backend/services/meta_scorer.py:221 ClaudeClient(` | claude-haiku-4-5 | pydantic-schema via dict (MetaScorerBatch) | 0 SUCCESSFUL calls in 30d. NOT a proven zero: ClaudeClient hardcodes ok=True at its log site (llm_client.py:~1905) and llm_client.py contains ZERO ok=False writers -- SDK errors re-raise (:1739/:1746/:1790) BEFORE the log block (:1886). So 0 rows means 'no call SUCCEEDED' and cannot distinguish 'never ran' from 'ran and every call failed' -- the latter being exactly the dead-credits outage this row's reason cites. UNMEASURABLE on the failure path (owner 78.8). | yes | **max_rail_cli** | 78.1 C-block rewire |
| C2 | news_screen | `backend/services/news_screen.py:267 ClaudeClient(` | claude-haiku-4-5 | pydantic-schema via dict (NewsSignalBatch) | 0 SUCCESSFUL calls in 30d. NOT a proven zero: ClaudeClient hardcodes ok=True at its log site (llm_client.py:~1905) and llm_client.py contains ZERO ok=False writers -- SDK errors re-raise (:1739/:1746/:1790) BEFORE the log block (:1886). So 0 rows means 'no call SUCCEEDED' and cannot distinguish 'never ran' from 'ran and every call failed' -- the latter being exactly the dead-credits outage this row's reason cites. UNMEASURABLE on the failure path (owner 78.8). | yes | **max_rail_cli** | 78.1 |
| C3 | macro_regime classify | `backend/services/macro_regime.py:506 ClaudeClient(` | claude-haiku-4-5 | pydantic-schema via dict (MacroRegimeOutput) | 0 SUCCESSFUL calls in 30d. NOT a proven zero: ClaudeClient hardcodes ok=True at its log site (llm_client.py:~1905) and llm_client.py contains ZERO ok=False writers -- SDK errors re-raise (:1739/:1746/:1790) BEFORE the log block (:1886). So 0 rows means 'no call SUCCEEDED' and cannot distinguish 'never ran' from 'ran and every call failed' -- the latter being exactly the dead-credits outage this row's reason cites. UNMEASURABLE on the failure path (owner 78.8). | yes | **max_rail_cli** | 78.1 |
| C4 | PEAD press-release signal | `backend/services/pead_signal.py:279 ClaudeClient(` | claude-haiku-4-5 | pydantic-schema via dict | 0 SUCCESSFUL calls in 30d. NOT a proven zero: ClaudeClient hardcodes ok=True at its log site (llm_client.py:~1905) and llm_client.py contains ZERO ok=False writers -- SDK errors re-raise (:1739/:1746/:1790) BEFORE the log block (:1886). So 0 rows means 'no call SUCCEEDED' and cannot distinguish 'never ran' from 'ran and every call failed' -- the latter being exactly the dead-credits outage this row's reason cites. UNMEASURABLE on the failure path (owner 78.8). | yes | **max_rail_cli** | 78.1 |
| C5 | analyst narrative scorer | `backend/services/analyst_narrative_scorer.py:135 ClaudeClient(` | claude-haiku-4-5 | pydantic-schema via dict | 0 SUCCESSFUL calls in 30d. NOT a proven zero: ClaudeClient hardcodes ok=True at its log site (llm_client.py:~1905) and llm_client.py contains ZERO ok=False writers -- SDK errors re-raise (:1739/:1746/:1790) BEFORE the log block (:1886). So 0 rows means 'no call SUCCEEDED' and cannot distinguish 'never ran' from 'ran and every call failed' -- the latter being exactly the dead-credits outage this row's reason cites. UNMEASURABLE on the failure path (owner 78.8). | yes | **max_rail_cli** | 78.1 |
| C6 | call-transcript GPR exposure | `backend/services/call_transcript_gpr.py:113 ClaudeClient(` | claude-haiku-4-5 | pydantic-schema via dict | 0 SUCCESSFUL calls in 30d. NOT a proven zero: ClaudeClient hardcodes ok=True at its log site (llm_client.py:~1905) and llm_client.py contains ZERO ok=False writers -- SDK errors re-raise (:1739/:1746/:1790) BEFORE the log block (:1886). So 0 rows means 'no call SUCCEEDED' and cannot distinguish 'never ran' from 'ran and every call failed' -- the latter being exactly the dead-credits outage this row's reason cites. UNMEASURABLE on the failure path (owner 78.8). | yes | **max_rail_cli** | 78.1 |

- **C1 — why:** TOP MONEY. Constructs ClaudeClient DIRECTLY, bypassing make_client, so PAPER_USE_CLAUDE_CODE_ROUTE never applies -- this is the phase-72 rail-bypass class that dies on dead credits and produced the 97%-cash incident. Dict schemas are rail-servable since 75.5.
- **C2 — why:** Same rail-bypass class as C1.
- **C3 — why:** Same class; gated macro_regime_filter_enabled=False by default.
- **C4 — why:** Same class.
- **C5 — why:** Same class.
- **C6 — why:** Same class.

### D. Layer-2 MAS + Slack bot

| # | Role | Anchor (re-derived) | Model today | Structured output | 30d volume | Instrumented | Decision | Owner step |
|---|------|--------------------|-------------|-------------------|-----------|--------------|----------|------------|
| D1 | MAS _call_agent | `backend/agents/multi_agent_orchestrator.py:1099` | mas_main/mas_qa=claude-opus-4-8; mas_communication/mas_research=claude-sonnet-4-6 | none (plain text) | UNMEASURABLE (call site writes no llm_call_log row -- see instrumented=false) | **NO** | **max_rail_cli** | 78.4 MAS rewire |
| D2 | MAS _call_agent_json | `backend/agents/multi_agent_orchestrator.py:1147` | same as D1 | pydantic/dict via output_config json_schema | UNMEASURABLE (call site writes no llm_call_log row -- see instrumented=false) | **NO** | **max_rail_cli** | 78.4 |
| D3 | MAS _call_agent_with_tools | `backend/agents/multi_agent_orchestrator.py:1268` | same as D1 | tool-use (AGENT_TOOLS) + interleaved thinking | UNMEASURABLE (call site writes no llm_call_log row -- see instrumented=false) | **NO** | stay_metered | 78.8 (instrumentation only) |
| D4 | Slack output leak detector | `backend/slack_bot/streaming_integration.py:527  [drift-corrected from the brief's original line]` | claude-haiku-4-5 (hardcoded :528) | forced tool-use (classify_output_leak) | UNMEASURABLE (call site writes no llm_call_log row -- see instrumented=false) | **NO** | stay_metered | 78.8 (instrumentation only) |
| D5 | openclaw_client chat/stream | `backend/agents/openclaw_client.py:72 / :154 (model table :47-52)` | hardcoded anthropic/claude-sonnet-4-6 (:48,:51) + opus-4-8 (:49,:50) | none | 0 -- DORMANT: openclaw_chat/openclaw_chat_stream have ZERO callers | **NO** | stay_metered | 78.6 disposition |

- **D1 — why:** Raw SDK, uninstrumented. Plain-text output is trivially rail-servable. Today a 401 latches a permanent per-instance Gemini fallback -- silent quality degradation rather than loud failure.
- **D2 — why:** Dict schemas are rail-servable (--json-schema since 75.5).
- **D3 — why:** Tool-use + interleaved-thinking loop -- the CC rail cannot serve either.
- **D4 — why:** Forced tool-use AND a per-response interactive latency budget; the CLI rail's subprocess spawn (~seconds) would be user-visible. Fail-open by design.
- **D5 — why:** NOT railed by this census: routes via the OpenClaw Gateway (:18789), not api.anthropic.com, and is dormant (zero callers). Listed so the hardcoded model table is not forgotten -- it is a live drift risk regardless of disposition.

### E. Ticket queue

| # | Role | Anchor (re-derived) | Model today | Structured output | 30d volume | Instrumented | Decision | Owner step |
|---|------|--------------------|-------------|-------------------|-----------|--------------|----------|------------|
| E1 | ticket queue processor | `backend/services/ticket_queue_processor.py:206 rail / :226 direct` | rail: CLI default (agent_model_map IGNORED); direct: opus-4-8 / sonnet-4-6 | none | 0 rows in 30d; rail branch logs via claude_code_invoke, direct branch does not | yes | **max_rail_cli** | 78.2 |

- **E1 — why:** ALREADY railed; same no---model defect as B1/B2 (the agent_model_map is silently ignored on the rail).

### F. Meta-evolution (Layer-4)

| # | Role | Anchor (re-derived) | Model today | Structured output | 30d volume | Instrumented | Decision | Owner step |
|---|------|--------------------|-------------|-------------------|-----------|--------------|----------|------------|
| F1 | directive_review | `backend/meta_evolution/directive_review.py:138  [drift-corrected from the brief's original line]` | claude-sonnet-4-6 (hardcoded) | json-prompt | UNMEASURABLE (call site writes no llm_call_log row -- see instrumented=false) | **NO** | **max_rail_cli** | 78.3 F/G rewire |
| F2 | directive_rewriter | `backend/meta_evolution/directive_rewriter.py:180  [drift-corrected from the brief's original line]` | claude-sonnet-4-6 (hardcoded) | json-prompt | UNMEASURABLE (call site writes no llm_call_log row -- see instrumented=false) | **NO** | **max_rail_cli** | 78.3 |
| F3 | skill_modification_review | `backend/agents/skill_modification_review.py:195  [drift-corrected from the brief's original line]` | claude-sonnet-4-6 (hardcoded) | json-prompt | UNMEASURABLE (call site writes no llm_call_log row -- see instrumented=false) | **NO** | **max_rail_cli** | 78.3 |

- **F1 — why:** Raw SDK behind a key-prefix gate: Anthropic only when the key startswith 'sk-ant-api', so an OAuth key silently skips to Gemini. Rail-servable (json-prompt).
- **F2 — why:** Same pattern as F1.
- **F3 — why:** Same pattern; anti-rubber-stamp reviewer.

### G. Harness Layer-3 planner

| # | Role | Anchor (re-derived) | Model today | Structured output | 30d volume | Instrumented | Decision | Owner step |
|---|------|--------------------|-------------|-------------------|-----------|--------------|----------|------------|
| G1 | Layer-3 planner_agent | `backend/agents/planner_agent.py:166 + :273` | claude-opus-4-8 | json-prompt | UNMEASURABLE (call site writes no llm_call_log row -- see instrumented=false) | **NO** | **max_rail_cli** | 78.3 |

- **G1 — why:** Raw SDK on the harness cadence -- exactly the rare-event/high-value shape the Max rail suits. (evaluator_agent.py is NOT an Anthropic site: Gemini via get_genai_client; its 'Claude Sonnet' docstring is stale.)

### H. RAG multimodal

| # | Role | Anchor (re-derived) | Model today | Structured output | 30d volume | Instrumented | Decision | Owner step |
|---|------|--------------------|-------------|-------------------|-----------|--------------|----------|------------|
| H1 | RAG multimodal index | `backend/agents/rag_agent_runtime.py:259` | claude-opus-4-8 | none (citations enabled) | UNMEASURABLE (call site writes no llm_call_log row -- see instrumented=false) | **NO** | stay_metered | 78.8 (instrumentation only) |

- **H1 — why:** Files API beta upload (files-api-2025-04-14) + citations; the CLI rail exposes neither, and citations x structured-outputs are mutually exclusive anyway (llm_client.py:1658-1665).

### I. Autoresearch nightly

| # | Role | Anchor (re-derived) | Model today | Structured output | 30d volume | Instrumented | Decision | Owner step |
|---|------|--------------------|-------------|-------------------|-----------|--------------|----------|------------|
| I1 | autoresearch nightly (fast/smart/strategic) | `scripts/autoresearch/run_memo.py:273-275` | haiku-4-5 / sonnet-4-6 / opus-4-8 | none (gpt-researcher library-internal) | 1 memo/night (cron); LLM calls are library-internal and unlogged | **NO** | **max_rail_proxy** | 76.9.2 (this cycle) |

- **I1 — why:** Third-party library (langchain_anthropic inside gpt-researcher) -- cannot be rewired to the CLI rail, so it routes over the HTTP bridge instead: ANTHROPIC_BASE_URL -> 127.0.0.1:18797 -> claude-code-proxy -> claude -p. Implemented in 76.9.2 behind AUTORESEARCH_USE_MAX_RAIL with loud-fail rc=78.

### J. Shell-script CLI call sites (already Max rail)

| # | Role | Anchor (re-derived) | Model today | Structured output | 30d volume | Instrumented | Decision | Owner step |
|---|------|--------------------|-------------|-------------------|-----------|--------------|----------|------------|
| J1 | MAS harness run_cycle.sh | `scripts/mas_harness/run_cycle.sh:66-71` | claude-opus-4-8 (explicit --model) | none | UNMEASURABLE (call site writes no llm_call_log row -- see instrumented=false) | **NO** | **max_rail_cli** | none (reference impl) |
| J2 | away-ops session driver | `scripts/away_ops/run_away_session.sh:146` | claude-opus-4-8 | json envelope (--output-format json) | UNMEASURABLE (call site writes no llm_call_log row -- see instrumented=false) | **NO** | **max_rail_cli** | none |

- **J1 — why:** ALREADY a direct `claude -p` Max-rail invocation. Passes --model explicitly (the correct pattern B1/B2/E1 should copy).
- **J2 — why:** ALREADY direct CLI Max rail.

## Scope disambiguation — things that look like call sites but are not

Every area named in the step scope is accounted for either as a role row (call sites) or a scope_disambiguation (things that look like call sites but are not). All claude_code_invoke callers were enumerated: services/autonomous_loop.py (B1,B2) and services/ticket_queue_processor.py (E1) — no others exist outside tests and the definition itself.

| Item | Anchor | Verdict | Why it is listed |
|------|--------|---------|------------------|
| backend/autonomous_loop.py (TOP-LEVEL, phase-3.3 planner entry) | `backend/autonomous_loop.py:75 (planner_model default) + :369 PlannerAgent(model=self.planner_model)` | NOT a call site — it configures and constructs PlannerAgent; the messages.create calls live in agents/planner_agent.py:166/:273, censused as row G1. | The step scope explicitly asks to disambiguate this from services/autonomous_loop.py (rows B1-B2). Both files are covered: this one as a caller, that one as two call sites. |
| backend/services/autonomous_loop.py:2696 _run_gemini_analysis | `backend/services/autonomous_loop.py:2696 make_client(...)` | NOT an Anthropic site — Gemini-only; the code raises on claude-* model names (:2692). | Closes the seed question of whether the lite path had a third LLM call. |
| backend/agents/evaluator_agent.py | `get_genai_client / GEMINI_WORKHORSE` | NOT an Anthropic site. Its docstring saying 'Uses Claude Sonnet' is STALE (already recorded in phase-75.5 llmeng-06). | A stale docstring is exactly the kind of thing a census gets fooled by; recorded so a future sweep does not re-add it. |
| backend/agents/debate.py:97, backend/agents/risk_debate.py:93, multi_agent_orchestrator.py:1114/:1164/:1282 | ``import anthropic` occurrences` | NOT distinct call sites — these imports exist for exception TYPING only (catching anthropic error classes). | An import-name sweep counts these; a call-site census must not. |
| MCP servers (scripts/mcp_servers/*) | `whole directory` | EXCLUDED — verified zero outbound LLM call surfaces. | Named as excluded in the step scope; re-verified by the gate's sweep rather than assumed. |
| scripts/away_ops/healthcheck.sh:95 | ``claude auth status`` | NOT an LLM call — an auth probe only. | A `claude` grep hits it; it bills nothing and routes nothing. |
| K3-K5 transport/infra (GitHub Models branch llm_client.py:2148; Files API upload helpers llm_client.py:1413 + tools/sec_insider.py:331; module-level SDK import llm_client.py:316) | `see each` | Routing CONTEXT, not inference call sites. K4 does reach api.anthropic.com but uploads files rather than performing inference. | K3 matters because it means claude-* ids CAN route via models.github.ai — a third transport a future router decision must account for. |

## Follow-up steps authored from this census

- `76.9.2 (this cycle)` — covers I1
- `78.1` — covers C2, C3, C4, C5, C6
- `78.1 C-block rewire` — covers C1
- `78.2` — covers B2, E1
- `78.2 rail model-args` — covers B1
- `78.3` — covers F2, F3, G1
- `78.3 F/G rewire` — covers F1
- `78.4` — covers D2
- `78.4 MAS rewire` — covers D1
- `78.5 fix-or-retire` — covers A5
- `78.6 disposition` — covers D5
- `78.7` — covers A2
- `78.7 (sonnet-5 swap only)` — covers A1
- `78.8 (instrumentation only)` — covers A6, D3, D4, H1
- `none` — covers J2
- `none (justified)` — covers A4
- `none (reference impl)` — covers J1
- `none (watch item)` — covers A3
