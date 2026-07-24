# 78.0 routing decision table — Main DRAFT (staged for GENERATE; scratchpad only)

Basis: research_brief_78.0.md census A-H + 30d llm_call_log volumes + measured flag
PAPER_USE_CLAUDE_CODE_ROUTE=true (backend/.env L75) + external docs (Batches/CLI/limits).

## Already Max-railed TODAY (flag ON via make_client) — decision: max_rail_cli KEEP
| Row | Role | Evidence |
|---|---|---|
| A1/A2 | Layer-1 enrichment/debate/quant tiers (per-ticker) | make_client honors the flag; 30d volume: 2,241 cc_rail sonnet-4-6 calls / 4.1M tok = THIS traffic |
| B1/B2 | lite trader + risk judge | dual rail, rail branch active; 8+8 calls provider=claude-code |
| E1 | ticket queue | dual rail, rail branch active |
KEEP-items: B1/B2/E1 pass NO --model on the rail (session-default drift risk) — fix step candidate; A1/A2 sonnet-4-6 pin -> sonnet-5 swap applies HERE (rail-routed, $0).

## The REAL routing gap — make_client-BYPASS sites; decision: max_rail (rewire through make_client or ClaudeCodeClient)
| Row | Role | Today | Why route |
|---|---|---|---|
| C1 | meta_scorer (haiku, pydantic-dict) | direct ClaudeClient = metered, DIES on dead credits | phase-72 97%-cash class; CC rail serves dict schemas since 75.5 |
| C2-C6 | news_screen, macro_regime, pead, analyst_narrative, call_transcript_gpr | same | same class; all haiku + dict-schema = rail-servable |
| F1-F3 | directive_review/rewriter, skill_mod_review (sonnet-4-6, json-prompt) | raw SDK + sk-ant-api key-prefix gate | rare-event; rail-servable; sonnet-5 swap rides along |
| G1 | planner_agent (opus-4-8, json-prompt) | raw SDK | harness-run cadence; rail-servable |
| D1/D2 | MAS _call_agent / _call_agent_json | raw SDK (Gemini latch on 401) | rail-servable (D2 dict schemas OK) |

## Justified STAY-METERED (CC rail structurally cannot serve)
| Row | Role | Blocker |
|---|---|---|
| A4 | advisor_call | beta advisor tool (tool-use); already hard-raises under the route flag; DARK today |
| A5 | BatchClient | Batches API (50% discount, 24h) has NO CLI equivalent; DOUBLY DARK + latent no-args TypeError — fix-or-retire step |
| A6 | HaikuScorer | forced tool_choice tool-use |
| D3 | _call_agent_with_tools | tool-use + interleaved thinking |
| D4 | slack leak detector | forced tool-use; per-message latency |
| H1 | multimodal_index_claude | Files API beta + citations |
STAY-items need the phase-72 credit token to actually function; near-zero 30d volume = blocked demand, disclosed.

## Special
| Row | Decision |
|---|---|
| autoresearch (langchain) | max_rail_proxy — DONE in 76.9.2 (flag AUTORESEARCH_USE_MAX_RAIL) |
| D5 openclaw_client | DORMANT (zero callers) — retire-or-keep decision step; hand-edit sonnet-4-6 table regardless (goal) |
| A3 deep_think/synthesis | Gemini default; latent claude path only under operator override — note only |
| K1/--bare trap | routing table RULE: CC-rail invocations must NEVER pass --bare (silently falls to metered key — S6 doc) |

## Top-3 money roles (DoD): (1) C1-C6 overlay block (rewire = un-dies the signal stack), (2) A1/A2 (already railed; sonnet-5 swap + keep), (3) autoresearch (76.9.2). Remediation steps to author at 78.0 close: rewire-C-block, rail-model-args (B/E), F/G/D1-D2 rewire, BatchClient fix-or-retire, D5 disposition, sonnet-5 swap + 76.11+_VALID_MODELS, 75.5.12 already queued.
