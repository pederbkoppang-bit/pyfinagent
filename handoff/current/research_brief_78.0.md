# Research Brief — Step 78.0: Anthropic Call-Site Census (AUDIT-CLASS)

Status: COMPLETE (internal 9-round audit-class sweep + external half; gate_passed=true)
Tier: complex (audit-class, coverage.audit_class=true set by caller)
Date: 2026-07-24
Researcher: Layer-3 researcher (workflow structured-output launch)

## Mission

Complete census of every outbound Anthropic/Claude LLM call site in the repo,
with per-site facts (area, entry_point, client_path, model, structured_output
needs, notes) so Main can write the routing-decision table for step 78.0.
Prior fact-check wf_3b9205bc-666 found 4 sites a hand-list missed; this gate
sweeps until TWO consecutive dry rounds.

## Sweep log (adaptive coverage)

| Round | Lens | New call sites found | Dry? |
|-------|------|---------------------|------|
| 1 | import sweep (anthropic/langchain_anthropic/ChatAnthropic/claude_code_invoke/ClaudeCodeClient) | debate.py:97, risk_debate.py:93 (error-typing only, NOT distinct sites); 3 extra import sites in multi_agent_orchestrator (:1114/:1164/:1282 = error-typing only); confirmed seed sites | NO |
| 2 | direct-construction sweep (`ClaudeClient(`) | 6 sites: analyst_narrative_scorer:135, meta_scorer:221, pead_signal:279, macro_regime:506, call_transcript_gpr:113, news_screen:267 — 4 NOT in seed. ALL bypass make_client => CC-rail flag NEVER applies to them | NO |
| 3 | env-var sweep (ANTHROPIC_API_KEY) | scripts/ops/anthropic_max_bridge.py (76.9.2 transport infra, unwired — flag AUTORESEARCH_USE_MAX_RAIL default OFF); other hits = checkers/display only (secrets_rotation_check, cron_dashboard_api, settings_api, main.py, data_sources) | NO |
| 4 | HTTP-literal sweep (api.anthropic.com) | bridge again (dedup); comments only elsewhere | DRY-ish (dedup only) |
| 5 | model-id sweep (`claude-` strings, all files) | 0 new call sites (cost_tracker/harness_memory/app_home/metered_spend/finra_audit = pricing+display) | DRY |
| 6 | config sweep (settings *_model + resolve_model callers) | gemini_model DEFAULT = "claude-sonnet-4-6" (settings.py:31) => Layer-1 standard tier IS Claude; 6 haiku overlay settings confirmed; resolve_model callers = agent_definitions/agent_map/settings_api/run_memo only | NO |
| 7 | shell/plist lens (`claude -p` in .sh) | scripts/mas_harness/run_cycle.sh:66-71 (claude -p --model claude-opus-4-8, launchd); scripts/away_ops/run_away_session.sh:146 (probe) + session body; healthcheck.sh = `claude auth status` only (NOT an LLM call); run_nightly.sh max-rail env wiring (flag OFF) | NO |
| 8 | async/constructor/beta sweep (AsyncAnthropic, .Anthropic(, beta.messages) | 0 new — all 15 constructor sites already censused; zero AsyncAnthropic in repo | DRY (1) |
| 9 | frontend + curl-in-shell + GitHub-catalog lens | 0 new call sites (frontend = display/cost components; no curl; GITHUB_MODELS_CATALOG includes claude-opus-4-8..haiku-4-5 = detail of make_client row) | DRY (2) |

**coverage.dry = true** (rounds 8 and 9 consecutive dry). MCP spot-check honored: scripts/mcp_servers/* have zero LLM-call surfaces (reconcile_alpaca_deny_list.py hits are .claude/settings.json path strings; smoke tests 0 hits).

## COMPLETE CENSUS (all call sites, verified file:line)

### A. Layer-1 analysis pipeline (highest frequency: per-ticker, ~20-39 calls/analysis)
| # | Entry point | Client path | Model today | Structured output | Notes |
|---|------------|-------------|-------------|-------------------|-------|
| A1 | orchestrator.py:652 `general_client = make_client(settings.gemini_model,...)` | make_client -> ClaudeCodeClient (CC rail, flag ON) or ClaudeClient | claude-sonnet-4-6 (settings.py:31 DEFAULT — field name is legacy) | pydantic-schema (9 pipeline schemas; ClaudeClient output_config json_schema :1682-1701; CC rail --json-schema since 75.5 handles dict+pydantic-CLASS) | Enrichment+debate tier. Frequency = per-ticker full pipeline. Consumed by debate.py/risk_debate.py (their `import anthropic` = error-typing only, NOT distinct sites) |
| A2 | orchestrator.py:659 `quant_exec_client = make_client(settings.gemini_model,...)` | same as A1 | claude-sonnet-4-6 | same as A1 | quant executor leg |
| A3 | orchestrator.py:653-654 deep_think/synthesis clients | make_client | GEMINI_DEEP_THINK default (NOT claude unless operator overrides DEEP_THINK_MODEL) | — | Anthropic only under operator override; census-relevant as a LATENT claude path |
| A4 | orchestrator.py:1520 `advisor_call(...)` synthesis | advisor_call (llm_client.py:2191; beta.messages.create :2273, betas=["advisor-tool-2026-03-01"]) | executor=claude-sonnet-4-6 + advisor=claude-opus-4-8 (defaults :2194-2195) | tool-use (advisor_20260301 beta tool) | DARK: enable_advisor_tool=False (settings.py:391). HARD-RAISES under paper_use_claude_code_route (llm_client.py:2233-2240) — NO CC-rail equivalent. Budget guard _check_cost_budget :2225 |
| A5 | orchestrator.py:1043 `BatchClient()` via _run_enrichment_batch (:1006) | BatchClient (llm_client.py:1931; messages.batches.create :1978) | per-request model param | n/a (params passthrough) | DOUBLY DARK: backtest_batch_mode flag has ZERO consumers; _run_enrichment_batch has NO caller; AND `BatchClient()` no-args is a latent TypeError (__init__(model_name, api_key) :1958 has no defaults). Batch API = 50% discount, 24h window — NO Max-rail equivalent |
| A6 | news/sentiment.py:798 HaikuScorer.score `messages.create` | raw SDK (anthropic.Anthropic :786) | claude-haiku-4-5-20251001 (DATED pin, :93) | tool-use (forced tool_choice classify_sentiment :809) | Sentiment cascade tier-3 (escalation only, below-confidence cases). sentiment_haiku_batch_mode flag (settings.py:138) is UNWIRED (zero consumers). Deliberately avoids ClaudeClient (generic prefix, :761 comment) |

### B. Lite paper-trading path (daily cycle x ~13 tickers x 2 calls)
| # | Entry point | Client path | Model today | Structured output | Notes |
|---|------------|-------------|-------------|-------------------|-------|
| B1 | services/autonomous_loop.py:2454 lite trader (claude_code_invoke) / :2472 direct `client.messages.create` | dual rail: claude_code_invoke when paper_use_claude_code_route else raw SDK (:2423) | rail: CLI session default (NO --model passed!); direct: settings.gemini_model -> claude-sonnet-4-6 | json-prompt (regex JSON extract :2480; parse-fail => degraded HOLD marker 70.4) | metered via _log_claude_code_call agent="lite_trader" (56.2) |
| B2 | services/autonomous_loop.py:2530 lite risk judge / :2548 direct | same dual rail | same as B1 | json-prompt | independent 2nd call (25.A); agent="lite_risk_judge" |
| B3 | services/autonomous_loop.py:2696 _run_gemini_analysis | make_client | Gemini-ONLY (guard raises on claude-* :2692) | — | NOT an Anthropic site; recorded to close the seed question |

### C. Signal-overlay services (daily cycle; ALL bypass make_client -> CC-rail flag NEVER applies; all raw-key ClaudeClient)
| # | Entry point | Client path | Model today | Structured output | Notes |
|---|------------|-------------|-------------|-------------------|-------|
| C1 | services/meta_scorer.py:221 | direct ClaudeClient(generate_content) | meta_scorer_model=claude-haiku-4-5 (settings.py:409) | pydantic-schema via dict (MetaScorerBatch.model_json_schema stripped; ClaudeClient sends output_config json_schema — dict accepted :1686) | phase-72: meta-scorer rail-bypass was a root cause of 97%-cash incident. Batch conviction scorer |
| C2 | services/news_screen.py:267 | direct ClaudeClient | news_screen_model=claude-haiku-4-5 (:402) | pydantic-schema via dict (NewsSignalBatch) | 48K max-tokens cap + parse-retry (69.3) |
| C3 | services/macro_regime.py:506 | direct ClaudeClient | macro_regime_model=claude-haiku-4-5 (:395) | pydantic-schema via dict (MacroRegimeOutput) | daily regime classify; gated macro_regime_filter_enabled=False default; consumed at screener.py:321 |
| C4 | services/pead_signal.py:279 | direct ClaudeClient | pead_signal_model=claude-haiku-4-5 (:398) | pydantic-schema via dict (PeadSignalOutput) | PEAD press-release scoring |
| C5 | services/analyst_narrative_scorer.py:135 | direct ClaudeClient | analyst_narrative_model=claude-haiku-4-5 (:490) | pydantic-schema via dict (hand-written) | 8-K exhibit-99 tone |
| C6 | services/call_transcript_gpr.py:113 | direct ClaudeClient | call_transcript_gpr_model=claude-haiku-4-5 (:501) | pydantic-schema via dict (hand-written) | GPR exposure classify |

### D. Layer-2 MAS + Slack bot (per Slack message / harness trigger)
| # | Entry point | Client path | Model today | Structured output | Notes |
|---|------------|-------------|-------------|-------------------|-------|
| D1 | multi_agent_orchestrator.py:1099 _call_agent | raw SDK (client built :245) | agent_definitions resolve_model: mas_main/mas_qa=claude-opus-4-8, mas_communication/mas_research=claude-sonnet-4-6 | none (plain text) | 401 => permanent per-instance Gemini fallback latch (_anthropic_unavailable) |
| D2 | multi_agent_orchestrator.py:1147 _call_agent_json | raw SDK | same | pydantic/dict via output_config json_schema (constrained decoding, 71.2) | fail-safe degrade to D1 |
| D3 | multi_agent_orchestrator.py:1268 _call_agent_with_tools | raw SDK | same | tool-use (AGENT_TOOLS) + thinking (adaptive on opus-4-8; fable/sonnet-5 branches :1236-1263) | TOOL-USE — CC rail CANNOT serve. Interleaved thinking loop |
| D4 | slack_bot/streaming_integration.py:526 detect_llm_leak (client :506) | raw SDK, max_retries=1 | claude-haiku-4-5 (hardcoded :528) | tool-use (forced classify_output_leak) | per-streamed-Slack-response leak check; fail-OPEN |
| D5 | agents/openclaw_client.py:72 openclaw_chat / :154 stream (model table :47-52) | OpenClaw Gateway HTTP :18789 (OpenAI-compat) | hardcoded anthropic/claude-sonnet-4-6 (:48,:51) + anthropic/claude-opus-4-8 (:49,:50) | none | DORMANT: zero callers of openclaw_chat/_stream (only check_gateway_health mao:320 + list_openclaw_sessions mas_events.py:177 used). Hardcoded table = drift risk |

### E. Ticket queue (1-2 tickets/day)
| # | Entry point | Client path | Model today | Structured output | Notes |
|---|------------|-------------|-------------|-------------------|-------|
| E1 | services/ticket_queue_processor.py:206 claude_code_invoke / :226 raw SDK | dual rail (rail added 56.2) | rail: CLI default (agent_model_map IGNORED on rail — no model arg!); direct: map :172-176 main/q-and-a=claude-opus-4-8, research=claude-sonnet-4-6 | none | 60s timeout both rails |

### F. Meta-evolution Layer-4 (review cycles; rare)
| # | Entry point | Client path | Model today | Structured output | Notes |
|---|------------|-------------|-------------|-------------------|-------|
| F1 | meta_evolution/directive_review.py:139 messages.create (client :137) | raw SDK | claude-sonnet-4-6 (hardcoded) | json-prompt | key-prefix gate: Anthropic ONLY if key startswith "sk-ant-api" (OAuth sk-ant-oat skips to Gemini); Gemini GEMINI_WORKHORSE fallback |
| F2 | meta_evolution/directive_rewriter.py:181 | raw SDK | claude-sonnet-4-6 (hardcoded) | json-prompt | same pattern |
| F3 | agents/skill_modification_review.py:196 | raw SDK | claude-sonnet-4-6 (hardcoded) | json-prompt | same pattern; anti-rubber-stamp reviewer |

### G. Harness Layer-3 planner (rare: harness runs)
| # | Entry point | Client path | Model today | Structured output | Notes |
|---|------------|-------------|-------------|-------------------|-------|
| G1 | agents/planner_agent.py:166 + :273 messages.create (Anthropic() :87, env-key) | raw SDK | claude-opus-4-8 (default :78; backend/autonomous_loop.py:75 passes same) | json-prompt | evaluator_agent.py is NOT Anthropic (GEMINI_WORKHORSE via get_genai_client, 75.5 llmeng-06; "Claude Sonnet" docstring STALE). Entry: scripts/harness/run_autonomous_loop.py |

### H. RAG multimodal (rare, on-demand)
| # | Entry point | Client path | Model today | Structured output | Notes |
|---|------------|-------------|-------------|-------------------|-------|
| H1 | agents/rag_agent_runtime.py:259/:261 multimodal_index_claude (client :229) | raw SDK (env ANTHROPIC_API_KEY only) | claude-opus-4-8 (param default :204) | none (citations enabled) | beta files-api-2025-04-14 upload for PDFs (:237) — Files API + citations => CC rail cannot serve; citations x structured-outputs mutually exclusive (llm_client :1658-1665) |

### I. Autoresearch nightly (cron)
| # | Entry point | Client path | Model today | Structured output | Notes |
|---|------------|-------------|-------------|-------------------|-------|
| I1 | scripts/autoresearch/run_memo.py:273-275 env FAST/SMART/STRATEGIC_LLM | gpt-researcher lib (langchain_anthropic inside), `anthropic:<model>` provider strings | haiku-4-5 / sonnet-4-6 / opus-4-8 (resolve_model autoresearch_*) | none (library-internal) | ANTHROPIC_API_KEY required :288. run_nightly.sh:79-93 max-rail block: AUTORESEARCH_USE_MAX_RAIL=1 => ANTHROPIC_BASE_URL/API_URL=127.0.0.1:18797 + dummy key, LOUD-FAIL rc=78 if bridge down. Flag default OFF (76.9.2 pending) |

### J. Shell-script CLI call sites (Max rail direct)
| # | Entry point | Client path | Model today | Structured output | Notes |
|---|------------|-------------|-------------|-------------------|-------|
| J1 | scripts/mas_harness/run_cycle.sh:66-71 `claude -p --dangerously-skip-permissions --model claude-opus-4-8` | claude CLI (Max rail) | claude-opus-4-8 (explicit flag) | none (text) | launchd-scheduled MAS harness cycle; 3600s gtimeout; stdin prompt |
| J2 | scripts/away_ops/run_away_session.sh:146 auth probe + session body | claude CLI | claude-opus-4-8 (--max-turns 1 probe) | json envelope (--output-format json) | away-window driver (idle outside away periods). healthcheck.sh:95 = `claude auth status` only — NOT an LLM call |

### K. Transport infra (not call sites; routing decision context)
| # | Item | Notes |
|---|------|-------|
| K1 | agents/claude_code_client.py:215 claude_code_invoke | THE CC rail: subprocess `claude --print --output-format json --disallowedTools Bash,Edit,...`; NO --model flag => CLI session default model decides; --json-schema (dict) for structured output; max_tokens is a NO-OP at CLI layer; rail guard 66.1 can skip; ClaudeCodeClient (:458) converts pydantic-CLASS + dict schemas to --json-schema since 75.5 (tool-use NOT servable) |
| K2 | scripts/ops/anthropic_max_bridge.py | phase-76.9.2 (PENDING): stdlib localhost adapter 127.0.0.1:18797 -> claude-code-proxy https://localhost:18796 -> `claude -p` Max rail; aggregates SSE to non-streaming Messages JSON (anthropic SDK 0.96.0 silent-corruption workaround); plist template exists, OPERATOR bootstraps (OPS-BRIDGE-BOOTSTRAP token) |
| K3 | llm_client.py:2148 GitHub Models branch | GITHUB_MODELS_CATALOG includes claude-opus-4-8..haiku-4-5 (+legacy) => claude-* CAN route via models.github.ai OpenAI-compat with GITHUB_TOKEN when Anthropic-direct key absent + CC-rail off. OpenAIClient(base_url=...) :1174-1186 |
| K4 | tools/sec_insider.py:331 upload_large_filing_to_files_api + llm_client.py:1413 upload_file_to_anthropic_files_api | Files API uploads (client.beta.files.upload) — outbound api.anthropic.com but NOT inference; helpers take/construct SDK client |
| K5 | llm_client.py:316 module-level `import anthropic as _anthropic_sdk` | shared SDK import for ClaudeClient/BatchClient; not a site |

## (superseded early list — kept for audit trail)

1. **Layer-2 MAS `_call_agent`** — multi_agent_orchestrator.py:1099 `client.messages.create` (client built :245 `anthropic.Anthropic`). Raw SDK, plain text. Models = agent_config.model (mas_main/mas_qa=claude-opus-4-8, mas_communication/mas_research=claude-sonnet-4-6). Gemini fallback on 401 (`_anthropic_unavailable` latch).
2. **Layer-2 MAS `_call_agent_json`** — multi_agent_orchestrator.py:1147, `output_config={"format":{"type":"json_schema"}}` constrained decoding (phase-71.2). Fail-safe degrade to _call_agent.
3. **Layer-2 MAS `_call_agent_with_tools`** — multi_agent_orchestrator.py:1268, `tools=AGENT_TOOLS` + thinking (adaptive on opus-4-8). TOOL-USE — CC rail CANNOT serve this.
4. **openclaw_client.py:47-52** — AGENT_MODEL_OVERRIDES hardcodes anthropic/claude-sonnet-4-6 (:48 communication, :51 research) + anthropic/claude-opus-4-8 (:49 main, :50 qa). Routes via OpenClaw Gateway HTTP :18789, NOT api.anthropic.com. **openclaw_chat/openclaw_chat_stream have ZERO callers** — dormant path; only check_gateway_health (mao:320) + list_openclaw_sessions (mas_events.py:177) used.
5. **ClaudeClient** — llm_client.py:1331 (SDK client :1367, max_retries=3, prompt caching). Constructed by make_client:2139 (Anthropic-direct branch) + direct constructions (TBD sweep). Pydantic/dict response_schema honored via prompt-embedding (json-prompt) — verify.
6. **BatchClient** — llm_client.py:1931, messages.batches.create :1978. Caller orchestrator.py:1043 `BatchClient()` — **LATENT TypeError: __init__(model_name, api_key) has no defaults**. Gated dark by backtest_batch_mode=False (settings.py:144). Batch API = 50% discount; NO Max-rail equivalent.
7. **advisor_call** — llm_client.py:2191, `client.beta.messages.create` :2273 with betas=["advisor-tool-2026-03-01"], executor=claude-sonnet-4-6, advisor=claude-opus-4-8. Caller orchestrator.py:1520 (synthesis), gated by enable_advisor_tool=False (settings.py:391). HARD-RAISES when paper_use_claude_code_route=True (:2233-2240) — beta tool has NO CC-rail equivalent.
8. **make_client CC rail** — llm_client.py:2099-2113 routes claude-* -> ClaudeCodeClient (subprocess `claude` CLI) when paper_use_claude_code_route=True; :2128 routing-breach guard hard-fails Anthropic-direct fallthrough under the flag; :2148 GitHub Models path can serve claude-* via models.github.ai + PAT (OpenAI-compat, non-Anthropic transport).
9. **evaluator_agent.py** — NOT an Anthropic site (GEMINI_WORKHORSE via get_genai_client, phase-75.5 llmeng-06; docstring "Uses Claude Sonnet" is STALE). planner_agent.py:21 `from anthropic import Anthropic` IS a site (detail TBD).
10. **run_memo.py:273-275** — gpt-researcher env config `FAST_LLM/SMART_LLM/STRATEGIC_LLM = anthropic:<resolve_model(...)>` (haiku-4-5 / sonnet-4-6 / opus-4-8); ANTHROPIC_API_KEY required :288. Transport = gpt-researcher's internal langchain_anthropic.
11. **macro_regime_model** — settings.py:395 claude-haiku-4-5 (call site TBD — round-2 follow-up).

## Model resolution TODAY (model_tiers.py)

mas_main=claude-opus-4-8 (:110), mas_qa=claude-opus-4-8 (:115), mas_communication=claude-sonnet-4-6 (:98), mas_research=claude-sonnet-4-6 (:117), autoresearch_fast=claude-haiku-4-5 (:122), autoresearch_smart=claude-sonnet-4-6 (:123), autoresearch_strategic=claude-opus-4-8 (:128).

## External sources — read in full (>=5 required)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|-----|----------|------|-------------|-------------|
| S1 | https://platform.claude.com/docs/en/build-with-claude/batch-processing | 2026-07-24 | Official doc (Anthropic) | WebFetch (full, 70.6KB) | Message Batches API: 50% off ALL active models; batch = up to 100,000 requests OR 256 MB, whichever first; async, most <1h but hard 24h expiry; results retained 29 days; Workspace-scoped to an API key; each request needs `max_tokens>=1` (`max_tokens:0` unsupported). Endpoint is `POST https://api.anthropic.com/v1/messages/batches` — a **metered direct-API** async endpoint. Confirms census row A5's "NO Max-rail equivalent": the CC/Max rail is synchronous `claude --print`; there is no batch submission/poll surface on the subscription rail, so A5 must stay metered if ever wired. Also validates A5's max_tokens caveat is orthogonal to the `BatchClient()` no-arg TypeError. |
| S2 | https://platform.claude.com/docs/en/build-with-claude/structured-outputs | 2026-07-24 | Official doc (Anthropic) | WebFetch (full) | Structured outputs supplied via `output_config.format = {type:"json_schema", schema:{...}}` (legacy `output_format` moved here; beta header `structured-outputs-2025-11-13` no longer required). Pydantic natively supported via `client.messages.parse(output_format=Model)`; SDKs auto-transform schemas (strip unsupported constraints). Uses **constrained decoding** with compiled-grammar caching (24h, invalidated by schema/tool-set change). GA on Claude 4.5+ incl. opus-4-8/sonnet-4-6/haiku-4-5. JSON-Schema limits: NO `minimum/maximum/multipleOf`, NO `minLength/maxLength`, NO recursive/external `$ref`, `additionalProperties` only `false`, `minItems` only 0/1 — unsupported keys => 400. Directly grounds C1-C6 + A1/A2 (all send `output_config json_schema`; the ClaudeClient dict-schema path at llm_client.py:1686 is the API-native shape). This is the API surface; the CLI `--json-schema` shape is a distinct wrapper (see S3/S6). |
| S3 | https://code.claude.com/docs/en/agent-sdk | 2026-07-24 | Official doc (Anthropic / Claude Code) | WebFetch (full) | Agent SDK overview. Confirms the headless CLI programmatic path: "run the CLI programmatically with the `-p` flag and `--output-format json`". SDKs bundle a native Claude Code binary. ROUTING-CRITICAL note: "Unless previously approved, Anthropic does not allow third party developers to offer claude.ai login or rate limits for their products, including agents built on the Claude Agent SDK. Please use the API key authentication methods described in this document instead." => the Max/claude.ai subscription rail is licensed for the operator's OWN interactive use (which is exactly how the CC rail K1 is used here — the operator's own Max login on their own Mac), NOT for redistributing subscription throughput. Grounds the CC-rail=K1 vs metered-API distinction that the census's A/C rows hang on. Auth alternatives = ANTHROPIC_API_KEY, Bedrock, Vertex, Foundry — all metered. |
| S4 | https://www.truefoundry.com/blog/claude-code-limits-explained | 2026-07-24 | Practitioner blog | WebFetch (full) | Claude Max usage-limit shape: dual limit = 5-hour ROLLING window (counter starts on first prompt, not a fixed clock) + weekly active-compute cap. CRITICAL for routing: "usage pools across Claude Code, Claude.ai chat, and Cowork, so heavy use in one drains capacity in the others" — a shared bucket, so pushing high-frequency Layer-1 (A1/A2, ~20-39 calls/ticker) onto the CC/Max rail competes with the dev-harness's own Max budget. Approx allowances: Max 5x ~50-225 prompts/5h + ~140-240 Sonnet + 15-35 Opus weekly hrs; Max 20x ~200-900 prompts/5h + 240-480 Sonnet + 24-40 Opus hrs. NO published per-second throughput ceiling. (Recency caveat: this blog dates the peak-hours reduction to Mar 2026; newer sources say it was REMOVED May 6 2026 — see recency scan.) |
| S5 | https://docs.litellm.ai/docs/routing | 2026-07-24 | Official doc (LiteLLM, gateway prior-art) | WebFetch (full) | LLM-gateway routing prior-art. `model_list` maps a logical `model_name` to N physical deployments (`litellm_params.model = anthropic/<model>`, `api_key`, optional `api_base`); multiple deployments share one name for load-balance/failover. Strategies: `simple-shuffle` (default, weight-aware), `latency-based-routing` (`lowest_latency_buffer`, `ttl`), `usage-based-routing-v2` (lowest-TPM, Redis), `least-busy`, `cost-based` (`litellm_model_cost_map`). Failover: `order` priority, `allowed_fails`+cooldowns (5s default), `num_retries` w/ exp-backoff. Prior-art lesson for pyfinagent's routing table: a single logical name fanning to {CC-rail, metered-direct, GitHub-Models K3} deployments with cost-based preference + health-cooldown failover is the industry pattern the make_client + rail-guard (66.1) already approximate ad-hoc; census confirms pyfinagent has NO central router — routing is per-site (make_client flag vs direct ClaudeClient bypass), which is why C1-C6 silently escape the CC-rail flag. |
| S6 | https://code.claude.com/docs/en/headless | 2026-07-24 | Official doc (Anthropic / Claude Code) | WebFetch (full) | The CLI structured-output reference behind the CC rail (K1). `claude -p`/`--print` non-interactive; `--output-format text|json|stream-json`; `--json-schema '<JSON Schema>'` constrains output, delivered in the **`structured_output`** envelope field alongside `result` (text), `session_id`, `total_cost_usd` + per-model cost breakdown. Confirms census K1 verbatim: the rail's schema path is `--output-format json --json-schema`, and there is NO output-token-cap flag in `-p` mode (grounds K1/B1's "max_tokens is a NO-OP at the CLI layer"). Hardening dates (recency): `--json-schema` validation only became strict in **v2.1.205** (before that, invalid schema => silent unstructured text); `format` keyword accepted but NOT enforced. ROUTING-CRITICAL addition: `--bare` mode "skips OAuth and keychain reads. Anthropic authentication must come from `ANTHROPIC_API_KEY` or an `apiKeyHelper`" — so the CC rail keeps the operator's Max/OAuth login ONLY because K1's invocation does NOT pass `--bare`; a hermetic/CI `--bare` call would silently fall to a metered key. |

## Snippet-only sources

| URL | Kind | Why not read in full |
|-----|------|----------------------|
| https://anthropic.com/news/message-batches-api | Official announcement | Superseded by the live batch-processing doc (S1); launch post, older numbers |
| https://platform.claude.com/docs/en/api/client-sdks | Official doc (SDK landing) | Fetched but is a nav landing page — no `base_url`/`ANTHROPIC_BASE_URL` body content; the actual param lives on the per-language SDK subpage. base_url usage is already internally confirmed (bridge K2 + autoresearch I1 set `ANTHROPIC_BASE_URL`/`ANTHROPIC_API_URL`) |
| https://www.morphllm.com/claude-code-usage-limits | Practitioner blog | Corroborates S4; used for the May-6-2026 recency delta (peak-hours reduction REMOVED) |
| https://www.developersdigest.tech/blog/claude-code-usage-limits-playbook-2026 | Practitioner blog | Duplicate coverage of Max limits (S4 sufficed) |
| https://intuitionlabs.ai/articles/claude-max-plan-pricing-usage-limits | Practitioner blog | $100 (Max 5x) / $200 (Max 20x) pricing detail; not load-bearing for the census |
| https://techsy.io/en/blog/claude-2x-usage-limits-explained | Practitioner blog | 2x-limit-change explainer; secondary to S4 |
| https://www.jdhodges.com/blog/claude-ai-usage-limits/ | Practitioner blog | General usage-limit overview; secondary |
| https://stevekinney.com/courses/self-testing-ai-agents/structured-cli-output-as-pipeline-glue | Course/blog | Confirms `--bare` + `--json-schema` pipeline idiom; official S6 authoritative |
| https://www.datallmlab.com/blog/claude-structured-output.html | Practitioner blog | Structured-output JSON tutorial; official S2 authoritative |
| https://dev.to/mukundakatta/when-and-how-to-use-the-anthropic-batch-api-in-your-agent-5fgn | Community (DEV) | Batch-API usage walkthrough; official S1 authoritative |
| https://www.codewords.ai/blog/anthropic-batch-api | Practitioner blog | "50% cost" batch explainer; official S1 authoritative |
| https://code.claude.com/docs/en/cli-reference | Official doc | Referenced by S6 for the full flag list; not separately fetched (S6 covered the structured-output flags) |

## Recency scan (2025-2026)

Searched the last-2-year window (2025-2026) on all three topics (structured
outputs, CLI structured output, Max usage limits). Findings that
COMPLEMENT/SUPERSEDE the census's routing facts:

1. **Structured Outputs went GA and moved fields (2025-11 -> 2026).** The beta
   header `structured-outputs-2025-11-13` is NO LONGER required and `output_format`
   migrated to `output_config.format` (S2). Constrained decoding + 24h grammar
   caching is now the GA mechanism on Claude 4.5+. => C1-C6 + A1/A2 sit on a GA,
   not beta, API surface; no beta-gate risk. No census row contradicted.
2. **CLI `--json-schema` hardened in Claude Code v2.1.205 (2026)** (S6). Before
   v2.1.205 an invalid schema silently returned unstructured text; now it errors
   loudly. This directly de-risks the census's 75.5 note (ClaudeCodeClient emits
   `--json-schema` for dict + pydantic-class schemas): as long as the operator's
   `claude` binary is >= v2.1.205, a malformed schema fails loud rather than
   silently degrading — a real improvement over the state assumed when 75.5 shipped.
3. **Max usage limits changed May 6 2026 (supersedes S4's dating).** S4
   (truefoundry) dates the weekday peak-hours (5-11am PT) reduction to Mar 2026;
   newer sources (morphllm, WebSearch corpus) report that on **May 6 2026** the
   5-hour limits were DOUBLED for Pro/Max/Team/Enterprise AND the peak-hours
   reduction was REMOVED for Pro and Max Claude Code accounts. Routing-relevant:
   the CC/Max rail has MORE 5-hour headroom in mid-2026 than S4 implies, but the
   shared-bucket constraint (Claude Code + claude.ai + Cowork drain one pool) is
   unchanged — so a routing decision to push high-frequency Layer-1 (A1/A2) onto
   the Max rail still competes with the dev-harness's own budget.
4. **Batch pricing table now lists 2026 models** (S1): Fable 5 / Mythos 5 / Opus 5
   / Sonnet 5 (Sonnet 5 introductory $1/$5 through Aug 31 2026). All active models
   support batch — no model-availability gap for A5 if it is ever wired. No census
   row contradicted.

No 2025-2026 source CONTRADICTED an inherited census routing fact; the window
sources tightened dates and confirmed GA status.

## Inherited-census spot-check (signature-covering the internal half)

Per the caller's mandate, two census rows were re-verified against current code
before signing `gate_passed`:

- **C1 -> backend/services/meta_scorer.py:221** — CONFIRMED. `client = ClaudeClient(model_name=getattr(settings, "meta_scorer_model", "claude-haiku-4-5"), api_key=..., enable_prompt_caching=False)` at :220-225 (direct `ClaudeClient`, NOT via `make_client` => CC-rail flag never applies — census claim holds). Schema path confirmed at :218 `cleaned_schema = _strip_unsupported_schema_keys(MetaScorerBatch.model_json_schema())` fed to `generate_content(..., {"response_schema": cleaned_schema, "response_mime_type": "application/json"})` at :229-236 — matches the "pydantic-schema via dict (stripped)" note. haiku-4-5 default matches.
- **B1 -> backend/services/autonomous_loop.py:2454** — CONFIRMED. `if use_claude_code_route:` at :2446 -> `claude_code_invoke(prompt, max_tokens=200, timeout_s=120)` at :2453-2458 with NO model arg (CLI session default — census claim holds), metered via `_log_claude_code_call(..., agent="lite_trader", ...)` at :2462; else-branch raw SDK `client.messages.create(model=model_name, max_tokens=200, ...)` at :2471-2476; regex JSON extract `re.search(r'\{[^}]+\}', text)` at :2480; parse-fail => degraded HOLD marker `_parse_failed: True` (phase-70.4) at :2484-2489. Every B1 fact matches.

Both spot-checks passed, so the signature below covers the inherited internal
census as well as the external half completed in this leg.

## Envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 12,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 45,
  "coverage": {
    "audit_class": true,
    "rounds": 9,
    "dry_rounds": 2,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": true
  },
  "summary": "Anthropic call-site census for step 78.0. Internal half (inherited, 9-round audit-class sweep, coverage.dry=true) censuses ~30 distinct Anthropic/Claude call sites across tables A-K: Layer-1 pipeline (A1-A6, incl. DARK advisor_call + latent-TypeError BatchClient A5 + Haiku sentiment A6), lite paper trader dual-rail (B1-B2), six make_client-BYPASSING overlay services (C1-C6 -> CC-rail flag never applies to them), Layer-2 MAS (D1-D5, incl. tool-use D3 the CC rail cannot serve + dormant OpenClaw D5), ticket queue (E1), meta-evolution (F1-F3), harness planner (G1), RAG multimodal (H1), autoresearch nightly (I1), shell-CLI Max-rail sites (J1-J2), transport infra (K1 CC rail, K2 pending Max bridge, K3 GitHub Models). External half (this leg): 6 sources read in full — Batch API (A5 has NO Max-rail equivalent; 50% off, 24h expiry, max_tokens>=1), Structured Outputs (output_config.format, constrained decoding, GA — grounds A1/A2/C1-C6), Agent SDK + headless CLI (--output-format json + --json-schema + structured_output field = census K1 verbatim; --bare forces metered key), Max usage limits (shared 5h+weekly bucket, doubled May-6-2026), LiteLLM router prior-art (pyfinagent has NO central router -> per-site routing is why C1-C6 escape the flag). Recency scan superseded S4's peak-hours date and confirmed GA status of structured outputs + v2.1.205 --json-schema hardening. Spot-checks of C1 (meta_scorer:221) and B1 (autonomous_loop:2454) both CONFIRMED against code. No external source contradicted a census routing fact.",
  "brief_path": "handoff/current/research_brief_78.0.md",
  "gate_passed": true
}
```
