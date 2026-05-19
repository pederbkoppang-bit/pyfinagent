# Research Brief -- phase-31.0.2 Stage 2 Smoketest (lite-path subagent synthesis)

**Tier:** deep | **Effort:** max | **Date:** 2026-05-20
**Scope:** Stage 2 of 13-stage smoketest. For each ticker
[AAPL, MSFT, NVDA, JPM] returned by Stage 1, spawn a Claude Code
subagent that produces a 4-field JSON synthesis:
`{ticker, recommendation, final_score, risk_assessment, price_at_analysis}`.

The substitution rule is load-bearing: the production
`_run_claude_analysis` lite path calls `anthropic.Anthropic().messages.create()`
directly. This smoketest replaces those Anthropic API calls with
Claude Code subagent spawns via `Agent({subagent_type: "general-purpose", ...})`.

## Search-query composition (three-variant discipline)

| Variant | Topic | Sample query |
|---------|-------|--------------|
| 2026 frontier | LLM structured JSON output | "LLM JSON output reliability prompt engineering best practice 2026"; "Claude Code subagent JSON output regex parse general-purpose" |
| 2025 last-2-yr | LLM-as-judge robustness | "'LLM as judge' financial analyst recommendation buy hold sell paper 2025" |
| year-less canonical | JSON-only prompt patterns + JSON repair | "JSON repair LLM output post-processing library python" |

## Code-audit findings (file:line anchors) -- CONFIRMED

### Production lite path -- `backend/services/autonomous_loop.py`

`_run_claude_analysis` at lines **1288-1470**. The shape of the
returned dict (the EXACT shape Stage 2 subagent output must
ultimately match to satisfy `decide_trades`):

```python
return {
    "ticker": ticker,                          # str
    "_path": "lite",                           # str sentinel
    "recommendation": analysis["action"],      # "BUY" | "SELL" | "HOLD"
    "final_score": analysis["score"],          # 1-10 numeric
    "risk_assessment": {                       # dict
        "decision": ...,                       # "APPROVE_FULL" | "REJECT" | ...
        "reasoning": str,
        "reason": str,                         # back-compat alias
        "recommended_position_pct": float,
        "risk_level": str,
        "risk_limits": dict,
    },
    "price_at_analysis": current_price,        # float (USD)
    "analysis_date": "<ISO8601 UTC>",
    "total_cost_usd": 0.01,
    "full_report": {
        "source": model_name,
        "analysis": analysis,                  # nested inner trader dict
        "market_data": {...},
    },
}
```

**Inner trader-LLM prompt** at `autonomous_loop.py:1339-1359` demands:
```json
{"action": "BUY", "confidence": 75, "score": 7, "reason": "..."}
```
Parsed with `re.search(r'\{[^}]+\}', text)` at line 1372 (single-
level brace match -- nested objects in the trader inner JSON would
break; the risk-judge JSON uses `re.DOTALL` for nested matching at
line 1407).

### Consumer -- `backend/services/portfolio_manager.py::decide_trades`

Required fields read by `decide_trades` (lines 138-181):

| Field | Path in analysis dict | Default if missing | Used for |
|-------|----------------------|--------------------|----------|
| `ticker` | `analysis["ticker"]` | `""` | Per-position lookup |
| `recommendation` | `analysis["recommendation"]` | `"HOLD"` | Buy/sell gate (line 140-146); upper-cased |
| `risk_assessment` | `analysis["risk_assessment"]` | `{}` | Sizing + stop-loss |
| `risk_assessment.recommended_position_pct` | nested | None | Position size |
| `risk_assessment.decision` | nested | `""` | Logging at line 183-188 |
| `risk_assessment.risk_limits.stop_loss` or `stop_loss_pct` | nested | None | Stop derivation |
| `final_score` | `analysis["final_score"]` | `0` | Sort key (line 191) |
| `price_at_analysis` | `analysis["price_at_analysis"]` | `None` | Stop derivation |

**Stage 2 simplified contract (per user prompt):** the subagent
emits a **4-field** subset:
`{ticker, recommendation, final_score, risk_assessment, price_at_analysis}`.
The user spec calls `risk_assessment` a **string** ("1-2 sentences").
This DIVERGES from the production schema where `risk_assessment` is
a **dict** with nested keys. **This is intentional for Stage 2**:
Stage 2 verifies subagent SPAWN + JSON SHAPE; full risk-assessment
fidelity is downstream (Stage 4+ when the risk-judge call
substitution is exercised). The Stage 2 output schema is therefore
the **simplified contract** the user prompt specifies, NOT the
production contract. Downstream stages will reconcile by adapting
the simple-string `risk_assessment` into the dict form before
`decide_trades` receives it.

## Pass 1 -- Broad coverage (12+ sources read in full)

### A. Anthropic structured-output + subagent canonical patterns

| # | URL | Accessed | Kind | Key finding |
|---|-----|----------|------|-------------|
| 1 | https://www.anthropic.com/engineering/built-multi-agent-research-system | 2026-05-20 | Anthropic engineering blog | "Each subagent needs an objective, an output format, guidance on the tools and sources to use, and clear task boundaries." Subagents use 3+ tools in parallel (90% time savings). Tasks with heavy interdependencies NOT suitable for subagent decomposition. Multi-agent uses ~15x more tokens than single-agent chats. |
| 2 | https://platform.claude.com/docs/en/build-with-claude/tool-use | 2026-05-20 | Anthropic official docs | Canonical tool_use pattern: model emits `stop_reason: "tool_use"` + `tool_use` blocks with `input` matching `input_schema`. **Tip block: "Guarantee schema conformance with strict tool use -- Add `strict: true` to your tool definitions."** Opus 4.7 + `auto`/`none` tool_choice = 346 system-prompt tokens. |
| 3 | https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use | 2026-05-20 | Anthropic official docs (DEFINITIVE) | **Grammar-constrained sampling guarantees schema conformance.** "Setting `strict: true` on a tool definition guarantees Claude's tool inputs match your JSON Schema by constraining the model's token sampling to schema-valid outputs (a technique called grammar-constrained sampling)." Without strict: types may differ ("2" vs 2); with strict: "response always contains passengers: 2". Mandatory schema clauses: `additionalProperties: false`, all keys in `required`. **For Stage 2:** strict mode is the canonical answer -- BUT Stage 2 uses a Claude Code subagent (not direct API), so strict-mode does NOT apply at the subagent boundary. Must use prompt-discipline + post-hoc regex parse. |
| 4 | https://code.claude.com/docs/en/sub-agents | 2026-05-20 | Anthropic Claude Code docs | **Definitive subagent lifecycle:** "Subagents work within a single session" -- single-turn synchronous invocation. "Each subagent runs in its own context window with a custom system prompt, specific tool access, and independent permissions." Built-in `general-purpose` agent inherits all tools. **Output is the agent's final assistant message -- there is no `tool_use` schema-enforcement at the subagent boundary** (subagents invoked via in-session Agent tool, not the messages API with `tools:[]`). Stage 2 must rely on prompt discipline + regex parse OR a post-spawn JSON-repair step. |
| 5 | https://platform.claude.com/docs/en/build-with-claude/structured-outputs | 2026-05-20 | Anthropic official docs | **Two complementary features:** (1) JSON Outputs (`output_config.format` / `messages.parse()` with pydantic schema), (2) Strict Tool Use (`strict: true`). Both use grammar-constrained sampling. **Guarantees:** always valid JSON; no parse errors; type-safe; no retries. **Hard limits:** no recursive schemas; no `minimum`/`maximum`/`minLength`/`maxLength`; no `minItems > 1`; 20 strict tools per request max. **First request:** grammar compile latency; cached 24h. **Stop reasons that BREAK the guarantee:** `refusal` and `max_tokens`. |

### B. Adversarial cross-validation: format restrictions DEGRADE reasoning

| # | URL | Accessed | Kind | Adversarial finding |
|---|-----|----------|------|---------------------|
| 6 [ADVERSARIAL] | https://arxiv.org/html/2408.02442 | 2026-05-20 | Peer-reviewed arXiv preprint (Oct 14 2024, v3) | **DIRECT CONTRADICTION of strict-mode-is-best.** "Let Me Speak Freely?" (Tam et al.): format restrictions DEGRADE reasoning. **GSM8K (math reasoning):** GPT-3.5-Turbo text 76.6% -> JSON 49.3% (**-27.3pt**); Claude-3-Haiku 86.5% -> 23.4% (**-63.1pt**); LLaMA-3-8B 74.7% -> 48.9% (-25.8pt). **Classification tasks (DDXPlus, MultiFin):** JSON-mode IMPROVES (Gemini-1.5-Flash 41.6% -> 60.3%). **Implication:** Stage 2 lite path is a **classification (BUY/HOLD/SELL) + regression (final_score)** task -- both answer-space-bounded; per Tam et al. classification results JSON-mode helps not hurts. **Warning:** Claude-3-Haiku reasoning collapse (-63pt) -- Stage 2 must NOT include CoT reasoning inside the JSON schema. |
| 9 | https://arxiv.org/html/2510.02209v1 | 2026-05-20 | arXiv (Oct 2025) | **StockBench** benchmark: LLM trading agents. Action space: increase / decrease / hold (3-action -- maps to BUY/SELL/HOLD). JSON-formatted decisions. **Schema-error finding (CRITICAL FOR STAGE 2):** "thinking models exhibited higher frequency of schema errors -- they tend to overthink and produce more complex outputs." Plain-instruct models had fewer schema errors but more arithmetic errors. 14 models tested: GPT-5, O3, Claude-4-Sonnet + 11 open-weight. Top return: Kimi-K2 1.9%, Qwen3-235B 2.4% vs 0.4% passive baseline. **Implication:** Stage 2 should NOT enable extended thinking on the subagent. |

### C. JSON prompting + parsing best practice

| # | URL | Accessed | Kind | Key finding |
|---|-----|----------|------|-------------|
| 7 | https://dev.to/pockit_tools/llm-structured-output-in-2026-stop-parsing-json-with-regex-and-do-it-right-34pk | 2026-05-20 | Industry blog (Feb 2026) | Three-level reliability hierarchy: **L1 prompt-only** 80-95%, silent failures; **L2 function/tool_use** 95-99%, schema is "hint not constraint"; **L3 native constrained-decoding** 100% via finite-state-machine token masking. Validation pattern: Prompt -> Generate -> Validate -> Repair -> Parse. Stage 2 lite-path subagent operates at L1. Must implement repair step. |
| 10 | https://genaiunplugged.substack.com/p/structured-outputs-json-prompts-guide | 2026-05-20 | Industry blog | Four-layer JSON-prompt approach: (1) clear schema definition, (2) perfect example, (3) strict field rules per type, (4) validation instruction. **Anti-preamble phrases (use verbatim):** "Output ONLY the JSON object, nothing before or after"; "Start your response with { and end with }. No text outside the JSON object."; "Do not wrap the output in code fences or backticks"; "Return raw JSON without any markdown formatting." **Temperature:** 0.0-0.1 for JSON tasks. **Null handling:** never `null` -- declare 0 / "" / false / [] defaults. |
| 11 | https://github.com/mangiucugna/json_repair | 2026-05-20 | Open-source library (v0.59.10, May 14 2026; 162 releases; Python 3.10+) | **THE canonical JSON repair lib.** Handles: missing quotes / commas / brackets / comments, stray prose, truncated values, incomplete kv pairs, unescaped chars. Drop-in `json_repair.loads(text)` replaces `json.loads()`. Example: `'{"users":[{"name":"Ada","role":"admin",}],"ok":tru'` -> `{'users':[{'name':'Ada','role':'admin'}],'ok':True}`. **Stage 2 implication:** wrap response parse in `json_repair.loads()` with stdlib fallback. Production code at `autonomous_loop.py:1372` uses single-level regex `\{[^}]+\}` which fails on nested objects -- Stage 2 prototyping `json_repair` becomes investment-grade upgrade. |
| 12 | https://www.unpromptedmind.com/structured-output-claude-json-validation/ | 2026-05-20 | Industry blog | Three-tier Claude JSON approach: **(1) native tool_use** ~99.5% but +100-200 tokens overhead; **(2) manual prompt + Pydantic** ~95-98%, lower overhead, must strip markdown fences; **(3) regex extraction fallback** ~85-90% -- use `json-repair` lib not pure stdlib. Production recommendation: cascade tool_use -> manual -> regex with exponential backoff over 3 attempts. **For Stage 2:** subagent can't use tool_use; operate at tier 2 + tier 3 fallback. Plan for 3-attempt retry. |

### D. Finance-domain BUY/HOLD/SELL precedent

| # | URL | Accessed | Kind | Key finding |
|---|-----|----------|------|-------------|
| 8 | https://arxiv.org/html/2507.01990v1 | 2026-05-20 | arXiv survey (Jul 2025) | Survey of LLMs in financial investments. **Papasotiriou et al.** frame stock rating as ordinal classification (Strong Sell / Moderate Sell / Hold / Moderate Buy / Strong Buy) -- 5-class. GPT-4-32k on S&P 500. MAE metric. **MarketSenseAI** on S&P 100 produces BUY/SELL/HOLD signals + explanations -- 72% cumulative returns / 10-30% excess alpha. The BUY/HOLD/SELL trichotomy is canonical industry practice; pyfinagent matches. 5-class extension is downstream work. |
| 13 | https://arxiv.org/abs/2509.11420 | 2026-05-20 | arXiv (Sep 2025) Tauric Research | **Trading-R1**: SFT + RL three-stage easy-to-hard curriculum. Tauric-TR1-DB: 100k samples / 18mo / 14 equities / 5 data sources. Evaluated on 6 major equities + ETFs. "improved risk-adjusted returns and lower drawdowns compared to both open-source and proprietary instruction-following models." Generates "structured, evidence-based investment theses" -- supports the lite-path pattern of `recommendation + final_score + risk_assessment` text. |

### E. Claude Code-specific: subagent + Agent SDK structured output

| # | URL | Accessed | Kind | Key finding |
|---|-----|----------|------|-------------|
| 14 | https://code.claude.com/docs/en/agent-sdk/structured-outputs | 2026-05-20 | Anthropic Claude Code Agent SDK docs (DEFINITIVE) | **GAME-CHANGER for Stage 2.** The Claude Agent SDK exposes `output_format={"type":"json_schema","schema":...}` on `query()` calls. "SDK validates the output against it, re-prompting on mismatch. If validation does not succeed within the retry limit, the result is an error instead of structured data." Python: `output_format={"type":"json_schema","schema":FeaturePlan.model_json_schema()}`. Output appears in `message.structured_output`. Errors raise `error_max_structured_output_retries` subtype. **Limitations same as platform structured outputs:** no recursive schemas, no numeric constraints, simple regex only. **Stage 2 lite-path subagent CAN use this** -- it shifts the subagent from L1 (prompt-only ~80-95%) to L2/L3 (SDK-validated ~99%+) without writing repair code. |
| 15 [ADVERSARIAL] | https://github.com/anthropics/claude-code/issues/30030 | 2026-05-20 | Anthropic GitHub issue (closed, not-planned) | **Known reliability gap:** Claude Code Agent tool fails to parse JSON after empty stream response. Reproducible 100% on v2.1.63 / Linux. Root cause: full conversation context (incl. images) packaged into subagent request -> 4xx (likely 413) rejection -> non-JSON error body -> Bun JSON parser fails. Error signature: `[ERROR] Stream completed without receiving message_start event` -> `[ERROR] Failed to parse JSON`. **Workaround:** "Avoid using the Agent tool entirely. Use Bash / Grep / Glob / Read directly." **Critical for Stage 2:** the smoketest must NOT package large context into each subagent spawn -- pass minimal Stage-1 row + ticker only. Issue closed/stale = Anthropic accepts the boundary, fix is operator-side. |

### F. Empirical benchmarks of JSON reliability across LLMs

| # | URL | Accessed | Kind | Key finding |
|---|-----|----------|------|-------------|
| 16 | https://medium.com/@lyx_62906/which-llms-actually-produce-valid-json-7c7b1a56c225 | 2026-05-20 | Industry benchmark blog (Aug 2025) | Empirical benchmark of LLM JSON-validity across 200 runs/model, 6 scenarios (simple, nested arrays, single tool-call, multi tool-call, agent-style). **No retry layer.** Top reliability tier (98-100%): GLM-4.5, Grok-3 Mini, GPT-5 Nano. Major model numbers (Claude Sonnet/Opus, GPT-4, Gemini) were NOT published in detail in the article -- only category-level. **Key insight:** "Some cheap models can be surprisingly reliable at JSON" -- model size != JSON reliability. **For Stage 2:** the chosen subagent model (Sonnet 4.6 inherited from main, or Opus 4.7) is one of the largest/most capable Claude models -- JSON reliability expected >=95% in zero-retry; with SDK output_format retry tier >=99%. |
| 17 | https://arxiv.org/pdf/2601.06112 | 2026-05-20 | arXiv (Jan 2025) | **ReliabilityBench** by Aayush Gupta: three-dimensional eval framework. (1) **Consistency:** k-trial pass rates across multiple runs. (2) **Robustness:** adversarial input / perturbation handling. (3) **Fault tolerance:** behavior under degraded conditions. **Key methodology insight:** "most existing benchmarks focus on task accuracy under ideal conditions, while production systems must maintain reliability under stress." **For Stage 2:** Stage 2's 4-ticker smoketest is single-trial per ticker -- to harden, future stages should add k=3 trials per ticker and assert k-trial pass-rate >= floor (per ReliabilityBench framing). Not in scope for Stage 2 but cited as upgrade path. |

