# Research Brief -- step 72.0.2 (tier: moderate)

**Step**: Fail-forward LLM routing -- standard-tier `claude-*` calls fall to the
Vertex-Gemini workhorse under a quality floor when the CC rail probe is dead AND
direct-API is credit-exhausted, instead of degrading lite -> HOLD ($0 book).
Ship FLAG-GATED DARK.

**Researcher**: Layer-3 (Workflow rail). **Date**: 2026-08-07.
**Status**: COMPLETE -- gate_passed=true.

---

## 0. Session log (write-first trail)

- [x] Read `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` in full.
- [x] Read masterplan step 72.0.2 + immutable criteria.
- [x] Internal audit: llm_client provider order / fail-closed path / rail probe.
- [x] External: multi-provider failover, quality floors, outage playbooks.
- [x] Envelope.

---

## 1. Internal code inventory (the Explore half)

| File | Anchor | Role | Status |
|---|---|---|---|
| `backend/agents/llm_client.py` | `:2044` `make_client()` | THE provider-order seam | live |
| `backend/agents/llm_client.py` | `:2114-2133` | CC-rail branch (claude-* + flag -> `ClaudeCodeClient`) | **the fail-forward seam** |
| `backend/agents/llm_client.py` | `:2136-2171` | Anthropic-direct branch + routing-breach `raise` at `:2144` | live |
| `backend/agents/llm_client.py` | `:2190-2192` | Vertex-Gemini fallback (`GeminiClient(model=vertex_model)`) | live, **None-trap** |
| `backend/agents/llm_client.py` | `:2195-2207` | terminal `raise ValueError` (no compatible key) | live |
| `backend/agents/claude_code_client.py` | `:85-183` | rail guard: probe gate + circuit breaker (`_RAIL_GUARD`, module global) | live |
| `backend/agents/claude_code_client.py` | `:735-741` | blocked -> `LLMResponse(text="", thoughts="rail_guard_skipped: ...")` | live |
| `backend/services/autonomous_loop.py` | `:393-427` | cycle-start `claude_code_health_probe` + `rail_guard_disable` + P1 | live |
| `backend/services/autonomous_loop.py` | `:2196-2213` | `_select_lite_analyzer(settings.gemini_model)` prefix dispatch | live |
| `backend/services/autonomous_loop.py` | `:2549-2551` | fabricated `HOLD/score=5/_parse_failed` placeholder | live (61.2 marks it) |
| `backend/services/autonomous_loop.py` | `:2216-2227` | `_fold_degraded_for_trading` (degraded -> dropped from decide_trades) | live, 61.2-gated |
| `backend/config/settings.py` | `:31` `gemini_model` | **the "standard tier"** -- default `"claude-sonnet-4-6"` | live |
| `backend/config/settings.py` | `:32` `deep_think_model` | phase-37.2 precedent (static default moved to Gemini) | live |
| `backend/config/settings.py` | `:176-198` | `paper_use_claude_code_route`, `claude_rail_breaker_threshold`, `claude_code_timeout_s`, `claude_code_empty_retry_max` | live |
| `backend/config/model_tiers.py` | `:52,:60` | `GEMINI_WORKHORSE="gemini-2.5-flash"`, `GEMINI_DEEP_THINK="gemini-2.5-pro"` | live (retires 2026-10-16) |
| `backend/agents/orchestrator.py` | `:625-659` | the ONLY caller that passes a real `GeminiModelBundle` to `make_client` | live |

### 1.1 STEP-PREMISE CORRECTION (contract must record)

The step text cites the provider-order seam as `backend/agents/llm_client.py:1983-2042`.
**That range is wrong in the current file**: `:1981-2041` is `BatchClient.poll()` /
`BatchClient.fetch()`. `make_client` begins at `:2044`; the CC-rail branch is
`:2114-2133`. Likewise the "78.1 comment cites llm_client.py:2163 raise" -- `:2163`
is now inside the phase-78.16 *comment block*; the actual routing-breach `raise` is
`:2144-2152`. Line numbers drifted (78.16 inserted ~20 comment lines). Re-derive
before quoting.

### 1.2 What ACTUALLY happens today when the rail is dead (the fail-closed path)

There is **no raise on the rail-dead path**. The chain is silent-empty, not
exception:

1. `autonomous_loop.py:395-411` -- cycle start, `claude_code_health_probe()` fails ->
   `rail_guard_disable(detail)` sets `_RAIL_GUARD.disabled_reason` + consumes the
   page latch; ONE P1 fires (`error_type="rail_down"`).
2. `make_client` is **unaffected**: `llm_client.py:2114-2128` returns a
   `ClaudeCodeClient` regardless of rail health. It never reads `rail_guard_status()`.
3. Every call: `claude_code_client.py:737-741` returns
   `LLMResponse(text="", thoughts="rail_guard_skipped: probe gate: ...")` with zero
   subprocess spawns.
4. The orchestrator retry-on-empty (`orchestrator.py:924-940`) **deliberately does not
   retry** `rail_guard_skipped` ("no calls through an open breaker" -- Fowler).
5. Lite path: `_run_claude_analysis` regex finds no JSON in `""` ->
   `autonomous_loop.py:2549-2551` fabricates `{"action":"HOLD","confidence":0,
   "score":5,"_parse_failed":True}`.
6. `_degraded_scoring_check` (`:2229`) counts it; with
   `paper_synthesis_integrity_enabled` ON, `_fold_degraded_for_trading` (`:2216`)
   drops it from `decide_trades`. Either way: **no trade, $0**.

The terminal `raise ValueError` at `:2195-2199` is reached only when the flag is OFF
and no `ANTHROPIC_API_KEY` -- a *different* failure (fresh checkout), not rail-dead.
The `:2144` routing-breach raise needs flag-ON **and** a failed `ClaudeCodeClient`
import **and** a live key. Neither is the 72.0.2 failure class.

### 1.3 Who is "standard tier"

`settings.gemini_model` (`settings.py:31`) is the standard-tier selector -- the field
name is legacy; its **default value is `"claude-sonnet-4-6"`**. Consumers:

- `orchestrator.py:652` `general_client` (enrichment + debate), `:659` `quant_exec_client`
- `autonomous_loop.py:1882` and `:1983` -> `_select_lite_analyzer(settings.gemini_model)`
  (prefix dispatch: `gemini-*` -> `_run_gemini_analysis:2691`, else -> `_run_claude_analysis:2381`)
- `autonomous_loop.py:3000`, `backtest/quant_optimizer.py:639`
- Separately, six C-block services pin their own `claude-haiku-4-5` default and call
  `make_client(..., None, settings, enable_prompt_caching=False)`:
  `meta_scorer.py:242`, `news_screen.py:288`, `macro_regime.py:527`,
  `pead_signal.py:300`, `analyst_narrative_scorer.py:156`, `call_transcript_gpr.py:135`.
  These are the phase-78.1 rail-routed callers and they ALSO go dark on rail-death.

`model_tiers.EFFORT_DEFAULTS` (`:318`) is a **Layer-2 role->effort map, not a
model-tier map** -- and `model_tiers.py:309-312` explicitly notes it is consumed only
at the `llm_client` effort seam. It does NOT select the standard-tier model; do not
route the fail-forward through it.

### 1.4 The Vertex-bundle None-trap (biggest design risk)

`make_client`'s Vertex leg is `GeminiClient(model=vertex_model)` (`:2190-2192`) --
it uses the **pre-built bundle passed by the caller**. Only `orchestrator.py:625-651`
passes real bundles. Every C-block service and `autonomous_loop.py:2764` pass
`vertex_model=None`. So a naive fail-forward that just rewrites the model string to
`gemini-2.5-flash` yields:

- `GEMINI_API_KEY` set -> branch 1 (`:2091-2106`) -> **AI Studio direct**, *not* Vertex;
- `GEMINI_API_KEY` unset -> branch 5 -> `GeminiClient(model=None)` -> a client with
  `self._model = None` -> failure at generate-time, i.e. **the same $0 outcome with
  extra steps**.

Criterion 1 says "run on **Vertex**-Gemini". The fail-forward therefore has to
CONSTRUCT its own `GeminiModelBundle` from ADC inside the seam (mirroring
`orchestrator.py:625-628`) rather than relying on the caller's `vertex_model`, or
explicitly accept the AI-Studio leg and say so in the live_check. Recommend:
build the bundle in the seam, ADC-based, so the branch is caller-independent.

### 1.5 Precedent inventory

- **No dynamic fail-forward exists.** `grep -rl "fail_forward|fail-forward" backend/`
  returns ZERO files. The only precedent is phase-37.2's **static default swap**
  (`settings.py:32`: `deep_think_model` default moved from `claude-opus-4-7` to
  `GEMINI_DEEP_THINK` "caused silent regression to Anthropic credit-exhaustion").
  Precedent = change the default, not reroute at runtime. 72.0.2 is the first
  runtime reroute -- design it from the literature, not from repo idiom.
- **Circuit-breaker idiom already exists and is good** (`claude_code_client.py:85-183`,
  Fowler/Azure per-cycle window, transition-only paging). The fail-forward must be a
  READER of that state, never a mutator.
- **Retry idiom exists and is correctly scoped** (`orchestrator.py:838-940`, single
  retry layer, `errored:` retryable / `rail_guard_skipped:` never). The fail-forward
  fills the gap the retry deliberately leaves.

---

## 2. External research

### 2.1 Search-query variants run (3-variant discipline)

| Variant | Query |
|---|---|
| current-year (2026) | `multi-provider LLM failover routing circuit breaker fallback model degradation 2026` |
| last-2-year (2025/2024) | `LLM fallback model quality floor cross-provider substitution structured output scoring 2025` |
| last-2-year (2025/2024) | `LLM provider outage postmortem playbook degrade to secondary model 2025 2024 production incident` |
| year-less canonical | `circuit breaker fallback graceful degradation microservices pattern` |

### 2.2 Read in full (7; counts toward the gate)

| # | URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|---|
| 1 | https://arxiv.org/html/2511.07585v1 | 2026-08-07 | preprint (finance) | WebFetch (arXiv HTML) | "LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows". INVERSE scale/determinism: Granite-3-8B + Qwen2.5-7B **100%** consistency at T=0.0; Llama-3.3-70B 75%; Mistral-Medium 56%; GPT-OSS-120B **12.5%** (p<0.0001). "SQL generation maintains 100% consistency at T=0.2, while RAG tasks show drift (25-75%)" (§4.3). "Consistency transfers between local and cloud deployments" (§4.1). |
| 2 | https://arxiv.org/html/2604.25359v1 | 2026-08-07 | preprint (benchmark) | WebFetch (arXiv HTML) | Structured Output Benchmark. "every model produces nearly perfect JSON, yet a sizeable fraction of the leaf values inside that JSON are wrong" -- best Value Accuracy only **83.0% on text**. Hardening rule: "if a response fails to parse, lacks a structured root, or violates the schema, all semantic scores are driven to zero." Recommends explicit coverage gates: "hard (floor = 0.95) for text". Schema-constrained decoding changes Value Accuracy only "-0.007 to +0.033". |
| 3 | https://arxiv.org/html/2512.16959v1 | 2026-08-07 | preprint (SLR) | WebFetch (arXiv HTML) | Systematic review of microservice recovery patterns. "Fallbacks (cached or approximate responses) maintain UX during partial outages" (§VI-D). "Pattern effectiveness depends on failure semantics; over-tight circuit-breaker thresholds reduce throughput" (T1). Bounded retries + breaker = best (P99 1100ms, 3% err) vs no-jitter backoff (17% err). **Explicit gap: no hysteresis/flapping/half-open tuning evidence in the reviewed corpus.** |
| 4 | https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker | 2026-08-07 | official doc | WebFetch | Closed/Open/Half-Open. Directly on point for 72.0.2: "an application might temporarily degrade its functionality, **invoke an alternative operation to try to perform the same task or obtain the same data**, or report the exception". Also: "rather than returning a failure and raising an exception, the Open state can return a default value that's meaningful to the application"; "a circuit breaker can fluctuate and reduce the response times of applications if it switches from the Open state to the Half-Open state too quickly" (the flapping warning); "Resource differentiation: be careful when you use a single circuit breaker for one type of resource if there might be multiple underlying independent providers." |
| 5 | https://futureagi.com/blog/what-is-llm-fallback-strategy-2026/ | 2026-08-07 | industry | WebFetch | The **quality-floor** source. Fallback triggers include a "quality-floor miss" alongside 5xx/429/timeout. "Model-downgrade without a quality floor degrades output silently on hard prompts. The cheaper fallback model is the right answer on the median prompt and the wrong answer on the tail." Enforcement = "the policy engine reads the rolling score per route per workload and downweights routes scoring below the floor". Pre-deploy validation = shadow mode + chaos injection. Observability = `gen_ai.fallback.reason/hop/route/score/mttr_ms`. |
| 6 | https://platform.claude.com/docs/en/api/errors | 2026-08-07 | official doc (Anthropic) | WebFetch (301 from docs.claude.com) | **402 `billing_error` DOES exist**: "There's an issue with your billing or payment information." Retry policy: "The official SDKs automatically retry transient failures (**such as connection errors, rate limits, and 5xx server errors**) with exponential backoff, twice by default, honoring the `retry-after` header." 402 is absent from that list -> not auto-retried. 429 `rate_limit_error`, 529 `overloaded_error`, 504 `timeout_error` also documented. |
| 7 | https://portkey.ai/blog/failover-routing-strategies-for-llms-in-production/ | 2026-08-07 | industry (gateway vendor) | WebFetch | Ordered `targets` fallback chains; "429 Too Many Requests (rate limit): Instantly reroute traffic to a secondary provider. 500-level errors: Retry on another provider." **[ADVERSARIAL-adjacent / negative evidence]**: explicitly provides NO framework for cross-provider model equivalence, NO flapping/cost-blowup mitigation, and admits an observability gap -- "Without centralized logs, it's hard to know when failover was triggered, how often, or what it cost." The gateway-vendor consensus is thinner than it looks. |

### 2.3 Identified but snippet-only (does NOT count toward the gate)

| URL | Kind | Why not fetched in full |
|---|---|---|
| https://dev.to/kuldeep_paul/adaptive-model-routing-and-fallback-logic-routing-around-llm-provider-outages-with-bifrost-4g3m | vendor blog | duplicate of #7's content, lower tier |
| https://www.getmaxim.ai/articles/retries-fallbacks-and-circuit-breakers-in-llm-apps-a-production-guide/ | industry | superseded by #5 (has the quality-floor mechanics) |
| https://www.getmaxim.ai/articles/failover-routing-strategies-for-llms-in-enterprise-ai-applications/ | industry | same author/site as above |
| https://neuralrouting.io/blog/llm-failover-high-availability-architecture | vendor | marketing-tier |
| https://www.truefoundry.com/blog/llm-failover-load-balancing-provider-outages | vendor | marketing-tier |
| https://nerdleveltech.com/llm-fallback-routing-ai-model-recall | blog | low authority |
| https://mixroute.ai/blog/handle-llm-api-failures/ | vendor | low authority |
| https://explore.n1n.ai/blog/multi-provider-llm-failover-high-availability-2026-06-22 | vendor | low authority |
| https://www.buildmvpfast.com/blog/llm-fallback-strategies-primary-model-secondary-model-2026 | blog | low authority |
| https://www.buildmvpfast.com/blog/building-with-unreliable-ai-error-handling-fallback-strategies-2026 | blog | low authority |
| https://futureagi.com/blog/evaluating-llm-structured-output-modes-2026/ | industry | superseded by #2 (peer-track benchmark) |
| https://futureagi.com/blog/evaluating-litellm-multi-provider-2026/ | industry | adjacent |
| https://www.researchgate.net/publication/397521379_LLM_Output_Drift... | mirror | same paper as #1 |
| https://theroadtoenterprise.com/blog/model-agnostic-ai-layer-fallbacks | blog | low authority |
| https://tianpan.co/blog/2026-04-19-ai-incident-response-playbook-llm-production | blog | outage-playbook context |
| https://tianpan.co/blog/2026-04-27-llm-postmortem-template-fields-sre-missed | blog | outage-playbook context |
| https://futureagi.substack.com/p/the-llm-incident-runbook-six-steps | industry | outage-playbook context |
| https://concepttocloud.com/news/how-to-improve-llm-uptime | blog | low authority |
| https://www.codersarts.com/post/debugging-incident-response-and-postmortem-for-llm-systems | blog | low authority |
| https://latitude.so/blog/llm-failure-modes-root-cause-analysis-guide | blog | low authority |
| https://medium.com/@josesousa8/enhancing-microservice-resilience-with-the-circuit-breaker-pattern-aa14c4660e83 | community | superseded by #4 |
| https://www.geeksforgeeks.org/system-design/what-is-circuit-breaker-pattern-in-microservices/ | community | superseded by #4 |
| https://www.geeksforgeeks.org/system-design/microservices-resilience-patterns/ | community | superseded by #3 |
| https://aerospike.com/blog/circuit-breaker-pattern/ | vendor | superseded by #4 |
| https://oneuptime.com/blog/post/2026-02-20-microservices-circuit-breaker/view | vendor | superseded by #4 |
| https://talent500.com/blog/circuit-breaker-pattern-microservices-design-best-practices/ | community | superseded by #4 |

**URLs collected: 33** (7 read in full + 26 snippet-only).

### 2.4 Recency scan (2024-2026) -- MANDATORY

Performed. Two dedicated last-2-year passes plus one 2026 pass (queries in
2.1). **Result: 4 findings in the window that materially change the design,
and they SUPERSEDE the pre-2024 canonical framing.**

1. The pre-2024 canon (Fowler/Hystrix/Azure) frames breaker-open as *fail
   fast + default value*. The 2026 LLM-gateway literature adds a distinct
   trigger class the canon does not have: **quality-floor miss** as a
   first-class failure alongside 5xx/429/timeout (source #5). That is a new
   requirement, not a restatement.
2. **arXiv:2511.07585 (Nov 2025)** inverts the naive "bigger model = safer
   fallback" intuition for FINANCIAL workflows: 120B scored 12.5% output
   consistency vs 100% for 7-8B at T=0.0. Model *size* is not the quality
   axis; *determinism under a fixed decode config* is.
3. **arXiv:2604.25359 (2026)** supplies the measurement that the quality
   floor must be built on: schema-validity is NOT quality ("nearly perfect
   JSON, yet a sizeable fraction of the leaf values inside that JSON are
   wrong"), best-in-class text Value Accuracy 83.0%.
4. **arXiv:2512.16959 (Dec 2025 SLR)** confirms a genuine literature GAP:
   no hysteresis / flapping / half-open tuning evidence in the reviewed
   corpus. So the anti-flapping design must lean on Azure's qualitative
   warning + pyfinagent's own per-cycle window, not on a cited number.

No 2024-2026 source contradicts the canonical circuit-breaker state machine;
they extend it.

### 2.5 Key findings (cited per claim)

1. **Fallback to an alternative provider is the doctrinally correct Open-state
   action, not an exotic one.** "an application might temporarily degrade its
   functionality, **invoke an alternative operation to try to perform the same
   task or obtain the same data**" (Azure Architecture Center, accessed
   2026-08-07, https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker).
   72.0.2 is textbook, not novel.
2. **A fallback without a quality floor is the named anti-pattern.** "Model-
   downgrade without a quality floor degrades output silently on hard prompts.
   The cheaper fallback model is the right answer on the median prompt and the
   wrong answer on the tail." (futureagi 2026, accessed 2026-08-07,
   https://futureagi.com/blog/what-is-llm-fallback-strategy-2026/).
3. **Schema-validity is NOT a quality floor.** "every model produces nearly
   perfect JSON, yet a sizeable fraction of the leaf values inside that JSON
   are wrong" (arXiv:2604.25359, accessed 2026-08-07). A floor that only checks
   "did it parse" is vacuous -- which is exactly the shape of pyfinagent's
   current `re.search(r'\{[^}]+\}', text)` check at `autonomous_loop.py:2543`.
4. **But parse-success IS the correct FIRST gate.** "if a response fails to
   parse, lacks a structured root, or violates the schema, all semantic scores
   are driven to zero" (arXiv:2604.25359 "hardening rule"). Floor = hard
   structural gate THEN a semantic gate.
5. **Determinism, not parameter count, is the finance-relevant quality axis.**
   7-8B models hit 100% consistency where a 120B hit 12.5% (arXiv:2511.07585
   §4.2, accessed 2026-08-07). This *qualifies* the repo's own memory that
   "8B-class local models scored 30-47% vs Gemini 92% on finance tasks" --
   those are different metrics (task accuracy vs run-to-run consistency). The
   substitution risk for `gemini-2.5-flash` (a hosted frontier-family model,
   not an 8B local) is therefore materially LOWER than the local-LLM verdict
   implies. Do not reuse the local-LLM rejection as evidence against this step.
6. **Retry and fallback are different tools; do not merge them.** "Circuit
   breakers protect services from cascading failures by tripping after repeated
   errors"; "Retries should be bounded, avoid non-idempotent operations"
   (arXiv:2512.16959 §VI-A/B). Azure: "the retry logic should be sensitive to
   any exceptions that the circuit breaker returns and stop retry attempts if
   the circuit breaker indicates that a fault isn't transient."
7. **402 `billing_error` is real and is NOT auto-retried.** Anthropic docs
   (accessed 2026-08-07) list `402 - billing_error` and scope SDK auto-retry to
   "connection errors, rate limits, and 5xx server errors ... twice by default".
   Criterion 2's premise HOLDS. Nuance for the contract: on the CC-rail the 402
   never surfaces as an SDK exception at all -- the rail is a CLI subprocess, so
   credit exhaustion arrives as a non-zero exit / empty envelope.
8. **Flapping warning is qualitative only.** Azure: "a circuit breaker can
   fluctuate and reduce the response times of applications if it switches from
   the Open state to the Half-Open state too quickly." The SLR found **no**
   quantitative half-open/hysteresis evidence (arXiv:2512.16959, §VI-B gap).
   Design implication: prefer pyfinagent's existing *latched per-cycle* window
   (no mid-cycle re-close) over inventing a hysteresis constant.
9. **Per-provider breaker separation is doctrine.** "Be careful when you use a
   single circuit breaker for one type of resource if there might be multiple
   underlying independent providers" (Azure). The fail-forward must NOT let a
   Gemini failure feed `_rail_guard_record_failure`.
10. **The gateway-vendor consensus is thin.** Portkey (accessed 2026-08-07)
    ships ordered fallback chains but offers **no** cross-provider equivalence
    framework, **no** flapping/cost mitigation, and concedes "Without
    centralized logs, it's hard to know when failover was triggered, how often,
    or what it cost." Treat vendor "just add a fallback array" advice as
    incomplete; the quality floor + observability are the hard parts.

### 2.6 Consensus vs debate

**Consensus**: breaker -> Open -> serve from an alternative; retries only for
transient/idempotent; alert on transition; never retry through an open breaker;
instrument which route served.
**Debate / unsettled**: (a) whether the fallback should be same-provider-cheaper
or cross-provider (#5 vs #7 differ on default hop order); (b) how to set the
quality floor numerically -- #2 proposes 0.95 text / 0.90 image; #5 uses a
rolling per-route score with no universal constant; (c) hysteresis tuning is
an acknowledged evidence gap (#3).

### 2.7 Pitfalls from the literature (mapped to this step)

| Pitfall | Source | Bites 72.0.2 as |
|---|---|---|
| Silent quality degradation | #5 | Gemini serves a plausible-but-worse score; nobody notices because the cycle looks healthy |
| Parse-only floor is vacuous | #2 | `re.search(r'\{[^}]+\}')` at `autonomous_loop.py:2543` would "pass" a garbage payload |
| Shared breaker across providers | #4 | Gemini failures inflating `_RAIL_GUARD.consecutive_failures` -> false P1s |
| Retry storms / double-billing | #3, #5 | fail-forward stacked on the 61.2 `claude_code_empty_retry_max` retry -> N x metered Gemini calls |
| Flapping on too-fast half-open | #4 | mid-cycle probe re-close bouncing traffic between rails within one cycle |
| Cost blowup unmeasured | #7 | Vertex is METERED; criterion 3 exists exactly for this |
| Non-idempotent retry | #3 | a fail-forward that re-runs an already-partially-executed trade decision |

---

## 3. Application to pyfinagent (contract-ready)

### 3.1 SECOND STEP-PREMISE CORRECTION -- the lite path BYPASSES `make_client`

The step frames the whole fix as a "provider-order seam" change in `llm_client.py`.
**That is necessary but NOT sufficient.** `_run_claude_analysis` -- the function that
actually emits the HOLD in the failure story -- never calls `make_client`:

- `autonomous_loop.py:2389` `import anthropic` (direct SDK)
- `autonomous_loop.py:2464` `use_claude_code_route = bool(getattr(settings, "paper_use_claude_code_route", False))`
- `autonomous_loop.py:2478` `client = anthropic.Anthropic(api_key=api_key) if not use_claude_code_route else None`
- `:2504/:2509/:2587/:2592` call `claude_code_invoke` directly.

A fail-forward implemented ONLY at `llm_client.py:2114` therefore leaves the
lite->HOLD chain **unchanged** and criterion 1 unsatisfied. Two seams are required.

### 3.2 The two seams

| Seam | Anchor | Covers |
|---|---|---|
| **A** | `backend/agents/llm_client.py:2114-2133` -- inside the CC-rail branch, BEFORE `return ClaudeCodeClient(...)` at `:2125` | orchestrator `general_client` (`:652`) + `quant_exec_client` (`:659`), the six 78.1 C-block services, `quant_optimizer.py:639`, `autonomous_loop.py:3000` |
| **B** | `backend/services/autonomous_loop.py:2196-2213` (`_select_lite_analyzer`) | the lite path (`:1882`, `:1983`) -- the actual HOLD producer |

Seam-B trap: `_run_gemini_analysis` re-reads `settings.gemini_model` itself at
`:2755` and **hard-raises** at `:2756-2762` ("standard model ... is not a Gemini
model") when it is `claude-*`. So flipping the dispatcher alone raises instead of
fixing. `_run_gemini_analysis` needs an explicit model override parameter (default
`None` = today's behaviour) so the flag-OFF path stays byte-identical.

Seam-A trap: see 1.4 -- build the `GeminiModelBundle` inside the seam from ADC
(mirror `orchestrator.py:625-628`), do not trust the caller's `vertex_model` (it is
`None` for every non-orchestrator caller).

### 3.3 Flag design

```
paper_rail_failforward_enabled: bool = Field(False, description="phase-72.0.2: ...")
paper_failforward_model: str = Field(GEMINI_WORKHORSE, description="...")
```
- Placement: `backend/config/settings.py` beside the rail knobs at `:176-198`.
- Name follows the live `paper_*_enabled` dark-ship family
  (`paper_data_integrity_enabled:60`, `paper_synthesis_integrity_enabled`,
  `paper_scale_out_enabled:35`). Env var `PAPER_RAIL_FAILFORWARD_ENABLED`.
- Description must state behaviour in BOTH states, that OFF is byte-identical,
  the METERED-cost consequence, and that promotion is an operator decision
  recorded in `live_check_72.0.2.md` (the `paper_data_integrity_enabled` field is
  the template for this wording).
- `paper_failforward_model` defaults to `GEMINI_WORKHORSE`, never a literal --
  the 2.5 family retires 2026-10-16 and `model_tiers.py:57-90` is the tripwire.

### 3.4 Quality floor -- recommended mechanism (two-stage, deterministic, $0)

**Stage 1 -- hard structural gate.** Adopt arXiv:2604.25359's hardening rule
verbatim in spirit: the fail-forward payload must parse to a dict AND carry the
required leaf keys with in-range values (`action` in {BUY,SELL,HOLD};
`confidence` int 0-100; `score` 1-10; non-empty `reason`). Any miss => this is
NOT a real score => hand back to the existing honest-degraded path (61.2), never
fabricate. Rationale: "if a response fails to parse, lacks a structured root, or
violates the schema, all semantic scores are driven to zero."

**Stage 2 -- deterministic semantic invariants (no extra LLM call).**
(a) Pin the decode config to `temperature=0.0, top_k=1` -- already the idiom at
`llm_client.py:2099` -- per arXiv:2511.07585's Tier-1 control ("enforce T=0.0 ...
fixed seeds"), because determinism (not parameter count) is the finance-relevant
axis.
(b) **Reject the degenerate signature the current fabrication uses**:
`confidence == 0` with an UPPERCASE recommendation is `_degraded_scoring_check`'s
own tell (`autonomous_loop.py:2229-2245`). A fail-forward result matching it must
be rejected, so a substituted answer can never masquerade as a real one.
(c) Stamp provenance on every fail-forward-served analysis --
`_failforward: True`, `_failforward_provider`, `_failforward_reason` (the
`rail_guard_status()` reason). This is the repo-local analogue of
`gen_ai.fallback.reason/hop/route/score` (#5) and is what makes criterion 1's
"produce real (non-degraded) scores" auditable rather than asserted.

**Explicitly REJECT an LLM-judge quality floor for this step.** #5's rolling
per-route score presumes a scored held-out eval corpus that pyfinagent does not
have for this task, and a judge call doubles metered spend on the exact path
where budget is the binding constraint (criterion 3). Record shadow-mode +
rolling-score as the phase-N follow-up, per #5's shadow-mode recommendation.

**Do NOT cite the repo's local-LLM verdict as evidence against this step.** That
memory ("8B-class scored 30-47% vs Gemini 92% on finance") measured task accuracy
of *local* 8B models; the substitute here is Gemini itself, the same workhorse
already trusted for `deep_think_model` since phase-37.2 and for
`_run_gemini_analysis`. Finding #5 above is the correct framing.

### 3.5 Isolation requirements (criterion 2)

The fail-forward is a strict READER of `rail_guard_status()`
(`claude_code_client.py:137-148`). It must NEVER call
`_rail_guard_record_failure/_success`, never change
`claude_rail_breaker_threshold`, never touch the probe at
`autonomous_loop.py:395-427` or its P1. Basis: Azure "Resource differentiation --
be careful when you use a single circuit breaker for one type of resource if
there might be multiple underlying independent providers." A Gemini failure
feeding the Claude breaker would corrupt the 66.1 alerting contract.

No new retry arm. `rail_guard_skipped` empties are already never-retried
(`orchestrator.py:924-940`, and `settings.py:196`'s own note), so the
fail-forward is a SINGLE substitution per call, not a retry hop -- consistent
with Azure ("the retry logic should ... stop retry attempts if the circuit
breaker indicates that a fault isn't transient") and arXiv:2512.16959 T3
(retry-storm amplification).

**402 verification**: Anthropic documents `402 - billing_error` ("There's an issue
with your billing or payment information") and scopes SDK auto-retry to
"connection errors, rate limits, and 5xx server errors ... twice by default".
402 is NOT in that set, so criterion 2's premise HOLDS. Contract nuance: on the
CC rail the 402 never surfaces as an SDK exception at all (CLI subprocess), so
the criterion is satisfied by *not adding* a retry, and by asserting the existing
direct-API path adds none.

### 3.6 Test strategy

**Seams available (all $0, no live calls):**
- `make_client(model, None, fake_settings)` + client-TYPE assertion -- exact idiom
  at `backend/tests/test_phase_78_1_c_block_rail.py:140-185` (`_S` stand-in class).
- `rail_guard_disable(reason)` / `rail_guard_reset(cycle_id)` are PUBLIC
  (`claude_code_client.py:115-135`) -- a test can drive rail-dead state with no
  subprocess and no monkeypatching of privates. `test_phase_66_1_rail_guard.py`
  is the precedent.
- `_select_lite_analyzer(name)` returns the FUNCTION uncalled -> assert function
  identity, no I/O.
- The quality-floor predicate MUST be extracted as a pure module-level function
  (pattern: `_fold_degraded_for_trading:2216`, `_degraded_scoring_check:2229`) --
  61.2's cycle-2 Q/A proved a source-scan test vacuous ("a comment-only module
  satisfied it"). Behavioural, not textual.

**Fixtures:** `_S` fake settings with `paper_use_claude_code_route=True`,
`anthropic_api_key="sk-ant-api-test-not-real"`, **`gemini_api_key=""`** (forces
the Vertex leg that criterion 1 names, not the AI-Studio leg), plus a stub
genai client/bundle so nothing dials out.

**Named mutations (each MUST turn a specific test RED):**
| ID | Mutation | Test that must fail |
|---|---|---|
| M1 | run the fail-forward branch with the flag `False` | `test_flag_off_returns_claude_code_client` (the byte-identity guard) |
| M2 | fire the branch when `rail_skipped=False and breaker_tripped=False` | healthy-path identity test |
| M3 | have the fail-forward call `_rail_guard_record_failure` | breaker-isolation test (criterion 2) |
| M4 | accept a payload with `confidence=0` + UPPERCASE action | quality-floor test |
| M5 | return `GeminiClient(model=None)` (the 1.4 trap) | "real, usable client" assertion |
| M6 | mutate the STUB: make the fake client return valid JSON unconditionally | floor test must still discriminate (per `feedback_mutation_test_guards_and_fixtures`) |
| M7 | drop the `_failforward` provenance stamp | live_check-evidence test |

Flag-OFF byte-identity must be asserted on client TYPE **and** constructor
kwargs -- phase-78.16 proved a silent kwargs drift is a real regression class on
this exact function.

### 3.7 Per-criterion mapping

| Criterion (verbatim intent) | What satisfies it |
|---|---|
| C1: flag ON + probe dead -> standard-tier runs on Vertex-Gemini with real scores; flag OFF byte-identical | BOTH seams (3.2) -- Seam A alone fails because the lite path bypasses `make_client` (3.1); bundle built in-seam (1.4); provenance stamp makes "non-degraded" measurable; M1/M2/M5/M7 |
| C2: breaker + probe alerting unchanged; no retry loop on 402 | isolation rules (3.5) + M3 + a `git diff`-scoped assertion that `claude_code_client.py` guard fns and the `:395-427` probe/alert block are untouched + Anthropic-docs citation |
| C3: cost note recorded; activation line states expected per-cycle cost | derive, do NOT assert: count standard-tier calls/cycle from `llm_call_log` and multiply by the published Vertex `gemini-2.5-flash` rate; state the derivation and its date. Put the line in the decision sheet AND `live_check_72.0.2.md`. (`feedback_measure_dont_assert_claims`) |

Verification command is only `ast.parse(llm_client.py)` -- it CANNOT prove any of
the three criteria. Do not treat a green command as evidence; the criteria are
carried by the tests + the live_check.

### 3.8 live_check capture plan

`handoff/current/live_check_72.0.2.md` must show, from ONE cycle:
1. `rail_guard_status()` / `summary["claude_rail_healthy"]=false` for that cycle_id;
2. the `[LLMClient] Routing ... -> Gemini (Vertex AI fallback)` log line (or the
   Seam-B dispatcher line) with the fail-forward provenance;
3. a BQ row from `pyfinagent_data.llm_call_log` with `provider='gemini'` +
   that `cycle_id` on a standard-tier agent;
4. the resulting analysis with a NON-null, NON-zero score and `_failforward=true`
   -- i.e. demonstrably not the `score=5/confidence=0` fabrication;
5. the per-cycle cost of (3) for criterion 3.

Because the flag ships DARK, the honest capture is an INDUCED one: operator-gated
single cycle with the flag ON and the rail forcibly disabled via the existing
public `rail_guard_disable()`. State explicitly that it was induced, and the
inducement mechanism. (The gate helper only checks file existence -- content
quality is on us.)

### 3.9 Overlaps the contract must scope around

1. **61.2 promotion pending** (`paper_synthesis_integrity_enabled`, operator ask
   #10). Orthogonal in intent -- 72.0.2 changes WHICH model serves, 61.2 changes
   what happens when NONE does -- but they share the SAME failure branch
   (`autonomous_loop.py:1996-2015`). Contract must: (a) NOT flip 61.2's flag;
   (b) test at least the two diagonal cells of the 2x2 (failforward x integrity);
   (c) record that promoting 72.0.2 SHRINKS the degraded-row population that
   61.2's 142/170 baseline was measured on -- that baseline must be re-derived,
   not reused, after this ships.
2. **4000.x CC-rail steps** (4000.1/4000.2/4000.3). 4000.3 needs a single-writer
   window and measures rail throughput; a fail-forward that reroutes OFF the rail
   mid-window would corrupt that measurement. Contract must state the flag stays
   OFF during any 4000.x measurement window (which is free -- OFF is
   byte-identical) and that 4000.x's rail-marker assertions are unaffected.
3. **78.1** already routes the C-block through `make_client`; that is what makes
   Seam A cover them. Do not re-plumb.
4. **model_tiers retirement** -- reference `GEMINI_WORKHORSE`, never the literal
   `"gemini-2.5-flash"` (2.5 family EOL 2026-10-16; tripwire at
   `model_tiers.py:57-90`).
5. **Away-ops `$0 metered` constraint** (`project_away_ops_plan`). Vertex is
   metered. Default-OFF satisfies it today; the activation line in the decision
   sheet is the operator's cost gate. Flag ON during an unattended away window
   without an operator token would breach the standing constraint -- say so.

---

## 4. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (**7**)
- [x] 10+ unique URLs total (**33**)
- [x] Recency scan (2024-2026) performed + reported (§2.4)
- [x] Full pages/papers read, not abstracts (arXiv HTML per the phase-29.7 chain)
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every module the caller named (llm_client
      provider order, rail probe + guard, model_tiers, C-block callers, Vertex
      path, lite->HOLD chain, flag conventions, test seams)
- [x] Contradictions / consensus noted (§2.6) -- incl. the vendor-consensus
      thinness (#7) and the SLR's acknowledged hysteresis gap (#3)
- [x] Claims cited per-claim with URL + access date
- Gap noted: no PBO/DSR-style statistical work exists on model-substitution
  effect on downstream trade P&L; the quality floor here is a *guard*, not a
  proof that Gemini-served scores are as profitable. That is a genuine residual
  risk the contract should name rather than paper over.

---

## 5. JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 26,
  "urls_collected": 33,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "Two step premises are wrong and both change the plan. (1) The cited seam llm_client.py:1983-2042 is BatchClient; make_client starts at :2044 and the CC-rail branch is :2114-2133. (2) The lite path that actually emits the HOLD bypasses make_client entirely (autonomous_loop.py:2478 builds anthropic.Anthropic directly), so a provider-order-only fix cannot satisfy criterion 1 -- a second seam at _select_lite_analyzer:2196 is required, plus a model-override param because _run_gemini_analysis hard-raises at :2758. Rail-dead never raises: it returns rail_guard_skipped empties that regex-fail into a fabricated score-5 HOLD (:2549). A naive fail-forward hits the Vertex None-trap (every non-orchestrator caller passes vertex_model=None). Literature: fallback-to-alternative-provider is textbook Azure Open-state behaviour, but a fallback WITHOUT a quality floor is the named anti-pattern, and schema-validity is NOT quality (arXiv:2604.25359). Recommend a two-stage deterministic $0 floor plus provenance stamping; keep the Claude breaker strictly read-only.",
  "brief_path": "handoff/current/research_brief_72.0.2.md",
  "gate_passed": true
}
```
