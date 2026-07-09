# Research Brief — phase-67.6 (API request-shape guards for Fable 5 / Sonnet 5 + SDK pin)

Tier: moderate (caller-specified). Date: 2026-07-09/10. Researcher session (write-first, incremental).

## Question

Guard `ClaudeClient.generate_content` so model-family-incompatible params (sampling params, explicit thinking configs, legacy budget_tokens) are never sent to claude-fable-5 / claude-sonnet-5, with zero behavior change for currently-reachable models; reconcile the anthropic SDK pin (requirements ==0.87.0 vs installed 0.96.0 vs code-comment >=0.96.0); design the behavioral test module.

## Verbatim immutable success_criteria (from .claude/masterplan.json 67.6)

1. "backend/agents/llm_client.py sends NO sampling params (temperature/top_p/top_k) in request shapes for claude-fable-5, claude-sonnet-5, claude-opus-4-8, claude-opus-4-7 (strip-list extended; behavioral test proves it)"
2. "llm_client.py never sends legacy {type:enabled, budget_tokens} thinking to claude-fable-5 or claude-sonnet-5: fable-5 omits the thinking param entirely (always-on per Anthropic docs); sonnet-5 receives adaptive; behavioral tests prove both"
3. "backend/requirements.txt anthropic pin matches the installed-and-required SDK (exact pin 0.96.0, supply-chain-hardening rationale comment preserved); the documented-vs-pinned mismatch is gone"
4. "No behavior change for currently-pinned models: opus-4-8/4-7/4-6, sonnet-4-6, haiku-4-5 request shapes preserved and asserted by the same test module"
5. "Fresh Q/A PASS with the 67.1 gates applied to this diff (lint + runtime smoke over the changed files, output in the critique)"

Verification command: `bash -c 'source .venv/bin/activate && python -m pytest backend/tests/test_claude_request_shapes.py -q -x --timeout=60 && grep -q "anthropic==0.96.0" backend/requirements.txt'`

live_check: builder-level request-payload dumps for fable-5 + sonnet-5 shapes (no temperature, no budget_tokens, effort in output_config) + pip show anthropic matching pin.

## Read in full (7; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|---|---|---|---|---|
| https://platform.claude.com/docs/en/about-claude/models/migration-guide.md | 2026-07-09 | official doc | WebFetch (.md raw) | Fable 5: "`thinking: {type: \"disabled\"}` returns an error on `claude-fable-5`, and requests without a `thinking` field run with adaptive thinking"; "Manual extended thinking ... is not supported on `claude-fable-5` and returns a 400 error". Sonnet 5: "manual extended thinking ... and sampling parameters (`temperature`, `top_p`, `top_k`) set to non-default values are no longer accepted and return a 400 error" (two breaking changes vs 4.6). Opus 4.7+: enabled+budget_tokens 400s; adaptive off-by-default on Opus 4.7/4.8 (explicit `{type:"adaptive"}` required) |
| https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5 | 2026-07-09 | official doc | WebFetch | model ID `claude-sonnet-5`; adaptive ON by default (no thinking field -> runs WITH adaptive); `{type:"disabled"}` allowed (unlike fable); non-default sampling -> 400, "the default value (or omitting the parameter) is accepted"; NEW TOKENIZER ~+30% tokens (max_tokens budgets must be revisited); prefill 400s -> use output_config.format |
| https://platform.claude.com/docs/en/about-claude/models/introducing-claude-fable-5-and-claude-mythos-5 | 2026-07-09 | official doc | WebFetch | "Adaptive thinking is the only thinking mode on Claude Fable 5... applies whenever the `thinking` parameter is unset. `thinking: {\"type\": \"disabled\"}` is not supported"; raw CoT never returned (`thinking.display` summarized/omitted); refusal = HTTP 200 + `stop_reason:"refusal"` (llm_client :1666-1674 already handles); `fallbacks` param (beta) for server-side retry; 1M ctx / 128K out; $10/$50 |
| https://platform.claude.com/docs/en/build-with-claude/effort | 2026-07-10 | official doc | WebFetch | Effort supported on: Fable 5, Mythos 5, Opus 4.8, Mythos Preview, Opus 4.7, Opus 4.6, **Sonnet 5**, Sonnet 4.6, Opus 4.5 (Haiku 4.5 NOT listed). **xhigh available on: Fable 5, Mythos 5, Opus 4.8, Opus 4.7, Sonnet 5** — the llm_client :1507 xhigh guard (opus-4-8/4-7 only) is stale for fable/sonnet-5. Sonnet 5 default effort = high. "Setting `effort` to `\"high\"` produces exactly the same behavior as omitting the parameter" |
| https://platform.claude.com/docs/en/build-with-claude/structured-outputs | 2026-07-10 | official doc | WebFetch | "Structured outputs are generally available on the Claude API for Claude Fable 5, Claude Mythos 5, Claude Opus 4.8, Claude Mythos Preview, Claude Opus 4.7, Claude Opus 4.6, Claude Sonnet 5, Claude Sonnet 4.6, Claude Sonnet 4.5, Claude Opus 4.5, and Claude Haiku 4.5" -> extending `_fmt_eligible` (:1548) with fable-5/sonnet-5 is doc-backed |
| https://github.com/anthropics/anthropic-sdk-python/releases | 2026-07-09 | release notes | WebFetch | **v0.114.0 (2026-06-30): "add support for claude-sonnet-5"** (typed model literal); v0.116.0 latest (2026-07-02). Installed 0.96.0 predates typed sonnet-5/fable-5 literals — functionally irrelevant: `model` accepts arbitrary str, thinking/output_config are TypedDict pass-through (local probe below confirms) |
| https://raw.githubusercontent.com/anthropics/anthropic-sdk-python/main/CHANGELOG.md | 2026-07-09 | changelog | WebFetch | 0.87.0 = 2026-03-31; 0.96.0 = 2026-04-16 ("add claude-opus-4-7, token budgets and user_profiles"); beta advisor tool added 0.93.0 (2026-04-09) — all inside the 0.87->0.96 bump; fable-5 typed support landed ~0.108.0 |

## Snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched |
|---|---|---|
| https://help.apiyi.com/en/claude-opus-4-7-deprecated-parameters-guide-en.html | practitioner blog | corroborates temperature-400 fixes; official docs suffice |
| https://github.com/Comfy-Org/ComfyUI/issues/13923 | community issue | real-world example of exactly this trap class (node sends deprecated temperature -> 400) — validates P0 framing |
| https://docs.litellm.ai/blog/claude_fable_5 | vendor | day-0 fable support notes |
| https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-fable-5.html | vendor doc | Bedrock-specific; we use direct API |
| https://www.developersdigest.tech/blog/migrating-to-claude-fable-5, .../fable-5-api-production-patterns-rate-limits | blog | secondary to official migration guide |
| https://www.anthropic.com/news/claude-fable-5-mythos-5, /news/redeploying-fable-5 | official news | capability/redeploy context, no request-shape detail beyond docs |
| https://espressio.ai/blog/migrating-opus-4-8-to-claude-fable-5/, https://claudefable-5.ai/features/adaptive-thinking/, https://www.buildthisnow.com/blog/models/claude-fable-5-api-guide, https://help.apiyi.com/en/claude-fable-5-comeback-api-guide-en.html, https://aiwiz.uk/blog/claude-fable-5/, https://www.buildfastwithai.com/blogs/claude-fable-5-returns-july-2026-what-changed, https://explainx.ai/blog/is-fable-5-back-2026, https://thenewstack.io/anthropic-extends-fable-5/, https://www.forbes.com/sites/sandycarter/2026/07/07/claude-fable-5-extends-by-five-more-days-10-moves-to-make-now/ | blogs/news | migration/return coverage, redundant with official docs |
| https://platform.claude.com/docs/en/build-with-claude/extended-thinking, /docs/en/claude_api_primer, https://code.claude.com/docs/en/errors, https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-8, https://docoreai.com/best-settings-claude-api/ | official/blog | surfaced in recency scan; effort+migration docs already cover the claims |

Search-query variants run (research-gate 3-variant rule): (1) year-less canonical: "Anthropic Claude Fable 5 API migration thinking parameter temperature rejected"; (2) doc-scoped: "whats-new-claude-fable-5 ... platform.claude.com docs"; (3) current-year: "Claude API request shape 400 error temperature thinking budget_tokens gotchas 2026". Note: the topic (models launched June 2026) is genuinely too new for older canonical prior-art; the canonical precedent is the Opus 4.7 sampling-param removal, covered inside the migration guide.

## Recency scan (2024-2026)

Performed (query variant 3 + all read-in-full sources are 2026-era). Findings that supersede/extend the caller's assumptions:
1. **Sonnet-5 nuance:** the 400 is on *non-default* sampling values; explicit default (temperature=1) is accepted. Irrelevant to the fix (strip entirely per criterion 1 and per migration guide "safest migration path is to omit these parameters entirely") but tests should not assert "server would 400 on temperature=1".
2. **Sonnet 5 new tokenizer (~+30% tokens)** — not in 67.6's criteria, but any 67.4 Sonnet-5 exercise must revisit max_tokens budgets; flag in contract as a 67.4 hand-off note.
3. **SDK moved on:** sonnet-5 typed support landed in SDK 0.114.0 (2026-06-30); the criterion-mandated 0.96.0 pin predates it — works fine (str pass-through) but a future bump beyond 0.96.0 needs its own step (immutable criterion fixes 0.96.0; do not silently over-bump).
4. **Fable refusal path:** HTTP 200 + stop_reason "refusal" — llm_client.py:1666-1674 already returns the sentinel; no new work.
No findings contradict the planned guard shape.

## Internal code inventory

| File | Lines | Role | Status |
|---|---|---|---|
| backend/agents/llm_client.py | 1344-1620 (flow), 1349/1396 (temperature default 0.0 always set), 1444-1457 (thinking branch + temp=1), 1448 (adaptive allowlist — no fable/sonnet-5), 1451-1454 (legacy else = trap #1), 1467-1470 (strip tuple opus-4-8/4-7 only = trap #2), 1484-1514 (effort), 1507 (stale xhigh guard), 1548-1551 (_fmt_eligible — no fable/sonnet-5), 1599-1602 (create dispatch), 1643-1652 (retry re-uses kwargs), 1666-1674 (refusal sentinel), 1272-1278+1831 (">=0.96.0" comments) | THE change site | traps confirmed |
| backend/agents/llm_client.py | 396-430 (_check_cost_budget; env escape hatch `COST_BUDGET_HARD_BLOCK_DISABLED=1` at :411), 1229 (_opus_adaptive_max_tokens — opus-prefix no-op elsewhere), 1257 (`__init__(model_name, api_key, enable_prompt_caching=True)`), 1704-1752 (response parsing: needs .content blocks w/ .type/.text, .usage.input_tokens/output_tokens/cache_*, .stop_reason), 1756-1769 (log_llm_call inside try — patch to no-op in tests) | test-seam facts | verified |
| backend/agents/multi_agent_orchestrator.py | 1089-1116 | SEPARATE messages.create site: opus-4-8/4-7 -> adaptive; ELSE -> `thinking={"type":"enabled","budget_tokens":2048}, temperature=1` UNCONDITIONALLY per tool-loop turn (no ENABLE_THINKING gate) | **adjacent trap — fires the instant any mas_* role repins to sonnet-5/fable-5.** Outside 67.6's immutable criteria (they name llm_client.py only); MUST be flagged in contract as residual scope decision |
| backend/agents/orchestrator.py | 101-125 (Critic 8192 / Synthesis 4096 budgets), 806-816 (gate) | thinking producer; model = settings.deep_think_model | gated by enable_thinking (default False, settings.py:47) |
| backend/agents/debate.py / risk_debate.py | 67-71 / 63-71 | produce `{"type":"enabled","budget_tokens":N}` into config -> generate_content | reachability: enable_thinking + standard-tier model = settings.gemini_model (default claude-sonnet-4-6, settings.py:30; UI-selectable) — trap fires if operator selects sonnet-5 |
| backend/config/model_tiers.py | 55-103 (_BUILD_TIER: opus-4-8 x3, sonnet-4-6 x3, haiku-4-5 x1, gemini x3 — post-07-08 repin, NO fable/sonnet-5 pins live), 212-226 (EFFORT_SUPPORTED_MODELS has claude-fable-5:218, **no claude-sonnet-5**), 268-282 (MODEL_EFFORT_FALLBACK has fable xhigh:273, **no sonnet-5**) | effort registry | sonnet-5 addition REQUIRED for live_check (else model_supports_effort=False -> effort dropped -> no output_config in dump) |
| backend/services/ticket_queue_processor.py | 168-177 | opus-4-8 / sonnet-4-6 pins; rides claude-code CLI rail (:205) not ClaudeClient | out of blast radius |
| backend/agents/claude_code_client.py | 264-271 (argv: --print/--output-format/--disallowedTools/--append-system-prompt/--json-schema), 464-468 (supports_thinking=False) | CLI rail | CONFIRMED never sends temperature/thinking — traps are direct-API-rail only |
| backend/requirements.txt | 39 | `anthropic==0.87.0  # exact pin: supply-chain hardening (phase-3.7.6; CVE-2026-34450/34452 fix)` — only anthropic pin repo-wide (grep) | the lie to fix |
| backend/tests/ | dir listing | NO existing request-shape test; newest template = test_agent_definitions_classification.py (67.2); test_claude_code_client.py covers CLI rail only | new module required |

### SDK probes (venv, no network — 2026-07-09)

- `pip show anthropic` -> **0.96.0**; `anthropic.__version__` == "0.96.0". `pip check`: only pre-existing gpt-researcher/numpy complaint — anthropic deps (httpx 0.28.1, pydantic 2.12.5) satisfied; **no conflicting pins**.
- `inspect.signature(Messages.create)`: **`output_config` present, `thinking` present**.
- `anthropic.types.OutputConfigParam` = `{effort: Literal['low','medium','high','xhigh','max'], format: JSONOutputFormatParam}` — **output_config.format + all 5 effort levels supported at 0.96.0**.
- `ThinkingConfigParam` = Enabled | Disabled | **Adaptive** union — adaptive typed at 0.96.0.
- `anthropic.types.beta`: **BetaAdvisorTool20260301Param present** (advisor_20260301 confirmed, beta namespace); **BetaUsage.model_fields contains `iterations`** (usage.iterations confirmed, beta namespace; stable `Usage` does NOT have it — callers must use beta endpoint to see it).
- Typed model literals for sonnet-5/fable-5 absent at 0.96.0 (added 0.114.0/0.108.0) — non-blocking: `model` is a str union, dicts pass through.

### Pin-downgrade time bomb (strengthens P0)

Installed 0.96.0 vs pinned ==0.87.0 means ANY `pip install -r backend/requirements.txt` (e.g. the nightly maintenance install per cron runbook) would DOWNGRADE to 0.87.0, where `messages.create` has no `output_config` kwarg -> TypeError on every effort-carrying Claude call. Reconciling the pin removes a live regression path, not just a doc mismatch.

## Consensus vs debate (external)

Full consensus across official docs; zero contradictions. Community evidence (ComfyUI #13923) shows the identical trap class shipping 400s in production elsewhere. Only nuance: "non-default value" wording (sonnet-5/opus) vs blanket strip — resolved in favor of omission (official migration guidance + immutable criterion 1).

## Pitfalls (from literature + code)

1. Fable: do NOT map budget_tokens->adaptive; OMIT the thinking key entirely (`disabled` also 400s).
2. Sonnet-5 without thinking field runs WITH adaptive by default -> thinking+text share max_tokens; plus +30% tokenizer. Not a 67.6 criterion; hand to 67.4.
3. `:1457 temperature=1` override fires inside the thinking branch — strip must stay AFTER it (already does) and the extended tuple then covers fable/sonnet-5.
4. Prefix collision check: "claude-sonnet-5" does not collide with "claude-sonnet-4-6"/"claude-sonnet-4-5" under startswith. Safe.
5. Legacy else-branch must SURVIVE for opus-4-5/3-7-class models (operator UI override can still route them). Don't invert to default-adaptive.
6. Retry path (:1643) copies kwargs — guards automatically propagate; test doesn't need a separate retry case.
7. mas orchestrator :1089-1116 is a second, UNGATED trap site not covered by the criteria — decide scope in contract (recommend: fix in same diff as a flagged bonus or open follow-up; Q/A will notice).

## Recommendation (concrete)

**Guard shape (minimal diff, house idiom = inline startswith tuples; no new helper):** in `generate_content`:
1. `:1447-1454` — insert fable as first branch: `if model_id.startswith("claude-fable-5"): pass  # adaptive always-on; any explicit thinking config 400s` ; add `"claude-sonnet-5"` to the adaptive elif tuple; keep legacy else verbatim.
2. `:1467` — extend strip tuple to `("claude-fable-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7")`.
3. `:1507` — extend xhigh-allowed tuple identically (effort doc: xhigh GA on fable-5/sonnet-5/opus-4-8/4-7); without this, fable's xhigh fallback (model_tiers:273) is spuriously downgraded to high. No current-model impact.
4. `:1548` — add `"claude-fable-5", "claude-sonnet-5"` to `_fmt_eligible` (structured-outputs doc verbatim list). No current-model impact.
5. model_tiers.py — add `"claude-sonnet-5"` to EFFORT_SUPPORTED_MODELS and `("claude-sonnet-5", "high")` to MODEL_EFFORT_FALLBACK (doc default = high). **Load-bearing for the live_check** (else sonnet-5 dump has no output_config.effort). Fable entries already exist (:218, :273).
6. llm_client.py:1272-1278 comment — no change needed (already says >=0.96.0, now true).

**Pin line:** `anthropic==0.96.0            # exact pin: supply-chain hardening (phase-3.7.6 discipline; CVE-2026-34450/34452 fix; phase-67.6 reconciled 0.87.0->0.96.0 to the installed+code-required SDK)` — preserves the rationale per criterion 3 and keeps the verification grep (`anthropic==0.96.0`) green. No pip install needed (venv already 0.96.0); do NOT bump past 0.96.0 (immutable criterion pins it exactly).

**Test module (`backend/tests/test_claude_request_shapes.py`; template: 67.2's test_agent_definitions_classification.py):**
- Seam: `monkeypatch.setattr(ClaudeClient, "_get_client", lambda self: fake)` — client is built inside generate_content (:1370) so class-level patch suffices. Fake exposes `.messages.create(**kw)` AND `.beta.messages.create(**kw)` (betas path :1599), appends kwargs to a list, returns `SimpleNamespace(content=[SimpleNamespace(type="text", text="ok", citations=None)], stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0), _request_id="test")`.
- Hermeticity: `monkeypatch.setenv("COST_BUDGET_HARD_BLOCK_DISABLED", "1")` (existing escape hatch :411) + monkeypatch `backend.services.observability.log_llm_call` to no-op (imported inside the function at :1756, so patching the module attribute works).
- Construction: `ClaudeClient(model_name=<id>, api_key="test-key")`; call `generate_content("hi", {...})`; assert on captured kwargs.
- Matrix (one param-driven test per assertion group): fable-5 {plain, thinking-budget-4096} -> no temperature/top_p/top_k, "thinking" not in kwargs, output_config.effort present; sonnet-5 {plain, thinking-budget} -> no sampling params, plain has no thinking key, budget case has thinking == {"type":"adaptive"}, effort=="high" present; opus-4-8/4-7 {plain, thinking} -> no sampling params, adaptive, effort xhigh (preserved); opus-4-6 / sonnet-4-6 / haiku-4-5 {plain, thinking} -> temperature PRESENT (0.0 plain / 1 thinking), adaptive on request, effort high/medium/absent respectively (preserved); legacy (claude-opus-4-5 or claude-3-7-sonnet) thinking -> {"type":"enabled","budget_tokens":N} + temperature==1 preserved; one schema case per family group asserting output_config.format presence for the eligible tuple.
- live_check generation falls out of the same seam: dump `json.dumps(captured_kwargs)` for the fable-5 and sonnet-5 cases + `pip show anthropic`.

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (7; 5 official Anthropic docs + SDK releases + changelog)
- [x] 10+ unique URLs total (28 collected across 3 search variants + direct fetches)
- [x] Recency scan (last 2 years) performed + reported (topic is 2026-native; scan section above)
- [x] Full pages read (not snippets) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (llm_client, model_tiers, mas orchestrator, orchestrator/debate/risk_debate producers, ticket queue, CLI rail, settings, requirements, tests)
- [x] Contradictions/consensus noted (none contradicting; one nuance on "non-default" wording)
- [x] Claims cited per-claim

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 21,
  "urls_collected": 28,
  "recency_scan_performed": true,
  "internal_files_inspected": 12,
  "report_md": "handoff/current/research_brief_67_6.md",
  "gate_passed": true
}
```
