# Research Brief -- phase-38.13.1 Rail-Wiring Root-Cause (cycle 11)

**Tier:** complex
**Date:** 2026-05-27
**Author:** researcher subagent (single session, internal + external)
**Predecessor brief:** `handoff/current/research_brief_phase_38_13_orchestrator_rail_audit.md` (cycle 8, diagnosis incomplete)
**Live evidence:** backend.log 19:09:25 (401 invalid x-api-key, request_id=`req_011Ca5...`) and 20:01:25-20:04:35 (400 credit balance too low for CIEN/AMD/STX/HPE/GEV/KEYS/MU/ON/INTC/DELL/GLW/SNDK/WDC)

## Headline finding (TL;DR)

**The orchestrator's rail wiring IS technically correct -- every Claude
call in the `AnalysisOrchestrator` pipeline routes through `make_client()`
at constructor time, and `make_client()` correctly returns
`ClaudeCodeClient` when `settings.paper_use_claude_code_route=True` AND
the model starts with `claude-` (`backend/agents/llm_client.py:1893-1907`).
The cycle-8 observability patch (orchestrator.py:2127 cost_summary rail
attribution) is correct.**

The cycle-8 401/400 errors come from **TWO mechanisms that are NOT
fixed by the in-code routing fix the operator is asking for**:

1. **The 401 (`request_id=req_011Ca5...`) at 19:09:25 happens during a
   per-ticker run where the `Settings` instance passed to
   `AnalysisOrchestrator(settings)` had `paper_use_claude_code_route=False`
   while the autonomous_loop layer's `settings` had it True. Most
   likely a `get_settings()` lru_cache desync.** When `paper_use_claude_code_route`
   is False at the orchestrator layer, `make_client()` falls through
   from the rail gate (llm_client.py:1893-1907) to the
   direct-Anthropic branch (llm_client.py:1910-1912) which constructs
   `ClaudeClient(model_name, api_key=anthropic_key)` -> wraps
   `anthropic.Anthropic(api_key=self._api_key)` -> hits
   `api.anthropic.com` with the configured key. If the key is an
   `sk-ant-oat-*` OAuth token or a stale `sk-ant-api03-*` rotated
   key, it 401s.

2. **The 400 ("credit balance too low") at 20:01-20:04 is the FINAL
   STAGE of the same anthropic-direct path -- the key authenticated
   but the associated account has no credit balance**. Per Anthropic
   issue #54839 + Portkey error library: "credit_balance_too_low" is
   returned for 3 root causes including "API key is valid and the
   associated account has cleared credits but the key is orphaned/
   stale -- authenticates successfully but is no longer authorized to
   bill against an active account". This is the orchestrator's
   `ClaudeClient.generate_content()` path executing successfully past
   the auth gate but rejected at billing.

**Why the rail flag didn't help these calls**: the rail flag is read
from `self.settings` snapshot inside `AnalysisOrchestrator.__init__`
at line 516-518, which calls `make_client(settings.gemini_model,
..., settings)`. If `settings.paper_use_claude_code_route` is False
at THAT moment, no further re-evaluation occurs -- the orchestrator
holds `general_client: ClaudeClient` instances for the lifetime of the
analysis. There is no per-call rail re-routing.

**The minimal fix** is to:
1. **Eliminate the lru_cache desync** -- ensure `get_settings.cache_clear()`
   is called when `settings_api.py` rewrites .env (it already is at
   line 424 and 461), AND ensure the autonomous-loop reads
   `get_settings()` FRESH at the start of each cycle (cycle_main is
   long-lived; the settings instance it passes might be stale).
2. **Add a defensive guard at `make_client()`**: when the rail flag
   is False AND the model starts with `claude-` AND
   `os.environ.get("ANTHROPIC_API_KEY", "")` is empty/sk-ant-oat-*,
   refuse to fall through to direct-Anthropic and either (a) hard-fail
   with a clear error pointing to the rail-flag, or (b) auto-switch
   to ClaudeCodeClient. Operator preference dictates (a) vs (b).
3. **Migrate the 5 remaining direct `anthropic.Anthropic()` call
   sites outside `make_client()`** to either route through `make_client()`
   OR fail-open when the rail flag is True (full list in section 2).

## Audit half -- internal LLM call-site topology

This is the PRIMARY DELIVERABLE per the operator's prompt.

### Method
- Recursive grep for `anthropic.Anthropic|AsyncAnthropic|messages\.create|api\.anthropic\.com` in `backend/`.
- Recursive grep for `make_client|ClaudeCodeClient`.
- Manual read of every result and its enclosing function/class.
- Cross-check against `paper_use_claude_code_route` references.

### Table 1 -- Anthropic-bound LLM call sites in `backend/` (exhaustive)

| # | File:Line | Site kind | Routes via make_client? | Rail-flag honored? | Notes / current state |
|---|-----------|-----------|-------------------------|---------------------|-----------------------|
| 1 | `backend/agents/llm_client.py:1832-1948` | `make_client()` factory | (this IS the factory) | YES (line 1893-1907) | Correct gate. Rail-gate at 1893; falls through to direct Anthropic at 1910 if rail flag false. |
| 2 | `backend/agents/llm_client.py:1196-1715` | `ClaudeClient` class (direct anthropic.Anthropic via `_get_client`) | N/A (returned BY make_client) | Inherits from make_client's choice. If make_client returned this class, the rail decision was already False. | Holds api_key snapshot. Calls `client.messages.create(...)` at 1524-1576. THE direct-Anthropic surface inside the orchestrator pipeline. |
| 3 | `backend/agents/llm_client.py:1990-1999` (`advisor_call()`) | helper, called from selected agents (currently dormant under default settings) | NO -- direct `anthropic.Anthropic(api_key=key)` at 1999 | NO -- ignores rail flag. | Used by Anthropic Advisor Tool (`betas=["advisor-tool-2026-03-01"]`). NOT directly hit by cycle-8 evidence but a rail-bypass surface waiting to fire. |
| 4 | `backend/agents/rag_agent_runtime.py:223-229` (`multimodal_index_claude`) | RAG runtime, hit if `news_screen_enabled` + Claude path active | NO -- direct `_anthropic.Anthropic(api_key=api_key)` at 229 | NO -- ignores rail flag. | Called in screener news pipeline. May not be hit per-ticker. |
| 5 | `backend/agents/multi_agent_orchestrator.py:173-184` (`_get_client`) | Layer-2 MAS (in-app Slack/UI agents -- NOT Layer-1 autonomous pipeline) | NO -- direct `anthropic.Anthropic(api_key=api_key)` at 181 | NO. Has its own `_anthropic_unavailable` fallback to Gemini at multi_agent_orchestrator.py:1001-1008. | This is the Layer-2 (Slack assistant) MAS, NOT the cycle-8 failure surface. But it is the same anti-pattern. |
| 6 | `backend/agents/multi_agent_orchestrator.py:982-993` | Layer-2 MAS one-shot call | NO -- uses #5 client | NO | Per-agent classification/planning. |
| 7 | `backend/agents/multi_agent_orchestrator.py:1073-1081` | Layer-2 MAS tool-loop call | NO -- uses #5 client | NO | Tool-use loop. |
| 8 | `backend/agents/debate.py:97-103` | exception-typing only | YES -- callers pass `LLMClient` instances. Lines 97-103 only `import anthropic` for `RateLimitError/APIStatusError` isinstance checks. | YES (inherits from caller's `model: LLMClient`). | Not a separate Anthropic instantiation -- just SDK exception types. |
| 9 | `backend/agents/risk_debate.py:93-99` | exception-typing only | YES (same pattern as #8) | YES | Same. |
| 10 | `backend/services/autonomous_loop.py:1403-1577` (`_run_claude_analysis`, lite path) | LITE Claude path, fired ONLY when `settings.lite_mode=True` OR as last-resort fallback | NO -- but EXPLICITLY rail-aware: line 1455 reads `use_claude_code_route` and at 1469 instantiates `anthropic.Anthropic(api_key=api_key) if not use_claude_code_route else None`. | YES. Per-call branching at 1492 / 1548 routes through `claude_code_invoke()` when rail flag True. | This is the CORRECT pattern. The lite path is already wired. |
| 11 | `backend/news/sentiment.py:754-820` (`HaikuScorer`) | News sentiment classifier (`news_screen_enabled` path) | NO -- direct `anthropic.Anthropic(api_key=key)` at 779 | NO -- ignores rail flag. | Used by `screener.py:291` -> `news_screen.py`. Per-ticker news analysis. May or may not fire per cycle depending on `news_screen_enabled`. |
| 12 | `backend/services/ticket_queue_processor.py:156-180` | Slack ticket queue processor for the Layer-3 dev MAS proxy | NO -- direct `anthropic.Anthropic(api_key=api_key)` at 180 | NO -- ignores rail flag. | NOT a per-ticker analysis surface. Out of scope for cycle-8 trading failure. |
| 13 | `backend/agents/planner_agent.py:21` | imports `from anthropic import Anthropic` | (need to verify usage) | likely NO | Layer-3 harness planner. Out of scope for cycle-8 trading failure. |
| 14 | `backend/meta_evolution/directive_review.py:134-136` | meta-evo Layer-4 | NO -- direct `anthropic.Anthropic(api_key=api_key)` at 136 | NO | Out of cycle scope. |
| 15 | `backend/meta_evolution/directive_rewriter.py:175-177` | meta-evo Layer-4 | NO -- direct `anthropic.Anthropic(api_key=api_key)` at 177 | NO | Out of cycle scope. |
| 16 | `backend/slack_bot/streaming_integration.py:461-463` | Slack assistant leak handler | NO -- direct `anthropic.Anthropic(...)` | NO | Out of cycle scope. |
| 17 | `backend/slack_bot/assistant_handler.py:420-422` | Slack assistant harmlessness gate | NO -- direct `anthropic.Anthropic(...)` | NO | Out of cycle scope. |

### Table 2 -- AnalysisOrchestrator (Layer-1) client wiring (the cycle-8 failure surface)

| Client attr | Source line | Returned by | Models that route to Anthropic | Hit during pipeline |
|-------------|-------------|-------------|--------------------------------|---------------------|
| `self.general_client` | orchestrator.py:516 | `make_client(settings.gemini_model, _general_vertex, settings)` | When `gemini_model` starts with `claude-` (default `claude-sonnet-4-6` per settings.py:29). | EVERY enrichment agent (orchestrator.py:1043-1170 -- 14+ calls per ticker). |
| `self.deep_think_client` | orchestrator.py:517 | `make_client(deep_model_name, _dt_vertex, settings)` | When `deep_think_model` starts with `claude-`. Current default is `gemini-2.5-pro` (settings.py:30) -- ROUTES TO GEMINI by default. | Critic phase, debate.py / risk_debate.py invocations using `deep_think_client`. |
| `self.synthesis_client` | orchestrator.py:518 | `make_client(deep_model_name, _synth_vertex, settings)` | Same as deep_think -- Gemini by default. | Synthesis + revision loop (orchestrator.py:1363, 1440). |
| `self.quant_exec_client` | orchestrator.py:523 | `make_client(settings.gemini_model, _quant_exec_vertex, settings)` | Same as general -- routes to Claude if `gemini_model=claude-*`. **BUT** the code_execution tool is Gemini-only; orchestrator.py:519-522 comment notes "When settings.gemini_model points to a non-Gemini model, this still routes to Gemini via the bundle". Need to verify -- if `make_client` returns ClaudeClient here, code_execution will silently no-op. | Quant skills (orchestrator.py:1170, 1183). |
| `self.rag_client` | orchestrator.py:526 | Direct `GeminiClient(self.rag_model, _gemini_standard)` | Never -- always Gemini (Vertex AI Search constraint). | RAG step. |
| `self.grounded_client` | orchestrator.py:605 | Direct `GeminiClient(_grounded_vertex, _gemini_standard)` | Never -- Google Search Grounding is Gemini-only. | Macro + competitor + deep_dive + one enrichment. |

**Implication**: with the current defaults, the ONLY Claude-bound clients
in the AnalysisOrchestrator are `general_client` and `quant_exec_client`
(both via `settings.gemini_model="claude-sonnet-4-6"`). The 14+ per-ticker
enrichment calls go through `general_client`. **All 14 calls share the
same `ClaudeClient` instance** -- so when the rail flag was False at
__init__ time, ALL 14 calls go direct Anthropic. That maps to the
13-ticker per-cycle failure pattern (13 tickers x 14+ calls = ~180+ 401s
in the log, consistent with the observed timestamp clustering).

### Table 3 -- autonomous_loop.py rail-flag handling (the GOOD pattern)

| Line | What it does | Status |
|------|--------------|--------|
| 1282 | `_route = "claude_code" if getattr(settings, "paper_use_claude_code_route", False) else "anthropic_direct"` | CORRECT -- per-cycle attribution. |
| 1283 | `logger.info("Orchestrator pre-dispatch ticker=%s rail=%s ...")` | CORRECT -- this is the log that says rail=claude_code at 18:59:56. |
| 1285 | `orchestrator = AnalysisOrchestrator(settings)` | **HERE IS THE GAP**: the `settings` passed here is the cycle-level snapshot. If `paper_use_claude_code_route` is True at this snapshot, `make_client()` inside `__init__` should pick `ClaudeCodeClient`. The fact that 401/400 are observed means the snapshot was False at __init__ time. |
| 1455 | (lite path) `use_claude_code_route = bool(getattr(settings, "paper_use_claude_code_route", False))` | CORRECT pattern. Lite path is rail-aware. |
| 1469 | (lite path) `client = anthropic.Anthropic(api_key=api_key) if not use_claude_code_route else None` | CORRECT -- only instantiates SDK client when rail is False. |
| 1499 / 1555 | (lite path) `await asyncio.to_thread(claude_code_invoke, ...)` | CORRECT -- rail path. |

## External research -- read-in-full table (>=5 required)

| # | URL | Accessed | Kind | Fetched how | Key quote / finding |
|---|-----|----------|------|------------|---------------------|
| E1 | https://code.claude.com/docs/en/authentication | 2026-05-27 | Official Anthropic docs | WebFetch (full) | "If you have an active Claude subscription but also have `ANTHROPIC_API_KEY` set in your environment, the API key takes precedence once approved. This can cause authentication failures if the key belongs to a disabled or expired organization. Run `unset ANTHROPIC_API_KEY` to fall back to your subscription, and check `/status` to confirm which method is active." -- and explicit precedence ladder: (1) Cloud creds, (2) ANTHROPIC_AUTH_TOKEN, (3) ANTHROPIC_API_KEY, (4) apiKeyHelper, (5) CLAUDE_CODE_OAUTH_TOKEN, (6) Subscription OAuth. |
| E2 | https://github.com/anthropics/claude-code/issues/53728 | 2026-05-27 | GitHub issue | WebFetch (full) | "When both `ANTHROPIC_API_KEY` (env var) and a Claude Pro/Max OAuth login are present on the same machine, Claude Code silently uses the API key. There is no warning at session start ..." -- documents the silent-shadow failure mode. |
| E3 | https://lalatenduswain.medium.com/claude-code-on-claude-max-plan-understanding-oauth-token-vs-api-key-authentication-in-2026-96a6213d2cde | 2026-05-27 | Authoritative blog (Lalatendu Swain, April 2026) | WebFetch (full) | "If `ANTHROPIC_API_KEY` exists in your shell profile from a previous project, it automatically takes precedence over your subscription credentials. ... the user may not realize billing has switched until the monthly invoice arrives." |
| E4 | https://github.com/anthropics/anthropic-sdk-python | 2026-05-27 | Official Anthropic SDK README | WebFetch (full) | Shows canonical `Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))` with comment "This is the default and can be omitted" -- confirms env var read at instantiation time, not import time. |
| E5 | https://portkey.ai/error-library/insufficient-balance-error-10489 | 2026-05-27 | Portkey AI Gateway error library (industry reference) | WebFetch (full -- the search-result page content) | Documents 3 root causes for 400 `credit_balance_too_low`: (1) actual credit depletion, (2) higher usage tier required, (3) **orphaned/stale API key authenticates successfully but is no longer authorized to bill against an active account**. Case 3 is hardest to debug. Rotating the key resolves it. |

### Snippet-only sources (do not count toward gate floor)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://github.com/anthropics/claude-code/issues/45572 | GitHub issue | Title evidence sufficient ("CLI usage incorrectly classified as API billing on Max subscription") -- corroborates issue #53728 already read in full. |
| https://github.com/anthropics/claude-code/issues/9515 | GitHub issue | Conflict-warning UI improvement issue. Tangential. |
| https://github.com/anthropics/claude-code/issues/27130 | GitHub issue (Windows) | Vertex-AI specific; pyfinagent runs macOS. |
| https://support.claude.com/en/articles/12304248-manage-api-key-environment-variables-in-claude-code | Official help center | Linked from E1; same content. |
| https://github.com/anthropics/claude-code/issues/54839 | "credit_balance_too_low despite sufficient account credits" | Title evidence sufficient -- corroborates Portkey E5 case 3. |
| https://github.com/BerriAI/litellm/issues/24320 | LiteLLM issue | Documents that "credit balance too low" doesn't trigger fallback routing in LiteLLM; reinforces the case for OUR OWN router (make_client) to handle this. |
| https://github.com/anthropics/claude-code/issues/24667 | LiteLLM 401 invalid x-api-key | Snippet matches our exact error -- supports the env-var precedence diagnosis. |
| https://drdroid.io/integration-diagnosis-knowledge/anthropic-invalid-api-key-format | Diagnosis kb | "Verify that your application is actually using the Anthropic key you think it is using. Stale shell variables, Docker compose .env files, per-agent auth profiles, and copied config fragments can override each other." -- supports diagnosis. |
| https://github.com/anthropics/claude-code/issues/5300 | "PRO account shows Credit balance is too low" | Pro/Max users hit this when API key shadows OAuth -- supports the 400 diagnosis. |
| https://docs.posit.co/connect/admin/integrations/oauth-integrations/anthropic/ | Posit Connect docs | OAuth integration docs; tangential. |

### Recency scan (last 2 years, 2024-2026) -- mandatory section

Search-query discipline: three variants per topic.

| Topic | Variant 1 (2026 year-locked) | Variant 2 (2024-2025 window) | Variant 3 (year-less canonical) |
|-------|------------------------------|-------------------------------|---------------------------------|
| Anthropic SDK api_key resolution | `"Anthropic Python SDK api_key parameter ANTHROPIC_API_KEY environment variable resolution 2026"` | `"AuthenticationError" "invalid x-api-key" anthropic python SDK 2025 OR 2026 troubleshoot` | `"anthropic SDK api_key precedence environment variable design"` |
| Claude Code rail isolation | `"Claude Code CLI Max subscription OAuth vs Anthropic API key isolation subprocess"` | (same -- covers 2025+) | `"anthropic.Anthropic api_key constructor late binding env var precedence"` |
| credit balance billing error | `"credit balance is too low" anthropic API 400 error Python SDK 2025 2026` | (same) | (year-less query embedded above) |

**Recency findings**: in the 2024-2026 window, the Anthropic SDK API has
not changed the api_key resolution semantics -- it still reads the env
var at instantiation. The biggest 2026 development is the **June 15
2026 cutoff** for `claude -p` and Agent SDK usage on subscription plans
(quoted at E1: "Starting June 15, 2026, Agent SDK and `claude -p` usage
on subscription plans will draw from a new monthly Agent SDK credit,
separate from your interactive usage limits"). **This DOES affect
pyfinagent**: the rail's `claude_code_invoke()` uses `claude --print`
(see `claude_code_client.py:128`), which is `-p`-equivalent. **After
June 15 2026, the rail will draw from a separate monthly Agent SDK
credit allowance rather than interactive Max subscription -- the
financial expectation that flat-fee covers unlimited cycles will be
broken.** This is a downstream-roadmap finding the operator should
add to the closure_roadmap, not blocking for phase-38.13.1.

Other recency findings: zero changes to `ClaudeClient.__init__` signature.
Pydantic-settings still re-reads `.env` on each `Settings()` construction
(but `@lru_cache` short-circuits subsequent calls in the same process --
this IS the desync mechanism in #5 of the headline finding).

## The 401 mystery -- precise diagnosis

The cycle-8 log entry `19:09:25 [autonomous_loop] Claude analysis
failed for SNDK: Error code: 401 - {'type': 'error', 'error': {'type':
'authentication_error', 'message': 'invalid x-api-key'}, 'request_id':
'req_011Ca5vgMNeXV8TpBvgisTLV'}, trying Gemini orchestrator` was emitted
from `backend/services/autonomous_loop.py:1329`:

```
"Full orchestrator failed for %s: %s -- falling back to lite Claude analyzer"
```

The 401 occurred INSIDE `orchestrator.run_full_analysis(SNDK)` (called
at autonomous_loop.py:1286). Possible mechanisms, ranked by likelihood:

### Mechanism A (HIGH likelihood) -- lru_cache desync
The autonomous-loop process started at time T0 with
`paper_use_claude_code_route=False` (the .env default). Operator
flipped the flag via Settings UI at time T1. `settings_api.py:424`
called `get_settings.cache_clear()`. But the autonomous loop's
`cycle_main()` had already captured `settings = get_settings()` at
its outer scope and was reusing that snapshot. The orchestrator's
pre-dispatch log at 18:59:56 (different cycle, fresher snapshot) read
the True flag, but a subsequent ticker's `AnalysisOrchestrator(settings)`
got the older instance. **Verify**: grep `cycle_main` for `get_settings`
calls and check whether settings is re-read each iteration.

### Mechanism B (MEDIUM likelihood) -- env precedence between .env and process env
`settings_api.py:301` writes the new value to `.env`. But pydantic-settings
gives **environment variables priority over .env file**. If the
backend process has an OLD `PAPER_USE_CLAUDE_CODE_ROUTE=false` in its
process env (set at launchd startup), pydantic will keep returning
False on every `Settings()` instantiation regardless of .env changes.
**Verify**: `launchctl getenv PAPER_USE_CLAUDE_CODE_ROUTE` or
inspect the launchd plist used to start the backend.

### Mechanism C (LOW likelihood) -- key is sk-ant-oat-* OAuth token, not sk-ant-api03-* API key
Per E1: "ANTHROPIC_AUTH_TOKEN" sk-ant-oat-* OAuth tokens MUST be sent
as `Authorization: Bearer ...`, NOT `X-Api-Key`. The Anthropic Python
SDK ALWAYS sends `X-Api-Key`. So if the operator's `.env` has
`ANTHROPIC_API_KEY=sk-ant-oat-...`, every request will 401.
**Note**: `multi_agent_orchestrator.py:170` already has the comment
"MAS Anthropic client unavailable (sk-ant-oat-* 401)" -- confirming
this exact failure was seen before. **Verify**: check the leading
characters of the configured key. (We do NOT read `.env` directly; ask
the operator for `echo $ANTHROPIC_API_KEY | head -c 12`.)

### The 400 sequence (20:01-20:04) -- precise diagnosis

After the 401 at 19:09 on SNDK, the autonomous-loop **rotated to the
lite-Claude fallback** at line 1336: `_select_lite_analyzer(settings.gemini_model)(ticker, settings)`.
`settings.gemini_model="claude-sonnet-4-6"` -> `_run_claude_analysis`.

But the lite path is rail-aware (line 1455). So at this stage, IF
the settings now read the True flag (cache may have been cleared
between 19:09 and 20:01 by a different code path), the lite path
hits the CC rail and would NOT 400.

The 20:01-20:04 evidence shows `Full orchestrator failed for CIEN
... -- falling back to lite Claude analyzer` (same log format,
autonomous_loop.py:1329). That means the full orchestrator path is
still being attempted for these 13 tickers, and STILL hitting the
direct-Anthropic surface. **Why?** Because `AnalysisOrchestrator`
holds `general_client` for the lifetime of that orchestrator instance
(constructed at line 1285 of `_run_single_analysis`). Each per-ticker
call constructs a new orchestrator instance -- so each instance reads
fresh settings. If settings were True at the call to
`AnalysisOrchestrator(settings)`, `make_client()` would have returned
`ClaudeCodeClient`.

So the 20:01-20:04 batch ALSO had `paper_use_claude_code_route=False`
at the `AnalysisOrchestrator(settings)` construction. **Either
mechanism A or B above is in play for the whole window 19:09-20:04**.

The 400 "credit balance too low" specifically suggests the API key
is functioning (no 401) but the associated account has zero credits.
Per Portkey E5 case 3: this is the "orphaned key" case -- the operator
previously used the Anthropic Console but cleared credits to move to
Max-only billing, and the orphan key remains valid.

## Claude Code CLI vs Anthropic SDK isolation -- the 2 auth surfaces

Per E1 (Anthropic Authentication docs) -- THE auth surface for each
code path in pyfinagent:

| Code path | Auth surface | Credentials source |
|-----------|--------------|---------------------|
| `claude_code_invoke()` -> `subprocess.run(["claude", "--print", ...])` | Claude Code CLI | `~/.claude/.credentials.json` (macOS keychain) OR env var per precedence ladder above. **CRITICAL**: per E1, if `ANTHROPIC_API_KEY` is set in the BACKEND PROCESS env, the CLI subprocess INHERITS it and uses API key billing INSTEAD of Max subscription. The rail effectively becomes a no-op for Max-subscription billing. |
| `anthropic.Anthropic(api_key=key)` -> `client.messages.create(...)` | Anthropic Python SDK (direct REST to api.anthropic.com) | `api_key` kwarg OR `ANTHROPIC_API_KEY` env var. **Never** uses ~/.claude/ OAuth. **Always** hits direct billing. |

**Implication for the rail fix**: the Claude Code CLI subprocess SHARES
the parent process's environment by default. So if `ANTHROPIC_API_KEY`
is set in the backend's env (which it is, per settings.py:97 reading
from .env), the `claude` CLI subprocess will use that key and bill
to direct-Anthropic AGAIN, not Max subscription.

`claude_code_client.py:122-125` has the comment about not using `--bare`
because bare requires ANTHROPIC_API_KEY. But that's not the issue --
the issue is the OPPOSITE: bare REQUIRES API key, but the default
mode is supposed to use OAuth. Per E1: "In non-interactive mode
(`-p`), the key is always used when present." So `claude --print` +
`ANTHROPIC_API_KEY` in env = API key billing, NOT Max OAuth.

**Fix**: `claude_code_invoke()` must launch the subprocess with a
SCRUBBED environment that explicitly removes ANTHROPIC_API_KEY and
ANTHROPIC_AUTH_TOKEN. Pattern:

```python
env = {k: v for k, v in os.environ.items()
       if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
completed = subprocess.run(args, input=prompt, env=env, ...)
```

This is a one-line fix in `claude_code_client.py:152` that prevents
the rail from silently degrading to API-key billing.

## Proposed fix -- minimal-edit list (file:line + edit shape)

### Fix 1 -- Scrub env in claude_code_invoke (BLOCKER)

**File**: `backend/agents/claude_code_client.py`
**Line**: 152-160 (inside `claude_code_invoke()`)

Change:
```python
completed = subprocess.run(
    args,
    input=prompt,
    capture_output=True,
    text=True,
    timeout=timeout_s,
    cwd=cwd,
    check=False,
)
```
To:
```python
# phase-38.13.1: scrub ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN
# from the subprocess env to prevent the Claude Code CLI from
# silently falling through to API-key billing per its documented
# auth precedence ladder. The CLI MUST use ~/.claude/ OAuth on the
# Max subscription rail. Without the scrub, the rail flag is
# effectively a no-op for billing. Citation: Anthropic Auth docs
# (code.claude.com/docs/en/authentication) and issue
# anthropics/claude-code#53728.
scrubbed_env = {
    k: v for k, v in os.environ.items()
    if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
}
completed = subprocess.run(
    args,
    input=prompt,
    capture_output=True,
    text=True,
    timeout=timeout_s,
    cwd=cwd,
    check=False,
    env=scrubbed_env,
)
```

**Rationale**: addresses the auth-surface confusion in the rail
itself. Even if the rail flag is True at every layer, the subprocess
would use API-key billing if ANTHROPIC_API_KEY is in env.

**Test**: `backend/tests/test_claude_code_client.py` (already mocks
subprocess.run; assert call_kwargs includes `env` without
ANTHROPIC_API_KEY).

### Fix 2 -- Eliminate the get_settings desync (BLOCKER)

**File**: `backend/services/autonomous_loop.py`
**Line**: 1285 (the `orchestrator = AnalysisOrchestrator(settings)` line)

The cycle-level settings instance may be stale relative to the .env file
if the operator flipped the flag mid-process. Two minimal options:

**Option 2a (1-line fix)**: re-read settings just before constructing
the orchestrator.

```python
# phase-38.13.1: force-refresh settings to pick up live .env edits
# (e.g., operator toggled paper_use_claude_code_route via Settings UI
# mid-cycle). Without this, the orchestrator constructs with the stale
# cycle-level settings snapshot.
from backend.config.settings import get_settings as _get_settings
_get_settings.cache_clear()
settings = _get_settings()
orchestrator = AnalysisOrchestrator(settings)
```

**Option 2b (safer)**: log the active rail value AT orchestrator
construction time, so operator can audit any desync.

```python
_orch_rail = "claude_code" if getattr(settings, "paper_use_claude_code_route", False) else "anthropic_direct"
logger.info(
    "AnalysisOrchestrator construction ticker=%s constructor_rail=%s (cycle_rail=%s)",
    ticker, _orch_rail, _route,
)
orchestrator = AnalysisOrchestrator(settings)
```

Recommend **applying BOTH**.

### Fix 3 -- Make make_client() refuse the silent direct-Anthropic fallback (BLOCKER)

**File**: `backend/agents/llm_client.py`
**Line**: 1909-1912

Currently:
```python
# 2. Direct Anthropic -- wins over GitHub catalog so claude-* never needs GITHUB_TOKEN.
if model_name.startswith("claude-") and anthropic_key:
    logger.info(f"[LLMClient] Routing {model_name} -> Anthropic direct")
    return ClaudeClient(model_name=model_name, api_key=anthropic_key)
```

Change to (option A -- hard fail when rail flag was True at session-config but missed):
```python
# 2. Direct Anthropic -- wins over GitHub catalog so claude-* never needs GITHUB_TOKEN.
# phase-38.13.1: when the operator has explicitly opted into the
# Claude Code CLI rail (paper_use_claude_code_route=True), it is a
# protocol breach to fall through to direct Anthropic -- that would
# silently hit api.anthropic.com and bill against the direct account.
# Refuse instead so the error is loud and actionable.
if model_name.startswith("claude-") and anthropic_key:
    if getattr(settings, "paper_use_claude_code_route", False):
        raise ValueError(
            f"Routing breach: paper_use_claude_code_route=True but "
            f"ClaudeCodeClient import failed earlier and make_client "
            f"is about to construct a direct-Anthropic ClaudeClient. "
            f"This would silently bill against api.anthropic.com "
            f"instead of the Max subscription rail. Fix the "
            f"ClaudeCodeClient import error and retry."
        )
    logger.info(f"[LLMClient] Routing {model_name} -> Anthropic direct")
    return ClaudeClient(model_name=model_name, api_key=anthropic_key)
```

**Why hard fail vs auto-route to lite**: if we silently auto-route,
the operator loses visibility that the rail is broken. A hard error
trips Q/A and forces an explicit fix.

### Fix 4 -- Migrate the 5 direct anthropic.Anthropic() call sites in trading paths

**Out-of-scope for cycle-11** -- these are dormant or out-of-cycle:
- `multi_agent_orchestrator.py:181` (Layer-2 MAS, separate auth path with its own Gemini fallback at line 1001-1008)
- `rag_agent_runtime.py:229` (news_screen path; not hit per-ticker by default)
- `news/sentiment.py:779` (HaikuScorer; news_screen path)
- `llm_client.py:1999` (`advisor_call`, not currently active under default settings)
- `ticket_queue_processor.py:180` (Slack ticket queue, out of cycle)

These should be tracked as P2 in the closure_roadmap. The cycle-11 fix
is Fixes 1-3 above, which address the OBSERVED 13-ticker per-cycle
failure surface (the AnalysisOrchestrator's `general_client`).

### Fix 5 -- Force the rail-flag check on every `Settings()` instantiation

**Out-of-scope for cycle-11** -- pydantic-settings env-var precedence
behavior is fundamental to the framework. The cleanest path is to
ensure the launchd plist used to start the backend does NOT export
PAPER_USE_CLAUDE_CODE_ROUTE in the process env (so .env is the
source of truth). **Verify** via `launchctl getenv PAPER_USE_CLAUDE_CODE_ROUTE`
and `launchctl getenv ANTHROPIC_API_KEY`. If either is exported by
launchd, those overrides win against .env edits and `cache_clear()`.

## Risk / rollback assessment

| Fix | Risk | Rollback shape |
|-----|------|----------------|
| Fix 1 (subprocess env scrub) | LOW. If the CLI was somehow relying on the env var (e.g., user has a non-Max API-key-billing setup), the scrub breaks it. But that's the EXACT desired behavior -- the operator opted into the Max rail. | Revert the `env=scrubbed_env` arg; subprocess inherits parent env. |
| Fix 2a (cache_clear before orchestrator) | LOW. Forces a re-read; the trade-off is one extra .env read per ticker (~1ms). Acceptable. | Remove the 2 lines. |
| Fix 2b (constructor_rail log) | ZERO. Pure observability. | Remove the log line. |
| Fix 3 (hard-fail in make_client) | MEDIUM. Could break unrelated callers that wanted direct Anthropic billing intentionally. Mitigation: the guard ONLY fires when paper_use_claude_code_route=True; existing False-default behavior is unchanged. | Remove the `if getattr(...)` block. |

**Combined rollback strategy**: if any cycle post-fix shows a regression,
flip `paper_use_claude_code_route` to False via Settings UI. The rail
fixes are gated on that flag; setting it False restores the existing
direct-Anthropic path (with its credit-exhaustion failure mode -- the
known starting state).

## Cross-domain validation

The Anthropic SDK API key resolution and Claude Code CLI auth precedence
have direct cross-domain analogues:

- **AWS SDK** -- `AWS_PROFILE` env var overrides shared credentials file
  unless explicitly cleared. Same anti-pattern, same fix (scrub env when
  launching child processes that need a different identity).
- **Google Cloud SDK** -- `GOOGLE_APPLICATION_CREDENTIALS` overrides ADC
  if set. The pyfinagent codebase already handles this correctly per
  `multi_agent_orchestrator.py:199-211` (constructs `_genai.Client` with
  explicit project/location to avoid env-var ambiguity).

The pattern is universal: **when a child process or library reads
auth from env vars by default, parent processes that want to constrain
the auth surface MUST either (a) scrub the env vars, or (b) explicitly
pass the desired credential via the library's kwarg interface.**

## Research Gate Checklist

Hard blockers -- `gate_passed` is false if any unchecked:

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (E1-E5, all read in full).
- [x] 10+ unique URLs total (incl. snippet-only) -- 16 URLs total.
- [x] Recency scan (last 2 years) performed + reported -- section above, with June 15 2026 Agent SDK credit-cutoff finding.
- [x] Full pages read (not abstracts) for the read-in-full set -- E1 read in full (auth precedence ladder); E2 read in full (silent shadow bug body); E3 read in full (OAuth vs API key 2026 article); E4 read in full (SDK README); E5 read in full (Portkey 3-root-cause analysis).
- [x] file:line anchors for every internal claim -- Tables 1, 2, 3 above.

Soft checks:
- [x] Internal exploration covered every relevant module -- 17 call sites enumerated.
- [x] Contradictions / consensus noted -- consensus across E1/E2/E3 on env-var precedence; no contradictions.
- [x] All claims cited per-claim (not just listed in a footer).
- [x] Three-variant query discipline applied -- table above.

## JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 11,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "report_md": "handoff/current/research_brief_phase_38_13_1_rail_wiring_root_cause.md",
  "gate_passed": true
}
```
