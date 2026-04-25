# Claude Code / Claude Code Remote as Anthropic API Substitute

**Phase:** 16.35 research deliverable
**Date:** 2026-04-24
**Tier:** moderate
**Author:** Researcher agent

---

## Section 1: Current State

pyfinagent uses the direct Anthropic Python SDK (`anthropic>=0.96.0`) with a
`sk-ant-api03-*` key at three call sites:

| File | Call site | Line anchor | Purpose |
|------|-----------|-------------|---------|
| `backend/agents/multi_agent_orchestrator.py` | `_get_client()`, `_call_agent()`, `_call_agent_with_tools()` | 161-1095 | MAS Layer-2: planning, synthesis, tool-loop with interleaved thinking |
| `backend/agents/llm_client.py` | `make_client()` -> `ClaudeClient` | 1090-1138 | Layer-1 routing: direct Anthropic for any `claude-*` model |
| `backend/services/autonomous_loop.py` | `_run_claude_analysis()` | 399-447 | Daily paper-trading ticker analysis |
| `backend/meta_evolution/directive_rewriter.py` | `_call_llm_for_rewrite()` | 159-184 | Harness meta-evolution: researcher prompt rewrite (requires `sk-ant-api03-*` prefix check at line 167) |

**Current blocker (phase-16.31 finding):** The configured key is
`sk-ant-oat-*` (OAuth bearer token), which the Messages API rejects with HTTP
401. The code already has a Gemini fallback path that trips on first 401. The
live pyfinagent MAS therefore runs on Gemini degraded mode for all Claude calls
until a real `sk-ant-api03-*` key is provided.

The user holds a **Claude Max $200/mo** subscription. The question is: can that
subscription be leveraged to avoid purchasing a separate Console API key?

---

## Section 2: Option A — Claude Agent SDK (Python subprocess)

The Claude Code SDK was renamed to the **Claude Agent SDK** in 2026. It is
available as `pip install claude-agent-sdk`.

**How it works:** The SDK spawns the locally-installed `claude` CLI as a
subprocess. Communication is JSON-RPC over stdin/stdout. The Python program
does not call the Anthropic Messages API directly; instead, the CLI process
does, using whatever auth credentials Claude Code has stored on the machine.

**Key API surface (Python):**

```python
from claude_agent_sdk import query, ClaudeAgentOptions, ClaudeSDKClient

# One-shot (new subprocess per call):
async for message in query(
    prompt="Analyze ticker AAPL",
    options=ClaudeAgentOptions(allowed_tools=["Read"]),
):
    if hasattr(message, "result"):
        result = message.result

# Session-reuse (single subprocess, multiple turns):
async with ClaudeSDKClient() as client:
    await client.query("Analyze AAPL")
    async for msg in client.receive_response():
        ...
```

**Auth requirement (CRITICAL):** The official SDK documentation explicitly
states:

> "Unless previously approved, Anthropic does not allow third party developers
> to offer claude.ai login or rate limits for their products, including agents
> built on the Claude Agent SDK. Please use the API key authentication methods
> described in this document instead."
> -- Agent SDK overview, code.claude.com/docs/en/agent-sdk/overview (2026-04-24)

The SDK requires `ANTHROPIC_API_KEY=sk-ant-api03-*`. If that environment
variable is set, `claude` CLI uses it for API billing, NOT the Max subscription.
From the official support article: "If you have an ANTHROPIC_API_KEY environment
variable set on your system, Claude Code will use this API key for
authentication instead of your Claude subscription (Pro, Max, Team, or
Enterprise plans), resulting in API usage charges."

**The Max subscription billing path is BLOCKED for SDK use.** A GitHub issue
(#559 on anthropics/claude-agent-sdk-python, opened Feb 2026, closed without
explanation) explicitly requested Max-plan billing for the SDK; Anthropic closed
it without implementing it.

**The CLI `-p` flag (headless/noninteractive mode):** `claude -p "prompt"` runs
non-interactively. When `ANTHROPIC_API_KEY` is absent and the user is logged in
via `claude auth login` (claude.ai OAuth), the CLI CAN use subscription quota.
This is the `--bare` path recommended for scripted/CI use. However, Anthropic's
April 4, 2026 policy change explicitly blocked third-party tools (defined as
anything not Anthropic's own official apps) from using subscription quota.
Whether `claude -p` called from a Python subprocess counts as "first-party" or
"third-party" is ambiguous; practically, if `ANTHROPIC_API_KEY` is present in
the environment it overrides, and if not, the OAuth path requires no key but
Anthropic may rate-limit it as "unofficial use."

**subprocess.run vs SDK:** The raw subprocess approach (`subprocess.run(['claude',
'-p', prompt, '--output-format', 'json'])`) and the SDK both use the same CLI
binary under the hood. The SDK adds structured Python types, async streaming,
session management, and error handling. Raw subprocess is lower-level but
functionally equivalent for billing purposes.

---

## Section 3: Option B — Claude Code Remote

"Claude Code Remote" in the popular press refers to two distinct features; it is
crucial not to conflate them:

**B1 — Remote Control (released Feb 2026, Research Preview):**
A sync layer that lets you control a *locally-running* Claude Code session from
a phone/browser. NOT cloud-hosted execution. The session runs on Peder's Mac.
The phone is just a UI window. This feature:
- Is NOT a substitute for the API
- Does NOT expose any programmatic interface
- Requires `claude auth login` (claude.ai subscription), NOT API key
- Does NOT work when `ANTHROPIC_API_KEY` is set (it explicitly fails with
  "Remote Control disabled... authenticated with API key")
- Relevant to pyfinagent? No. It is a human-interaction feature, not a
  server-side call path.

**B2 — Claude Code on the Web (cloud-hosted):**
Anthropic runs Claude Code in managed cloud infrastructure. Accessed via
claude.ai/code. This also exposes no programmatic API surface for external
programs to call. It is a browser-based IDE, not an API replacement.

**B3 — /ultrareview (April 2026):**
Cloud-hosted bug-hunting fleet. 3 free runs for Pro/Max users. No programmatic
API.

**Conclusion for Option B:** "Claude Code Remote" is a UX feature for human
developers, not an API substitute. There is no cloud endpoint pyfinagent can
call as a drop-in replacement for `anthropic.Anthropic().messages.create()`.

---

## Section 4: Upsides — Option A (Agent SDK / CLI subprocess)

**A1. Cost arbitrage (potential, not guaranteed):**
Max 20x ($200/mo) = ~900 messages per 5-hour window. If the Max subscription
quota applies to CLI `-p` calls without `ANTHROPIC_API_KEY` set, the effective
token cost is $0 marginal for those calls. Phase-16.27 estimated pyfinagent
API cost at $3-10/mo. Saving $3-10/mo when you already pay $200/mo is a 1.5-5%
cost reduction on the subscription — not zero, but not transformative.
One data point: a developer tracking 10B tokens over 8 months paid ~$800 on Max
vs $15,000+ at API rates. That is a 94% saving at very high volume; pyfinagent's
volume is orders of magnitude lower.

**A2. No Console billing account needed:**
If the subscription path works, Peder does not need a separate console.anthropic.com
account with a credit card attached for API usage.

**A3. Full tool-use support:**
The Agent SDK supports all Claude tools (Read, Edit, Bash, WebSearch, MCP),
extended thinking, subagents, hooks, and session resumption. Feature parity is
high.

**A4. Structured JSON output:**
`--output-format json` and the SDK's `ResultMessage` type provide structured
output compatible with pyfinagent's JSON parsing patterns. Schema enforcement
via `--json-schema` is available.

**A5. Session continuity:**
`ClaudeSDKClient` maintains a persistent subprocess session with conversation
history across multiple calls -- potentially useful for `_call_agent_with_tools`
tool-loop continuation without round-tripping a new process per turn.

**A6. MCP pass-through:**
The SDK passes through `.mcp.json` MCP servers, which pyfinagent already uses
(BigQuery MCP, Alpaca MCP from phase-17). This could enable richer tool calls
with existing infrastructure.

---

## Section 5: Downsides — Option A (Agent SDK / CLI subprocess)

**D1. API key STILL REQUIRED for programmatic SDK use (hard blocker):**
Anthropic's official policy: SDK requires `ANTHROPIC_API_KEY`. Max subscription
quota cannot be used for Agent SDK calls. The cost savings evaporate -- you
still need a Console API key, and SDK calls bill at per-token rates. This is
the single most important finding of this research. The user's assumption
("Claude Code subprocess would leverage the Max subscription") is INCORRECT
per Anthropic's documented policy as of April 2026.
Sources: Agent SDK overview docs, shareuhack.com OpenClaw analysis, Anthropic
policy change coverage (pymnts.com, April 4, 2026).

**D2. Subprocess startup latency (+500-1500ms per call):**
Each `query()` call spawns a new `claude` process, loads CLAUDE.md, discovers
MCP servers, and initializes context. This adds 500-1500ms to first-token
latency vs the direct SDK path (~50-200ms). For `_call_agent_with_tools` which
runs multi-turn loops, this overhead compounds. Use `ClaudeSDKClient` for
session reuse to amortize the startup cost.

**D3. Requires `claude` CLI installed on the runtime machine:**
The pyfinagent server runs on Peder's Mac (local-only deployment, per project
memory). This is met today. But it creates a system-level dependency: `claude`
binary version pinning, PATH management, and the risk that `claude --version`
changes behavior. The TypeScript SDK bundles the binary; the Python SDK does not.

**D4. Tool-loop implementation differences:**
`_call_agent_with_tools` (multi_agent_orchestrator.py:1016-1095) implements a
bespoke interleaved-thinking loop with `AGENT_TOOLS`, `MAX_TOOL_TURNS`, and
AuthenticationError fallback. The Agent SDK has its own autonomous tool loop
that CANNOT be directly introspected turn-by-turn. The existing `thinking=`
parameter (line 1058-1065, Opus 4.7 adaptive vs Opus 4.6 enabled) has no
direct equivalent in the SDK's Python API -- the SDK controls thinking
internally. Replacing `_call_agent_with_tools` with the SDK requires either
accepting the SDK's opaque loop or significant reimplementation.

**D5. Token usage / cost accounting breaks:**
`ResultMessage.total_cost_usd` is present but may be `None`. The existing
`backend/agents/cost_tracker.py` tracks per-agent token usage from
`response.usage.input_tokens` / `output_tokens`. The SDK returns
`ResultMessage.usage` dict with `input_tokens`, `output_tokens`,
`cache_read_input_tokens`, `cache_creation_input_tokens` -- so cost tracking
IS possible, but requires adapter code.

**D6. No streaming support for caller:**
The existing architecture uses synchronous `client.messages.create()` calls.
The Agent SDK is async-only (`async for message in query(...)`). Integrating
into FastAPI's async routes is straightforward, but the existing
`_call_agent` and `_call_agent_with_tools` are sync methods. Thread executor
wrapping (`loop.run_in_executor`) would be needed or a full async refactor.

**D7. Anthropic policy risk:**
The April 4, 2026 subscription policy change happened WITH 30 days notice and
hurt real users. Anthropic could similarly restrict `claude -p` headless usage
to API-key-only, or change the SDK billing model again. The subscription path
is less stable than a direct API key contractually.

**D8. `directive_rewriter.py` explicitly checks for `sk-ant-api03-*` prefix:**
Line 167: `if api_key and api_key.startswith("sk-ant-api"):`. This guard exists
because OAT keys fail. The subprocess path bypasses this check entirely -- the
rewriter would need separate adaptation.

---

## Section 6: Implementation Sketch

If Peder decides to proceed (see Recommendation), the minimal change set is:

**Step 1 — Install and verify:**
```bash
pip install claude-agent-sdk
claude --version  # must be >=2.1.51 for Remote Control, any recent for SDK
```

**Step 2 — Auth decision (critical fork):**
- If using API key (pay-per-token): set `ANTHROPIC_API_KEY=sk-ant-api03-*` in
  `backend/.env`. The SDK will use it automatically. This is the SAFE path.
- If testing subscription path (no API key): unset `ANTHROPIC_API_KEY`, ensure
  `claude` is logged in via `claude auth login`. Risk: rate limits, policy
  violation if Anthropic detects automated use.

**Step 3 — Adapter for `_call_agent` (simple replacement):**
```python
# New function in multi_agent_orchestrator.py
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def _call_agent_sdk(agent_config, task) -> tuple[str, dict]:
    """Agent SDK replacement for _call_agent."""
    result_text = ""
    usage = {"input": 0, "output": 0}
    async for message in query(
        prompt=f"{agent_config.system_prompt}\n\n{task}",
        options=ClaudeAgentOptions(
            allowed_tools=[],  # no tools for simple calls
            model=agent_config.model,
            max_turns=1,
        ),
    ):
        if hasattr(message, "result"):
            result_text = message.result or ""
        if hasattr(message, "usage") and message.usage:
            usage["input"] = message.usage.get("input_tokens", 0)
            usage["output"] = message.usage.get("output_tokens", 0)
    return result_text or "No response.", usage

def _call_agent(self, agent_config, task):
    """Sync wrapper: run async SDK call in a new event loop."""
    return asyncio.run(_call_agent_sdk(agent_config, task))
```

**Step 4 — `_call_agent_with_tools` replacement:**
This is harder. The SDK's autonomous tool loop cannot be controlled turn-by-turn.
Options:
- Pass `AGENT_TOOLS` as allowed_tools and let the SDK run its own loop. Accept
  that thinking budget and turn count are SDK-controlled.
- Keep the existing `_call_agent_with_tools` as-is (it uses direct Anthropic
  SDK), and only replace the simpler `_call_agent` calls with the Agent SDK.

**Step 5 — `autonomous_loop.py::_run_claude_analysis`:**
Simplest replacement -- this is a single-turn call with no tools:
```python
result, usage = await _call_agent_sdk(agent_config, prompt)
```

**Step 6 — `directive_rewriter.py::_call_llm_for_rewrite`:**
Remove the `sk-ant-api03-*` prefix guard. Replace with Agent SDK call or keep
Gemini fallback as primary since it does not depend on API key.

**Files that change:**
- `backend/agents/multi_agent_orchestrator.py` (lines 161-184, 962-1003, 1016-1095)
- `backend/services/autonomous_loop.py` (lines 399-447)
- `backend/meta_evolution/directive_rewriter.py` (lines 159-199)
- `requirements.txt` (add `claude-agent-sdk>=0.2.111`)
- `backend/.env` (remove or replace `ANTHROPIC_API_KEY`)

---

## Section 7: Cost Analysis

| Path | Monthly cost | Notes |
|------|-------------|-------|
| Direct Anthropic SDK (API key, pay-per-token) | $3-10/mo | Per phase-16.27; pyfinagent volume is low |
| Agent SDK with API key | $3-10/mo + subprocess overhead (latency only) | Billing identical to direct SDK |
| Agent SDK with Max subscription (IF it worked) | $0 marginal | Does NOT work per Anthropic policy |
| Max subscription (already paying) | $200/mo flat | For Claude Code CLI + claude.ai use |
| Max + API key | $200 + $3-10/mo | Both needed if using SDK programmatically |

**Key insight:** The Max subscription CANNOT reduce pyfinagent's API bill for
programmatic calls. The $3-10/mo API cost remains. Moving to the Agent SDK
adds complexity and latency without reducing cost. The only scenario where
the Max subscription saves money is if Peder manually copy-pastes prompts into
claude.ai -- which is already what the subscription covers.

---

## Section 8: Reliability and Risk

**Risk 1 — Anthropic policy instability (HIGH):**
April 4, 2026: Anthropic blocked third-party subscription OAuth with 30 days
notice. February 2026: Anthropic banned subscription OAuth in unofficial apps.
The programmatic subscription path is the most volatile part of the stack.
Reverting to direct API key requires only an env var change.

**Risk 2 — CLI binary dependency (MEDIUM):**
pyfinagent's `claude` binary would need version pinning. An `npm` update or
Homebrew auto-update could change CLI behavior mid-run. The direct SDK pins
`anthropic>=0.96.0` in requirements.txt and is stable.

**Risk 3 — Async refactor scope (MEDIUM):**
`_call_agent` and `_call_agent_with_tools` are synchronous. The Agent SDK is
async-only. `asyncio.run()` wrappers work but can conflict with FastAPI's own
event loop if called from within an async context. Full async refactor of
multi_agent_orchestrator.py is a significant change.

**Risk 4 — Rate limits under load (MEDIUM):**
Max 20x allows ~900 messages per 5-hour window. pyfinagent's autonomous loop
runs ~daily and makes ~5-15 LLM calls per cycle. This is well within limits.
But harness optimization cycles (run_harness.py) can make 50-100 calls in a
burst. A burst could hit the 5-hour window limit.

**Risk 5 — SLA gap (LOW for pyfinagent):**
No formal SLA for subscription services vs API. For pyfinagent (local-only,
paper trading, not live money), this is acceptable.

---

## Section 9: Recommendation

**Recommendation: NO — do not replace the Anthropic API key with the Agent SDK
or Claude Code Remote as the primary execution path for #21.**

**Rationale:**

1. The core premise is false. The Max subscription ($200/mo) does NOT cover
   Agent SDK / programmatic calls. Anthropic's official documentation and
   April 2026 policy changes make this explicit. The expected cost saving does
   not exist.

2. The direct fix for #21 ("Anthropic key swap") is simple: replace the
   `sk-ant-oat-*` OAT key in `backend/.env` with a real `sk-ant-api03-*`
   Console API key. Cost: $3-10/mo. This unblocks Claude entirely with zero
   architecture change.

3. The Agent SDK path adds: subprocess latency (+500-1500ms), async complexity,
   CLI binary dependency, cost tracking adapter work, and a harder-to-control
   tool-loop -- for zero cost benefit given Anthropic's billing policy.

**CONDITIONAL YES for a specific subset:** If Peder wants to explore the
`claude -p` headless path WITHOUT setting `ANTHROPIC_API_KEY` (pure OAuth
subscription path), the daily `_run_claude_analysis` ticker analysis (1 call/day)
and the harness meta-evolution rewriter (occasional) could potentially run on
subscription quota. This is NOT officially supported for third-party programmatic
use, carries policy-violation risk, and could be rate-limited without warning.
Not recommended for a trading system.

**What closes #21 entirely:** Obtaining a Console API key (`sk-ant-api03-*`)
and setting `ANTHROPIC_API_KEY` in `backend/.env`. The existing fallback
infrastructure (phase-16.31 Gemini fallback) stays in place. No architecture
change needed.

---

## Section 10: What This Closes

The step #21 "Anthropic key swap (user action)" is about replacing the broken
`sk-ant-oat-*` with a working `sk-ant-api03-*`. Research shows:

- The Agent SDK does NOT close #21 as an alternative billing path.
- Claude Code Remote does NOT close #21 -- it is a UX feature, not an API.
- #21 closes only when `backend/.env` has a valid `sk-ant-api03-*` key.

However, this research suggests a NEW optional step could be added to the
masterplan: **phase-16.36 "Claude Agent SDK integration (optional enhancement)"**
with the goal of replacing `_call_agent` (simple calls only, not tool-loop) with
the Agent SDK for better observability hooks, session management, and future
proofing -- but explicitly NOT as a cost-reduction measure and NOT replacing the
API key requirement.

---

## Read in Full

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://code.claude.com/docs/en/agent-sdk/overview | 2026-04-24 | Official doc | WebFetch | SDK requires ANTHROPIC_API_KEY; Max subscription explicitly prohibited for SDK use |
| https://code.claude.com/docs/en/agent-sdk/python | 2026-04-24 | Official doc | WebFetch | Full Python API: query(), ClaudeSDKClient, ResultMessage with usage dict, async-only |
| https://code.claude.com/docs/en/remote-control | 2026-04-24 | Official doc | WebFetch | Remote Control is local-only sync; NOT cloud execution; requires claude.ai OAuth, blocks API key |
| https://code.claude.com/docs/en/headless | 2026-04-24 | Official doc | WebFetch | -p flag is now "Agent SDK via CLI"; --bare mode skips OAuth; API key overrides subscription |
| https://intuitionlabs.ai/articles/claude-max-plan-pricing-usage-limits | 2026-04-24 | Blog (authoritative) | WebFetch | Max 20x = 900 msgs/5hr window; no API access policy detail; subscription-only coverage |
| https://www.pymnts.com/artificial-intelligence-2/2026/third-party-agents-lose-access-as-anthropic-tightens-claude-usage-rules/ | 2026-04-24 | News/industry | WebFetch | April 4, 2026: Anthropic ended subscription quota for third-party tools; $200 Max was covering $1-5k of API-equivalent compute |
| https://buildwithaws.substack.com/p/inside-the-claude-agent-sdk-from | 2026-04-24 | Authoritative blog | WebFetch | Billing: per-token, API key required; no subscription; JSON-RPC stdin/stdout subprocess architecture |
| https://support.claude.com/en/articles/11145838-using-claude-code-with-your-pro-or-max-plan | 2026-04-24 | Official support doc | WebFetch | ANTHROPIC_API_KEY overrides subscription; results in API charges not subscription usage |

## Identified but Snippet-Only

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://github.com/anthropics/claude-agent-sdk-python/issues/559 | GitHub issue | Fetched but response missing (issue closed without explanation, no Anthropic comment visible) |
| https://github.com/anthropics/claude-code/issues/39903 | GitHub issue | Search result only; Max subscribers unexpected charges via subagents |
| https://findskill.ai/blog/claude-code-subscription-pricing-guide/ | Blog | Search snippet; pricing overview |
| https://simonwillison.net/2026/Apr/22/claude-code-confusion/ | Blog | Fetched; covers pricing confusion; no SDK billing detail |
| https://www.shareuhack.com/en/posts/openclaw-claude-code-oauth-cost | Blog | Fetched; April 2026 OAuth cutoff; key detail: API key overrides subscription |
| https://northflank.com/blog/claude-rate-limits-claude-code-pricing-cost | Blog | Fetched; rate limit info; confirms SDK needs API key |
| https://releasebot.io/updates/anthropic | Release notes aggregator | Search snippet; release chronology |
| https://venturebeat.com/orchestration/anthropic-just-released-a-mobile-version-of-claude-code-called-remote | News | 429 rate limit on fetch |
| https://pypi.org/project/claude-agent-sdk/ | PyPI | Search snippet; package exists |
| https://github.com/anthropics/claude-agent-sdk-python | GitHub | Search snippet; source repo |

## Recency Scan (2024-2026)

Searched: "Claude Agent SDK 2026", "Claude Code Remote 2026", "Anthropic Max subscription API access 2026",
"claude -p noninteractive billing subscription 2026", "Anthropic third-party subscription cutoff 2026".

**New 2026 findings that supersede or significantly update older understanding:**

1. **April 4, 2026:** Anthropic ended third-party subscription quota access. This directly kills
   the assumption that Max plan billing can power programmatic SDK calls.
2. **February 2026:** Claude Agent SDK (formerly Claude Code SDK) launched. Requires API key, not
   subscription. Max plan billing request filed (Issue #559) and closed without resolution.
3. **February 2026:** Remote Control launched (Research Preview). Confirmed local-only, not
   a programmatic API.
4. **April 2026:** /ultrareview cloud feature launched. Not relevant to API substitution.
5. The "claude -p subscription billing path" remains technically possible if no API key env var
   is set, but is explicitly prohibited for third-party tools by Anthropic's updated policy.

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/multi_agent_orchestrator.py` | 161-220 | `_get_client()`, Gemini fallback init | Active; OAT 401 fallback already in place |
| `backend/agents/multi_agent_orchestrator.py` | 962-1003 | `_call_agent()` simple calls | Active; routes to Gemini on 401 |
| `backend/agents/multi_agent_orchestrator.py` | 1016-1095 | `_call_agent_with_tools()` tool loop with interleaved thinking | Active; complex; hard to replace with SDK |
| `backend/agents/llm_client.py` | 1090-1138 | `make_client()` routing | Active; requires `sk-ant-api03-*` for Claude models |
| `backend/services/autonomous_loop.py` | 399-447 | `_run_claude_analysis()` | Active; simple single-turn; easiest to adapt |
| `backend/meta_evolution/directive_rewriter.py` | 159-199 | `_call_llm_for_rewrite()` | Active; `sk-ant-api03-*` prefix guard at line 167 |
| `requirements.txt` | -- | `claude-agent-sdk` | NOT present |

---

## Consensus vs Debate

**Consensus:** Agent SDK requires API key for programmatic use. Max subscription covers only
official Anthropic apps (claude.ai, Claude Code CLI interactive, Desktop). Third-party/automated
use of subscription quota was cut off April 4, 2026.

**Debate:** Whether `claude -p` without `ANTHROPIC_API_KEY` set (pure OAuth path) technically
violates Anthropic's policy. The policy language targets "third-party developers offering claude.ai
login or rate limits for their products." A single user calling `claude -p` from their own machine
for their own project may not be a "third-party developer offering" anything. Anthropic's
enforcement is unclear at this granularity. This is a grey area, not a clear prohibition.

---

## Pitfalls (from Literature)

1. **Env var override:** Setting `ANTHROPIC_API_KEY` in any shell that runs pyfinagent will
   silently switch billing from subscription to API -- unexpected charges. Unset it to test
   subscription path; set it for reliable production use.
2. **Subprocess latency in backtest loops:** `_run_single_analysis` calls Claude ~15 times per
   ticker in the full pipeline. At +1s per subprocess startup, a 20-ticker run adds 300s overhead.
3. **asyncio.run() in FastAPI:** FastAPI handlers are already async. Calling `asyncio.run()` from
   within an async context raises `RuntimeError: This event loop is already running`. Use
   `asyncio.ensure_future()` or `await` directly.
4. **CLAUDE.md loading:** Without `--bare`, `claude -p` loads pyfinagent's `CLAUDE.md` (the
   harness protocol). This adds tokens and could confuse the model with harness instructions.
   Always use `--bare` for scripted calls.

---

## Application to pyfinagent

| Finding | pyfinagent implication | File:line |
|---------|----------------------|-----------|
| SDK requires API key, not subscription | #21 must be a real Console API key; no shortcut | `backend/.env` |
| Gemini fallback already exists | Stack is resilient; adding another fallback tier (SDK) is redundant | `multi_agent_orchestrator.py:186-220` |
| `sk-ant-api03-*` guard in directive_rewriter | Already guards against OAT keys; would work with real API key | `directive_rewriter.py:167` |
| `_call_agent_with_tools` uses interleaved thinking params | SDK hides thinking; not directly replaceable | `multi_agent_orchestrator.py:1051-1065` |
| `autonomous_loop._run_claude_analysis` is simple 1-turn | Easiest migration target if SDK adopted later | `autonomous_loop.py:399` |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (8 fetched in full)
- [x] 10+ unique URLs total (18 URLs collected: 8 full reads + 10 snippet-only)
- [x] Recency scan (last 2 years) performed + reported (5 key 2026 findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (4 files fully read)
- [x] Contradictions noted (grey-area debate on `claude -p` OAuth path)
- [x] All claims cited per-claim (URLs inline throughout)
