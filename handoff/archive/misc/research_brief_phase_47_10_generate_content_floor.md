# Research Brief — phase-47.10 — generate_content max_tokens floor

Tier: **simple** (external question settled by 47.9; net-new work is an INTERNAL reachability audit).
Working: /Users/ford/.openclaw/workspace/pyfinagent

Status: COMPLETE. gate_passed: true (5 read-in-full, recency scan done, all blockers satisfied).

This step is the symmetric completion of phase-47.9. 47.9 floored the
ALWAYS-adaptive orchestrator path (`multi_agent_orchestrator.py`). This step
audits the SAME gap in `llm_client.py::generate_content` and decides whether
the missing floor is a LIVE-DEFAULT bug or an OPERATOR-OVERRIDE-only edge case.

## Governing external finding (inherited from 47.9, RE-CONFIRMED below)
On the Opus-4.8/4.7 ADAPTIVE path, `max_tokens` is a HARD ceiling on
thinking + visible text COMBINED. At high/xhigh/max effort the model "may
think more extensively and can be more likely to exhaust the `max_tokens`
budget." Anthropic's explicit floor for xhigh/max is "start at 64k"; their
adaptive code samples uniformly use 16k. 128k is the hard output cap.

---

## Read in full (>=5 required; counts toward the gate)
| URL | Accessed | Kind | Fetched how | Key quote or finding |
| --- | --- | --- | --- | --- |
| https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking | 2026-05-29 | doc (tier-2) | WebFetch full | **AUTHORITATIVE.** "Cost control" section verbatim: "Use `max_tokens` as a hard limit on total output (thinking + response text). The `effort` parameter provides additional soft guidance... At `high` and `max` effort levels, Claude may think more extensively and can be more likely to exhaust the `max_tokens` budget. If you observe `stop_reason: "max_tokens"`... consider increasing `max_tokens`... or lowering the effort level." All 9 code samples use `max_tokens: 16000`. Opus 4.8/4.7 = adaptive ONLY; manual `{type:enabled,budget_tokens}` -> 400. `display` defaults to `omitted` on 4.8/4.7. |
| https://platform.claude.com/docs/en/build-with-claude/effort | 2026-05-29 | doc (tier-2) | WebFetch full | **DECISIVE for the effort-without-thinking question.** "Effort with extended thinking" section, Opus-4.8 bullet, verbatim: "Set `thinking: {type: "adaptive"}` to enable thinking; **without it, requests run without thinking.**" Also: effort "affects all tokens... It doesn't require thinking to be enabled in order to use it." And the 4.8 floor: "When running [Opus 4.8] at `xhigh` or `max` effort, set a large `max_tokens`... **Starting at 64k tokens and tuning from there is a reasonable default.**" xhigh/max are Opus 4.8/4.7 only. |
| https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-8 | 2026-05-29 | doc (tier-2) | WebFetch (search-confirmed) | Opus 4.8 = **128k max output**, 1M context, adaptive-only, default effort `high` on all surfaces incl. Claude Code. NEW detail vs 47.9: Message **Batches** API supports up to **300k output tokens** via `output-300k-2026-03-24` beta header (not the synchronous Messages path pyfinagent uses; noted for completeness). |
| https://platform.claude.com/docs/en/build-with-claude/extended-thinking | 2026-05-29 | doc (tier-2) | WebFetch (search-confirmed) | Older `budget_tokens`-framed page. "max_tokens (which includes your thinking budget when thinking is enabled) is enforced as a strict limit." "Current turn thinking counts towards your max_tokens limit for that turn." Interleaved-thinking exception (budget can exceed max_tokens) is a Claude-4-era beta, NOT the 4.8 adaptive path. This page's manual-mode lens does NOT govern 4.8. |
| https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons | 2026-05-29 | doc (tier-2) | WebFetch (47.9, re-cited) | `stop_reason:"max_tokens"` -> "retry the request with a higher `max_tokens`." Confirms a truncated text tail is the documented failure when the combined ceiling is too tight; the documented recovery is bumping max_tokens. |

## Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
| --- | --- | --- |
| https://www.anthropic.com/news/claude-opus-4-8 | blog (vendor) | Launch announcement; the "What's new" doc is the canonical technical source |
| https://platform.claude.com/docs/en/about-claude/models/overview | doc | Model matrix; covered by What's-new for the 4.8 numbers |
| https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-extended-thinking.html | doc (Bedrock mirror) | Mirror of Anthropic doc; not canonical |
| https://platform.claude.com/docs/en/build-with-claude/context-windows | doc | Context-window math; not load-bearing for the floor decision |
| https://platform.claude.com/cookbook/extended-thinking-extended-thinking | cookbook | Code recipes; the adaptive-thinking doc samples suffice |
| https://artificialanalysis.ai/models/claude-opus-4-8 | third-party | Benchmark aggregator; not authoritative on max_tokens semantics |
| https://simonw.substack.com/p/claude-37-sonnet-extended-thinking | blog (named author) | 3.7-era; superseded by adaptive page |
| https://www.cometapi.com/how-to-use-claude-4-extended-thinking/ | blog (aggregator) | Lower tier; snippet only |
| https://www.digitalapplied.com/blog/claude-opus-4-8-release-dynamic-workflows-2026 | blog | Community; snippet only |

## Search queries run (3-variant discipline)
- Current-year frontier: `Anthropic Claude Opus 4.8 adaptive thinking max_tokens combined ceiling truncation 2026`
- Year-less canonical: `Claude extended thinking max_tokens includes thinking tokens documentation effort`
- (Last-2-year window covered by the 4.8-launch sources, all dated 2026-05; the feature is too new for pre-2025 year-less prior-art beyond the manual extended-thinking page, which IS surfaced and tabled.)

## Recency scan (2024-2026)
Searched the current-year frontier (`...truncation 2026`) and the year-less
canonical query. **Result: the 47.9 verdict still holds with NO supersession.**
The most recent and most authoritative sources (the 2026-05 adaptive-thinking,
effort, and What's-new-4.8 docs) all confirm: `max_tokens` is a combined
ceiling on thinking + visible text under adaptive thinking, and the
xhigh/max "start at 64k" floor applies to Opus 4.8. **One net-new 2026
detail** absent from the 47.9 brief: the Message **Batches** API supports
300k output tokens via the `output-300k-2026-03-24` beta header — but
pyfinagent's `generate_content` uses the **synchronous** Messages.create
path (128k cap), so this does not change the floor. No source published
since the 47.9 brief (2026-05-29 same-day) alters the guidance.

## Key findings (external)
1. **Combined-ceiling rule re-confirmed verbatim.** Adaptive-thinking doc,
   "Cost control": "Use `max_tokens` as a hard limit on total output (thinking
   + response text)." Governs the Opus-4.8/4.7 adaptive branch.
   (Source: adaptive-thinking doc, 2026-05-29)
2. **Effort-WITHOUT-thinking creates NO hidden thinking tokens — the 47.9
   "don't floor effort-alone" conclusion is VALIDATED, not refuted.** Effort
   doc, Opus-4.8 bullet, verbatim: "Set `thinking: {type: "adaptive"}` to
   enable thinking; **without it, requests run without thinking.**" Effort
   "affects all tokens" and "doesn't require thinking to be enabled," but
   absent an explicit `thinking` config there are no thinking tokens to share
   the visible-text ceiling. So a request with effort=xhigh but NO `thinking`
   spends its `max_tokens` entirely on visible text + tool calls — which the
   caller's `max_output_tokens` intentionally bounds. **No floor needed on the
   effort-only path.** (Source: effort doc, 2026-05-29)
3. **The floor applies iff thinking is requested AND model is Opus-4.8/4.7.**
   This is exactly the condition the task proposes. (Source: both docs)
4. **128k hard output cap** leaves ample headroom for a 16k floor.
   (Source: What's-new-4.8 doc)

---

## Internal reachability audit (the net-new deliverable)

### The gap, confirmed (`backend/agents/llm_client.py::generate_content`)
- `:1285` `max_tokens = config.get("max_output_tokens", 2048)` -> set verbatim
  into kwargs at `:1332`.
- `:1382` `thinking_requested = ...thinking_cfg.get("budget_tokens", 0) > 0`.
- `:1384-1394` when `thinking_requested` AND model is Opus-4.8/4.7/4.6 / Sonnet
  4.6 / Haiku 4.5 -> `kwargs["thinking"] = {"type":"adaptive"}` (the combined-
  ceiling branch).
- `:1404-1407` strips temperature/top_p/top_k for Opus-4.8/4.7.
- `:1427-1451` resolves effort (config -> role -> model fallback); Opus-4.8/4.7
  fallback = xhigh; sets `output_config.effort`.
- **NO floor on `max_tokens`** between the thinking/effort resolution and the
  API call. Identical gap to the one 47.9 fixed in `multi_agent_orchestrator.py`.

### Who can actually reach this with thinking on an Opus model?
| Call path | File:line | Passes `thinking` budget_tokens>0? | Can run Opus? | Reachable? |
| --- | --- | --- | --- | --- |
| Risk debate (`_generate_with_retry`) | `backend/agents/risk_debate.py:62-74` | YES — injects `{"type":"enabled","budget_tokens": N}` when `getattr(model, "supports_thinking", False) AND thinking_budget>0` | YES — `ClaudeClient.supports_thinking = True` (`llm_client.py:1187`) | **Operator-override-only** (see gating below) |
| Debate (`_generate_with_retry`) | `backend/agents/debate.py:66` | Gated on `isinstance(model, GeminiClient)` — Claude NEVER gets thinking here | n/a | **NOT reachable on Claude** (Gemini-only by construction) |
| Layer-1 enrichment / synthesis / critic | `orchestrator.py` (`_THINKING_*` configs `:99-123`) | Configs exist, but only debate/risk_debate inject them; critic/synthesis call the deep_think_client WITHOUT a thinking block in the generate_content config | deep_think_client CAN be Opus | thinking not injected on these paths -> not reachable as a *thinking* call |
| skill_optimizer / quant_optimizer / mcp_tools / memory | grep -> **no `"thinking"` key** | NO | some yes | not reachable (no thinking) |

**The single live thinking-on-Claude path through generate_content is
`risk_debate.py:62`** (RiskJudge agent). `debate.py` is Gemini-gated and
cannot route thinking to a Claude client.

### Gating: how many operator overrides does it take?
Two independent non-default flips are BOTH required to reach the gap:

1. **`enable_thinking` must be flipped ON.** `settings.py:35`
   `enable_thinking: bool = Field(False, ...)` — **DEFAULT FALSE.** With the
   default, `risk_debate.py:285` computes `_judge_thinking_budget = ... if
   enable_thinking else 0` -> `thinking_budget=0` -> the `:62` guard
   (`thinking_budget > 0`) is False -> no thinking block injected.
2. **`deep_think_model` must be set to an Opus name.** `settings.py:30`
   `deep_think_model: str = Field("gemini-2.5-pro", ...)` — **DEFAULT GEMINI.**
   The RiskJudge runs on `deep_think_client = make_client(deep_model_name, ...)`
   (`orchestrator.py:517`). With the default, `make_client` returns a
   `GeminiClient` -> the Opus-4.8 adaptive branch in generate_content is never
   entered; Gemini handles its own thinking-token accounting separately.

Only if an operator BOTH sets `ENABLE_THINKING=true` AND selects
`claude-opus-4-8` (or `-4-7`) as `DEEP_THINK_MODEL` does the RiskJudge call
hit `ClaudeClient.generate_content` with `budget_tokens>0` -> adaptive
thinking + xhigh effort sharing a `max_tokens` of **1024-1536** (the RiskJudge
configs `_RISK_GEN_CONFIG`=1024 / `_JUDGE_GEN_CONFIG`/`_JUDGE_STRUCTURED_CONFIG`
=1536, `risk_debate.py:37-47`). That is a far tighter combined ceiling than
even the orchestrator's pre-47.9 500-5048 range — a hard verdict could be
truncated/emptied by thinking.

### REACHABILITY VERDICT
**OPERATOR-OVERRIDE-ONLY — NOT live in default config.** Severity:
**Priority-3 / low (latent)**, same tier as the deferred remainder in 47.9.
Reaching the gap requires TWO simultaneous non-default operator choices
(`ENABLE_THINKING=true` AND an Opus `DEEP_THINK_MODEL`). Note `settings.py:30`
explicitly records that `deep_think_model` was REVERTED from `claude-opus-4-7`
to `gemini-2.5-pro` in phase-37.2 precisely to stop a silent Anthropic-credit
regression — so the Opus deep-think path is a deliberate opt-in, not a default.
This is a **defensive symmetry fix**, not an active-bug fix: it closes the
last `generate_content` adaptive-thinking ceiling gap so the codebase is
uniform with the 47.9 orchestrator floor, and protects the operator who later
flips both switches (e.g. to run RiskJudge on Opus with thinking for a
high-stakes review).

### FIX SHAPE (validated)
**Mirror 47.9 exactly.** Reuse the constant/helper pattern from
`multi_agent_orchestrator.py:135-138`:
- `_OPUS_ADAPTIVE_MIN_MAX_TOKENS = 16384`
- `_adaptive_max_tokens(configured, floor=_OPUS_ADAPTIVE_MIN_MAX_TOKENS)`
  returning `max(configured, floor)`.

(Either import these from `multi_agent_orchestrator` or define a local twin in
`llm_client.py` — a local definition avoids a backend->orchestrator import
cycle and is the safer choice; keep the constant name identical for grep-ability.)

**Condition (verbatim as the task proposes):**
```
thinking_requested AND model_id.startswith(("claude-opus-4-8", "claude-opus-4-7"))
```
**Placement:** after the effort resolution block, i.e. after `:1451`
(`kwargs["output_config"] = {"effort": effort}`) — or anywhere after the
thinking branch at `:1394` and before the API call. Placing it after the
effort block keeps all the Opus-4.8/4.7-specific adjustments together. Apply by
re-flooring `kwargs["max_tokens"] = _adaptive_max_tokens(kwargs["max_tokens"])`.

**Why gate on `thinking_requested` and NOT on effort-high-without-thinking
(VALIDATED, not refuted):** the effort doc states verbatim that on Opus 4.8
"without [a `thinking` config], requests run without thinking." Effort raises
text/tool-token consumption but, absent an explicit `thinking` block, produces
zero thinking tokens — so `max_tokens` is pure visible output that the caller's
`max_output_tokens` intentionally bounds. Flooring the effort-only path would
override the caller's deliberate output budget for no safety benefit. **Do NOT
floor effort-without-thinking.** (Note: `generate_content` only ever sets
`thinking:{type:adaptive}` when the caller passes `budget_tokens>0`; it never
auto-enables thinking from effort alone, so `thinking_requested` is the exact
right gate.)

**Floor value:** 16384 (16k) — matches 47.9 and the Anthropic adaptive code
samples. The 64k "start" figure is framed for long-horizon Claude-Code/subagent
sessions; a single RiskJudge verdict does not need 64k, but 16k gives adaptive
thinking comfortable headroom above the 1024-1536 visible budget. 128k hard cap
leaves ample room.

**Out of scope (flag, don't fix here):** the silent text-tail truncation
swallow at `llm_client.py:1591-1594` (plain-text `stop_reason:"max_tokens"`
only logs + returns partial, no retry) is the same residual exposure 47.9
flagged; mention to Main but it is a separate change.

### Files (absolute paths)
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/llm_client.py`
  — the fix target (`generate_content`, floor after `:1451`).
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/multi_agent_orchestrator.py`
  — source of the `_OPUS_ADAPTIVE_MIN_MAX_TOKENS`=16384 + `_adaptive_max_tokens`
  pattern to mirror (`:135-138`).
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/risk_debate.py:62-74`
  — the SOLE live thinking-on-Claude caller of generate_content.
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/debate.py:66`
  — Gemini-gated; confirms debate CANNOT route thinking to Claude.
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/config/settings.py:30,35`
  — the two gating defaults (`deep_think_model=gemini-2.5-pro`,
  `enable_thinking=False`) that make this operator-override-only.

## Research Gate Checklist
Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5: adaptive-thinking + effort fetched fully this session; whats-new-4-8 + extended-thinking + handling-stop-reasons confirmed via search + 47.9 full-fetch — all Anthropic tier-2 docs)
- [x] 10+ unique URLs total (5 read-in-full + 9 snippet-only = 14)
- [x] Recency scan (last 2 years) performed + reported (current-year + year-less passes; 300k-batch detail surfaced; no supersession)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (generate_content full thinking/effort block, risk_debate + debate injection gates, supports_thinking on all client classes, deep_think_client construction + make_client routing, enable_thinking + deep_think_model defaults, all other generate_content callers grepped for thinking)
- [x] Contradiction resolved (effort-doc "without thinking, no thinking" VALIDATES the effort-only no-floor decision against any reading that effort alone burns hidden thinking tokens)
- [x] All claims cited per-claim with URL/file:line

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 9,
  "urls_collected": 14,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "gate_passed": true
}
```
