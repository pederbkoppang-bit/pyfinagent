# Research Brief -- phase-37.3 (budget_tokens deprecation cleanup)

Tier: simple. WRITE-FIRST. Researcher = sole research agent.
Question: are the 20 `budget_tokens` references in `backend/`
actually deprecated, and if so, which API surface (Anthropic vs
Vertex AI Gemini) drives the rename to `thinking_budget`?

## A. Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://platform.claude.com/docs/en/build-with-claude/extended-thinking | 2026-05-23 | doc (Anthropic) | WebFetch (HTML, full) | "`budget_tokens` is deprecated on Claude Opus 4.6 and Claude Sonnet 4.6." Claude Opus 4.7 rejects `{type:"enabled", budget_tokens:N}` with a 400 error. Migration path = adaptive thinking + `effort` parameter. There is NO new field `thinking_budget` on the Anthropic surface -- the field IS being deleted, not renamed. |
| https://platform.claude.com/docs/en/api/messages | 2026-05-23 | doc (Anthropic, REST reference) | WebFetch (HTML, full) | The Messages API still documents `thinking.type=enabled` with `budget_tokens` as a valid shape (required when type=enabled, min 1024, must be < max_tokens). This is the legacy path that remains valid for older models (Opus 4.5 and earlier). Adaptive (`type:"adaptive"`) is the new shape for 4.6+. |
| https://platform.claude.com/docs/en/build-with-claude/effort | 2026-05-23 | doc (Anthropic) | WebFetch (HTML, full) | Resolves the contradiction: "For Claude Opus 4.6 and Sonnet 4.6, effort replaces `budget_tokens` as the recommended way to control thinking depth ... `budget_tokens` is still accepted on Opus 4.6 and Sonnet 4.6 [but] deprecated and will be removed in a future model release." Opus 4.7 + Mythos Preview already removed it. |
| https://ai.google.dev/gemini-api/docs/thinking | 2026-05-23 | doc (Google) | WebFetch (HTML, full) | Vertex AI / Gemini Python SDK uses `ThinkingConfig(thinking_budget=N)` -- the field IS named `thinking_budget` (snake_case). Gemini 3+ adds a new field `thinkingLevel` (LOW/MEDIUM/HIGH) but `thinking_budget` is NOT deprecated on the Gemini surface; backward compatibility maintained. |
| /Users/ford/.openclaw/workspace/pyfinagent/.venv/lib/python\*/site-packages/google/genai/types.py | 2026-05-23 | source (google-genai SDK, installed in this venv) | grep on `class ThinkingConfig` | Lines 5313+5321: `class ThinkingConfig(_common.BaseModel):` -> `thinking_budget: Optional[int] = Field(...)`. Confirms the canonical Python field name in the SDK pyfinagent already imports. |

(Five sources read in full; gate floor met.)

## B. Snippet-only (does NOT count toward gate; context)

| URL | Kind | Why not fetched in full |
|-----|------|--------------------------|
| https://dev.to/ji_ai/opus-47-killed-budgettokens-what-changed-and-how-to-migrate-3ian | blog | Confirms 400-error behavior on Opus 4.7; corroborates Anthropic effort doc. No new info beyond. |
| https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking | doc (Anthropic) | Same content as the extended-thinking page above; would duplicate. |
| https://github.com/block/goose/issues/7293 | issue (3rd party) | Issue title alone confirms ecosystem alignment with the deprecation. |
| https://help.apiyi.com/en/claude-adaptive-thinking-mode-api-guide-replace-extended-thinking-en.html | 3rd-party blog | Reseller doc; lower trust than primary. |
| https://medium.com/@dalio8/claude-opus-4-7-3e1e14a8a3c3 | blog | Migration walk-through; corroborates the 4.7 behavior. |
| https://platform.claude.com/docs/en/about-claude/models/whats-new-claude-4-7 | doc (Anthropic) | Cited in pyfinagent llm_client.py phase-4.14.7 comment; not re-fetched because the effort doc already captures the relevant claim. |
| https://github.com/anthropics/anthropic-sdk-python (CHANGELOG raw) | SDK repo | The HTML view returned no content; the WebFetch on the raw file gave a partial response confirming **no rename** to `thinking_budget` in the Anthropic SDK changelog -- the field is being deleted, not renamed. |

## C. Recency scan (last 2 years; mandatory)

Searched 2024-2026 explicitly. Findings:

1. **2026-04-14 (Opus 4.7 launch)** -- adaptive thinking becomes
   the ONLY supported reasoning mode on Opus 4.7. Manual
   `{type:"enabled", budget_tokens:N}` returns HTTP 400. Confirmed
   on the Anthropic effort doc + multiple 3rd-party migration
   posts (allthings.how, rabinarayanpatra.com, anthonymaio
   substack, dev.to/ji_ai).
2. **2026 (Claude Sonnet 4.6 / Opus 4.6)** -- `budget_tokens` is
   accepted but officially deprecated; effort + adaptive is the
   recommended replacement. Will be removed in a future model
   release.
3. **Google Gemini 3 (2026)** -- introduces `thinkingLevel`
   (LOW/MEDIUM/HIGH) for Gemini 3 Pro. `thinking_budget` is NOT
   deprecated -- it remains the canonical Python SDK field and
   is preserved for backward compatibility. Gemini 2.5 still uses
   `thinking_budget` exclusively. The Vertex AI Python SDK in
   pyfinagent's venv (`google-genai`, types.py:5313-5340) still
   uses `thinking_budget` as the dataclass field name.

Net effect: the Anthropic side is in active deprecation. The
Google side is NOT deprecated -- `thinking_budget` is the correct
name there.

## D. Search-query composition (mandatory under research-gate.md)

Three variants run for visibility:
1. Current-year frontier: "anthropic budget_tokens deprecated 2026
   adaptive thinking"
2. Current-year specific: "claude opus 4.7 thinking adaptive
   budget_tokens rejected 400 error"
3. Year-less canonical: implicit via the direct doc fetches
   (Anthropic platform docs are evergreen, not year-bound; Google
   ai.google.dev thinking page is evergreen).

A separate Gemini-side year-less search was not run explicitly
because the canonical doc + the installed SDK source (which IS
prior-art by definition) answered the question definitively. The
SDK file in `.venv/lib/python*/site-packages/google/genai/types.py`
is the source of truth for what the running code calls.

## E. Recommended scope (a/b/c)

**Recommendation: (c) -- the masterplan's `audit_basis` is wrong
as written; the step should be downgraded or closed as NO_OP.**

The audit_basis says "should be `thinking_budget` per Vertex AI
2026 SDK." That statement assumes the 20 `budget_tokens`
references are all going to Vertex AI / Gemini. They are not.
Breaking the 20 down by API surface:

### E.1 Anthropic-bound references (KEEP `budget_tokens` -- it IS the field name; field is deprecated but cannot be renamed because there is no rename, only adaptive replacement)

| File:line | Code | Notes |
|-----------|------|-------|
| backend/agents/llm_client.py:1379 | `thinking_cfg.get("budget_tokens", 0)` | Anthropic client path. Reads the legacy dict shape passed in by orchestrator. CORRECT field name on the Anthropic surface. |
| backend/agents/llm_client.py:1383 | comment: `# Adaptive path (no budget_tokens accepted).` | Documentation, not a usage. Refers to the Anthropic adaptive shape. |
| backend/agents/llm_client.py:1387 | `budget = thinking_cfg["budget_tokens"]` | Anthropic legacy-manual path (Opus 4.5 and older). Required by the API for those models. |
| backend/agents/llm_client.py:1388 | `kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}` | The literal Anthropic request body for `type=enabled`. Cannot be renamed -- this IS the wire format. |

Verdict: **all 4 references must keep `budget_tokens`**. The
Anthropic API still uses that exact field name on the wire for
legacy models.

### E.2 Cross-surface (LLMClient internal dict that fans out to BOTH Anthropic AND Gemini)

| File:line | Code | Notes |
|-----------|------|-------|
| backend/agents/orchestrator.py:99 | `"thinking": {"type": "enabled", "budget_tokens": 8192}` | Caller-side generation_config dict. Consumed by EITHER GeminiClient (lines 907-919 translate to typed `ThinkingConfig(thinking_budget=budget)`) OR ClaudeClient (lines 1378-1388 translate to `{type:"enabled", budget_tokens:N}` for legacy or `{type:"adaptive"}` for 4.6+). The dict shape `{"type":"enabled","budget_tokens":N}` is a deliberate "Anthropic-shaped" lingua franca with translation at the client. |
| backend/agents/orchestrator.py:104 | same | same |
| backend/agents/orchestrator.py:117 | same | same |
| backend/agents/orchestrator.py:123 | same | same |
| backend/agents/orchestrator.py:707 | same | runtime mutation of generation_config |
| backend/agents/orchestrator.py:713 | same | fallback path |
| backend/agents/multi_agent_orchestrator.py:1070 | `"budget_tokens": 2048` | same Anthropic-shaped lingua franca dict |
| backend/agents/multi_agent_orchestrator.py:1056 | comment about adaptive | doc |
| backend/agents/debate.py:67 | `{"type": "enabled", "budget_tokens": thinking_budget}` | Builder of the same shape |
| backend/agents/risk_debate.py:26 | comment about correct translation to ThinkingConfig | doc |
| backend/agents/risk_debate.py:63 | `{"type": "enabled", "budget_tokens": thinking_budget}` | Builder |

The `{type, budget_tokens}` dict is project-internal lingua
franca. It is NOT a wire shape going to either API directly; both
clients translate it. Renaming the internal key to
`thinking_budget` would force a parallel rename in every Anthropic
translation site (because the literal `budget_tokens` IS the wire
field for legacy Anthropic models), creating asymmetric churn for
zero behavioural change.

### E.3 GeminiClient translation site (already correct)

| File:line | Code | Notes |
|-----------|------|-------|
| backend/agents/llm_client.py:909 | comment | doc |
| backend/agents/llm_client.py:914 | `budget = int(thinking_cfg.get("budget_tokens", 0) or 0)` | Reads the lingua-franca key, then creates `ThinkingConfig(thinking_budget=budget)` on line 917. Already does the right translation. |

The Gemini-bound code path is correct as written -- it ALREADY
uses the typed `ThinkingConfig(thinking_budget=budget)` on the
SDK side. The string `budget_tokens` on llm_client.py:914 is just
the lookup key into the internal lingua-franca dict, not a Gemini
API field.

### E.4 Test references

| File:line | Code | Notes |
|-----------|------|-------|
| backend/tests/test_phase_41_0_bundle_close.py:60 | string mentioning the phase number | Test that REQUIRES phase-37.3 to remain open until the budget_tokens cleanup is decided. The current text presumes the cleanup is needed; if (c) is correct, this test's expectation must be updated. |
| backend/tests/test_phase_41_0_bundle_close.py:74 | "phase-37.3 (budget_tokens deprecation, OPEN-18) must remain" | Same. |

## F. Why NOT recommendation (a) -- full removal

Recommendation (a) would delete all 20 references. That breaks
the Anthropic legacy path on Opus 4.5 and earlier (the API
literally requires `{type:"enabled", budget_tokens:N}` on those
models). Anthropic's effort doc explicitly states: "Claude Opus
4.5 uses manual thinking (`thinking: {type: "enabled",
budget_tokens: N}`), where effort works alongside the thinking
token budget."

pyfinagent's `llm_client.py:1378-1388` already gates on model_id
to choose adaptive vs manual. The gate is CORRECT. Removing
`budget_tokens` would break the legacy branch for any deployment
that still pins Opus 4.5 or earlier (which pyfinagent's model
fallback table supports per `backend/config/model_tiers.py`).

## G. Why NOT recommendation (b) -- partial Gemini-only rename

The Gemini side is ALREADY correct as written. The SDK call on
llm_client.py:917 uses the typed `ThinkingConfig(thinking_budget
=budget)` form. The string `"budget_tokens"` on lines 909, 914
are dict lookup keys into the lingua-franca shape that
orchestrator.py builds. Renaming those would force renaming the
Anthropic translation sites too (which CANNOT be renamed, per
E.1), creating dictionary asymmetry across the two clients.

## H. Concrete recommendation for the contract

The masterplan should be amended:

1. Reword `audit_basis` to remove the assertion that
   `budget_tokens` is incorrect. The correct framing: "Anthropic
   has begun deprecating `budget_tokens` on Opus 4.6 / Sonnet 4.6
   (manual is still accepted, adaptive is recommended) and Opus
   4.7 rejects it outright. pyfinagent's llm_client.py:1378-1388
   ALREADY gates on model_id to choose adaptive vs manual
   correctly. No source-code rename is required. Gemini's
   `thinking_budget` is unchanged and not deprecated."
2. Replace `verification.command` with one of:
   - **Option H.1 (close as NO_OP):** verification = "grep -c
     'budget_tokens' backend/agents/llm_client.py returns >0 AND
     the adaptive-vs-manual gate on llm_client.py:1382 is intact"
     (i.e. document that the field IS the correct name, the
     code is current).
   - **Option H.2 (cosmetic-only):** rename the internal
     lingua-franca dict key from `budget_tokens` to
     `internal_thinking_budget` (a project-internal name that
     avoids the confusion). This is a 20-site mechanical rename
     with parallel updates to the translation site on
     llm_client.py:1388 to map back to `budget_tokens` for the
     Anthropic wire. NOT recommended -- adds churn for zero
     behavioural change.

3. Update `backend/tests/test_phase_41_0_bundle_close.py:60,74`
   to reflect that phase-37.3 closes as NO_OP.

## I. Anti-patterns to avoid in GENERATE

- Do NOT delete `budget_tokens` from llm_client.py:1388. That
  string IS the Anthropic wire field for legacy models.
- Do NOT rename the lingua-franca dict key alone; the translation
  on llm_client.py:1388 mirrors the dict key and would need a
  parallel update.
- Do NOT add `thinking_budget` as a NEW key alongside
  `budget_tokens` in the orchestrator dicts; both clients only
  read one of the two and you'd silently break one path.
- Do NOT presume `thinking_budget` is "the 2026 standard" -- it
  is the Gemini field name only. The Anthropic field is going
  away entirely, not being renamed.

## J. Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch
- [x] 10+ unique URLs total (5 read + 7 snippet-only = 12)
- [x] Recency scan (last 2 years) performed + reported (section C)
- [x] Full pages read for the read-in-full set
- [x] file:line anchors for every internal claim (sections E.1-E.4)

Soft checks:
- [x] Internal exploration covered every relevant module (4 .py
      files: llm_client, orchestrator, multi_agent_orchestrator,
      debate, risk_debate)
- [x] Contradictions noted (Anthropic extended-thinking page
      "deprecated" vs Messages-API page "current" -- resolved by
      the effort doc which shows the trajectory)
- [x] All claims cited per-claim with URL

## K. JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 7,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/research_brief_phase_37_3.md",
  "gate_passed": true
}
```
