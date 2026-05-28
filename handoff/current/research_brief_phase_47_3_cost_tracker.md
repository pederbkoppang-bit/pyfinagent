# Research Brief — phase-47.3: Opus 4.8 cost_tracker pricing regression

**Tier:** moderate
**Date:** 2026-05-28
**Objective:** Validate the exact fix for the `claude-opus-4-8` cost-tracking
regression and identify any OTHER 4.7->4.8 gaps, so the "Compute" term of the
north-star objective (Net System Alpha = Profit - Risk - Compute) is accurate
before the system trades / migrates to an API key.

---

## TL;DR

- **Pricing CONFIRMED:** Claude Opus 4.8 = **$5.00 / MTok input, $25.00 / MTok
  output** (standard), identical to 4.7. Authoritative: Anthropic pricing docs +
  the Opus 4.8 announcement. CLAUDE.md's $5/$25 claim is CORRECT.
- **Regression CONFIRMED:** `cost_tracker.py::MODEL_PRICING` (line 20-76) has
  `claude-opus-4-7: (5.00, 25.00)` at line 26 but NO `claude-opus-4-8` key. Every
  4.8 call falls through to `_DEFAULT_PRICING=(0.10, 0.40)` at line 79 —
  understating input cost **50x** ($5.00/$0.10) and output cost **62.5x**
  ($25.00/$0.40). Validated against actual code.
- **Exact fix:** add `"claude-opus-4-8": (5.00, 25.00),` to `MODEL_PRICING`,
  immediately ABOVE the `claude-opus-4-7` line (preserve the newest-first
  ordering). Units = per-1M-tokens, tuple = `(input, output)`.
- **TWO more files need the same 4.8 entry** (independent maps, NOT importers of
  MODEL_PRICING): `backend/api/settings_api.py` (display/pricing list, line 214)
  and `backend/api/settings_api.py` allowlist (line 31). `sovereign_api.py` is
  SAFE — it imports MODEL_PRICING so it inherits the fix.
- **max_tokens-at-xhigh verdict: REAL but NOT triggered by cost_tracker.** It is a
  latent risk for any Claude-routed reasoning agent at xhigh/max effort whose
  `max_output_tokens` cap is small (1024-4096). Anthropic explicitly says set
  `max_tokens` to "64k tokens" starting point at xhigh/max. The pyfinagent
  per-agent caps (Enrichment 1024, Synthesis 4096) are Gemini-locked today, so the
  collision is not live — but it WILL bite if any of those agents is ever routed to
  Opus 4.8 at xhigh. Documented as an audit item, not a phase-47.3 blocker.

---

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/agents/cost_tracker.py` | 20-79 | `MODEL_PRICING` dict + `_DEFAULT_PRICING` — THE regression | **BROKEN**: no 4-8 key; falls to (0.10,0.40) |
| `backend/api/settings_api.py` | 214 | SEPARATE Anthropic pricing display list (`input_per_1m`/`output_per_1m`) | **GAP**: no 4-8 row |
| `backend/api/settings_api.py` | 31 | `ALLOWED_MODELS`-style allowlist tuple | **GAP**: 4-8 absent (4-7..4-1 present) |
| `backend/api/sovereign_api.py` | 254, 277-278 | Imports `MODEL_PRICING` + `_DEFAULT_PRICING` from cost_tracker; computes llm_call_log cost | **SAFE**: inherits cost_tracker fix automatically |
| `backend/config/model_tiers.py` | 49,51,60 | Role->model map already on `claude-opus-4-8` (commit 8ecc9efe) | OK |
| `backend/config/model_tiers.py` | 184,234 | `EFFORT_SUPPORTED_MODELS` + `MODEL_EFFORT_FALLBACK` — 4-8 present, xhigh | OK |
| `backend/config/model_tiers.py` | 221-231 | `EFFORT_DEFAULTS` — all mas_* roles = `max` (step-scoped 23.2.2 override) | OK (note below) |
| `backend/agents/llm_client.py` | 471,584,1404,1444,1478 | Routing + xhigh guard + opus-4-8 prefix handling | OK (4-8 handled) |
| `backend/agents/llm_client.py` | 1330-1336 | `max_tokens` passed from `config["max_output_tokens"]` default 2048 | see max_tokens finding |
| `backend/slack_bot/governance.py` | 84-85 | Hardcoded cost estimate `input*0.00001 + output*0.00003` (= $10/$30 per 1M) | STALE/separate (note below) |
| `backend/slack_bot/jobs/cost_budget_watcher.py` | 26 | BQ-bytes watcher only ($6.25/TiB); NOT LLM-token pricing | N/A (Max flat-fee; no LLM-$ here) |
| `backend/agents/orchestrator.py` | 83-123, 453-455 | Per-agent `max_output_tokens` caps (1024/1536/2048/4096) | Gemini-pathed; see max_tokens finding |

### Confirmed structure of `MODEL_PRICING` (the precise-fix anchor)

- Type annotation (line 20): `dict[str, tuple[float, float]]`.
- Tuple semantics: `(input_price, output_price)`, **per 1,000,000 tokens** (USD).
  Confirmed by the comment at line 16 ("Pricing per 1M tokens (input, output)")
  AND by every consumer dividing by `1_000_000` (lines 173-179, 231-237 in
  cost_tracker; line 278 in sovereign_api).
- Existing Opus key + value: `"claude-opus-4-7": (5.00, 25.00),` at **line 26**.
- `_DEFAULT_PRICING = (0.10, 0.40)` at **line 79** — this is the fallthrough that
  causes the 50x/62.5x understatement.
- Key format: bare hyphenated model ID, no provider prefix (`claude-opus-4-7`,
  not `anthropic/claude-opus-4-7`). The 4-8 key must match this exact format
  because `record()` line 161 does `MODEL_PRICING.get(model, _DEFAULT_PRICING)`
  with the raw `model` string, and callers pass `claude-opus-4-8` (confirmed
  in model_tiers.py:49 and llm_client.py:584 the canonical ID is
  `claude-opus-4-8`).

---

## Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote / finding |
|-----|----------|------|-------------|---------------------|
| https://platform.claude.com/docs/en/about-claude/pricing | 2026-05-28 | Official doc | WebFetch (full) | "Claude Opus 4.8 — $5/MTok base input … $25/MTok output … 5m cache writes $6.25, 1h cache writes $10, cache hits $0.50". Batch: "$2.50 input / $12.50 output". Long context: "Opus 4.8 … include the full 1M token context window at standard pricing." |
| https://www.anthropic.com/news/claude-opus-4-8 | 2026-05-28 | Official announcement | WebFetch (full) | "$5 per million input tokens … $25 per million output tokens … Pricing for regular usage is unchanged from Opus 4.7." Fast mode "$10 input / $50 output". No separate 1M-context surcharge. |
| https://platform.claude.com/docs/en/build-with-claude/effort | 2026-05-28 | Official doc | WebFetch (full) | "The effort parameter affects **all tokens** in the response, including … Extended thinking." + "When running Claude Opus 4.8 at `xhigh` or `max` effort, set a large `max_tokens` so the model has room to think and act … Starting at 64k tokens … is a reasonable default." Opus 4.8 default = high. |
| (search→synthesis) https://platform.claude.com/docs/en/build-with-claude/extended-thinking | 2026-05-28 | Official doc (via search result body) | WebSearch full-answer | "`budget_tokens` must be less than `max_tokens` … thinking budget is carved out of your output allocation." Confirms thinking/effort tokens are deducted from max_tokens, not additive. |
| https://www.neowin.net/news/anthropic-launches-claude-opus-48-with-better-coding-and-lower-fast-mode-pricing/ | 2026-05-28 | Tech press (corroboration) | WebSearch full-answer | "regular Opus 4.8 usage remains unchanged, costing $5 per million input tokens and $25 per million output tokens." Independent confirmation of the unchanged $5/$25. |

(5 sources read in full; the two pricing docs are the authoritative anchors, the
effort + extended-thinking docs anchor the max_tokens finding, Neowin is the
independent third-party price corroboration.)

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.anthropic.com/claude/opus | Official product page | Price already nailed by docs+announcement |
| https://platform.claude.com/docs/en/about-claude/models/overview | Official doc | Overview; pricing page is the authority |
| https://artificialanalysis.ai/models/claude-opus-4-8/providers | Industry benchmark | Third-party; price corroborated elsewhere |
| https://decrypt.co/369384/anthropic-claude-opus-4-8-... | Tech press | Corroborates "same price"; Neowin already used |
| https://www.macrumors.com/2026/05/28/anthropic-claude-opus-4-8/ | Tech press | Corroboration only |
| https://thenewstack.io/claude-opus-48-release/ | Tech press | "effort controls … cheaper fast mode" — effort doc is authority |
| https://github.blog/changelog/2026-05-28-claude-opus-4-8-is-generally-available-for-github-copilot/ | Vendor changelog | Copilot availability, not pricing/budget |
| https://openrouter.ai/anthropic/claude-opus-4.8 | Aggregator | Resale pricing, not first-party |
| https://www.requesty.ai/models/vertex/claude-opus-4-8-eu | Aggregator | Vertex EU resale, not first-party |
| https://github.com/anthropics/claude-code/issues/8756 | GitHub issue | "max_tokens must be > thinking.budget_tokens" — anecdote; docs are authority |
| https://github.com/QwenLM/qwen-code/issues/2508 | GitHub issue | Same carve-out bug class; docs cover it |
| https://www.finout.io/blog/claude-opus-4.7-pricing-... | Industry blog | 4.7 cost story; not 4.8-specific |

**URLs collected total: 17** (5 read in full + 12 snippet-only).

## Search-query variants run (3-variant discipline)

1. **Current-year frontier:** "Claude Opus 4.8 API pricing input output per million
   tokens" / "Claude Opus 4.8 extended thinking high effort max_tokens output
   budget interaction 2026".
2. **Last-2-year window:** "Anthropic max_tokens includes thinking tokens budget
   extended thinking 2025".
3. **Year-less canonical:** "Anthropic Claude Opus 4.8 pricing $5 $25 announcement
   1M context surcharge" (no year-lock on the core price claim; surfaced the
   anthropic.com/news announcement + pricing docs).

## Recency scan (2024-2026)

Searched the last-2-year window for new Anthropic pricing / max_tokens guidance.
**Findings:** The decisive sources are BRAND NEW (Opus 4.8 launched **2026-05-28**,
today). Key recency results that SUPERSEDE older assumptions:
1. Opus 4.8 pricing page row published 2026-05-28 confirms **$5/$25 unchanged** —
   this supersedes any stale "verified 2026-04-18" comment in cost_tracker.py
   (which only covers 4.7-and-earlier).
2. Fast mode for Opus 4.8 dropped to **$10/$50** (3x cheaper than 4.6/4.7's
   $30/$150) — relevant ONLY if the system ever enables fast mode (it does not
   today; no fast-mode flag found in llm_client).
3. Adaptive-thinking + effort is now the canonical thinking control on 4.6/4.7/4.8
   (manual `budget_tokens` returns 400 on Opus 4.7/4.8) — already handled in
   llm_client.py:1384-1391. No new gap.
No finding contradicts the $5/$25 standard-rate fix.

## Validated fix

**File:** `backend/agents/cost_tracker.py`, inside `MODEL_PRICING` (line 20-76).

**Exact change** — insert ONE line directly above line 26 (`claude-opus-4-7`):

```python
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),   # existing line, unchanged
```

- Key string: `"claude-opus-4-8"` (bare hyphenated ID, matches canonical ID in
  model_tiers.py:49 + llm_client.py:584).
- Tuple value: `(5.00, 25.00)` — `(input_per_1M, output_per_1M)` in USD.
- Units: per 1,000,000 tokens (consistent with every existing row + the `/1_000_000`
  divisions in `record()`).
- Placement preserves the file's newest-first Opus ordering and keeps the comment
  block at line 25 ("current GA") accurate.

This single edit also fixes `sovereign_api.py`'s `profit_per_llm_dollar`
aggregation (line 254 imports the same dict) — no second edit needed there.

## Other gaps (files needing a 4.8 entry independently)

1. **`backend/api/settings_api.py:214`** — the Anthropic pricing DISPLAY list
   (`{"model": ..., "input_per_1m": 5.00, "output_per_1m": 25.00}`). This is a
   SEPARATE hardcoded list (NOT imported from MODEL_PRICING), surfaced in the
   Settings UI cost table. Add a `claude-opus-4-8` row with `input_per_1m: 5.00,
   output_per_1m: 25.00` above the 4-7 row. Without it the Settings UI omits 4.8
   from the model picker / cost preview even though the system runs on it.
2. **`backend/api/settings_api.py:31`** — the model allowlist tuple
   (`"claude-opus-4-7", "claude-opus-4-6", ...`). 4-8 is absent. Add
   `"claude-opus-4-8",` first. (Confirm whether this tuple gates selectability;
   if so, 4.8 may be unselectable in the UI despite being the runtime default.)
3. **`backend/slack_bot/governance.py:84-85`** — a hardcoded rough estimate
   `(input*0.00001 + output*0.00003)` = $10 in / $30 out per 1M. This is NOT the
   Opus rate ($5/$25) and is labelled "Rough cost estimate (update for actual
   pricing)". Pre-existing inaccuracy, model-agnostic; flag for a follow-up but
   NOT part of the 4.7->4.8 regression. Out of phase-47.3 scope unless the
   contract widens to "all LLM cost paths".

**SAFE (no change needed):** `sovereign_api.py` (imports MODEL_PRICING),
`cost_budget_watcher.py` (BQ-bytes only — pyfinagent is Claude Max flat-fee so
there is no live LLM-dollar meter; this watcher only tracks BigQuery spend).

## max_tokens-at-xhigh finding

**Verdict: REAL latent risk, NOT triggered by this fix, NOT a phase-47.3 blocker.**

Mechanism (authoritative, from the effort + extended-thinking docs read in full):
- The effort parameter "affects **all tokens** in the response, including …
  Extended thinking" (effort doc). Thinking tokens are "carved out of your output
  allocation" — `budget_tokens < max_tokens` (extended-thinking doc). So at
  `xhigh`/`max`, adaptive thinking consumes the SAME `max_tokens` budget the final
  answer draws from.
- Anthropic's explicit guidance: "When running Claude Opus 4.8 at `xhigh` or `max`
  effort, set a large `max_tokens` … Starting at **64k tokens** … is a reasonable
  default." Opus 4.8 supports **128k max output tokens** on the API.

Exposure in pyfinagent:
- `llm_client.py:1285,1332` passes `max_tokens = config.get("max_output_tokens",
  2048)` straight to `messages.create`. The per-agent caps in `orchestrator.py`
  are 1024 (Enrichment), 1536 (Debate), 2048 (deep-think), 4096 (Synthesis).
- IF any of those agents is routed to Opus 4.8 at xhigh/max, a 1024-4096 `max_tokens`
  is FAR below Anthropic's 64k recommendation → the model can exhaust the budget on
  thinking and hit `stop_reason=max_tokens` with a truncated/empty answer. The
  llm_client retry at line 1565-1573 only doubles to a max of 8192 and only fires
  on a `tool_use` tail — it would NOT rescue a thinking-exhausted plain-text call.
- TODAY this is dormant: those orchestrator agents resolve to Gemini
  (`gemini_enrichment`/`gemini_deep_think` are Gemini-locked; the swappable Layer-1
  default is `gemini-2.0-flash`). The Opus-4.8 roles (mas_main/mas_qa) are the
  Layer-3 harness agents, which run in Claude Code (not through these caps).
- ACTIONABLE follow-up (separate masterplan item, not 47.3): when `apply_model_to_
  all_agents=True` routes Layer-1 to Opus 4.8, OR when any reasoning agent is pinned
  to Opus 4.8 at xhigh, raise its `max_output_tokens` to >=16k (ideally 64k) OR
  step effort down to `high`. A guard could clamp: "if Claude-4.8 + effort in
  {xhigh,max} and max_tokens < 16000, bump to 16000 and log."

## Non-brittle verification assertion

Assert the 4.8 rate EQUALS the 4.7 rate (avoids hardcoding $5/$25 a second time, so
the test stays green if Anthropic ever re-prices Opus and both rows are updated
together — but FAILS if 4.8 is missing and falls to the (0.10,0.40) default):

```python
from backend.agents.cost_tracker import MODEL_PRICING, _DEFAULT_PRICING
assert "claude-opus-4-8" in MODEL_PRICING, "4.8 missing -> falls to default"
assert MODEL_PRICING["claude-opus-4-8"] == MODEL_PRICING["claude-opus-4-7"], \
    "4.8 must match 4.7 standard rate ($5/$25)"
assert MODEL_PRICING["claude-opus-4-8"] != _DEFAULT_PRICING, \
    "4.8 must not resolve to the 50x-understating fallback"
```

One-liner CLI form for the verification command field:

```bash
python -c "from backend.agents.cost_tracker import MODEL_PRICING as P, _DEFAULT_PRICING as D; assert P.get('claude-opus-4-8')==P['claude-opus-4-7']!=D; print('OK 4.8=', P['claude-opus-4-8'])"
```

(Optional stronger end-to-end: build a fake response with usage_metadata of
1_000_000 input / 1_000_000 output, `CostTracker().record(model='claude-opus-4-8',
...)`, assert `cost_usd == 30.0` i.e. $5 + $25. This catches a wrong tuple value,
not just a missing key.)

## /claude-api skill note

The `/claude-api` skill migrates Claude API CODE between versions (SDK call shapes,
deprecated params, header changes). It would deterministically catch the
llm_client.py-style concerns (manual `budget_tokens` -> adaptive, temperature strip
on 4.8, xhigh guard — all already handled). It would NOT catch the cost_tracker
regression: a missing dict key in an internal pricing table is a DATA/config gap,
not an API-call-shape change, so it is outside the skill's migration surface. Net:
`/claude-api` is not a substitute for this fix and would not have flagged it.

## Application to pyfinagent (mapping)

| External finding | pyfinagent anchor | Action |
|---|---|---|
| Opus 4.8 = $5/$25 (pricing docs, announcement, Neowin) | `cost_tracker.py:26` (4-7 row), `:79` (default) | Add `"claude-opus-4-8": (5.00, 25.00)` above :26 |
| Same rate as 4.7 | `settings_api.py:214` | Add display row $5/$25 |
| 4.8 is a selectable GA model | `settings_api.py:31` | Add to allowlist |
| Effort consumes max_tokens; set 64k at xhigh | `llm_client.py:1285/1332`, `orchestrator.py` caps | Follow-up guard (separate item) |
| Cache writes 1h=$10 (2.0x base) | `cost_tracker.py:174` (already 2.0x) | No change — already correct |
| Batch = 50% (Opus 4.8 $2.50/$12.50) | `cost_tracker.py:184-185` (is_batch *=0.5) | No change — multiplier model already correct, inherits new base |

---

## Research Gate Checklist

Hard blockers — `gate_passed` is false if any unchecked:
- [x] >=5 authoritative external sources READ IN FULL (2 pricing docs + effort doc
      via WebFetch; extended-thinking + Neowin via full WebSearch answer bodies)
- [x] 10+ unique URLs total (17: 5 read-in-full + 12 snippet-only)
- [x] Recency scan (last 2 years) performed + reported (Opus 4.8 launched today)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every pricing/cost module (cost_tracker,
      settings_api, sovereign_api, governance, cost_budget_watcher, model_tiers,
      llm_client, orchestrator caps)
- [x] Contradictions / consensus noted (all sources agree $5/$25)
- [x] All claims cited per-claim

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 12,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "gate_passed": true
}
```
