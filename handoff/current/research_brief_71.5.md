# Research Brief — Step 71.5: effort/model posture reconciliation

Tier: **moderate** | Gate: Layer-3 harness research gate | Accessed: 2026-07-17

Objective: RECONCILE the stale `EFFORT_DEFAULTS` "max" override vs the authoritative
CLAUDE.md effort policy; pin Main's Layer-3 tier deterministically; tighten the
fallbackModel chain; prune stale Fable-window comments in researcher.md/qa.md.

---

## Read in full (5 required; counts toward the gate)
| # | URL | Kind | Fetched how | Key finding |
|---|-----|------|-------------|-------------|
| 1 | https://platform.claude.com/docs/en/build-with-claude/effort | Official doc (tier 2) | WebFetch full | Default=high on all surfaces incl. Claude Code; Opus 4.8 "start with xhigh for coding/agentic"; **max "Reserve for genuinely frontier problems ... adds significant cost for relatively small quality gains, and on some structured-output or less intelligence-sensitive tasks it can lead to overthinking."** Effort affects ALL tokens incl. tool calls. Sonnet 4.6 default=medium; Sonnet 5 default=high. xhigh accepted on Fable5/Mythos5/Opus4.8/4.7/Sonnet5 only. |
| 2 | https://code.claude.com/docs/en/model-config | Official doc (tier 2) | WebFetch full | fallbackModel = up to 3 models tried in order; triggers on overload/unavailable/non-retryable server errors; NOT on auth/rate-limit/billing; chain capped at 3 "after duplicate removal, extra entries ignored"; `availableModels` filters fallback targets. |
| 3 | https://www.anthropic.com/news/claude-opus-4-8 | Official announcement (tier 2) | WebFetch full | "Opus 4.8 defaults to high effort, which we judge to be the best overall balance"; use "extra"(=xhigh) for difficult/long-running async; max for even better at more tokens. Pricing $5/$25 (regular), $10/$50 (fast mode). ~4x less likely to let code flaws pass. |
| 4 | https://claude.com/blog/claude-model-and-effort-level-in-claude-code | Official blog (tier 2) | WebFetch full | "For most tasks use the model's default effort level"; raise effort when Claude skipped a file / didn't run tests / didn't double-check; Sonnet = "a really good generalist". No explicit fallback-chain guidance. |
| 5 | https://www.aiforanything.io/blog/claude-code-fallback-model-overload-fix-2026 | Practitioner (tier 4) | WebFetch full | Fallback triggers on HTTP 529 overloaded_error + non-retryable errors; NOT auth/rate-limit/request-size/transport. **Recommended chain `claude-opus-4-8 -> claude-sonnet-4-6 -> claude-haiku-4-5`; "Haiku is almost never overloaded" = the always-available floor.** |

## Snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched |
|-----|------|-----------------|
| github.com/anthropics/claude-code/issues/65782 | Issue | fallbackModel ordering docs-gap tracker; snippet sufficient |
| findskill.ai/blog/claude-opus-4-8-effort-settings | Blog | Secondary effort explainer; official doc #1 authoritative |
| www.mindstudio.ai/blog/claude-opus-4-8-effort-levels-explained | Blog | Secondary; superseded by #1 |
| vibe.cerridan.com/posts/claude-code-fallback-model-chain | Blog | Fallback overview; #2/#5 cover it |
| www.digitalapplied.com/blog/claude-code-safe-mode-fallback-models | Blog | Fallback + safe-mode; #5 stronger |
| ofox.ai/blog/claude-api-error-529-overloaded-fix-2026 | Blog | 529 handling; #5 covers |
| gist.github.com/mculp/...settings.json Reference | Gist | v2.1.104 settings ref; stale vs current |

## Recency scan (last 2 years)
Three-variant discipline: current-year frontier queries run ("...Opus 4.8 xhigh max docs" +
"...fallbackModel behavior ordering overload rate limit", both 2026). The effort parameter
and `fallbackModel` are both **2026-native** Claude Code/API features — there is no pre-2024
canonical prior-art to surface with a year-less query (noted per the rule). **Result: 1
material finding.** The current Opus 4.8 effort doc (fetched today 2026-07-17) SUPERSEDES the
phase-23.2.2 "all mas_* = max" step-scoped override: the doc explicitly reserves `max` for
frontier problems and warns it causes overthinking on structured-output / less-intelligence-
sensitive tasks (exactly the per-ticker MAS classification/synthesis workload). fallbackModel
semantics (introduced v2.1.166; current v2.1.205+) are stable and confirmed current.

---

## Internal code inventory (file:line anchors)
| File:line | Role | Status / finding |
|-----------|------|------------------|
| `backend/config/model_tiers.py:261-271` | EFFORT_DEFAULTS | ALL mas_* = `"max"`. STALE step-scoped override (revert target). |
| `backend/config/model_tiers.py:255-260` | comment | Documents the 23.2.2 override + **pre-revert values: communication=low, main=xhigh, qa=high, research=medium** + "revert after closure". |
| `backend/config/model_tiers.py:293-310` | resolve_effort(role) | Reads EFFORT_DEFAULTS. |
| `backend/config/model_tiers.py:212-231` | EFFORT_SUPPORTED_MODELS | CORRECT (fable-5, sonnet-5, opus 4.8/4.7/4.6/4.5/4.1, sonnet 4.6/4.5). No silent-drop risk. |
| `backend/config/model_tiers.py:273-290` | MODEL_EFFORT_FALLBACK | CORRECT (fable/opus-4.8/4.7=xhigh, sonnet-5=high, sonnet-4.6=medium, haiku=None). |
| `backend/agents/llm_client.py:1504-1531` | effort resolution | **The ONLY consumer of resolve_effort/EFFORT_DEFAULTS.** Fires ONLY when `config["_role"]` is set AND the call goes through the ClaudeClient wrapper. |
| `backend/agents/multi_agent_orchestrator.py:223-246` | _get_client() | Returns **raw `anthropic.Anthropic()`** — docstring: "used only for tool-loop calls that need native API features." BYPASSES ClaudeClient. |
| `backend/agents/multi_agent_orchestrator.py:1098,1146,1267` | MAS create calls | `client.messages.create(...)` passes model/max_tokens/system/messages(/tools/thinking/format) — **NO `output_config.effort`.** Runtime effort = API default = `high`. |
| `backend/agents/multi_agent_orchestrator.py:164` | comment | **STALE/WRONG: "Layer-2 agents run at effort=max"** — they run at `high` (API default). Fix in this step. |
| `backend/tests/test_phase_59_1_fable_adoption.py:50` | test | Asserts `resolve_effort("mas_main")=="max"`. **REQUIRED co-change**: update to `"xhigh"` on revert or the suite breaks. |
| `.claude/settings.json:2` | Main Layer-3 pin | `effortLevel="xhigh"` — already deterministic. |
| `.claude/settings.json:3` | fallbackModel | `["claude-opus-4-8","claude-sonnet-5"]` — first hop == Main's primary (Opus 4.8) = redundant; no availability floor. |
| `.claude/settings.json:11` | statusMessage | Note "chain covers OVERLOAD-class only; rate/usage-limit never switch models (v2.1.199 partial-work retention)" — ACCURATE per sources #2/#5. |
| `.claude/agents/qa.md:5 / :45` | Q/A frontmatter | `model: opus` (reverted 67.4) + `effort: max`. Lines 7-44 = stale Fable narration to PRUNE. |
| `.claude/agents/researcher.md:5 / :43` | Researcher frontmatter | `model: opus` (reverted 67.4) + `effort: max`. Lines 7-42 = stale Fable narration to PRUNE. |

## CRUX — is config==runtime on the MAS path? NO, but in the SAFE direction.
EFFORT_DEFAULTS says `mas_* = max`, but the **live Layer-2 MAS path never reads it**:
`multi_agent_orchestrator._get_client()` returns the raw Anthropic SDK (mao:243), and every
MAS `messages.create` (mao:1098/1146/1267) omits `output_config.effort` → runtime = API
default `high`. `resolve_effort()` is invoked ONLY at `llm_client.py:1509` (ClaudeClient
wrapper), which the raw-SDK MAS calls bypass. Grep confirms **no live caller sets
`config["_role"]` to any `mas_*` key** — the only live `_role` setters are
`autonomous_loop.py:2698/2738` (`lite_trader`/`lite_risk_judge`, not EFFORT_DEFAULTS keys →
KeyError-caught → model-prefix fallback). So `EFFORT_DEFAULTS[mas_*]` is consumed by NOTHING
at runtime except the unit test. **Reverting mas_* max->baseline is a runtime no-op on the
live per-ticker trading path** ($0 metered delta, no behavior change); it only realigns the
config's stated intent + touches one test assertion + the stale mao:164 comment.

## CLAUDE.md-authoritative target values (per role)
| Role | Model pin | Current effort | CLAUDE.md / doc target | Source |
|------|-----------|----------------|------------------------|--------|
| mas_communication | sonnet-4-6 | max | **low** | model_tiers baseline; Sonnet 4.6 low-volume notify |
| mas_main | opus-4-8 | max | **xhigh** | CLAUDE.md Layer-2 "mas_main runs at xhigh"; Opus 4.8 doc |
| mas_qa | opus-4-8 | max | **high** | CLAUDE.md Layer-2 "mas_qa historically high ... per ticker" |
| mas_research | sonnet-4-6 | max | **medium** | baseline; Sonnet 4.6 recommended default=medium |
| autoresearch_fast | haiku-4-5 | None | None (unchanged) | Haiku not effort-supported |
| autoresearch_smart | sonnet-4-6 | medium | medium (unchanged) | — |
| autoresearch_strategic | opus-4-8 | high | high (unchanged) | — |

NOTE: the reconciled target == the "Pre-23.2.2 values (for revert)" already written at
`model_tiers.py:257-258`. Layer-3 subagents (qa.md/researcher.md `effort: max`) are a
SEPARATE system — CLAUDE.md phase-29.2 codifies them PERMANENT at opus+max on the flat-fee
Max rail ($0 metered, rare-event). **Do NOT "revert" the Layer-3 subagent effort:max** —
that would contradict CLAUDE.md. Only Layer-2 `EFFORT_DEFAULTS[mas_*]` is in scope.

## Recommendation — criterion-1 path: **(a), executed as a documented no-op** (a+doc hybrid)
Revert `EFFORT_DEFAULTS` mas_* to the CLAUDE.md baseline (communication=low, main=xhigh,
qa=high, research=medium) NOW, AND record the config-vs-runtime drift as intentional:
1. It is verified SAFE — a runtime no-op on the live MAS path (that raw-SDK path uses the
   API default `high` by design and does not consult EFFORT_DEFAULTS). No metered-cost delta,
   no trading-behavior change → no CLAUDE.md sign-off needed because we are MATCHING CLAUDE.md,
   not deviating from it.
2. It removes a latent TRAP: if a future "config==runtime" fix wires `resolve_effort(_role)`
   into the MAS path while EFFORT_DEFAULTS still says max, the live per-ticker roles would
   silently jump high->max (a real cost spike the effort doc warns causes "overthinking").
3. **Do NOT wire resolve_effort into the MAS path in this step** — that (and only that) would
   convert the revert into a real functional change; it must stay a separate, sign-off-gated
   step. Satisfy criterion 1 via the doc-the-drift branch, not the wire branch.
4. Required co-changes: update mao:164 comment ("effort=max" -> "API default high; EFFORT_DEFAULTS
   not consulted on this raw-SDK path"); update `test_phase_59_1_fable_adoption.py:50`
   assertion max->xhigh; verify EFFORT_SUPPORTED_MODELS / MODEL_EFFORT_FALLBACK unchanged.

Reject **(b)** (document max as intentional): contradicts actual runtime (high), contradicts
the code's own "revert after closure" note, and preserves the max trap. Reject pure **(c)**
(gate revert behind operator sign-off): over-cautious for a verified no-op that ALIGNS to the
already-authoritative CLAUDE.md; leaves the trap in place. If Main wants max caution, (c) is an
acceptable fallback but I advise against it.

## Criterion 2 — Main's Layer-3 tier + fallbackModel
- **effortLevel: keep `xhigh`** (already pinned at settings.json:2 — deterministic). State the
  choice: Opus 4.8 doc says "start with xhigh for coding and agentic use cases"; Main is a
  long-running agentic coding orchestrator = the textbook xhigh case. `max` is "reserved for
  genuinely frontier problems ... adds significant cost for relatively small quality gains" +
  overthinking risk → NOT chosen. Consistent with CLAUDE.md (Main=xhigh).
- **fallbackModel: change `["claude-opus-4-8","claude-sonnet-5"]` -> `["claude-sonnet-5","claude-haiku-4-5"]`.**
  Drop the primary-equal first hop (when primary Opus 4.8 hits 529, falling back to Opus 4.8
  re-hits the same overloaded pool = wasted hop; dedup is within-list, not vs primary). Add
  `claude-haiku-4-5` as the availability FLOOR — "almost never overloaded" (source #5), matching
  Anthropic's own opus->sonnet->haiku chain shape. Fallback fires per-turn only + covers
  overload-class only (rate/usage-limit never switch — accurate, keep the statusMessage note).
  Keep Fable OUT of the chain (it auto-falls-back to Opus on flagged prompts + draws credits).

## Criterion 3 — prune stale Fable-window comments
qa.md:7-44 and researcher.md:7-42 narrate the expired Fable window. Model pins are already
`opus` (67.4). PRUNE the Fable narration to a one-line opus-steady-state note; **KEEP the
`effort: max` VALUES** (CLAUDE.md-authoritative for Layer-3 subagents) and the STALL-WATCH /
roster-snapshot operational notes if concise. No functional model/effort change — comments only.

## Research Gate Checklist
Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5: 4 official + 1 practitioner)
- [x] 10+ unique URLs total (5 read + 7 snippet = 12)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim
Soft checks:
- [x] Internal exploration covered every relevant module (model_tiers, llm_client, mao, settings, qa.md, researcher.md, test)
- [x] Contradictions/consensus noted (config-vs-runtime drift; stale mao:164)
- [x] All claims cited per-claim

## JSON envelope
```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 7,
  "urls_collected": 12,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "coverage": {
    "audit_class": false,
    "rounds": 1,
    "dry_rounds": 0,
    "K_required": 2,
    "new_findings_last_round": 0,
    "dry": false
  },
  "summary": "EFFORT_DEFAULTS mas_*=max is a stale 23.2.2 step-scoped override. CRUX: the live Layer-2 MAS path (multi_agent_orchestrator raw Anthropic SDK, mao:243/1098/1146/1267) passes NO output_config.effort, so runtime=API-default high; resolve_effort/EFFORT_DEFAULTS is read ONLY at llm_client.py:1509 (ClaudeClient wrapper), which the MAS bypasses, and no live caller sets config['_role'] to a mas_* key. So reverting mas_* max->baseline (comm=low/main=xhigh/qa=high/research=medium, already spelled at model_tiers.py:257) is a runtime no-op on the live per-ticker path -> criterion 1 via path (a) documented-drift, NOT wiring (wiring+max = latent cost trap). Update mao:164 stale comment + test_phase_59_1:50 assertion. Criterion 2: keep effortLevel=xhigh (Opus 4.8 doc: start xhigh for agentic; max=frontier-only, overthinking risk); fallbackModel -> [sonnet-5, haiku-4-5] (drop primary-equal opus hop, add Haiku floor). Criterion 3: prune Fable comments in qa.md/researcher.md but KEEP effort:max (CLAUDE.md Layer-3 permanent). Layer-3 subagent effort != Layer-2 EFFORT_DEFAULTS.",
  "brief_path": "handoff/current/research_brief_71.5.md",
  "gate_passed": true
}
```
