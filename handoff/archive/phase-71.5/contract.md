# Contract — step 71.5 (effort/model posture reconciliation: max-where-it-helps, cost-appropriate elsewhere)

**Phase:** phase-71 | **Step:** 71.5 | **Priority:** P3 | harness_required: true | depends_on: 71.0 (done)
**Cycle:** 1 | Date: 2026-07-17 | **Type:** config/effort-posture reconciliation + Layer-3 fallback hygiene +
Fable-comment prune. $0; local-only; NO trading-behavior change; historical_macro FROZEN; live book untouched.

## Research-gate summary (gate PASSED)

Researcher subagent (Agent tool, Opus 4.8, effort:max, $0 Max rail), brief `research_brief_71.5.md`. Envelope:
**gate_passed=true**, tier=moderate, **5 external sources read in full** (4 Anthropic/Claude Code official docs + 1
practitioner), 7 snippet-only, 12 URLs, recency scan performed, 7 internal files inspected. Main INDEPENDENTLY
re-verified all 5 crux claims (grep/sed) before writing this contract.

**Decisive finding (criterion-1 crux): config != runtime on the MAS path, in the SAFE direction.** The live Layer-2
MAS uses the RAW Anthropic SDK (`multi_agent_orchestrator.py:243 _get_client`); every `messages.create`
(mao:1098/1146/1267/1379) OMITS `output_config.effort` → runtime = API-default `high`. `EFFORT_DEFAULTS`/
`resolve_effort` is consumed ONLY at `llm_client.py:1506-1509` (the ClaudeClient wrapper), which the MAS BYPASSES;
grep confirms NO live caller sets `config["_role"]` to a `mas_*` key (only `lite_trader`/`lite_risk_judge` at
autonomous_loop.py:2698/2738, which are not EFFORT_DEFAULTS keys). Therefore `EFFORT_DEFAULTS[mas_*]=max` is **DEAD at
runtime** except the unit test `test_phase_59_1_fable_adoption.py:50`. **Reverting it to baseline is a verified
runtime no-op on the live per-ticker trading path** → NO CLAUDE.md sign-off needed (we are MATCHING CLAUDE.md, not
deviating), $0 metered, zero trading-behavior change.

## Hypothesis

Reconciling the stale `EFFORT_DEFAULTS[mas_*]=max` override back to the CLAUDE.md-documented baseline (already spelled
in the comment at model_tiers.py:257), correcting a factually-wrong comment, tightening the Layer-3 `fallbackModel`
chain, and pruning the expired Fable-window comments — all done as documentation/config hygiene with a VERIFIED
runtime no-op on the metered path — satisfies all 3 immutable criteria at $0 with no functional/trading change and no
operator sign-off (because nothing functional changes at runtime).

## Plan

### A. `backend/config/model_tiers.py` — reconcile EFFORT_DEFAULTS to the CLAUDE.md baseline [criterion 1]
- Revert the `mas_*` values from `"max"` to the documented baseline: `mas_communication="low"`, `mas_main="xhigh"`,
  `mas_qa="high"`, `mas_research="medium"` (autoresearch_* + gemini_* UNCHANGED). This is the exact set the comment
  block at :246-258 already prescribes ("Pre-23.2.2 values (for revert)...").
- Rewrite the phase-23.2.2 "STEP-SCOPED override; revert after closure" note → a phase-71.5 reconciliation note that
  RECORDS THE RATIONALE: the override closed long ago; the values are reverted to the CLAUDE.md posture; AND documents
  the drift-is-intentional fact — runtime on the live MAS path is API-default `high` (raw SDK omits effort), so these
  dict values are the *documented intent* for any future EFFORT_DEFAULTS consumer, not the current runtime. This
  satisfies criterion 1's "reconciled ... rationale recorded" AND "config==runtime (resolve_effort wired OR the drift
  documented as intentional)" via the **documented-as-intentional** branch. Do NOT wire resolve_effort into the MAS
  path (research: unnecessary; API-default high is correct; wiring adds runtime cost/complexity).
- Verify `EFFORT_SUPPORTED_MODELS` + `MODEL_EFFORT_FALLBACK` UNCHANGED (no silent effort drop).

### B. `backend/agents/multi_agent_orchestrator.py:164` — fix the factually-wrong comment [criterion 1 hygiene]
- "Layer-2 agents run at effort=max but with small configured output budgets" → correct to API-default `high` (the
  create calls omit `output_config.effort`). Pure comment fix; no code change.

### C. `backend/tests/test_phase_59_1_fable_adoption.py:50` — update the assertion to the reconciled value [criterion 1]
- `assert mt.resolve_effort("mas_main") == "max"` → `== "xhigh"`; update the inline comment
  ("mas_main runs max per EFFORT_DEFAULTS" → "mas_main = xhigh, the reconciled CLAUDE.md baseline (phase-71.5)").
- Re-run the whole test file to confirm green (the fable-effort-support assertions are independent of the change).

### D. `.claude/settings.json` — Main's Layer-3 tier + fallbackModel [criterion 2]
- Keep `effortLevel: "xhigh"` and STATE the xhigh-vs-max choice: Main = xhigh per the Opus 4.8 effort doc ("start with
  xhigh for coding/agentic"); `max` reserved for genuinely-frontier problems (doc warns of overthinking/cost). This is
  already deterministic (settings.json key, not env-dependent).
- `fallbackModel: ["claude-opus-4-8","claude-sonnet-5"]` → `["claude-sonnet-5","claude-haiku-4-5"]`: DROP the
  redundant primary-equal first hop (Opus 4.8 == Main's primary; re-hitting the same 529 pool is a no-op) and ADD a
  Haiku availability floor ("almost never overloaded"; matches Anthropic's opus→sonnet→haiku shape). The phase-67.5
  tripwire statusMessage note (overload-class-only) stays accurate.

### E. `.claude/agents/qa.md` + `.claude/agents/researcher.md` — prune expired Fable-window comments [criterion 3]
- Remove/condense the stale Fable-window narration comments (qa.md ~7-44, researcher.md ~7-42) to the opus steady
  state (the `model:` pins are already `opus` post-67.4). **KEEP the `effort: max` VALUES** — Layer-3 subagent effort
  is CLAUDE.md-PERMANENT (phase-29.2, flat-fee Max rail, rare-event roles) and is a SEPARATE system from Layer-2
  EFFORT_DEFAULTS; do NOT conflate. No functional model/effort change. Edits agent files → **separation-of-duties +
  roster-snapshot note** in the harness_log (Peder review + verify_qa_roster_live.sh; Agent-tool roster snapshots at
  session start, the Workflow path reads from disk live).

## Immutable success criteria (verbatim from masterplan.json 71.5)

1. "EFFORT_DEFAULTS is reconciled against the CLAUDE.md effort policy with the rationale recorded; config==runtime on
   the MAS path (resolve_effort wired or the drift documented as intentional); no silent effort drop
   (EFFORT_SUPPORTED_MODELS / MODEL_EFFORT_FALLBACK still correct)"
2. "Main's Layer-3 effort tier is pinned deterministically and the xhigh-vs-max choice is stated; the fallbackModel
   chain has no redundant primary-equal first hop and has an availability floor"
3. "The stale Fable-window frontmatter comments in researcher.md/qa.md are pruned/updated to the opus steady state
   (pins already reverted in 67.4); no functional model/effort change smuggled in without CLAUDE.md sign-off"

**Verification command (immutable):**
`bash -c 'python -c "import ast; ast.parse(open('backend/config/model_tiers.py').read())" && grep -Eqi "effortLevel|fallbackModel" .claude/settings.json'`

## Boundaries (binding)
$0 metered; local-only. The EFFORT_DEFAULTS revert is a VERIFIED runtime no-op on the live metered MAS path (raw SDK
omits effort; dict dead at runtime except one unit test) → NO trading-behavior change, NO operator sign-off (matching
CLAUDE.md, not deviating). settings.json fallbackModel is Layer-3 ($0 Max rail; overload-class-only). Fable-comment
pruning is pure documentation; the `model:opus` + `effort:max` VALUES stay (Fable pins stay reverted per the drain
directive; Layer-3 effort:max is CLAUDE.md-permanent). NO functional model/effort change smuggled in. Do-no-harm:
kill-switch/stops/sector-caps/DSR/PBO byte-untouched. historical_macro FROZEN; live book untouched. Agent-file edits →
separation-of-duties + verify_qa_roster_live.sh note.

## References
research_brief_71.5.md; CLAUDE.md "Effort policy (Layer-3 harness MAS)" + "Fable 5 policy" + Layer-2 in-app MAS
paragraph; backend/config/model_tiers.py:246-331; backend/agents/multi_agent_orchestrator.py:164,243,1098/1146/1267;
backend/agents/llm_client.py:1506-1531; .claude/settings.json:2-3; Anthropic effort doc
(platform.claude.com/docs/en/build-with-claude/effort); Claude Code model-config/fallbackModel docs.
