# Experiment results — step 71.5 (effort/model posture reconciliation)

**Step:** 71.5 (P3, depends_on 71.0). $0; local-only; NO trading-behavior change; historical_macro FROZEN; live book
untouched. Research gate PASSED this session (research_brief_71.5.md, gate_passed=true, 5 external sources read in
full + recency scan; Main independently re-verified all 5 crux claims via grep/sed before the contract).

## What changed (6 files)

1. **`backend/config/model_tiers.py`** [criterion 1] — reverted `EFFORT_DEFAULTS[mas_*]` from the expired
   phase-23.2.2 all-`max` override back to the CLAUDE.md baseline: `mas_communication="low"`, `mas_main="xhigh"`,
   `mas_qa="high"`, `mas_research="medium"` (autoresearch_* + gemini_* UNCHANGED). Rewrote the comment block to a
   phase-71.5 reconciliation note RECORDING THE RATIONALE + the verified-no-op fact (the MAS uses the raw Anthropic
   SDK and omits `output_config.effort` → runtime = API-default `high`; `EFFORT_DEFAULTS` is consumed only at
   `llm_client.py:1506-1509`, which the MAS bypasses; so these values are the DOCUMENTED INTENT, not the current
   runtime). This is the "drift documented as intentional" branch of criterion 1; no `resolve_effort` wiring added.
2. **`backend/agents/multi_agent_orchestrator.py:164`** [criterion 1 hygiene] — corrected the factually-wrong comment
   "Layer-2 agents run at effort=max" → "run at the API-default effort (`high`)... EFFORT_DEFAULTS is not consulted
   here". Comment-only (+3/−1); no code change.
3. **`backend/tests/test_phase_59_1_fable_adoption.py`** [criteria 1 + 3 co-changes] — (a) line ~51 assertion
   `resolve_effort("mas_main") == "max"` → `== "xhigh"` (the reconciled baseline); (b) `test_layer3_agents_pin_...`
   line ~94: replaced the assertion that the EXPIRED Fable-window narration is present (`2026-06-23`/`June`/`USAGE
   CREDITS`) with the DURABLE economics rationale (`phase-29.2` / `Max rail`), since 71.5 prunes that narration.
4. **`.claude/settings.json`** [criterion 2] — `fallbackModel` `["claude-opus-4-8","claude-sonnet-5"]` →
   `["claude-sonnet-5","claude-haiku-4-5"]`: dropped the redundant primary-equal Opus-4.8 first hop (== Main's
   primary; re-hits the same 529 pool) and added a Haiku availability floor. `effortLevel` KEPT `"xhigh"` (Main =
   xhigh per the Opus 4.8 doc; `max` reserved for frontier).
5. **`.claude/agents/qa.md`** [criterion 3] — pruned the expired Fable-window comments (was ~38 lines) to a concise
   opus steady-state note. **KEPT `model: opus` + `effort: max` VALUES** (Layer-3 subagent effort is CLAUDE.md-
   permanent per phase-29.2; separate system from Layer-2 EFFORT_DEFAULTS). Documented why max-not-xhigh.
6. **`.claude/agents/researcher.md`** [criterion 3] — same prune to the opus steady state; KEPT `model: opus` +
   `effort: max`.

## Verification (verbatim)

- IMMUTABLE cmd `bash -c 'python -c "import ast; ast.parse(open('backend/config/model_tiers.py').read())" && grep -Eqi "effortLevel|fallbackModel" .claude/settings.json'` → **exit 0 (PASS)**.
- Reconciled values via `resolve_effort`: mas_main=**xhigh**, mas_qa=**high**, mas_communication=**low**,
  mas_research=**medium**. No silent effort drop: `EFFORT_SUPPORTED_MODELS` (9 entries) + `MODEL_EFFORT_FALLBACK`
  (10 entries; claude-opus-4-8→xhigh) UNCHANGED.
- `.claude/settings.json` valid JSON: effortLevel=**xhigh**, fallbackModel=**["claude-sonnet-5","claude-haiku-4-5"]**.
- Agent-file YAML valid: qa.md → model=**opus**, effort=**max**, maxTurns=30; researcher.md → model=**opus**,
  effort=**max**, maxTurns=40. `model:opus` + `effort:max` VALUES unchanged (only comments pruned).
- `uvx ruff check` on the SUBSTANTIVELY-changed files (model_tiers.py + test) → **All checks passed** (clean).
- `pytest test_phase_59_1_fable_adoption.py` → **6 passed**. Regression
  `test_phase_59_1/71_2/71_3/71_4/71_6` → **48 passed**.
- Runtime smoke: `get_settings()` loads; `model_tiers` imports; `resolve_model("mas_main")`=claude-opus-4-8,
  `resolve_effort("mas_main")`=xhigh.

## Scope honesty — pre-existing lint (NOT introduced by 71.5)

`uvx ruff check backend/agents/multi_agent_orchestrator.py` reports **17 pre-existing errors** (3× F841 unused-var at
lines 430/471/474; 14× F541 f-string-without-placeholder at 685–1580). ALL are ≥ line 430; my only change to that
file is the comment at line 164. They are legacy issues UNRELATED to effort reconciliation — sweeping 17 changes
across a 1900-line file during a config-reconciliation step would be scope creep (and the F841 fixes carry a small
behavior-change risk). Left untouched; flagged here for transparency. My 71.5 changes introduce ZERO lint errors.

## Do-no-harm / boundaries

$0 metered. The EFFORT_DEFAULTS revert is a VERIFIED runtime no-op on the live metered MAS path (raw SDK omits effort;
dict dead at runtime except the unit test) → NO trading-behavior change, NO CLAUDE.md sign-off needed (MATCHING the
policy, not deviating). settings.json fallbackModel is Layer-3 ($0 Max rail; overload-class-only per the phase-67.5
tripwire note, still accurate). Fable-comment pruning is pure documentation; `model:opus` + `effort:max` stay (Fable
pins stay reverted per the drain directive; Layer-3 effort:max is CLAUDE.md-permanent). No functional model/effort
change smuggled in. kill-switch/stops/sector-caps/DSR/PBO byte-untouched; historical_macro FROZEN; live book
untouched. **Separation of duties:** 71.5 edits `.claude/agents/qa.md` + `researcher.md` (comment prune only) → Peder
review requested + `scripts/qa/verify_qa_roster_live.sh` next session (Agent-tool roster snapshots at session start;
the Workflow qa-verdict.js/researcher paths read from disk live).

## Artifact shape
Config/doc reconciliation — no new runtime artifact. Verifiable via the immutable command + `resolve_effort(role)`
values + the settings.json/agent-file contents.
