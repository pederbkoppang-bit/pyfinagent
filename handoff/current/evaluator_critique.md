# Evaluator Critique — Step 71.2 (Layer-2 honesty: structured outputs + kill silent-failure classes)

**Evaluator:** fresh, independent Q/A via `.claude/workflows/qa-verdict.js` (Workflow structured-output, Opus 4.8,
`effort:max`, `model:opus`, $0 Max rail). Verdict = captured return value; transcribed VERBATIM by Main.

---

## Cycle 1 — VERDICT: CONDITIONAL (run wf_4c499286-704; transcribed verbatim)

**ok:** false | **verdict:** CONDITIONAL | **harness_compliance_ok:** true | **certified_fallback:** false
**violated_criteria:** `["qa_md_1a_python_lint_gate_ruff_F401"]`

**reason (verbatim):**
> All 4 immutable criteria are independently VERIFIED MET (C1 both Claude sites route through _call_agent_json with
> output_config.format json_schema, subset-compliant schemas, decision logic byte-identical to legacy; C2 clobber
> else-branch is now `return None` with a genuine red->green test that would fail pre-fix; C3 both spot-check
> methods fully deleted, zero external callers, immutable grep exit 0 with no 1.02/0.95/0.99 literal; C4
> LOOSE_DSR_MIN==0.95, rubric is an f-string so {LOOSE_DSR_MIN} renders 0.95 -> prompt byte-identical, shared
> _call_agent untouched, no model/effort change). Harness compliance is clean 5/5 and no unintended production code
> changed. The SINGLE blocker: the qa.md §1a Python lint gate returns exit 1 -- ruff F401 flags an unused
> `import pytest` at test_phase_71_2_layer2_honesty.py:18 (this step) plus a pre-existing unused `import os` at
> evaluator_agent.py:34. Per qa.md §1a a non-zero lint exit blocks PASS; because no immutable criterion is missed
> and the fix is 1-2 import lines, this is a fixable CONDITIONAL, not a FAIL.

**violation_details (verbatim):** `{violation_type: Threshold_Not_Met, action: "uvx ruff check --select
F821,F401,F811 ...", state: "exit=1; 2 F401 -- unused import pytest (test, NEW) + unused import os
(evaluator_agent.py:34, pre-existing, in a modified file)", constraint: "qa.md §1a Python lint gate: non-zero ruff
exit blocks PASS"}`

**notes (verbatim excerpt):** SUBSTANCE IS SOUND -- this CONDITIONAL is purely the lint-gate hygiene blocker; the
four immutable criteria are all met and fail-safe by construction. Verified in depth: _call_agent_json degrades to
_gemini_text_call on auth-error and to the shared _call_agent on any other error; the shared _call_agent def is NOT
in the diff. The clobber red->green test genuinely reaches the fixed else-branch. The prompt is an f-string so
{LOOSE_DSR_MIN} renders 0.95 (byte-identical). FO-71.2-A is an HONEST deferral (the grep is satisfied by the
orchestrator; _call_model is Gemini; deferring the live evaluate_proposal path is prudent).
> NON-BLOCKING SCOPE-HONESTY OBSERVATION on C1: "worst-case == today's behavior" is slightly optimistic for one
> rare sub-path -- when Anthropic is DOWN, the reworded JSON prompt goes to Gemini WITHOUT constrained decoding,
> and if Gemini wraps output in ```json fences, json.loads fails and the legacy text parser finds no scores -> the
> gate returns None (keeps original). Still fail-safe (no crash/clobber) but loses answer-improvement on that
> sub-path. The classifier path is robust (parse_llm_classification strips fences). Rare degraded path, fail-safe,
> violates no immutable criterion.

---

## Cycle-1 → fix (Main; per canonical cycle-2 flow — fix blocker + address the honest finding + fresh respawn)

1. **Lint blocker (required for PASS):** removed the unused `import pytest` (test file) + the pre-existing unused
   `import os` (evaluator_agent.py:34). `uvx ruff check --select F821,F401,F811` now **exit 0** (re-run, verified).
2. **C1 Gemini-fallback robustness (the honest non-blocking finding — fixed at the source, not just documented):**
   the gate's structured parse now **strips ```json code fences** before `json.loads` (mirrors
   `parse_llm_classification`). So on the Anthropic-DOWN + Gemini-fenced-JSON sub-path the structured parse still
   works → the gate keeps its answer-improvement ability. With this, post-71.2 is **≥ today on every path**. New
   test `test_c1_structured_fenced_json_still_parses` proves it. experiment_results "worst-case" claim corrected.

Re-verified deterministically after the fixes: ruff exit 0; immutable verification exit 0; **28 tests pass**
(11 new 71.2 incl. the fence test + 17 regression across evaluator/fallback/classification); both modules import.
The changes are import-lines + an additive fence-strip (strictly a superset) → the four criteria remain verified.
Evidence CHANGED → a **fresh** Cycle-2 Q/A runs on it (documented fix-then-respawn; NOT verdict-shopping).

## Cycle 2 — VERDICT: PASS (fresh Q/A on the lint-clean + fence-robust code; run wf_583e066f-aa5; transcribed verbatim)

**ok:** true | **verdict:** PASS | **harness_compliance_ok:** true | **certified_fallback:** false
**violated_criteria:** [] | **violation_details:** []

**reason (verbatim excerpt):**
> All 4 immutable criteria are independently VERIFIED MET and the cycle-1 lint blocker is RESOLVED (qa.md §1a ruff
> F821,F401,F811 now exits 0 -- "All checks passed!"). C1: both Claude JSON sites now route through the fail-safe
> helper _call_agent_json with output_config={"format":{"type":"json_schema","schema":...}}; I confirmed via $0
> local SDK introspection that the PINNED anthropic 0.96.0 exposes output_config on messages.create, so on
> claude-sonnet-4-6 (GA) this is genuine constrained decoding; QUALITY_VERDICT_SCHEMA + CLASSIFY_SCHEMA are
> subset-compliant and CLASSIFY_SCHEMA field names match parse_llm_classification exactly. C2: the clobber
> else-branch is now `return None, usage`; I traced the caller -> the ORIGINAL analyst answer stands; test_c2 is a
> genuine red->green. C3: both spot-check methods fully deleted (zero external callers), the 1.02/0.95/0.99 dict +
> CONDITIONAL->PASS flip are gone, immutable grep exit 0. C4: LOOSE_DSR_MIN==0.95 (byte-identical, asserted); no
> model/effort change; thresholds byte-identical; shared _call_agent untouched. Deterministic: lint exit 0,
> immutable exit 0, 28 tests pass, both import, no unintended production code changed. Harness 5/5. FO-71.2-A
> deferral is HONEST. Every change is fail-safe; worst case is today's behavior, strictly better (fail-safe None,
> no clobber) on the unparseable path.

**notes (verbatim excerpt):** CYCLE 2, fresh Q/A on CHANGED evidence (documented fix-then-respawn). Harness 5/5
(research gate gate_passed=true 5 sources; contract-before-generate mtime-proven; results present; log-last — 71.2
not in harness_log, masterplan in-progress; no verdict-shop). NON-BLOCKING OBSERVATIONS (no verdict effect): (a)
the structured parse decides PASS/FAIL from the four scores and ignores the schema's `verdict` enum — intentional,
byte-identical to the legacy score-driven block. (b) _call_agent_json returns text-or-"" vs _call_agent's
text-or-"No response." — benign, still fail-safe. (c) no live end-to-end MAS Claude call (metered, needs Peder
approval per CLAUDE.md); the 28 direct-method tests + import smoke are the substantive exercise for an internal
fail-safe correctness change. Live book untouched; historical_macro FROZEN; harness stays exactly 3 agents.

## Main's disposition (recorded; not a verdict edit)
- Both Cycle-1 items were fixed at the SOURCE, not just documented: the ruff blocker (unused imports removed → exit
  0) and the honest degraded-path finding (fence-stripping added → the gate now works on the Gemini-fallback JSON
  path; post-71.2 is ≥ today on every path). The three Cycle-2 non-blocking observations are accurate and accepted
  as-is (all intentional/benign/fail-safe).
- **FO-71.2-A** (Gemini structured output on `evaluator_agent._call_model`) remains an honest deferral — not
  criterion-required (`_call_model` is Gemini; the grep is satisfied by the orchestrator), and prudently avoids
  risking the high-frequency live `evaluate_proposal` path. Recommended approach recorded in experiment_results.
