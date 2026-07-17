# Experiment results — step 71.4 (independent evaluator for the self-improvement loop + coverage gate)

**Phase/step:** phase-71 → 71.4 | **Date:** 2026-07-17 | **Type:** LIVE Layer-2/4 code (skill self-improvement
loop) — flag-gated DARK — + research-gate docs. $0-delta when OFF; paper-only; historical_macro FROZEN; live book
untouched.

## What was changed (every change fail-safe / additive)

### `backend/agents/skill_modification_review.py` (NEW — mirrors directive_review, FAIL-CLOSED)
`review_skill_modification(content, old_text, new_text, description, *, modifiable_sections, llm_call_override)`
→ `SkillReviewResult`. Two stages:
1. **Deterministic pre-check ($0, no LLM):** hard-REJECT on a `{{variable}}` placeholder-set delta, a section-scope
   escape (old_text's enclosing section not in the modifiable set), or a new non-allowed section header. Only rejects
   on unambiguous violations (no prose false-positives).
2. **LLM semantic judge (mirrors directive_review):** 2 dims — `safety` (doesn't weaken a constraint) + `factuality`
   (description matches the diff). ACCEPT iff mean ≥ 0.70 AND min-dim ≥ 0.5 (a single weak dim can't be averaged
   away). FAIL-CLOSED: empty/None/exception/non-dict/missing-dim/out-of-range → REJECT, scores 0.0. `llm_call_override`
   test seam. The modifiable set is the SAFE narrow `("## Prompt Template",)` (matches skill_optimizer's own rule —
   deliberately NOT widened, which would loosen).

### `backend/agents/skill_optimizer.py`
Inserted inside `apply_modification` AFTER the mechanical checks (:422) and BEFORE the write (:424/425): when
`get_settings().skill_modification_review_enabled` → call the review; on non-ACCEPT → log + `return False` (NO write,
NO commit). Gates ONLY the forward write, never the read/revert. **Flag OFF (default) → byte-identical to today**
(the block is skipped — proven by a test). Also swept a pre-existing unused `import json` (§1a lint gate, file
touched by this diff).

### `backend/config/settings.py`
`skill_modification_review_enabled: bool = Field(False, ...)` — DARK-until-token; description documents fail-closed +
gates-only-the-write + no-threshold-moved + OFF=byte-identical.

### Coverage-gate / loop-until-dry docs (ADDITIVE; >=5 floor PRESERVED)
`.claude/agents/researcher.md` (a `coverage` envelope object + an audit-class gate clause + a new "Adaptive coverage"
section), `.claude/rules/research-gate.md` (an "Adaptive coverage gate" how-to + the gate-logic clause), `ARCHITECTURE.md`
(MADR decision #5). Audit-class steps loop-until-dry (K=2 dry rounds) after the floor; `gate_passed` for an audit step
additionally requires `coverage.dry==true`; the >=5-source floor + recency scan stay HARD (an audit at 4 sources still
FAILS). Cross-linked, no duplication.

### `backend/tests/test_phase_71_4_skill_review.py` (NEW, 14 tests)
accept / reject-safety-low / reject-factuality-low / min-dim-gate / fail-closed (None + raise + non-dict + missing-dim
+ out-of-range) / pre-check (var-delta + section-scope + header-injection) / **apply_modification flag-OFF writes &
never reviews** / **apply_modification flag-ON REJECT skips the write (byte-identical file)**.

## Verification command output (verbatim)
```
$ bash -c 'grep -Eqi "review|evaluat|adversar" backend/agents/skill_optimizer.py && ls backend/tests/ | grep -Eqi "71_4|skill_optim|evaluator" && python -c "import ast; ast.parse(...)"'
VERIFICATION: PASS (exit 0)
$ uvx ruff check --select F821,F401,F811 <the 4 files>     -> All checks passed!
$ python -m pytest test_phase_71_4_skill_review.py test_skill_optimizer.py -q  -> 25 passed
$ python -c "import backend.agents.skill_optimizer"          -> imports OK; flag default = False
```
git scope: `backend/agents/{skill_optimizer,skill_modification_review}.py`, `backend/config/settings.py`,
`.claude/agents/researcher.md`, `.claude/rules/research-gate.md`, `ARCHITECTURE.md`, the new test + handoff. NO
frontend / paper-trading / risk code changed.

## Criterion evidence
- **C1** — the independent review runs BEFORE the write (apply_modification:422→424); rejects a
  constraint-weakening diff (safety-low + pre-check section-scope/var-delta) and a description-mismatch (factuality-low)
  — rejected-and-skipped (`return False`, no write); LLM-error fails CLOSED (None/raise/non-dict/missing/out-of-range →
  REJECT) — all proven by the 14 tests, incl. the flag-OFF byte-identical + flag-ON reject-skips-write integration
  tests.
- **C2** — audit-class loop-until-dry critic + the adaptive coverage gate documented in researcher.md +
  research-gate.md + ARCHITECTURE.md; the >=5-source floor is preserved (coverage can only ADD a requirement).
- **C3** — grounded in Anthropic evaluator-optimizer + multi-agent doer/judge separation (cited in the module + docs);
  REUSES the directive_review pattern (mirrored fail-closed shape, `_coerce_score`/`llm_call_override` seam,
  `_parse_llm_json`, the Anthropic→Gemini caller).

## Do-no-harm / scope honesty
LIVE code but **flag-gated DARK** (`skill_modification_review_enabled=False` default → OFF byte-identical, proven by a
test). The review is FAIL-CLOSED and can ONLY BLOCK a bad self-modification, never force one (gates the forward write
only). NO risk-limit VALUE change; the metered review LLM call fires only when a proposal exists AND the flag is ON
(OFF = $0-delta). Docs additive; the >=5 floor stays HARD. historical_macro FROZEN; live book untouched. The narrow
modifiable set was chosen for SAFETY (not widened). `researcher.md` is an agent file → separation-of-duties + roster
note in the harness_log. Activation follow-on: operator flips `skill_modification_review_enabled=true` to close the
un-reviewed-self-modification gap.
