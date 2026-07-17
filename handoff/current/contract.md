# Contract — step 71.4 (independent evaluator for the self-improvement loop + coverage-gate/loop-until-dry critic)

**Phase:** phase-71 | **Step:** 71.4 | **Priority:** P2 | harness_required: true | depends_on: 71.0 (done)
**Cycle:** 1 | Date: 2026-07-17 | **Type:** LIVE Layer-2/4 code (skill self-improvement loop) — flag-gated DARK —
+ research-gate docs. $0 metered when OFF (byte-identical); paper-only; historical_macro FROZEN; live book untouched.

## Research-gate summary (gate PASSED)

Researcher via Workflow structured-output (Opus 4.8, $0), run wf_c171b7b9-98e. Envelope: **gate_passed=true**,
tier=complex, **9 external sources read in full**, 10 snippet-only, 19 URLs, recency scan, 11 internal files.
Brief: `research_brief_71.4.md`. Grounded in Anthropic evaluator-optimizer + multi-agent doer/judge separation +
6 self-modifying-agent-safety preprints (PolicyGuard/VeriGuard/ShieldAgent — independent verifier, gate-on-mutation,
fail-closed). **fail_closed=true, risk_threshold_value_change=false.** Key anchors (HEAD, 894-line file):
- `apply_modification` (skill_optimizer.py:399-451) is the write path: `skill_path.write_text(new_content)` at
  **:425** + git add/commit :446-447. Modification object `{old_text, new_text, description, hypothesis}`. Mechanical
  gating today (old_text present :410 / unique :417 / post-write re-parse-or-REVERT :436-442) — NO pre-write judge.
- **Insertion point: after the mechanical checks (:422), before the write (:424/425)** — all fields in scope.
- **ONE caller** `_run_one_iteration:754` (both propose + think_harder converge → one gate covers both). LIVE
  autonomous trigger: MetaCoordinator "skill_opt" weekly → auto-apply + auto-commit, NO human (the exact gap).
- `directive_review.py` reuse: `review_directive_diff(...) -> ReviewResult`, 5-dim rubric (accept iff mean>=0.70),
  `_fail_closed`/`_coerce_score`, anti-rubber-stamp (strips proposer self-score), `llm_call_override` seam, 13-test
  template. It has NO production caller (directive path is HITL); MIRROR the SHAPE into a new module for the skill
  path (which lacks a judge). Drift to fix: skill_optimizer.py:7/335 says "ONLY ## Prompt Template" — narrower than
  backend-agents.md's 3 modifiable sections; enforce the union-SAFE rule.

## Plan (line-anchored, fail-closed, flag-gated DARK)

### A. `backend/agents/skill_modification_review.py` (NEW, pure module — mirrors directive_review)
`review_skill_modification(old_text, new_text, description, modifiable_sections, *, llm_call_override=None) ->
SkillReviewResult`. Two-stage, FAIL-CLOSED:
1. **Deterministic pre-check ($0, no LLM):** hard-REJECT if the diff touches a section-header OUTSIDE the
   modifiable set (## Prompt Template / ## Skills & Techniques / ## Anti-Patterns), OR changes the `{{variable}}`
   token-SET (compare sets; reject only on a delta → no prose false-positives).
2. **LLM semantic judge (metered, mirrors directive_review):** 2 focused dims — `safety` ("does NOT weaken any
   existing constraint": section-scope / anti-pattern-guard removal / rule-loosening) + `factuality`
   ("`description` accurately + completely characterizes the literal old→new diff"; under-description = smuggling).
   ACCEPT iff both >= threshold. FAIL-CLOSED: empty/None/exception/non-dict/missing-dim/out-of-range → REJECT,
   scores 0.0. Reuse `_coerce_score`/`_fail_closed` shape; `llm_call_override` for deterministic tests.

### B. `backend/agents/skill_optimizer.py`
- Insert, inside `apply_modification` AFTER the mechanical checks (:422) and BEFORE the write (:424/425): if
  `settings.skill_modification_review_enabled` → call `review_skill_modification(...)`; on REJECT → log + return a
  skip result (NO write, NO commit). Gate ONLY the forward write (rider-trap #1: never the read/revert path). Fix
  the docstring drift (:7/335) to the 3 modifiable sections.
- **Flag OFF (default) → byte-identical to today** (the review block is skipped).

### C. `backend/config/settings.py`
- `skill_modification_review_enabled: bool = Field(False, ...)` — DARK-until-token.

### D. Coverage-gate / loop-until-dry docs (ADDITIVE, >=5 floor PRESERVED)
- `.claude/agents/researcher.md` + `.claude/rules/research-gate.md` + `ARCHITECTURE.md`: add a `coverage` envelope
  field `{audit_class, rounds, dry_rounds, K_required=2, new_findings_last_round, dry}` + a loop-until-dry
  completeness critic (audit-class steps only; stop at K dry rounds) + an audit-class gate clause. **The >=5-source
  floor + recency scan stay HARD requirements** — coverage can only ADD a requirement, never lower the floor.
  Cross-link (mechanics→research-gate.md, spec→researcher.md, MADR→ARCHITECTURE.md); no duplication.

### E. `backend/tests/test_phase_71_4_skill_review.py` (satisfies the `grep 71_4|skill_optim|evaluator` check)
accept / reject-weakens-constraint / reject-description-mismatch / fail-closed (LLM None+exception+non-dict) /
pre-check section-scope-reject / pre-check {{var}}-delta-reject / flag-OFF-byte-identical (apply_modification with
the flag off does not call the review). All via `llm_call_override` — deterministic, $0.

## Immutable success criteria (verbatim from masterplan.json 71.4)

1. A proposed skill/prompt modification is independently reviewed before apply_modification writes it; a diff that
   weakens a constraint or whose description does not match the diff is rejected-and-skipped; LLM-error fails closed
   (no write) -- proven by a test
2. Audit-class steps have a loop-until-dry completeness-critic option, and the research gate documents an adaptive
   coverage gate (keep going until K dry rounds; >=5-source floor preserved)
3. Grounded in the Anthropic evaluator-optimizer + multi-agent 'separate the doer from the judge' pattern (cited);
   reuses the existing directive_review pattern where possible

Verification command (immutable):
`bash -c 'grep -Eqi "review|evaluat|adversar" backend/agents/skill_optimizer.py && ls backend/tests/ | grep -Eqi "71_4|skill_optim|evaluator" && python -c "import ast; ast.parse(open(\'backend/agents/skill_optimizer.py\').read())"'`

## Boundaries (binding)
LIVE Layer-2/4 code but **flag-gated DARK-until-token** (`skill_modification_review_enabled=False` default → OFF is
byte-identical to today; proven by a flag-OFF test). The review is FAIL-CLOSED + can ONLY BLOCK a bad
self-modification, never force one (rider-trap #1: gate the forward write only, never the revert). NO risk-limit
VALUE change (adds a guard). Metered review LLM call fires ONLY when a proposal exists (i.e. the skill_opt loop is
already spending) AND the flag is ON — so OFF = $0-delta. Reuse directive_review's proven fail-closed shape. Docs
are ADDITIVE; the >=5-source floor + recency scan stay HARD. historical_macro FROZEN; harness stays 3 agents.
Independent Q/A REQUIRED (live code) — verdict transcribed VERBATIM + persisted as evaluator_critique.json (71.3).
**No agent-file edit** (researcher.md IS an agent file → separation-of-duties + roster note in harness_log).

## References
research_brief_71.4.md; design_harness_mas_71.md §71.4 (kept #7/#11); harness_proposals.json; directive_review.py
(reuse template); Anthropic building-effective-agents (evaluator-optimizer) + multi-agent-research (doer/judge).
