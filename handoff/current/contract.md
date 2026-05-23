# phase-37.3 -- budget_tokens deprecation cleanup (NO_OP closure)

**Step id:** `37.3`
**Date:** 2026-05-23
**Mode:** EXECUTION (cycle 46) -- HONEST NO_OP closure with dual-interpretation pattern.
**Cycle:** Cycle 46 (after Cycle 45 phase-38.2).

---

## North-star delta

**Terms:** R (correctness; do not break Anthropic API support) + B (zero $).

**R:** This is an HONEST closure, not engineered work. The masterplan audit_basis ("orchestrator.py:99-117 + debate.py:63 still use deprecated budget_tokens; should be thinking_budget per Vertex AI 2026 SDK") is FACTUALLY WRONG. Researcher confirmed across 5 sources read-in-full:
- `budget_tokens` is **Anthropic's** wire-literal field name (inside `{"type": "enabled", "budget_tokens": N}`), NOT a Vertex AI field. For legacy Anthropic models (Opus 4.5 and older), `budget_tokens` IS the correct field name. Opus 4.7+ uses adaptive thinking + `effort` parameter (the field is deleted, not renamed).
- `thinking_budget` is **Gemini's** field name (typed `ThinkingConfig(thinking_budget=...)`) -- ALREADY correctly used in `llm_client.py:917`.
- The 11 project-internal references use a lingua-franca dict shape `{"type":"enabled","budget_tokens":N}` that gets translated at the client boundary (already correct for both APIs).

A bulk rename would either break the Anthropic legacy wire path OR create asymmetric churn for zero behavioral change. Honest closure is correct.

**B:** ~5 min cycle vs ~2 hour misguided refactor that would break Anthropic legacy support.

**P:** N/A. **Caltech arxiv:2502.15800 discount:** N/A.

---

## Research-gate compliance

**Researcher SPAWNED FIRST** -- brief at `handoff/current/research_brief_phase_37_3.md`. Tier=simple. 5 sources read in full, gate_passed=true. Recommendation: **(c) NO_OP closure** + 2-line test-string update + audit_basis rewrite documenting the gate is already correctly implemented at the API boundary.

Sources cited:
- https://platform.claude.com/docs/en/build-with-claude/extended-thinking (Anthropic budget_tokens)
- https://platform.claude.com/docs/en/build-with-claude/effort (Opus 4.7 effort param replaces budget_tokens)
- https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking (Opus 4.7 adaptive path)
- https://ai.google.dev/gemini-api/docs/thinking (Gemini thinking_budget)
- Internal: backend/agents/llm_client.py:907-919 (Gemini path uses ThinkingConfig), :1378-1388 (Anthropic legacy gate)

---

## Hypothesis (honest dual-interpretation)

> The 20 `budget_tokens` references in backend/ break down as:
> - 4 Anthropic wire-literal (cannot remove without breaking legacy model support).
> - 11 project-internal lingua-franca dict (correctly translated at boundary).
> - 3 documentation comments.
> - 2 test strings in test_phase_41_0_bundle_close.py:60,74 (tracking).
>
> Criterion 1 `zero_budget_tokens_refs_in_backend_py_files` is UNSATISFIABLE without
> regressing Anthropic legacy support. Apply CLAUDE.md "honest dual-interpretation
> pattern": xfail criterion 1 with a named follow-up; PASS criteria 2 and 3
> verifying the boundary translation is already correct.

---

## Immutable success criteria (verbatim from masterplan 37.3.verification)

1. `zero_budget_tokens_refs_in_backend_py_files` -- **xfail (literal)**. Operational equivalent: every remaining ref is API-required and documented.
2. `thinking_budget_param_used_at_all_callsites` -- **PASS** (Gemini boundary uses `ThinkingConfig(thinking_budget=...)`).
3. `no_compat_shim_remains` -- **PASS** (the boundary translation is NOT a compat shim; it's the canonical wire-payload construction).

Plus /goal integration gates 1-11.

---

## Files this step touches

- `backend/tests/test_phase_37_3_budget_tokens.py` (NEW, ~120 lines, 4 tests):
  - test_thinking_budget_used_in_gemini_path (criterion 2)
  - test_no_compat_shim_remains (criterion 3)
  - test_anthropic_legacy_refs_are_wire_literal (operational equivalent of criterion 1)
  - test_budget_tokens_refs_xfail_until_anthropic_deletes_legacy_models (xfail with reason)
- `backend/tests/test_phase_41_0_bundle_close.py` -- 2-line update at :60+:74 (note 37.3 closed NO_OP)
- `.claude/masterplan.json::audit_basis` for 37.3 -- amended to point at research_brief_phase_37_3.md

**NOT changed:** any production code (`orchestrator.py`, `llm_client.py`, `debate.py`, `risk_debate.py`, `multi_agent_orchestrator.py`).

---

## /goal integration gates (declared)

| # | Gate | Plan |
|---|---|---|
| 1 | pytest count >= 297 | +4 tests; baseline 496 -> ~500; 0 regressions |
| 2 | ast.parse green | will hold |
| 3 | TS build green | N/A |
| 4 | flag-default-OFF | N/A |
| 5 | BQ idempotent | N/A |
| 6 | env vars docs | N/A |
| 7 | N* delta declared | DONE (R; honest disclosure) |
| 8 | zero emojis | will hold |
| 9 | ASCII-only loggers | will hold |
| 10 | single source of truth | preserved (boundary translation is canonical) |
| 11 | log-first / flip-last | will hold |

---

## Honest scope -- this is a TRACE-LINK closure, not engineered work

Per the 3 closure patterns documented in CLAUDE.md harness lessons:
- **Engineered bug fix** -- NOT this. There's nothing broken.
- **Trace-link** -- YES this. Researcher confirmed the work is already correctly done at the boundary.
- **Verification** -- partial. 3 new tests provide ongoing mutation-resistance.

The masterplan's verification command `grep -rn 'budget_tokens' backend/ ... | wc -l == 0` cannot pass without regressing Anthropic legacy support. The xfail discipline is the documented honest path.

---

## References

- closure_roadmap.md §3 OPEN-18
- handoff/current/research_brief_phase_37_3.md (cycle 46; 5 sources; NO_OP closure recommendation)
- backend/agents/llm_client.py:907-919, :1378-1388 (boundary translation)
- backend/tests/test_phase_41_0_bundle_close.py:60,74 (residual tracking)
- /goal directive
