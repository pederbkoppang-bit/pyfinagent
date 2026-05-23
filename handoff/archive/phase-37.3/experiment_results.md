# phase-37.3 -- experiment results (Cycle 46) -- NO_OP closure

**Date:** 2026-05-23
**Cycle:** 46
**Step:** phase-37.3 -- budget_tokens deprecation cleanup (OPEN-18)
**Verdict:** PASS with honest xfail (3 operational PASS + 1 literal xfail strict)

---

## What changed (production: ZERO; tests: +130 lines / 4 tests)

| File | Change | Lines |
|---|---|---|
| `backend/tests/test_phase_37_3_budget_tokens.py` | NEW; 3 PASS + 1 xfail strict. Documents NO_OP closure rationale. | +130 |

**No production code changed.** Per researcher's NO_OP recommendation: the work is already correctly done at the API boundary.

---

## Verbatim test output

```
$ source .venv/bin/activate
$ pytest backend/tests/test_phase_37_3_budget_tokens.py -v
============================= test session starts ==============================

  test_phase_37_3_thinking_budget_used_in_gemini_path PASSED      [criterion 2]
  test_phase_37_3_no_compat_shim_remains PASSED                   [criterion 3]
  test_phase_37_3_anthropic_legacy_refs_are_wire_literal PASSED   [criterion 1 operational]
  test_phase_37_3_literal_criterion_1_unsatisfiable_until_anthropic_eol XFAIL  [criterion 1 literal, expected]

========================= 3 passed, 1 xfailed in 0.05s =========================

$ pytest backend/ --collect-only -q | tail -2
500 tests collected   (was 496; +4 net new; 0 regressions)
```

---

## NO_OP rationale (researcher-confirmed across 5 sources)

The masterplan audit_basis "should be thinking_budget per Vertex AI 2026 SDK" is FACTUALLY WRONG. Researcher confirmed (sources: platform.claude.com/docs/en/build-with-claude/extended-thinking + adaptive-thinking + effort; ai.google.dev/gemini-api/docs/thinking):

1. `budget_tokens` is the **Anthropic** API wire-literal field name inside `{"type":"enabled","budget_tokens":N}`. REQUIRED for legacy Anthropic models (Opus 4.5 and older). Opus 4.7+ deletes the field (NOT renames) in favor of adaptive thinking + `effort` parameter.
2. `thinking_budget` is the **Gemini / Vertex AI** field name in typed `ThinkingConfig(thinking_budget=...)`. Already correctly used at Gemini boundary in `llm_client.py:917`.
3. The 11 project-internal references use a lingua-franca dict `{"type":"enabled","budget_tokens":N}` that gets translated at the client boundary (Anthropic wire OR Gemini typed).

A bulk rename would break Anthropic legacy support. NO_OP closure is correct.

---

## Immutable success criteria

1. **LITERAL** `zero_budget_tokens_refs_in_backend_py_files` -- xfail strict with reason. Follow-up phase-37.3.1 (re-evaluate when Anthropic legacy-model EOL announced).
1. **OPERATIONAL** equivalent -- PASS (every remaining ref is wire-literal / dict-form / docstring; offenders=0).
2. `thinking_budget_param_used_at_all_callsites` -- PASS (Gemini boundary uses `thinking_budget=` inside `ThinkingConfig`).
3. `no_compat_shim_remains` -- PASS (direct boundary construction, not try/except alias).

---

## Honest dual-interpretation pattern

Per CLAUDE.md "honest dual-interpretation pattern (literal vs operational criterion; xfail with named follow-ups)" — this is the DOCUMENTED honest path when a literal criterion is unsatisfiable without regressing other API surface.

- `test_phase_37_3_literal_criterion_1_unsatisfiable_until_anthropic_eol` uses `@pytest.mark.xfail(strict=True, reason=...)`. The reason cites the named follow-up phase-37.3.1.
- `strict=True` means: if the test SUDDENLY passes (Anthropic legacy-model wire refs got silently deleted), pytest will fail loudly -- catches the failure mode where a future cleanup PR removes the wire refs and silently breaks Anthropic API support.
- The 3 operational tests assert the boundary translation is correct AND no compat shim has crept in.

---

## /goal integration-gate scoreboard

| # | Gate | Verdict |
|---|---|---|
| 1 | pytest count baseline (>=297) | **PASS** (500; +4 net new) |
| 2 | ast.parse green | **PASS** |
| 3 | TS build | N/A |
| 4 | Flag-default-OFF | N/A |
| 5 | BQ idempotent | N/A |
| 6 | env vars docs | N/A |
| 7 | N* delta declared | **PASS** (R; honest disclosure preserves Anthropic legacy support) |
| 8 | Zero emojis | **PASS** |
| 9 | ASCII-only loggers | **PASS** |
| 10 | Single source of truth | **PASS** |
| 11 | log-first / flip-last | **WILL HOLD** |

---

## Honest scope + closure pattern

**Trace-link closure**, not engineered. The 3 closure patterns documented in CLAUDE.md harness lessons:
- Engineered bug fix -- NOT this.
- **Trace-link** -- YES this. Researcher confirmed the work is already correctly done.
- Verification (mutation-resistance) -- partial. 3 PASS tests guard the boundary against drift.

**Follow-up: phase-37.3.1 (P3)** -- when Anthropic announces EOL for legacy-model (Opus 4.5 and older) thinking support, re-evaluate. Will be tracked in masterplan if Q/A confirms NO_OP closure here.

---

## Research-gate

Researcher SPAWNED FIRST this cycle (operator override `feedback_never_skip_researcher` honored for 2 consecutive cycles). Brief at `handoff/current/research_brief_phase_37_3.md` -- 5 sources read in full, gate_passed=true. Recommended scope (c) NO_OP closure -- implemented verbatim.

---

## Files for archive (handoff/archive/phase-37.3/)

- contract.md
- experiment_results.md (this file)
- live_check_37.3.md
- evaluator_critique.md (after Q/A PASS)
- research_brief_phase_37_3.md
