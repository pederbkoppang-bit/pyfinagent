---
step: phase-10.7.7
cycle_date: 2026-04-26
forward_cycle: true
expected_verdict: PASS
deliverables:
  - backend/meta_evolution/directive_review.py (NEW, ~225 LOC)
  - tests/agents/__init__.py (NEW, empty marker)
  - tests/agents/test_evaluator_directive_review.py (NEW, ~225 LOC, 13 tests)
---

# Experiment Results -- phase-10.7.7

## What was done

Built an independent second-opinion evaluator review gate for directive
diffs. The gate scores a `DirectiveVersion` proposal on a 5-dimension
rubric (clarity / alignment / safety / proportionality / factuality),
aggregates by mean, and ACCEPTs iff aggregate >= 0.70 (stricter than the
proposer's self-floor of 0.60). Fail-CLOSED on any LLM error -- this is
a SAFETY gate, NOT a monitoring gate (opposite discipline from the
weekly cron in 10.7.6).

The proposer's `judge_score` field is STRIPPED from the judge prompt
(test #10 verifies the literal value never appears) -- enforcing
Anthropic's "separate the agent doing the work from the agent judging
it" principle.

## Deliverables

### `backend/meta_evolution/directive_review.py` (NEW, ~225 LOC)

Module exports:
- `ACCEPT_THRESHOLD = 0.70`
- `RUBRIC_DIMENSIONS = ("clarity","alignment","safety","proportionality","factuality")`
- `ReviewResult` frozen dataclass: verdict, reason, 5 score fields, aggregate_score, raw_llm_response
- `review_directive_diff(proposal, current_directive_text, *, llm_call_override=None) -> ReviewResult`

Internal helpers:
- `_build_review_prompt(proposal, current_text)` -- explicitly excludes `proposal.judge_score`
- `_call_llm_for_review(prompt)` -- Anthropic primary -> Gemini fallback (mirrors `directive_rewriter._call_llm_for_rewrite` lines 165-209)
- `_fail_closed(reason, raw=None)` -- returns ReviewResult with verdict="REJECT" + all scores=0.0
- `_coerce_score(parsed, key)` -- pulls float in [0,1] or returns None (out-of-range -> None)

Reuses `_parse_llm_json` from `directive_rewriter.py` (single source of truth for JSON-with-fences parsing).

ASCII-only logger messages (per `.claude/rules/security.md`).

### `tests/agents/__init__.py` (NEW, empty)

New directory marker so `python -m pytest tests/agents/...` works.

### `tests/agents/test_evaluator_directive_review.py` (NEW, ~225 LOC, 13 tests)

10 from research brief test list + 3 additional defensive:

1. `test_accept_on_high_scores` -- aggregate 0.82 -> ACCEPT
2. `test_reject_on_low_aggregate` -- aggregate 0.55 -> REJECT
3. `test_accept_threshold_boundary_exact` -- 0.70 ACCEPT; 0.699 REJECT (boundary exactness)
4. `test_reject_on_llm_none_fail_closed` -- LLM returns None -> REJECT
5. `test_reject_on_invalid_json_fail_closed` -- LLM returns non-dict -> REJECT
6. `test_reject_on_override_exception_fail_closed` (defensive) -- override raises -> REJECT (no propagation)
7. `test_reject_on_missing_dimension` -- LLM omits 'factuality' -> REJECT (specific reason mentions dim)
8. `test_reject_on_out_of_range_score` (defensive) -- safety = 1.5 -> REJECT
9. `test_missing_proposed_text_reject` -- empty proposed_text -> REJECT before LLM call (spy verifies)
10. `test_proposer_self_score_stripped_from_prompt` -- literal "0.95" + word "judge_score" absent from prompt
11. `test_current_text_present_in_prompt` -- CURRENT_TEXT + proposed_text + diff_summary all in prompt
12. `test_idempotent_same_proposal_same_verdict` -- same input/override -> identical ReviewResult (dataclass equality)
13. `test_review_result_stores_raw_llm_response` (defensive) -- raw_llm_response is the parsed dict

Mock surface: `llm_call_override=lambda p: {...}`. No monkeypatch needed; no API cost.

## Verification (verbatim, immutable from masterplan)

```
$ python -m pytest tests/agents/test_evaluator_directive_review.py -v
============================== 13 passed in 0.02s ==============================
```

## Files touched

| Path | Action | Note |
|------|--------|------|
| `backend/meta_evolution/directive_review.py` | CREATED | ~225 LOC pure module |
| `tests/agents/__init__.py` | CREATED | empty marker |
| `tests/agents/test_evaluator_directive_review.py` | CREATED | ~225 LOC, 13 tests |
| `handoff/current/contract.md` | rewrite (rolling) | -- |
| `handoff/current/experiment_results.md` | rewrite (this) | -- |
| `handoff/current/phase-10.7.7-research-brief.md` | created (researcher) | -- |

NO new dependencies. NO BQ schema changes. NO modifications to
`directive_rewriter.py` (the gate is opt-in via separate function call;
wiring into `rewrite_directive` is out-of-scope per contract).

## Success criteria assessment

| # | Criterion | Result |
|---|-----------|--------|
| 1 | `tests/agents/test_evaluator_directive_review.py` exists + all pass | PASS (13/13) |
| 2 | `review_directive_diff(proposal, current_text, *, llm_call_override=None) -> ReviewResult` signature matches contract | PASS |
| 3 | 5-dim rubric scored independently | PASS |
| 4 | ACCEPT_THRESHOLD = 0.70 (boundary exactness verified by test #3) | PASS |
| 5 | Fail-CLOSED on LLM None / invalid / exception (3 separate tests) | PASS |
| 6 | Proposer's judge_score never appears in judge prompt | PASS (test #10 grep) |
| 7 | Empty proposed_text short-circuits before LLM call | PASS (spy verified no call) |
| 8 | ASCII-only logger messages | PASS |

## Honest disclosures

1. **13 tests vs 10 in contract** -- added 3 defensive: override-raises-fail-closed, out-of-range score, raw_llm_response present. Floor exceeded; not a violation.

2. **No cycle-2 fix needed.** First pytest run was 13/13 PASS.

3. **No live wiring into `rewrite_directive`.** This cycle ships the gate as a callable. Integration into the rewriter pipeline (e.g., automatic post-step or surface in 10.7.6 cycle) is out-of-scope per contract; will be addressed in 10.7.8 runbook or a follow-up.

4. **No BQ persistence of ReviewResult rows.** The dataclass is in-memory only this cycle; future telemetry sink is a 10.7.x follow-up.

5. **Adversarial-flip-attack hardening (Dec 2025 paper).** Research brief noted this concern; mitigation is judge-model-class choice (Sonnet not Haiku in production). The `_call_llm_for_review` already defaults to `claude-sonnet-4-6`. Out of scope for unit-test cycle.

6. **Reuses `_parse_llm_json` from directive_rewriter.** Avoids duplication; single source of truth for fence-stripping JSON parsing. Imported via private name (acknowledging the underscore convention) since the parser is implementation detail of the rewriter family.

## Closes

Task list item #76. Masterplan step phase-10.7.7.

## Next

Spawn Q/A.
