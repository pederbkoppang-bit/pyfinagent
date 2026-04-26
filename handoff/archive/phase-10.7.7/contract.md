---
step: phase-10.7.7
title: Evaluator review gate for directive diffs
cycle_date: 2026-04-26
harness_required: true
verification: python -m pytest tests/agents/test_evaluator_directive_review.py -v
research_brief: handoff/current/phase-10.7.7-research-brief.md
---

# Contract -- phase-10.7.7

## Step ID

`phase-10.7.7` -- "Evaluator review gate for directive diffs" (`.claude/masterplan.json:3269-3277`).

## Research-gate summary

Spawned `researcher` (moderate tier). Brief at
`handoff/current/phase-10.7.7-research-brief.md`. Gate: 7 sources read in
full via WebFetch (incl. Anthropic multi-agent-research, Constitutional
AI, Anthropic harness, EvidentlyAI judge guide, Kinde calibration,
LabelYourData 2026, SIPDO arXiv 2505.19514), 17 unique URLs, recency
scan (2024-2026) performed (5 relevant findings), 5 internal files
inspected. `gate_passed: true`.

Decisive findings (verbatim from brief):
- Reuse the `llm_call_override` mock-surface pattern from `directive_rewriter.py:246` -- no monkeypatch required
- New module `backend/meta_evolution/directive_review.py` (do NOT extend `evaluator_agent.py` -- it's async, Gemini-only, scoped to backtest metrics)
- 5-dim rubric: clarity, alignment, safety/no-regression, proportionality, factuality
- Aggregate = mean; ACCEPT_THRESHOLD = 0.70 (stricter than rewriter's self-floor 0.60)
- **Fail-CLOSED** on LLM error (this is a SAFETY gate; opposite of cron's fail-open)
- The proposer's `judge_score` field MUST be stripped from the judge prompt (no rubber-stamp on self-score)
- `tests/agents/` directory does NOT exist -- create `__init__.py`

## Hypothesis

A standalone `directive_review.review_directive_diff(proposal, current_text, *, llm_call_override=None) -> ReviewResult` function provides an independent second-opinion gate on rewriter proposals. The proposer's self-score is never shown to the judge. Aggregate score >= 0.70 ACCEPTs; below or any LLM failure REJECTs (fail-closed). Pure function with mock-injection for tests.

## Immutable success criteria (verbatim from masterplan)

```
verification: python -m pytest tests/agents/test_evaluator_directive_review.py -v
```

The test module exists at `tests/agents/test_evaluator_directive_review.py` and all tests pass.

## Plan steps

1. Create `backend/meta_evolution/directive_review.py` (~180 LOC):
   - `ReviewResult` frozen dataclass (verdict, reason, 5 score fields, aggregate_score, raw_llm_response)
   - `ACCEPT_THRESHOLD = 0.70` module-level constant
   - `_build_review_prompt(proposal, current_directive_text)` -- 5-dim rubric, strips proposer's judge_score from prompt, includes both current_text + proposed_text
   - `_call_llm_for_review(prompt)` -- mirror `directive_rewriter._call_llm_for_rewrite` (Anthropic primary -> Gemini fallback). Returns parsed JSON dict or None.
   - `review_directive_diff(proposal, current_directive_text, *, llm_call_override=None) -> ReviewResult`:
     - Empty proposed_text -> REJECT (no LLM call)
     - LLM returns None / error -> REJECT (fail-closed; all scores 0.0)
     - Aggregate < 0.70 -> REJECT
     - Aggregate >= 0.70 -> ACCEPT
   - ASCII-only logger messages

2. Create `tests/agents/__init__.py` (empty marker).

3. Create `tests/agents/test_evaluator_directive_review.py` (~250 LOC, 10 tests per research brief test list):
   1. `test_accept_on_high_scores`
   2. `test_reject_on_low_aggregate`
   3. `test_reject_on_llm_none_fail_closed`
   4. `test_reject_on_invalid_json_fail_closed`
   5. `test_proposer_self_score_stripped_from_prompt`
   6. `test_missing_proposed_text_reject`
   7. `test_idempotent_same_proposal_same_verdict`
   8. `test_accept_threshold_boundary_exact` (mean=0.70 ACCEPT; mean=0.699 REJECT)
   9. `test_current_text_present_in_prompt`
   10. `test_review_result_stores_raw_llm_response`

4. Verify: `python -m pytest tests/agents/test_evaluator_directive_review.py -v`

## References

- `.claude/masterplan.json:3269-3277` -- step entry
- `handoff/current/phase-10.7.7-research-brief.md` -- research gate
- `backend/meta_evolution/directive_rewriter.py:55-97` -- DirectiveVersion dataclass shape
- `backend/meta_evolution/directive_rewriter.py:165-236` -- _call_llm_for_rewrite mirror pattern
- `backend/meta_evolution/directive_rewriter.py:239-333` -- rewrite_directive function (mock pattern source)
- `tests/meta_evolution/test_directive_rewriter.py` -- mock LLM stub template
- Anthropic multi-agent research: https://www.anthropic.com/engineering/multi-agent-research-system
- Anthropic Constitutional AI: https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback
- SIPDO: https://arxiv.org/abs/2505.19514

## Out of scope

- Wiring `review_directive_diff()` into `directive_rewriter.rewrite_directive()` as an automatic post-step. That belongs in 10.7.8 (operator runbook) or a separate cycle.
- BQ schema migration to persist ReviewResult rows (not introduced this cycle).
- Live LLM calls (tests use mock; no API cost incurred).
- Adversarial-flip-attack hardening (Dec 2025 paper finding) -- noted in research brief but out of scope; addressed by judge-model-class choice (Sonnet) in production.
