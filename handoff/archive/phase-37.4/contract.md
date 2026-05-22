# phase-37.4 -- Moderator response_schema (pre-closed by phase-37.1; regression test + observation only)

**Step id:** `phase-37.4`
**Date:** 2026-05-22
**Mode:** EXECUTION (test-only; no new source edits expected).
**Cycle:** Cycle 19 (after Cycle 18 phase-37.2).

---

## North-star delta

**Terms:** B (defensive Burn-protection) + R (audit trail).

**B:** Same B-delta as 37.1 -- closing the Moderator invalid-JSON fallback path saves the duplicated LLM call cost on every Moderator invocation. Already realized by phase-37.1's include_thoughts guard; this step locks in regression-resistance.

**R:** Audit-trail completeness. Moderator decisions are part of every cycle's debate; persisting them in structured form unblocks future "decisions consistent with portfolio exposure?" analytics.

**P:** N/A (configuration discipline, not trading logic).

**Caltech arxiv:2502.15800 discount:** N/A (no decision-quality change).

**How measured:** Pre-existing test `test_phase_37_1_debate_generate_with_retry_same_guard` already covers debate.py's include_thoughts guard; this step adds 5 explicit Moderator-named assertions for regression-resistance + grep-based live-check on next cycle.

---

## Research-gate compliance

**Researcher SPAWNED** per `feedback_never_skip_researcher`. Simple-tier brief at `handoff/current/research_brief_phase_37_4.md`:

- `gate_passed: true` (6 of 5-source floor met; +20% buffer)
- 6 external sources in full (Gemini structured-output docs, python-genai issues #782 + #637, TradingAgents arxiv:2412.20138 + CHANGELOG, AI-trading multi-agent debate)
- 6 internal files inspected
- 3-variant queries logged; recency scan performed
- Verdict: **TRUE** -- phase-37.4 is pre-closed by phase-37.1 in source code; reduces to test-file authoring + live observation

Key finding: `_MODERATOR_STRUCTURED_CONFIG` at `debate.py:47-51` already has `response_mime_type` + `response_schema=ModeratorConsensus` (since phase-3). The actual Moderator invalid-JSON-fallback observed in cycle 2 was caused by the unconditional `include_thoughts=True` -- which phase-37.1's `_generate_with_retry` guard at `debate.py:65-72` now correctly skips when `response_schema` is in the input config.

External corroboration: python-genai issue #782 explicitly notes the structured-output + thinking incompatibility. TradingAgents v0.2.5 (2026-05-11) independently adopts the same `response_schema for Gemini` pattern -- production-grade pattern at the 2026 frontier.

---

## Hypothesis

> Phase-37.1 already closed the Moderator invalid-JSON root cause (via the
> include_thoughts guard at `debate.py:65-72`). Phase-37.4 requires no new
> source-code edits; the work is:
> (a) explicit regression tests asserting Moderator structural properties
>     (`response_mime_type`, `response_schema=ModeratorConsensus`, no
>     `include_thoughts` injection when schema is set),
> (b) a live_check that captures the post-37.1 zero-warning count from
>     a future cycle.

---

## Immutable success criteria (verbatim from masterplan 37.4.verification)

1. `moderator_structured_config_gains_response_schema` -- **ALREADY MET in source** (since phase-3). Verified by new tests.
2. `live_cycle_post_change_shows_zero_moderator_invalid_json_warnings` -- **DEFERRED to live observation** (next cycle's backend.log grep, operator runbook in live_check_37.4.md).

Plus /goal integration gates 1-10.

---

## Plan steps

| # | Step | Status |
|---|---|---|
| 1 | Researcher spawn (simple tier, 5+ sources) | DONE |
| 2 | Write contract | IN FLIGHT |
| 3 | Write `backend/tests/test_phase_37_4_moderator_schema.py` (5 assertions) | NEXT |
| 4 | pytest verify (count >= 326) | NEXT |
| 5 | live_check_37.4.md + Q/A + harness_log Cycle 19 + flip status | NEXT |

---

## Files this step touches

- `backend/tests/test_phase_37_4_moderator_schema.py` (NEW, ~70 lines, 5 tests)
- `handoff/current/contract.md` (this)
- `handoff/current/live_check_37.4.md` (post-Q/A)
- `handoff/current/evaluator_critique.md` (Q/A overwrite)
- `handoff/harness_log.md` (Cycle 19 append)
- `.claude/masterplan.json` (status flip 37.4)
- `handoff/current/research_brief_phase_37_4.md` (already written by researcher)

**NOT changed:** any source code. `debate.py`, `schemas.py`, any other backend file -- UNTOUCHED. ZERO frontend changes.

---

## References

- closure_roadmap.md §3 OPEN-16 (the Moderator invalid-JSON observation in cycle 2)
- research_brief_phase_37_4.md (this cycle's researcher output, 6 sources)
- backend/agents/debate.py:47-51 (existing _MODERATOR_STRUCTURED_CONFIG -- unchanged)
- backend/agents/debate.py:65-72 (include_thoughts guard added by phase-37.1)
- backend/agents/schemas.py (ModeratorConsensus BaseModel)
- backend/tests/test_phase_37_1_risk_judge_schema.py:test_phase_37_1_debate_generate_with_retry_same_guard (pre-existing coverage of the guard)
- /goal directive (10 integration gates; researcher mandatory per feedback_never_skip_researcher)
