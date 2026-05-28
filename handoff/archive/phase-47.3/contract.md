# Contract — phase-47.3: Opus 4.8 cost_tracker pricing regression

**Cycle:** 3 of the production-ready+money push (free / no project LLM spend).
**Step:** 47.3 | **Phase:** phase-47 | **Status:** in-progress | **Harness:** required | **Tier:** moderate.

NOTE: phase-47.2 (first trade) is PARKED in-progress, blocked on operator LLM-spend approval for the
validation cycle (or tomorrow's free daily cron). Its contract content lives in
`research_brief_phase_47_2_first_trade.md` + `active_goal.md`. This contract.md now covers the active
step 47.3. Advancing 47.3 (free, code-only) keeps the goal moving per HARD STOP "priorities 3-7 shipped".

## Research-gate summary (PASSED)
Researcher `a028675973fbfefd1`, tier=moderate, `gate_passed: true`. 5 sources in full, 17 URLs,
recency scan (Opus 4.8 launched 2026-05-28), 9 internal files. Brief:
`research_brief_phase_47_3_cost_tracker.md`.

Validated: Opus 4.8 = **$5.00 / $25.00 per 1M** (input/output), identical to 4.7 (Anthropic pricing
docs + 4.8 launch + press). `cost_tracker.py:20-76` `MODEL_PRICING` has `claude-opus-4-7: (5.00,25.00)`
at :26 but NO `claude-opus-4-8` -> 4.8 falls to `_DEFAULT_PRICING=(0.10,0.40)` at :79 -> ~50x in /
~62.5x out understatement. **Two more gaps the diagnostic missed:** `settings_api.py` model allowlist
(~:31) + display-pricing table (~:214) both lack a 4-8 entry. `model_tiers.py` already 4-8 (no action).
`governance.py:84-85` stale model-agnostic estimate (out of scope). max_tokens-at-xhigh = real latent
risk but Gemini-locked today -> separate follow-up, NOT this step.

## Hypothesis
Adding `claude-opus-4-8: (5.00, 25.00)` to `cost_tracker.MODEL_PRICING` + the two `settings_api.py`
tables makes 4.8 calls cost at the real rate instead of the 50x-too-low default, restoring the
accuracy of the Compute term of Net System Alpha. A pytest guard prevents the next bump from
re-introducing the gap.

## Immutable success criteria (verbatim from masterplan.json phase-47.3)
1. MODEL_PRICING['claude-opus-4-8'] == MODEL_PRICING['claude-opus-4-7'] == (5.00, 25.00) and != _DEFAULT_PRICING in backend/agents/cost_tracker.py
2. claude-opus-4-8 present in backend/api/settings_api.py model allowlist AND display-pricing table at 5.00/25.00 (>=2 occurrences)
3. a pytest regression guard asserts the 4.8 pricing entry (== 4.7, != default) and passes; ast.parse clean on all edited files

## Plan steps
1. `backend/agents/cost_tracker.py` — insert `"claude-opus-4-8": (5.00, 25.00),` above the 4-7 entry (:26).
2. `backend/api/settings_api.py` — add `"claude-opus-4-8"` to the current-GA allowlist line; add a
   `{"model": "claude-opus-4-8", "provider": "Anthropic", "input_per_1m": 5.00, "output_per_1m": 25.00}`
   row above the 4-7 display row.
3. NEW `tests/agents/test_cost_tracker_pricing.py` — regression guard: 4.8 entry == 4.7, != default,
   AND a behavioral assertion via the actual cost-lookup path (not a manual multiply).
4. Verify: immutable command exit 0; pytest green; ast.parse clean. Fresh Q/A.

## Blast radius
Cost telemetry only (`cost_tracker.py`, `settings_api.py` display/allowlist). No trade-execution /
risk / perf-metrics path touched. No LLM spend. No production data write.

## References
- `research_brief_phase_47_3_cost_tracker.md` (gate); `roadmap_master.md` (Opus-4.8 shortlist #1)
- `backend/agents/cost_tracker.py:20-79`; `backend/api/settings_api.py:~31,~214`
- CLAUDE.md effort policy (Opus 4.8 = claude-opus-4-8, $5/$25)
