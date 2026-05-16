# Sprint Contract -- phase-26.0
Step: Verify Opus 4.7 migration complete across all callers

## Research Gate
researcher_a9261ec55c07a241b (tier=simple) gate_passed=true.
Brief: `handoff/current/research_brief_step_26_0.md`.
- 5 unique external URLs read in full via WebFetch (all Tier-1/2 official Anthropic). 10 snippet-only URLs collected; 15 URLs total. 3-variant search discipline applied (current-year, last-2-year, year-less canonical).
- Recency scan (2026-04-01 -> 2026-05-16) performed: Opus 4.7 GA 2026-04-16; Opus 4 / Sonnet 4 `20250514` deprecated 2026-04-14 with 2026-06-15 retirement.
- Key findings (cited per claim in the brief):
  1. Opus 4.7 model ID is `claude-opus-4-7` (dateless snapshot). Same $5/$25 per MTok pricing as 4.6.
  2. Three breaking API changes: `budget_tokens` removed (use `thinking: {"type": "adaptive"}`); `temperature/top_p/top_k` removed (any non-default -> HTTP 400); thinking content omitted by default (need `display: "summarized"`).
  3. The two production Opus callers in `_inventory.json` (`MultiAgentOrchestrator`, `PlannerAgent`) are ALREADY pinned to `claude-opus-4-7`. No Layer-1 / Layer-4 service uses Opus.
  4. `llm_client.py:1258-1278` already implements the breaking-change guards (adaptive thinking routing + temperature/top_p/top_k strip for 4.7 calls). Comment explicitly cites phase-4.14.7.
  5. Remaining `claude-opus-4-6` references in `backend/` (17 hits across 6 files) are exclusively in legacy registry/pricing/context-limit/tier tables that preserve 4.6 alongside 4.7 for backward-compat — `claude-opus-4-7` entries are already PRESENT in every one of them (verified file:line in this contract's Plan step 1). None are "active callers defaulting to 4.6 as primary".

## Hypothesis
The Opus 4.6 -> 4.7 migration is functionally complete: both production Opus callers in `_inventory.json` are pinned to `claude-opus-4-7`; the three breaking-change guards (`thinking` adaptive routing, sampling-param strip, model-string update) are already in place in `llm_client.py`; and every legacy config table that lists `claude-opus-4-6` also lists `claude-opus-4-7`. The remaining `claude-opus-4-6` references are legitimate backward-compat entries (legacy model still active per Anthropic deprecation table — no announced retirement). A single live Opus 4.7 API call should therefore succeed with `response.model == "claude-opus-4-7"`, satisfying live_check.

## Success Criteria (immutable, copied verbatim from .claude/masterplan.json step 26.0)
```
source .venv/bin/activate && grep -rn 'claude-opus-4-6\|claude-3-opus' backend/ --include='*.py' | grep -v 'tests/'
```
Plus sub-criteria:
- `no_active_callers_reference_opus_4_6_or_3` — interpretation per audit_basis: no AGENT in `_inventory.json` declares Opus 4.6 as its model (registry/pricing/tier table entries preserving 4.6 alongside 4.7 are legitimate backward-compat). Verified via inventory inspection in Plan step 2.
- `_inventory.json shows opus role agents pinned to claude-opus-4-7` — verified verbatim in Plan step 2.
- `smoke_test_one_opus_call_succeeds` — single Opus 4.7 call via `llm_client.py`, asserting `response.model == "claude-opus-4-7"` and no HTTP 400 from missing breaking-change guards.

live_check (Operator-auditable artifact): `handoff/current/live_check_26.0.md` — verbatim Python output of one Opus 4.7 call showing `response.model = 'claude-opus-4-7'` and a non-empty `response.content`.

## Plan (PRE-commit; will NOT diverge in Generate)
1. Run the verification command verbatim; capture full output to `experiment_results.md`. Classify EVERY hit as one of: {registry entry / pricing table / context-limit table / tier-routing table / passthrough-routing alias / model-family startswith check / UI dropdown / dead code / active-caller-defaulting-to-4.6}. Expected: zero hits in the last category.
2. Read `backend/agents/_inventory.json`; assert no agent declares `claude-opus-4-6`, `claude-opus-4-5`, or `claude-3-opus` as its model. Report the model assignment for every agent whose role includes Opus.
3. Confirm `llm_client.py:1258-1278` breaking-change guards cover Opus 4.7 (adaptive thinking + sampling-param strip). Quote the relevant lines into `experiment_results.md`.
4. Run a single Opus 4.7 smoke call via `python -m backend.agents.llm_client` (or equivalent thin call) with `model="claude-opus-4-7"` + minimal prompt; capture `response.model`, `response.content[0].text[:80]`, and exit code. Write to `live_check_26.0.md`.
5. Write `experiment_results.md` with: file list, verbatim verification-command output, the 17-hit classification table, inventory inspection result, breaking-change-guards quote, smoke-call output, and a one-line PASS/FAIL self-summary (note: NOT a Q/A verdict — Main does not self-evaluate; Q/A spawns next).

## Scope honesty / out-of-scope
- This step VERIFIES migration completeness; it does NOT consolidate or remove legacy `claude-opus-4-6` / `claude-opus-4-5` registry entries (those are still legitimately active models per Anthropic's deprecation table — no announced retirement for 4.6).
- This step does NOT adopt the Advisor Tool (that is step 26.2).
- This step does NOT add a UI default-model preference flip (the slack_bot dropdown already includes 4.7; whether 4.7 is the FIRST option is a UI affordance question deferred to phase-27 polish).
- This step does NOT touch `claude-opus-4-20250514` (deprecated, retires 2026-06-15) — separate cleanup, not in 26.0 scope.

## References
- Research brief: `handoff/current/research_brief_step_26_0.md`
- Masterplan step JSON: `.claude/masterplan.json` step `26.0`
- llm_client.py guard implementation: `backend/agents/llm_client.py:1258-1278`
- Inventory: `backend/agents/_inventory.json`
- External grounding (Tier-1 Anthropic docs cited in brief): models/overview, model-deprecations, migration-guide, whats-new-claude-4-7, anthropic.com/news/claude-opus-4-7
