# Evaluator Critique -- phase-26.0 Verify Opus 4.7 migration complete across all callers

**Q/A agent:** qa (merged qa-evaluator + harness-verifier)
**Date:** 2026-05-16
**Spawn:** first Q/A spawn for step 26.0 (no prior verdicts)
**Authoritative verdict:** PASS

---

## Prelude

Step 26.0 is a verification step (no code changes required if migration is already complete). Main's hypothesis is that the Opus 4.6 -> 4.7 migration is functionally complete: both production Opus callers (`MultiAgentOrchestrator`, `PlannerAgent`) pinned to `claude-opus-4-7` in `_inventory.json`, breaking-change guards already in place in `llm_client.py:1258-1278`, and remaining 4.6 references are legitimate backward-compat registry entries. Q/A reproduced all deterministic checks; evidence aligns with the contract's pre-committed interpretation.

---

## Phase 1: Harness-compliance audit (run FIRST)

| # | Item | Verdict | Evidence |
|---|------|---------|----------|
| 1 | Researcher spawn | PASS | `researcher_a9261ec55c07a241b`, tier=simple, gate_passed=true. Brief: `handoff/current/research_brief_step_26_0.md`. Closing JSON envelope shows `external_sources_read_in_full: 5`, `unique_external_urls_read_in_full: 5`, `urls_collected: 15`, `recency_scan_performed: true`. 5 URLs are unique Anthropic Tier-1/2. |
| 2 | Contract pre-commit | PASS | `handoff/current/contract.md` present. Success Criteria section copies the verification command verbatim from masterplan.json step 26.0; includes Plan section (5 numbered steps) and References section. Hypothesis + Scope-honesty section explicit. |
| 3 | Results recorded | PASS | `handoff/current/experiment_results.md` present with verbatim verification-command output, 12-hit classification table, inventory inspection result, breaking-change-guards quote, smoke-call summary, file list. Spot-checks: (a) lines 41-52 match my D1 reproduction byte-for-byte; (b) inventory walk verified independently (44 total, 2 Opus, 0 offenders); (c) llm_client.py:1258-1278 quote matches file. |
| 4 | Log-last discipline | PASS | `grep -nE "phase=?26.0\|phase-26.0" handoff/harness_log.md` returns only line 18286 ("Next action: Start phase-26.0") -- no cycle entry yet. Correct: LOG appends AFTER Q/A PASS and BEFORE status flip. |
| 5 | No-verdict-shopping | PASS | First Q/A spawn for 26.0. No prior `result=PASS/CONDITIONAL/FAIL` entries for this step-id. No 3rd-CONDITIONAL trap. |

All five items PASS -> proceed to Phase 2.

---

## Phase 2: Deterministic checks

### D1. Re-run the immutable verification command

Command:
```
source .venv/bin/activate && grep -rn 'claude-opus-4-6\|claude-3-opus' backend/ --include='*.py' | grep -v 'tests/'
```

Verbatim Q/A reproduction stdout (2026-05-16):
```
backend/config/model_tiers.py:103:        The model ID string (e.g. "claude-opus-4-6").
backend/config/model_tiers.py:178:    "claude-opus-4-6",
backend/config/model_tiers.py:199:    ("claude-opus-4-6",   "high"),
backend/agents/cost_tracker.py:27:    "claude-opus-4-6": (5.00, 25.00),
backend/agents/llm_client.py:431:    "claude-opus-4-6",
backend/agents/llm_client.py:543:    "claude-opus-4-6":   "anthropic/claude-opus-4-6",
backend/agents/llm_client.py:1258:            if model_id.startswith(("claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5")):
backend/agents/llm_client.py:1349:            "claude-opus-4-7", "claude-opus-4-6",
backend/agents/harness_memory.py:53:    "claude-opus-4-6": 1_000_000,
backend/slack_bot/app_home.py:21:    "claude-opus-4-6",
backend/api/settings_api.py:31:    "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5", "claude-opus-4-1",
backend/api/settings_api.py:201:    {"model": "claude-opus-4-6",              "provider": "Anthropic",     "input_per_1m": 5.00,  "output_per_1m": 25.00},
```

12 hits, byte-for-byte identical to `experiment_results.md:41-52`. Classification verified hit-by-hit against the file contents -- zero `active-caller-defaulting-to-4.6` rows. **D1 PASS**.

### D2. Inventory inspection

Python walk over `backend/agents/_inventory.json`:
```
total_models_seen= 44
opus_agents= [('multi_agent_orchestrator', 'claude-opus-4-7'), ('planner_agent', 'claude-opus-4-7')]
offenders= []
```

Both production Opus callers pinned to `claude-opus-4-7`. Zero agents declare 4.6 / 4.5 / 3-opus. **D2 PASS**.

### D3. llm_client.py:1258-1278 guards

Read verbatim:
- Line 1258: `if model_id.startswith(("claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5")):` -- adaptive-thinking routing covers 4.7 (listed first in the tuple).
- Lines 1275-1278: `if model_id.startswith("claude-opus-4-7"): kwargs.pop("temperature", None); kwargs.pop("top_p", None); kwargs.pop("top_k", None)` -- hard strip of all three sampling params for any 4.7-prefixed model. Comment cites phase-4.14.7 + Anthropic "What's new in Claude Opus 4.7" doc.

Both breaking-change guards (#1 budget_tokens removal, #2 sampling-param rejection) covered. Breaking change #3 (thinking-content-default-empty) correctly scoped out -- llm_client reads `response.content[0].text`, not thinking blocks. **D3 PASS**.

### D4. live_check artifact

`handoff/current/live_check_26.0.md` exists. Verbatim Python stdout block (lines 42-52):
```
=== Opus 4.7 smoke call SUCCESS ===
response.model           = 'claude-opus-4-7'
response.content[0].text = 'PONG'
response.stop_reason     = 'end_turn'
response.role            = 'assistant'
response.id              = 'msg_01ViYYn5PNUWP53ijAJ5qxAy'
response.usage           = input=22, output=7
wall_clock_seconds       = 1.49
=== assertions passed ===
```

`response.model == 'claude-opus-4-7'` present and explicit. Non-empty content. Clean `end_turn`. Auto-commit-and-push hook gate will clear. **D4 PASS**.

### D5. Smoke-call reproduction

SKIPPED -- Main's live_check_26.0.md evidence is concrete, includes a specific `msg_01ViYYn5PNUWP53ijAJ5qxAy` response.id and `wall_clock_seconds: 1.49` consistent with a real API call (not fabricated values). Re-running the call would burn ~$0.0003 with no additional information. **D5 SKIPPED**.

---

## Phase 3: LLM judgments

### J1. Contract alignment
**PASS.** The 5 plan steps in `contract.md` are executed in order in `experiment_results.md`: step 1 (verification command + classification) at lines 32-72; step 2 (inventory) at 74-87; step 3 (breaking-change guards) at 89-120; step 4 (smoke call) at 122-131; step 5 (self-summary) at 133-144. No divergence from the pre-commit plan.

### J2. Scope honesty
**PASS.** Contract's "Scope honesty / out-of-scope" section (lines 37-41) explicitly disclaims: (a) no consolidation/removal of legacy 4.6/4.5 registry entries; (b) Advisor Tool is step 26.2; (c) UI default-model flip deferred to phase-27; (d) `claude-opus-4-20250514` cleanup separate. Main stayed within scope -- no code edits, only verification.

### J3. Mutation-resistance / interpretation
**PASS.** The sub-criterion `no_active_callers_reference_opus_4_6_or_3` is interpreted as "no AGENT in `_inventory.json` defaults to 4.6" rather than the literal "zero grep hits". This interpretation is:
- Pre-committed in `contract.md:24` BEFORE the verification command was run (no ad-hoc reinterpretation after-the-fact);
- Consistent with the audit_basis quote in the masterplan ("Inventory currently shows mixed claude-opus-4-7 and claude-sonnet-4-6"), which targets agent pins, not grep hits;
- Backed by hit-by-hit classification: 12 hits are docstring example (1), registry/allowed-list entries (3), pricing table (2), context-limit table (1), tier-routing (1), passthrough alias (1), model-family startswith branches that correctly include 4.7 first (2), UI dropdown that lists 4.7 first (1). None are active callers defaulting to 4.6.

The interpretation is robust. If Q/A used the literal "zero grep hits" reading, the step would be impossible to satisfy without ripping out backward-compat entries for a still-active model (4.6 has no announced retirement per Anthropic deprecation table) -- which is explicitly out of scope. **No CONDITIONAL warranted.**

### J4. Anti-rubber-stamp
**PASS.** Spot-checked the 12-hit classification table for "active caller" mislabeled as "registry":
- llm_client.py:1258 (model-family startswith for thinking routing): Correctly labeled as model-family check, NOT active caller. Routes both 4.6 and 4.7 to adaptive thinking; 4.6 also accepts adaptive per Anthropic docs.
- llm_client.py:1349 (another model-family branch): Verified line is part of a list, not a defaulting assignment.
- model_tiers.py:178/199 and cost_tracker.py:27: All have a `claude-opus-4-7` entry on the immediately preceding line (verified file:line from experiment_results.md "4.7 also present?" column).

No mislabels found.

### J5. Sycophancy check
**PASS.** Main's `verdict_by_main: PASS` is explicitly flagged at experiment_results.md:8 as "Q/A is the authoritative verdict; this is a self-summary", and again at line 144 with reference to Anthropic's harness-design doctrine and CLAUDE.md "Main does NOT self-evaluate". Evidence chain (deterministic checks + smoke call with concrete response.id) supports PASS. Correct documentation pattern, not sycophancy.

### J6. Research-gate compliance
**PASS.** Brief's "Sources read in full" section lists 5 unique URLs, all on `platform.claude.com/docs/en/about-claude/*` or `www.anthropic.com/news/`. No duplicates. All Tier-1/2 official Anthropic. Recency scan section present (2026-04-01 -> 2026-05-16). 3-variant search discipline visible at brief lines 16-18 (current-year, last-2-year, year-less canonical).

---

## Final JSON envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "violated_criteria": [],
  "violation_details": "All three immutable success criteria from masterplan.json step 26.0 are satisfied: (1) verification command reproduced exactly (12 hits, zero active-caller-defaulting-to-4.6); (2) _inventory.json shows both production Opus callers (MultiAgentOrchestrator, PlannerAgent) pinned to claude-opus-4-7, zero offenders; (3) smoke call live evidence in live_check_26.0.md shows response.model='claude-opus-4-7' with non-empty content and clean end_turn. Five-item harness-compliance audit passed: researcher gate_passed=true with 5 unique Tier-1/2 URLs read in full, contract pre-committed with verbatim success criteria and Plan, results recorded faithfully, log-last discipline observed (no 26.0 cycle in harness_log.md yet), first Q/A spawn (no verdict-shopping).",
  "certified_fallback": false,
  "checks_run": 11,
  "phase_1_audit": {
    "researcher_spawn": "PASS",
    "contract_pre_commit": "PASS",
    "results_recorded": "PASS",
    "log_last_discipline": "PASS",
    "no_verdict_shopping": "PASS"
  },
  "phase_2_checks": {
    "D1": "PASS",
    "D2": "PASS",
    "D3": "PASS",
    "D4": "PASS",
    "D5": "SKIPPED"
  },
  "phase_3_judgments": {
    "J1": "PASS -- contract plan executed verbatim, no divergence",
    "J2": "PASS -- out-of-scope section explicit and respected",
    "J3": "PASS -- interpretation pre-committed in contract, backed by audit_basis, hit-by-hit classification supports it",
    "J4": "PASS -- 12-hit classification spot-checked, no active callers mislabeled as registry",
    "J5": "PASS -- verdict_by_main flagged twice as self-summary; Anthropic doctrine cited",
    "J6": "PASS -- 5 unique Anthropic Tier-1/2 URLs read in full, recency scan present, 3-variant search visible"
  }
}
```
