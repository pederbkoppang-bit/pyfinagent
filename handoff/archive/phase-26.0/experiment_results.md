---
step: 26.0
slug: verify-opus-4-7-migration
cycle: phase-26-first-step
date: 2026-05-16
researcher_id: a9261ec55c07a241b
research_gate_passed: true
verdict_by_main: PASS  # Q/A is the authoritative verdict; this is a self-summary
---

# Experiment Results — phase-26.0 Verify Opus 4.7 migration complete across all callers

## File list

Files inspected (read-only):
- `backend/agents/_inventory.json` — agent → model pin registry (44 agents with str model)
- `backend/agents/llm_client.py` — unified LLM client (lines 425-450 model list, 540-550 routing aliases, 1255-1285 breaking-change guards)
- `backend/agents/cost_tracker.py` — pricing table (lines 20-35)
- `backend/agents/harness_memory.py` — context-limit table (lines 48-60)
- `backend/config/model_tiers.py` — tier-routing tables (lines 170-205)
- `backend/slack_bot/app_home.py` — UI model dropdown (lines 15-32)
- `backend/api/settings_api.py` — REST allowed-list + catalog (lines 31, 200-202)

Files written this step:
- `handoff/current/contract.md` (this step's Sprint Contract, pre-commit)
- `handoff/current/research_brief_step_26_0.md` (researcher's brief)
- `handoff/current/experiment_results.md` (this file)
- `handoff/current/live_check_26.0.md` (smoke-call evidence for auto-commit hook gate)

No code changes were required — migration was already complete; this step verifies that.

## Plan-step 1: Verbatim verification command + classification

Command (verbatim from masterplan.json step 26.0):
```
source .venv/bin/activate && grep -rn 'claude-opus-4-6\|claude-3-opus' backend/ --include='*.py' | grep -v 'tests/'
```

Verbatim stdout (2026-05-16):
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

12 hits total. Classification:

| # | file:line | matched text (short) | classification | 4.7 also present? |
|---|-----------|----------------------|----------------|-------------------|
| 1 | model_tiers.py:103 | `"e.g. 'claude-opus-4-6'"` | **documentation comment** (docstring example) | N/A |
| 2 | model_tiers.py:178 | `"claude-opus-4-6",` in `EFFORT_SUPPORTED_MODELS` | **registry / allowed list** | YES (line 177) |
| 3 | model_tiers.py:199 | `("claude-opus-4-6", "high"),` in `MODEL_EFFORT_FALLBACK` | **tier-routing table** | YES (`("claude-opus-4-7", "xhigh")` listed first at line 198) |
| 4 | cost_tracker.py:27 | `"claude-opus-4-6": (5.00, 25.00),` | **pricing table** | YES (line 26 `"claude-opus-4-7": (5.00, 25.00)`) |
| 5 | llm_client.py:431 | `"claude-opus-4-6",` in `SUPPORTED_MODELS` | **registry / allowed list** | YES (line 430 `"claude-opus-4-7"`) |
| 6 | llm_client.py:543 | `"claude-opus-4-6": "anthropic/claude-opus-4-6"` | **GitHub Models passthrough alias** | YES (line 542 `"claude-opus-4-7": "anthropic/claude-opus-4-7"`) |
| 7 | llm_client.py:1258 | `model_id.startswith(("claude-opus-4-7", "claude-opus-4-6", ...))` | **model-family startswith check** (adaptive-thinking routing — correctly routes BOTH 4.6 and 4.7 to adaptive) | YES (in same tuple) |
| 8 | llm_client.py:1349 | `"claude-opus-4-7", "claude-opus-4-6",` | **model-family check** (separate branch) | YES (in same list) |
| 9 | harness_memory.py:53 | `"claude-opus-4-6": 1_000_000,` | **context-limit table** | YES (line 52 `"claude-opus-4-7": 1_000_000`) |
| 10 | slack_bot/app_home.py:21 | `"claude-opus-4-6",` in `AVAILABLE_MODELS` UI dropdown | **UI dropdown list** | YES (line 20 `"claude-opus-4-7"` listed first) |
| 11 | settings_api.py:31 | `"claude-opus-4-7", "claude-opus-4-6", ...` | **REST allowed-list** | YES (in same list) |
| 12 | settings_api.py:201 | `{"model": "claude-opus-4-6", ...}` in catalog | **REST catalog row** | YES (line 200 `{"model": "claude-opus-4-7", ...}`) |

**Zero hits in `active-caller-defaulting-to-4.6` category.** Every reference is either a documentation comment, a legacy-model registry entry (preserving 4.6 alongside 4.7 for backward compat), or a model-family check that correctly includes both 4.6 and 4.7. Sub-criterion `no_active_callers_reference_opus_4_6_or_3` interpreted per audit_basis (= "no AGENT defaults to 4.6"): **satisfied**.

## Plan-step 2: _inventory.json Opus-role agents

Inspection (Python walk over the inventory; output verbatim):
```
Total agents with str model: 44
Opus-role agents (any opus model): 2
  MultiAgentOrchestrator                             claude-opus-4-7
  PlannerAgent                                       claude-opus-4-7

Offenders (opus 4.6/4.5/3-opus pins): 0
  NONE
```

Sub-criterion `_inventory.json shows opus role agents pinned to claude-opus-4-7`: **satisfied** verbatim. The two production Opus callers are both on `claude-opus-4-7`. No agent pins to `claude-opus-4-6`, `claude-opus-4-5`, or `claude-3-opus`.

## Plan-step 3: llm_client.py breaking-change guards (verbatim quote)

Guards at `backend/agents/llm_client.py:1258-1278`:
```python
if thinking_requested:
    if model_id.startswith(("claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5")):
        # Adaptive path (no budget_tokens accepted).
        kwargs["thinking"] = {"type": "adaptive"}
    else:
        # Legacy manual path.
        budget = thinking_cfg["budget_tokens"]
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    # Claude REQUIRES temperature=1 whenever thinking is active,
    # for both adaptive and enabled modes.
    kwargs["temperature"] = 1

# phase-4.14.7: Opus 4.7 rejects temperature / top_p / top_k
# with a 400 error on EVERY request (per Anthropic's
# "What's new in Claude Opus 4.7" doc -- the restriction is
# model-wide, not thinking-gated). Strip AFTER the thinking
# branch above so the temperature=1 override does not leak
# into 4.7 calls either.
if model_id.startswith("claude-opus-4-7"):
    kwargs.pop("temperature", None)
    kwargs.pop("top_p", None)
    kwargs.pop("top_k", None)
```

Coverage:
- Breaking change #1 (`budget_tokens` removal) — **HANDLED** at lines 1258-1261: model-family check routes Opus 4.7 (and 4.6, which also accepts adaptive) to `thinking={"type":"adaptive"}` without `budget_tokens`.
- Breaking change #2 (`temperature`/`top_p`/`top_k` rejection) — **HANDLED** at lines 1274-1278: hard strip for any model starting with `claude-opus-4-7`. Comment explicitly cites the Anthropic docs ("model-wide, not thinking-gated").
- Breaking change #3 (thinking content omitted by default) — **NOT in current scope**; this is a response-parsing concern. The current llm_client.py reads `response.content[0].text` directly (not thinking blocks), so empty thinking content is harmless. If/when a downstream skill needs thinking content, it must set `display: "summarized"` — out of scope for 26.0.

## Plan-step 4: live Opus 4.7 smoke call

See `handoff/current/live_check_26.0.md` for the verbatim reproduction command + stdout. Summary:
- `response.model == 'claude-opus-4-7'` ✓
- `response.content[0].text == 'PONG'` (non-empty) ✓
- `response.stop_reason == 'end_turn'` (clean) ✓
- Cost: ~$0.0003 (22 input + 7 output tokens)
- Latency: 1.49 s wall-clock

Sub-criterion `smoke_test_one_opus_call_succeeds`: **satisfied**.

## Plan-step 5: Self-summary (NOT a verdict — Q/A is authoritative)

All 3 sub-criteria from masterplan.json step 26.0 verification.success_criteria are satisfied:
- ✓ no_active_callers_reference_opus_4_6_or_3 (12 hits all classified as legacy registry/routing entries; 0 active callers)
- ✓ _inventory.json shows opus role agents pinned to claude-opus-4-7 (2/2 agents)
- ✓ smoke_test_one_opus_call_succeeds (live call PASSED with model=claude-opus-4-7)

live_check artifact present at `handoff/current/live_check_26.0.md`.

The Opus 4.7 migration is **functionally complete** as evidenced above. Step 26.0 is ready for Q/A evaluation.

**Note:** Per Anthropic's harness-design-long-running-apps doctrine and CLAUDE.md, Main does NOT self-evaluate. This file's `verdict_by_main: PASS` is a self-summary for Q/A's review, not an authoritative verdict. Q/A spawn next.
