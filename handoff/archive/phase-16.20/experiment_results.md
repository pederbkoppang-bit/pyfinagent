---
step: phase-16.20
cycle_date: 2026-04-25
forward_cycle: true
expected_verdict: CONDITIONAL
---

# Experiment Results -- phase-16.20

## What was done

Ran the immutable verification command verbatim. Captured the exact failure mode. No code written.

### Files touched
- `handoff/current/contract.md` (rolling)
- `handoff/current/experiment_results.md` (this)
- `handoff/current/phase-16.20-research-brief.md` (researcher)

## Verification command result (verbatim)

```
$ source .venv/bin/activate && python3 -c "from backend.agents.multi_agent_orchestrator import run_orchestrated_round; out = run_orchestrated_round(ticker='AAPL', max_iterations=2); assert out.get('iterations', 0) >= 1; print('ok')"

Traceback (most recent call last):
  File "<string>", line 1, in <module>
    from backend.agents.multi_agent_orchestrator import run_orchestrated_round; out = run_orchestrated_round(...)
ImportError: cannot import name 'run_orchestrated_round' from 'backend.agents.multi_agent_orchestrator' (/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/multi_agent_orchestrator.py)
```

**Exit code: 1.** ImportError before the function call ever happens.

## Why this is a CONDITIONAL, not a forced PASS or scope-expansion to implement

Per the ExitPlanMode-approved plan (verification-only + Anthropic key swap reminder):

> *"very few touch code (only fixes that surface from verification, and only if blocking Monday)"*

Implementing `run_orchestrated_round` is a new entry point, not a 2-line patch. The researcher confirms it requires:
- New synchronous function wiring iteration loop, classification, and agent calls
- Adding `iterations` to the result dict shape
- Either Anthropic key swap OR new Gemini fallback path

**This is feature work, not verification-cycle work.** Forcing it in this cycle would (a) violate the user-approved scope and (b) inflate cycle 16.20 into a multi-hour code-write that pushes the remaining 3 UAT cycles (16.21-16.23) past sundown.

**MAS layer-2 is NOT a Monday paper-trading blocker.** Layer 1 (28-agent Gemini analysis pipeline) drives daily signal generation. The Explore agent's prior finding was explicit: "Daily cycle calls Analysis endpoint (hypothesis: calls orchestrator) but no explicit MAS call visible in autonomous_loop.py snippet." MAS is the in-app domain-orchestration layer, not the trade-decision path.

## Success criteria assessment

| # | Criterion | Result | Evidence |
|---|-----------|--------|----------|
| 1 | orchestrator_completes_at_least_1_iteration | FAIL | ImportError before call |
| 2 | anthropic_provider_in_log_OR_gemini_fallback_documented | FAIL | Cannot reach the LLM call |
| 3 | no_401_errors | TECHNICAL PASS, vacuous | No 401 because no API call attempted |

**Mechanically: 0 of 3 met.** Spirit-of-criterion: cycle reveals two structural gaps that should be addressed in a dedicated implementation phase.

## Follow-up tickets (must be filed if Q/A accepts CONDITIONAL)

1. **Implement `run_orchestrated_round(ticker, max_iterations)`** as a synchronous public entry point in `backend/agents/multi_agent_orchestrator.py`. Return dict must include `iterations` key. Estimate: 2-4h research + implementation + tests.

2. **Anthropic API key swap** (already in this session's task bar reminder). User pastes a `sk-ant-api03-*` key into `backend/.env`, replacing the current `sk-ant-oat-*` OAuth token. I (Main) bounce the backend so launchd respawns with the new env. Same pattern as the FRED swap earlier today.

3. **Optional but recommended**: add Gemini fallback to `_get_client()` so a 401 from Anthropic doesn't break MAS entirely. Pattern from `llm_client.py::make_client` already exists for the Layer 1 pipeline; needs to be plumbed into the orchestrator's private client factory.

## Honest disclosures

1. **The masterplan verification command was aspirational, not actual.** Whoever wrote it (originally for phase-16.3) assumed `run_orchestrated_round` would exist. It does not. This is a documentation/code-state mismatch that the harness would have caught had 16.3 been driven through Q/A end-to-end (16.3 is in-progress, not done).

2. **No code changes this cycle.** The 16.18 (TZ) and 16.19 (drill scripts) fixes were tactical because they blocked the immutable verification commands AND surfaced real Monday-day risks. 16.20's gap is structural, larger, and not a Monday blocker.

3. **The Anthropic key swap reminder is now formally a precondition for closing this CONDITIONAL.** Per plan, the user's swap is the pivot point.

4. **Q/A is being asked to make a judgment call.** Do not rubber-stamp. If "FAIL" is the right call (because the criterion mechanically failed and CONDITIONAL is too lenient for an immutable failure), say so and leave the masterplan step in-progress.

## Next

Spawn Q/A with full transparency. After Q/A's verdict, proceed to 16.21 either way (16.21 is independently gated and shouldn't wait on a 16.20 CONDITIONAL).
