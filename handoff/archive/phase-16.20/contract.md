---
step: phase-16.20
title: MAS orchestrator live round-trip (closes 16.3)
cycle_date: 2026-04-25
harness_required: true
forward_cycle: true
parent_phase: phase-16
---

# Sprint Contract -- phase-16.20

## Research-gate summary

`handoff/current/phase-16.20-research-brief.md`. Envelope: tier=simple, 5 in-full, 15 URLs, recency scan, gate_passed=true.

**Two structural blockers uncovered by researcher:**

1. **`run_orchestrated_round` does NOT exist** in `backend/agents/multi_agent_orchestrator.py`. The masterplan's verification command imports a function that was never written. Confirmed by Main running the literal command -- `ImportError: cannot import name 'run_orchestrated_round'`.
2. **`sk-ant-oat-*` keys are hard 401** by Anthropic Messages API (verified via official 2026 docs + GitHub #28091). The orchestrator's `_get_client()` has no Gemini fallback. So even if the function existed, current `backend/.env` would 401.

**Plan-scope reminder** (per ExitPlanMode-approved plan):
> "Verification-only sweep + Anthropic key swap reminder."
> "very few touch code (only fixes that surface from verification, and only if blocking Monday)"

Implementing `run_orchestrated_round` is a new feature, not a "few touch code" fix. MAS layer-2 (in-app Claude orchestration for domain queries) is NOT on the Monday paper-trading critical path -- Layer 1 (28-agent Gemini analysis pipeline) drives signal generation. Confirmed by the Explore agent's earlier finding: "Daily cycle calls Analysis endpoint (hypothesis: calls orchestrator) but no explicit MAS call visible in autonomous_loop.py snippet."

Therefore the disciplined call: 16.20 closes **CONDITIONAL** with the missing-function gap as a documented follow-up, NOT a forced PASS or in-scope implementation.

## Hypothesis

The verification command will fail with ImportError. Q/A is asked to rule whether (a) CONDITIONAL with explicit remediation path is acceptable for a non-Monday-blocker step, OR (b) FAIL is required because the immutable verification command cannot succeed.

## Success Criteria (verbatim, immutable)

```
source .venv/bin/activate && python3 -c "from backend.agents.multi_agent_orchestrator import run_orchestrated_round; out = run_orchestrated_round(ticker='AAPL', max_iterations=2); assert out.get('iterations', 0) >= 1; print('ok')"
```

- orchestrator_completes_at_least_1_iteration
- anthropic_provider_in_log_OR_gemini_fallback_documented
- no_401_errors

**As-written outcome:** ImportError at `from backend.agents.multi_agent_orchestrator import run_orchestrated_round`. Cannot reach the orchestrator call.

## Plan steps

1. Run the verification command verbatim (already done above).
2. Capture exit code + traceback.
3. Spawn Q/A with full transparency. Ask Q/A explicitly: is CONDITIONAL acceptable here?
4. If Q/A says CONDITIONAL is acceptable, append log + flip 16.20 to done with notes pointer + 2 follow-up tickets.
5. If Q/A says FAIL is required, leave 16.20 in-progress and continue to 16.21 (which has independent gates).

## What Q/A must audit

1. The ImportError is real (re-run the command independently).
2. The "MAS not on Monday critical path" claim is true (read autonomous_loop.py + paper_trading.py to confirm no MAS round-trip in the daily cycle).
3. The Anthropic key state in backend/.env (without reading the file -- check that ANTHROPIC_API_KEY env var is set; if its value starts with `sk-ant-oat-` per Explore report, this is the OAuth token Peder needs to swap).
4. Decide: CONDITIONAL or FAIL? Document the rationale either way.
5. If CONDITIONAL accepted: confirm the two follow-up tickets cover the gap (implement `run_orchestrated_round` + swap key).

## References

- `handoff/current/phase-16.20-research-brief.md`
- `backend/agents/multi_agent_orchestrator.py` -- function NOT present
- Earlier this session: Explore agent's "MAS orchestrator half-state" finding
- `backend/services/autonomous_loop.py` (daily cycle path -- does NOT call MAS)
- ExitPlanMode-approved plan: "verification-only sweep" scope
