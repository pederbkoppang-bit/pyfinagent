# Contract -- 66.1 Restore the decision path (goal-phase66-reactivation)

Step: 66.1 | Cycle 68 | 2026-07-06/07 | Operator present

## Research-gate summary

research_brief_66.1.md (tier moderate, gate_passed: true; 7 read-in-full / 80 URLs /
recency scan / 15 internal files). Load-bearing findings:
- ZERO-PAGES ROOT CAUSE: `backend/services/alerting.py` DOES NOT EXIST; four in-cycle
  P1 sites import it (autonomous_loop.py:220 rail-probe, :751 conviction, :923
  degraded-scoring, :957 fallback-rate) and the ModuleNotFoundError dies inside
  fail-open excepts. Correct module: `backend.services.observability.alerting`
  (already used correctly at :1415/:1438). Pure bug fix -> ships live, test-covered.
- The health probe ALREADY runs at cycle start (autonomous_loop.py:209-234, phase-56.2)
  and stamps summary["claude_rail_healthy"]; it gates nothing. 66.1 adds the gate.
- cc_rail failures propagate as EMPTY LLMResponse, not exceptions
  (claude_code_client.py:399-413); orchestrator's _generate_with_retry
  (orchestrator.py:755, max_retries=3) never retries them. ~162 calls/cycle = fan-out
  width (default gemini_model `claude-sonnet-4-6` -> general/quant_exec clients via
  make_client, llm_client.py:1963-1977). Single choke point: claude_code_invoke (:80);
  breaker counter lives beside _log_cc_call(ok=False) (:404).
- P1 delivery: backend.services.observability.alerting auto-selects the live-proven
  _bot_token_fallback (:123-163) when webhook empty (:197-205); P1 BYPASSES the
  consecutive-failure deduper (:46/:75-80) -> exactly-once requires a CALLER-SIDE latch
  on the closed->open transition (Fowler/Azure/PagerDuty alert-on-transition-only).
- Conventions: bounded Field + default-OFF flag (settings.py:305/:311);
  record_cycle_end kwarg precedent (cycle_health.py:301/:322) for
  rail_skipped/breaker_tripped; test pattern test_phase_60_4:40-113 + autouse
  isolation fixture (test_phase_62_2:15-19) to avoid the .env-bleed defect.
- External canon: trip on threshold-within-window, reset per window; probe doubles as
  the half-open recovery check; auth failures trip faster than timeouts; page on state
  TRANSITION only.

## Hypothesis

One import-path bug silenced every in-cycle P1 for the whole away window; with the
import fixed, a probe gate at the single choke point plus a consecutive-failure breaker
with a transition latch makes >20 silent consecutive cc_rail failures structurally
impossible, at zero behavior change when the rail is healthy (gate/breaker only alter
the already-failing path: doomed 5s subprocess calls become instant empty responses).

## Immutable success criteria (verbatim from .claude/masterplan.json phase-66/66.1)

1. "Pre-cycle health probe gates the rail: on probe failure the cycle SKIPS all cc_rail
   calls for that cycle (test evidence: forced probe failure -> zero cc_rail invocation
   attempts)"
2. "Circuit breaker: after a configurable threshold (default <=20) of consecutive
   cc_rail failures, the rail stops retrying for the cycle and emits exactly ONE P1
   page via the proven bot-token path, deduped (drill evidence: forced failure -> one
   Slack P1, permalink in live_check); no cycle may ever log >20 consecutive failed
   cc_rail calls without a page"
3. "A SCHEDULED (not manual) trading cycle after deploy writes ok=true agent LIKE
   'cc_rail%' rows to pyfinagent_data.llm_call_log (BQ row paste in live_check;
   scheduled-run evidence per the 39.1 lesson)"
4. "Degraded-mode policy (Claude-rail-down => hold, current behavior) documented; any
   Gemini-fallback alternative ships config-gated default OFF requiring an operator
   token to enable"

Verification command (immutable):
source .venv/bin/activate && python -m pytest backend/tests/test_phase_66_1_rail_guard.py -q

live_check: live_check_66.1.md with pytest output, drill P1 permalink, and the
post-deploy scheduled-cycle BQ rows.

## Plan

1. Fix the four `backend.services.alerting` imports -> `backend.services.observability
   .alerting` (bug fix, live).
2. RailGuard state in claude_code_client.py: module-level per-cycle breaker
   (consecutive-failure count, open/closed, transition latch); checked at
   claude_code_invoke entry -- when OPEN (or cycle marked probe-failed), return the
   failure envelope immediately WITHOUT spawning the CLI; on open transition, emit ONE
   P1 via observability.alerting (bot-token path) with cycle id + failure class.
3. Wire the existing probe result: autonomous_loop probe-failure branch calls
   the guard's disable_for_cycle(); cycle start calls reset_for_cycle(); summary +
   record_cycle_end gain rail_skipped / breaker_tripped.
4. settings: claude_rail_breaker_threshold (bounded Field, default 20). Gemini
   fallback NOT implemented; policy doc: rail-down => hold (docs section + flag
   placeholder only if convention requires none -> none).
5. backend/tests/test_phase_66_1_rail_guard.py (monkeypatched CLI + alerting, autouse
   env isolation): probe-fail -> zero invocations; threshold trip -> exactly one page
   (second trip in same cycle: no page); reset on cycle boundary; healthy path
   untouched; import-path regression test for the four sites.
6. Live drill: forced failure -> one Slack P1 (permalink). Restart backend so the
   running loop holds the new code (66.1 deploy precondition for criterion 3).
7. Criterion 3 is wall-clock-gated to the next SCHEDULED cycle (2026-07-07 18:00 UTC):
   expected first Q/A verdict CONDITIONAL with criteria 1/2/4 closed; a fresh Q/A
   closes criterion 3 on the BQ evidence after the cycle.

## Scope boundaries

No gate/threshold/risk-cap changes (66.2's constraint), no Gemini fallback
implementation, no sentinel/healthcheck edits (66.3/66.4), no trailing-stop code.
Healthy-path behavior byte-identical: guard logic engages only on probe failure or
consecutive-failure threshold.

## References

research_brief_66.1.md; autonomous_loop.py:209-234/:220/:751/:923/:957/:1415;
claude_code_client.py:80/:236/:399-413; orchestrator.py:592/:599/:755;
llm_client.py:1963-1977; observability/alerting.py:46/:75-80/:123-163/:197-205;
settings.py:305/:311; cycle_health.py:301/:322; Fowler CircuitBreaker; Azure
circuit-breaker + health-endpoint-monitoring; PagerDuty event dedup; Claude Code
headless docs.
