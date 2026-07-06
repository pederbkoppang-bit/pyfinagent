---
name: cc-rail-guard-66-1
description: phase-66.1 audit — zero-pages root cause is 4 broken `backend.services.alerting` imports in autonomous_loop; probe already runs at cycle start but gates nothing; cc failures propagate as EMPTY LLMResponse not exceptions
metadata:
  type: project
---

phase-66.1 research (2026-07-07), cc_rail 100%-failure 06-15..07-06 (~162 doomed calls/cycle):

- **ZERO-PAGES ROOT CAUSE**: `backend/services/alerting.py` DOES NOT EXIST (find_spec=None; services/__init__.py empty, no shim). FOUR in-cycle P1 sites import it and are swallowed by fail-open excepts: autonomous_loop.py:220 (56.2 rail_down probe P1), :751 (conviction_overlay_degraded), :923 (degraded_scoring), :957 (fallback_rate). Correct module: `backend.services.observability.alerting` (used at :1415/:1438, cycle_health.py, kill_switch.py). ALL FOUR alarm systems designed to catch the outage were dead code.
- Probe `claude_code_health_probe` (claude_code_client.py:236, `claude auth status`, never raises) ALREADY runs at cycle start since 56.2 (autonomous_loop.py:209-234, stamps summary["claude_rail_healthy"] :217) — it just gates nothing and its page is dead.
- **Failure propagation**: `ClaudeCodeClient.generate_content` CATCHES ClaudeCodeError and returns EMPTY LLMResponse (:399-413) — no exception upstream, so `_generate_with_retry` (orchestrator.py:755, debate.py:58, risk_debate.py:53, max_retries=3) does NOT retry cc failures; ~162/cycle = fan-out width not retry amplification. Breaker counter must live INSIDE claude_code_client (beside `_log_cc_call(ok=False)` :404) — nothing upstream sees failures. Breaker-open short-circuit must return the same empty-response shape (a transient-named exception would TRIGGER 3x retries via orchestrator.py:856 substring match).
- **P1 dedupe**: P1 in `_CRITICAL_SEVERITIES` bypasses threshold AND repeat window (alerting.py:46, :75-80) — every P1 call pages. Exactly-once = caller-side latch on the closed->open transition. Webhook empty on this machine -> `_bot_token_fallback` :123-163 (chat.postMessage, channel default C0ANTGNNK8D) auto-selected for critical sevs at :197-205.
- **Rail topology**: settings.gemini_model DEFAULT "claude-sonnet-4-6" (settings.py:30) -> orchestrator general_client :592 + quant_exec_client :599 on cc rail; deep_think_model=gemini-2.5-pro (Moderator/Critic/Synthesis on Gemini). make_client cc-route gate llm_client.py:1963-1977; routing-breach ValueError :1987-1996 forbids Anthropic-direct fallthrough — any rail-down fallback must be Gemini, config-gated.
- cycle_history row extension precedent: `meta_scorer_degraded` kwarg on record_cycle_end (cycle_health.py:301/:322). 2026-07-06 row still shows meta_scorer_degraded=true post-credential-restore — check in 66.2.
- Claude Code docs (2026): auth status exits 0/1; no enumerated exit-code table (branch zero-vs-nonzero); stream-json api_retry carries `authentication_failed` etc.; docs now recommend `--bare` for scripts but --bare SKIPS keychain OAuth -> would kill the Max rail (keep no---bare rule, claude_code_client.py:122-126).

**Why:** these six facts are non-derivable without the full trace and directly shape 66.1 GENERATE + 66.2 funnel diagnosis.
**How to apply:** any future alerting wiring must import `backend.services.observability.alerting`; any cc-rail guard/counter goes in claude_code_client.py; verify import paths with find_spec in tests. See [[backend-restart-safety]] for restart interplay.
