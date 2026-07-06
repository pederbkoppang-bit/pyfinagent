# Claude rail degraded-mode policy (phase-66.1)

Scope: what the trading pipeline does when the Claude Code CLI rail (cc_rail,
`backend/agents/claude_code_client.py`) is unavailable -- probe failure at cycle start,
or the circuit breaker tripping mid-cycle.

## Policy: rail-down => HOLD (fail-safe, current behavior)

When the rail is gated (failed pre-cycle `claude_code_health_probe`) or the breaker is
open (>= `settings.claude_rail_breaker_threshold` consecutive failures, default 20),
every `ClaudeCodeClient.generate_content` call returns an empty `LLMResponse`
immediately -- no subprocess spawn, no `llm_call_log` row. Downstream, empty responses
flow into the pipeline's existing degraded fallbacks (no-LLM scorer paths, lite
fallbacks), which do not produce BUY conviction; the practical effect is the system
HOLDS. This is deliberate: a decision engine without its decision model must not trade
on synthetic neutral values (phase-61 decision-input-integrity doctrine).

What changed vs. the 2026-06 outage: the *outcome* (no trades) is the same fail-safe;
the *cost and observability* are fixed -- no 162 doomed 5s subprocess calls per cycle,
no phantom cost rows, a P1 page on the first cycle of rail-down (probe path,
autonomous_loop rail-probe site) or on the breaker's closed->open transition
(exactly once per cycle, bot-token delivery), and `rail_skipped` / `breaker_tripped`
persisted per cycle in cycle_history for the 66.2 funnel diagnosis.

## Gemini fallback: NOT implemented; config-gated + operator-token if ever proposed

Routing Claude-rail work to the Gemini deep path on rail-down is a TRADING BEHAVIOR
CHANGE (different model family making buy/sell-relevant judgments) and is deliberately
NOT implemented in phase-66.1. Per the immutable 66.1 criterion 4 and the standing
do-no-harm rule, any future implementation must ship config-gated DEFAULT OFF and may
only be enabled by an operator token (pending_tokens ask + recorded reply). Until
then: rail-down => hold, page the operator, recover the rail.

## Operator runbook on a rail-down page

1. `claude auth status` on the host (the probe runs exactly this, with
   ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN scrubbed -- the OAuth/keychain path).
2. If logged out / 401: `claude /login` interactively; consider `claude setup-token`
   (1-year credential) -- evaluated in phase-66.4.
3. Verify: `python -c "from backend.agents.claude_code_client import
   claude_code_health_probe; print(claude_code_health_probe())"` -> `(True, 'ok')`.
4. The next cycle's `rail_guard_reset` re-enables the rail automatically -- no flag to
   flip, no restart needed (the probe is the half-open check, Azure circuit-breaker /
   health-endpoint-monitoring pattern).

References: research_brief_66.1.md (Fowler CircuitBreaker; Azure circuit-breaker +
health-endpoint-monitoring; PagerDuty alert-on-transition; Claude Code headless docs).
