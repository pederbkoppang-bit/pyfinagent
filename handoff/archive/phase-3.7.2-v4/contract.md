# Sprint Contract -- Cycle 63 / phase-3.7 step 3.7.4

Step: 3.7.4 "A2A (or AutoGen-style) task-delegation layer with retry/expiry"

## Research gate (PASSED, dual)

**researcher** (16 URLs):
- a2a-sdk v0.3.26 (April 2026) is production-ready but mandates
  HTTP/JSON-RPC transport -- no in-process binding.
- A2A spec has Task envelope {id, contextId, status, artifacts,
  history} + tasks/cancel RPC (idempotent) but NO built-in retry
  or expiry; caller implements both.
- For a fixed 3-node in-process topology, asyncio.Queue is lowest-
  latency (<100us scheduling overhead vs A2A's 0.5-2ms loopback
  HTTP).
- Canonical retry/expiry: asyncio.wait_for + asyncio.shield +
  explicit task.cancel()/await-cancelled. Avoid @tenacity.retry on
  asyncio.Task (swallows CancelledError).
- 2s p95 budget has ~1.99s headroom after infra; risk is LLM on
  critical path, not the bus.

**Explore**:
- scripts/harness/a2a_roundtrip_test.py does NOT exist.
- 3 MCP servers (data/backtest/signals/risk) are isolated; no
  peer-to-peer communication exists today.
- asyncio.Queue used in backend/agents/mas_events.py (SSE pub/sub),
  not task delegation.
- handoff/a2a_roundtrip.json schema NOT documented; p95_ms key is
  used elsewhere (perf_tracker summarize).

## Hypothesis

Build an asyncio.Queue task-delegation bus + scripts/harness/
a2a_roundtrip_test.py that drives a Data -> Strategy -> Risk chain
20 times, each hop using asyncio.wait_for for expiry + explicit
retry loop for transient failures, + one induced transient-fail
sample. Emit handoff/a2a_roundtrip.json with p95_ms <= 2000 and
retry_on_transient_failure flag.

## Success criteria (immutable)

- task_handoff_data_to_strategy_to_risk_green
- p95_le_2s
- retry_on_transient_failure

## Verification (immutable)

python scripts/harness/a2a_roundtrip_test.py && python -c "import json; d=json.load(open('handoff/a2a_roundtrip.json')); assert d['p95_ms'] <= 2000"

## Plan

1. Create backend/agents/task_bus.py:
   - `TaskEnvelope` dataclass mirroring A2A Task shape (id,
     context_id, status, artifacts, history, created_at, expires_at).
   - `AsyncTaskBus` class: per-agent asyncio.Queue; `delegate(target,
     envelope, timeout=0.5)` method with retry + expiry + cancel.
   - Fixed 3-node topology wiring: data_agent, strategy_agent, risk_agent.

2. Create scripts/harness/a2a_roundtrip_test.py:
   - Spawn 3 "agents" as asyncio tasks each reading their Queue.
   - Data: receives {ticker} -> emits {ticker, signal_seeds}.
   - Strategy: receives seeds -> emits {ticker, candidates[5]}.
   - Risk: receives candidates -> emits {ticker, approved: [...]}.
   - Run 20 round-trips; inject 1 transient failure that retries.
   - Write handoff/a2a_roundtrip.json with
     {verdict, samples, p95_ms, p50_ms, retry_observed,
      transient_failure_retried, sample_rows[:5]}.

3. Run verification; capture verbatim output.

4. EVALUATE: qa-evaluator + harness-verifier IN PARALLEL.

5. Write experiment_results.md + evaluator_critique.md; append
   harness_log.md; flip status.

## Anti-patterns guarded

- Self-approval (forbidden -- evaluators decide).
- Single-researcher or single-Explore (both ran).
- Using @tenacity or similar decorator that swallows CancelledError.
- Hard-coding retry_observed=True (must actually induce + observe
  a retry by e.g. one deliberate raise in Strategy on a flagged
  sample).
- Measuring only happy-path latency (real p95 must include the
  retry sample to be honest).

## Out of scope

- A2A SDK wire-protocol binding (researcher rec: keep for external
  boundary only; future sub-step under phase-3.7 if we ever expose
  agents externally).
- LLM calls in the delegation hop (kept out of critical path per
  researcher's latency analysis).

## References

- https://www.anthropic.com/engineering/harness-design-long-running-apps
- https://pypi.org/project/a2a-sdk/
- https://a2a-protocol.org/latest/specification/
- https://docs.python.org/3/library/asyncio-task.html
- https://threeofwands.com/careful-with-tasks/
- backend/agents/mas_events.py (existing Queue pattern)
