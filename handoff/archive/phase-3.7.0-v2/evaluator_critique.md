# Evaluator Critique -- Cycle 60 / phase-3.7 step 3.7.1

Step: 3.7.1 "Promote data_server.py to first-class Data MCP"

## qa-evaluator verdict: CONDITIONAL (lean PASS)

Immutable criteria satisfied:
- data_mcp_handshake_ok (transitively via phase-4.6.2 smoketest)
- parity_ge_95_vs_python_client (1.0 >= 0.95)

Flagged follow-up (non-blocking):
- Real FastMCP Client vs DataServer class-instance A/B would be
  stronger proof than divergent-mock canonicalization. Phase-4.6.2
  already proved the real in-memory client path works. Wire that
  into mcp_ab_test.py `data` branch before 3.7.4/3.7.6 so the
  phase-3.7 layer lands with honest handshake coverage.

## harness-verifier verdict: PASS

All 5 mechanical checks green:
- exit 0
- handoff/mcp_ab_test_data.json parses
- parity_rate 1.0 >= 0.95
- server == "data"
- latency_within_1_5x true (noise_dominated regime)

## Decision: PASS

Immutable criteria satisfied. CONDITIONAL concerns tracked as
follow-ups for 3.7.6 (tool-call-storm + size-cap + real-client A/B
wiring). Not blocking 3.7.1 acceptance.
