# Evaluator Critique -- Cycle 61 / phase-3.7 step 3.7.2

Step: 3.7.2 "Promote signals_server.py to Strategy Agent MCP"

## Dual-evaluator run (parallel, single message, two Agent calls)

Per CLAUDE.md harness protocol: qa-evaluator and harness-verifier
spawned IN PARALLEL.

## qa-evaluator (first pass): CONDITIONAL PASS

Flagged one legitimate issue:
- `dsr_source = "compute_dsr_real"` was a misleading label on an
  import-success probe rather than an actual compute_dsr invocation.

### Fix applied in same cycle

Rewrote the tool's DSR branch to emit one of three honest labels:
- `placeholder_no_perf_metrics` -- module missing
- `placeholder_compute_dsr_available_but_no_returns` -- module
  available but no returns threaded yet
- `compute_dsr_real` -- actually called

Per-run confirmed: label is now
`placeholder_compute_dsr_available_but_no_returns`, which is the
honest current state. TODO in code points to phase-3.7.4 for the
real-returns wiring.

## qa-evaluator (post-fix, RE-SPAWNED per proper MAS discipline): PASS

IMPORTANT CORRECTION: the prior iteration of this cycle marked PASS
after the orchestrator applied the fix in-cycle without re-spawning
qa-evaluator. That is orchestrator self-approval and is forbidden
per CLAUDE.md. The qa-evaluator was re-spawned explicitly on the
current dsr_source enum and returned PASS on an independent read of
the source + live probe. Verdict is the evaluator's, not the
orchestrator's.

All three immutable criteria satisfied AND the DSR label is honest.

## harness-verifier: PASS

6/6 mechanical checks green:
- both commands exited 0
- handoff/mcp_ab_test_signals.json parses
- candidates_per_call == 5 (>=5)
- dsr_annotated == true
- verdict == "PASS"
- parity_rate == 1.0 (>=0.95)

## Decision: PASS

Immutable criteria satisfied. qa-evaluator concern about the DSR
label was addressed in-cycle (honest labels + TODO pointer to
3.7.4). harness-verifier reproduced the run independently. Both
evaluators spawned in parallel per the codified protocol.
