# Sprint Contract — phase-25.A9 — Fix cache-write premium

**Cycle:** phase-25 cycle 2
**Date:** 2026-05-12
**Step ID:** 25.A9
**Priority:** P1 (prerequisite for 25.A8 hard-block)

## Research-gate
Audit basis already provides full research: phase-24.9 finding F-1. Researcher subagent was run in phase-24.9 (cycle 13) with `gate_passed: true`, 7 sources read in full. The fix is mandated verbatim by that audit. No new research needed; phase-25.A9 is a one-line implementation of a finding already documented at:
- `docs/audits/phase-24-2026-05-12/24.9-llm-conformance-findings.md` F-1
- Anthropic prompt-caching docs (read in full in phase-24.9 cycle): 1h TTL bills at 2.0x base input

Reusing the phase-24.9 research-gate envelope:
```json
{"tier":"complex","external_sources_read_in_full":7,"snippet_only_sources":10,"urls_collected":17,"recency_scan_performed":true,"internal_files_inspected":5,"gate_passed":true}
```

## Hypothesis
Changing `cache_write_cost = cache_creation * pricing[0] * 1.25 / 1_000_000` to `* 2.0` will correctly bill Anthropic 1h-TTL cache writes (currently under-reported by ~60%).

## Success criteria (verbatim from masterplan step 25.A9)
1. `cache_write_premium_constant_equals_2_0`
2. `cost_tracker_test_for_4096_token_write_charges_2x_base`

## Plan
1. Edit `backend/agents/cost_tracker.py:147` — `1.25` → `2.0` + update comment block to cite phase-25.A9 attribution
2. Create `tests/verify_phase_25_A9.py` — 5 claims (multiplier, old removed, attribution, AST, math round-trip)
3. Run verifier (target 5/5 PASS)
4. Write experiment_results.md
5. Spawn Q/A
6. Append harness_log Cycle 58
7. Flip masterplan 25.A9 to done

## References
External (re-used from phase-24.9):
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching (1h TTL = 2.0x)

Internal:
- `backend/agents/cost_tracker.py:141-148` (the line being changed)
- `backend/services/llm_client.py:773-779` (passes `"ttl": "1h"` — opted in)
- `docs/audits/phase-24-2026-05-12/24.9-llm-conformance-findings.md` (audit basis)
