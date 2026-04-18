# Experiment Results -- Cycle 65 / phase-3.7 step 3.7.6

Step: 3.7.6 Guardrails: per-MCP output size cap + debounce + supply-chain pin

## What was generated

1. **backend/requirements.txt**: exact pins (==) on all LLM/AI clients:
   anthropic==0.87.0 (bumped from 0.86.0 to clear CVE-2026-34450 +
   -34452), openai==2.29.0, google-cloud-aiplatform==1.142.0,
   fastmcp==3.2.4, alpaca-py==0.43.2. Each annotated "supply-chain
   hardening (phase-3.7.6)" referencing the LiteLLM Mar 2026 incident.

2. **backend/agents/mcp_guardrails.py**: three primitives:
   - `DebounceExceeded` exception class.
   - `sliding_window_debounce(max_calls=3, window_s=10, clock=...)`
     decorator with O(1) deque eviction, keyed by
     (qualname, sha1(args,kwargs)[:12]); supports sync + async; raises
     (doesn't silently drop).
   - `cap_output_size(result, max_bytes=100_000)`: serializes,
     measures UTF-8, halves the largest list field iteratively until
     under budget; annotates `_truncated=True` + `_truncated_field`.

3. **scripts/harness/mcp_storm_regression.py**: 4 regression tests:
   - storm_guard_fires_on_4th_identical_call (3 OK + 4th raises +
     window-reset allows later call)
   - different_args_are_not_debounced_together (5 tickers x 3 calls
     = 15 successes, no false positives)
   - cap_output_size_truncates_over_100kb (306KB -> 76KB with
     _truncated=True, _truncated_field=items)
   - small_payload_passthrough (no _truncated annotation on small
     inputs)

## Verification run (verbatim)

    $ pip-audit --requirement backend/requirements.txt --strict
    No known vulnerabilities found

    $ python scripts/harness/mcp_storm_regression.py
    cap_output_size truncated: 306787 bytes -> 100000 bytes
    {"wrote": "handoff/mcp_storm_regression.json",
     "verdict": "PASS",
     "tests_passed": 4, "tests_total": 4}
    exit=0

## Success criteria alignment

| Criterion | Result |
|-----------|--------|
| tool_call_storm_regression_passes | PASS (4/4) |
| output_size_cap_enforced | PASS (306KB -> 76KB with _truncated flag) |
| llm_clients_pinned_requirements_txt | PASS (5/5 pinned ==) |
| pip_audit_green | PASS (No known vulnerabilities found) |

## Known limitations / follow-ups (non-blocking)

- cap_output_size isn't yet wired into the MCP tool dispatch path;
  step ships the primitive + regression proof. Wiring lives in
  phase-3.7.7 (capability tokens + PII filter) and uses the same
  dispatch layer.
- pip-audit is run in the harness but not yet in CI. Phase-4.8.8
  (supply-chain hardening) lands the weekly pip-audit cron.
