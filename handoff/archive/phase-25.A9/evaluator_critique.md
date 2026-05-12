---
step: phase-25.A9
cycle: 58
evaluator: qa (merged qa-evaluator + harness-verifier)
verdict: PASS
date: 2026-05-12
---

# Q/A Critique -- phase-25.A9 (cache-write premium 1.25x to 2.0x)

## 5-item harness-compliance audit

1. **Researcher gate**: CONFIRM. The contract explicitly reuses the
   phase-24.9 cycle-13 research gate envelope
   (`gate_passed:true, sources_read_in_full=7, urls=17,
   recency_scan=true`). Phase-25.A9 is a verbatim 1-line
   implementation of phase-24.9 audit finding F-1; no new external
   knowledge is required. Reuse is justified and disclosed --
   passes the "research-gate compliance" leg.
2. **Contract pre-commit**: CONFIRM. `handoff/current/contract.md`
   contains both verbatim success_criteria
   (`cache_write_premium_constant_equals_2_0`,
   `cost_tracker_test_for_4096_token_write_charges_2x_base`).
3. **experiment_results.md complete**: CONFIRM. Front-matter has
   `step: phase-25.A9`, verifier command, and verbatim 5/5 PASS
   output block.
4. **Log-last**: CONFIRM. `handoff/harness_log.md` does NOT yet
   contain a `phase=25.A9` entry -- last entry is Cycle 57
   phase=25.1. Correct ordering: Q/A PASS first, then log, then
   status-flip.
5. **No verdict-shopping**: CONFIRM. First Q/A spawn for 25.A9 in
   cycle 58. No prior CONDITIONAL/FAIL on this step-id.

## Deterministic checks

- Verifier run: `source .venv/bin/activate && python3
  tests/verify_phase_25_A9.py` -- **5/5 PASS, EXIT=0**, verbatim:
  ```
  [PASS] cache_write_premium_constant_equals_2_0
  [PASS] old_1_25_multiplier_removed_from_calculation
  [PASS] phase_25_A9_attribution_comment_present
  [PASS] cost_tracker_py_syntax_clean
  [PASS] cost_tracker_math_for_4096_token_write_at_5_per_mtok_equals_0_04096_usd
  ```
- AST parse on `backend/agents/cost_tracker.py`: SYNTAX_OK.
- `backend/agents/cost_tracker.py:149` reads
  `cache_write_cost = cache_creation * pricing[0] * 2.0 / 1_000_000`
  -- correct.
- `grep -n "1\.25" backend/agents/cost_tracker.py`: only 3 hits,
  none in compute paths --
    - L24 Gemini pro pricing tuple (unrelated, $1.25/MTok input)
    - L139 comment documenting Anthropic's 5-min TTL rate
      (historical context)
    - L143 comment quantifying the under-report closure
  Zero residual `1.25` in the active math. PASS.
- Attribution comment block (L137-145) correctly cites phase-25.A9,
  closes phase-24.9 F-1, and quantifies the ~60% under-report.

## LLM-judgment legs

1. **Contract alignment**: PASS. Implementation matches phase-24.9
   F-1 exactly -- single multiplier swap with attribution comment.
2. **Mutation-resistance**: PASS. Verifier claims 1+2 (constant
   equals 2.0 AND old 1.25 removed from calculation) would both
   trip on a revert. The math claim (#5) provides a third
   independent guard via numeric round-trip
   ($0.04096 != $0.0256).
3. **Anti-rubber-stamp / claim verifiability**: PASS. Verified
   `backend/agents/llm_client.py:778` literally passes
   `"ttl": "1h"` inside the `cache_control` block. The 2.0x rate
   is therefore the correct Anthropic charge, not a hypothesis.
   Minor: contract references `backend/services/llm_client.py` but
   the file lives at `backend/agents/llm_client.py` -- cosmetic
   path-drift, does not affect the fix. Flag for cleanup but
   non-blocking.
4. **Scope honesty**: PASS. The live-check (re-aggregation script
   to confirm cumulative cost increase) is explicitly deferred to
   post-deployment in `experiment_results.md` -- no overclaim.
5. **Research-gate compliance**: PASS. Reuse of phase-24.9 gate
   is appropriate because (a) the audit finding is the
   authoritative source, (b) Anthropic's published pricing has
   not changed since cycle 13, and (c) the change is mechanical.
   A fresh full research gate for a 1-line constant swap would
   be wasteful.

## Violations
None.

## Verdict envelope

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "5/5 verifier PASS, 5/5 harness-compliance audit PASS, all LLM-judgment legs satisfied. 1.25x to 2.0x cache-write multiplier correctly implemented per phase-24.9 F-1; llm_client.py:778 confirmed to pass ttl=1h opting into Anthropic's 2.0x extended-cache-TTL rate; mutation-resistance covered by 3 independent verifier claims.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": ["syntax", "verification_command", "constant_grep", "ttl_opt_in_verification", "harness_log_state", "contract_alignment", "scope_honesty"]
}
```

## Non-blocking note for follow-up
Contract references `backend/services/llm_client.py:773-779`; actual path is `backend/agents/llm_client.py:773-779`. Worth correcting in the next handoff cycle but does not affect this PASS.
