# Evaluator Critique -- Cycle 66 / phase-3.7 step 3.7.7

Step: 3.7.7 Capability tokens per session + PII filter on MCP input

## Dual-evaluator run (parallel, evaluator-owned)

## qa-evaluator: PASS

All 3 immutable criteria satisfied. Line-by-line findings:

1. **Token honesty**: HMAC-SHA256 real (line 122); `hmac.compare_digest`
   constant-time (line 146); TTL strict (`t >= exp` raises, lines
   154-157); scope membership strict (line 160). Injected `now=`
   enables deterministic expiry test.

2. **Role -> scope map**: Exactly 6 roles. `paper_trader` is the ONLY
   role with `trading.write`. `orchestrator` scopes explicitly exclude
   `trading.write`. Test `role_scope_map_structural` asserts this
   across all roles, not just a spot-check.

3. **PII filter honesty**: Real substitution via `pat.sub(_REDACTED,
   out)` (line 192), deep-walks dict/list/tuple (lines 209-220).
   Assertions verify both `[REDACTED]` present AND original secret
   literal absent. WARNING logged per hit.

4. **Test discrimination**: `forged_token` flips sig byte -> catches
   missing HMAC. `wrong_scope` catches role-map regression. `clean_
   args_passthrough` with `"date": "2026-01-15"` catches over-greedy
   regex (this test already caught a bug in the first GENERATE pass
   -- the phone regex matching ISO dates -- and was the reason the
   second regex tightening was applied). `token_ttl_honored` decodes
   payload and asserts exp-iat ~= 1800.

5. **Regression artifact**: verdict=="PASS", 10/10.

## harness-verifier: PASS

All 5 mechanical checks green:
- syntax_mcp_capabilities: valid AST
- syntax_secret_leak_regression: valid AST
- regression_run: exit=0, "verdict": "PASS", "tests_passed": 10
- json_artifact_assertions: all hold
- role_map_invariant: paper_trader has trading.write; researcher,
  orchestrator, strategy do not.

## Decision: PASS (evaluator-owned)

All 3 immutable criteria met. Both evaluators ran independently and
both returned PASS. No CONDITIONAL from either -- no orchestrator
revision/re-submission cycle this step.
