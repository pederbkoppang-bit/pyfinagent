# Experiment Results -- Cycle 66 / phase-3.7 step 3.7.7

Step: 3.7.7 Capability tokens per session + PII filter on MCP input

## What was generated

1. **backend/agents/mcp_capabilities.py** (NEW, stdlib-only):
   - `issue_token(session_id, role, ttl_seconds=1800)`: HMAC-SHA256
     signed token carrying (sid, role, scopes, iat, exp, nonce),
     payload + sig both base64url-encoded.
   - `verify_token(token, required_scope)`: uses `hmac.compare_digest`
     (constant time); raises `TokenInvalidError` / `TokenExpiredError`
     / `ScopeViolationError` appropriately.
   - `scrub_args(args)`: deep-walks dict/list/tuple; regex-substitutes
     email, phone, anthropic `sk-ant-*`, openai `sk-*`, JWT, SSN
     patterns with `[REDACTED]`; logs WARNING on any redaction.
   - `ROLE_SCOPES` authoritative map: paper_trader is the ONLY role
     with `trading.write`.
   - `enforce(required_scope)` composable decorator for future wiring
     into MCP server factories.

2. **scripts/harness/secret_leak_regression.py** (NEW, 10 tests):
   unsigned / expired / wrong-scope / forged / email / phone /
   anthropic-key / clean-passthrough / role-map-structural / ttl-honored.

## Verification run (verbatim)

    $ python scripts/harness/secret_leak_regression.py
    WARNING:backend.agents.mcp_capabilities:...redacted PII kinds=['email']
    WARNING:backend.agents.mcp_capabilities:...redacted PII kinds=['phone']
    WARNING:backend.agents.mcp_capabilities:...redacted PII kinds=['anthropic_key']
    {"wrote": "handoff/secret_leak_regression.json",
     "verdict": "PASS", "tests_passed": 10, "tests_total": 10}
    exit=0

## Success criteria alignment

| Criterion | Result |
|-----------|--------|
| capability_tokens_scoped_per_session | PASS (HMAC+TTL+scope all strict) |
| pii_filter_active | PASS (email/phone/anthropic redacted; clean passthrough byte-equal) |
| secret_leak_regression_passes | PASS (10/10) |

## Iterations during GENERATE

- Initial phone regex `\d[\d\-\s().]{8,}\d` over-matched ISO dates
  (`2026-01-15`). Tightened to three explicit formats: `+` country
  code, `(NNN)` parens, or NANP `NNN-NNN-NNNN`. Re-ran -> 10/10.

## Known limitations / follow-ups (non-blocking)

- `enforce()` decorator is primitive-only this step. Wiring it around
  every `@mcp.tool` in data/signals/risk/backtest servers is a same-
  step follow-up (pattern identical to 3.7.6 cap_output_size which is
  also shipped as primitive awaiting dispatch wiring in 3.7.8).
- `MCP_CAPABILITY_SECRET` defaults to a visibly-insecure dev string.
  Production deploy must inject the real value via Secret Manager.
  Phase-4.8.x will add the deploy-time check.
