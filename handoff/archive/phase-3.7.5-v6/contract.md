# Contract -- Cycle 66 / phase-3.7 step 3.7.7

Step: 3.7.7 Capability tokens per session + PII filter on MCP input

## Hypothesis

Two stdlib-only defensive layers (capability tokens + regex PII filter)
wrapped around every `@mcp.tool` close the two largest 2026 MCP attack
classes observed in the wild:
- Cursor/Supabase class (over-permissioned agent exfiltrates data via
  prompt-injected tool call) -> blocked by capability scope mismatch
- CVE-2026-5023 class (unsanitized input flows to downstream systems)
  -> blocked by regex PII scrub + audit log

Zero new PyPI deps: `hmac`, `hashlib`, `secrets`, `re`, `time` only.

## Scope

Files this step creates/modifies:

1. **NEW** `backend/agents/mcp_capabilities.py` -- capability token
   issuance/verification + PII filter + composite decorator
2. **NEW** `scripts/harness/secret_leak_regression.py` -- 10 immutable
   assertions covering token + scope + PII paths
3. **NO** changes to existing MCP server files in this step. The
   decorator is shipped as a primitive; wiring into server factories
   is a follow-up (same rationale as 3.7.6 cap_output_size).

## Immutable success criteria (verification owned by evaluator)

1. `capability_tokens_scoped_per_session`: every issued token carries
   `(session_id, role, scopes, expires_at)` and is verifiable via HMAC;
   token TTL <= 30 minutes; role->scope map enforced.
2. `pii_filter_active`: emails, phone numbers, JWT-ish tokens,
   Anthropic `sk-ant-*` keys, and OpenAI `sk-*` keys are redacted from
   inbound arg dicts before tool body runs; clean args pass through
   byte-identical.
3. `secret_leak_regression_passes`: `python scripts/harness/secret_leak_regression.py`
   exits 0 with `"verdict": "PASS"` and writes
   `handoff/secret_leak_regression.json`.

## 10 required regression assertions

1. Unsigned request raises CapabilityError
2. Expired token (issued in past with TTL=1s) raises TokenExpiredError
3. Wrong-scope token (researcher calling trading.write) raises
   ScopeViolationError
4. Forged token (one char flipped in signature) raises TokenInvalidError
5. Email address in args is redacted + WARNING log emitted
6. Phone number in args is redacted + WARNING log emitted
7. Anthropic API key (`sk-ant-*`) in args is redacted + WARNING emitted
8. Clean args (`{"ticker":"AAPL","date":"2026-01-15"}`) unchanged
9. `paper_trader` role has `trading.write`; `researcher` does NOT
10. Token TTL honored: `expires_at - issued_at == 1800` (+-1s)

## Role -> scope map (authoritative)

- researcher   -> {data.read, signals.read, backtest.read}
- strategy     -> {data.read, signals.read, signals.write, backtest.read}
- risk         -> {data.read, signals.read, risk.read, risk.write}
- evaluator    -> {data.read, signals.read, backtest.read, risk.read}
- orchestrator -> all EXCEPT trading.write
- paper_trader -> {data.read, signals.read, trading.write, risk.read}

paper_trader is the only role with trading.write -- structural block
on "researcher accidentally placing orders" via prompt-injection.

## References

- modelcontextprotocol.io/specification/draft/basic/authorization
- NIST SP 800-63B-4 (short-lived tokens for automated agents)
- CoSAI/OASIS WS4 MCP Security report (2026)
- OWASP LLM01:2025 Prompt Injection
- CVE-2026-5023 codebase-mcp OS command injection
- Cursor/Supabase agent exfiltration (mid-2025 disclosure)
