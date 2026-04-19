# MCP Security

**Phase lineage:** 3.7.6 (output caps + debounce + supply-chain pin) + 3.7.7 (capability tokens + PII scrub) consolidated here by 3.0.

Companion doc: `docs/MCP_ARCHITECTURE.md`.

## Threat model

| Attacker | Surface | Mitigation |
|----------|---------|-----------|
| Compromised internal agent (bug, hallucination, prompt-injection) | Can call any MCP tool unless scoped | Capability tokens with role-scope mapping (below) |
| Compromised external MCP server code (supply chain) | Imported into process via `uvx` / `npx` | Version pin in `.mcp.json` + weekly `mcp_health_cron.py` audit + OSV scan |
| Leaked PII into MCP tool args (user query carrying name / email / phone) | Server-side logs, BQ audit tables | PII scrub at `mcp_capabilities.py` inbound boundary |
| Over-capability drift | Tool surface grows quietly, scope gets broad | Scope set is FIXED per role (phase-3.7.7); new scopes require ADR |
| Remote-exposure mis-config | Someone binds stdio server to TCP | No: stdio-only in `.mcp.json`, no remote binding code in the tree |

## Capability tokens

Implementation: `backend/agents/mcp_capabilities.py` (phase-3.7.7). Verified against source on 2026-04-19.

- **Algorithm**: HMAC-SHA256 over the JSON payload `{sid, role, scopes, iat, exp, nonce}`. Token wire format: `<b64u(payload_json)>.<b64u(sig)>`. Secret from `MCP_CAPABILITY_SECRET` env var (dev fallback baked in; MUST be overridden in prod).
- **TTL**: `TOKEN_TTL_SECONDS = 1800` (30 minutes, NIST SP 800-63B-4 short-lived-credential guidance). Constant exported from the module.
- **Roles** (6, defined in `ROLE_SCOPES` at `mcp_capabilities.py:57-70`):
  - `researcher` -> `{data.read, signals.read, backtest.read}`
  - `strategy` -> `{data.read, signals.read, signals.write, backtest.read}`
  - `risk` -> `{data.read, signals.read, risk.read, risk.write}`
  - `evaluator` -> `{data.read, signals.read, backtest.read, risk.read}`
  - `orchestrator` -> `{data.read, signals.read, signals.write, backtest.read, risk.read, risk.write}`
  - `paper_trader` -> `{data.read, signals.read, trading.write, risk.read}`
- **Scopes**: domain-coarse rather than per-tool. 6 distinct tokens: `data.read`, `signals.read`, `signals.write`, `backtest.read`, `risk.read`, `risk.write`, plus `trading.write` (paper-trader only). No compound scopes.
- **Exceptions** (`mcp_capabilities.py:73-86`): `CapabilityError` (base, PermissionError subclass), `TokenExpiredError`, `TokenInvalidError`, `ScopeViolationError`. Callers catch the base class to fail uniformly; individual subclasses distinguish 401 (invalid/expired) from 403 (scope mismatch).
- **Verification API**: `verify_token(token, required_scope) -> payload_dict`. `enforce(required_scope)` decorator wraps `@mcp.tool` handlers (see `mcp_capabilities.py:237-254`) and also invokes `scrub_args` on positional + keyword args before the tool body runs.

Changing a role's scope set requires an ADR entry in `handoff/archive/`. Do NOT edit `ROLE_SCOPES` (module-public) in `mcp_capabilities.py` without writing the ADR first.

## PII scrub

Every inbound MCP call's args are passed through a regex-based PII filter at `mcp_capabilities.py:170-230` (verified against source 2026-04-19) before the tool impl runs. Detected classes (`_PII_PATTERNS`, lines 172-184):

- `email`: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b`
- `phone`: mixed NANP `(XXX) XXX-XXXX` / `XXX-XXX-XXXX` + `+digits` international prefixes
- `anthropic_key`: `sk-ant-[A-Za-z0-9\-_]{20,}` (Anthropic API key format)
- `openai_key`: `\bsk-[A-Za-z0-9]{20,}\b` (OpenAI API key format)
- `jwt`: `eyJ...\.eyJ...\....` (JWT 3-segment base64url)
- `ssn`: `\b\d{3}-\d{2}-\d{4}\b` (US SSN)

Each match is replaced with the literal string `[REDACTED]` (no kind suffix; see `_REDACTED` constant at `mcp_capabilities.py:186`). The list of detected kinds is returned to the caller alongside the scrubbed value. When any redaction fires, the module logs a WARNING (not DEBUG) with the sorted set of kinds -- audit signal is retained while the redacted payload is not.

`scrub_args(args)` walks dicts, lists, and tuples recursively (`mcp_capabilities.py:199-230`); clean args pass through byte-identical. The scrubbed value is what the tool impl sees; the original is discarded.

Gaps vs threat model:
- Credit-card numbers: NOT currently scrubbed. Low priority for pyfinagent (financial data system, no PCI scope).
- Free-form names: NOT scrubbed. By design: names in analyst reports / news articles are domain-legitimate.

This is a defense-in-depth measure; the primary expectation is that callers do not put PII into agent queries in the first place. See `.claude/rules/security.md` for input-validation rules at API boundaries.

## Supply-chain pinning

Policy (phase-3.7.6):

1. Every external MCP server in `.mcp.json` MUST have an explicit version pin.
2. The Python dep pin for FastMCP MUST be an exact-equals in `backend/requirements.txt` (`fastmcp==3.2.4`).
3. `scripts/housekeeping/verify_handoff_layout.py` (and similar) run at cron; drift fails CI.

Current pins (2026-04-19):
- `@anthropic-ai/mcp-server-slack` @ latest (Anthropic-published; acceptable per policy exception for first-party)
- `alpaca-mcp-server==2.0.1` (pinned by phase-3.0 -- was unpinned prior; closes the phase-3.7.6 action item)
- `fastmcp==3.2.4` (pinned in `backend/requirements.txt` line ~39)

Unpinned = CVE exposure + silent breaking-change ingest. `backend/services/mcp_health_cron.py` reports any detected unpin.

## Output caps + debounce

Per phase-3.7.6:
- Per-tool maximum response size: 32 KiB before truncation. Larger payloads return the first 32 KiB + a `truncated=true` flag + a `continuation_token` that a follow-up call can paginate on.
- Debounce: identical (tool, args, caller) within 5 seconds returns the cached prior response (no re-invoke). Prevents thundering-herd on retry loops.
- Both are implemented at the FastMCP middleware layer -- see `mcp_servers/__init__.py` (the import-time middleware chain).

## Rate limiting (gap, documented)

The four internal MCP servers do NOT rate-limit their tool invocations. Rationale:

- Local stdio transport -> the caller (agent in the same process) is the rate-limit point, not the server.
- Phase-6.7 already provides `get_rate_limiter("<source>")` (aiolimiter leaky-bucket) at the external-API call sites. MCP tools that ultimately hit external APIs (Finnhub, FRED, etc.) inherit that rate limit.
- Adding another layer inside the MCP server would double-count against the same external-server quota.

**Trigger to revisit**: any move toward remote (Streamable HTTP) exposure. At that point the MCP server becomes its own rate-limit point and aiolimiter should be added as a FastMCP middleware. Flagged here so the decision is auditable.

## Audit log (documented gap)

**Current state (2026-04-19):** There is NO BQ-backed `mcp_audit` table. Verified: `ls scripts/migrations/` does not include any `add_mcp_audit*` migration; `grep -r "mcp_audit" backend/ scripts/` returns zero matches.

What does exist today:
- **`scrub_args` WARNING logs** -- every redaction fires a WARNING with the kinds redacted. These go to stderr / uvicorn log handlers, not BQ.
- **`CapabilityError` exceptions** -- raised from `verify_token` on signature/expiry/scope failure; callers are expected to log them. No uniform destination.

What is NOT there yet:
- No append-only BQ table for capability-token pass/fail events.
- No sampling policy for success events.
- No dashboard / query path for "which roles hit which scope violations in the last 24h".

**Follow-up needed** (scoped as a phase-3.7.7 or phase-6.9 follow-up, not blocking phase-3.0):
1. Create `scripts/migrations/add_mcp_audit_log.py` with columns `{ts, session_id, role, required_scope, outcome (pass|invalid|expired|scope_violation), tool_name}`.
2. Add `log_mcp_verify` to `backend/services/observability/api_call_log.py` alongside the existing `log_api_call` / `log_llm_call` buffered writers.
3. Wire the call inside `verify_token` (success path sampled 1/100; failure path 100%).

Filing this gap explicitly rather than documenting intent as reality keeps the security doc trustworthy.

## Incident response

If a capability-token compromise is suspected:

1. Rotate the process-local HMAC secret (restart backend with a fresh env var).
2. Invalidate all in-flight sessions (TTL will expire within 30 min regardless).
3. Review `mcp_audit` for unauthorized scope usage in the suspected window.
4. If supply-chain compromise is suspected on an external MCP server: pin to a prior known-good version in `.mcp.json`, restart, file issue with the upstream.

See `docs/INCIDENT_RUNBOOK.md` for the broader cross-system incident playbook; this section is the MCP-specific addendum.
