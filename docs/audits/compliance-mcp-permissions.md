# Compliance Audit: MCP + Permission Policies + Capability Tokens
**Phase 4.15.4 — 2026-04-18**

---

## Scope

Full read of: `.mcp.json`, `.claude/settings.json`, `.claude/settings.local.json`,
`backend/agents/mcp_capabilities.py`, `backend/agents/mcp_guardrails.py`,
all four `backend/agents/mcp_servers/*.py`, `backend/slack_bot/mcp_tools.py`,
`scripts/audit/mcp_inventory.py`, `scripts/audit/mcp_risk_score.py`,
`backend/services/mcp_health_cron.py`, `handoff/mcp_watchlist.md`,
`handoff/mcp_inventory.json`.
External references: Anthropic permissions docs, managed-agents permission-policy docs (read 2026-04-18).

---

## Pattern Inventory (20 findings)

### P-01 — `enableAllProjectMcpServers: true` + `enabledMcpjsonServers` contradiction [MEDIUM]
**File:** `.claude/settings.local.json`

`enableAllProjectMcpServers: true` tells Claude Code to mount every server in `.mcp.json`
automatically. At the same time `enabledMcpjsonServers: ["slack"]` lists only one server.
These two directives conflict: the `enableAll` flag wins at runtime and both `slack` and
`alpaca` are mounted, making the allowlist a dead letter. This was flagged as a phase-4.10
finding and has not been resolved. The effective surface is wider than the explicit list
suggests; any new server added to `.mcp.json` is automatically live without a corresponding
`enabledMcpjsonServers` update.

**Maps to:** MF-2 (surface control).

---

### P-02 — Alpaca write-tool deny rule absent [HIGH]
**File:** `.mcp.json`, `.claude/settings.json`

`alpaca-mcp-server` exposes order-placement tools. Neither `.claude/settings.json` nor
`.claude/settings.local.json` contains a `deny: ["mcp__alpaca__create_order",
"mcp__alpaca__*_order"]` (or equivalent wildcard `mcp__alpaca__*`) rule. The environment
variable `ALPACA_PAPER_TRADE: "true"` is set, which limits financial exposure to paper
accounts, but it does not prevent the tool from being called. An adversarial prompt injection
could call order tools; the paper-mode flag is the only barrier. Anthropic permission docs
confirm that deny rules must be explicit: `mcp__<server>__<tool>` syntax matches individual
tools, and `mcp__alpaca` matches all tools from that server. Either form is absent.

**Maps to:** MF-18 (write-tool veto).

---

### P-03 — `bypassPermissions` mode with no container isolation [HIGH]
**File:** `.claude/settings.json`, line 77

`"defaultMode": "bypassPermissions"` skips all permission prompts except writes to `.git`,
`.claude`, `.vscode`, `.idea`, and `.husky`. Anthropic docs say explicitly: "Only use this
mode in isolated environments like containers or VMs where Claude Code cannot cause damage."
pyfinAgent runs directly on the developer's macOS machine, not in a container. The harness
doc (`per-step-protocol.md`) mandates this mode for autonomous cycling, but the isolation
requirement is unmet. No `disableBypassPermissionsMode` guard exists in any managed settings
layer to prevent escalation.

**Severity note:** in the current setup the developer (Peder/Ford) is the sole user, so
the practical blast radius is limited. The risk materialises if CI or a shared machine ever
runs the harness.

---

### P-04 — BigQuery MCP present in injected harness but not declared in `.mcp.json` [LOW]
**File:** `CLAUDE.md`, `.mcp.json`

`CLAUDE.md` documents a BigQuery MCP server injected by the harness environment with read
and write DML access to project `sunny-might-477607-p8`. This server has no entry in
`.mcp.json` and therefore no explicit permission rules in `settings.json`. The inventory
script (`mcp_inventory.py`) will not detect it. Harness-injected servers are outside the
file-based audit surface, creating a documentation gap: any MCP-aware audit that reads only
the JSON files will undercount active servers by one.

**Maps to:** MF-19 (inventory completeness).

---

### P-05 — Legacy `backend/mcp/*.py` stubs are dead code but not removed [LOW]
**File:** `backend/mcp/data_server.py`, `backend/mcp/signals_server.py`, `backend/mcp/backtest_server.py`

All three legacy stubs (`stub: true` in `mcp_inventory.json`) expose FastAPI HTTP apps on
hardcoded ports (8101, 8103, implied). They have zero tools, zero resources, and contain
`# TODO: Implement phase 3` comments. The authoritative FastMCP servers live in
`backend/agents/mcp_servers/`. The legacy stubs are import-dead (no callers) and create
confusion: any contributor reading `backend/mcp/` thinks these are live servers. The
inventory correctly marks them `stub: true` but the stubs have not been deleted.

**Maps to:** MF-19 (inventory hygiene).

---

### P-06 — HMAC capability tokens not wired to in-process MCP server tools [HIGH]
**File:** `backend/agents/mcp_capabilities.py`, `backend/agents/mcp_servers/`

`mcp_capabilities.py` provides `issue_token`, `verify_token`, and the `@enforce` decorator.
Inspection of all four authoritative servers (`data_server.py`, `backtest_server.py`,
`risk_server.py`, `signals_server.py`) shows that none of their `@mcp.tool`-decorated
functions use `@enforce` or call `verify_token`. The capability token system is fully
implemented and unit-testable but is not deployed at any tool call-site. Any agent that
obtains a reference to the in-process MCP transport can call any tool regardless of role or
scope. The `ROLE_SCOPES` mapping (`researcher`, `strategy`, `risk`, `evaluator`,
`orchestrator`, `paper_trader`) is therefore unenforced at runtime.

**Maps to:** MF-18 (scope enforcement).

---

### P-07 — `mcp_guardrails.py` debounce and output-cap not wired to tool callsites [MEDIUM]
**File:** `backend/agents/mcp_guardrails.py`, `backend/agents/mcp_servers/`

Same pattern as P-06. `sliding_window_debounce` and `cap_output_size` are implemented but
not applied to any `@mcp.tool` decorator in the four authoritative servers. The `run_backtest`
tool is explicitly noted as "slow (~30s)" and is precisely the kind of call that would benefit
from debounce protection. Without `@sliding_window_debounce`, a retry loop or prompt injection
can trigger concurrent backtest runs, each consuming significant compute.

---

### P-08 — `settings.local.json` allows destructive Bash commands without scope [MEDIUM]
**File:** `.claude/settings.local.json`, lines 8-12

Three `Bash(...)` allow rules permit:
- `mkdir -p docs/runbooks docs/audits`
- `mv .claude/agents/per-step-protocol.md docs/runbooks/per-step-protocol.md`
- `rm .claude/agents/qa-evaluator.md .claude/agents/harness-verifier.md`

These are exact-match historical rules from a past session. The `rm` rule allows deletion of
agent config files without re-prompting. Per the permissions doc, allow rules in
`settings.local.json` have lower precedence than `settings.json` but higher than user-level.
Because `defaultMode: bypassPermissions` is set in `settings.json`, these rules are actually
redundant (bypass already skips prompts), but they would persist and become effective if
`defaultMode` were later changed to `default` or `acceptEdits`. Stale allow rules for `rm`
and `mv` on config files should be cleaned up.

---

### P-09 — `Managed Agents permission_policy` not used; custom capability tokens fill the gap [INFORMATIONAL]
**File:** `backend/agents/mcp_capabilities.py`

The Anthropic Managed Agents API (beta header `managed-agents-2026-04-01`) supports
`permission_policy: {type: "always_allow" | "always_ask"}` on `mcp_toolset` entries,
offering binary allow/deny per server or per tool name. pyfinAgent does not use the Managed
Agents API (agents run via Claude Code harness, not API sessions). The HMAC token system in
`mcp_capabilities.py` is strictly more expressive: it carries session ID, role, scopes, TTL,
and a nonce, and supports `ScopeViolationError` granularity that binary `always_ask` cannot
provide. The custom tokens are compliant with the spirit of the Managed Agents policy system
and exceed it in capability, but only if they are actually enforced at call-sites (see P-06).

**Compliance status:** Implementation exceeds the platform equivalent in design; falls short
in deployment.

---

### P-10 — `slack-mcp` write tools (`post_message`, `create_canvas`) have no deny rule [MEDIUM]
**File:** `.mcp.json`, `backend/slack_bot/mcp_tools.py`

The `@anthropic-ai/mcp-server-slack` package exposes write-capable tools including
`post_message` and `create_canvas`. No `deny: ["mcp__slack__post_message"]` or
`deny: ["mcp__slack__create_canvas"]` rule exists in either settings file. During autonomous
harness cycles an agent could post to Slack channels without operator confirmation. The Slack
bot (`backend/slack_bot/`) already handles deliberate Slack posting via the Socket Mode app;
double-posting via the MCP surface is an unintended side-channel.

---

### P-11 — `ALPACA_PAPER_TRADE: "true"` is the sole guard on trading tools [HIGH]
**File:** `.mcp.json`

Related to P-02 but distinct: the Alpaca paper-trade flag is an environment variable passed
to the `alpaca-mcp-server` process. It is not a Claude Code permission rule. The flag
controls which Alpaca API endpoint the server targets but does not prevent tool calls from
being made. If the env var is accidentally unset or the server ignores it for certain tool
variants, live-capital orders could be placed. A `deny: ["mcp__alpaca__*"]` with targeted
`allow: ["mcp__alpaca__get_*", "mcp__alpaca__list_*"]` (read-only tools only) would be a
defense-in-depth layer that cannot be misconfigured away.

**Maps to:** MF-18.

---

### P-12 — MCP health cron samples only 3 GitHub repos per run [LOW]
**File:** `backend/services/mcp_health_cron.py`, line 102

`check_once(gh_sample_limit=3)` is hardcoded. With two adopted servers (`slack`, `alpaca`)
and a watchlist of 12 entries, any run that iterates beyond 3 items silently skips GitHub
freshness checks for the remainder. The cron only covers 3 servers per Sunday sweep, not the
full active inventory. This is acceptable for watchlist entries (they are not live) but means
the two adopted servers may not both be checked in the same run if iteration order shifts.

---

### P-13 — `mcp_risk_pull.py` referenced in audit instructions but does not exist [LOW]
**File:** `scripts/audit/`

The audit brief listed `scripts/audit/mcp_risk_pull.py` as a file to read. No such file
exists in the repository. `mcp_risk_score.py` exists (phase-3.5 step 3.5.2 scoring). The
missing file may have been planned for phase-3.5 step 3.5.1 (candidate extraction) and never
created, or was renamed. The inventory pipeline has a gap: `mcp_risk_score.py` requires
`handoff/mcp_candidates.csv` as input, but there is no script to produce that CSV
automatically. Manual production of the CSV is an audit continuity risk.

---

### P-14 — `backend/slack_bot/mcp_tools.py` uses a placeholder `MCPToolExecutor` [LOW]
**File:** `backend/slack_bot/mcp_tools.py`, lines 126-185

Every `_search_*`, `_post_message`, and `_read_thread` handler returns a hardcoded stub
string. The `# TODO: Call Slack MCP server` comments confirm these are unimplemented. The
file also contains an emoji in `logger.info` on line 143, violating the project no-emoji
rule. Additionally, the `build_claude_mcp_config` function targets
`claude-sonnet-4-20250514` which may be a stale model ID.

---

### P-15 — `enforce` decorator pops `_cap_token` kwarg but FastMCP tools don't receive it [MEDIUM]
**File:** `backend/agents/mcp_capabilities.py`, lines 235-252

The `@enforce(scope)` decorator expects the wrapped function to accept `_cap_token: str |
None = None` as a keyword argument. FastMCP's `@mcp.tool` decorator passes tool inputs from
the MCP wire protocol as positional/keyword arguments matching the tool's JSON schema. A
client calling the tool via MCP would have no way to include `_cap_token` in the tool input
unless the schema explicitly declares it. This means `@enforce` cannot be applied
transparently to FastMCP tools without schema changes that expose `_cap_token` as an
explicit tool parameter, which would be visible to all callers. The design needs revision
before P-06 can be closed: either via out-of-band session context or a middleware intercept
at the FastMCP transport layer.

**Maps to:** MF-18.

---

### P-16 — `risk_server.py` `portfolio_cvar` and `factor_exposure` are stubs with no timeline enforcement [LOW]
**File:** `backend/agents/mcp_servers/risk_server.py`, lines 94-129

Both tools return `"status": "stub_placeholder"` and note `"todo": "phase-4.8.2"`. There is
no runtime assertion or test that would fail if phase-4.8.2 is skipped. An evaluator agent
consuming these tools would receive apparent success responses with `null` data, potentially
silently skipping the CVaR and factor-exposure gates in the composite `evaluate_candidate`
function.

---

### P-17 — `signals_server.py` idempotency state is in-memory only [LOW]
**File:** `backend/agents/mcp_servers/signals_server.py`, lines 76-93

The `_seen_signal_ids` set and `_recent_responses` dict are cleared on process restart. On
a crash-restart cycle, `publish_signal` can re-publish a signal it already sent. The code
acknowledges this ("durable BQ signal_history table is Phase 4.2"), but phase-4.2 is not
marked done in `masterplan.json`. Until the BQ write lands, duplicate Slack posts and
duplicate paper-trade orders are possible on restart.

---

### P-18 — Permission rule evaluation order not reflected in settings layout [INFORMATIONAL]
**File:** `.claude/settings.json`, `.claude/settings.local.json`

Anthropic docs specify: deny > ask > allow, first match wins. The `settings.json`
`permissions` block has only `allow` entries; there is no `deny` array. The `settings.local.json`
`permissions` block has only `allow` entries for three Bash commands. With `bypassPermissions`
as `defaultMode`, the evaluation order is moot today, but the configuration gives no
indication to a reader that writes to production infrastructure (BigQuery DML, Alpaca orders,
Slack posts) are intended to be prompt-free. Adding a `deny` section with explicit comments
would improve auditability.

---

### P-19 — `_DEFAULT_DEV_SECRET` in `mcp_capabilities.py` is a hard-coded fallback [MEDIUM]
**File:** `backend/agents/mcp_capabilities.py`, line 47

```python
_DEFAULT_DEV_SECRET = "dev-only-mcp-cap-secret-CHANGE-IN-PROD"
```

If `MCP_CAPABILITY_SECRET` is absent from the environment (e.g., a developer machine with
no `.env`), all HMAC tokens are signed with this known string. Any party who reads this file
can forge valid tokens. The code comment says "CHANGE-IN-PROD" but there is no startup
assertion (`assert os.getenv("MCP_CAPABILITY_SECRET")`) that would fail-fast when the env
var is missing in a non-dev context. The `backend/.env` file is gitignored, so there is no
guarantee the variable is set in every harness environment.

---

### P-20 — `mcp_inventory.py` secret-pattern scan misses `ALPACA_PAPER_TRADE` env key type [INFORMATIONAL]
**File:** `scripts/audit/mcp_inventory.py`, lines 27-33

The `SECRET_PATTERNS` list catches `sk-*` (OpenAI), `xox*` (Slack tokens), `AIza*` (Google),
`ghp_*` (GitHub PATs), and PEM headers. Alpaca API keys follow the format
`APCA-API-KEY-ID` / `APCA-API-SECRET-KEY` (short alphanumeric strings). These would not be
caught by any existing pattern. If an Alpaca key were accidentally inlined into a script
rather than referenced via env-var template, the inventory scanner would not flag it.

---

## Summary Table

| ID | Area | Severity | Maps to |
|----|------|----------|---------|
| P-01 | `enableAllProjectMcpServers` vs `enabledMcpjsonServers` | MEDIUM | MF-2 |
| P-02 | Alpaca write-tool deny absent | HIGH | MF-18 |
| P-03 | `bypassPermissions` without container isolation | HIGH | — |
| P-04 | BigQuery MCP invisible to file audit | LOW | MF-19 |
| P-05 | Legacy `backend/mcp/` stubs not deleted | LOW | MF-19 |
| P-06 | HMAC tokens not wired to tool callsites | HIGH | MF-18 |
| P-07 | Guardrails not wired to tool callsites | MEDIUM | — |
| P-08 | Stale `rm`/`mv` allow rules in settings.local | MEDIUM | — |
| P-09 | Managed Agents policy vs custom tokens (design OK, deployment not) | INFO | — |
| P-10 | Slack write-tool deny absent | MEDIUM | MF-18 |
| P-11 | `ALPACA_PAPER_TRADE` env var as sole order guard | HIGH | MF-18 |
| P-12 | Health cron GitHub sample limit of 3 | LOW | — |
| P-13 | `mcp_risk_pull.py` missing; CSV pipeline manual | LOW | MF-19 |
| P-14 | `mcp_tools.py` stubs + emoji violation | LOW | — |
| P-15 | `@enforce` decorator incompatible with FastMCP schema | MEDIUM | MF-18 |
| P-16 | CVaR/factor-exposure stub gates silently pass | LOW | — |
| P-17 | Signal idempotency state lost on restart | LOW | — |
| P-18 | No `deny` section despite wide allow mode | INFO | — |
| P-19 | Hard-coded dev HMAC secret with no startup assertion | MEDIUM | — |
| P-20 | Alpaca key pattern absent from secret scanner | INFO | MF-19 |

**HIGH:** 4 (P-02, P-03, P-06, P-11)
**MEDIUM:** 6 (P-01, P-07, P-08, P-10, P-15, P-19)
**LOW:** 7 (P-04, P-05, P-12, P-13, P-14, P-16, P-17)
**INFO:** 3 (P-09, P-18, P-20)

---

## Recommended Immediate Actions (HIGH severity)

1. **P-02 / P-11:** Add `deny: ["mcp__alpaca__*"]` to `settings.json` and add back only the
   specific read-only tool names that the harness legitimately needs. This is one line in the
   JSON and closes the most material trading-safety gap.

2. **P-06 / P-15:** Before wiring `@enforce` to FastMCP tools, decide the transport mechanism
   for capability tokens. Options: (a) pass token via a dedicated `X-Cap-Token` header in
   the in-process call if FastMCP supports custom headers, or (b) use a request-scoped context
   variable set by a FastMCP middleware, or (c) accept the token as a first-class tool
   parameter in the JSON schema (transparent to MCP callers, documented as required). Until
   one of these is resolved, the capability token system is documentation-only.

3. **P-03:** Document the container/VM isolation gap in `CLAUDE.md` as a known deviation and
   add `"disableBypassPermissionsMode": "disable"` to any CI-facing managed settings if/when
   CI adoption happens.

---

*Audit generated by Researcher Agent — pyfinAgent phase-4.15.4 — 2026-04-18*
