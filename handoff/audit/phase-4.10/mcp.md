# MCP Audit — Claude Doc Alignment (phase-4.10.4)

Audit only (no source modified). Docs fetched from `https://code.claude.com/docs/en/mcp`.

## Documentation summary

**Transports.** Three options, declared via `type`:
- `http` (recommended for remote) — `claude mcp add --transport http <name> <url>`. Config key `url`, optional `headers`, `oauth`, `headersHelper`.
- `sse` (deprecated; SSE remote).
- `stdio` (local process) — keys `command`, `args`, `env`. On Windows-native, `npx` must be wrapped with `cmd /c`.

**Scopes & storage.** Three levels with explicit precedence (Local > Project > User > Plugin > claude.ai connectors):
- `local` — per-user, per-project; written to `~/.claude.json` projects map.
- `project` — shared via `.mcp.json` at repo root; requires user approval prompt on first use; reset with `claude mcp reset-project-choices`.
- `user` — cross-project, per-user; also in `~/.claude.json`.
- Plugin-provided: `.mcp.json` at plugin root or inline in `plugin.json`; `${CLAUDE_PLUGIN_ROOT}` / `${CLAUDE_PLUGIN_DATA}` expansion.

**`.mcp.json` schema.** Top-level `{"mcpServers": { "<name>": { "type", "command"|"url", "args", "env", "headers", "oauth": {"clientId","callbackPort","scopes","authServerMetadataUrl"}, "headersHelper" } }}`. Env expansion supports `${VAR}` and `${VAR:-default}` in `command`, `args`, `env`, `url`, `headers`.

**Authentication.**
- OAuth 2.0 via `/mcp` with browser flow; tokens stored in OS keychain. `--callback-port`, `--client-id`, `--client-secret` (secret via TTY prompt or `MCP_CLIENT_SECRET` env). `oauth.scopes` pins approved scopes (space-separated); `authServerMetadataUrl` overrides discovery.
- `headersHelper` executes a shell command returning JSON key/value headers — runs fresh on each connection, 10s timeout, requires workspace-trust acceptance. Env: `CLAUDE_CODE_MCP_SERVER_NAME`, `CLAUDE_CODE_MCP_SERVER_URL`.

**Permissions / managed config.**
- Enterprise exclusive-control: `/Library/Application Support/ClaudeCode/managed-mcp.json` (macOS), `/etc/claude-code/managed-mcp.json` (Linux).
- Policy-based: `allowedMcpServers` / `deniedMcpServers` in managed settings, each entry with exactly one of `serverName`, `serverCommand`, `serverUrl` (wildcards).
- Project-scope opt-in: `.claude/settings.local.json` fields `enabledMcpjsonServers: [names]`, `disabledMcpjsonServers`, `enableAllProjectMcpServers: bool`.
- Per-tool permission: standard `permissions.allow/deny/ask` with `mcp__<server>__<tool>` patterns; `ToolSearch` itself is denyable.

**Primitives.** Tools (`mcp__server__tool`), Resources (`@server:protocol://path`, fuzzy @-mention), Prompts (`/mcp__server__prompt [args]`), plus `list_changed` notifications, elicitation (form/URL), and channels (`claude/channel` capability + `--channels` flag).

**Deferred tool loading.** `ENABLE_TOOL_SEARCH` env: unset = defer for first-party hosts, `true` always defer, `auto[:N]` threshold, `false` disable. Server-author hints via instructions (<2KB). Output: 10k-token warn, 25k default cap, `MAX_MCP_OUTPUT_TOKENS` override, per-tool `_meta["anthropic/maxResultSizeChars"]` up to 500k.

**Debug.** `claude mcp list|get|remove`, `/mcp`, `MCP_TIMEOUT` env, auto-reconnect (5 attempts, exponential backoff) for HTTP/SSE only.

## Codebase audit

### MCP server inventory

| Server | Config file | Scope | Transport | Auth | Tools | Status |
|---|---|---|---|---|---|---|
| `slack` | `.mcp.json` (repo root) | project | stdio (npx `@anthropic-ai/mcp-server-slack`) | env: `${SLACK_BOT_TOKEN}`, `${SLACK_APP_TOKEN}` | external | Compliant |
| `alpaca` | `.mcp.json` | project | stdio (uvx `alpaca-mcp-server`) | env: `${ALPACA_API_KEY_ID}`, `${ALPACA_API_SECRET_KEY}`; `ALPACA_PAPER_TRADE=true` | external | Compliant |
| BigQuery MCP | **undeclared** (CLAUDE.md-only) | n/a | injected by harness | n/a | `list_dataset_ids`, `list_table_ids`, `get_dataset_info`, `get_table_info`, `execute_sql_readonly`, `execute_sql` | Drift — documented but not in any config file |
| `backtest_server` | `backend/agents/mcp_servers/backtest_server.py` | in-process (not a Claude Code MCP) | FastMCP in-process | capability tokens (HMAC) | 5 | In-process — not Claude-Code-visible |
| `data_server` | `backend/agents/mcp_servers/data_server.py` | in-process | FastMCP | capability tokens | 1 + 7 resources | In-process |
| `signals_server` | `backend/agents/mcp_servers/signals_server.py` | in-process | FastMCP | capability tokens | 5 | In-process |
| `risk_server` | `backend/agents/mcp_servers/risk_server.py` | in-process | FastMCP | capability tokens | n/a | In-process |
| Legacy stubs | `backend/mcp/*.py` | — | — | — | — | Flagged `stub: true` in inventory; dead code |

`.claude/settings.local.json` shows `enabledMcpjsonServers: ["slack"]` with `enableAllProjectMcpServers: true` — the two fields conflict; `enableAllProjectMcpServers: true` silently supersedes the per-server allowlist, so `alpaca` is enabled even though the allowlist omits it.

### Custom MCP servers written by pyfinAgent

All FastMCP servers in `backend/agents/mcp_servers/` are **in-process only** (no stdio/HTTP endpoint, no `.mcp.json` entry). They expose `backtest`, `data`, `signals`, `risk` primitives to the internal multi-agent orchestrator, wrapped by:
- `backend/agents/mcp_capabilities.py` — HMAC-SHA256 capability tokens (30-min TTL), 6 role→scope bindings, PII scrubber (email/phone/JWT/API-key regexes) with deep-walk of args.
- `backend/agents/mcp_guardrails.py` — `sliding_window_debounce(max_calls=3, window_s=10)` for tool-storm suppression, `cap_output_size(max_bytes=100_000)` with truncation annotation.

These guardrails are **well beyond** what Claude Code's MCP layer provides and are sound defense-in-depth. They do not, however, wrap the *external* Claude-Code-facing MCP servers (`slack`, `alpaca`) — those run as independent subprocesses.

### CLAUDE.md MCP guidance vs actual behavior

| Claim in CLAUDE.md | Reality | Note |
|---|---|---|
| "harness environment injects a BigQuery MCP server" | No entry in `.mcp.json`, `settings.json`, or `settings.local.json` | Drift — BQ MCP is ephemeral/session-only; docs imply it's always present |
| "discover via `ToolSearch` with query `bigquery`" | Correct pattern per docs | OK |
| "Default to `execute_sql_readonly`" | Good least-privilege guidance; tool exists in doc'd inventory | OK |
| "Fall back to `bq` CLI" | Reasonable; 30s timeout rule quoted | OK |
| `mcp__<server-id>__<tool>` pattern | Matches Claude Code convention | OK |

## Findings

| Aspect | Status | Evidence | Notes |
|---|---|---|---|
| `.mcp.json` schema | Correct | `.mcp.json:1-23` | Valid `type`/`command`/`args`/`env`, uses `${VAR}` expansion |
| No embedded secrets | Correct | all server configs use `${...}` | `mcp_inventory.py:27-33` also runs secret-pattern check |
| Project scope choice | Correct | shared `.mcp.json`, team-visible | Appropriate for slack + alpaca |
| Local opt-in allowlist | **Incorrect** | `.claude/settings.local.json` | `enableAllProjectMcpServers: true` makes `enabledMcpjsonServers` field dead |
| OAuth config | Not applicable | both servers use env-var bearer tokens | Slack-hosted `https://mcp.slack.com/mcp` (HTTP + OAuth) in `backend/slack_bot/mcp_tools.py:24` is NOT wired into `.mcp.json` — only used server-side in the bot's LLM calls |
| Per-tool permission allowlist | Missing | `settings.json:76-90` allow list has no `mcp__slack__*` / `mcp__alpaca__*` entries, `defaultMode: bypassPermissions` | In bypass mode the gap is moot today, but if bypass is removed, nothing constrains which Slack/Alpaca tools are callable |
| `permissions.deny` for mutation tools | Missing | no `mcp__alpaca__place_order` or similar in deny | Paper-trade flag `ALPACA_PAPER_TRADE=true` is the sole guard — one env flip = live orders |
| `enabledMcpjsonServers` conflict | Drift | `.claude/settings.local.json:2-4` | Pick one pattern; `enableAllProjectMcpServers: true` + allowlist is contradictory |
| Deferred tool loading | Implicit default | no `ENABLE_TOOL_SEARCH` override | Fine (first-party default defers); project-scope MCP docs in CLAUDE.md mention ToolSearch — accurate |
| `MAX_MCP_OUTPUT_TOKENS` | Not set | — | Default 25k is adequate for current tools; document if BigQuery schemas exceed |
| Custom FastMCP guardrails | Correct and exceptional | `mcp_capabilities.py`, `mcp_guardrails.py` | Capability tokens + PII scrub + debounce + output cap — strong defense-in-depth |
| Legacy stubs | Drift | `backend/mcp/*.py` flagged `stub: true` in inventory | Dead code, should be pruned |
| CLAUDE.md BigQuery MCP claim | Drift | Referenced but never config'd in-repo | Either document that it's harness-injected-only or add a project-scope entry |
| Scoped audit tooling | Correct | `scripts/audit/mcp_inventory.py`, `mcp_risk_pull.py`, `mcp_risk_score.py`, `mcp_ab_test.py`, `mcp_storm_regression.py`, `backend/services/mcp_health_cron.py` | Solid inventory + risk-scoring + health monitoring scaffold (phase-3.5); outputs in `handoff/mcp_*.json` |

## Gaps & Opportunities

**MUST FIX**
1. **Resolve `settings.local.json` contradiction** — either remove `enableAllProjectMcpServers: true` (keep the `enabledMcpjsonServers: ["slack"]` allowlist, re-add `alpaca` explicitly), or drop the allowlist. As written, the allowlist is dead.
2. **Add per-tool `mcp__alpaca__*` deny-list for write operations** — right now only `ALPACA_PAPER_TRADE=true` prevents live trades; one env leak plus `defaultMode: bypassPermissions` = financial exposure. Deny `mcp__alpaca__place_order`, `mcp__alpaca__cancel_order`, etc., in `.claude/settings.json` `permissions.deny`.
3. **Resolve BigQuery MCP drift** — CLAUDE.md documents `execute_sql` (full DML/DDL) as available, but no config file declares the server. Either (a) pin it in `.mcp.json` with explicit OAuth config and `oauth.scopes` restricted to read-only where possible, or (b) update CLAUDE.md to state it's harness-injected and may be absent. Add `mcp__bigquery__execute_sql` (mutating) to `permissions.deny` by default.
4. **Prune legacy `backend/mcp/*.py` stubs** flagged by the inventory — they duplicate `backend/agents/mcp_servers/` and risk confusing future agents.

**NICE TO HAVE**
- **Custom pyfinAgent MCP server exposed to Claude Code** — wrap the existing FastMCP `data_server` / `signals_server` over stdio and register in `.mcp.json` (project scope). Would let the orchestrating Claude Code session query signals, run mini-backtests, and hit the harness verifier without shelling to Python. Guardrails already exist — only the transport wrapper is missing.
- **`managed-mcp.json` deployment** for go-live — once on production hosts, enforce `allowedMcpServers: [slack, alpaca, bigquery]` via `/etc/claude-code/managed-mcp.json` so future agents cannot add unvetted servers mid-session.
- **Adopt `oauth.scopes` pinning** when/if the Slack-hosted MCP (`https://mcp.slack.com/mcp`, referenced in `backend/slack_bot/mcp_tools.py`) is added to `.mcp.json` — pin to `channels:read chat:write search:read` only.
- **Document `MAX_MCP_OUTPUT_TOKENS`** and `ENABLE_TOOL_SEARCH` expectations in CLAUDE.md; current defaults are fine but undocumented.
- **Activate the watchlist gate** — `handoff/mcp_watchlist.md` lists 6 servers with explicit `adopt_condition`s. Wire `scripts/audit/mcp_risk_score.py` into CI so new adoptions auto-score before merge.

## References

- Claude Code MCP docs: `https://code.claude.com/docs/en/mcp`
- `/Users/ford/.openclaw/workspace/pyfinagent/.mcp.json`
- `/Users/ford/.openclaw/workspace/pyfinagent/.claude/settings.json`
- `/Users/ford/.openclaw/workspace/pyfinagent/.claude/settings.local.json`
- `/Users/ford/.openclaw/workspace/pyfinagent/CLAUDE.md` (BigQuery MCP section, lines 84-123)
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/mcp_capabilities.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/mcp_guardrails.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/agents/mcp_servers/{backtest,data,signals,risk}_server.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/backend/slack_bot/mcp_tools.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/scripts/audit/mcp_inventory.py`
- `/Users/ford/.openclaw/workspace/pyfinagent/handoff/mcp_inventory.json`
- `/Users/ford/.openclaw/workspace/pyfinagent/handoff/mcp_watchlist.md`
- `/Users/ford/.openclaw/workspace/pyfinagent/handoff/mcp_risk_scores.json`
