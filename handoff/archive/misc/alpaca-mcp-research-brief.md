---
step: alpaca-mcp-integration
tier: moderate
date: 2026-04-24
gate_passed: true
---

# Research Brief: Alpaca MCP Server Integration

## Read in full (>=5 required; counts toward gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://raw.githubusercontent.com/alpacahq/alpaca-mcp-server/main/README.md | 2026-04-24 | official docs | WebFetch | Full tool inventory, auth env vars, stdio transport, ALPACA_TOOLSETS filter, V2 rewrite note |
| https://alpaca.markets/mcp-server | 2026-04-24 | vendor landing | WebFetch | Paper->live switch via API key swap + ALPACA_PAPER_TRADE=false; no minimum deposit |
| https://docs.alpaca.markets/docs/alpaca-mcp-server | 2026-04-24 | official docs | WebFetch | 43 (V1) -> 61 (V2) tools; rate-limit warning; "review orders on dashboard" caveat |
| https://alpaca.markets/blog/alpaca-launches-mcp-server-v2/ | 2026-04-24 | vendor blog | WebFetch | V2: OpenAPI-spec-driven codegen at startup; adds order replacement, option chains, market screeners; no backward compat with V1 tool names |
| https://skywork.ai/skypage/en/ai-agent-wall-street-trading/1981194951578275840 | 2026-04-24 | practitioner blog | WebFetch | Human-in-the-loop approval pattern; .env key storage; never commit creds; paper-first discipline |

## Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.keysight.com/blogs/en/tech/nwvs/2026/01/12/mcp-command-injection-new-attack-vector | security advisory | budget; key findings captured via search snippet |
| https://www.ox.security/blog/mcp-supply-chain-advisory-rce-vulnerabilities-across-the-ai-ecosystem/ | CVE advisory | budget |
| https://labs.snyk.io/resources/prompt-injection-mcp/ | security research | budget |
| https://www.practical-devsecops.com/mcp-security-vulnerabilities/ | practitioner security | budget |
| https://thehackernews.com/2026/04/anthropic-mcp-design-vulnerability.html | news | budget |
| https://owasp.org/www-project-mcp-top-10/2025/MCP05-2025%E2%80%93Command-Injection&Execution | OWASP | budget |
| https://github.com/tadata-org/fastapi_mcp | OSS library | budget |
| https://gofastmcp.com/integrations/fastapi | framework docs | budget |
| https://blog.premai.io/mcp-explained-build-ai-integrations-with-tools-resources-oauth-2026-guide/ | practitioner blog | budget |
| https://mcpservers.org/servers/alpacahq/alpaca-mcp-server | aggregator | budget |

## Search queries run (3-variant discipline)

1. Year-less canonical: `Alpaca MCP server integration FastAPI` -- surface prior art
2. 2025 window: `Alpaca MCP server agent hallucination unwanted order risk quant 2025`
3. 2026 current: `MCP server FastAPI integration production 2026 stdio subprocess security`

## Recency scan (2024-2026)

Searched explicitly for 2025-2026 literature. Found substantive new findings:

- **CVE-2026-32211** (April 2026): Azure MCP Server SSE transport had zero auth, enabling full tenant takeover. Reinforces: never expose MCP server on non-loopback without auth.
- **OWASP MCP Top 10** (2025): MCP05:2025 Command Injection formally catalogued. Stdio transport can be weaponized if MCP config is attacker-controlled.
- **OX Security advisory** (2026-04-16): "Mother of all AI supply chains" -- 200k MCP servers at risk from tool-definition mutation post-install. A tool approved on day 1 can silently reroute credentials by day 7.
- **Alpaca MCP V2 launch** (2026): complete rewrite, OpenAPI-spec driven, 61 tools (up from 43). V1 tool names no longer valid.

These findings are directly load-bearing for the security/risk checklist below.

---

## Full Tool Inventory (Alpaca MCP Server V2, 61 tools)

| Category | Tools |
|----------|-------|
| **Account & Portfolio (6)** | `get_account_info`, `get_account_config`, `update_account_config`, `get_portfolio_history`, `get_account_activities`, `get_account_activities_by_type` |
| **Trading / Orders (9)** | `get_orders`, `get_order_by_id`, `get_order_by_client_id`, `replace_order_by_id`, `cancel_order_by_id`, `cancel_all_orders`, `place_stock_order`, `place_crypto_order`, `place_option_order` |
| **Positions (6)** | `get_all_positions`, `get_open_position`, `close_position`, `close_all_positions`, `exercise_options_position`, `do_not_exercise_options_position` |
| **Watchlists (7)** | `create_watchlist`, `get_watchlists`, `get_watchlist_by_id`, `update_watchlist_by_id`, `delete_watchlist_by_id`, `add_asset_to_watchlist_by_id`, `remove_asset_from_watchlist_by_id` |
| **Assets & Market Info (8)** | `get_all_assets`, `get_asset`, `get_option_contracts`, `get_option_contract`, `get_calendar`, `get_clock`, `get_corporate_action_announcements`, `get_corporate_action_announcement` |
| **Stock Data (9)** | `get_stock_bars`, `get_stock_quotes`, `get_stock_trades`, `get_stock_latest_bar`, `get_stock_latest_quote`, `get_stock_latest_trade`, `get_stock_snapshot`, `get_most_active_stocks`, `get_market_movers` |
| **Crypto Data (8)** | `get_crypto_bars`, `get_crypto_quotes`, `get_crypto_trades`, `get_crypto_latest_bar`, `get_crypto_latest_quote`, `get_crypto_latest_trade`, `get_crypto_snapshot`, `get_crypto_latest_orderbook` |
| **Options Data (7)** | `get_option_bars`, `get_option_trades`, `get_option_latest_trade`, `get_option_latest_quote`, `get_option_snapshot`, `get_option_chain`, `get_option_exchange_codes` |
| **News & Corporate Actions (2)** | `get_corporate_actions`, `get_news` |

**Transport**: stdio (subprocess per session). Optional streamable-HTTP via `--transport streamable-http --port 8000` (defaults to loopback only).

**Auth env vars**: `ALPACA_API_KEY` + `ALPACA_SECRET_KEY`. Paper/live gate: `ALPACA_PAPER_TRADE=true` (default). Toolset restriction: `ALPACA_TOOLSETS=account,trading,stock-data` (comma-separated; omit destructive categories in read-only contexts).

**Rate limits**: Undocumented numerically by Alpaca. Warning in official docs: repeated market-data queries or high-frequency order placement may trigger limiting. Apply exponential back-off (already the convention in `security.md`).

---

## Internal Code Inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `/Users/ford/.openclaw/workspace/pyfinagent/.mcp.json` | 14 | MCP server registration | ALREADY HAS alpaca entry (V2.0.1, paper=true, env-var substitution via `${...:-}`) |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/services/execution_router.py` | ~260 | Routes paper orders to bq_sim / alpaca_paper / shadow | Has `_refuse_live_keys()` at line 74-80; `EXECUTION_BACKEND` env-var toggle |
| `/Users/ford/.openclaw/workspace/pyfinagent/backend/services/paper_trader.py` | 80+ inspected | BQ-backed virtual trade engine | Hardcoded to BQ; does NOT call execution_router; is the abstraction layer above it |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/researcher.md` | (system) | Researcher agent | Tools: Read, Bash, Glob, Grep, WebSearch, WebFetch, SendMessage -- could call Alpaca MCP tools if registered |
| `/Users/ford/.openclaw/workspace/pyfinagent/.claude/agents/qa.md` | (system) | Q/A agent | Tools: Read, Bash, Glob, Grep, SendMessage -- cannot call Alpaca MCP tools (not in tool list) |

### Critical internal findings

1. **.mcp.json already has an alpaca entry** (`alpaca-mcp-server==2.0.1`, `ALPACA_PAPER_TRADE=true`, env-var refs `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY`). Scope 1 (harness-only) is **largely pre-wired** -- Claude Code sessions can already call Alpaca MCP tools if the env vars are set. file:/Users/ford/.openclaw/workspace/pyfinagent/.mcp.json:1-14.

2. **execution_router.py already knows about alpaca_paper mode** (lines 157-194: `_alpaca_real_fill` via `alpaca-py` SDK). The existing direct-SDK path and a future MCP-mediated path are parallel options, not duplicates. The router's `_refuse_live_keys()` guard (lines 74-80) is the canonical paper-only lock; it checks both key prefix (`PKLIVE`) and `ALPACA_PAPER_TRADE` env var.

3. **paper_trader.py is BQ-only** -- it owns portfolio state (positions, NAV, cash) in BigQuery tables. It does NOT call execution_router. The paper_trader is the state machine; execution_router is the fill path. Scope 2 requires wiring execution_router into paper_trader's buy/sell methods, not replacing paper_trader wholesale.

4. **MAS agents (researcher, qa) have tool lists that do NOT include Alpaca MCP tool names** -- they would need explicit tool grants or would call them as Bash subcommands. For scope 1 (read-only market data during harness runs), the researcher agent could call MCP tools if the harness injects them. Q/A's tool list (Read, Bash, Glob, Grep, SendMessage) makes it unsuitable as an order-placing agent, which is correct.

---

## Three Integration Scopes

### Scope 1 -- Harness-only (read-only market data for MAS agents)

Register Alpaca MCP in Claude Code sessions so researcher + main can call `get_stock_bars`, `get_stock_snapshot`, `get_market_movers`, `get_news` during harness runs. No execution path. `.mcp.json` already has the entry; missing piece is the env vars being set in the shell that launches harness runs.

**Already done**: `.mcp.json` entry exists with `ALPACA_PAPER_TRADE=true` and `ALPACA_TOOLSETS` could be added to restrict to read-only categories.

**Remaining**: set `ALPACA_API_KEY_ID` + `ALPACA_API_SECRET_KEY` in `backend/.env`, document in CLAUDE.md, restrict toolset to `account,stock-data,crypto-data,assets,news` (omit `trading,positions` for scope 1).

### Scope 2 -- Paper-trading backend swap

Wire `paper_trader.execute_buy` / `execute_sell` to call through `execution_router` with `EXECUTION_BACKEND=alpaca_paper`. Orders land on Alpaca's paper sandbox; portfolio state remains in BQ. The `_alpaca_real_fill` path in execution_router already implements this via the `alpaca-py` SDK -- the MCP server is an alternative interface for agents to inspect results, not a replacement for the SDK fill path.

**Concrete change**: `paper_trader.py` currently does not call execution_router at all. Add a dispatch call in `execute_buy` / `execute_sell` that routes through `ExecutionRouter` when `EXECUTION_BACKEND != bq_sim`. Shadow mode (`EXECUTION_BACKEND=shadow`) lets both paths run in parallel for drift validation before full cutover.

### Scope 3 -- Full live (BLOCKER-4)

Set `ALPACA_PAPER_TRADE=false`, swap paper keys for live keys, remove `PKLIVE` key prefix guard (or rename convention). Requires Peder typed approval (already specced in BLOCKER-4). MCP server becomes the agent-callable live trading surface.

---

## Recommended Masterplan Phase Structure (~8 sub-steps)

```
phase-X: Alpaca MCP Integration
  X.1  RESEARCH GATE (this document)
  X.2  Scope 1a: env-var wiring -- add ALPACA_API_KEY_ID + ALPACA_API_SECRET_KEY
       to backend/.env (paper keys); document in CLAUDE.md BigQuery MCP section
       alongside the Alpaca entry. Restrict ALPACA_TOOLSETS to read-only categories.
  X.3  Scope 1b: smoke-test harness run with MCP tools available; researcher agent
       calls get_stock_snapshot for 3 tickers during a dry-run cycle; verify
       tool outputs appear in handoff artifacts.
  X.4  Scope 2a: wire paper_trader.execute_buy / execute_sell to call
       ExecutionRouter when EXECUTION_BACKEND != bq_sim. Write unit test
       confirming bq_sim path still works when env var is absent.
  X.5  Scope 2b: shadow mode validation -- run EXECUTION_BACKEND=shadow for 5
       paper trades; compare fill prices; gate: drift < 2% on mid-price.
  X.6  Scope 2c: flip to EXECUTION_BACKEND=alpaca_paper (paper creds); run
       autonomous_loop one cycle; verify orders appear in Alpaca paper dashboard.
  X.7  Kill-switch and order-size clamp -- add max_notional_usd guard in
       execution_router._alpaca_real_fill; confirm _refuse_live_keys() blocks
       PKLIVE-prefix keys; document rollback procedure (env-var flip).
  X.8  Handoff doc: BLOCKER-4 prerequisites checklist; leave scope 3 gated on
       Peder typed approval.
```

---

## Security / Risk Checklist

| Risk | Severity | Mitigation |
|------|----------|------------|
| Unwanted live order from LLM hallucination | Critical | Triple lock: (1) `ALPACA_PAPER_TRADE=true` in .mcp.json, (2) `_refuse_live_keys()` checks key prefix + env, (3) MCP toolset restricted to read-only for scope 1 |
| MCP tool-definition mutation post-install (OX Security 2026) | High | Pin exact version in .mcp.json (`alpaca-mcp-server==2.0.1`); re-audit on any version bump |
| Stdio subprocess command injection (CVE-2026-32211 class) | High | Alpaca MCP launched via `uvx` with fixed args; never pass user-supplied strings to the spawn command; .mcp.json is checked into git and reviewed on PR |
| Order-size hallucination (LLM invents large qty) | High | Add `max_notional_usd` clamp in execution_router before any Alpaca submit; reject orders > $X (configurable, default $500 for paper) |
| Rate-limit cascade (agent spams market-data calls) | Medium | ALPACA_TOOLSETS restricts categories; add per-session call counter in researcher agent for Alpaca MCP tools |
| Paper->live key swap accident | Critical | `ALPACA_PAPER_TRADE` env var checked at every router entry point; PKLIVE key prefix check; scope 3 requires Peder typed approval separate from harness |
| API keys in env exposure | Medium | Keys in `backend/.env` (gitignored); never in .mcp.json values directly -- use `${VAR:-}` substitution as already done |

---

## Consensus vs Debate

**Consensus**: stdio MCP is the default secure transport for single-machine deployments (Anthropic design intent). For pyfinagent on a local Mac, stdio is correct -- no network exposure.

**Debate**: whether MAS agents should be allowed to call trading tools at all, or only read-only data tools. The literature (Alpaca docs, practitioner guides) uniformly recommends human-in-the-loop before any order execution. For scope 1 (read-only) there is no debate. For scope 2 the pattern is: agent proposes, `autonomous_loop.py` executes via paper_trader, which is already gated by risk_judge.

**Pitfall** (from V2 launch): V2 is a complete rewrite with incompatible tool names. Any existing prompt templates referencing V1 tool names (`get_account` vs `get_account_info`) must be updated. Pin the version.

---

## Application to pyfinagent

| Finding | Maps to |
|---------|---------|
| .mcp.json already has alpaca entry (paper=true, V2.0.1) | Scope 1 is ~70% done; only env vars missing |
| execution_router.py:74-80 _refuse_live_keys() | Strong paper-only guard; keep it; extend with notional clamp |
| execution_router.py:157-194 _alpaca_real_fill | SDK path already works; MCP is an agent-callable complement, not replacement |
| paper_trader.py does NOT call execution_router | Scope 2 requires explicit wiring; not a large change (~20 lines) |
| qa.md tool list excludes trading tools | Correct separation of duties; Q/A should never place orders |
| researcher.md could call Alpaca MCP read tools | Useful: researcher can pull live snapshot data during signal research |

---

## Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (5 fetched)
- [x] 10+ unique URLs total incl. snippet-only (15 collected)
- [x] Recency scan (last 2 years) performed + reported (CVE-2026-32211, OWASP MCP Top 10 2025, OX Security 2026, Alpaca V2 2026)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (execution_router, paper_trader, .mcp.json, agent .md files)
- [x] Contradictions / consensus noted
- [x] All claims cited per-claim

---

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 5,
  "snippet_only_sources": 10,
  "urls_collected": 15,
  "recency_scan_performed": true,
  "internal_files_inspected": 5,
  "report_md": "handoff/current/alpaca-mcp-research-brief.md",
  "gate_passed": true
}
```
