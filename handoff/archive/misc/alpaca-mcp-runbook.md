# Alpaca MCP Integration Runbook — phase-17

One-page operator summary for staging Alpaca MCP into pyfinagent.

## Three scopes (staged)

1. **Scope-1 (read-only, harness)** — 17.2 → 17.4. LLM agents can query
   account/positions/market-data via MCP. No execution changes.
2. **Scope-2 (paper-trading backend swap)** — 17.5 → 17.7. `paper_trader`
   dispatches through `ExecutionRouter`. Flipping `EXECUTION_BACKEND=alpaca_paper`
   lands orders on Alpaca's paper sandbox instead of the BQ-sim mock.
3. **Scope-3 (live)** — 17.8 is the prerequisites checklist; the live
   flip itself is **BLOCKER-4 (task #46)** and needs typed owner approval.

## Credential types (important)

The Alpaca MCP server needs **traditional trading API keys**:
- Paper: `PK*`-prefix key + `ALPACA_API_SECRET_KEY`, from
  https://app.alpaca.markets/paper/dashboard/overview → "View API Keys".
- Live: `AK*` or `PKLIVE*`-prefix keys from the live dashboard.

Do NOT confuse with OAuth2 Broker-API credentials
(`client_id` / `client_secret` at `authx.alpaca.markets`). Those are for
user-authorize-your-app flows (letting other traders connect their Alpaca
accounts through pyfinagent) and are stored separately in backend/.env as
`ALPACA_OAUTH_CLIENT_ID` / `ALPACA_OAUTH_CLIENT_SECRET` for a future
Broker integration.

## Existing infrastructure

- `.mcp.json` already registers `alpaca-mcp-server==2.0.1` with
  `ALPACA_PAPER_TRADE=true` default and env substitution for the keys.
  **70% of scope-1 is pre-wired.** 17.2 just needs the paper keys pasted
  into `backend/.env`.
- `backend/services/execution_router.py:74-80` already has the live-key
  lockout (`_refuse_live_keys`) that raises on `PKLIVE*` or
  `ALPACA_PAPER_TRADE=false`. This is the triple-lock foundation.
- `backend/services/paper_trader.py` currently bypasses the router
  entirely (BQ-sim only). 17.5 wires it through.

## Execution order

1. **17.2** — paste paper keys. Verify: `ALPACA_API_KEY_ID` starts with
   `PK` (not `PKLIVE`), `ALPACA_PAPER_TRADE` not `false`. 2-minute task.
2. **17.3** — in a fresh Claude Code session, call
   `mcp__alpaca*__get_account_info`. Write the paper-account output to
   `handoff/current/alpaca-mcp-smoketest.md`.
3. **17.4** — run the harness dry-run; confirm researcher-sub agent
   invoked `mcp__alpaca*` tool(s) during the research phase.
4. **17.5** — code change: `PaperTrader.execute_buy/sell` calls
   `ExecutionRouter` (default mode=bq_sim, so behavior unchanged).
   Re-run zero_orders drill to confirm no regression.
5. **17.6** — set `EXECUTION_BACKEND=alpaca_paper`, restart backend,
   drive 5 synthetic BUYs via the new
   `scripts/go_live_drills/alpaca_shadow_drill.py`. Reconcile 5 Alpaca
   paper orders with BQ paper_trades rows. Revert env after.
7. **17.7** — implement `max_notional_usd` clamp in
   `ExecutionRouter._alpaca_real_fill` + write
   `docs/runbooks/alpaca-mcp-rollback.md`.
8. **17.8** — write the scope-3 prerequisites checklist that feeds
   BLOCKER-4. No live flip here.

## Top 3 risks

1. **Live-key accident** — quadruple lock:
   (a) `_refuse_live_keys()` at router;
   (b) `.mcp.json` pinning `ALPACA_PAPER_TRADE=true`;
   (c) 17.2 gate rejecting `PKLIVE*` prefix;
   (d) `max_notional_usd` clamp in 17.7.
2. **MCP tool-definition mutation post-install** (OX Security 2026-04-16) —
   pin exact version `==2.0.1` in `.mcp.json`; review diff on every
   version bump before harness runs.
3. **Order-size hallucination** — `max_notional_usd` clamp ($10000 default)
   raises RuntimeError before any Alpaca `submit()` call.

## Rollback

Env-var flip: `EXECUTION_BACKEND=bq_sim`, restart backend. Paper orders
already on Alpaca dashboard are not auto-cancelled — use
`mcp__alpaca*__cancel_all_orders` or the Alpaca dashboard to flatten.

## What's explicitly out of scope for phase-17

- Live (real-capital) trading — owned by BLOCKER-4.
- Broker-API OAuth integration — park for later.
- Options/crypto trading — Alpaca MCP exposes them but we're not
  wiring them through pyfinagent yet.
