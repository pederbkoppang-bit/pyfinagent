# Alpaca MCP integration masterplan phase -- Experiment results

## What was built

`phase-17` added to `.claude/masterplan.json` (status=pending) with 8
sub-steps, plus a 1-page operator runbook at
`handoff/current/alpaca-mcp-runbook.md`.

## Files changed

1. `.claude/masterplan.json` (inserted phase-17 after phase-16, 8 steps).
2. `handoff/current/alpaca-mcp-runbook.md` (new).
3. `backend/.env` (OAuth2 Broker creds stored separately as
   `ALPACA_OAUTH_CLIENT_ID` / `ALPACA_OAUTH_CLIENT_SECRET` /
   `ALPACA_OAUTH_TOKEN_URL`; gitignored; NOT committed).

## Sub-step structure

| Sub-step | Name | Scope |
|---|---|---|
| 17.1 | Research gate (this cycle) | meta |
| 17.2 | Paste traditional PK paper keys into backend/.env | scope-1 |
| 17.3 | Smoke-test Alpaca MCP tools from Claude Code session | scope-1 |
| 17.4 | Researcher subagent uses Alpaca MCP during a dry-run | scope-1 |
| 17.5 | Wire paper_trader -> ExecutionRouter (bq_sim default) | scope-2 |
| 17.6 | Shadow mode: EXECUTION_BACKEND=alpaca_paper for 5 trades | scope-2 |
| 17.7 | max_notional_usd clamp + rollback runbook | scope-2 |
| 17.8 | Scope-3 prereqs checklist (feeds BLOCKER-4; no live flip) | meta |

## Credential clarification

The user pasted Alpaca Broker-API OAuth2 credentials
(`client_id: AKSF5FVOKXYKG4ATGCJQBA`, secret redacted). Testing showed
the token endpoint returned `unauthorized_client` on standard
`client_credentials` grant — these creds are configured for a 3-legged
`authorization_code` flow, not machine-to-machine. They are for a
Broker-API integration (users authorize pyfinagent to manage their
Alpaca accounts on their behalf) — orthogonal to the MCP server's
needs. Stored separately in backend/.env for a future Broker
integration.

The Alpaca MCP server needs **traditional trading API keys**
(`PK*`-prefix paper key + secret from the Alpaca paper dashboard).
17.2's verification criterion explicitly rejects `PKLIVE*`-prefix keys
and enforces `ALPACA_PAPER_TRADE` != `false`.

## Verification command output (verbatim)

```
$ python3 -c "<contract-verification script>"
ALL_ASSERTS_OK
$ test -f handoff/current/alpaca-mcp-runbook.md && echo RUNBOOK_OK
RUNBOOK_OK
$ python3 -c "import json; json.loads(open('.claude/masterplan.json').read()); print('JSON_OK')"
JSON_OK
```

## Success-criteria coverage

| # | Criterion | Evidence |
|---|---|---|
| 1 | phase-17 entry status=pending | PASS |
| 2 | sub-step ids 17.1..17.8 | PASS |
| 3 | every step has verification.command | PASS |
| 4 | every step has criteria >= 2 | PASS |
| 5 | 17.2 mentions `ALPACA_PAPER_TRADE` + `PK` (not PKLIVE) | PASS |
| 6 | 17.6 mentions `alpaca_paper` or `Alpaca dashboard` | PASS |
| 7 | 17.7 mentions `max_notional_usd` | PASS |
| 8 | 17.8 mentions `BLOCKER-4` | PASS |
| 9 | masterplan.json is valid JSON | PASS |
| 10 | runbook file exists | PASS |

## Scope discipline

- Did NOT execute any phase-17 sub-step (planning cycle only).
- Did NOT edit `.mcp.json` (already has alpaca-mcp-server==2.0.1).
- Did NOT create traditional Alpaca paper keys (user task via dashboard).
- Did NOT touch phase-16 (still in-progress awaiting user ack).
- Did NOT flip any existing masterplan statuses.

## Notes / follow-ups

- User must paste traditional PK paper keys (30-second task at
  https://app.alpaca.markets/paper/dashboard/overview → View API Keys)
  before sub-step 17.2 can pass.
- Broker-API OAuth credentials stored in `backend/.env` are
  valuable for a future "let other users connect their Alpaca accounts
  to pyfinagent" feature but unrelated to this MCP integration.
- Scope-3 (live) explicitly deferred to BLOCKER-4 so the live-capital
  gate stays single-source-of-truth.
