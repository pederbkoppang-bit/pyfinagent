# Contract -- Alpaca MCP integration masterplan phase (task #50)

## Research gate

- Researcher spawn: 2026-04-24. Brief at `handoff/current/alpaca-mcp-research-brief.md`.
- JSON envelope: tier=moderate, external_sources_read_in_full=5 (floor 5), urls_collected=15, recency_scan=true, internal_files_inspected=5, gate_passed=true.
- Key findings:
  - Alpaca MCP v2 exposes ~61 tools across 9 categories (account, trading, positions, stock-data, crypto-data, options, watchlists, assets, news).
  - stdio transport (subprocess) by default; optional streamable-HTTP on loopback.
  - Auth: `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` env vars (traditional keys, not OAuth). `ALPACA_PAPER_TRADE=true` default.
  - Toolset restriction via `ALPACA_TOOLSETS=account,stock-data,...` — useful to disable `trading` for read-only phases.
  - `.mcp.json` at project root **already registers** alpaca-mcp-server==2.0.1 with paper-mode env substitution — 70% of scope 1 is pre-wired.
  - `execution_router.py:74-80` already has the paper-only lockout. `paper_trader.py` does NOT call execution_router today (BQ-sim only).
- **Credential clarification (in-session finding):** the OAuth2 creds Peder pasted (`AKSF5FVOKXYKG4ATGCJQBA` + secret) are a Broker API OAuth app (user-authorize pattern), not traditional trading keys. They are stored in `backend/.env` as `ALPACA_OAUTH_CLIENT_ID` / `ALPACA_OAUTH_CLIENT_SECRET` for a future Broker integration. The MCP server still needs traditional `PK*`-prefix paper keys from the Alpaca paper dashboard.

## Phase structure (3 scopes, staged)

Scope 1 (harness-only, read-only): MCP server available to Claude Code + MAS agents for market data. No execution path changes.
Scope 2 (paper-trading backend swap): `paper_trader` dispatches to Alpaca MCP paper endpoint instead of BQ-sim. Orders land on Alpaca's paper sandbox.
Scope 3 (live): BLOCKER-4 territory -- Peder typed approval required. Out of scope this planning cycle.

## Planned change (PLANNING only; no app code this cycle)

Add `phase-17` to `.claude/masterplan.json` with 8 sub-steps:

| Sub-step | Name | Scope | Gates |
|---|---|---|---|
| 17.1 | Research gate (this cycle) | meta | brief exists |
| 17.2 | Paste traditional PK paper keys into backend/.env | scope-1 | `ALPACA_API_KEY_ID` starts with `PK` AND `ALPACA_PAPER_TRADE=true` (no `PKLIVE*`) |
| 17.3 | Smoke-test: MCP tools reachable from Claude Code session | scope-1 | `mcp__alpaca*__get_account_info` returns paper account, non-zero buying_power |
| 17.4 | Researcher spawn uses Alpaca MCP during a dry-run | scope-1 | researcher brief cites >=1 Alpaca MCP tool call |
| 17.5 | Wire paper_trader.execute_buy/execute_sell through ExecutionRouter | scope-2 | unit test: BUY AAPL $100 -> BQ paper_trades row + execution_router mode=bq_sim (still mock until 17.6 flip) |
| 17.6 | Shadow mode: flip `EXECUTION_BACKEND=alpaca_paper` for 5 trades | scope-2 | 5 paper trades land in Alpaca dashboard AND drift from BQ-sim ledger < 2% |
| 17.7 | Add max_notional_usd clamp + rollback runbook | scope-2 | clamp raises on > $10000 single order; env-var flip documented |
| 17.8 | Scope-3 prerequisites checklist (feeds BLOCKER-4) | planning | list of preconditions written; does NOT flip paper->live |

## NOT in scope this cycle

- Editing `.mcp.json` (already has the entry).
- Creating traditional keys (Peder manual task).
- Flipping to live (BLOCKER-4).
- OAuth Broker integration with the creds Peder pasted (valuable but orthogonal; park as future work).

## Immutable success criteria

1. `.claude/masterplan.json` contains a top-level entry with `id: "phase-17"` whose `status` is `pending`.
2. That entry has exactly 8 sub-steps with ids `17.1` through `17.8`.
3. Every sub-step has a non-null `verification.command` string.
4. Every sub-step has `verification.success_criteria` list of length >= 2.
5. Sub-step `17.2` criterion includes the literal string `ALPACA_PAPER_TRADE` and `PK` (not `PKLIVE`).
6. Sub-step `17.6` criterion mentions `alpaca_paper` or `Alpaca dashboard`.
7. Sub-step `17.7` criterion mentions `max_notional_usd` clamp.
8. Sub-step `17.8` criterion mentions `BLOCKER-4` by name (so scope-3 stays linked to the existing live-gate task).
9. JSON validity: `python -c "import json; json.loads(open('.claude/masterplan.json').read())"` exits 0.
10. `handoff/current/alpaca-mcp-runbook.md` exists with a one-page operator summary.

## Verification command (Q/A reproduces)

```bash
source .venv/bin/activate
python3 -c "
import json
mp = json.loads(open('.claude/masterplan.json').read())
def find(n, tid):
    if isinstance(n, dict):
        if n.get('id') == tid: return n
        for v in n.values():
            r = find(v, tid)
            if r: return r
    elif isinstance(n, list):
        for i in n:
            r = find(i, tid)
            if r: return r
p17 = find(mp, 'phase-17')
assert p17 is not None and p17.get('status') == 'pending'
steps = p17.get('steps', [])
ids = [s.get('id') for s in steps]
assert ids == [f'17.{i}' for i in range(1, 9)], ids
for s in steps:
    v = s.get('verification') or {}
    assert v.get('command'), f'{s[\"id\"]} missing command'
    assert isinstance(v.get('success_criteria'), list) and len(v['success_criteria']) >= 2, f'{s[\"id\"]} < 2 criteria'
# literal checks
s2_text = ' '.join(next(s for s in steps if s['id']=='17.2')['verification']['success_criteria']).lower()
assert 'alpaca_paper_trade' in s2_text and 'pk' in s2_text and 'pklive' not in s2_text, '17.2 missing paper/PK gate'
s6_text = ' '.join(next(s for s in steps if s['id']=='17.6')['verification']['success_criteria']).lower()
assert 'alpaca_paper' in s6_text or 'alpaca dashboard' in s6_text, '17.6 missing alpaca_paper or Alpaca dashboard'
s7_text = ' '.join(next(s for s in steps if s['id']=='17.7')['verification']['success_criteria']).lower()
assert 'max_notional_usd' in s7_text, '17.7 missing max_notional_usd'
s8_text = ' '.join(next(s for s in steps if s['id']=='17.8')['verification']['success_criteria']).lower()
assert 'blocker-4' in s8_text, '17.8 missing BLOCKER-4 link'
print('ALL_ASSERTS_OK')
"
test -f handoff/current/alpaca-mcp-runbook.md
```

## References

- `handoff/current/alpaca-mcp-research-brief.md` (research deliverable)
- `.mcp.json` (existing Alpaca MCP registration)
- `backend/services/execution_router.py` (paper-only lockout)
- `backend/services/paper_trader.py` (BQ-sim path to swap in 17.5)
- BLOCKER-4 (task #46) — live-capital gate that scope-3 feeds into
