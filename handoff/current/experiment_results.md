# Experiment Results -- Phase 4.2.3.3 (SN-audit ASCII hardening)

**Step:** 4.2.3.3 -- ASCII hardening of `signals_server.py` module docstring
**Commit:** `852e04f` on main (base `9a53cf6`)
**Outcome:** PASS (qa-evaluator 30/30, scores 10/10/10/10/10)

## What landed

Single surgical `replace_all` Edit on `backend/agents/mcp_servers/signals_server.py`. Replaced 7 U+2192 RIGHTWARDS ARROW glyphs in the module header docstring (lines 5-13) with the ASCII substring `->`. Final diff: **+7 / -7**, single file, zero method bodies touched.

## Diff

```diff
--- a/backend/agents/mcp_servers/signals_server.py
+++ b/backend/agents/mcp_servers/signals_server.py
@@ -2,15 +2,15 @@
 MCP Signals Server: Callable tools for signal generation, validation, publishing

 Tools (FastMCP @mcp.tool):
-- generate_signal(ticker, date) \u2192 BUY/SELL/HOLD with confidence
-- validate_signal(signal) \u2192 Check constraints (market hours, liquidity, exposure)
-- publish_signal(signal) \u2192 Post to Slack + portfolio
-- risk_check(portfolio, proposed_trade) \u2192 Can we add this position?
+- generate_signal(ticker, date) -> BUY/SELL/HOLD with confidence
+- validate_signal(signal) -> Check constraints (market hours, liquidity, exposure)
+- publish_signal(signal) -> Post to Slack + portfolio
+- risk_check(portfolio, proposed_trade) -> Can we add this position?

 Resources:
-- portfolio://current \u2192 Current holdings (tickers, shares, PnL)
-- constraints://risk \u2192 Risk limits (max exposure, max drawdown, Sharpe floor)
-- signals://history \u2192 All generated signals this month
+- portfolio://current -> Current holdings (tickers, shares, PnL)
+- constraints://risk -> Risk limits (max exposure, max drawdown, Sharpe floor)
+- signals://history -> All generated signals this month
```

## Why this cycle

Phase 4.2.3.2 qa-evaluator's `ascii_only` audit flagged 7 non-ASCII characters in `signals_server.py`, classified as "pre-existing, not a regression". The prior session noted this as a soft audit miss and recommended a follow-on "SN-audit cleanup" micro-fix as option #1 on the next-run list. This cycle closes that note.

Per `.claude/rules/security.md`, logger messages must be ASCII-only (Windows cp1252 crash risk in uvicorn handlers). The rule strictly binds logger calls, but docstrings are copy-paste vectors into logger calls, so defense-in-depth hardens them too.

## Verification

Lead-self smoke (stdlib only, before commit):

```
Method count: pre=21, post=21
All 21 methods byte-identical at ast.dump level
Top-level structure unchanged: 21 entries
Imports byte-identical: 12
Logger ASCII scan: 0 non-ASCII
U+2192 count: pre=7, post=0
ASCII ' -> ' delta: +7 (pre=50, post=57)
ADV9: SN4 _parse_iso_date regression smoke PASS
ADV10: generate_signal returns dict PASS
ADV7: Class docstring unchanged
ADV8: Module docstring has ASCII arrow
ALL VERIFICATION CHECKS PASS
```

Independent qa-evaluator (Opus, dedicated subagent type, isolated git worktree): **30/30 PASS**. See `evaluator_critique.md`. 5 tool uses, ~62s, no `Stream idle timeout`.

## Budget utilization

- Contract diff cap: `<= 7 / <= 7` -- **exactly 7/7 (100% utilization, 0% overage)**
- Contract SC count: 20 / 20 passed
- Contract ADV count: 10 / 10 passed
- Files touched: 1 code file (plus contract.md, handoff artifacts -- out of scope)
- New imports: 0
- New methods: 0
- New comments: 0
- Renames: 0

## Regression smoke

- `_parse_iso_date("2026-4-1")` -> `date(2026, 4, 1)` -- Phase 4.2.3.2 SN4 scaffold intact
- `_parse_iso_date("not-a-date")` -> `None` -- error path intact
- `generate_signal("AAPL", "2026-01-01")` returns dict with keys `['ticker', 'date', 'signal', 'confidence', 'factors', ...]` -- Phase 4.1 scaffold intact

## Files modified

- `backend/agents/mcp_servers/signals_server.py` (+7 / -7) -- GENERATE commit `852e04f`
- `handoff/current/contract.md` (rewritten for Phase 4.2.3.3) -- GENERATE commit
- `handoff/current/experiment_results.md` (this file) -- LOG commit
- `handoff/current/evaluator_critique.md` (rewritten by qa-evaluator, replacing Phase 4.2.3.2 content) -- LOG commit
- `handoff/current/research.md` (rewritten for Phase 4.2.3.3) -- LOG commit
- `handoff/harness_log.md` (Cycle 4 appended) -- LOG commit
- `.claude/context/sessions/2026-04-14-HHMM.md` (session log) -- LOG commit
