# Contract -- Phase 4.4.5.5: "How to Trade pyfinAgent Signals" Guide

## Target
Checklist item 4.4.5.5: "Documentation: 'How to trade pyfinAgent signals' guide for Peder"

WHO: joint (Ford writes, Peder reviews)
WHEN: launch-week
HOW: the guide lives at `docs/TRADING_GUIDE.md` and covers: signal anatomy, confidence thresholds, sizing, stop-loss execution, and when to override Ford. Peder's sign-off is a Slack acknowledgement in `#ford-approvals`.

## Success Criteria

### A. File structure (SC1-4)
- SC1: `docs/TRADING_GUIDE.md` exists, is valid Markdown, <= 500 lines
- SC2: Exactly 1 new file created. Zero `.py` files touched. Zero existing `.md` files modified (except this contract + handoff artifacts).
- SC3: No emojis in the document (Phosphor Icons rule does not apply to docs, but no emojis per CLAUDE.md)
- SC4: ASCII-only content (no Unicode arrows, em dashes, etc.)

### B. Required sections (SC5-12)
The guide must have all of the following sections, in roughly this order:

- SC5: **Overview** -- what pyfinAgent is, what it does, what it does NOT do (it generates signals, Peder trades manually)
- SC6: **Signal anatomy** -- explains every field in the Slack signal alert: ticker, action (BUY/SELL/HOLD), confidence (0.00-1.00), price, size (USD), stop price, thesis/reason, signal_id, date
- SC7: **Confidence thresholds** -- explains what confidence means (AI model's assessed probability), what ranges are strong vs weak, and how confidence feeds into position sizing (half-Kelly)
- SC8: **Position sizing** -- explains the three-arm hybrid formula: (a) hard percent cap (max 5% equity / max $1000), (b) half-Kelly (0.5 * confidence * equity), (c) inverse-vol. Final size is the MIN of all applicable arms. Default 10% fallback in portfolio_manager when Risk Judge doesn't specify.
- SC9: **Stop-loss execution** -- explains the -8% stop-loss mechanism in portfolio_manager.decide_trades (current_price <= stop_loss_price triggers SELL), the inclusive boundary, and that stop takes precedence over re-eval BUY signals
- SC10: **Risk limits** -- the 4 hardcoded limits: per-ticker 10%, total 100%, drawdown -15% kill switch, max 5 daily trades. Explains these are NOT configurable without a code change (defense-in-depth).
- SC11: **When to override Ford** -- concrete scenarios where Peder should NOT follow a signal (e.g., known earnings event, market circuit breaker, position already has external exposure, personal risk tolerance exceeded)
- SC12: **Daily workflow** -- step-by-step: (1) check morning digest in Slack, (2) review any BUY/SELL signals, (3) decide to act or skip, (4) execute via broker, (5) check evening digest

### C. Accuracy (SC13-18)
- SC13: Strategy parameters cited match `optimizer_best.json`: tp_pct=10.0, sl_pct=12.92, holding_days=90
- SC14: Risk limits cited match `get_risk_constraints`: per-ticker=10.0, total=100.0, drawdown=-15.0, daily_trades=5
- SC15: Stop-loss semantic cited correctly: `current_price <= stop_loss_price` (inclusive boundary, from portfolio_manager.py:80)
- SC16: Position sizing formula cited correctly: min(hard_cap, half_kelly, inverse_vol) from signals_server.py:928
- SC17: Sell-first-then-buy order cited correctly (portfolio_manager.py convention)
- SC18: The guide does NOT promise guaranteed returns, does NOT present the system as infallible, includes a clear disclaimer

### D. Tone and audience (SC19-21)
- SC19: Written for a non-technical reader (Peder). No Python code blocks. No AST references. No internal file paths in the body (those go in an optional appendix if needed).
- SC20: Actionable -- each section tells Peder what to DO, not just what the system does
- SC21: Cross-references: mentions the rollback plan (`docs/ROLLBACK_PLAN.md`), the go-live checklist, and the Slack channels

### E. Global invariants (SC22-23)
- SC22: `git diff --stat` shows exactly 1 file added (`docs/TRADING_GUIDE.md`) plus handoff artifacts
- SC23: No checklist checkbox flipped -- 4.4.5.5 stays `[ ]` until Peder's Slack acknowledgement

## Adversarial Probes

- ADV1: Guide does not contain the word "guaranteed" or "risk-free" in any context suggesting certain profits
- ADV2: Guide does not cite internal Python file paths in the main body (appendix only if present)
- ADV3: Guide correctly states that HOLD signals require NO action from Peder
- ADV4: Guide mentions that paper trading precedes live trading and references the 2-week floor
- ADV5: Guide does not suggest Peder should blindly follow every signal
- ADV6: Guide mentions the escalation path (what to do if something looks wrong)
- ADV7: No code blocks in the main body (pseudocode OK in an appendix only)
- ADV8: ASCII-only -- zero bytes > 127 in the file
