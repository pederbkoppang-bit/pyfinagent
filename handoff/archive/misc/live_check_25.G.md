# Live-check placeholder — phase-25.G

**Step:** 25.G — Fix Slack digest P&L (endpoint + field key + slash command)
**Date:** 2026-05-12

## Live-check field
> "Slack screenshot showing non-zero P&L in next morning digest"

## Pre-deployment evidence
- 9/9 verifier PASS
- AST clean across scheduler.py, formatters.py, commands.py
- Zero legacy `/api/portfolio/performance` references in `backend/slack_bot/`
- Both digests + `/portfolio` slash all route to `/api/paper-trading/portfolio` (BQ-backed paper trader)
- All 3 `total_return` reads have `total_pnl_pct` primary + `total_return_pct` fallback

## Post-deployment confirmation (to be filled in)
Operator captures next morning digest (06:00 ET per .env override pending — see phase-25.I separately) showing live P&L from the 11 active positions, not `+$0.00 (+0.0%)`.

**Audit anchor for next bucket:** 25.H (recent-analyses ticker dedup).
