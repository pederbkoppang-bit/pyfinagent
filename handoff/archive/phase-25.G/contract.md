# Sprint Contract — phase-25.G — Fix Slack digest P&L

**Cycle:** phase-25 cycle 3
**Date:** 2026-05-12
**Step ID:** 25.G
**Priority:** P0

## Research-gate
Reuses phase-24.5 cycle 4 researcher gate (gate_passed=true, 5 sources). Fix is verbatim per phase-24.5 audit F-1 (wrong endpoint) + F-2 (wrong field key) + F-6 (same in /portfolio slash command).

## Hypothesis
Two-level bug + slash-command parallel:
- `scheduler.py:235,260` calls `/api/portfolio/performance` (legacy in-memory) → should call `/api/paper-trading/portfolio`
- `formatters.py:106,321,365` reads `total_return_pct` → should read `total_pnl_pct`
- `commands.py:138` `/portfolio` slash same wrong endpoint

## Success criteria (verbatim from masterplan)
1. morning_digest_p_and_l_matches_paper_trading_portfolio_endpoint
2. evening_digest_p_and_l_matches
3. portfolio_slash_command_p_and_l_matches

## Plan
1. Replace `/api/portfolio/performance` → `/api/paper-trading/portfolio` (3 sites: scheduler.py morning + evening, commands.py /portfolio)
2. Add `total_pnl_pct` fallback chain in formatters.py (3 sites: L106, L321, L365)
3. Verifier `tests/verify_phase_25_G.py` (9 claims)
4. experiment_results.md
5. Q/A
6. harness_log Cycle 59
7. Flip masterplan 25.G

## References
- `docs/audits/phase-24-2026-05-12/24.5-slack-notifications-findings.md` F-1, F-2, F-6
- `backend/slack_bot/scheduler.py:235,260`
- `backend/slack_bot/formatters.py:106,321,365`
- `backend/slack_bot/commands.py:138`
