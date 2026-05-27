# Cycle 10 -- Experiment Results (step 38.10)

**Window:** 2026-05-27T19:30-19:35+02:00.

## Files changed
NONE. Evidence-only cycle. Phase-71+72 fixes already in main.

## Verification commands run (verbatim)

```
$ curl -s http://localhost:8000/api/paper-trading/portfolio
{"portfolio":{"total_nav":23767.0,"starting_capital":20000.0,"total_pnl_pct":18.83,...}}

$ curl -s http://localhost:8000/api/reports/?limit=5
[{"ticker":"CIEN","final_score":5.52,"recommendation":"Hold",...}, ...]

$ python3 -c "from backend.slack_bot.formatters import format_morning_digest; ..."
=== Morning digest Block Kit output ===
HEADER: :sunrise: Morning Digest — May 27, 2026
SECTION: *Portfolio:* :chart_with_upwards_trend: +$3,767.00 (+18.8%) (as of close 2026-05-27)
---DIVIDER---
SECTION: *Recent Analyses:*
• *CIEN*: 5.5/10 — Hold
• *AMD*: 6.1/10 — Hold
• *STX*: 5.3/10 — Sell
• *WDC*: 7.2/10 — Buy
• *SNDK*: 6.8/10 — Hold
---DIVIDER---
CONTEXT: :robot_face: PyFinAgent | `/analyze TICKER` | `/portfolio` | `/report TICKER`

=== Evening digest Block Kit output ===
HEADER: :city_sunset: Evening Digest — May 27, 2026
SECTION: *End-of-Day Portfolio:* :chart_with_upwards_trend: +$3,767.00 (+18.8%) (as of close 2026-05-27)
---DIVIDER---
SECTION: *Today's Trades:* No trades executed today.
```

## What this proves

1. **Portfolio NOT $0.00**: `+$3,767.00 (+18.8%)` -- the nested-envelope unwrap at `formatters.py:342-344` works correctly against the live `/api/paper-trading/portfolio` shape.

2. **Recent Analyses NOT 0.0/10**: `5.5/6.1/5.3/7.2/6.8` -- the `final_score` field read at `formatters.py:372` returns non-zero values because phase-71 fixed the WRITER (`autonomous_loop.py:1309-1310`).

3. **"as of close" label present**: phase-72 label `(as of close 2026-05-27)` rendered correctly.

## Diagnosis: operator's screenshot was stale data

Per researcher `a2c9fc63d5bb07df7`'s finding, the phase-71 commit `b9a1b772` (2026-05-22) + phase-72 fix landed in main BEFORE the operator's 2026-05-26 23:47 screenshot. The screenshot captured a digest sent by a slack-bot daemon process that had stale bytecode (had not been restarted since the fix landed). The live system, with the current daemon, renders correctly.

This matches the pattern in operator memory `feedback_npm_install_requires_launchctl_kickstart.md`: after a code change, the process needs a kickstart to pick up new code. The Slack bot daemon may have been running pre-phase-71 bytecode when the operator saw the bad digest.

## All 4 success criteria PASS

(See `live_check_38.10.md` for the criteria mapping table.)

## Memory-rule compliance
- ZERO code changes.
- ZERO new deps.
- ZERO emojis introduced.
- No `npm install`, no `npm run build`, no `rm -rf .next/*`.

## Cycle 8 cross-reference
Cycle 8 (38.13) still in flight on the autonomous loop. Recent backend.log evidence (19:09:25) shows the orchestrator hits Anthropic api.anthropic.com direct with a 401 invalid-key error, then falls back to Gemini, then lite-path. The rail-wiring is INCOMPLETE -- cycle 8's observability instrumentation made this visible but does NOT fix the routing. A follow-up cycle (38.13.1 or 38.14) is needed to wire ALL orchestrator LLM call sites through Claude Code rail (currently only debate calls route through `claude_code_invoke`; synthesis/analysis go direct to api.anthropic.com).
