# KICKOFF -- goal-post-away-review (paste this into a fresh local Claude Code session)

You are resuming pyfinagent after the operator's 8-day absence (2026-06-01 -> 2026-06-10)
during which the autonomous paper-trading run continued. The next goal is fully specified
in the repo; execute it under the mandatory harness protocol (CLAUDE.md).

FIRST ACTIONS (in order):
1. git checkout main && git pull origin main, then merge origin/claude/sweet-feynman-zhs8p3
   (it carries this goal). On conflicts, masterplan STATUS fields from main win.
2. Read CLAUDE.md and handoff/current/goal_post_away_review.md IN FULL before any work.
3. Install the goal (one commit): append the fenced phase-55/56/58 JSON payload to
   .claude/masterplan.json byte-for-byte, replace handoff/current/active_goal.md with the
   refresh payload, add the CLAUDE.md Playwright bullet. If masterplan.json already has a
   phase-55+ or drifted statuses, STOP and ask the operator before renumbering.
   Commit: "feat(masterplan): add phase-55/56/58 post-away-review goal; refresh active_goal"
4. Execute in order: 55.1 -> 55.2 -> 55.3. (53.5 already shipped on main 2026-06-10 and
   53.4 is deferred -- do not resurrect either without an operator ask.)

HARD GATES (non-negotiable):
- Full harness loop per step: researcher FIRST (>=5 sources read in full) -> contract.md
  (criteria verbatim) -> GENERATE -> ONE fresh qa -> harness_log.md append -> masterplan
  flip. No self-evaluation, no verdict-shopping.
- Phase-55 is review-only: $0, NO fixes, NO LLM trading-cycle spend.
- Phase-56+ opens only after phase-55 is done; every fix must cite a 55.x finding ID.
- Phase-57 is NOT pre-installed: 55.3 recommends + specs both variants (LEVER vs FEATURE);
  author and install only the one the operator picks by replying "PHASE-57: LEVER|FEATURE".
- Phase-58 runs NO live cycle until the operator's verbatim "LLM SPEND: APPROVED <budget>"
  is on record; "LLM SPEND: DECLINED" -> the $0 branch.
- DO-NO-HARM: the US momentum core stays byte-identical unless a flag is explicitly
  enabled. 53.1's no-trade-band REJECT is binding -- no naive re-proposal.
- Verify ALL UI claims in the live UI via Playwright MCP (NextAuth wall); put captures in
  the live_checks.

CONTEXT (digest/screenshot evidence is SECONDARY -- re-verify against primary data in 55.1):
Away week: +21.9% (06-01) -> peak +23.4% (06-03) -> +19.2% (06-09). Whipsaw round trips:
MU -6.3% in one day, 000660.KS -9.9%, DELL traded 4x in 9 days; SNDK score flipped
7.0-BUY -> 5.0-HOLD day-over-day; ~100% semis/memory concentration. CODE-CONFIRMED:
paper_trader.py:265 stores total_value in LOCAL currency (KRW-as-USD in paper_trades);
:386-414 SELL fee unconverted. OPEN: UI NAV shows 345,968.86 USD on a $10K fund
(mark_to_market converts correctly, so the root cause is elsewhere -- trace it, suspects
listed in the goal doc). "VS KOSPI" card shows holdings return, not index excess. Slack
"Approve" failed with a missing anthropic API key (the operator's only away-week control
was broken). Nightly watchdog ReadTimeouts (05-27/28, 06-04). 05-28: ALL analyses scored
0.0/10 silently. Kill switch did not trip on the -3.5% 06-05 day (verdict open: audit,
don't presume). Go-live gate 1/5 baseline. Lite mode runs rag/earnings_tone/insider/
patent/news but skips deep_dive/devil's-advocate/risk-assessment -- audit what ACTUALLY
fired via pyfinagent_data.llm_call_log.

OPERATOR CHECKPOINT (after 55.3): post the decision block to Slack (burn estimate from
llm_call_log, expected DoD-2/5/6/7/9 value, finetune-vs-features recommendation + both
phase-57 payloads) and wait for the two verbatim replies before installing phase-57 or
running any phase-58 live cycle. Phase-56 may proceed meanwhile. HARD STOP after 58.1:
refresh cycle_block_summary.md with a crisp operator ask list and stop.
