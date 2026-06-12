# PM away session (goal-away-ops) -- evidence collection + digest, NO dev step

You are the scheduled evening session (22:00 local; the 18:00 UTC trading cycle ended
~21:10 local). Authority: docs/runbooks/away-ops-rules.md -- read FIRST, it overrides
this prompt. The same 10 rails bind you (see prompt_am.md for the inline text; rails 1,
4, 5, 7, 10 are the ones PM sessions most often touch). You ship NO code this session --
evidence, health, and communication only. Hard cap 2h; front-load items 1-4 in the first
hour (the 23:00 evening digest reads the files you write).

## Reading order

1. docs/runbooks/away-ops-rules.md
2. handoff/current/active_goal.md
3. handoff/away_ops/session.log (today's AM session outcome -- report it)
4. New operator tokens (unapplied_tokens(); apply per the AM prompt's Step-0 order --
   tokens are applied by WHICHEVER session sees them first)

## Tasks, in order

1. HEALTH: run bash scripts/away_ops/healthcheck.sh (if present); verify backend/
   frontend/slack-bot launchd state; restart only via kickstart -k (rail 9); append
   findings to handoff/away_ops/session_notes.md.
2. CYCLE EVIDENCE: pull today's trading-cycle BQ rows (financial_reports.paper_trades +
   analysis_results) and update whichever live_check files are wall-clock-gated on cycle
   evidence (61.1 criterion 4 until closed; 65.4 during its proof window; 35.3 when
   active). ALSO record the 02:00 autoresearch scheduled-night outcome (tail
   handoff/autoresearch.log -- START/preflight-only/END OK lines, verbatim) into
   live_check_39.1.md; after the THIRD consecutive post-fix scheduled night
   (06-13/14/15), spawn ONE fresh qa to close 39.1 AND 62.6 together (the 62.6
   criterion-3 coupling; see live_check_62.6.md). Verbatim rows, never summaries. If a
   gated step's evidence is now COMPLETE: spawn ONE fresh qa for that step, and on PASS
   append harness_log.md + flip it. WHILE 62.1/62.2 REMAIN OPEN (2026-06-12 state):
   62.1 needs one digest-from-the-launchd-bot evidence line (grep 'digest sent' in
   handoff/logs/slack_bot.log + the Slack permalink via a server-side read) pasted into
   live_check_62.1.md section C; 62.2 needs the operator's TEST TOKEN: PING round-trip
   (check handoff/operator_tokens.jsonl -- if a line exists, paste it + the threaded ACK
   verbatim into live_check_62.2.md). Then fresh qa per step as above.
3. PENDING ASKS: refresh handoff/away_ops/pending_tokens.json -- every open operator
   decision with its EXACT reply string and age in days.
4. COST: append today's session COST/LIMIT_HIT lines from session.log into
   session_notes.md (June-15 Agent SDK credit burn visibility).
5. NIGHTLY TESTS (only once phase-64 ships them): cd frontend &&
   LIGHTHOUSE_SKIP_AUTH=1 npx playwright test --project=functional --reporter=line;
   record pass/fail counts in session_notes.md.
6. DEFECT WATCH: skim today's backend.log tail for new ERROR-class lines (bounded:
   tail -500); new defect classes go into handoff/away_ops/defect_register.md as
   candidate rows (phase-63 owns triage).
7. WEEKLY (Fridays): write the week summary section in session_notes.md (steps closed,
   fixes shipped, churn metrics vs baseline, EU funnel state).

At ~80% budget: stop, commit everything written (`chore(away-pm): evidence + notes
<date>`), push, exit. Unsure about anything => rail 10.
