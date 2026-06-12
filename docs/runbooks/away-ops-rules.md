# Away-Ops Rules (goal-away-ops, 2026-06-12 .. operator return ~2026-07-06)

BINDING on every Claude session (scheduled or interactive) operating on pyfinagent while
the operator is away. Created by step 62.0 from the operator-approved plan
(handoff/away_ops/approved_plan_2026-06-12.md). Violating any rail is an automatic FAIL
for the active step and must be reported in the next digest.

## The 10 rails (verbatim from the approved plan)

1. No .env behavior-flag change without a verbatim matching operator_tokens.jsonl line;
   live_check cites the line. No reply = stays dark/paused.
2. Masterplan freeze: no phases/steps beyond 62-65 + existing backlog; criteria immutable.
3. Git: no force-push, no history rewrite, never touch files a session didn't edit, main-only.
4. $0 metered: dev sessions never call Gemini/Anthropic APIs; sentinel asserts daily
   compute-burn <= pinned pipeline baseline; breach => digest-only + P1 ask. ($25 58.1 window
   + existing Gemini pipeline exempt.)
5. Kill-switch stays paused after any breach; only `KILL SWITCH: RESUME` token resumes;
   auto-resume hysteresis stays OFF.
6. Live-fix authority: failing->passing regression test + fresh Q/A PASS + live evidence;
   order/sizing/risk/screener/fee files = trading-behavior by definition => dark+token.
7. claude.ai connectors (Figma, Slack MCP) never load-bearing headless; Slack I/O =
   bot artifacts only.
8. One step per AM session; evidence-only PM; exit at budget; never chain.
9. launchctl authority limited to backend/frontend/slack-bot kickstart; never touch
   watchdog/session plists.
10. Unsure => do nothing, write a token ask, move on. Operator can reply `HALT-DEV` anytime.

## Enforcement layers

- Prompt-level: every kickoff prompt (scripts/away_ops/prompt_*.md, built by 62.3) reads
  this file FIRST and quotes the rails inline (mas-harness cycle_prompt.md precedent).
- Hook-level: .claude/hooks/pre-tool-use-danger.sh blocks force-push variants (incl.
  position-free flags and +refspec), launchctl bootout/unload/remove/disable on
  com.pyfinagent.* labels, and backend/.env write shapes (Bash >>/>, sed -i, tee, perl -i,
  and Edit/Write tool calls) absent a fresh token cursor at
  handoff/away_ops/tokens_cursor (mtime < 6h). Layer-2 mirrors live in
  .claude/settings.json permissions.deny (subagent caveat, issue #40580).
- Sentinel-level: scripts/away_ops/sentinel.sh (62.4) reconciles backend/.env behavior
  flags against handoff/operator_tokens.jsonl daily; mismatch => digest-only mode + P1.

## Token mechanics

- Operator replies in the Slack bot channel; the bot (62.2) appends
  {ts,user,channel,raw,step,key,value} to handoff/operator_tokens.jsonl.
- Sessions read new lines past handoff/away_ops/tokens_cursor, apply them FIRST (flag flip
  + restart + live_check citing the jsonl line number), then advance the cursor (touch
  updates mtime = opens the hook gate for the authorized write, 6h window).
- Open asks live in handoff/away_ops/pending_tokens.json with the EXACT reply string.
- Reserved tokens: `KILL SWITCH: RESUME` (sole trading-resume path), `HALT-DEV` (stop all
  dev sessions; sessions check for it before any step work), `RESUME-DEV`.

## Trading-behavior file list (rail 6 "by definition" surface)

backend/services/paper_trader.py (execution/sizing/stop branches),
backend/services/portfolio_manager.py (swap/decide logic), backend/services/kill_switch.py,
backend/tools/screener.py thresholds, backend/services/execution_router.py,
backend/config/settings.py trading-parameter defaults, any fee/turnover logic.
Touching these = dark build + token, regardless of how bug-like the change looks.
