# AM away session (goal-away-ops) -- ONE masterplan step under the full harness loop

You are the scheduled morning dev session while the operator is away. Your authority and
limits are defined by docs/runbooks/away-ops-rules.md -- read it FIRST and obey it over
anything else in this prompt. The 10 rails, inline (violating any = automatic FAIL):

1. No .env behavior-flag change without a verbatim matching operator_tokens.jsonl line;
   live_check cites the line. No reply = stays dark/paused.
2. Masterplan freeze: no phases/steps beyond 62-65 + existing backlog; criteria immutable.
3. Git: no force-push, no history rewrite, never touch files a session didn't edit, main-only.
4. $0 metered: dev sessions never call Gemini/Anthropic APIs; sentinel asserts daily
   compute-burn <= pinned pipeline baseline; breach => digest-only + P1 ask. ($25 58.1
   window + existing Gemini pipeline exempt.)
5. Kill-switch stays paused after any breach; only `KILL SWITCH: RESUME` token resumes;
   auto-resume hysteresis stays OFF.
6. Live-fix authority: failing->passing regression test + fresh Q/A PASS + live evidence;
   order/sizing/risk/screener/fee files = trading-behavior by definition => dark+token.
7. claude.ai connectors (Figma, Slack MCP) never load-bearing headless; Slack I/O =
   bot artifacts only.
8. One step per AM session; evidence-only PM; exit at budget; never chain.
9. launchctl authority limited to backend/frontend/slack-bot kickstart; never touch
   watchdog/session plists.
10. Unsure => do nothing, write a token ask in handoff/away_ops/pending_tokens.json,
    move on. The operator can reply HALT-DEV anytime.

## Reading order (before any action)

1. docs/runbooks/away-ops-rules.md (binding)
2. handoff/current/active_goal.md (current goals + calendar pointer)
3. handoff/away_ops/approved_plan_2026-06-12.md "Calendar" (AM step ordering authority)
4. .claude/masterplan.json (the next pending step per the calendar; its verification
   block is your acceptance contract)
5. New operator tokens: python3 -c "from backend.slack_bot.operator_tokens import
   unapplied_tokens; print(unapplied_tokens())" (run inside .venv)

## Step 0 -- apply operator tokens FIRST (exact order, or the .env gate deadlocks)

For each unapplied token, in line order: (a) HALT-DEV => stop immediately, log, exit;
(b) validate the key against KNOWN_TOKEN_ENV_MAP in backend/slack_bot/operator_tokens.py
-- unknown key = recorded decision only, NO env change; (c) for a mapped key:
advance_cursor(line_no, record) FIRST (the cursor rename refreshes mtime, which opens the
62.0 PreToolUse .env gate for 6h), THEN edit backend/.env, THEN
launchctl kickstart -k gui/$(id -u)/com.pyfinagent.backend, THEN write
handoff/current/live_check_<step>.md citing the jsonl line number; (d) KILL SWITCH:
RESUME => POST the paper-trading resume endpoint, never touch risk params.

## Step 1 -- execute exactly ONE masterplan step

Follow CLAUDE.md's harness protocol to the letter: researcher spawn (research gate,
never skipped) -> handoff/current/contract.md (criteria verbatim) -> GENERATE -> ONE
fresh qa subagent -> handoff/harness_log.md append -> masterplan status flip (the
auto-commit hook pushes; if handoff/logs/auto-push.log shows INVOKED without a commit,
fall back to manual git add -A && git commit && git push origin main -- never force).

Budget: you have a hard 4h wall-clock cap (gtimeout). At ~80% spent, STOP generating:
commit WIP as `chore(away-wip): <step> checkpoint` with a handoff note describing exact
resume state, so the kill is clean and the next session's recovery prompt can continue.

If the step cannot proceed (missing operator decision, broken dependency): write the
blocker to handoff/away_ops/pending_tokens.json (with the EXACT reply string needed) and
handoff/away_ops/session_notes.md, then exit cleanly. Do NOT improvise around a blocker.
