# Evaluator Critique — Cycle 4.15.12

## Q/A verdict: PASS (+ new MF-49 confirmed)

All expected confirmations hold:
- SubagentStop hook present (MF-42 ✓)
- permissionMode:plan on both agents (MF-40 ✓)
- Session-restart + separation-of-duties notes in CLAUDE.md ✓
- bypassPermissions default still live (MF-3 pending)
- sandbox=null still unfixed (MF-4 pending)
- 2 .bak files still present (v3 item 23 pending)

**MF-49 (new, LOW)**: All 3 Bash allow rules in
`.claude/settings.local.json` reference files already moved/deleted
in 4.15.0 (rm qa-evaluator.md, rm harness-verifier.md, mv
per-step-protocol.md → docs/runbooks/). Dead pre-approvals. Prune.

## Combined verdict: PASS

## Next

4.15.13 Claude Code surfaces + CI + Slack + routines/cron.
