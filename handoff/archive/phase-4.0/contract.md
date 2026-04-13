# Phase 4.0: Move MAS to OpenClaw — CONTRACT

**Date:** 2026-04-06 22:16 GMT+2  
**Phase:** Phase 4.0  
**Status:** Contract signed

## Hypothesis

Migrating the Multi-Agent System from a standalone Python orchestrator to OpenClaw-native agents will:
1. Eliminate process management complexity (3 processes → 1 gateway + 1 backend)
2. Make Slack always responsive (dedicated agent, never blocked by main)
3. Unify cost tracking and session visibility
4. Remove ~2000 lines of orchestration code (replaced by OpenClaw native features)

## Acceptance Criteria

| # | Criterion | Measurement |
|---|-----------|-------------|
| 1 | Slack messages answered by pyfinagent OpenClaw agent | Send test message, get response within 10s |
| 2 | Main agent (webchat/iMessage) unaffected | Send webchat message during Slack activity, responds normally |
| 3 | Backtest triggerable from Slack | "run a backtest" → agent calls API → reports results |
| 4 | No separate Slack bot process running | `ps aux \| grep slack_bot` returns nothing |
| 5 | MAS Dashboard shows OpenClaw session data | `/api/mas/dashboard` returns agent sessions |
| 6 | Cost tracking unified | All pyfinagent Slack agent calls visible in OpenClaw |

## Boundaries

- **In scope:** Agent migration, Slack binding, workspace setup, dashboard update
- **Out of scope:** New features, optimizer improvements, backtest engine changes
- **Not changing:** FastAPI backend, frontend, BigQuery, backtest engine

## Rollback Plan

If migration fails:
1. Restart the old Slack bot: `cd pyfinagent && python -m backend.slack_bot.app`
2. Remove pyfinagent agent from openclaw.json
3. Restart gateway: `openclaw gateway restart`

Everything is additive until Step 4 (killing old bot), so rollback is safe.

## Sign-off

- [ ] Plan reviewed
- [ ] Workspace created with proper SOUL.md
- [ ] Slack binding configured and tested
- [ ] Old Slack bot stopped
- [ ] Dashboard updated
- [ ] Dead code cleaned up
- [ ] Evaluator critique written
