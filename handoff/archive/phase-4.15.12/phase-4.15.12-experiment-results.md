# Experiment Results — Cycle 4.15.12

Step: phase-4.15.12 Claude Code core compliance

## What was built

`docs/audits/compliance-claude-code-core.md` (~1580 words, 25 patterns).

## Newly compliant post-4.15.0 + MF-40-44 fixes

- **SubagentStop hook** now wired (MF-42 ✓)
- **`permissionMode: plan`** on both agents (MF-40 ✓)
- **Session-restart doctrine** in CLAUDE.md (MF-44 ✓ — 2 grep hits)
- **qa.md Bash constraint** narrowed (MF-41 ✓ — "NEVER Edit or Write")
- **Separation-of-duties note** in CLAUDE.md (MF-43 ✓)

## Still missing / pending

| Pattern | Status | MF-# |
|---------|--------|------|
| `InstructionsLoaded` hook (research-gate enforcement) | ❌ missing | MF-27/v3 item 22 |
| `PreToolUse` deny on dangerous Bash (`rm -rf`, `git push --force`, `git reset --hard`) | ❌ missing | v3 item 21 |
| `SessionStart` / `UserPromptSubmit` / `ConfigChange` hooks | ❌ missing | v3 item 22 |
| `bypassPermissions` default (on dev Mac, outside envelope) | ❌ unchanged | MF-3 |
| Sandboxing (`"sandbox": null`) | ❌ missing | MF-4 |
| Stray `.claude/*.bak-*` files (2 still present) | ❌ present | v3 item 23 |
| Memory layout split: 2 systems | ⚠️ unresolved | v2 skills audit |
| `strictKnownMarketplaces` not set | ⚠️ low urgency | v3 cluster F |
| ZDR enrollment | N/A (operator decision) | — |
| settings.local.json: 3 dead one-time migration Bash allow rules | ⚠️ cleanup | new |

## Hook events coverage

| Hook event | Wired? |
|------------|--------|
| PostToolUse | ✓ (Bash commit + masterplan.json Write) |
| TaskCompleted | ✓ |
| Stop | ✓ |
| SubagentStop | ✓ (new post-MF-42) |
| TeammateIdle | ✓ (dead weight per phase-4.10) |
| PreToolUse | ❌ |
| InstructionsLoaded | ❌ |
| SessionStart | ❌ |
| UserPromptSubmit | ❌ |
| ConfigChange | ❌ |
| PreCompact / PostCompact | ❌ |
| SubagentStart / CwdChanged / FileChanged | ❌ |
| WorktreeCreate / Remove | ❌ |
| Elicitation | ❌ |

5 wired of 28 documented. (Doc grew since phase-4.11 — 28 events
now vs ~10 prior; most added events are low priority for us.)

## New MUST-FIX candidate

**MF-49 (MINOR)**: 3 dead Bash allow rules in
`.claude/settings.local.json` reference files deleted in 4.15.0
(e.g., `rm .claude/agents/qa-evaluator.md`). No runtime impact
under `bypassPermissions` but become effective if default mode
changes. Clean up.

## Success criteria

1. every_doc_pattern_status_evidenced — PASS (25 patterns)
2. qa_runs_live_code_checks_not_review — PARTIAL (Q/A next)
3. deviations_cite_doc_page — PASS

## Artifact

- `docs/audits/compliance-claude-code-core.md`
