# Evaluator Critique — Cycle 4.15.3

Step: phase-4.15.3 Sub-agents + Agent teams + Skills compliance

## Q/A verdict: PASS

Acting agent: `qa-evaluator` (filling in for merged `qa.md` — see
session-cache finding below).

```json
{
  "ok": true,
  "verdict": "PASS",
  "reason": "All 25 doc patterns evidenced; 5 novel findings independently verified via live checks; audit-only (no code modified); researcher exceeded 18-pattern threshold.",
  "violated_criteria": [],
  "violation_details": [],
  "certified_fallback": false,
  "checks_run": [
    "ls -la .claude/agents/",
    "wc -l .claude/agents/*.md",
    "jq '.hooks | keys' .claude/settings.json",
    "jq '.env' .claude/settings.json",
    "jq '.hooks.TaskCompleted' .claude/settings.json",
    "ls ~/.claude/teams/",
    "ls .claude/skills/",
    "wc -l backend/agents/skills/*.md",
    "grep -l '^name:|^description:' backend/agents/skills/*.md | wc -l",
    "grep -c '{{' backend/agents/skills/*.md (sum=252)",
    "grep -c 'MUST BE USED|use proactively' .claude/agents/*.md",
    "grep -i 'permissionMode' .claude/agents/*.md",
    "grep -c 'Bash' .claude/agents/qa.md",
    "git log --oneline -10 -- .claude/agents/",
    "git log -5 --format='%h %an %ae %s' -- .claude/agents/"
  ]
}
```

## Independent live-check results

Every researcher claim confirmed:

| Claim | Live verdict |
|-------|--------------|
| permissionMode missing on both agents | ✓ zero grep matches |
| qa.md tools:Bash contradicts body "NEVER modify files" | ✓ Bash count=5; body contains NEVER statement |
| SubagentStop hook absent | ✓ hook keys = `[PostToolUse, Stop, TaskCompleted, TeammateIdle]` |
| No ~/.claude/teams/ dir (flag unreachable) | ✓ dir does not exist |
| No .claude/skills/ Claude Code skills | ✓ only `masterplan/` exists |
| 28 skill files, 0 YAML frontmatter, 252 {{}} placeholders | ✓ all counts verified |
| Trigger phrasing present on merged agents | ✓ 1 line each containing both phrases |
| `.claude/agents/` self-authored by current session | ✓ git log shows same Claude Code session |

## Session-cache finding (novel, flagged by Q/A)

Q/A escalated this as a FLAG separate from the 25 patterns:

> "Filling in for merged 'qa' agent (phase-4.15.0 merged
> qa-evaluator + harness-verifier into single qa.md). Agent
> registration is session-snapshot; the merge happened mid-session
> so the Agent tool cannot yet dispatch to 'qa'. This evaluation
> was performed by qa-evaluator as a stand-in. A session restart
> is required before 'qa' is dispatchable via the Agent tool.
> Additionally, Pattern 23 (separation-of-duties) is self-
> incriminating: the same Claude Code session authored
> `.claude/agents/qa.md` and is now the one evaluating work that
> depends on it — exactly the gap the audit flags."

This becomes:
- **MF-40 candidate**: Document session-restart requirement for
  agent-definition changes (Claude Code doc behavior; not a code
  bug, but an operator footgun).
- Confirms Pattern 23 (separation of duties) is a real gap, not
  just theoretical.

## Combined verdict: PASS (with meta-flags)

Both halves (researcher + Q/A standin) independently arrive at
the same 25-pattern deviation set. The audit is complete for
this topic.

## Next

Proceed to 4.15.6 (Batches + Files + Citations + Search results).
Full harness loop per step — one step at a time.
