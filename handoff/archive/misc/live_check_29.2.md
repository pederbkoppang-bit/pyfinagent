# Live check — phase-29.2 (Opus + max effort codify)

**Step ID:** phase-29.2
**Date:** 2026-05-18
**Gate field:** `verification.live_check = "post-restart operator recipe confirms freshly-spawned Researcher reports model: opus + effort: max"`

This is a **frontmatter-edit** step. Pre-restart evidence is on-disk grep; live activation requires session restart.

## Pre-restart on-disk evidence (this cycle)

```
$ grep -nE '^(model|effort|maxTurns):' .claude/agents/researcher.md
5:model: opus
6:maxTurns: 30
13:effort: max

$ grep -nE '^(model|effort|maxTurns):' .claude/agents/qa.md
5:model: opus
6:maxTurns: 12
13:effort: max

$ ! grep -q 'Revert after step closes' .claude/agents/researcher.md && echo "comment removed (researcher)"
comment removed (researcher)

$ ! grep -q 'Revert after step closes' .claude/agents/qa.md && echo "comment removed (qa)"
comment removed (qa)

$ grep -c 'operator override' CLAUDE.md
1

$ grep -c 'Max subscription' CLAUDE.md
1
```

All 8 on-disk checks PASS.

## Post-restart operator recipe (for the morning)

After `/clear` or full Claude Code restart, run:

```bash
# Verify on-disk + origin/main commit visibility
scripts/qa/verify_qa_roster_live.sh
```

The script (existing per phase-23.3.0) checks on-disk state + origin/main + embeds the literal self-disclosure prompt to send a fresh Q/A subagent to confirm the new policy is in its snapshot. The expected self-disclosure responses:

- **Researcher**: should report `model: opus` (or the literal `claude-opus-4-7`) and `effort: max` in its self-introduction.
- **Q/A**: should report `model: opus` and `effort: max` (this was already correct pre-29.2; the change is removal of the "temporarily raised" framing in the inline comment).

If the verification self-disclosure shows the OLD values, the session snapshot didn't refresh — `/clear` again or restart the terminal.

## Why a frontmatter edit doesn't activate immediately

Per `CLAUDE.md` "Agent definition changes require session restart" rule: `.claude/agents/*.md` files are snapshotted by the Agent-tool loader at session START. Adding/merging/renaming/changing-frontmatter mid-session won't make them dispatchable until you `/clear` or restart Claude Code. The retry-on-FAIL doctrine (`docs/runbooks/per-step-protocol.md` §4) handles this.

## Honest disclosure

The Researcher that ran THIS cycle (and will run the next 6 cycles in this overnight session) is still on the Sonnet/max snapshot from session start. That's expected. The next session (morning) is the first one to see the new Opus/max policy live.
