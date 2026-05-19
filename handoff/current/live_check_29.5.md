# Live check — phase-29.5 (deep research tier)

**Step ID:** phase-29.5
**Date:** 2026-05-19

## Pre/post line count diff

```
Before: 202 lines (researcher.md)
After:  265 lines (+63)
```

## Verbatim new table row

```
| deep | <=3500 w | <=200 | 40+ | at least 20 (typically 20-50) |
```

## Verbatim new section markers

```
$ grep -nE '^### `deep` tier — additional requirements' .claude/agents/researcher.md
150:### `deep` tier — additional requirements (phase-29.5)

$ grep -nE '^\*\*`deep` gate check' .claude/agents/researcher.md
207:**`deep` gate check:** `gate_passed: true` only if (a) >=20 sources
```

## Post-restart caveat

Per `CLAUDE.md` "Agent definition changes require session restart" rule, the new `deep` tier is on-disk and committed, but the Researcher subagent dispatch loader snapshots `.claude/agents/*.md` at session start. The new tier becomes selectable only after the operator runs `/clear` or restarts Claude Code in the morning.

## Post-restart sanity check (operator recipe)

After restart:
```
1. Ask: "Spawn researcher tier=deep on a small test topic (3 sentences of context)"
2. Verify the researcher's first line mentions tier=deep (not 'moderate' fallback)
3. Verify the brief contains a [ADVERSARIAL] tagged source
4. Verify the brief contains pass 1 / pass 2 / pass 3 markers
```

If the researcher silently downgrades to `complex` instead of refusing or executing `deep`, that's a finding — file as phase-29.5.1 follow-up.

## Honest disclosure

This live_check is **descriptive**, not a live execution. The deep tier cannot be exercised this overnight session because the snapshotted researcher dispatch doesn't know the new tier exists. The first real `deep` run happens in some future cycle after restart.
