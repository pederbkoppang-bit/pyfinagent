# live_check — step 67.4 (Post-window Fable revert)

Date: 2026-07-13 (revert executed on schedule; free Fable window ended 2026-07-12).
No `FABLE PERMANENT: AUTHORIZE` token was found in `handoff/harness_log.md` at
session start, so the scheduled revert-to-Opus fired per the phase-67 plan.

## Frontmatter grep (post-revert)

```
$ grep -n "^model:" .claude/agents/researcher.md .claude/agents/qa.md
.claude/agents/qa.md:5:model: opus
.claude/agents/researcher.md:5:model: opus
```

## Gate command (67.4 verification.command) — PASS

```
grep -q "^model: opus" .claude/agents/qa.md \
  && grep -q "^model: opus" .claude/agents/researcher.md \
  && grep -qiE "ruff|pyflakes" .claude/agents/qa.md \
  && grep -q "consumer-contract-break" .claude/skills/code-review-trading-domain/SKILL.md
# -> exit 0 (PASS)
```

## Phase-67 artifact retention (criterion #2) — all PRESENT

- `qa.md` lint gate (ruff/pyflakes): PRESENT
- code-review `consumer-contract-break` heuristic in SKILL.md: PRESENT
- researcher WRITE-FIRST discipline: retained (frontmatter comment + rules/research-gate.md)

## CLAUDE.md steady-state (criterion #3)

CLAUDE.md line 56 "Fable 5 policy" bullet updated: the "Current state" sentence now
reads REVERTED 2026-07-13, no present-tense claim of an active/expired Fable pin.

## Revert commit

Pre-flip HEAD: `838d2398 chore: auto-changelog hook entry for 8463b952`.
The revert commit is the auto-commit-and-push commit created by this step's
status flip to `done` (subject = step 67.4 name). See `git log --oneline` for the hash.

## Roster-live note

Agent-file pins take effect at the NEXT session start (roster snapshot). THIS session
was snapshotted with `model: fable`; the on-disk revert governs every subsequent
session. Next session should run `scripts/qa/verify_qa_roster_live.sh` to confirm the
Opus roster is live before any GENERATE step depends on it.
