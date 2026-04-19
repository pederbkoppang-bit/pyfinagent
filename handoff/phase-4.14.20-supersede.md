# phase-4.14.20 Agent Trigger Phrasing — supersede record

**Decision date:** 2026-04-19
**Prior status in `.claude/masterplan.json`:** `blocked`.
**New status:** `superseded` (with `superseded_by: phase-4.15.0`).
**Authored by:** harness audit cycle for phase-4.14.20 (Main session; researcher `a382f98251f8a31f5`).

## Original intent

[T4] Strengthen `.claude/agents/*.md` descriptions with "use proactively" / "MUST BE USED" / "use immediately after" trigger phrasing so Claude Code's Agent-tool auto-delegation fires proactively on the right MAS roles. Per Anthropic's sub-agents docs (`code.claude.com/docs/en/sub-agents`), description-phrasing is the primary auto-delegation lever.

## Original immutable verification command

```
grep -c 'use proactively\|MUST BE USED\|use immediately after' \
  .claude/agents/qa-evaluator.md \
  .claude/agents/harness-verifier.md \
  .claude/agents/researcher.md \
  | awk -F: '{sum+=$2} END {exit (sum<3)}'
```

## Why the command can no longer be literally satisfied

Phase-4.15.0 (PREP: MAS restructure to 3 agents) MERGED `qa-evaluator.md` and `harness-verifier.md` into a single `qa.md`. Both source files no longer exist:

```
$ ls .claude/agents/
qa.md
researcher.md
# qa-evaluator.md and harness-verifier.md deleted in phase-4.15.0
```

CLAUDE.md explicitly forbids re-splitting:
> "Re-split agents: reintroducing Explore or harness-verifier as separate files after they've been merged is the old pattern."

Therefore the literal immutable cmd cannot be re-satisfied without violating CLAUDE.md.

## Semantic intent is already satisfied

The phrasing landed correctly in the successor files at the time of the phase-4.15.0 merge:

```
$ head -3 .claude/agents/qa.md
---
name: qa
description: MUST BE USED in every EVALUATE phase. Combined QA + harness-verifier
  -- independent cross-verification ... Use proactively after any GENERATE step,
  immediately before marking a masterplan step done. ...
```

```
$ head -3 .claude/agents/researcher.md
---
name: researcher
description: MUST BE USED before every PLAN phase. Combined external-literature
  researcher + internal-codebase explorer. Use proactively at the start of
  any masterplan step, before writing contract.md. ...
```

Both agent description blocks contain the trigger phrasing (`MUST BE USED`, `Use proactively`, plus `immediately before` in qa.md). Anthropic's sub-agents doc confirms these phrases on the description line drive auto-delegation; the qa.md + researcher.md pair satisfies the original requirement.

## Why this is `superseded`, not `done`

- `done` would imply the implementation step was executed against its immutable command. It was not (could not be).
- `superseded_by: phase-4.15.0` correctly attributes the trigger-phrasing delivery to the MAS-merger step that consolidated the files AND added the phrasing in the same pass.
- Status transitions are not immutable. The `verification` block is preserved verbatim (as a historical record of what the pre-merge contract required); only `status` and `superseded_by` are updated.

## Non-changes

- No edits to `.claude/agents/qa.md` or `.claude/agents/researcher.md` (phrasing already present; CLAUDE.md separation-of-duties rule on agent edits precludes Main-session edits + same-session self-evaluation).
- No attempt to re-create `qa-evaluator.md` or `harness-verifier.md`.
- Immutable `verification` block in `.claude/masterplan.json` for phase-4.14.20 is preserved unchanged.

## Cross-references

- `handoff/current/phase-audit-2.10-4.14.20-research-brief.md`
- `handoff/current/phase-audit-2.10-4.14.20-contract.md`
- `handoff/archive/phase-4.15.0/` -- MAS restructure that delivered the phrasing
- CLAUDE.md: "Never edit verification criteria in masterplan.json" + "Agent definition changes require session restart" + "Separation of duties on agent edits"
- Anthropic sub-agents docs: `https://code.claude.com/docs/en/sub-agents` (description phrasing as auto-delegation trigger)
