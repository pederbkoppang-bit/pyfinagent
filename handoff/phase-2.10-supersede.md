# phase-2.10 Karpathy Autoresearch Integration — supersede record

**Decision date:** 2026-04-19
**Status in `.claude/masterplan.json`:** `superseded` (was already set; this record formalizes the decision).
**Authored by:** harness audit cycle for phase-2.10 (Main session; researcher `a382f98251f8a31f5`; Q/A to be spawned after write).

## Original intent

Integrate Andrej Karpathy's "autoresearch" paradigm (establish baseline -> propose modification -> measure metric -> keep/discard -> loop) into the pyfinagent meta-orchestration layer. Originally scoped as a standalone step under phase-2 (three-agent harness).

## What actually happened

The Karpathy autoresearch pattern landed in `backend/agents/skill_optimizer.py`, which explicitly self-identifies as the absorber. Verified against source on 2026-04-19:

- `backend/agents/skill_optimizer.py:4` -- module docstring: "Mirrors Karpathy's autoresearch pattern: establish baseline -> propose modification -> measure metric -> keep/discard/crash -> LOOP FOREVER."
- `backend/agents/skill_optimizer.py:129` -- `establish_baseline()` method; docstring calls out the "BASELINE FIRST (autoresearch rule)" step.
- `backend/agents/skill_optimizer.py:270` -- `read_in_scope_files()` method; docstring "READ CONTEXT (autoresearch rule): gather everything relevant before proposing a modification."
- `backend/agents/skill_optimizer.py:453` -- `handle_crash()` method; docstring "CRASH RECOVERY (autoresearch rule): revert the modification, log the crash, and move on."

These four docstrings are direct citations of the autoresearch algorithm's five stages (BASELINE -> READ CONTEXT -> PROPOSE -> MEASURE -> KEEP/DISCARD/CRASH + LOOP). The skill_optimizer implements all five, not merely references them.

Additional touches:
- `backend/agents/meta_coordinator.py` references autoresearch in its docstring but is a deprecated stub not in the active MAS path.
- `.claude/rules/backend-agents.md` lists `skill_optimizer.py` as "Autoresearch-style prompt optimization loop".

No piece of Karpathy's 2023-2026 autoresearch corpus (github.com/karpathy/autoresearch) that was realistically in scope for pyfinagent has been left unimplemented. The external library itself (released 2026-03-07, ~630-line ML training loop) is a narrow training harness, not an SDK, and is not a dep.

## Gap closed by this record

The `superseded` status was set without an audit trail. Before this record, nothing pointed downstream readers at `skill_optimizer.py` as the absorber. `handoff/phase-2.10-supersede.md` (this file) is the formal pointer.

## Forward dependency

`phase-8.5.0` in `.claude/masterplan.json` has an immutable verification command `test -f handoff/phase-2.10-supersede.md`. Creating this file unblocks that step.

## Non-changes

- Masterplan `phase-2.10` status stays `superseded` (not flipped to `done`). The step is not a done implementation step -- it is retired in favor of the absorber.
- No code changes in this audit cycle. `skill_optimizer.py` is not touched.
- No test changes. The skill optimizer's existing tests continue to cover the absorbed behavior.

## Cross-references

- `handoff/current/phase-audit-2.10-4.14.20-research-brief.md` -- research gate for this audit.
- `handoff/current/phase-audit-2.10-4.14.20-contract.md` -- contract for this audit.
- `handoff/archive/phase-2.10/` -- whatever earlier phase-2.10 artifacts exist (if any).
- `handoff/harness_log.md` -- audit cycle entry `Cycle N+45`.
