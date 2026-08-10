---
name: conditional-counter-86-21
description: 3rd-CONDITIONAL counter reads its source at step CLOSE so it is blind mid-flight (git-confessed in a1b92d14); the rule has THREE copies with TWO different predicates; harness_log is 48.9% unparseable
metadata:
  type: project
---

Step 86.21 measured facts about the 3rd-CONDITIONAL auto-FAIL counter.

**The counter is structurally blind to the steps it governs.** Its prescribed source
`handoff/harness_log.md` is appended in the LOG phase (`CLAUDE.md:223`, runbook `:278`),
i.e. at step CLOSE, so during the verdict sequence it returns zero. This is not a
hypothesis: commit **a1b92d14** (2026-08-09) backfilled cycles 190-194 for phase=36.17
in ONE commit and its own message says *"harness_log.md had ZERO 36.17 rows across five
Q/A cycles, so the grep-based 3rd-CONDITIONAL counter was blind to the entire step and
every Q/A had to be told its own verdict history in the spawn prompt."*

**Why:** the compensating control (Main pasting the history into the spawn prompt) is
the independence defect itself -- the counted party is handed its own count, unattested.

**How to apply:** three things are easy to get wrong on any fix here.

1. **The rule has THREE copies and TWO predicates.** `CLAUDE.md:358-364` = *consecutive*
   run, resets on PASS/FAIL. `.claude/agents/qa.md:512-519` = *cumulative* grep count
   ("2+ result=CONDITIONAL entries"), no reset. `docs/runbooks/per-step-protocol.md:238-240`
   drops the word "consecutive"; `:261` restores it. They give **opposite answers on the
   only real case in the repo** (36.17 at cycle 194: cumulative 2 -> FAIL, consecutive 1
   -> CONDITIONAL allowed). Fix the predicate, not just the source.

2. **The source is 48.9% unparseable.** Of 1205 cycle headers in the 32,308-line log,
   **589 carry no `phase=` token**; 16 are `###` not `##`; a `^## Cycle ... phase=`
   regex has recall **20/26 = 77%** on the CONDITIONAL population (6 real records sit on
   `## phase-10.5.7 -- ... result=CONDITIONAL`-shaped headers). A bare `grep 36.17` as
   qa.md prescribes has **precision 0.50** (12 lines vs 6 real rows). And `result=` has
   **19 distinct values** -- `PASS_AFTER_RETRY`, `PASS_WITH_FINDINGS`, `CERTIFIED_FALLBACK`,
   `BLOCKED`, `SUPERSEDED`, `PENDING` -- so `== "PASS"` does not reset and the reset rule
   has no defined behaviour for most of them. Same class as [[rec-vocabulary-86-20]].

3. **`evaluator_critique_<id>.md` is NOT a drop-in replacement.** >=17 filename shapes
   across 13,483 archived files (`_cycleN`, `_passN`, `_ERRORED`, `.json` vs `.md`,
   4-digit phase-4000 family, plus `_audit`/`_final`/`_main`/`_upgrade`). Worse,
   **one file per cycle is not an invariant** -- 36.17 ran six cycles and leaves a single
   overwritten `evaluator_critique_36.17.md`. It is a presence signal, not a counter.
   `handoff/current/` also still holds critiques for closed steps.

4. **Who can write a ledger is severely constrained.** The Q/A has **no `Write` tool**
   (`qa.md:543`) and the Workflow runtime has **no `fs`** (`qa-verdict.js:36`: the two
   scripts "cannot share a module because the Workflow runtime forbids imports"). So the
   writer must be Main or a hook -- which is architecturally correct under SLSA's
   trusted-control-plane rule, not a workaround. `handoff/audit/*.jsonl` is the existing
   append-only JSONL partition and is not swept by `archive-handoff.sh` the way
   `handoff/current/` is.

Related: [[workflow-runtime-constraints]], [[claude-code-hooks-run-in-parallel]].
