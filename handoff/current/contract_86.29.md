# Contract -- step 86.29

**Step**: `86.29` (phase-86, P2, `harness_required: true`) | **Phase**: PLAN
**Date**: 2026-08-10 (23:0x CEST) | **Driver**: Main (`pyfinagent-06`), Opus 5 / effort max
**Written BEFORE any code.** `git diff -- .claude/hooks/archive-handoff.sh` is
empty at this moment; the only 86.29 files on disk are the two research briefs.

---

## 1. Research gate

**PASSED** before this contract -- `handoff/current/research_brief_86.29.md`
plus `research_brief_86.29_rerun.md` (the first run's gate FAILED and was re-run;
both are on disk, and the re-run is the one that passed). Not re-run tonight: the
gate is already satisfied and the standing rule forbids re-running a passed gate
to feel better about it.

**A caution the researcher raised against MY OWN work, carried forward rather
than smoothed:** it critiqued `derive_archive_misattribution_86_29.py` --
"precision is unmeasured and its 2 known positives are one instance". That is
fair and it is why criterion 1 exists. **153 is not settled.**

## 2. Mechanism -- DEMONSTRATED before planning, not asserted (criterion 2)

Everything below was measured tonight, before this contract was written.

**M1 -- the two step-specific globs match ZERO files, for every sid.**
`archive-handoff.sh:160` iterates
`"$CURRENT_DIR/${sid}-"*.md` and `"$CURRENT_DIR/phase-${sid}-"*.md`.

```
sid=86.29   ${sid}-*.md -> 0    phase-${sid}-*.md -> 0
sid=86.6    ${sid}-*.md -> 0    phase-${sid}-*.md -> 0
sid=82.54   ${sid}-*.md -> 0    phase-${sid}-*.md -> 0
```

The globs expect the sid at the **front**, followed by a hyphen
(`4.5.9-contract.md`, per the hook's own comment). The convention since then puts
the sid at the **end**, after an underscore: `contract_82.54.md`. The two never
meet, so `moved=0` always and **the rolling branch is the only one that fires**.

**M2 -- the rolling files are stale, and from THREE DIFFERENT STEPS.** This is
worse than the step text records:

| rolling file | declares | sha256[:16] |
|---|---|---|
| `contract.md` | **phase-82.54** | `9a819a708324f0ba` |
| `experiment_results.md` | **phase-82.6** | `7f1722baf5d632e2` |
| `evaluator_critique.md` | **phase-82.6** | `25833c0305d06a76` |
| `research_brief.md` | **phase-80.2** | `98106f378f1c9446` |

So an archive dir does not merely get "the wrong step" -- it gets a **mixture of
three unrelated steps**, and the mixture is whatever was last written to each
rolling name.

**M3 -- the poison is byte-traceable to the live rolling file.** Both known
positives carry `9a819a708324f0ba`, identical to `handoff/current/contract.md`:

```
rolling  handoff/current/contract.md        9a819a708324f0ba
archive  phase-86.6/contract.md             9a819a708324f0ba
archive  phase-86.26/contract.md            9a819a708324f0ba
diff -q phase-86.6 phase-86.26              silent (byte-identical)
head -1 both                                "# Contract -- phase-82.54"
```

**Recall gate satisfied (criterion 1):** any population method must flag BOTH of
these. They are flagged by construction here because the digest is the discriminator.

## 3. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. The population is RE-DERIVED by a method whose recall is validated against the two known positives (handoff/archive/phase-86.6 and phase-86.26, both of which contain phase-82.54's contract) BEFORE the report is used -- a method that reports either of them clean is rejected, not adjusted. The 610 currently-unparsed dirs are re-classified or explicitly reported as still-unclassified with the reason.
2. The mechanism is demonstrated, not asserted: show that archive-handoff.sh's step-specific globs ($CURRENT_DIR/${sid}-*.md and phase-${sid}-*.md) match ZERO files under the current suffix naming convention, and that the rolling-file branch is therefore the only branch that fires.
3. After the fix, archiving a step produces a directory whose contract.md declares THAT step -- proven by driving the hook against a synthetic step in a scratch tree, never against handoff/archive itself, and showing the before (wrong step) and after (right step) content.
4. The fix does not rely on the hook and the file-naming convention agreeing by convention: either the hook DERIVES the names it copies from the step id, or a check fails loudly when it finds nothing to archive for a step. An archive run that silently copies nothing, or copies another step's files, must be a visible failure.
5. State explicitly whether the 89 already-wrong archive directories are backfilled or left as-is, and if backfilled, show the mapping was derived from git history rather than guessed. Leaving them wrong is an acceptable, stated outcome; leaving them wrong while implying they are fixed is not.
6. Mutation-test the new behaviour: revert the fix and show the archiving test goes red. A guard whose mutant survives does not count.

**Verification command** (immutable):
`bash -c 'test -f .claude/hooks/archive-handoff.sh && bash -n .claude/hooks/archive-handoff.sh'`
-- a syntax check only. It proves criterion 2 and nothing else, so the live_check
carries the real evidence. **Stated here so no future reader mistakes a green
command for a green step.**

## 4. Plan

**P1 -- DERIVE the names from the step id (criterion 4's first branch).** The
hook should copy `contract_<sid>.md` -> `contract.md` etc., i.e. build the source
name from `sid` rather than hoping a glob written for a 2024 convention still
matches. This removes the convention-agreement dependency instead of re-tuning it.

**P2 -- FAIL LOUDLY on an empty archive (criterion 4's second branch).** Both
branches are implemented, not one: if a step archives **zero** step-specific
files AND the rolling files do not declare that step, emit a visible warning. A
silent `copied=4 moved=0` that copies three other steps' work is the exact defect,
and it currently prints as success.

**P3 -- do NOT trust the rolling files.** If `contract_<sid>.md` is absent, the
hook must not silently substitute `contract.md`. That substitution IS the bug.

**P4 -- criterion 3 in a SCRATCH TREE.** Drive the hook against a synthetic step
in a temp repo; show before (wrong step's content) and after (right step's).
**Never against `handoff/archive/` itself** -- the criterion says so explicitly and
the archive is an audit trail.

**P5 -- criterion 1's population, recall-gated.** Re-derive with the digest
method, assert both known positives are flagged BEFORE using the report, and
report the 610 unparsed dirs as classified or explicitly still-unclassified.
**Measure precision too** -- the gate's critique of my census stands, and 86.19
already burned me on a recall-only validation (46 correct dirs reported damaged).

**P6 -- criterion 5: state the backfill decision.** My prior is **leave them
as-is and say so plainly**: the archive is a historical record, the correct
content for each dir would have to be reconstructed from git history, and a wrong
reconstruction is worse than a known-wrong dir. The step explicitly blesses this
outcome provided it is stated rather than implied.

**P7 -- criterion 6 mutation:** revert the derivation and require the archiving
test to go red, with a green control first.

### Explicitly NOT doing

- **Not** running the hook against the real `handoff/archive/`.
- **Not** backfilling the 89 dirs in this step (P6 states the decision).
- **Not** touching `phase-phase-*` directory naming -- that is 86.19's Class A
  root cause (`archive-handoff.sh` interpolating a raw `$sid` that already
  carries the `phase-` prefix). Same file, different defect; splitting it here
  would entangle two steps.

### Risk

`archive-handoff.sh` is live PostToolUse infrastructure on the masterplan write
path. A bug here can strand handoff files or break the auto-commit. Every change
must preserve the existing early-exit discipline (`[ -d "$CURRENT_DIR" ] || exit 0`)
and must never fail the masterplan Write itself.

## 5. References

- `handoff/current/research_brief_86.29.md` + `_rerun.md` (gate PASSED)
- `.claude/hooks/archive-handoff.sh:127-171` (the two branches)
- Measurements M1-M3 above, all re-derivable from the commands quoted
