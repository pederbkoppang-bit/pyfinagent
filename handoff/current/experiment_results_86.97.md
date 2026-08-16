# experiment_results — phase-86.97

## What changed

| file | change |
|---|---|
| `scripts/qa/verify_decision_log_86_97.py` | **NEW.** The end-to-end guard: preconditions, source-derived enumeration, real-hook driver, mutation cells. |
| `.claude/hooks/post-commit-changelog.sh` | **Docstring only.** Corrects the `_log_decision` scope overclaim. No behavioural change. |
| `handoff/current/experiment_results_86.91.md` | §5 heading + claim bounded (criterion 5). |
| `handoff/current/contract_86.91.md` | P2 plan claim bounded (criterion 5). Its **verbatim immutable criterion at :105 is untouched.** |

Full measured evidence: `live_check_86.97.md` §A–G.

---

## The finding

**The 86.91 guard is not weak — it is structurally blind, and the mutant is
INVISIBLE rather than surviving.**

`detector_source()` collects only definition nodes (`FunctionDef` / `Assign` /
`AnnAssign`). A bare `_log_decision(bump_type)` is `ast.Expr(Call)` and binds no
name, so it can never match. Deleting the production call leaves the extracted
source **byte-identical** (7,597 B, sha1 `f7458a6ab1f5fe96`), while an edit
inside the definition changes it (+24 B) — so the extraction is live, and it is
specifically the *call* it cannot see.

The consequence is the reason criterion 4 is worded as it is: **no assertion
added to that file could ever kill this mutant.** Patching `NEEDED` would be
theatre. The only instrument that can observe a deleted call is one that runs
the hook and reads the file it writes — which is also the only instrument that
can reach the three pre-detector `exit 0` paths at all, since those live in bash
*outside* the heredoc and no Python-side test can execute them.

Reproduced with the control observed GREEN first, and the mutant checked to
**build** before being scored (an unbuildable mutant is UNSCORABLE, never a
kill):

```
control INSIDE the worktree : ALL GREEN: 42 passed, 0 failed
MUTANT (call deleted, 25 B) : ALL GREEN: 42 passed, 0 failed   <- SURVIVED
bash -n on the mutant        : parses -> BUILDABLE, so the score is valid
```

After this step, the same mutant is **KILLED**.

---

## The guard

`scripts/qa/verify_decision_log_86_97.py`, four sections, 20 assertions, exit 0.

1. **Preconditions.** The classification rule is lexical, so its three soundness
   conditions are *asserted*: no bash functions, no `trap`/`source`/`.`/`eval`,
   exactly one heredoc with a uniquely-located terminator. If any becomes false
   the gate goes red and says why, instead of the classification silently
   becoming wrong.
2. **Enumeration by a written-down rule, with a self-test.** The rule is stated
   in source; it is cross-checked against a deliberately dumber scan, and every
   line the dumb scan finds but the rule misses must be a comment. A scan that
   matches nothing fails rather than reporting a clean bill of health.
3. **End-to-end.** Runs the **real hook** in a temp git repo (`CLAUDE_PROJECT_DIR`
   isolation) and asserts on the **decision-log file**. Isolation is asserted,
   not hoped for: the real log is snapshotted and required byte-identical, since
   the hook ends in `git add` + `git commit`.
4. **Mutation.** Control first, then two cells: delete the call, and neuter the
   write. The second exists because the first alone would let a
   "the call text is present" check pass.

### The classification is keyed on condition text, and that was proven live

Line numbers are the fixture that rotted in phase-86.92, so the table keys on
each guard's **condition text**. This got an unplanned live test: my own
docstring correction added 16 lines, moving the heredoc from `43..371` to
`43..387` and the post-detector exits from `378/380/381` to `394/396/397`. The
enumeration tracked the move **with no edit** and the classification still
matched.

**An unclassified pre-detector exit FAILS the gate** — a fourth early exit
cannot be waved through; it must be classified deliberately.

---

## Classification (criteria 2, 3)

| path | class | reason |
|---|---|---|
| recursion guard (`^chore: (auto-changelog\|changelog drift)`) | **LEGITIMATELY-SILENT** | The hook is re-entering itself; such a commit is by construction not a bump candidate, so there is no decision to explain. **A bound to state, not a defect to fix** — and it accounts for essentially the entire gap. |
| `! -f "$CHANGELOG"` | **MUST-LOG** | The CHANGELOG is gone. Every later commit silently produces nothing and the decision log is the only place that would surface it. |
| no `### Recent Activity` | **MUST-LOG** | Same class: silent structural breakage that looks identical to "nothing to do". |
| the three post-detector exits | n/a | The detector already ran; a decision line already exists. |

Driven, not argued: an auto-changelog commit exits 0 and writes **no** line.

---

## The gap, re-derived rather than copied (criterion 1)

The filed figure "10 commits vs 5 decision lines" is a stale snapshot. The
checker recomputes at run time with the window pinned to the log's own first
timestamp (a bare date would slide with the clock). Measured twice, hours apart:

```
21:03 -> commits=47  decision lines=24  gap=23  recursion-guard commits=24
21:52 -> commits=51  decision lines=26  gap=25  recursion-guard commits=26
```

It moves — which is the point, and why the checker asserts the *relationship*
(gap ≈ recursion-guard count) rather than any number.

---

## Criterion 5 — the correction REPLACES

Three sites, enumerated by the claim's semantics rather than by my own wordings.
That mattered: my first sweep found two, and searching for the unbounded
assertion form surfaced a third (`contract_86.91.md:145`, in the plan section
rather than the criteria copy).

`experiment_results_86.91.md` already *contained* the correct bound — 265 lines
below the claim it qualifies, where a reader of §5 would never meet it. That is
exactly the "accompany, don't replace" failure the criterion names.

Not edited, deliberately: the verbatim immutable criterion at
`contract_86.91.md:105` and in `.claude/masterplan.json` (amending a criterion is
forbidden); `evaluator_critique_86.91.md` (an evaluator's verdict is not mine to
rewrite); and the verbatim checker output in `live_check_86.91.md`.

---

## Scope honesty — what I did NOT do

- Did **not** patch `detector_source` to "see" the call. It is a category error;
  criterion 4 explicitly forbids pretending otherwise.
- Did **not** add `bats-core`. The research names it as the idiom, but a plain
  Python driver matches this repo's existing `scripts/qa/` shape and adds no
  dependency. Disclosed as a deviation from the researched recommendation.
- Did **not** change any bump semantics, flip any step, or alter any verdict.
- Did **not** fix the two MUST-LOG paths. The step's criteria require them
  *enumerated and classified*, not repaired; making the hook log from a
  pre-detector bash path is a behavioural change to a hook that runs on every
  commit, and it belongs in its own step with its own criteria. **Filed as
  86.103** rather than left as prose.
- `bump_type = _flip_magnitude()` (hook `:214`) is the second call site the
  research surfaced. It is covered incidentally by the end-to-end driver (if it
  were deleted the hook would fail), but it has **no dedicated mutation cell**.
  Stated rather than implied.

---

## Verification

```
$ bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'
parses

$ python scripts/qa/verify_decision_log_86_97.py ; echo $?
ALL GREEN: 20 passed, 0 failed
0
```

No regression in the sibling gates:

```
verify_changelog_flip_86_91.py     ALL GREEN: 42 passed, 0 failed
verify_workflow_args_boundary.mjs  ALL GREEN: 96 passed, 0 failed
verify_research_gate_workflow.mjs  ALL GREEN: 124 passed, 0 failed
```
