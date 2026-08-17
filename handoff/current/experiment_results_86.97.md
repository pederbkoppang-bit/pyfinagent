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
source **byte-identical**, while an edit inside the definition changes it
(+24 B) — so the extraction is live, and it is specifically the *call* it cannot
see.

**The byte figure is pinned to a commit, because it moves.** At `52358053` (this
step's parent) the extraction is 7,597 B / sha1 `f7458a6ab1f5fe96`; at HEAD it is
**8,617 B / sha1 `072056e58af2befa`**. The mover was this step's *own* criterion-5
docstring correction inside `_log_decision` — one of the four extracted names —
which rode in the same commit that first stated the number. The **property** is
invariant at both commits (call-deleted is byte-identical to base); only the count
is not. Flagged by the cycle-2 Q/A.

The consequence is the reason criterion 4 is worded as it is: **no assertion
added to that file could ever kill this mutant.** Patching `NEEDED` would be
theatre. The only instrument that can observe a deleted call is one that runs
the hook and reads the file it writes — which is also the only instrument that
can reach the three pre-detector `exit 0` paths at all, since those live in bash
*outside* the heredoc and no Python-side test can execute them.

Reproduced with the control observed GREEN first, and the mutant checked to
**build** before being scored (an unbuildable mutant is UNSCORABLE, never a
kill). The buildability oracle is `bash -n` **plus** a `compile()` of the
heredoc body — see below for why the bash half alone is not enough:

```
control INSIDE the worktree : ALL GREEN: 42 passed, 0 failed
MUTANT (call deleted, 25 B) : ALL GREEN: 42 passed, 0 failed   <- SURVIVED
mutant buildability          : bash -n rc=0 AND heredoc compiles -> BUILDABLE, score valid
```

After this step, the same mutant is **KILLED**.

---

## The guard

`scripts/qa/verify_decision_log_86_97.py`, five sections, 35 assertions, exit 0.

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
4. **Mutation of the production guards.** An oracle self-test in both directions,
   control first, then two cells: delete the call, and retarget the write. The
   second exists because the first alone would let a "the call text is present"
   check pass.
5. **Mutation of the `[1]`/`[2]` guards themselves.** Criterion 6 says *every* new
   guard, and cycles 1–2 covered only `[3]`/`[4]`. Seven cells now mutate the
   preconditions, the enumeration recall, the classification keying and the
   isolation check. To make them reachable, the `[1]`/`[2]` logic was refactored
   into one `analyse(src)` function that **both** the shipped assertions and the
   cells consume — a section driving a re-implementation would test a copy, not
   the guard.

**The buildability oracle was wrong in cycle 1, and the failure was the step's
own subject.** `buildable()` was `bash -n` alone — which does **not** parse
inside a quoted heredoc (`<< 'PYEOF'`), and *both* mutants are Python-side edits
inside exactly that heredoc. Measured: the mutant `_log_decision(bump_type`
(unbalanced paren) gave `bash -n` rc=0 while `compile()` raised
`SyntaxError: '(' was never closed`; driven, it produced rc=1 and an empty log,
and the scoring rule `m_log.strip() == ""` recorded it as **KILLED**. A crash was
being counted as a kill — an oracle blind to the failure mode it guards, which is
the same category of defect this step exists to close, reproduced inside the
guard. Found by the cycle-1 Q/A.

Now: `buildable()` checks the bash half **and** compiles the heredoc body; the
oracle is self-tested in both directions (YES to the real hook, NO to a heredoc
`SyntaxError`); and every cell additionally asserts the mutant's **rc == 0**, so
an empty log can only mean the guard caught it and not that the hook crashed.

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
22:12 -> commits=53  decision lines=27  gap=26  recursion-guard commits=27
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
- Until cycle 3, the mutation matrix covered only the `[3]`/`[4]` guards; the
  `[1]`/`[2]` guards had no cell and that subsetting was not disclosed. Now
  covered by section `[5]`, seven cells, all killing.
- The `7,597 B` extraction figure was exact at the parent commit and was
  invalidated by this step's own criterion-5 docstring edit **in the commit that
  stated it**. All six sites now name the commit; the *property* was never wrong.
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
ALL GREEN: 35 passed, 0 failed
0
```

No regression in the sibling gates:

```
verify_changelog_flip_86_91.py     ALL GREEN: 42 passed, 0 failed
verify_workflow_args_boundary.mjs  ALL GREEN: 96 passed, 0 failed
verify_research_gate_workflow.mjs  ALL GREEN: 124 passed, 0 failed
```

---

## Cycle-4 remediation (after the overnight PARK at the 3-attempt cap)

Both blockers named in the park note are closed. Every block below was
regenerated from a live run in the same pass that wrote this file — three
consecutive 86.94 cycles capped on hand-maintained numbers and I am not
repeating that here.

### Blocker A — the guard could not see WHAT the decision was

The park note said the `:305` assertion was weak. **The research gate showed it
was VACUOUS**: `reason=` is a literal in the writer's format string
(`post-commit-changelog.sh:271`), so `"reason=" in log_text` is true for *every
non-empty line the writer can emit* and is strictly subsumed by the
"a line was written" check immediately above it.

Reproduced before fixing, through section [3]'s own `drive()` helper (which takes
the hook **source as a string**, so the production hook is never touched):

```
CONTROL   bump=none   reason=no_flip
N-1       bump=minor  reason=unrecorded     <- delete `bump_type = _flip_magnitude()`
DIFFERENT DECISION: True
BOTH satisfy the shipped assertion "'reason=' in log_text": True
```

A **spurious `minor` bump** (what 86.68 exists to prevent) with an **unexplained
reason** (what 86.91 criterion 4 exists to close), invisible to every assertion
in the file.

**The fix asserts the decision as DATA.** The line is parsed into
`(bump, reason, created_done, transitioned_done)` and compared by **exact
equality** against a table derived from the hook's **branch structure before any
scenario was driven** — the ordering matters, because an oracle written after
looking at a run drifts toward whatever that run produced (arXiv:2410.21136,
arXiv:2402.11041). The nine reason states and their sites are listed in the
source; state 9, `unrecorded`, is assigned by **no branch at all** — it is the
`.get` default at `:267` and therefore the signature of a detector that never ran.

Cycle 3's driver seeded `{"phases": []}` for both revisions, so it exercised
**1 of those 9** states and pinned **0** values. It now drives four, spanning all
four bump magnitudes:

```
       no_flip -- masterplan unchanged
         -> {'bump': 'none', 'reason': 'no_flip', 'created': '-', 'transitioned': '-'}
       flip_transitioned -- 99.1 pending -> done
         -> {'bump': 'patch', 'reason': 'flip_transitioned', 'created': '-', 'transitioned': '99.1'}
       flip_created -- 98.0 appears already done
         -> {'bump': 'minor', 'reason': 'flip_created', 'created': '98.0', 'transitioned': '-'}
       subject_forced_major -- a `!` subject
         -> {'bump': 'major', 'reason': 'subject_forced_major', 'created': '-', 'transitioned': '-'}
  ok   [3a] the scenarios DISCRIMINATE -- they do not all produce one reason
       RE-DERIVED at execution time (window pinned to 2026-08-16T08:23:33Z):
         commits=87  decision lines=44  gap=43
         commits matching the recursion guard=44
```

### Blocker B — the last unbounded carrier of the criterion-4 claim

Swept by claim class, seeded from an **independent** artifact
(`night_diagnostics.md:51`, which names the survivor) rather than from my own
phrasing. The cycle-3 sweep searched `"every invocation"` — my wording — while
the survivor says `"every decision"`; the QGS literature names exactly that
failure (seeding a search with the terms you search for makes recall ~100% and
meaningless).

Recall test: both members named in the masterplan note are found
(`live_check_86.91.md:104`, and `experiment_results_86.91.md:444` — the note said
`:441`, off by three).

`handoff/current/live_check_86.91.md:104` was the only remaining **unbounded**
carrier: a `grep -cE "reach(es|ed)? the detector|pre-detector|bash exit|recursion
guard|86\.97"` over that file returned **0**. Its heading is now bounded **in
place** — *"every decision THAT REACHES THE DETECTOR explains itself"* — with the
reason stated, not appended. The same grep now returns **5**.

### The commits-vs-lines gap, re-derived at execution time (criterion 1)

```
       RE-DERIVED at execution time (window pinned to 2026-08-16T08:23:33Z):
         commits=87  decision lines=44  gap=43
         commits matching the recursion guard=44
```

Not copied: the checker recomputes it per run against a pinned window, and asserts
the **relationship** (gap tracks the recursion-guard count), never a pinned number.

### Mutation matrix — control observed GREEN first

```
  rc=0   ALL GREEN: 48 passed, 0 failed
  CONTROL GREEN.
--- N-1 delete the _flip_magnitude() call (hook :214)
    KILLED   rc=1  FAILED: 41 passed, 7 failed
--- N-2 never record a reason (force the :267 .get default)
    KILLED   rc=1  FAILED: 42 passed, 6 failed
--- N-3 swap flip_created / flip_transitioned
    KILLED   rc=1  FAILED: 46 passed, 2 failed
--- N-4 subject-major branch stops recording its reason
    KILLED   rc=1  FAILED: 46 passed, 2 failed
  rc=0  ALL GREEN: 34 passed, 0 failed
  SURVIVED, as required -- without [3a] the same mutant is invisible.
  => the N-1 kill is attributable to the new decision-content assertions.
killed=4  survived=0  unscorable=0  of 4
N-5 attribution control: OK
production hook byte-identical after the run: True
PASS
```

**N-5 is the attribution control** the contract required: keep the N-1 mutant but
delete the new `[3a]` section, and the guard goes green again at 34/0. So the kill
is attributable to the new decision-content assertions rather than to some other
assertion that happened to move. The production hook is verified byte-identical
after every run.

**Guard: 35 → 48 assertions.** No criterion reinterpreted; no verdict semantics
touched; no masterplan step flipped.
