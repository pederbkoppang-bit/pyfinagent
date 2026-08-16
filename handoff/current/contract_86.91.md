# Contract -- phase-86.91

**Step:** `86.91` (P1)
**Title:** the changelog version detector silently swallows every step FILED AND
CLOSED IN THE SAME COMMIT
**Written:** 2026-08-16, AFTER the research gate returned. **Cycle:** 1

---

## 1. Research gate -- PASSED (enforced, not self-reported)

| Field | Value |
|---|---|
| Rail | `.claude/workflows/research-gate.js` launched by **scriptPath** |
| Run | `wf_6f758470-f84` |
| Brief | `handoff/current/research_brief_86.91.md` (21,062 chars, `brief_status: COMPLETE`) |
| Sources read in full | **8** (floor 5) · URLs collected **28** (floor 10) |
| Recency scan | performed · audit-class: NO |
| `gate_passed` | **true**, RECOMPUTED by the script; self-report agreed |

Sources read in full: PEP 661 (sentinel values, Final 2026-04-23);
python-patterns.guide sentinel object; semantic-release FAQ; git-scm `githooks`;
Schiller, *changesets vs semantic-release*; *intentional releases with
changesets*; arXiv 2605.02033 (commit classification); arXiv 2408.01760 (ISSTA
'24, equivalent mutants 4-39%).

---

## 2. Hypothesis (already reproduced -- see §3)

`_flip_magnitude` decides a version bump from the masterplan `id -> status` diff
between `HEAD~1` and `HEAD`. Its predicate is

```python
newly_done = [sid for sid, st in after.items()
              if st == "done" and before.get(sid) not in (None, "done")]
```

`before.get(sid)` returns `None` for a step that **did not exist** at `HEAD~1`.
The `None` exclusion was meant to say *"not a transition"*; it actually says
*"ignore any step that appeared this commit"* -- and that is the file-it-and-fix-it
workflow this project uses constantly. **Absence is being encoded as the same
value as a legitimate null.** The remedy is a key-space membership test with an
explicit sentinel, and three states rather than two: CREATED-done, TRANSITIONED-done,
ALREADY-done.

---

## 3. Reproduction ALREADY PERFORMED (criterion 1, before any change)

On the real commit `e4f2e844` (phase-86.86, 2026-08-15), verbatim:

```
86.86 before: None -> after: done
OLD rule newly_done: []
NEW rule newly_done: ['86.86']
```

Baseline replay over the current corpus, verbatim from
`scripts/qa/replay_changelog_rule_86_68.py`:

```
corpus: 621 commits since 2026-08-11 (re-derived at execution time)
  version bumps under OLD rule : 210
  version bumps under NEW rule : 5
```

And the symptom: `### v6.93.222 ... (2026-08-14)` is still the newest version
header while Recent-Activity rows landed through 2026-08-15.

---

## 4. Three research findings that shape the fix

- **I3 -- a stderr marker CANNOT close criterion 4.** `grep -c "flip-detect
  FAILED" handoff/logs/auto-push.log` = **0** over 976,895 bytes, so today's
  freeze is the silent `[]`, not errors. And git's "stderr reaches the user"
  guarantee does **not** apply: this is not a git hook (`.git/hooks/` holds only
  `pre-commit`), and `auto-commit-and-push.sh:396` runs it as
  `>> "$LOG_FILE" 2>&1`. A fix that adds a stderr line and stops there is **as
  invisible as the current silence**. => the decision must be written to a
  **named, deterministic decision log**, not merely emitted.
- **I5 -- `replay_changelog_rule_86_68.py:54` mirrors the same predicate.** Fix
  the hook alone and criterion 3's three numbers compare the fixed hook against a
  **stale baseline**. Both files change; the replay must express all three arms.
- **ISSTA '24: equivalent mutants are 4-39% of real mutants** and equivalence is
  undecidable. Criterion 6's "control observed GREEN first" is therefore
  load-bearing, not ceremony -- without it a cell is unscorable.

Prior art **inside the repo**: `handoff/archive/phase-86.68/contract.md:106-107`
predicted this exact step -- *"A silent failure mode in a never-raise detector is
worth a criterion of its own if it proves true; file it, do not absorb it here."*

Explicitly NOT imported: arXiv 2605.02033's methodology (one repo, no train/test
split, assumes the declared prefix is ground truth). This repo's existing
three-arm, full-corpus, mutation-gated replay is stronger; keep it.

---

## 5. Immutable success criteria (VERBATIM from `.claude/masterplan.json`)

1. the defect is REPRODUCED by execution before anything is changed, on a REAL commit from history, quoting the command and its verbatim output showing before=None -> after=done and the shipped rule returning an EMPTY newly_done
2. the fix makes a created-already-done step count as a flip, and the exact predicate change is stated; a fix that special-cases 86.86 or any single step id rather than the CLASS fails this criterion
3. the 86.68 defect is NOT reintroduced: the replay over the same 348-commit corpus is re-run and the bump count is STATED for the OLD subject-prefix rule, the SHIPPED flip rule, and the FIXED flip rule -- three numbers, from execution, not asserted. An increase over the shipped rule must be accounted for commit by commit, and every newly-bumping commit must be shown to have actually closed a step
4. the silent-swallow class is closed, not just this instance: when the detector decides 'none' it must be possible to tell WHY from the hook's own output (a genuinely-chore commit, an error, or a flip it declined) -- an unexplained 'none' is the defect and a fix that leaves it unexplained does not close it
5. CHANGELOG.md is NOT hand-edited: any correction to the version line is produced by running the hook, and the evidence shows the hook producing it
6. a regression guard is added that would go RED if the None exclusion (or an equivalent) is restored, and it is mutation-tested with the control observed GREEN first and the mutant KILLED
7. the hook still NEVER RAISES: an internal error must still print the FAILED marker and bump nothing, demonstrated by injecting a fault rather than by reading the source
8. verdict semantics and masterplan state are UNCHANGED: nothing here may flip a step or alter a verdict

**Immutable verification command** (run before the criteria were frozen; exit 0):

```
bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'
```

**Note on criterion 3's "348-commit corpus".** The replay script **re-derives its
corpus at execution time** as "commits since 2026-08-11", which is 621 today and
was 348 when phase-86.68 measured it. The criterion is immutable and is not
amended: it will be answered on **both** windows -- the pinned 348-commit window
that 86.68 actually measured, and the current full corpus -- with both stated.
Reporting only the larger window would silently answer a different question.

---

## 6. Plan

**P1. The predicate, as a three-state membership test** in
`.claude/hooks/post-commit-changelog.sh::_flip_magnitude`. A module-level
`_ABSENT` sentinel (PEP 661's documented pattern) replaces the `None` conflation,
and the three states are computed separately so the decision log can name which
one fired:

```python
created_done     = [sid for sid, st in after.items()
                    if st == "done" and before.get(sid, _ABSENT) is _ABSENT]
transitioned_done= [sid for sid, st in after.items()
                    if st == "done" and before.get(sid, _ABSENT) not in (_ABSENT, "done")]
newly_done       = created_done + transitioned_done
```

This fixes the CLASS -- any step created-and-closed in one commit -- and names no
step id. The magnitude rules (major/minor/patch) are untouched.

**P2. The decision log** (criterion 4). Every invocation **that reaches the
detector** appends one structured
line to `handoff/logs/changelog-decisions.log` naming the sha, the chosen bump,
and the REASON, from a closed set: `subject_forced_major`, `flip_created`,
`flip_transitioned`, `no_flip` (a genuinely-chore commit), `masterplan_unreadable`,
`first_commit`, `detector_error`. An unexplained `none` becomes impossible
**within the detector**. *(Scope corrected in phase-86.97: this plan originally
said "Every invocation ... An unexplained `none` becomes impossible", which
overstates the reach. Three bash `exit 0` paths run before the heredoc and never
reach any Python in the hook; they are enumerated and classified in
`scripts/qa/verify_decision_log_86_97.py`. The immutable criterion text quoted
verbatim at line 105 of this contract is deliberately NOT edited.)* The
stderr `[changelog] flip-detect FAILED` marker is KEPT -- it is additive, and I3
shows it is not sufficient on its own.

**P3. `replay_changelog_rule_86_68.py`** gains the third arm (`OLD` /
`SHIPPED` / `FIXED`) so criterion 3's three numbers come from one execution and
the baseline cannot go stale. Its `:54` mirror of the defect is fixed in the same
edit.

**P4. Regression guard** -- `scripts/qa/verify_changelog_flip_86_91.py`:
- CONTROL GREEN FIRST: a normal transition still bumps, and a chore commit still does not.
- It **drives the REAL shipped `_flip_magnitude`**, extracted by `ast` from the
  heredoc inside the `.sh` and exec'd with `git show` stubbed. A re-implemented
  copy would stay green while production drifted.
- Created-and-closed -> bump. Already-done -> no bump. Absent-in-both -> no bump.
- MUTATION, anchor-uniqueness checked first: restore the `None` exclusion -> RED;
  make every commit bump -> RED (the over-crediting direction).
- FAULT INJECTION (criterion 7): make `git show` raise; assert `_flip_magnitude`
  returns `"none"`, does not propagate, and prints the marker.

**P5.** The live proof for criterion 5 is the flip commit for 86.90/86.91 itself:
the hook is run, and the version header it produces is quoted. **CHANGELOG.md is
never hand-edited.**

---

## 7. Out of scope (named)

- The `MAX_ROWS=20` Recent-Activity trim. Pre-existing, orthogonal, and already
  documented as "a row count is not a census".
- Retro-bumping the versions the shipped rule swallowed since 2026-08-14.
  Rewriting released version history is an operator call, not mine; the count of
  swallowed commits is reported instead.

---

## 8. References

- `handoff/current/research_brief_86.91.md` (run `wf_6f758470-f84`)
- `handoff/archive/phase-86.68/contract.md:106-107` -- this step, predicted
- PEP 661 -- sentinel values; python-patterns.guide -- the sentinel object pattern
- arXiv 2408.01760 (ISSTA '24) -- equivalent mutants 4-39%, control-first is load-bearing
- `CLAUDE.md` "NEVER manually update CHANGELOG.md" -- the phase-86.68 rule this repairs
