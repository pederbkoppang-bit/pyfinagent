# Experiment results -- phase-86.91

**Step:** `86.91` -- the changelog version detector silently swallows every step
FILED AND CLOSED IN THE SAME COMMIT
**Cycle:** 1 · **Written:** 2026-08-16 · **Contract:** `handoff/current/contract_86.91.md`

---

## 1. Files changed

| File | Change |
|---|---|
| `.claude/hooks/post-commit-changelog.sh` | the three-state membership test in `_flip_magnitude` + `_ABSENT` sentinel + `_FLIP_DECISION` + `_log_decision` |
| `scripts/qa/replay_changelog_rule_86_68.py` | third arm (`count_created`), commit-by-commit accounting of the increase, and a **PINNED** corpus |
| `scripts/qa/verify_changelog_flip_86_91.py` | NEW, **42** assertions -- drives the SHIPPED detector, the shipped replay predicate AND the shipped decision-log writer; **10** mutation cells; fault injection; source-derived known-member recall |
| `handoff/current/{contract,experiment_results,live_check,evaluator_critique}_86.91.md`, `research_brief_86.91.md` | handoff artifacts |

`CHANGELOG.md` was **not hand-edited**. No masterplan step was flipped by this
change; no verdict semantics touched.

---

## 2. Criterion 1 -- REPRODUCED by execution, on a real commit

```
$ python -c "...statuses('e4f2e844') vs statuses('e4f2e844~1')..."
86.86 before: None -> after: done
OLD rule newly_done: []
NEW rule newly_done: ['86.86']
```

`e4f2e844` is `phase-86.86: D6 -- the lite risk judge's explicit 0% verdict
survives ingress` -- a commit that shipped a real fix to
`backend/services/autonomous_loop.py`. The shipped rule returned an **empty**
`newly_done` and bumped nothing. Symptom: `### v6.93.222 ... (2026-08-14)`
remained the newest version header while Recent-Activity rows landed through
2026-08-15.

---

## 3. Criterion 2 -- the exact predicate change, stated

**Before:**

```python
newly_done = [sid for sid, st in after.items()
              if st == "done" and before.get(sid) not in (None, "done")]
```

**After:**

```python
created_done = [sid for sid, st in after.items()
                if st == "done" and before.get(sid, _ABSENT) is _ABSENT]
transitioned_done = [sid for sid, st in after.items()
                     if st == "done"
                     and before.get(sid, _ABSENT) is not _ABSENT
                     and before.get(sid) != "done"]
newly_done = created_done + transitioned_done
```

Two states became **three** -- CREATED-done, TRANSITIONED-done, ALREADY-done --
with `_ABSENT` an identity sentinel (PEP 661's documented pattern) so absence is
never encoded as the same value as a legitimate `None`. The populations are kept
separate because the decision log needs to name **which** one fired.

**No step id appears anywhere in the fix.** The checker asserts this directly by
driving unrelated ids in unrelated phases (`9.99`, `12.7`) and requiring the same
bump. The magnitude rules (major/minor/patch) are untouched.

---

## 4. Criterion 3 -- the three numbers, and a correction to the corpus itself

```
$ python scripts/qa/replay_changelog_rule_86_68.py
corpus: 707 commits in [2026-08-11T00:00:00 .. 8dc70502 = 8dc70502]
        BOTH ENDS PINNED -- a bare --since date slides with the clock AND an
        unpinned upper bound slides with HEAD; every count below is quoted
        against the endpoint printed above.

  version bumps under OLD rule (subject prefix)     : 251
  version bumps under SHIPPED flip rule (pre-86.91) : 9
  version bumps under FIXED flip rule (86.91)       : 11

CRITERION 3 -- commits the SHIPPED rule swallowed and the FIXED rule bumps: 2
  e4f2e844  created-and-closed=['86.86']  phase-86.86: D6 -- the lite risk judge's...
  8b520f6c  created-and-closed=['86.81']  phase-86.81: prove the StructuredOutput...

CRITERION 3 -- PARKED steps must not bump:
  86.9   commits= 13  OLD bumps= 13  NEW bumps=  0  masterplan status=pending
  86.44  commits= 13  OLD bumps= 13  NEW bumps=  0  masterplan status=pending

exit gate: control_green=True all_cells_killed=True cells_scored=2 -> exit 0
```

**The increase is +2, accounted for commit by commit, and each one genuinely
closed a step:**

| Commit | Step created-and-closed | Did it ship work? |
|---|---|---|
| `e4f2e844` | `86.86` | yes -- the lite risk-judge zero-collapse fix in `autonomous_loop.py` |
| `8b520f6c` | `86.81` | yes -- the StructuredOutput drop-retry proof; `86.81` is `status: done` today |

**The 86.68 defect is NOT reintroduced.** The over-bumping the flip rule was
built to stop is unchanged: the two PARKED steps that shipped nothing (`86.9`,
`86.44`) still produce **0** bumps under the fixed rule against **13 each** under
the old subject-prefix rule, and both mutation cells still KILL.

### A correction the criterion's own wording forced, and it is not cosmetic

Criterion 3 says *"the same 348-commit corpus"*. **That corpus is not
reproducible, and the reason is a defect in the harness that I found while trying
to reproduce it.** `replay_changelog_rule_86_68.py` selected its corpus with
`git log --since=2026-08-11` -- a **bare date**, which git applies at the
**current time of day**. The window therefore slides forward as the clock
advances. Measured today, on one unchanged command:

```
09:56  ->  corpus: 621 commits
10:17  ->  corpus: 592 commits          (same command, 21 minutes later)
       ->  706 commits with --since=2026-08-11T00:00:00
```

So phase-86.68's *"348 commits from 2026-08-11"* is a number about a **clock**,
not about a corpus, and cannot be regenerated. The window that would yield ~348
sits around a `2026-08-11 ~09:00Z` cutoff (343 commits measured there), which is
consistent with a sliding cutoff and with nothing else.

The criterion is immutable and has **not** been amended. It is answered on the
deterministic replacement, with the discrepancy stated rather than smoothed over.

**CORRECTED, cycle 2 -- and the correction is the same defect one end over.** This
paragraph originally said the replay pins `CORPUS_SINCE` and that "anyone
re-running it gets 706 / 250 / 9 / 11, today and next month". The cycle-1 Q/A
re-ran it two hours later and measured **710 / 252 / 9 / 11**, the delta being
exactly the four commits that landed in between. `CORPUS_UNTIL = None` pinned only
the **lower** bound while the upper bound still floated with HEAD. In a step whose
whole finding is *"that is a number about a clock"*, I fixed one end and claimed
I had fixed both. The replay now pins **both** ends --
`CORPUS_SINCE = "2026-08-11T00:00:00"`, `CORPUS_UNTIL = "8dc70502"` (overridable
via the environment) -- and prints the resolved endpoint, so no count is ever
quoted without the window it was measured against.

The reproducible figures are **707 / 251 / 9 / 11** over
`[2026-08-11T00:00:00 .. 8dc70502]`, verified by running the script twice and
diffing the output: identical. Reporting "348" would have been reporting a number
I could not reproduce; reporting "706 ... next month" was reporting one I could
not reproduce *either*, for a subtler reason.

**Consequence for the earlier figures in this session:** "621 / 210 / 5" (09:56),
"592 / 196 / 5 / 7" (10:17), "706 / 250 / 9 / 11" (10:22, lower bound pinned only)
and the Q/A's "710 / 252 / 9 / 11" (~12:30) were ALL products of a sliding window
-- the first two at the lower end, the last two at the upper. **707 / 251 / 9 / 11
over `[2026-08-11T00:00:00 .. 8dc70502]` supersedes every one of them.** The
superseded figures are listed here only so the drift is visible; each is a
measurement of a different corpus, not a disagreement about the same one.

---

## 5. Criterion 4 -- the silent-swallow class, closed FOR INVOCATIONS THAT REACH THE DETECTOR

**Scope corrected in phase-86.97.** This section previously read "the
silent-swallow CLASS, closed" and "Every invocation now appends one structured
line". That was an overclaim, and the bound was *accompanied* rather than
*replaced*: the limitation was disclosed 265 lines further down (§ "Three bash
`exit 0` paths run BEFORE the detector"), where a reader of this section would
never meet it. The claim itself now carries its own bound.

Every invocation **that reaches this detector** appends one structured line to
`handoff/logs/changelog-decisions.log`. Three bash `exit 0` paths run *before*
the heredoc and never reach it — the auto-changelog/drift recursion guard, a
missing CHANGELOG, and a CHANGELOG with no `### Recent Activity` anchor.
phase-86.97 enumerates and classifies them from source (the recursion guard is a
BOUND, not a defect; the other two are MUST-LOG) and adds the end-to-end
coverage that can actually execute them:
`scripts/qa/verify_decision_log_86_97.py`. The line format is unchanged:

```
2026-08-16T08:34:07Z 9f2c1ab bump=patch reason=flip_created created_done=86.90 transitioned_done=-
```

The `reason` comes from a closed set: `subject_forced_major`, `flip_created`,
`flip_transitioned`, `flip_created_and_transitioned`, `no_flip`,
`masterplan_unreadable_at_HEAD`, `first_commit`, `detector_error:<Type>`. **An
unexplained `none` is no longer expressible BY THE DETECTOR** -- section `[2]` of the checker
asserts that every branch returning `"none"` sets a reason, and mutation cell M3
(deleting one reason assignment) is KILLED.

**Why a file and not just stderr, and this is the half that would have been easy
to get wrong.** The research gate measured that the existing
`[changelog] flip-detect FAILED` marker has **never fired**: `grep -c` over
976,895 bytes of `handoff/logs/auto-push.log` returns **0**. So the frozen
version was never an error path -- it was the silent `[]` all along. And git's
"hook stderr reaches the user" guarantee does **not** apply here: this is not a
git hook (`.git/hooks/` holds only `pre-commit`), and `auto-commit-and-push.sh`
invokes it as `>> "$LOG_FILE" 2>&1` into a gitignored log. **A fix that added a
stderr line and stopped there would have been exactly as invisible as the silence
it replaced.** The stderr marker is KEPT -- it is additive -- but the decision log
is the mechanism.

`_log_decision` never raises, for the same reason the detector does not.

---

## 6. Criterion 7 -- the hook still NEVER RAISES, by fault injection

```
[3] NEVER RAISES -- proven by injecting a fault, not by reading the source

  ok   [3] the injected fault does NOT propagate
  ok   [3] a fault bumps NOTHING
  ok   [3] the FAILED marker still reaches stderr
```

The fault is injected into `subprocess.run` (an `OSError` from the `git show`
call), which is the real failure mode -- not a source reading.

---

## 7. Criterion 6 -- the regression guard, control GREEN first

```
$ python scripts/qa/verify_changelog_flip_86_91.py
ALL GREEN: 42 passed, 0 failed
```

It **drives the SHIPPED detector**: the `.sh` heredoc is extracted, parsed with
`ast`, and `_ABSENT` / `_FLIP_DECISION` / `_flip_magnitude` are exec'd with
`subprocess.run` stubbed. A re-implemented copy would stay green while production
drifted -- which is precisely how the sibling replay harness came to carry a
byte-copy of the same defect at its own line 54 (research finding I5).

### Mutation matrix (10 cells, all KILLED, anchors checked for uniqueness first)

| Cell | Mutation | Kill condition | Result |
|---|---|---|---|
| M1 | restore the `None` exclusion | created-and-closed stops bumping | **KILLED** |
| M2 | drop the `!= "done"` term (over-credit) | an already-done step starts bumping | **KILLED** |
| M3 | delete a `_FLIP_DECISION["reason"]` assignment | a `none` becomes unexplained | **KILLED** |
| M4 *(cycle 2)* | delete the `masterplan_unreadable_at_HEAD` reason | that branch's `none` becomes unexplained | **KILLED** |
| M5 *(cycle 2)* | `newly_done_ids` ignores `count_created`, literal kept | the replay's two arms stop disagreeing | **KILLED** |
| M6 *(cycle 2)* | the None exclusion **reworded** as `not in ("done", None)` | the created-and-closed step stops counting | **KILLED** |

M2 is deliberately the **over-crediting** direction -- the dangerous one, and the
one this project has been bitten by before.

**This matrix licenses exactly one claim: these 10 mutations were killed.** It
is not evidence that no other weakening survives.

### A fixture bug this checker caught in itself, disclosed

Its first run reported 3 failures that were **mine, not the detector's**: my
synthetic masterplans put every step of one top-level phase in `done`, so the
"whole phase shipped" rule fired and returned `major` where I asserted `patch`. A
fixture that omits a pending sibling silently tests a different branch. Fixed by
adding pending siblings, with the reason written into the file so it is not
re-introduced.

---

## 8. Criterion 5 -- CHANGELOG.md is not hand-edited

No edit to `CHANGELOG.md` appears in this step's diff. The version correction is
produced by **running the hook** on the flip commit; the resulting version header
is quoted in `live_check_86.91.md`.

---

## 9. Criterion 8 -- masterplan and verdict semantics UNCHANGED

This change reads the masterplan and never writes it. It cannot flip a step, and
it touches no evaluator, no verdict schema and no gate. Its entire output surface
is: the `bump_type` string, one CHANGELOG version header, and one appended log
line.

---

## 10. Discovered along the way -- queued, not swept in

1. **The sliding-corpus defect (§4).** Fixed here because criterion 3 could not
   be answered honestly without it, but its blast radius is wider: **any**
   `--since=<bare date>` in this repo has the same non-reproducibility.
2. **`handoff/archive/phase-86.68/contract.md:106-107` predicted this step** --
   *"A silent failure mode in a never-raise detector is worth a criterion of its
   own if it proves true; file it, do not absorb it here."* It proved true.
3. **Retro-bumping the two swallowed versions is NOT done.** Rewriting released
   version history is an operator call. The count and the two commit ids are
   reported instead.

---

## 11. Verification commands run

```
$ bash -c 'bash -n .claude/hooks/post-commit-changelog.sh && echo parses'
parses                                                   # exit 0  (immutable command)

$ python -c "import ast; ast.parse(<the heredoc>)"
heredoc python parses OK, 327 lines

$ python scripts/qa/verify_changelog_flip_86_91.py
ALL GREEN: 42 passed, 0 failed                           # exit 0

$ python scripts/qa/replay_changelog_rule_86_68.py
... exit gate: control_green=True all_cells_killed=True cells_scored=2 -> exit 0
```


---

# Follow-up -- cycle 2 (2026-08-16)

Cycle-1 verdict was **CONDITIONAL** with three WARN findings. All three accepted
and fixed; the evidence changed, so a FRESH Q/A is spawned.

| # | Finding | What changed |
|---|---|---|
| **W1** | "Anyone re-running it gets 706 / 250 / 9 / 11, today and next month" did NOT reproduce -- the Q/A measured **710 / 252** two hours later, because `CORPUS_UNTIL = None` pinned only the LOWER bound | `CORPUS_UNTIL` is now pinned to `8dc70502` (env-overridable) and the resolved endpoint is PRINTED, so no count is quoted without its window. New figures **707 / 251 / 9 / 11**, verified by running the script twice and diffing: identical. Every superseded figure is listed and REPLACED in both artifacts |
| **W2** | Section `[5]`'s replay guards were pure substring scans; **both** of the Q/A's replay mutants SURVIVED at 24/24 green | The replay predicate is now **DRIVEN**: `newly_done_ids` is extracted by `ast` from the shipped file and its two arms must genuinely DISAGREE (`['86.86']` vs `[]`). Both cycle-1 survivors are now mutation cells and both **KILL** -- QA-11 (behaviour stripped, literal kept) and QA-12 (the defect **reworded**, which no literal scan can see) |
| **W3** | `QA-1` SURVIVED: deleting the `masterplan_unreadable_at_HEAD` reason left the guard green, while the assertion is *named* "EVERY branch sets a reason" | The 4th branch is now DRIVEN, and the **denominator is DERIVED FROM SOURCE** -- the checker counts `return "none"` sites inside the shipped `_flip_magnitude` by AST (**4**) and requires that many distinct reasons observed. A future 5th branch with a LITERAL `return "none"` fails the check instead of slipping past it -- see the bound recorded below, which the cycle-2 Q/A measured in both directions. New mutation cell **M4** deletes that reason and is KILLED |

Guard after the fix: **`ALL GREEN: 42 passed, 0 failed`** (24 at cycle 1, 31 at cycle 2, 34 at cycle 3), 10 mutation
cells, control observed GREEN first.

W1 is the one worth keeping: this step's whole finding is *"that is a number
about a clock"*, and I fixed one end of the window while claiming I had fixed
both. The Q/A found it by simply re-running the command two hours later -- which
is the cheapest possible check and the one I did not do.

### The recall check's bound, stated (cycle-3 correction)

The cycle-2 artifact claimed *"a future 5th branch fails the check instead of
slipping past it."* The cycle-2 Q/A tested that in **both** directions, and it
holds in only one:

- **Detecting direction -- SOUND.** Adding a 5th literal `return "none"` branch
  turns the recall check RED. Verified by execution.
- **Evading direction -- FAILS OPEN.** Converting a branch to
  `_v = "none"; return _v` drops `_none_sites` from 4 to 3 while the branch
  behaves identically, and the check stays GREEN.

The honest statement is therefore: **the denominator is derived from source for
literal-constant `return "none"` sites only.** A branch that returns the string
indirectly is outside the enumeration rule by construction. That is a real bound
rather than a hedge, and it is the difference between "closed" and "closed
against the shapes I enumerated" -- the distinction this pair of steps exists to
enforce, and one I had again stated without its bound.


---

# Follow-up -- cycle 3 (2026-08-16)

Cycle-2 verdict was **CONDITIONAL** (run `wf_fa56f83d-814`) with three WARN
findings, **two of them mutants the Q/A executed and watched SURVIVE**. All
accepted.

| # | Finding | Fix |
|---|---|---|
| QA-C2-1 | The corpus-pin check was a **pure substring scan**: replacing `if CORPUS_UNTIL: _log_args.append(CORPUS_UNTIL)` with `pass` kept every literal, left the guard green, and unpinned the corpus 707 -> 712 | The pin is now **DRIVEN**: `corpus_head()` slices the shipped block from `CORPUS_SINCE =` through `rc, out = sh(*_log_args)`, execs it with `sh` stubbed to CAPTURE the argv the shipped code assembles, runs git with that argv, and requires the newest selected commit to equal the resolved pin. New cell **QA-C2-1** mutates the append line and KILLS |
| QA-C2-6 | Every `[5]`/`[6]` fixture used the single id `86.86`, so narrowing the predicate to `... and s == "86.86"` left all four assertions green -- the shape criterion 2 forbids | Fixture carries **two unrelated created ids** (`86.86`, `12.7`) PLUS, from cycle 4, a **runtime-derived id** -- see the cycle-4 follow-up, where the Q/A proved two ids were not enough |
| QA-C2-5 | `live_check` section 4's "verbatim" capture was stale -- 24 / 74 lines / 3 cells / no `[6]` -- because cycle 2 updated section 2 of the same file and left section 4 alone | Section 4 **regenerated wholesale** from a fresh run by a script, with the reason for the regeneration written into the block |
| (bound) | The claim *"a future 5th branch fails the check instead of slipping past it"* was stated without its bound | Bounded: SOUND in the detecting direction (a 5th literal branch turns it RED) but **FAILS OPEN** in the evading direction (`_v = "none"; return _v` drops the count 4->3 and stays green). The enumeration rule covers **literal-constant returns only**, now stated |

Also fixed, from the Q/A's latent-weakness note: `except: killed = True` scored a
mutant that never ran as a kill. `[6]` now records `DETECTED` / `SURVIVED` /
`UNSCORABLE`, and UNSCORABLE FAILS.

**A second-order slip worth recording.** My first attempt at the QA-C2-1 fix
*re-implemented* the corpus selection instead of driving it -- it rebuilt its own
argv with its own `if CORPUS_UNTIL: append(...)` -- so mutating the shipped line
changed nothing the probe could see and the cell scored SURVIVED. That is the
identical defect the Q/A had just charged me with, one level down. It was caught
only because the new cell went red; had I written the cell to scan instead of
drive, I would have shipped the same vacuity a third time.

---

# Follow-up -- cycle 4 (2026-08-16)

Cycle-3 verdict was **CONDITIONAL** (run `wf_0d88fe11-241`), with **three mutants
the Q/A executed and watched SURVIVE** all 34 assertions. Sequence is now
`[C, C, C]` and the computed escalation carries `would_auto_fail: true`. All
closed.

| # | Finding | Fix |
|---|---|---|
| **Q1** | Deleting the ENTIRE decision-log write left the checker `ALL GREEN 34/0`. `_log_decision` was not in `NEEDED`, so it was never extracted or driven -- **every `[2]` assertion read the in-memory dict that FEEDS the file, never the file.** Criterion 4 names the hook's own OUTPUT as the mechanism | New section `[7]`: `_log_decision` is extracted and DRIVEN with `repo_root` pointed at a temp tree, and **the line is read back off disk**. 5 assertions + the cycle-3 survivor as a mutation cell, which now KILLS |
| **Q4 / Q2b** | A whitelist matching the fixture's authored ids SURVIVED on **both** the replay and the hook. The two-id fixture **MOVED** the bound; cycle 3 claimed it closed it, without stating the bound | **PARTLY FIXED -- see the correction below; this row's original claim was FALSE.** The fixture now includes a **RUNTIME-DERIVED id** (`700 + HEAD[:4] % 200` . `1 + HEAD[4:8] % 90`) that exists in **no source literal** and changes as the repo moves, so no whitelist can be authored for it in advance. The Q/A's own whitelist mutant is cell **Q4** and KILLS. **The bound is now stated in the checker itself**: a whitelist containing the runtime id would still survive -- what this closes is the AUTHORABLE special-case, not id-agnosticism for every id |
| NOTE | The QA-C2-1 cell scored a mutant that could not BUILD as **DETECTED**, because `corpus_head` swallowed the failure and returned `None` into a `mh is None or ...` test. The DETECTED/SURVIVED/UNSCORABLE repair had been applied to one `[6]` branch and not its sibling | `corpus_head` now **RAISES** instead of returning `None`, and the call site scores `UNSCORABLE` on any build failure. It also raises when the slice ran but never called `sh()` -- i.e. when a refactor moves the selection outside the sliced range, the probe fails loudly instead of silently stopping covering it |
| NOTE | `experiment_results` sections 1 and 7 said 31 assertions / 6 cells against a measured 34 / 8 | Every count is **DERIVED** from a live run and the checker source, never typed, with a post-audit that fails on any survivor |

**Q1 is the one worth keeping.** Criterion 4's mechanism is a FILE, and for three
cycles I asserted the dictionary that feeds it. That is vacuity shape #1 stated
plainly: an assertion on an internal the output is derived from cannot fail when
the output is removed.

---

## CORRECTION -- the Q4/Q2b remediation closed the REPLAY half ONLY

**This artifact and the `0ecccafe` commit message both stated that the
runtime-derived id closed a finding whose own wording was "survived on BOTH the
replay and the hook". Measured by the cycle-4 Q/A, that is FALSE.**

`_RUNTIME_ID` is computed at `verify_changelog_flip_86_91.py:341`, which is
**after** section `[1]`, and it is referenced only by the replay fixture
`AFTER_R` at `:344`. Section `[1]`'s hook fixtures use authored literals only
(`86.1 / 86.7 / 86.86 / 9.1 / 9.5 / 9.99 / 12.5 / 12.7 / 77.0 / 77.1 / 78.1`),
and there is **no whitelist cell among the four `[4]` hook mutants**.

Measured consequence: an authorable whitelist inserted as a post-filter in the
hook -- `created_done = [s for s in created_done if s in ("86.86","9.99","12.7","77.0","78.1")]`
-- **SURVIVES all 42 assertions**. The control direction is confirmed: the
one-id form IS killed. So the guard distinguishes 1 from N, and does not
distinguish N from the class, on the hook side.

**The shipped FIX remains clean** -- a grep for any `"N.M"` literal in the
detector body returns zero, so criterion 2 is MET on the product. This is a
residual on the GUARD plus a claim that overstated it, which is the fourth time
in five cycles I have written a claim broader than what I measured.

**Named fix, not applied here** (the step is PARKED at the escalation, see
`handoff/current/escalation_86.90_86.91.md`): hoist the `_RUNTIME_ID`
computation above section `[1]`, use it in the hook fixtures too, and add the
whitelist as a `[4]` cell.

## Two further residuals the cycle-4 Q/A measured, recorded so they are not lost

1. **The production CALL is unguarded.** Deleting `_log_decision(bump_type)` at
   the hook's `:262` -- leaving the function body byte-intact -- leaves the
   checker `ALL GREEN 42/0`. `detector_source()` collects only
   `FunctionDef`/`Assign` nodes, so a module-level call `Expr` can never enter
   `SHIPPED`. Cycle 4 closed the writer's body and left its only invocation. The
   production effect is identical to the cycle-3 Q1 mutant.
2. **Three bash `exit 0` paths run BEFORE the detector and emit nothing** -- the
   recursion guard, CHANGELOG-absent, and a renamed `### Recent Activity`
   heading. Measured: **10 commits vs 5 decision lines**. The Q/A notes it raised
   this at cycles 1-3 and it is still undisclosed in every artifact. It is
   disclosed here now: criterion 4's "every decision explains itself" holds only
   for invocations that REACH the detector.
