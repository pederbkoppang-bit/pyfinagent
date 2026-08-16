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
| `scripts/qa/verify_changelog_flip_86_91.py` | NEW, 24 assertions -- drives the SHIPPED detector, 3 mutation cells, fault injection |
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
corpus: 706 commits since 2026-08-11T00:00:00 (PINNED timestamp -- deterministic;
        a bare date slides with the clock)

  version bumps under OLD rule (subject prefix)     : 250
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
deterministic replacement, with the discrepancy stated rather than smoothed over:
the replay now pins `CORPUS_SINCE = "2026-08-11T00:00:00"`. Anyone re-running it
gets 706 / 250 / 9 / 11, today and next month. Reporting "348" would have been
reporting a number I could not reproduce.

**Consequence for the earlier figures in this session:** the "621 commits, OLD
210, NEW 5" I quoted at 09:56 and the "592 / 196 / 5 / 7" at 10:17 were both
products of the sliding window. The pinned numbers **supersede** them; the
superseded ones are recorded here only so the drift is visible.

---

## 5. Criterion 4 -- the silent-swallow CLASS, closed

Every invocation now appends one structured line to
`handoff/logs/changelog-decisions.log`:

```
2026-08-16T08:34:07Z 9f2c1ab bump=patch reason=flip_created created_done=86.90 transitioned_done=-
```

The `reason` comes from a closed set: `subject_forced_major`, `flip_created`,
`flip_transitioned`, `flip_created_and_transitioned`, `no_flip`,
`masterplan_unreadable_at_HEAD`, `first_commit`, `detector_error:<Type>`. **An
unexplained `none` is no longer expressible** -- section `[2]` of the checker
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
ALL GREEN: 24 passed, 0 failed
```

It **drives the SHIPPED detector**: the `.sh` heredoc is extracted, parsed with
`ast`, and `_ABSENT` / `_FLIP_DECISION` / `_flip_magnitude` are exec'd with
`subprocess.run` stubbed. A re-implemented copy would stay green while production
drifted -- which is precisely how the sibling replay harness came to carry a
byte-copy of the same defect at its own line 54 (research finding I5).

### Mutation matrix (3 cells, all KILLED, anchors checked for uniqueness first)

| Cell | Mutation | Kill condition | Result |
|---|---|---|---|
| M1 | restore the `None` exclusion | created-and-closed stops bumping | **KILLED** |
| M2 | drop the `!= "done"` term (over-credit) | an already-done step starts bumping | **KILLED** |
| M3 | delete a `_FLIP_DECISION["reason"]` assignment | a `none` becomes unexplained | **KILLED** |

M2 is deliberately the **over-crediting** direction -- the dangerous one, and the
one this project has been bitten by before.

**This matrix licenses exactly one claim: these three mutations were killed.** It
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
ALL GREEN: 24 passed, 0 failed                           # exit 0

$ python scripts/qa/replay_changelog_rule_86_68.py
... exit gate: control_green=True all_cells_killed=True cells_scored=2 -> exit 0
```
