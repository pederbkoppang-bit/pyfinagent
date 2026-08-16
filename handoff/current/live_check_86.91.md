# live_check -- phase-86.91

Evidence artifact for the `verification.live_check` gate. Verbatim command output
only, captured 2026-08-16.

---

## 1. The reproduction: `before=None -> after=done` and an EMPTY `newly_done`

```
$ python - <<'EOF'   # statuses() at e4f2e844 and its parent, then both predicates
86.86 before: None -> after: done
OLD rule newly_done: []
NEW rule newly_done: ['86.86']
EOF
```

`e4f2e844` = `phase-86.86: D6 -- the lite risk judge's explicit 0% verdict
survives ingress`, a commit that shipped a real fix to
`backend/services/autonomous_loop.py`.

The symptom it produced:

```
$ grep -n "^### v" CHANGELOG.md | head -1
33:### v6.93.222 — phase-86.68: the version number counts ATTEMPTS, not shipped work --... (2026-08-14)

$ sed -n '9,12p' CHANGELOG.md
| 2026-08-15 | `7a3e3cf2` | docs(session): day report 2026-08-15b + regenerate the 08-16 goal |
| 2026-08-15 | `9a18150f` | phase-86.85: cycle 4 (C8) -- derived guard coverage; CONDITIONAL, step ESCALATED |
| 2026-08-15 | `e4f2e844` | phase-86.86: D6 -- the lite risk judge's explicit 0% verdict survives ingress |
```

Rows through 2026-08-15; newest version header dated 2026-08-14.

---

## 2. The three bump counts (criterion 3)

```
$ python scripts/qa/replay_changelog_rule_86_68.py
corpus: 707 commits in [2026-08-11T00:00:00 .. 8dc70502 = 8dc70502]
        BOTH ENDS PINNED -- a bare --since date slides with the clock AND an
        unpinned upper bound slides with HEAD; every count below is quoted
        against the endpoint printed above.
RULE STATED: OLD = subject-only (phase-X.Y -> patch). NEW = subject may force
             MAJOR only; otherwise the parsed masterplan id->status diff decides.

  version bumps under OLD rule (subject prefix)     : 251
  version bumps under SHIPPED flip rule (pre-86.91) : 9
  version bumps under FIXED flip rule (86.91)       : 11

CRITERION 3 -- commits the SHIPPED rule swallowed and the FIXED rule bumps: 2
  e4f2e844  created-and-closed=['86.86']  phase-86.86: D6 -- the lite risk judge's explicit 0% verdict sur
  8b520f6c  created-and-closed=['86.81']  phase-86.81: prove the StructuredOutput drop retry actually fire

CRITERION 3 -- PARKED steps must not bump:
  86.9   commits= 13  OLD bumps= 13  NEW bumps=  0  masterplan status=pending
  86.44  commits= 13  OLD bumps= 13  NEW bumps=  0  masterplan status=pending

CRITERION 6 -- MUTATION (flip gate removed):
  86.9   CONTROL=0 (GREEN)  MUTANT=13  -> KILLED
  86.44  CONTROL=0 (GREEN)  MUTANT=13  -> KILLED

exit gate: control_green=True all_cells_killed=True cells_scored=2 -> exit 0
```

Both newly-bumping commits closed a real step: `86.81` is `status: done` today,
and `86.86` is `done` and has since been independently RE-GRADED to PASS on the
86.90-fixed rail.

### The corpus was not reproducible, and that is a finding, not a footnote

```
$ git log --since=2026-08-11 --format=%H | wc -l            # 09:56
     621
$ git log --since=2026-08-11 --format=%H | wc -l            # 10:17, same command
     592
$ git log --since=2026-08-11T00:00:00 --format=%H | wc -l
     706
```

A **bare** `--since` date is applied by git at the **current time of day**, so
the window slides forward as the clock advances. phase-86.68's *"348 commits from
2026-08-11"* is therefore a number about a clock. The cutoff that reproduces it
sits near `2026-08-11 09:00Z` (343 commits measured there) -- consistent with a
sliding window and with nothing else.

The immutable criterion is not amended. It is answered on a corpus **pinned at
BOTH ends** -- `CORPUS_SINCE = "2026-08-11T00:00:00"` and
`CORPUS_UNTIL = "8dc70502"` -- and the drift is reported rather than smoothed away.

**Cycle-2 correction.** Pinning only `CORPUS_SINCE` fixed the lower bound while
the upper still floated with HEAD, so the cycle-1 Q/A re-ran the script two hours
later and measured 710 / 252 / 9 / 11 against the 706 / 250 / 9 / 11 recorded
here. Same defect, other end. The reproducible figures are **707 / 251 / 9 / 11**,
verified by running the script twice and diffing: identical. Every earlier figure
in this session -- "621 / 210 / 5", "592 / 196 / 5 / 7", "706 / 250 / 9 / 11" and
the Q/A's "710 / 252 / 9 / 11" -- is SUPERSEDED, and each was a measurement of a
different corpus rather than a disagreement about one. *(Corrected phase-86.94: that claim was TZ-DEPENDENT when written. `2026-08-11T00:00:00` is a TZ-LOCAL timestamp, so the same both-ends-pinned command measured 707 under Europe/Oslo, UTC and America/New_York but **787** under Asia/Seoul -- an 80-commit spread decided by `$TZ`. `CORPUS_SINCE` is now `2026-08-11T00:00:00Z`; the figures are UNCHANGED on this machine (707 / 251 / 9 / 11, exit gate green) and are now identical in every timezone. Pinning both ends was necessary and was not sufficient.)*

---

## 3. Criterion 4 -- every decision now explains itself

The hook, run by hand immediately after commit `8dc70502`:

```
$ bash .claude/hooks/post-commit-changelog.sh
[main 3b69ddf9] chore: auto-changelog hook entry for 8dc70502
 1 file changed, 1 insertion(+), 1 deletion(-)

$ cat handoff/logs/changelog-decisions.log
2026-08-16T08:23:33Z 8dc70502 bump=none reason=no_flip created_done=- transitioned_done=-
```

`reason=no_flip` -- a genuinely-chore commit, now distinguishable from an error
and from a declined flip. Before this step, that decision produced **no output at
all**, which is why the freeze went unnoticed for two days.

Closed reason set: `subject_forced_major`, `flip_created`, `flip_transitioned`,
`flip_created_and_transitioned`, `no_flip`, `masterplan_unreadable_at_HEAD`,
`first_commit`, `detector_error:<Type>`.

**Why a file rather than stderr** (measured by the research gate):

```
$ grep -c "flip-detect FAILED" handoff/logs/auto-push.log
0                                    # over 976,895 bytes -- the marker has NEVER fired
$ ls .git/hooks/
pre-commit                           # this is NOT a git hook
$ grep -n 'post-commit-changelog' .claude/hooks/auto-commit-and-push.sh
396:  ... >> "$LOG_FILE" 2>&1        # stderr goes into a gitignored log
```

A stderr-only fix would have been as invisible as the silence it replaced.

---

## 4. Criterion 6 + 7 -- the guard, and the never-raises proof

*(REGENERATED from a fresh run each cycle. Never hand-edited.)*

```
$ python scripts/qa/verify_changelog_flip_86_91.py
phase-86.91 -- changelog flip detector, three-state membership test

  (driving the SHIPPED detector, 151 lines extracted from post-commit-changelog.sh)

[0] CONTROL -- behaviour that was already correct must still hold

  ok   [0] a NORMAL transition pending->done still bumps
  ok   [0] and it is recorded as a transition
  ok   [0] an ALREADY-done step does NOT bump
  ok   [0] and the 'none' is explained as no_flip
  ok   [0] a chore commit that moves nothing does NOT bump

[1] THE CLASS -- a step created AND closed in one commit (criteria 1, 2)

  ok   [1] created-and-closed BUMPS
  ok   [1] the created step is NAMED in the decision
  ok   [1] the reason distinguishes created from transitioned
  ok   [1] the rule is about the CLASS, not a hardcoded id
  ok   [1] magnitude: a created X.0 kickoff is minor, not patch
  ok   [1] magnitude: closing the last step of a phase is major

[2] NO UNEXPLAINED 'none' -- the silent-swallow class (criterion 4)

  ok   [2] 'no_flip' is reported as its own reason
  ok   [2] 'first_commit' is reported as its own reason
  ok   [2] 'masterplan_unreadable_at_HEAD' is reported as its own reason
  ok   [2] an internal error is reported as detector_error
  ok   [2] EVERY branch that returns 'none' sets a reason -- none is left unrecorded
  ok   [2] known-member RECALL: all 4 none-returning branches are DRIVEN (denominator derived from source, not hand-listed)

[3] NEVER RAISES -- proven by injecting a fault, not by reading the source

  ok   [3] the injected fault does NOT propagate
  ok   [3] a fault bumps NOTHING
  ok   [3] the FAILED marker still reaches stderr

[4] MUTATION -- each cell must turn a check above RED (criterion 6)

  ok   [4] restore-the-None-exclusion: KILLED (check [1] must go RED when created steps are excluded again)
  ok   [4] over-credit-already-done: KILLED (check [0] must go RED when a no-op commit starts bumping)
  ok   [4] drop-the-unreadable-reason: KILLED (check [2] must go RED when the masterplan-unreadable branch stops explaining itself)
  ok   [4] drop-the-reason: KILLED (check [2] must go RED when a 'none' stops explaining itself)

[5] THE SIBLING REPLAY HARNESS -- guarded BEHAVIOURALLY, not by substring

  ok   [5] the replay predicate is extractable and runnable
  ok   [5] count_created=True COUNTS created-and-closed steps in UNRELATED phases
  ok   [5] and it counts a RUNTIME-DERIVED id (811.38) present in no source literal
  ok   [5] count_created=False reproduces the SHIPPED (defective) result
  ok   [5] the two arms genuinely DISAGREE (not two names for one number)
  ok   [5] the corpus UPPER bound is pinned BEHAVIOURALLY (newest selected commit == the pin)
  ok   [5] the corpus LOWER bound is an explicit timestamp, not a bare date

[6] MUTATION of the REPLAY predicate -- the cycle-1 survivors

  ok   [6] QA-11 ignore-count_created (literal kept, behaviour stripped): KILLED (a scan for the word 'count_created' cannot see this; the drive can)
  ok   [6] QA-12 reworded None exclusion: KILLED (the defect reworded is invisible to a literal scan)
  ok   [6] Q4 whitelist matching the fixture's authored ids: KILLED (an N-id whitelist defeats an N-id fixture; a runtime-derived id defeats the whitelist)
  ok   [6] QA-C2-6 special-case a single step id (the shape criterion 2 forbids): KILLED (a single-id fixture cannot tell the CLASS from the instance)
  ok   [6] QA-C2-1 unpin the upper bound (literals all kept): KILLED (a substring scan cannot see this; the behavioural pin check can)

[7] THE DECISION LOG -- the hook's own OUTPUT, read back from disk

  ok   [7] a decision WRITES a line to changelog-decisions.log
  ok   [7] the line carries the bump
  ok   [7] the line carries the REASON, which is the whole point of criterion 4
  ok   [7] the line names the created step
  ok   [7] a 'none' decision is ALSO written -- an unexplained none is the defect
  ok   [7] delete-the-decision-log-write: KILLED (this mutant SURVIVED all 34 assertions in cycle 3)

ALL GREEN: 42 passed, 0 failed
```

## 5. Criterion 5 -- CHANGELOG.md not hand-edited

```
$ git log --oneline -3 -- CHANGELOG.md
3b69ddf9 chore: auto-changelog hook entry for 8dc70502
5d791aa3 chore: auto-changelog hook entry for a21a5889
75831f4c chore: auto-changelog hook entry for c627a810
```

Every CHANGELOG.md change in this session was produced by running the hook. No
step in this session's diff edits it by hand.

**The version-bump proof is the flip commit itself** -- see §6, appended after
the Q/A verdict.

---

## 6. The live bump (appended after the flip)

*Pending: filled in from the decision log and the version header produced by the
hook on the commit that flips 86.90/86.91 to `done`.*

**Stated precisely, because the tempting claim here is false.** That flip commit
will read `reason=flip_transitioned`, **not** `flip_created`: `86.90` was filed
on 2026-08-15 and `86.91` in commit `c627a810` earlier today, so both already
existed at `HEAD~1`. It is therefore an end-to-end proof that the hook still
bumps on a normal transition -- the CONTROL direction -- and **not** a live
demonstration of the created-and-closed class.

The created-and-closed class is demonstrated in the two places where it actually
can be: section `[1]` of the checker, driving the SHIPPED detector, and the
historical replay on `e4f2e844` / `8b520f6c` in §2. A future file-it-and-fix-it
commit will produce the first live `reason=flip_created` line.
