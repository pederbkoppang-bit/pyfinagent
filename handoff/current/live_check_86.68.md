# Live check — step 86.68


> **TIMESTAMP CORRECTION (2026-08-14 04:35 CEST).** Wall-clock times in this file were
> **narrated, not measured** — I read the clock once at session start and invented a
> progression from it. The real session spans **08-13 23:10 → 08-14 04:26** (~5h), not the
> 16+ hours the original times implied. Times below are now the **git commit timestamps**
> of this artifact, which are ground truth. Durations and orderings derived from the old
> figures should be disregarded; the measurements themselves are unaffected.
**Date:** 2026-08-14 ~03:27 CEST (git)
**Required shape:** *"before/after bump counts for the replayed 86.9 and 86.44 sequences"*
**Harness:** `scripts/qa/replay_changelog_rule_86_68.py` (read-only; `git show` + parse)

---

## The required evidence, verbatim

```
$ python scripts/qa/replay_changelog_rule_86_68.py

corpus: 482 commits since 2026-08-11 (re-derived at execution time)
RULE STATED: OLD = subject-only (phase-X.Y -> patch). NEW = subject may force
             MAJOR only; otherwise the parsed masterplan id->status diff decides.

  version bumps under OLD rule : 186
  version bumps under NEW rule : 8

CRITERION 3 -- PARKED steps must not bump:
  86.9   commits= 13  OLD bumps= 13  NEW bumps=  0  masterplan status=pending
  86.44  commits= 13  OLD bumps= 13  NEW bumps=  0  masterplan status=pending

CRITERION 6 -- MUTATION (flip gate removed):
  86.9   CONTROL=0 (GREEN)  MUTANT=13  -> KILLED
  86.44  CONTROL=0 (GREEN)  MUTANT=13  -> KILLED

REAL exit=0
```

`REAL exit` is captured directly, **not through a pipe** — `$?` after a pipe reports the
last command in the pipeline, which produced a false green earlier in this session.

## The immutable command

```
$ bash -c 'test -f .claude/hooks/post-commit-changelog.sh && bash -n .claude/hooks/post-commit-changelog.sh && echo classifier-parses'
classifier-parses
exit=0
```

It proves the classifier **parses**. It cannot observe behaviour — the replay above is what
carries criteria 1, 3, 4 and 6.

## The separation — CORRECTED after the cycle-1 Q/A

**The cycle-1 version of this section was confounded and is withdrawn.** It read
*"Recent-Activity rows dated 2026-08-14: 20 ← all still written"*. **20 is `MAX_ROWS=20`**
(`post-commit-changelog.sh:17`), the trim cap — a count identical to the cap cannot show
coverage. Re-derived:

```
POPULATION RULE: row-eligible unless the subject matches
  ^chore: (auto-changelog|changelog drift)   [the skip at post-commit-changelog.sh:27]

commits dated 2026-08-14                  : 86
skipped as chore                          : 43
ROW-ELIGIBLE                              : 43
rows surviving in the table               : 20   <- exactly MAX_ROWS=20
eligible commits TRIMMED                  : 23   <- got a row, then aged out

of the 43 eligible, commits that BUMPED   :  0
surviving rows whose commit did NOT bump  : 20 of 20
version at session start (34e5d0c6)       : v6.93.221
version now                               : v6.93.221   <- UNCHANGED
```

**Rows exist exactly where bumps do not**, and that conclusion does not depend on the
trimmed 23. Structurally: the row-insert (`:252-270`) is unconditional; the version header
(`:212`) and bullet (`:228`) are gated on `bump_type`.

## Provenance of the last bump

`v6.93.221` traces to **86.58**, whose masterplan status is **`done`** — a real flip, not a
subject claim. That is the rule working in the intended direction.

## Divergences from the step's own audit_basis — reported, not adopted

| figure | audit_basis | re-derived here |
|---|---:|---:|
| corpus | 348 | **482** |
| bumps, old rule | 136 | **186** |
| bumps, new rule | 7 | **8** |
| 86.9 + 86.44 bumps removed | 19 | **26** |

The corpus grew by 134 commits; the proportion losing a bump is stable (**73.1%** vs 73.4%).

## What this artifact does NOT license

- **It does not close the step.** A Q/A graded cycle 1 **CONDITIONAL** (`wf_aebf89bf-bfd`);
  the two blockers it named are fixed above and in the replay harness, and a **fresh Q/A
  must grade the changed evidence**.
- The counts are **tree-dependent** and will move; the durable claim is the rule.
- **A wording gap in `CLAUDE.md`** is disclosed in `experiment_results_86.68.md` §C5:
  it says detection is "from the masterplan diff", which reads as a *text* diff — the exact
  approach the implementation tested and rejected. Doc precision, queued not fixed.
