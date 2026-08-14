# Live check — step 86.68

**Date:** 2026-08-14 ~06:25 CEST
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

## The separation, live on this session

```
version at session start (34e5d0c6) : v6.93.221
version now                          : v6.93.221     <- UNCHANGED across 20 commits
Recent-Activity rows dated 2026-08-14: 20            <- all still written
substantive commits checked in CHANGELOG.md          : 10 of 10 present
```

Eight of those ten are `phase-86.x:` commits that produced **zero** version bumps and still
appear. Under the retired rule each would have bumped the patch number.

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

- **It does not close the step.** No Q/A has graded it.
- The counts are **tree-dependent** and will move; the durable claim is the rule.
- **A wording gap in `CLAUDE.md`** is disclosed in `experiment_results_86.68.md` §C5:
  it says detection is "from the masterplan diff", which reads as a *text* diff — the exact
  approach the implementation tested and rejected. Doc precision, queued not fixed.
