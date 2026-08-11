# live_check -- step 86.36

**Captured**: 2026-08-11, tree `d375be9e`, `scripts/qa/qa_wip.py`
sha256[:16] = `da6db96dddb9b9fc`.
**Every block below is a fresh run at this tree**, not a paste from development.

**Concurrency disclosure**: peer session `pyfinagent-51` has Q/A runs writing to
`.claude/agent-memory/qa/verdicts/` during this capture. Every retention
experiment therefore ran in a SCRATCH sink; the only thing touched in the real
corpus is the `audit_memory.py` leg (section C), which is read-only and asserts
NO CHANGE.

---

## A. Criterion 1 -- the destruction, DRIVEN

```
$ python scripts/qa/reproduce_wip_destruction_86_36.py
scratch sink: /var/folders/n4/9khkbgzj593cmjc28m9chntm0000gn/T/pyfin_86_36_repro_g6woe14i
(the REAL verdicts/ dir is untouched -- live Q/As are writing to it)

SPAWN 1 wrote  : verdict_wip_86.36.md
  bytes        : 4386
  status       : COMPLETE  recoverable=True

SPAWN 2 wrote  : verdict_wip_86.36.md
  bytes        : 124
  status       : INCOMPLETE

--- RESULT ---
  same path for both spawns : True   (verdict_wip_86.36.md)
  bytes before / after      : 4386 -> 124   (LOST 4262)
  spawn 1's analysis still recoverable : False

OK -- destruction reproduced: one path, second write erases the first's analysis. This is the pre-fix contract behaving exactly as 86.31 specified it, which is why the fix is a NAME change, not a timing one.
```

**Driven, not simulated-from-memory** -- criterion 1 asks which was used. Two
REAL instances corroborate it and neither was manufactured:

| instance | before | after | source |
|---|---|---|---|
| 86.31 cycle 4 | 6,239 B `INCOMPLETE` | stub | my own cycle-5 spawn destroyed it |
| **86.34** | 4,921 B (06:27:15Z) | **796 B** (06:40:32Z) | a **peer session's** retry, watched live by the researcher between two of its own tool calls |

The second matters most: it happened in another session, so the defect is not an
artifact of one operator's habits.

## B. Criteria 2, 3, 4 -- coexistence, resolution, non-verdict-ness

```
$ python scripts/qa/verify_wip_retention_86_36.py

[2] TWO CYCLES' RECORDS COEXIST
  ok   both records exist on disk  -- verdict_wip_86.36__20260811T060000Z.md=965B  verdict_wip_86.36__20260811T064000Z.md=110B
  ok   the paths are DISTINCT
  ok   cycle 1 survived cycle 2's first write  -- 965 bytes retained (pre-fix this went to ~124)
  ok   list_wip_records sees BOTH, newest first  -- ['verdict_wip_86.36__20260811T064000Z.md', 'verdict_wip_86.36__20260811T060000Z.md']
  ok   each carries its OWN WRITTEN stamp
  ok   cycle 1 carries COMPLETED, cycle 2 does not

[3] RESOLUTION FOR A GIVEN CYCLE, AND STALENESS STILL FIRES
  ok   spawn 2 resolves to cycle 2's record  -- status=INCOMPLETE
  ok   and lists cycle 1 as a PRIOR, not merged
  ok   spawn 1 resolves to cycle 1's record  -- status=COMPLETE
  ok   a spawn NEWER than every record reports STALE  -- got STALE
  ok   an unparseable spawned_at reports IDENTITY_UNKNOWN  -- got IDENTITY_UNKNOWN

[4] NO RECORD IS EVER READABLE AS A VERDICT
  ok   report(spawned_at='2026-08-11T06:00:00Z') has NO 'verdict' key
  ok   report(spawned_at='2026-08-11T06:00:00Z') is_verdict is False
  ok   report(spawned_at='2026-08-11T06:40:00Z') has NO 'verdict' key
  ok   report(spawned_at='2026-08-11T06:40:00Z') is_verdict is False
  ok   report(spawned_at=None) has NO 'verdict' key
  ok   report(spawned_at=None) is_verdict is False
  ok   criterion-4 cardinality floor: >=3 reports examined  -- examined=3

[2b] RETENTION IS BOUNDED
  ok   prune keeps exactly `keep` records  -- 8 -> 3, removed 5
  ok   prune removed the OLDEST, kept the newest
  ok   keep=0 is REFUSED (retaining zero is the defect, not a setting)

[2c] audit_memory.py OUTPUT UNCHANGED (real corpus, read-only)
  ok   a stamped record does NOT change the auditor's exit code  -- 1 vs 1
  ok   a stamped record does NOT change the auditor's output  -- identical

ALL GREEN -- 23 passed, 0 failed
```

## C. Criterion 2's auditor leg -- the ONLY thing run against the real corpus

Included in section B above: a stamped record is added under `verdicts/`, the
auditor is run before and after, and the output is **byte-identical** (exit 1
both sides -- the auditor is red for pre-existing reasons unrelated to this
step, which is exactly why "unchanged" is the assertion and not "green").
The probe file is removed in a `finally`.

## D. Criterion 6 -- mutation

```
$ python scripts/qa/mutation_matrix_86_36.py
subject : scripts/qa/qa_wip.py  sha256[:16]=da6db96dddb9b9fc

[CONTROL] unmutated checker -> exit 0
  ok -- green control established

  M1-REVERT-RUN-STAMP        KILLED      revert the path to fixed-per-step -- both cycles collide again  [named: the paths are DISTINCT]
  M2-RETAIN-ONLY-ONE         KILLED      cap retention at ONE -- the bounded-retention assertion must fire  [named: prune keeps exactly `keep` records]
  M3-LIST-ONLY-NEWEST        KILLED      make list_wip_records report only the newest -- coexistence must fail  [named: list_wip_records sees BOTH, newest first]
  M4-NEWEST-FIRST-SELECTION  KILLED      reintroduce the newest-first selection bug this step already hit once  [named: spawn 1 resolves to cycle 1's record]
  M5-LEAK-A-VERDICT-KEY      KILLED      leak a scrapeable verdict key -- criterion 4 must fire  [named: has NO 'verdict' key]

tracked subject UNCHANGED: True  (da6db96dddb9b9fc -> da6db96dddb9b9fc)

OK -- all 5 cells KILLED on a named assertion
```

**THE MATCHER WAS WRONG TWICE BEFORE THIS OUTPUT WAS TRUSTWORTHY**, and both
errors are mine:

1. `name in output` accepted the name anywhere -- and the checker prints every
   assertion name **including the passing ones** -- so any red at all scored as
   a named kill. **M5 was a FALSE KILL** under that rule and `MIS-ATTRIB` was
   unreachable.
2. `"FAIL " + name` required the name to immediately follow the verdict word,
   but these names are **suffixes** of their line, so a genuine kill was
   mis-scored `MIS-ATTRIB`.

Now: the name must appear on a line beginning `FAIL`/`FAILED:`. **Proven
non-vacuous by execution** -- a probe cell whose expected name never appears
reports `MIS-ATTRIB` rather than `KILLED`:

```
  M5-LEAK-A-VERDICT-KEY      MIS-ATTRIB  went red but not on a NAMED assertion from this cell
FAIL -- 1 cell(s) did not kill: ['M5-LEAK-A-VERDICT-KEY']
```

That probe was itself broken on its first attempt: copied to `/tmp`, its
`REPO = __file__.parents[2]` resolved outside the repo, it bailed at the control,
and I briefly read that as evidence the discriminator was still vacuous. Re-run
from inside `scripts/qa/`, it behaves as above. **Third probe error in this
step.**

## E. Criteria 3 and 5 -- no regression in phase-86.31

```
$ bash -c 'source .venv/bin/activate && python scripts/qa/verify_qa_write_first_86_31.py'
is mandatory for it). Covering control: Main's
     post-verdict `git status` rule. Queued as its own masterplan step.
  R3 Section [6] is still a TEXT SCAN. It now kills the reword-inversion class the
     cycle-1 Q/A demonstrated, but no scan is proof against every rewrite. The only
     non-circular evidence that the directive reaches the agent is section [9].

ALL GREEN -- 195 passed, 0 failed
```

```
$ git diff --stat -- .claude/hooks/qa-write-guard.sh
(no output -- the guard is byte-unchanged)
```

**Note the immutable command is 86.31's checker.** It proves the write/deny
separation did not regress and says **NOTHING** about retention; criteria 2 and 6
carry that evidence. Stated so a green command is not mistaken for a green step.

## F. Backward compatibility, checked against the LIVE sink

The peer's in-flight records still use the legacy un-stamped name. They must keep
resolving, or this change breaks their recovery today:

```
$ python scripts/qa/qa_wip.py 86.25   ->  status=COMPLETE bytes=8543  retained=1 is_verdict=False
$ python scripts/qa/qa_wip.py 86.34   ->  status=COMPLETE bytes=10346 retained=1 is_verdict=False
$ python scripts/qa/qa_wip.py 86.29   ->  status=COMPLETE bytes=11479 retained=1 is_verdict=False
```

## G. What this capture does NOT establish

- **No Q/A has run. No verdict is claimed.**
- **The stamped path has never been written by a REAL Q/A.** `qa.md` now directs
  it and the Workflow rail reads `qa.md` from disk at runtime, so it is live for
  the next spawn -- but every record on disk today is legacy-named. Until a real
  run lands, section B is simulation plus the live backward-compat check in F.
- **`prune_wip_records()` has no production caller.** Tested and available;
  retention is therefore **unbounded in practice right now**. Stated rather than
  implied, because "bounded retention" in section B is a property of the function,
  not of the live system.
- **A test-only seam exists** (`PYFIN_QA_WIP_OVERRIDE`), defaulting to the real
  module.
