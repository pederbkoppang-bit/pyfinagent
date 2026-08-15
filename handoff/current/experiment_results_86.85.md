# Experiment results -- step 86.85

**Step:** the verdict ledger is never written for the step being evaluated, so the
3rd-CONDITIONAL auto-FAIL rule has no input and cannot ever fire.
**Date:** 2026-08-15. **Verification command:** GREEN, `parses`, exit 0.

---

## 0. Headline

**The writer existed nowhere; it exists now, and the rule it feeds is proven to fire
by execution.** With the ledger populated, `enforceEscalation` -- the shipped
consumer, run unmodified -- computes `consecutive_conditionals = 2`,
`would_auto_fail = true` from 86.74's priors **unaided**. That is the auto-FAIL Main
had to compute BY HAND earlier today because the sequence arrived as prose.

**The step was localised before anything was built, and it survived the re-scope
test the criteria demand.**

---

## 1. What was changed

| File | Change |
|---|---|
| `scripts/qa/verdict_ledger_write.py` | **new** -- the writer. Append-one-row, dedup-refusing, fail-loud, append-only. Carries `--emit-sequence` and `--self-test`. |
| `scripts/qa/mutation_matrix_86_85.py` | **new** -- 5-cell mutation matrix, zero repo writes |
| `handoff/verdict_ledger.jsonl` | 35 -> 43 rows; 86.74 backfilled from 0 -> 8 labelled rows |

**No production/trading code was touched.** No flag promoted, no `.env` written, no
manual cycle, no restart.

---

## 2. Criterion-by-criterion

### C1 -- failure LOCALISED before anything was built ✅

Commands and output are in `contract_86.85.md` §2 and reproduced in
`live_check_86.85.md` §1. Result: **CAUSE = NEVER-WRITTEN.**

- **not wrong-key** -- the positive control `--step 86.21` returns `status=ok` with
  5 verdicts through the *same reader* and the *same key*.
- **not pruned** -- the file is append-only and its max date was 2026-08-11,
  predating every 86.74 verdict.
- **not only-after-close** -- 86.21 is `status=pending` and has rows.
- all 35 pre-existing rows carry `recorded_by=main`; 12 share one microsecond
  `recorded_at`, i.e. bulk backfills, never a per-verdict write.

**The criterion also asks: if 86.74's verdicts ARE on disk somewhere, say so and
re-scope rather than build. They partly are, and the step still stands.** They exist
as **10 WIP artifacts** and as **3 narrative cycle sections** in
`evaluator_critique_86.74.md`. Neither substitutes:

- `qa_wip.py`'s own guidance: a WIP record is *"EVIDENCE FOR THE NEXT SPAWN ONLY --
  never count it as a verdict"*, and `records_retained` is *"a gauge, not a counter"*.
- the critique is prose; no machine reads a sequence from it.

**And a third candidate was eliminated by fresh measurement this session:**
`handoff/harness_log.md` cannot carry a per-step sequence either -- 86.74's cycle 5
was **never given a row at all**, and cycle numbers there are **not unique** (two
independent 193/194/195 runs exist). All three substitutes fail.

### C2 -- population rule stated beside every count, enumeration command quoted ✅

**Population = EVERY non-blank line in `handoff/verdict_ledger.jsonl`.** Command:

```
$ python3 -c "rows=[json.loads(l) for l in open('handoff/verdict_ledger.jsonl') if l.strip()]"
  total rows            : 43   (was 35 before this step)
  rows added by 86.85   : 8
  step_ids present      : 11   (was 10)
  86.74 rows            : 8    (was 0)
  verdict distribution  : {CONDITIONAL 23, FAIL 5, PASS 8, NO_VERDICT 7}
  rows with recorded_at : 29 / 43
```

`rows with recorded_at` is 29/43 rather than 43/43 because **14 historical rows
predate the field**; all 8 new rows carry it. That is stated rather than rounded up.

### C3 -- cross-session persistence DEMONSTRATED across two process invocations ✅

Not asserted -- executed as two separate `python` invocations:

```
$ python scripts/qa/verdict_ledger_write.py --ledger <tmp> --step 77.1 \
      --verdict CONDITIONAL --run-id wf_p1 --cycle 1 --note proc1
{"step_id": "77.1", ..., "verdict": "CONDITIONAL", ...}          exit=0

$ python scripts/qa/verdict_ledger_write.py --ledger <tmp> --emit-sequence --step 77.1
["CONDITIONAL"]                                                   exit=0
```

*(A first attempt at this demonstration reported FAILED. That was a **probe defect**,
not a writer defect: the driver put the command in a shell variable and zsh does not
word-split unquoted parameters, so the write never ran and the read-back correctly
found nothing. Recorded rather than quietly re-run -- a red check that indicts the
probe is exactly as misleading as a green one that cannot fail.)*

### C4 -- the 3rd-CONDITIONAL rule proven to FIRE by DRIVING it ✅

The **shipped** `enforceEscalation` (lines 319-370 of `.claude/workflows/qa-verdict.js`)
was extracted and executed unmodified, fed sequences read back **out of the ledger**:

```
CONTROL  1 prior C + current CONDITIONAL       n=1  would_auto_fail=false
DRIVEN   2 prior C + current CONDITIONAL       n=2  would_auto_fail=true
CONTROL  2 prior C + current PASS              n=2  would_auto_fail=false
CONTROL  2 prior C + current FAIL              n=2  would_auto_fail=false
```

The first row is the anti-vacuity control: the rule does **not** fire early. And on
86.74's real backfilled priors:

```
priors from ledger: ["NO_VERDICT","NO_VERDICT","CONDITIONAL","CONDITIONAL","PASS","CONDITIONAL","CONDITIONAL"]
consecutive_conditionals = 2   would_auto_fail = true
```

**So the auto-FAIL that Main computed by hand today would have fired unaided.**

### C5 -- interaction with 86.79 and 86.45 resolved explicitly in writing ✅

- **86.45** owns whether a `NO_VERDICT` row grades. This step **records** a drop
  faithfully and **changes no counting semantics**. Measured rather than assumed:
  the shipped consumer already skips it (`if (v === 'NO_VERDICT') continue` -- it
  *neither extends nor resets*), so recording a drop **cannot** clear an escalation
  today. 86.85 preserves that; it does not re-decide it.
- **86.79** owns the `records_retained` off-by-one in the **parallel** `qa_wip.py`
  trail. This step writes a **different file** and never reads `records_retained`.
  No overlap, so no blocker.
- **86.71** (cumulative budget) would be the ledger's second consumer; out of scope.
- **86.21** owns the counter's in-flight blindness; out of scope.

### C6 -- a drop must not CLEAR an escalation; a missing row must not read as zero ✅

Both driven, not argued:

```
[C,C,NO_VERDICT] + CONDITIONAL -> n=2   would_auto_fail=true    (a drop does NOT reset)
absent sequence                -> n=null would_auto_fail=null
                                  sequence_status=not_supplied  (unknown, NOT zero)
```

**And the recorder was proven to have run before any zero was treated as evidence**
-- the positive control (`--step 86.21` -> `status=ok`, 5 verdicts) is what licenses
reading 86.74's zero as a measured zero rather than a broken reader.

**A design decision that this criterion forced, stated explicitly.** Cycle 7's row
records `CONDITIONAL` -- **what the rail returned** -- not the `FAIL` Main recorded
as the cycle outcome. Storing the derived FAIL would reset the consecutive run and
**clear the very escalation it represents**. The FAIL is computed by the consumer
from the sequence; it is deliberately not stored. This is why the ledger row and the
`harness_log` row differ, and they differ correctly: the ledger records the **verdict
event**, the log records the **cycle outcome**.

### C7 -- verdict semantics UNCHANGED ✅

- The writer **records**; it never transforms. The vocabulary guard **REJECTS**
  anything outside `{PASS, CONDITIONAL, FAIL, NO_VERDICT}` (exit 3) rather than
  coercing -- coercion is how a non-PASS would silently become something else.
- Demonstrated above: with 2 prior CONDITIONALs, a current **PASS** stays
  `would_auto_fail=false` and a current **FAIL** stays `false`. Nothing turns a
  non-PASS into a PASS.
- `qa.md` is untouched. `verdict_history_86_21.py` is untouched. `qa-verdict.js` is
  untouched. The only new surface is a writer.

### C8 -- mutation-tested, control GREEN first, byte-identical ✅

`scripts/qa/mutation_matrix_86_85.py` -- **5 cells, 5 KILLED, 0 survived,
0 unscorable**, control observed GREEN *before* any cell ran:

| cell | guard reverted | result |
|---|---|---|
| M1 | remove the dedup refusal | KILLED (4 checks red) |
| M2 | remove the verdict vocabulary guard | KILLED |
| M3 | allow an unkeyed row | KILLED |
| M4 | swallow a corrupt ledger line | KILLED |
| M5 | collapse event time into write time | KILLED |

**Zero repo writes** -- every mutant is a temp copy executed in the OS tmpdir, so
there is no restore step to get wrong. `sha256` of the target printed before and
after and compared in-run: `146cf84e...1904` both times, `UNCHANGED: True`.

A cell that exits non-zero **without** `SELF-TEST FAILED` in its output is scored
**UNSCORABLE, not KILLED** -- otherwise a typo'd mutation would score as a kill it
did not earn (cf. `pytest exit 5 scores as a KILL`).

---

## 3. The measurement that changed the design

The brief flagged as **UNPROVEN** whether a `PostToolUse` hook can see the returned
verdict, and required it be measured before being designed around. Measured with a
temporary probe hook in the gitignored `.claude/settings.local.json`, restored
**byte-identical** (sha256 `8f03f194...66` before and after, compared in-run):

| question | answer |
|---|---|
| Does a PostToolUse hook receive `tool_response`? | **YES** |
| Does the matcher fire on `Workflow`? | **YES** (`tool_name: "Workflow"` observed) |
| Does `tool_response` carry the **verdict**? | **NO** |

`tool_response` for a Workflow is the **launch receipt** --
`{runId, scriptPath, status, summary, taskId, taskType, transcriptDir, workflowName}`.
Workflows *always* run in the background, so PostToolUse fires at launch and there is
no foreground mode. **A hook can therefore never author a verdict row.** This is
structural, not a quirk.

**Consequence:** the writer is an explicit call by Main at the seam -- the brief's
own stated fallback (§B). A hook remains viable *later* as a pure **alarm**, since it
does see `runId` at launch, which is exactly the key needed to notice a launched run
that never got a row. **That alarm is out of scope here** (86.85 = the WRITER only)
and is recorded as a follow-up.

---

## 4. What I could NOT verify

1. **The writer is not yet WIRED into the seam.** It exists, is tested, and was used
   to backfill -- but nothing *automatically* calls it when a verdict returns. Today
   Main must remember. The measurement above establishes that a hook cannot close
   that gap by authoring the row; the honest position is that **un-forgettability is
   not yet solved**, only made possible.
2. **The backfilled 86.74 rows are a RECONSTRUCTION**, not contemporaneous records.
   They are labelled as such in every `note` field, with their source named
   (`harness_log` row or `evaluator_critique` section). Two of them
   (`1-drop-a`/`1-drop-b`) split a single `harness_log` line that says "rail dropped
   x2" and have no `run_id` -- they are keyed by cycle label instead. A reconstruction
   is weaker evidence than a live write and is marked so rather than blended in.
3. **`recorded_at` is absent on 14 of 43 rows** because they predate the field.
   Those rows cannot distinguish event time from write time retrospectively.
4. **Only one consumer is proven.** `enforceEscalation` is driven end-to-end;
   `attempt_budget.py` (86.71) is still inert and unwired, so the ledger's second
   intended consumer remains hypothetical.
5. **No end-to-end run has yet used the ledger to supply `args.verdict_sequence`
   on a live spawn.** The mechanism is proven by driving the shipped function with
   ledger-sourced data; it has not yet ridden a real Q/A launch.
