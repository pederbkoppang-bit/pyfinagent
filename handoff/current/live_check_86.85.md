# live_check -- step 86.85

Required by the masterplan `live_check` field: *"the localisation evidence, the
cross-process read-back, and the driven 3rd-CONDITIONAL auto-FAIL."* All three
below, verbatim, with the command above each output.

---

## 1. LOCALISATION -- cause is NEVER-WRITTEN

```
$ wc -c handoff/verdict_ledger.jsonl
10814 bytes, 35 rows           [state BEFORE this step]

$ python3 -c "rows=[json.loads(l) for l in open('handoff/verdict_ledger.jsonl') if l.strip()]; ..."
total rows: 35
recorded_by: {'main': 35}
verdict    : {'CONDITIONAL': 18, 'FAIL': 5, 'PASS': 7, 'NO_VERDICT': 5}
distinct step_ids: 10
86.74 rows : 0
max date   : 2026-08-11

$ python scripts/qa/verdict_history_86_21.py --step 86.74 --evidence-only
step            : 86.74
source          : handoff/verdict_ledger.jsonl
status          : no_rows_for_step
detail          : no rows recorded for step 86.74 in this ledger. That is NOT the same
                  as knowing it has no verdicts -- nothing writes this ledger
                  automatically yet, so absence here is weak evidence.
verdicts        : (none)
```

**POSITIVE CONTROL -- this is what licenses reading the zero above as a MEASURED
zero rather than a broken reader:**

```
$ python scripts/qa/verdict_history_86_21.py --step 86.21 --evidence-only
step            : 86.21
status          : ok
detail          : 5 verdict(s) from the ledger
verdicts        : CONDITIONAL -> CONDITIONAL -> FAIL -> CONDITIONAL -> CONDITIONAL
```

Same reader, same key, same file -- rows come back for 86.21 and not for 86.74.
So: **never written.** Not a wrong key. Not pruned (append-only, max date
2026-08-11, before 86.74's verdicts existed). Not only-after-close (86.21 is
`pending` and has rows).

---

## 2. CROSS-PROCESS READ-BACK -- two separate invocations

```
$ python scripts/qa/verdict_ledger_write.py --ledger <tmp> --step 77.1 \
      --verdict CONDITIONAL --run-id wf_p1 --cycle 1 --note proc1
{"step_id": "77.1", "cycle": "1", "verdict": "CONDITIONAL", "run_id": "wf_p1",
 "recorded_by": "main", "date": "2026-08-15",
 "recorded_at": "2026-08-15T13:39:59.363969+00:00", "note": "proc1"}
exit=0  file now has 1 row(s)

$ python scripts/qa/verdict_ledger_write.py --ledger <tmp> --emit-sequence --step 77.1
["CONDITIONAL"]
exit=0
```

Two distinct `python` process invocations: the first writes, the second reads it
back. The Layer-3 per-step loop runs across sessions, which is why this had to be
process-crossing rather than in-memory.

**A probe defect, recorded rather than quietly re-run.** The first attempt printed
`cross-process read-back FAILED`. The cause was the driver, not the writer: it put
the command in a shell variable and **zsh does not word-split unquoted parameters**,
so the write never executed and the read-back correctly found nothing. A red check
that indicts its own probe is exactly as misleading as a green one that cannot fail.

---

## 3. THE DRIVEN 3rd-CONDITIONAL AUTO-FAIL

The **shipped** `enforceEscalation` (`.claude/workflows/qa-verdict.js`, lines
319-370) extracted and executed **unmodified**, fed sequences read back out of the
ledger by the writer's `--emit-sequence`:

```
$ node <extracted enforceEscalation + ledger-sourced sequences>
CONTROL  1 prior C + current CONDITIONAL       n=1  would_auto_fail=false
DRIVEN   2 prior C + current CONDITIONAL       n=2  would_auto_fail=true
CONTROL  2 prior C + current PASS              n=2  would_auto_fail=false
CONTROL  2 prior C + current FAIL              n=2  would_auto_fail=false
```

Row 1 is the anti-vacuity control: the rule does **not** fire early. Rows 3 and 4
are the semantics controls: a PASS stays a PASS, a FAIL stays a FAIL, and neither
is converted.

**A rail drop must not CLEAR an escalation:**

```
[C,C,NO_VERDICT] + CONDITIONAL -> n=2    would_auto_fail=true    (drop does NOT reset)
absent sequence                -> n=null would_auto_fail=null
                                  sequence_status=not_supplied   (unknown, NOT zero)
```

**On 86.74's REAL backfilled priors:**

```
$ python scripts/qa/verdict_ledger_write.py --emit-sequence --step 86.74   [minus the current row]
["NO_VERDICT","NO_VERDICT","CONDITIONAL","CONDITIONAL","PASS","CONDITIONAL","CONDITIONAL"]
consecutive_conditionals = 2   would_auto_fail = true
```

**This is the point of the step.** Earlier today the same computation returned
`sequence_status: "not_supplied"`, `consecutive_conditionals: null`,
`would_auto_fail: null`, because the sequence had to be hand-carried and arrived as
prose in `extra`. Main computed the auto-FAIL by hand. With the ledger populated the
shipped consumer reaches the same answer **unaided**.

---

## 4. IDEMPOTENCY -- the dedup key refuses a replay

```
$ python scripts/qa/verdict_ledger_write.py --step 86.74 --verdict CONDITIONAL \
      --run-id wf_8c3730a1-32e --cycle 7 --note "duplicate attempt"
verdict_ledger_write: duplicate key ('86.74', 'run:wf_8c3730a1-32e') already in
  handoff/verdict_ledger.jsonl. Refusing to append. This ledger is append-only and a
  key identifies one logical event; a correction must be a NEW labelled row, never a
  rewrite.
exit=2
rows for 86.74 still: 8
```

A retried spawn gets a **new** `run_id` and is legitimately a new row; only a
re-transcription of the **same** run is refused.

---

## 5. THE HOOK MEASUREMENT -- settings restored byte-identical

Probe installed in the gitignored `.claude/settings.local.json`, then restored:

```
sha256 before : 8f03f1949599866fe3875266557ff23818d1d1dc5e1cf7a4eef337e68124d966
sha256 after  : 8f03f1949599866fe3875266557ff23818d1d1dc5e1cf7a4eef337e68124d966
BYTE-IDENTICAL RESTORE: True
probe hook remaining in settings: 0
```

Bash tool call (control -- proves the probe itself works):

```json
{"has_tool_response": true, "tool_name": "Bash",
 "tool_response_shape": {"keys": ["interrupted","isImage","noOutputExpected","stderr","stdout"]}}
```

Workflow tool call (the question):

```json
{"has_tool_response": true, "tool_name": "Workflow",
 "tool_response_shape": {"keys": ["runId","scriptPath","status","summary","taskId",
                                  "taskType","transcriptDir","workflowName"]}}
```

**The hook fires on `Workflow` and receives `tool_response` -- but that payload is
the LAUNCH RECEIPT and contains no verdict.** Workflows always run in the
background, so PostToolUse fires at launch; there is no foreground mode. **A hook
can never author a verdict row.** It could still alarm on a `runId` that never got a
row -- out of scope for 86.85, recorded as a follow-up.

---

## 6. MUTATION MATRIX -- 12/12 killed, control GREEN first, zero repo writes

> **THIS SECTION WAS STALE TWICE AND THE SECOND TIME IS THE WORSE ONE.** It shipped
> as "5/5"; cycle 1 proved those 5 blind to ORDERING (+M6, M7). The cycle-2
> remediation then named **both** `experiment_results` §2 **and** this file §8 --
> and I updated only the former while reporting the item done, silently narrowing
> the scope. Cycle 3 caught it: this header still read "7/7 ... Current state below"
> against a delivered 10. It is now **12 cells** -- cycle 3 additionally found the
> `_dedup_key` **cycle-fallback** branch uncovered (+M11, M12), a LIVE branch used
> by 5 real ledger rows, all on 86.74.

```
$ python scripts/qa/mutation_matrix_86_85.py
sha256 before: 2f0d1000f98ed03e3b92e25792e296e831775a784e8203968bdde9315d57c168
CONTROL      : rc=0 -> GREEN

M1  KILLED (rc=1)  remove the dedup refusal
M2  KILLED (rc=1)  remove the verdict vocabulary guard
M3  KILLED (rc=1)  allow an unkeyed row
M4  KILLED (rc=1)  swallow a corrupt ledger line
M5  KILLED (rc=1)  collapse event time into write time
M6  KILLED (rc=1)  REVERSE emit_sequence  <- cycle-1 QA-M1, SURVIVED the palindrome
M7  KILLED (rc=1)  remove the out-of-vocabulary loudness in emit_sequence
M8  KILLED (rc=1)  revert the fail-loud I/O guard   <- cycle-2 QA-M6
M9  KILLED (rc=1)  drop step_id from the dedup key  <- cycle-2 QA-M4
M10 KILLED (rc=1)  remove the empty-step_id guard
M11 KILLED (rc=1)  make the cycle fallback key CONSTANT <- cycle-3 QA-M2
M12 KILLED (rc=1)  DELETE the cycle fallback entirely   <- cycle-3 QA-M1

sha256 after : 2f0d1000f98ed03e3b92e25792e296e831775a784e8203968bdde9315d57c168
UNCHANGED    : True  (mutations ran on temp copies; the real file was never written)
12 cells: 12 killed, 0 survived, 0 unscorable
EXIT=0
```

---

## 7. IMMUTABLE VERIFICATION COMMAND

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"scripts/qa/verdict_history_86_21.py\").read()); print(\"parses\")"'
parses
EXIT=0
```

---

## 8. LEDGER STATE AFTER THIS STEP

Population = every non-blank line in `handoff/verdict_ledger.jsonl`.

```
[PRE-STEP  d1c4a79d~1]  total 35 | step_ids 10 | 86.74 rows 0 | recorded_by {main:35}
[AS SHIPPED d1c4a79d ]  total 43 | step_ids 11 | 86.74 rows 8 | recorded_at 29/43
                        verdict distribution {CONDITIONAL 23, FAIL 5, PASS 8, NO_VERDICT 7}
[WORKING TREE        ]  total 46 | this step's own cycle-1/2/3 FAIL rows included
```

**ANCHORED, because this step's ledger MOVES WHILE THE STEP RUNS** -- 86.85 records
its own verdicts into the file it counts, so an unanchored total goes stale the
moment the next cycle is graded. That is exactly what cycle 3 caught here: this
block read "43" against a measured 45, and carried no commit anchor even though
`experiment_results` §2 already had one. `recorded_at` is short of the total
because **14 historical rows predate the field**; every row this step writes has it.

---

## 9. HONEST LIMITS

1. **The writer is not WIRED to the seam.** Nothing calls it automatically when a
   verdict returns; Main must remember. The §5 measurement shows a hook cannot close
   that by authoring the row. **Un-forgettability is made possible, not solved.**
2. **The 8 backfilled 86.74 rows are a RECONSTRUCTION**, labelled as such in every
   `note`, with the source named. Two split one `harness_log` line reading "rail
   dropped x2" and carry no `run_id`, so they are keyed by cycle label.
3. **Only one consumer is proven.** `attempt_budget.py` (86.71) is still inert.
4. **No live spawn has yet consumed the ledger** for `args.verdict_sequence` -- the
   mechanism is proven by driving the shipped function, not by riding a real launch.
