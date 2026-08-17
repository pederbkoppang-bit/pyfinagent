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

*(cycle-5 annotation, 2026-08-17: this heading's 12/12 is the CYCLE-3 count and
is superseded -- the matrix grew to 14 cells / 14 KILLED in cycle 4 (M13, M14
added; see C8 below). The heading is kept because the section under it is the
cycle-3 capture; the cycle-4 Q/A flagged the un-annotated heading as its
non-blocking note 1, closed here.)*

*(cycle-7 annotation, 2026-08-17: the matrix has since grown again -- 20 cells
as of cycle 7 (event-order, same-date, and ISO-date-validation cells added in
cycles 6-7). EVERY count in this file's sections 6 and C8 is a dated capture of
its own cycle; the CURRENT figures are produced by running the commands, and
the latest captured run is the LAST C8.x section below (earlier C8.x sections are dated captures, superseded by construction). The cycle-6 Q/A found this
file stale a third time, so the rule is now stated where the numbers live:
a matrix count quoted without its cycle label or command is wrong by default.)*

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
[WORKING TREE        ]  total 52 as of 2026-08-17 after the cycle-5/6 writes | perishable BY DESIGN (the step records its own verdicts): quote it only with its date and the command `wc -l < handoff/verdict_ledger.jsonl` | the cycle-5 Q/A measured 47 when 45/46 stood here -- three artifacts disagreeing is what an unanchored moving number produces
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

---

# CYCLE 4 -- C8 ONLY (2026-08-15)

Prior verdicts on this step, passed to the Q/A as DATA in
`args.verdict_sequence` (from `verdict_ledger_write.py --emit-sequence --step
86.85`): **`["FAIL", "FAIL", "FAIL"]`**.

All three FAILs were ONE class: **a new guard shipped with no mutation cell**
(cycle 1 ordering; cycle 2 fail-loud-I/O + step_id-in-key; cycle 3
cycle-fallback). Each time the guard list was written BY HAND and the Q/A found
the one that was missed. A hand-list cannot be audited by the person who wrote
it -- its omissions are invisible from the inside.

**So cycle 4 does not write another hand-list.** It DERIVES guard-shaped
coverage instead.

**READ C8.5 BEFORE BELIEVING ANY COMPLETENESS CLAIM IN THIS SECTION.** The
cycle-4 Q/A refuted the strong reading and I re-measured it myself: known-member
recall against this step's own three prior FAILs is **1 of 4**. The gate catches
**guard-shaped** omissions (a `raise` or refusing `if` with no cell) and NOT
**behavioural** ones -- and 3 of the 4 historical misses were behavioural. Any
sentence below that reads as "the failure class is now structurally closed" is
governed by this paragraph and by C8.5.

## C8.1 -- the new checker: `scripts/qa/verify_matrix_coverage_86_85.py`

It enumerates the writer's guards FROM SOURCE by AST, then measures -- per cell
-- which guards each cell actually touches. **No cell declares what it covers**;
a declaration would be a hand-list wearing a different hat. Any enumerated guard
that no cell touches makes the checker exit non-zero.

**The enumeration rule, written down so it can be argued with:**

- `GUARD-RAISE` -- every `raise LedgerError(...)`. `LedgerError` is the module's
  own refusal type, so this is a rule about semantics, not a curated list. It
  excludes `raise SystemExit(main())` (CLI dispatch) **without anyone naming
  it**.
- `GUARD-BRANCH` -- every `ast.If` that refuses DIRECTLY: its body reaches a
  GUARD-RAISE or a failure-code `return` **without passing through a nested
  `ast.If`**. Ancestors route; they do not guard.

**Coverage is measured two ways, and each was added because the other gave a
wrong answer:**

- (a) STRUCTURAL -- the guard's fingerprint is present in the original and
  absent/altered in the cell's mutated source.
- (b) TEXTUAL -- the cell's anchor overlaps the guard's own span **or the span
  of an enclosing `ast.If`**.

## C8.2 -- three defects the checker found in its own first runs

Recorded because each is a case of a check that looked clean and was not.

1. **FALSE GAPS from own-span-only matching.** First run reported 8 uncovered of
   16 -- but `RAISE::append_row::duplicate key` is killed by M1, which neuters
   the enclosing `if key in existing_keys(path):`. The raise's own AST node and
   text are untouched, so own-span matching missed it. Fixed by adding
   enclosing-`If` spans.
2. **AN OVER-COUNTED GUARD.** `if args.emit_sequence:` was enumerated as a guard
   because its body ends `return EXIT_OK` and the rule tested `"EXIT_" in txt`.
   A success return is not a refusal. Rule corrected to exclude `EXIT_OK`;
   guards 16 -> 15.
3. **OVER-CREDITED COVERAGE -- the dangerous direction, and the one that would
   have hidden a real gap.** The ancestor rule initially included `ast.Try`.
   `main` wraps its whole body in one `try:`, so every anchor inside it
   "overlapped" every guard in it, and cell M13 was credited with covering a
   raise in a different branch. **The tell: dropping cell M14 left the gate
   GREEN.** This was invisible until the gate was tested for its ability to go
   RED. Ancestors are now `ast.If` only.

## C8.3 -- what the derived checker found in the MATRIX

With the rules corrected, the checker reported **`main`'s CLI argument
validation had NO cell aiming at it** -- while the matrix reported **12/12
KILLED**. A perfect score over an incomplete list: the 86.85 failure class in
miniature, found by derivation rather than by re-reading the file.

Two cells added, **M13** and **M14**. `M13`'s first anchor did not match
(`count=0`); the checker reported it as a CELL PROBLEM rather than silently
scoring it -- a no-match `str.replace` looks exactly like success.

**M14 then SURVIVED**, i.e. a guard that cannot fail. Cause, measured: the CLI
guards are defence-in-depth -- `build_row` refuses an empty `--step` and an
empty `--verdict` anyway -- so removing the CLI guard still refuses and only the
MESSAGE changes. The pre-existing self-test checks asserted only that the exit
code was 3, and **both paths return 3**, so they were vacuous. Three checks
added that pin WHICH refusal fires (plus an anti-vacuity check that a
well-formed append still succeeds). M14 is now KILLED.

## C8.4 -- the matrix is now RED when a guard has no cell

`mutation_matrix_86_85.py` calls the coverage checker and fails on a non-zero
result, so "every cell killed" can no longer be reported as success while a
guard sits uncovered.

## C8.5 -- proof the gate can go RED (drop-one-cell sweep)

A gate that has never been observed failing is a zero-assertion guard. Each cell
was removed in memory and the gate re-run:

```
CONTROL (14 cells) -> exit 0          [cycle-4 capture -- current counts move with the suite; see the LAST C8.x section]
    guards: 15   covered: 15   uncovered: 0   cell problems: 0

  drop M1   -> exit 1 RED   gaps=2        drop M8   -> exit 1 RED   gaps=1
  drop M2   -> exit 1 RED   gaps=2        drop M9   -> exit 0 GREEN (0 enumerated-guard coverage)
  drop M3   -> exit 1 RED   gaps=1        drop M10  -> exit 1 RED   gaps=2
  drop M4   -> exit 1 RED   gaps=1        drop M11  -> exit 0 GREEN (0 enumerated-guard coverage)
  drop M5   -> exit 0 GREEN (0 enumerated-guard coverage)   drop M12  -> exit 0 GREEN (0 enumerated-guard coverage)
  drop M6   -> exit 0 GREEN (0 enumerated-guard coverage)   drop M13  -> exit 1 RED   gaps=2
  drop M7   -> exit 1 RED   gaps=2        drop M14  -> exit 1 RED   gaps=2
```

*(cycle-5 relabel, 2026-08-17: the five GREEN rows above previously carried the
label "(coverage-redundant)". Those parenthetical labels were HAND-AUTHORED
annotations, not checker output -- no script in the repo prints that phrase
(`grep -rn "coverage-redundant" scripts/qa/*.py` matches only a comment in
`verify_cell_vacuity_86_89.py` describing the error) -- so the exit codes are
verbatim and the labels were editorial. They are relabelled to the measured
fact the correction below establishes: those five cells cover ZERO enumerated
guards. Leaving the refuted word inside the fence, above its own refutation,
is the surviving-copy defect this project keeps re-finding.)*

**CORRECTED BY THE CYCLE-4 Q/A, AND THE CORRECTION MATTERS MORE THAN THE
ORIGINAL CLAIM.** The paragraph that stood here said the five GREEN rows were
"coverage-redundant -- another cell touches the same *guard*". **That is false,
and I have re-measured it myself rather than taking the Q/A's word:**

```
guards covered by MORE THAN ONE cell: 0

  M1  covers 2 guard(s)      M8   covers 1 guard(s)
  M2  covers 2 guard(s)      M9   covers 0 guard(s)  <-- ZERO
  M3  covers 1 guard(s)      M10  covers 2 guard(s)
  M4  covers 1 guard(s)      M11  covers 0 guard(s)  <-- ZERO
  M5  covers 0 guard(s) <-- ZERO   M12  covers 0 guard(s)  <-- ZERO
  M6  covers 0 guard(s) <-- ZERO   M13  covers 2 guard(s)
  M7  covers 2 guard(s)      M14  covers 2 guard(s)
```

**There is no redundancy anywhere in the coverage map.** Those five cells cover
**zero** enumerated guards. The gate stays green without them not because
another cell covers the same guard, but because **their targets are invisible to
the enumeration rule** -- and those two readings differ materially: "redundant"
says the gate is still complete, "invisible" says the gate is structurally
blind. I wrote the reassuring one.

**THE CONSEQUENCE, MEASURED -- and it is the finding of this cycle.** The
known-member set is not one I chose after the fact: it is the three prior FAILs,
named in the checker's own docstring as its motivating class. Dropping the cell
for each historical miss and re-running the gate:

```
  drop ['M6']         cycle-1 QA-M1  ordering           -> GREEN  NOT demanded
  drop ['M8']         cycle-2 QA-M6  fail-loud I/O      -> RED    DEMANDED
  drop ['M9']         cycle-2 QA-M4  step_id-in-key     -> GREEN  NOT demanded
  drop ['M11','M12']  cycle-3 QA-M1/M2  cycle-fallback  -> GREEN  NOT demanded

RECALL = 1 of 4
```

**So the mechanism I offered as the structural end of this step's three-cycle
failure class would not have demanded 3 of the 4 guards whose omission caused
those three cycles.** By this project's own rule -- *a scan that cannot find its
own known members is a FAILED gate* -- a 1-of-4 recall does not support the
claim "completeness is now DERIVED", and that sentence is withdrawn. What the
gate genuinely does is narrower and still worth having: **it catches
guard-shaped omissions** (a `raise LedgerError` or a refusing `if` with no
cell), which is exactly how it found `main`'s uncovered CLI validation. It does
**not** catch behavioural omissions, which were 3 of the 4 historical misses.
Extending the enumeration to behavioural guards is queued as step **86.89**.

Also corrected: the behaviour list above named "sequence filtering", for which
**no cell exists**, and omitted M5's actual target. The five cells' real targets
are **ordering (M6), dedup-key composition (M9), cycle fallback (M11, M12), and
event/write-time separation (M5)**. All five remain KILLED by the matrix.

**Residual risk is bounded and stated rather than assumed:** behavioural
coverage of ordering does not depend on M6 alone, because the self-test carries
both "sequence is oldest->newest" and the "order fixture is NOT palindromic
(anti-vacuity)" guard-on-the-guard.

The two checks still answer different questions -- "can this guard fail?" versus
"is any guard unmutated?" -- and conflating them is how a 12/12 score coexisted
with an uncovered guard. That part of the original paragraph stands.

Additionally the checker runs a **self-control before reporting**: it plants a
synthetic guard into an in-memory copy and requires itself to report that guard
as UNCOVERED. If it cannot detect a guard it planted itself it exits 2 with
FAILED GATE rather than a clean bill.

## C8.6 -- verbatim output *(cycle-4 capture, 2026-08-15 -- superseded; current run in the LAST C8.x section)*

### writer self-test
```
SELF-TEST PASSED
checks emitted: 23        (was 20 before this cycle)
```

### mutation matrix, now gated on derived coverage
```
14 cells: 14 killed, 0 survived, 0 unscorable
guards: 15   covered: 15   uncovered: 0   cell problems: 0
RESULT: OK -- every enumerated guard is touched by at least one cell.
exit=0
```
```
UNCHANGED    : True  (mutations ran on temp copies; the real file was never written)
UNCHANGED    : True  (all mutation was in memory)
```
The writer's sha256 is printed before and after by both scripts and is unchanged
-- no restore step exists to get wrong, because nothing is ever written.

### pytest
```
selector: -k '86_85 or ledger or verdict_ledger'
34 passed, 3498 deselected, 1 warning in 6.16s
```

**Population rule for every count above (C2 discipline):** `checks emitted` is
`grep -cE '^  (ok  |FAIL)'` over the self-test's own output; `cells` is
`len(CELLS)` in `mutation_matrix_86_85.py`; `guards` is the AST enumeration
under the rule in C8.1; the pytest number is whatever the quoted `-k` selector
selects and is **not** comparable to a count taken under a different selector.

## C8.7 -- what this does NOT claim

- A matrix licenses exactly **"these 14 mutations were killed"**. The coverage
  checker adds a separate, narrower guarantee: **"no guard the enumeration can
  see is unmutated."** Neither is a claim that the writer's guard SET is
  complete or correct.
- The enumeration has its own blind spots -- it sees `raise LedgerError` and
  refusing `if` branches. A guard expressed some other way (a silent `return`
  with no failure code, a validation inside a helper called for effect) is
  outside it. The rule is written down in C8.1 precisely so that limit is
  auditable rather than implicit.
- **Scope:** this cycle touched C8 only. C1-C7 are unchanged from cycle 3 and
  were not re-litigated.


---

## C8.8 -- cycle-7/8 capture, 2026-08-17 -- SUPERSEDED (current: the LAST C8.x section below)

Every figure below was from a live run AT ITS CAPTURE; the cycle-8 Q/A measured several stale on arrival and the cycle-9 Q/A found this heading still claiming currency -- figures here are HISTORY; the current run is the LAST C8.x section.

```
$ python3 scripts/qa/verdict_ledger_write.py --self-test
SELF-TEST PASSED                                                    (exit 0; [cycle-11 correction: this figure said 30 -- the cycle-10 Q/A measured the cycle-7 tree at 29 by this artifact's own grep rule, so it was never true; current counts live in the LAST C8.x section] checks
                                incl. same-date file-order, backfill event-order,
                                non-ISO refusal at BOTH seams, undated-row loudness)

$ python -m pytest backend/tests -k '86_85 or ledger or verdict_ledger' -q
38 passed, 3514 deselected                                          (exit 0)

$ python3 scripts/qa/mutation_matrix_86_85.py
20 cells: 20 KILLED, 0 survived, 0 unscorable   [cycle-11 note: accurate for the cycle-7 tree; current count in the LAST C8.x section]                       (exit 0)
  (M5 retargeted to stay date-shaped -- its old timestamp replacement died at
   the NEW ISO guard, the wrong guard, and scored UNSCORABLE; M17 rewritten to
   the verdict-participates key -- the originally-named plain-sorted mutant is
   now EQUIVALENT because pos discriminates, and pos-to-constant alone degrades
   to stable file order, both stated in the cell comments)

$ python3 scripts/qa/verify_matrix_coverage_86_85.py
guards: 17   covered: 17   uncovered: 0   cell problems: 0          (exit 0)   [cycle-11 correction: the cycle-10 Q/A measured the cycle-7 tree at guards 21/21 -- this line was never true at any tree; current in the LAST C8.x section]

$ python3 scripts/qa/verdict_ledger_write.py --emit-sequence --step 86.85
["FAIL", "FAIL", "FAIL", "CONDITIONAL", "CONDITIONAL", "FAIL"]
```

**The cycle-6 FAIL's three findings, closed here:** the sort key now EXCLUDES
the verdict (`key=lambda t: (t[0], t[1])`) with a same-date fixture whose file
order (C, P, F) differs from alphabetical (C, F, P), so a verdict-fallthrough
mutant dies (M17) and a within-date reversal dies (M18); non-ISO event dates
are REFUSED at both seams (build_row and emit_sequence -- M19/M20; the 11
legacy range-shaped rows on 36.17/86.20/86.17 now produce a LOUD error if those
steps are ever emitted, recorded as a residual repair question rather than
silently mis-ordered); and this file's stale counts are corrected by dated
capture labels plus this current section.


---

## C8.9 -- cycle-9 captured run (2026-08-17, regenerated in full at write time)

The cycle-8 Q/A found C8.8's figures stale ON ARRIVAL (the fourth recurrence of
this file's own disease) and one anti-vacuity check tautological. The counts
below were captured by running the commands IMMEDIATELY before writing this
section, and each carries its population rule inline:

```
$ python3 scripts/qa/verdict_ledger_write.py --self-test 2>&1 | grep -cE '^  (ok  |FAIL)'
31                                             (all 'ok', exit 0; the count MOVES with the suite)

$ python3 scripts/qa/mutation_matrix_86_85.py 2>/dev/null | grep -cE 'KILLED \(rc'
21                                             (21 cells incl. M21 calendar-half; 0 SURVIVED, 0 UNSCORABLE; exit 0)

$ python3 scripts/qa/verify_matrix_coverage_86_85.py | grep '^guards:'
guards: 21   covered: 21   uncovered: 0   cell problems: 0

$ python3 scripts/qa/verdict_ledger_write.py --emit-sequence --step 86.85
["FAIL", "FAIL", "FAIL", "CONDITIONAL", "CONDITIONAL", "FAIL", "NO_VERDICT", "CONDITIONAL"]
   (corpus-relative: this array grows as the loop runs; quote it only with its date)
```

**Cycle-8 findings closed:** (QA-C7-1) `valid_event_date` now ANDs the shape
regex with `datetime.date.fromisoformat` at BOTH seams -- 2026-18-10,
2026-02-30 and 9999-99-99 are all refused (fixtures in the self-test and
pytest; matrix cell M21 kills the calendar half's removal). (QA-C7-3) the
anti-vacuity check now derives the date set FROM THE ROWS ON DISK instead of a
locally-built literal. (QA-C7-2) this section replaces C8.8's role as current;
every figure above was regenerated at write time and states that it moves.


---

## C8.10 -- cycle-10 captured run (2026-08-17, regenerated at write time)

```
$ python3 scripts/qa/verdict_ledger_write.py --self-test 2>&1 | grep -cE '^  (ok  |FAIL)'
32                                            (all 'ok', exit 0; +1: the compact-date
                                               shape-only refusal fixture)
$ python3 scripts/qa/mutation_matrix_86_85.py 2>/dev/null | grep -cE 'KILLED \(rc'
22                                            (22 cells incl. M22 shape-half; 0 SURVIVED,
                                               0 UNSCORABLE; exit 0)
$ python -m pytest backend/tests -k '86_85 or ledger or verdict_ledger' -q | tail -1
38 passed, 3514 deselected                    (exit 0)
```

**Cycle-9 WARNs closed:** (QA-C9-1) BOTH halves of `valid_event_date` are now
independently killable -- the new fixture `20260810` (compact ISO) is refused
ONLY by the regex half (`date.fromisoformat` accepts it, and it sorts
lexicographically LAST -- the escalation-clearing direction the evaluator
drove), and cell M22 removes the shape half and dies against that fixture,
exactly as M21 removes the calendar half and dies against `2026-18-10`.
(QA-C9-2) C8.8 is corrected AT THE SITE this time: retitled SUPERSEDED, its
currency sentence REPLACED, and every forward pointer renamed to the LAST
C8.x section -- after the cycle-9 evaluator measured my previous
"corrected at the site" claim as purely additive.


---

## C11. Cycle-11 captures (2026-08-17)

```
$ shasum -a 256 scripts/qa/verdict_ledger_write.py
9ade917c6dd07c6e485902d42c14ba229316606deb1b893fc3a84f3ace853dc8  scripts/qa/verdict_ledger_write.py
$ python3 scripts/qa/verdict_ledger_write.py --self-test | tail -1; echo EXIT=$?
SELF-TEST PASSED
EXIT=0
$ python -m pytest backend/tests -k "86_85 or ledger or verdict_ledger" -q --no-header | tail -1
38 passed, 3514 deselected, 1 warning in 7.62s
$ uvx ruff check --select F821,F401,F811 scripts/harness/attempt_gate.py scripts/qa/verdict_ledger_write.py scripts/qa/rail_turn_cap.py scripts/qa/mutation_matrix_86_71.py scripts/qa/mutate_rail_turn_cap.py | tail -1
All checks passed!
```

The sha is byte-identical to the cycle-10 evaluator's own measurement --
cycle 11 changed prose and a DIFFERENT step's code (86.71), not this
step's source. The mutation matrix was therefore not re-run this cycle;
the cycle-10 verdict's independently-executed run (22/22 KILLED, control
green first, sha match before/after) remains the current-source evidence.
