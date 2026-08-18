STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.85
WRITTEN: 2026-08-17T11:14:48Z

# Q/A write-first record -- step 86.85, cycle 10 (per spawn prompt)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status/diff scope, ruff, pytest, self-test, matrix
C. LLM judgment vs the 8 immutable criteria read VERBATIM from .claude/masterplan.json

## Findings (appended as established)

### B. DETERMINISTIC -- all run by me
- IMMUTABLE COMMAND: `bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(...verdict_history_86_21.py...); print(\"parses\")"'` -> `parses`, **EXIT=0**.
- ruff F821/F401/F811 over DERIVED scope (`git diff --name-only HEAD -- '*.py'`, 8 files, empty-set guard asserted, passed via xargs) -> `All checks passed!` exit 0.
- self-test: exit 0, **32 checks** under the artifact's own C8.6 grep rule (`grep -cE '^  (ok  |FAIL)'` -> 32). 0 FAIL.
- pytest `-k '86_85 or ledger or verdict_ledger'` -> **38 passed, 3514 deselected**, exit 0.
- mutation matrix: CONTROL observed **GREEN first** (`CONTROL : rc=0 -> GREEN`), then **22 cells / 22 KILLED / 0 SURVIVED / 0 UNSCORABLE**, exit 0. sha256 before == after == `9ade917c6dd07c6e485902d42c14ba229316606deb1b893fc3a84f3ace853dc8`; matrix's own report `UNCHANGED: True`. My independent pre/post shasum agrees.
- coverage checker: `guards: 21   covered: 21   uncovered: 0   cell problems: 0`.
- C8.10's three quoted figures ALL reproduce byte-exact under their own quoted commands: 32 / 22 / 38.

### QA-C9-1 (cycle-9 WARN) -- CLOSED, verified by an INDEPENDENT construction
Built my own shape-half mutant with a DIFFERENT construction from the author's M22
(`ISO_DATE_RE = re.compile(r".*", re.S)` instead of `if False:`), on a temp copy only,
CONTROL run first (rc=0, 0 fails), repo sha256 identical before/after:
  MUT-A permissive ISO_DATE_RE -> **rc=1, KILLED** by exactly the new cycle-10 fixture
  `FAIL  compact ISO date (regex-only refusal) refused at build_row`.
So the fix is real and not a construction artifact of M22's anchor.
Also ran: MUT-C (calendar half swallowed, different construction from M21) -> KILLED (2 fails);
MUT-D (emit-seam truthiness half removed) -> KILLED by the "no event date" MESSAGE pin;
MUT-E (build_row None-half removed) -> rc=1 but 0 named fails (crash, i.e. UNSCORABLE-shaped, still detected).

### A SURVIVOR I INVESTIGATED AND RULED **EQUIVALENT** (not a finding)
MUT-B: `\A\d{4}-\d{2}-\d{2}\Z` -> `\d{4}-\d{2}-\d{2}` (anchors removed) **SURVIVED** the
32-check self-test. Behavioural differential attempted and NOT found: for the mutant to
differ, a string must be unanchored-matched AND accepted by `date.fromisoformat` AND
refused by the anchored regex. Tested 11 candidates ('2026-08-10T00:00:00', ' 2026-08-10',
'2026-08-10Z', 'x2026-08-10', '+2026-08-10', ...): `fromisoformat` **rejects every one**.
The forms it does accept ('20260810', '2026-W32-1') contain no `\d{4}-\d{2}-\d{2}` substring.
=> EQUIVALENT mutant under the ANDed guard. Reported as a negative result, not a finding.

### CRITERIA 3/4/6/7 -- DRIVEN BY ME, not read
`enforceEscalation` brace-extracted from `.claude/workflows/qa-verdict.js:535` into a temp
module (naive `{`-grab hits `opts = {}` -- matched the PARAM LIST first; final body 2225 chars,
asserted to contain would_auto_fail AND burden_on).
- **C3 cross-process persistence**: 3 SEPARATE `python3 ... --step 99.888 --verdict CONDITIONAL
  --ledger <tmp>` invocations, then a **4th separate** `--emit-sequence` process ->
  `["CONDITIONAL", "CONDITIONAL", "CONDITIONAL"]`. MET.
- **C4 3rd-CONDITIONAL fires**: that ledger-sourced array + current CONDITIONAL -> n=3
  auto=**true**. Anti-vacuity controls all fire the other way: 1 prior -> auto=false;
  [C,C]+PASS -> false; [C,C]+FAIL -> false. MET by execution.
- **C6 drop must not clear / absence != zero**: [C,C,NO_VERDICT]+CONDITIONAL -> n=2 auto=**true**
  (the drop neither extends nor resets); absent -> n=**null** status=not_supplied; null -> null;
  garbage token -> null/unparseable; non-array -> null/unusable. **Never 0.** MET.
- **C7 verdict semantics unchanged**: 220-cell sweep (4 current verdicts x 11 sequences x 5 opt
  combos incl. attempt_number/max_attempts) -> **0 violations** on all four tests: input object
  never mutated, return never carries verdict/ok, no non-PASS ever becomes PASS, unknown never
  reported as 0. MET under every flag combination.

### PYTEST HALF IS NON-VACUOUS TOO (zero repo writes)
Copied the REAL test module into a scratch tree whose `parents[2]` resolves to my mutant
(the module derives WRITER from `__file__`), so the shipped pytest ran against mutants:
CONTROL 31 passed -> MUT-A (shape half) **2 failed** incl. `test_non_iso_date_refused_at_both_seams`;
MUT-C (calendar half) **2 failed**. repo sha unchanged. Both halves are killable in BOTH harnesses.
(File-scoped run is 31 vs the `-k` selector's 38 -- a stated selector difference, not a discrepancy.)

### C1 RE-DERIVED FROM GIT BY ME
`git show d1c4a79d~1:handoff/verdict_ledger.jsonl` -> 10814 bytes, **35 rows**, 10 step_ids,
**0 rows for 86.74**, {C18,F5,P7,NV5}, max date 2026-08-11. Positive control **86.21 -> 5 rows**,
which is what licenses reading the 86.74 zero as MEASURED rather than a broken query.
`d1c4a79d` -> 43 rows / 11 step_ids / 8 86.74 rows / {C23,F5,P8,NV7}. All reproduce EXACTLY.
The three 86.74 verdicts were NOT on disk, so no re-scope was owed. Cause = NEVER-WRITTEN.

### FINDING QA-C10-1 [WARN] -- the "second consumer is hypothetical" claim no longer reproduces
`experiment_results_86.85.md` s4 item 4: *"Only one consumer is proven. `enforceEscalation` is
driven end-to-end; `attempt_budget.py` (86.71) is still inert and unwired, so the ledger's second
intended consumer remains hypothetical."* And C5: *"86.71 (cumulative budget) would be the
ledger's second consumer; out of scope."*
MEASURED AGAINST HEAD:
- `192ef652` (2026-08-17 **12:35:43** +0200) is an ANCESTOR of HEAD (`git merge-base --is-ancestor` OK).
- `.claude/settings.json:39` registers `scripts/harness/attempt_gate.py` as a **live PreToolUse hook**.
- At that commit `attempt_gate.py:151-152` does `from verdict_ledger_write import emit_sequence` /
  `seq = emit_sequence(step_id, VERDICT_LEDGER)`, and `VERDICT_LEDGER` defaults to the **REAL**
  `handoff/verdict_ledger.jsonl` (`attempt_gate.py:90-91`).
- The cycle-10 artifacts were written at 13:09:56 local -- **34 minutes AFTER** that commit.
So 86.85's reader IS on the live tool-call path; the second consumer is not hypothetical.
SECOND HALF: that consumer wraps the call in `except Exception: return []`
(`attempt_gate.py:154`), converting every LOUD refusal cycles 1-10 built (non-ISO date,
calendar-invalid date, undated row, out-of-vocabulary verdict, corrupt line) into a silent
empty list -- the "missing row readable as no prior verdict" shape.
DIRECTION OF HARM **MEASURED, NOT ASSUMED**: `verdict_outcomes` feeds only the PASS exception and
`disposition()` checks PASS before exhaustion (`attempt_gate.py:47`), so `[]` can only REMOVE an
allowance (deny where allow was due) = fail-CLOSED; and the F1 3rd-CONDITIONAL path runs through
`enforceEscalation` on a Main-supplied sequence, not through this hook. **No escalation is cleared
today.** That is why this is WARN, not BLOCK, and why C5/C6 are graded MET as literally worded.
NAMED FIX: correct s4 item 4 and the C5 bullet in place to name `attempt_gate.py` as a LIVE
consumer; and queue (86.71's file, not this step's) replacing the blanket `except Exception` with
a narrow one that distinguishes "no rows" from "refused to order".

### FINDING QA-C10-2 [WARN] -- C8.8's three superseded FIGURES still not replaced
Cycle-9's named fix was verbatim: *"replace C8.8's three superseded figures **and** its
'CURRENT'/'this session' framing in place (or delete the block), and repoint :183-184 at C8.9."*
DONE: heading -> "cycle-7/8 capture ... SUPERSEDED", currency sentence replaced, and three forward
pointers repointed (:184, :367, :457). NOT DONE: the three figures. Still on the page --
"30 checks", "20 cells: 20 KILLED", "guards: 17 covered: 17".
I MEASURED THE CYCLE-7 TREE MYSELF rather than trusting cycle 8: `git show
f3c89229:scripts/qa/verdict_ledger_write.py` run under the artifact's OWN C8.6 grep rule -> **29**,
and `git show f3c89229:scripts/qa/verify_matrix_coverage_86_85.py` -> **guards: 21 covered: 21**.
So 2 of the 3 figures were **never true at any time**, and the replacement sentence's first clause
("Every figure below was from a live run AT ITS CAPTURE") is contradicted by my own measurement --
hedged, not retracted, by the clause that follows it. Recurrence #6 in this file.
Materially SMALLER than cycle 9's version (the framing WAS replaced in place this time, and every
current figure lives in C8.10 and reproduces byte-exact). NAMED FIX: state the measured values
(29 / 21 cells at that commit / guards 21-21) inline where the wrong ones stand, or delete them.

### A SECOND SURVIVOR HUNT THAT FOUND NOTHING (recorded as a negative result)
No further survivor. Enumerated the other compound guards: build_row's `is not None` half
(MUT-E detected), emit's truthiness half (MUT-D killed by the MESSAGE pin, not merely by
"a LedgerError was raised"), the sort key, the vocabulary guard.

### HARNESS COMPLIANCE -- all five clean
1. Research gate: brief_status COMPLETE, gate_passed true, 8 sources (>=5), 23 URLs (>=10),
   recency_scan true; contract s1 cites it. Brief FIRST COMMITTED 9034ddfb 2026-08-14 21:41 --
   BEFORE the contract's d1c4a79d 2026-08-15 15:44 (its later mtime is a later annotation).
2. Contract (15:44 commit / 15:59 mtime) < generated artifacts (2026-08-17 13:08-13:09).
3. experiment_results_86.85.md AND live_check_86.85.md both present.
4. Log-last: masterplan 86.85 still `"pending"`; `grep -F phase=86.85 handoff/harness_log.md`
   returns only the two 2026-08-15 rows -- the in-flight cycle is NOT logged, step NOT flipped.
5. No verdict-shopping: cycle-9 WIP 13:04:40 < 86.85 sources+artifacts 13:08:51/13:09:56, and the
   content changed (M22 cell, compact-ISO fixture in self-test AND pytest, C8.8 replaced in place,
   C8.10 added). Documented fresh-respawn-on-CHANGED-evidence flow.

### PRIOR-VERDICT EVIDENCE (gathered; not applied as a trigger)
`verdict_history_86_21.py --step 86.85 --evidence-only`: status **ok**, "9 verdict(s) from the
ledger", sequence = FAIL -> FAIL -> FAIL -> CONDITIONAL -> CONDITIONAL -> FAIL -> **NO_VERDICT** ->
CONDITIONAL -> CONDITIONAL. NO_VERDICT carried through as-is, not collapsed.
`qa_wip.py 86.85 --spawned-at 2026-08-17T11:14:48Z`: source_present **true**,
attempt_number_status **ok**, attempt_number **10**, prior_attempts **9**,
attempt_number_is_lower_bound true, records_pruned_known **null**, records_retained 10 (the
payload's own unit string calls this a GAUGE inclusive of my own record -- I did not use it).
CROSS-CHECK: 9 prior attempts vs 9 ledger rows -> they AGREE, so the ledger is NOT stale for this
step. Notable because qa.md documents the ledger as hand-appended and measured 86.62 at
qa_wip 4 vs ledger `no_rows_for_step`.

### VERDICT (mine): CONDITIONAL
All 8 criteria MET as literally worded and independently re-derived/driven; two WARN findings
(QA-C10-1 scope honesty, QA-C10-2 artifact accuracy) cap at CONDITIONAL by worst-severity dispatch.

MEMORY: recorded the new reusable class at
.claude/agent-memory/qa/feedback_unwired_is_a_claim_with_an_expiry.md and indexed it in MEMORY.md.

COMPLETED: 2026-08-17T11:27:38Z
