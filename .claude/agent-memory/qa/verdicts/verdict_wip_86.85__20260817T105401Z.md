STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.85
WRITTEN: 2026-08-17T10:54:01Z

# Q/A write-first record -- step 86.85, cycle 9 (per spawn prompt)

Spawn context (ADVISORY, from Main -- Main is the constrained party):
- attempt_number claimed: 9
- sequence_provenance claimed: verdict_ledger_write.py --emit-sequence --step 86.85
- operator_authorization claimed for attempt 9 (cycle 8 = all 8 criteria MET, 3 WARNs)

Plan:
A. harness-compliance audit (5 items)
B. deterministic: immutable command + git status/diff + ruff + pytest + self-test + matrix
C. LLM judgment vs the 8 immutable criteria read VERBATIM from .claude/masterplan.json

## Findings log (appended as established)

### Prior-attempt evidence (gathered, NOT applied as a trigger)
- `qa_wip.py 86.85 --spawned-at 2026-08-17T10:54:01Z`:
  source_present=true, attempt_number_status=ok, attempt_number=9,
  prior_attempts=8, records_retained=9 (gauge, not counter),
  records_pruned_known=null, identity_checked=true.
- `verdict_history_86_21.py --step 86.85 --evidence-only`: status=ok,
  detail "8 verdict(s) from the ledger",
  sequence = FAIL -> FAIL -> FAIL -> CONDITIONAL -> CONDITIONAL -> FAIL ->
  NO_VERDICT -> CONDITIONAL  (NO_VERDICT carried through as-is).
- CROSS-CHECK: attempt_number (9) vs ledger row count (8). 8 prior attempts
  produced 8 ledger rows => the ledger is NOT stale for this step. This is the
  first step-id I have graded where the two sources AGREE.

### Deterministic checks (all re-run by me, this process)
- IMMUTABLE COMMAND `bash -c 'source .venv/bin/activate && python -c "import
  ast; ast.parse(open(\"scripts/qa/verdict_history_86_21.py\").read());
  print(\"parses\")"'` -> stdout "parses", EXIT 0.
- self-test `verdict_ledger_write.py --self-test` -> SELF-TEST PASSED, exit 0,
  31 "ok" lines, 0 FAIL.
- pytest file-scoped `backend/tests/test_phase_86_85_verdict_ledger_write.py`
  -> 31 passed (22 `def test_`, parametrised to 31 collected).
- pytest -k scoped `backend/tests -k '86_85 or ledger or verdict_ledger'`
  -> **38 passed**, 3514 deselected, exit 0. REPRODUCES the artifact's "38".
  (The artifact quotes its selector at experiment_results:645, so the 31-vs-38
  difference is a stated scope difference, not a discrepancy.)
- mutation matrix `mutation_matrix_86_85.py` -> CONTROL rc=0 GREEN first;
  21 cells, 21 KILLED, 0 survived, 0 unscorable; sha256 before == after
  (c32b626c...fda8), UNCHANGED True; coverage guards 21 covered 21 uncovered 0.
- ruff F821,F401,F811 over DERIVED scope (`git diff --name-only HEAD -- '*.py'`,
  8 files, non-empty asserted, xargs to avoid the zsh no-word-split trap)
  -> "All checks passed!", exit 0.
- C8.9's four figures each re-run by me and each reproduces EXACTLY:
  31 / 21 / "guards: 21   covered: 21   uncovered: 0   cell problems: 0" /
  the 8-element emit-sequence array.

### *** SURVIVING MUTANT -- QA-C9-1 (my own cell, not in the matrix) ***
The cycle-9 fix introduces ONE new guard, `valid_event_date`, with TWO halves:
the shape regex AND `datetime.date.fromisoformat`. The matrix adds a cell for
the CALENDAR half only (M21). I built the missing cell for the SHAPE half.

MUTATION (in memory, temp copies only, repo sha256 c32b626c741d8386 before AND
after -- zero repo writes):
    "    if not ISO_DATE_RE.match(s):\n        return False"
 -> "    if False:\n        return False"
anchor count = 1 (unique).

RESULT -- control observed GREEN first in BOTH harnesses:
  CONTROL  --self-test  rc=0 GREEN   |  MUTANT --self-test rc=0  *** SURVIVED ***
  CONTROL  pytest       31 passed    |  MUTANT pytest      31 passed *** SURVIVED ***
(pytest run against a relocated tree so `parents[2]` resolved to the mutant,
control run the same way -> the relocation itself is proven inert.)

BEHAVIOURAL DIFFERENTIAL (this is NOT an equivalent mutant):
`date.fromisoformat` accepts '20260810' and '2026-W32-1'; both sort AFTER every
hyphenated date lexicographically (measured: sorted(['2026-08-10','20260810'])
== ['2026-08-10','20260810']). Driven end to end:
  CONTROL: two CONDITIONALs on 2026-08-14/15, then backfill an OLDER PASS dated
           '20260810' -> REFUSED at build_row; sequence stays
           ['CONDITIONAL','CONDITIONAL'].
  MUTANT : same backfill ACCEPTED; sequence becomes
           ['CONDITIONAL','CONDITIONAL','PASS'] -- the older PASS lands LAST.
A trailing PASS resets the consecutive-CONDITIONAL count. That is the exact
escalation-CLEARING direction criterion 6 forbids and the exact defect class
the guard was written for ("a driven '2026-8-10' backfill landed AFTER every
August ISO date and cleared an escalation" -- the guard's own docstring).

WHY IT SURVIVED, measured: the only shape-half fixture anywhere is '2026-8-10'
(pytest:166), and `date.fromisoformat('2026-8-10')` ALSO raises ValueError, so
the calendar half subsumes it. No fixture exists that only the regex can refuse.

NOT a product defect -- the SHIPPED code is correct and does refuse. This is a
criterion-8 miss ("mutation-test EVERY new guard") on the half of the new guard
that was not mutated, with a criterion-6 behavioural consequence.
NAMED FIX: add a matrix cell that neuters the regex half, plus one fixture at
each seam using '20260810' (and ideally '2026-W32-1').

### Criteria 3/4/6/7 -- driven MYSELF, not read
Sliced the REAL `enforceEscalation` body out of `.claude/workflows/qa-verdict.js`
by brace-matching from the signature (the naive slice found the `{}` in
`opts = {}` and yielded a 55-char body -- caught by asserting the slice contains
'would_auto_fail'/'burden_on'; final body 2225 chars, both present) and exported
it from a temp copy. This drives the shipped function, never a re-implementation.
- C3: THREE separate `python verdict_ledger_write.py` write processes, then a
  FOURTH separate process `--emit-sequence` -> ['CONDITIONAL','CONDITIONAL',
  'CONDITIONAL']. Cross-process persistence DEMONSTRATED by me.
- C4: that LEDGER-SOURCED array + current verdict CONDITIONAL -> n=3,
  would_auto_fail=TRUE. Anti-vacuity controls in the same sweep: C,C,PASS -> n=0
  false; PASS/FAIL/NO_VERDICT with 3 prior C -> false. The rule fires only where
  it should.
- C6: ['C','C','NO_VERDICT'] + CONDITIONAL -> n=2, would_auto_fail=TRUE (a rail
  drop does NOT reset). absent -> n=null, status=not_supplied; garbage -> n=null,
  status=unparseable. Never 0.
- C7: 176 cells swept (4 current verdicts x 11 sequences x 4 opt combos incl.
  attempt_number/max_attempts). Violations = 0 on all three tests: the input
  verdict object is never mutated, the returned object never carries a
  `verdict`/`ok` key, and no non-PASS ever becomes PASS.
- `node scripts/qa/verify_escalation_86_78.mjs` -> 51 checks, 0 failed.

### QA-C9-2 (WARN) -- "corrected at the site" does not reproduce for live_check
`git diff HEAD -- handoff/current/live_check_86.85.md` is PURELY ADDITIVE: the
only change is the appended C8.9 block. Zero lines of C8.8 changed. Yet
experiment_results' new "Cycle 9 GENERATE" section claims "(2) The C8.8 figures
are corrected at the site". Residual state in live_check_86.85.md:
- :508  heading still reads "## C8.8 -- CURRENT captured run (cycle 7, ...)"
- :510  "Every figure below is from a live run this session"
- :514/:522/:530  still 30 checks / 20 cells / guards 17 (measured now: 31/21/21)
- :183-184 the file's own forward pointer still reads "the latest captured run
  is in section C8.8 below" -- now false, it is C8.9.
The SUBSTANTIVE remedy WAS delivered (C8.9 regenerated; all four of its figures
reproduce byte-exact for me; experiment_results' own summary line WAS corrected
in place with an inline correction note). What did not happen is the in-place
correction of C8.8 that the prose asserts. This is recurrence #5 of this file's
own documented disease, and the correction ACCOMPANIES rather than REPLACES.
The cycle-8 Q/A's NAMED fix was literally "regenerate C8.8 by re-running the
five commands and pasting their actual output" -- that is the part not done.

### Harness compliance (5 items) -- CLEAN
1. research-gate-before-contract: research_brief_86.85.md brief_status COMPLETE,
   gate_passed true, external_sources_read_in_full 8 (>=5), urls_collected 23
   (>=10), recency_scan_performed true. Brief first committed 9034ddfb
   (2026-08-14) BEFORE the contract's first commit d1c4a79d (2026-08-15 15:44);
   the brief's LATER mtime (16:16) is a cycle-3 annotation (39999944), not a
   post-hoc gate. Contract §1 cites the brief + envelope.
2. contract-before-generate: contract d1c4a79d 15:44 < cycle-9 artifacts
   2026-08-17 12:51. OK.
3. experiment_results present (37,875 bytes) + live_check present.
4. log-last: masterplan 86.85 status still "pending"; harness_log holds only the
   two 2026-08-15 rows (Cycle 197 FAIL, Cycle 220 CONDITIONAL) -- the in-flight
   cycle is NOT logged and the step is NOT flipped. OK.
5. no-verdict-shopping: evidence CHANGED (git diff HEAD: writer +55/-15,
   matrix +14, tests +12, both artifacts, ledger +3). Fresh-respawn on changed
   evidence = the documented cycle-2 flow.

### Scope / unintended-change check
`git diff --name-only HEAD -- '*.py'` = 8 files. 3 belong to 86.85; 4
(attempt_gate.py, mutate_rail_turn_cap.py, mutation_matrix_86_71.py,
rail_turn_cap.py) belong to 86.71/86.84; backend/api/sovereign_api.py + 5
frontend files belong to the disclosed PEER SESSION. Attribution VERIFIED, not
taken: contract_86.85.md and experiment_results_86.85.md contain ZERO mentions
of sovereign_api or frontend/. `python -c "import backend.api.sovereign_api"`
-> OK. 86.85's own diff touches no frontend file, so gate 1b is not this step's
to run; I did not run repo-wide eslint on a peer's uncommitted tree.

### Cycle-8 WARN closure audit (against the cycle-8 verdict's OWN named fixes)
- QA-C7-1 "AND ISO_DATE_RE with datetime.date.fromisoformat() at both seams":
  DONE. All SIX members of the cycle-8-measured accept-set (2026-18-10,
  2026-80-10, 2026-08-32, 2026-02-30, 2026-00-00, 9999-99-99) now refused;
  2026-08-10 still accepted. Cell M21 kills the calendar half's removal.
  (Note: 9999-99-99 appears only in a docstring, not as a fixture -- C8.9's
  "fixtures in the self-test and pytest" is over-broad by one member. NOTE-level.)
- QA-C7-2 "regenerate C8.8 ... and paste their actual output": PARTIAL -- see
  QA-C9-2 above.
- QA-C7-3 "read the rows back for step 99.4 ... assert len({r['date']}) == 3":
  DONE verbatim. Non-vacuity confirmed by matrix M5, which now FAILs that exact
  check.

### CRITERION MAP
C1 MET  | C2 MET | C3 MET | C4 MET | C5 MET | C6 MET | C7 MET
C8 NOT MET (WARN) -- "mutation-test EVERY new guard": the new guard's shape
   branch has no cell and my cell for it SURVIVED (QA-C9-1). Control GREEN
   first, sha256 identical before/after, 21/21 KILLED are all satisfied.
Plus QA-C9-2 (WARN) on artifact-claim accuracy.
Worst severity = WARN -> CONDITIONAL. No BLOCK. No criterion unaddressed.

### Prior-verdict sequence reported as EVIDENCE only
ledger status=ok; sequence FAIL -> FAIL -> FAIL -> CONDITIONAL -> CONDITIONAL ->
FAIL -> NO_VERDICT -> CONDITIONAL (NO_VERDICT carried through, not collapsed).
Every count/comparison/rollup over it is the caller's to derive.

COMPLETED: 2026-08-17T11:04:37Z  (read from `date -u`, not narrated)
