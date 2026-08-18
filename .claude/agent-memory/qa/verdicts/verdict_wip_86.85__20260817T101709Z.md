STATUS: INCOMPLETE -- not a verdict
STEP: 86.85
WRITTEN: 2026-08-17T10:17:09Z

# Q/A write-first record -- step 86.85, cycle 7 (attempt 7 per Main's disclosure, ADVISORY)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status/diff scope, ruff, pytest, self-test, matrix
C. LLM judgment against the 8 immutable criteria, adversarial mutation of the NEW guards
   (the cycle-7 fixes: sort-key excludes verdict; ISO_DATE_RE on both seams; live_check
   capture labels)

## Findings (appended as established)

### Prior-attempt evidence (gathered, NOT a trigger)
- qa_wip.py 86.85 --spawned-at 2026-08-17T10:17:09Z: attempt_number=7, prior_attempts=6,
  attempt_number_status=ok, source_present=true, records_retained=7 (GAUGE, not used).
- verdict_history_86_21.py --step 86.85 --evidence-only: status=ok, 6 verdicts,
  FAIL -> FAIL -> FAIL -> CONDITIONAL -> CONDITIONAL -> FAIL.
- CROSS-CHECK: prior_attempts (6) == ledger verdict count (6). Ledger is NOT stale for
  this step. (Note the ledger IS the artifact this step builds, so this agreement is
  partly self-produced -- stated, not used as independent corroboration.)

### B. Deterministic (all run by me, in .venv)
- IMMUTABLE COMMAND: `bash -c 'source .venv/bin/activate && python -c "import ast;
  ast.parse(open(\"scripts/qa/verdict_history_86_21.py\").read()); print(\"parses\")"'`
  -> stdout `parses`, EXIT 0. GREEN.
- self-test: `python3 scripts/qa/verdict_ledger_write.py --self-test` -> SELF-TEST PASSED,
  exit 0. Check count by the artifact's OWN stated population rule
  (`grep -cE '^  (ok  |FAIL)'`) = **29**. FAIL lines = 0.
- pytest: `python -m pytest backend/tests -k '86_85 or ledger or verdict_ledger' -q`
  -> **38 passed, 3514 deselected**, exit 0. MATCHES the claim.
- matrix: `python3 scripts/qa/mutation_matrix_86_85.py` -> **20 cells: 20 killed,
  0 survived, 0 unscorable**, exit 0; sha256 before==after
  (0d034aae9c37deb4d19d87638b0e7d2b7742d2f11c8529ebfd1ea65d0ebfcb1e), UNCHANGED True.
  MATCHES the claim.
- coverage: `python3 scripts/qa/verify_matrix_coverage_86_85.py` ->
  **guards: 21   covered: 21   uncovered: 0   cell problems: 0**, exit 0.
- emit-sequence: ["FAIL","FAIL","FAIL","CONDITIONAL","CONDITIONAL","FAIL"] -- matches.

### FINDING QA-C7-1 (claim defect, 4th recurrence of the same class)
`live_check_86.85.md` C8.8 is titled "CURRENT captured run (cycle 7, 2026-08-17)" and
says "Every figure below is from a live run this session; re-derive with the command."
Two of its five fenced figures DO NOT REPRODUCE in the same working tree:
  - claims `(exit 0; 30 checks ...)`; measured **29** under the artifact's own rule
    (live_check:485-489). HEAD version of the writer emits 23, so 30 was never true
    at any commit I can reach.
  - claims verbatim line `guards: 17   covered: 17   uncovered: 0   cell problems: 0`;
    the tool emits `guards: 21   covered: 21`. 17 is the CYCLE-6 figure carried
    forward (HEAD live_check records 15/15 at cycle 4); cycle 7 ADDED the two ISO
    guards, which the enumeration counts, so the number necessarily rose.
`experiment_results_86.85.md:697-698` repeats both ("self-test PASSED (30 checks) ...
coverage 17/17") under the label "Verbatim, post-change".
This is the exact class the cycle-6 Q/A flagged (QA-C6-3, "stale live_check, third
recurrence") and which cycle 7's own remedy claims to close with capture labels. The
label was added; the number was not re-derived.
