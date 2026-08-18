STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.85
WRITTEN: 2026-08-17T10:33:11Z

# Q/A write-first record -- step 86.85, cycle 7 respawn

Spawned as the Layer-3 Q/A evaluator for masterplan step 86.85 (EVALUATE).
Prompt states: cycle 7 RESPAWN after wf_2fafe515-6a2 died on API 529 mid-eval
(NO VERDICT). Evidence claimed UNCHANGED since that spawn; cycle-6 FAIL's three
findings claimed closed in code.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status/diff scope, lint, scoped tests
C. Mutation / guard-vacuity work on the NEW guards
D. Criterion-by-criterion MET/NOT MET

## Findings log (appended as established)

### Prior-attempt / prior-verdict EVIDENCE (gathered, not applied)
- `qa_wip.py 86.85 --spawned-at 2026-08-17T10:33:11Z`:
  `source_present: true`, `attempt_number_status: ok`, `attempt_number: 8`,
  `prior_attempts: 7`, `records_retained: 8` (GAUGE, not a counter),
  `attempt_number_is_lower_bound: true`, `records_pruned_known: null`.
- `verdict_history_86_21.py --step 86.85 --evidence-only`: `status: ok`,
  7 rows, sequence
  `FAIL -> FAIL -> FAIL -> CONDITIONAL -> CONDITIONAL -> FAIL -> NO_VERDICT`.
- CROSS-CHECK: prior_attempts (7) == ledger rows (7). attempt_number (8) is
  inclusive of THIS spawn, whose verdict is not yet written, so the ">" is
  expected and the ledger is NOT stale for this step. Carrying the NO_VERDICT
  row through as-is per qa.md.
- NOTE: the masterplan `notes` field for 86.85 says the step is "PARKED at
  [CONDITIONAL x4]"; the ledger says F,F,F,C,C,F,NV. The masterplan prose
  disagrees with the ledger; the ledger governs (qa.md).

### B. Deterministic
- IMMUTABLE COMMAND: `bash -c 'source .venv/bin/activate && python -c "import
  ast; ast.parse(open(\"scripts/qa/verdict_history_86_21.py\").read());
  print(\"parses\")"'` -> stdout `parses`, **EXIT=0**. GREEN.
- self-test: `python3 scripts/qa/verdict_ledger_write.py --self-test` ->
  **EXIT=0**, `SELF-TEST PASSED`, **29 ok lines, 0 FAIL lines**.
  *** experiment_results Cycle-7 "Verbatim, post-change" claims "(30 checks)".
      MEASURED 29. Claim does not reproduce. ***
- pytest `backend/tests -k '86_85 or ledger or verdict_ledger' -q` ->
  **38 passed, 3514 deselected, EXIT=0**. MATCHES the claim.
- mutation matrix `python3 scripts/qa/mutation_matrix_86_85.py` -> EXIT=0,
  **20 cells: 20 killed, 0 survived, 0 unscorable**; sha256 before == after
  (`0d034aae...cb1e`), `UNCHANGED: True`. MATCHES the claim.
- coverage `python3 scripts/qa/verify_matrix_coverage_86_85.py` -> EXIT=0,
  **guards: 21  covered: 21  uncovered: 0  cell problems: 0**.
  *** experiment_results Cycle-7 claims "coverage 17/17". MEASURED 21/21.
      Claim does not reproduce. ***
- `enforceEscalation` is at `.claude/workflows/qa-verdict.js:535`, NOT
  "lines 319-370" as experiment_results §C4 states. The file's only export is
  `export const meta` (line 1) -- enforceEscalation is NOT exported, so any
  "extracted and executed unmodified" claim needs to be checked for
  re-implementation vs real slice-and-exec. NOTE ONLY: the cycle-6 Q/A already
  established the citation was accurate at d1c4a79d and went stale via other
  steps' edits; the quoted OUTPUT reproduces (see below).
- ruff `F821,F401,F811` over the scope DERIVED from the graded commit
  (`git show --name-only --format='' f3c89229 | grep -E '\.py$'` -> 3 files,
  non-empty set asserted, passed via xargs): **"All checks passed!" exit=0**.
- No frontend/** in scope -> gate 1b N/A. No UI claims -> gate 1c N/A. Only
  `backend/tests/**` under backend -> runtime smoke discharged by the scoped
  pytest run, which imports and executes the writer.

### TREE MOVED DURING EVALUATION (recorded, not a finding)
HEAD was `8000de69` at 10:34Z and `cadab378` by 10:41Z (peer/Main commits
landed mid-eval). The 86.85 work is now COMMITTED as **f3c89229**
("phase-86.85: cycles 5-7 ..."). `git diff HEAD` is EMPTY for every 86.85
file, and `shasum -a 256 scripts/qa/verdict_ledger_write.py` =
`0d034aae9c37deb4d19d87638b0e7d2b7742d2f11c8529ebfd1ea65d0ebfcb1e`, identical
to the sha the shipped matrix printed before/after its run -- so everything I
measured IS what is committed. f3c89229 touches exactly 7 files (3 .py + 3
handoff artifacts + the ledger); NO production/trading code, no frontend, no
masterplan flip. masterplan status for 86.85 = `pending`.

### HARNESS COMPLIANCE (5 items) -- CLEAN
1. Research gate: `research_brief_86.85.md` envelope `brief_status: COMPLETE`,
   `gate_passed: true`, `external_sources_read_in_full: 8` (>=5),
   `urls_collected: 23` (>=10), `recency_scan_performed: true`, 2 "Recency
   scan" sections. Brief ADDED 2026-08-14 21:41 (9034ddfb) BEFORE contract
   ADDED 2026-08-15 15:44 (d1c4a79d) -- git add-order, not mtime.
2. Contract before generate: contract present, cites the brief (3 refs).
3. experiment_results present (36,711 B) with a Cycle-7 GENERATE section.
4. Log-last: masterplan `pending`; `grep -F 'phase=86.85' handoff/harness_log.md`
   -> 2 rows (cycles 3 and 4), NO in-flight cycle-7 row. Correct.
5. No verdict-shopping: code changed 10:13-10:16Z BETWEEN the cycle-6 spawn
   (10:00:28Z) and the cycle-7 spawn (10:17:09Z). Since the cycle-7 DROP the
   only change is the ledger's own NO_VERDICT row. A respawn after a drop is
   the documented recovery (a drop is NO VERDICT), not a second opinion.

### C1 LOCALISATION -- reproduces
- positive control `verdict_history_86_21.py --step 86.21 --evidence-only`
  -> `status: ok`, 5 verdicts. Same reader/key/file. Licenses the 86.74 zero.

### C2 ANCHORED LEDGER COUNTS -- ALL REPRODUCE EXACTLY
Population = every non-blank line of handoff/verdict_ledger.jsonl.
- `git show d1c4a79d~1:handoff/verdict_ledger.jsonl` -> total **35**,
  step_ids **10**, 86.74 rows **0**, recorded_at 21, verdicts
  {C 18, F 5, P 7, NV 5}, max date **2026-08-11**, recorded_by {main}.
  run_id: key present **35**, non-empty **35**, wf_-prefixed **35**.  MATCHES.
- `git show d1c4a79d:...` -> total **43**, step_ids **11**, 86.74 rows **8**,
  recorded_at **29**, verdicts {C 23, F 5, P 8, NV 7}.  MATCHES.
- [WORKING TREE] "52" is now 56 -- but that figure is explicitly labelled
  perishable with its date, command and reason. Disclosed drift, NOT a defect.
- Non-ISO census re-derived BY MEMBER not cardinality: **11** rows,
  `{36.17: 6, 86.20: 3, 86.17: 2}`, single distinct value `2026-08-09/10`.
  MATCHES the claim exactly.

### C3/C4/C6/C7 -- DRIVEN through the REAL sliced enforceEscalation
Extracted by brace-matching from `function enforceEscalation` in
`.claude/workflows/qa-verdict.js` into a temp module (the shipped 86.78
checker's own technique; my first attempt failed because `opts = {}`
balanced the brace counter -- recorded, not quietly re-run).
    CONTROL correct-date backfill  [PASS,C,C]+COND  n=2  auto_fail=true
    CONTROL 2 priors + CONDITIONAL [C,C]+COND       n=2  auto_fail=true
    CONTROL 1 prior  + CONDITIONAL [C]+COND         n=1  auto_fail=false  <-- does not fire early
    SEMANTICS 2 priors + PASS                       n=2  auto_fail=false
    SEMANTICS 2 priors + FAIL                       n=2  auto_fail=false
    DROP [C,C,NO_VERDICT] + COND                    n=2  auto_fail=true   <-- drop does NOT clear
    ABSENT sequence                                 n=null status=not_supplied  <-- null, not 0
`if (v === 'NO_VERDICT') continue` verified in source at qa-verdict.js:535ff.
C3 cross-process read-back re-driven by me: proc1 writes, proc2 --emit-sequence.
Shipped 86.78 checker `node scripts/qa/verify_escalation_86_78.mjs` -> 51/51 PASS.

### MY OWN MUTATION WORK (control GREEN first; sha256 0d034aae..cb1e before
### AND after every cell; UNCHANGED True; zero repo writes -- temp copies only)
    CONTROL                       GREEN
    QA-M-ISO-ANCHOR (drop \A \Z)  KILLED  (non-ISO stored date check)
    QA-M-KEY-POSONLY  (t[1],)     KILLED  (backfill event-order check)
    QA-M-KEY-VERDICT-LAST         KILLED  (same-date file-order check)
    QA-M-KEY-DATEONLY (t[0],)     SURVIVED -- EQUIVALENT (stable sort)
    cycle-6 QA-M-POS-const pos->0 SURVIVED -- **EQUIVALENT, PROVEN**: byte-identical
        output vs baseline on 86.85 / 86.74 / 86.21 / 36.17. The cycle-6 blocker
        is genuinely CLOSED; M17 replaces it and kills with a real differential
        on all three readable steps. CREDIT.
    QA-M-FIXTURE-SAMEDATE         **SURVIVED** -- see F3.
    Composite (both fixtures collapsed + M15) KILLED -> F3 opens no live hole.

## FINDINGS

### F1 [WARN] QA-C7-1 -- the ISO guard validates SHAPE, not a real date;
### the escalation-clearing backfill is STILL reachable.
Driven on a temp ledger:
  seed C(2026-08-11), C(2026-08-12) -> ["CONDITIONAL","CONDITIONAL"]
  backfill an older PASS `--date 2026-18-10`  -> **ACCEPTED, exit 0, no warning**
  --emit-sequence -> ["CONDITIONAL","CONDITIONAL","PASS"]   n=0  auto_fail=FALSE
  control, same backfill `--date 2026-08-10`  -> ["PASS","CONDITIONAL","CONDITIONAL"]
                                                            n=2  auto_fail=TRUE
Accept-set measured: 2026-18-10 / 2026-80-10 / 2026-08-32 / 2026-02-30 /
2026-00-00 / 9999-99-99 are ALL accepted by ISO_DATE_RE; `date.fromisoformat`
refuses all six and accepts 2026-08-10. `sorted(['2026-18-10','2026-08-12'])`
puts the typo LAST -- the escalation-clearing direction.
Same class + same fail-open direction as cycle-5 QA-MUT-B and cycle-6 QA-C6-1.
The self-test check NAME "(a backfill cannot clear an escalation)" and the
emit_sequence docstring both still OVERCLAIM. Mitigating: the author
implemented literally the regex the cycle-6 Q/A named. Fix: AND the regex with
`datetime.date.fromisoformat()` (datetime already imported).

### F2 [WARN] QA-C7-2 -- two figures in the block that asserts it is live do
### not reproduce. FOURTH recurrence in this same file.
`live_check_86.85.md` C8.8 is headed "CURRENT captured run (cycle 7)" and says
"**Every figure below is from a live run this session; re-derive with the
command.**" Re-derived with the exact quoted commands:
  claimed "SELF-TEST PASSED (exit 0; 30 checks ...)"  -> MEASURED **29**
      using the artifact's OWN stated population rule (C8.6:
      `grep -cE '^  (ok  |FAIL)'`); 29 ok, 0 FAIL, exit 0.
  claimed "guards: 17   covered: 17"                  -> MEASURED **21   21**
      (deterministic over two consecutive runs)
  reproduced OK: pytest 38 passed/3514 deselected; matrix 20 cells 20 killed
      0 survived 0 unscorable; ruff clean; immutable command parses exit 0.
  emit-sequence [F,F,F,C,C,F] is now [F,F,F,C,C,F,NO_VERDICT] -- ledger moved
      after the capture; NOT counted against the artifact.
`experiment_results_86.85.md` Cycle-7 repeats both wrong figures ("self-test
PASSED (30 checks) ... coverage 17/17"). Cycle 2 was FAILed for this defect,
cycle 3 caught it, cycle 6 FAILed for it as the "third recurrence" -- and the
cycle-7 remediation FOR it ships two instances of it.

### F3 [WARN] QA-C7-3 -- the "anti-vacuity for the date axis" check is a
### TAUTOLOGY (qa.md 4c shape #4/#6).
`verdict_ledger_write.py`:
    check("order fixture carries distinct event dates (anti-vacuity for the date axis)",
          len({f"2026-08-1{i}" for i in range(3)}) == 3)
It asserts the cardinality of a locally-built literal set and NEVER references
the fixture rows it names -- true by construction for every fixture state.
EXECUTED: collapsing the 99.4 fixture to a single `event_date="2026-08-10"`
SURVIVES with all checks green, including that one.
HONEST BOUND: the 99.7 backfill fixture independently covers the date axis --
collapsing 99.7 is KILLED, and the composite (both collapsed + M15) is KILLED.
So this is a vacuous guard ALONGSIDE a genuine behavioural guard -> WARN, not
BLOCK, per qa.md 4c verdict wiring. Fix: derive the asserted set from the rows
actually appended (or read them back from the temp ledger for step 99.4).

## CRITERION MAP
C1 MET  C2 MET  C3 MET  C4 MET  C5 MET  C6 MET  C7 MET  C8 MET-with-F3
(every criterion MET as literally worded and independently re-derived;
F1/F2/F3 are WARN-severity findings, worst severity WARN -> CONDITIONAL)

Worst-of-N lenses: correctness=CONDITIONAL (F1 residual);
does-it-reproduce=CONDITIONAL (F2); scope-honesty=CONDITIONAL (C8.8 heading
overclaims, check name overclaims -- though §4/§9 HONEST LIMITS are unusually
candid). min = CONDITIONAL.

Sycophancy check: cycle 6 was FAIL, I return CONDITIONAL -- the code DID change
(f3c89229) and I verified the cycle-6 blocker closed BY EXECUTION. Documented
cycle-2 flow, not a flip on unchanged evidence.

Discrepancy recorded: masterplan step 86.85 `notes` says "PARKED at
[CONDITIONAL x4]"; the ledger says F,F,F,C,C,F,NV. The ledger governs.

COMPLETED: 2026-08-17T10:47:00Z

