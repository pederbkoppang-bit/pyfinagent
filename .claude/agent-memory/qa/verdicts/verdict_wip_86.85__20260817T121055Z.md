STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.85
WRITTEN: 2026-08-17T12:10:55Z

# Q/A cycle-12 write-first record for step 86.85

Launch: Workflow rail, agentType qa. Prior verdict from me (cycle 11) was FAIL with
two findings QA-C11-A (vacuous FILTER-axis guards) and QA-C11-B (stale claims in
live_check §9 items 3-4).

## Prior-attempt evidence (gathered, NOT used as a trigger)
- `qa_wip.py 86.85 --spawned-at 2026-08-17T12:10:55Z`: source_present=true,
  attempt_number=12 (status ok, is_lower_bound=true), prior_attempts=11,
  records_retained=12 (gauge), records_pruned_known=null.
- `verdict_history_86_21.py --step 86.85 --evidence-only`: status=ok, 11 rows:
  FAIL -> FAIL -> FAIL -> CONDITIONAL -> CONDITIONAL -> FAIL -> NO_VERDICT ->
  CONDITIONAL -> CONDITIONAL -> CONDITIONAL -> FAIL
- Cross-check: prior_attempts (11) == ledger rows (11). Ledger NOT stale.
- NOTE: the spawn prompt's ADDITIONAL CONTEXT asserts "FIFTH counted attempt".
  That does not reconcile with attempt_number=12 / 11 ledger rows. Recorded as an
  observation; it is caller-side bookkeeping and not mine to apply.

## B. Deterministic (all re-derived by me)
- IMMUTABLE COMMAND `python -c "import ast; ast.parse(open('scripts/qa/verdict_history_86_21.py').read()); print('parses')"`
  -> stdout `parses`, EXIT=0. MET.
- shasum -a 256 scripts/qa/verdict_ledger_write.py ->
  0cc08f20b32e6229f2b21c23920566d215e1270db0d66b308eed9abe6b8c5bde
  == the sha claimed in experiment_results C12 and live_check C12. REPRODUCES.
- writer --self-test: ST_EXIT=0, `grep -c "^  ok "` = 34. REPRODUCES the claim.
- pytest -k "86_85 or ledger or verdict_ledger": PT_EXIT=0, "38 passed, 3514
  deselected, 1 warning in 7.10s". REPRODUCES (artifact said 7.20s -- timing only).
- mutation_matrix_86_85.py: MX_EXIT=0, CONTROL rc=0 -> GREEN observed FIRST,
  "24 cells: 24 killed, 0 survived, 0 unscorable", sha256 before == after ==
  0cc08f20..., UNCHANGED: True. guards 21 covered 21 uncovered 0. REPRODUCES.
- M23 / M24 present verbatim as permanent cells, both KILLED (rc=1). REPRODUCES.

## C. MY OWN MUTATIONS of the NEW cycle-12 guards (in-memory exec, zero repo writes)

CONTROL (unmutated source, exec'd in-memory): rc=0, SELF-TEST PASSED. Observed FIRST.

- **D2 = MUT-A (filter -> startswith) against the SHIPPED fixture**: rc=1,
  `FAIL  sequence filters by EXACT step id -- extension not swept in`.
  => the shipped fixture IS prefix-related and the two DIRECTION checks are
  genuinely behavioural. The cycle-11 primary finding IS fixed at the site.

- **D1 = drift the self-test fixture back to a prefix-UNRELATED pair
  (append "99.2", reverse-query "99.2"), production filter UNTOUCHED**:
  rc=0, and the check named
  `ok    filter fixture is prefix-related (anti-vacuity for the filter axis)`
  STILL REPORTS ok. The check cannot fail when its subject is broken.

- **D3 = D1 fixture drift + MUT-A(startswith)**: rc=0, SELF-TEST PASSED, all
  three filter-axis checks green. **The exact cycle-11 vacuous state is fully
  restorable with every guard green**, including the guard added this cycle to
  prevent precisely that.

ROOT CAUSE (source, verdict_ledger_write.py:558-559):

    check("filter fixture is prefix-related (anti-vacuity for the filter axis)",
          "99.40".startswith("99.4") and "99.40" != "99.4")

The predicate is over two STRING LITERALS. It references no fixture, no row on
disk, no variable. Constant-true. Same in pytest
(test_phase_86_85_verdict_ledger_write.py, test_sequence_filters_by_step):

    assert "4.10".startswith("4.1") and "4.10" != "4.1", "fixture must be prefix-related"

This is vacuity shape #4 (tautology) from qa.md 4c, and it is the SAME shape the
cycle-8 Q/A already caught on the DATE axis in this very file -- fixed in cycle 9
by deriving from the ROWS ON DISK (line 456-457):

    check("order fixture carries distinct event dates (anti-vacuity for the date axis)",
          len({r["date"] for r in read_rows(p) if r.get("step_id") == "99.4"}) == 3)

The ORDER-axis sibling (line 452-453) is also real -- it reads `ordered`, which is
derived from the fixture. The cycle-12 FILTER-axis check is the only one of the
three that is a literal tautology. The correct form is ~100 lines above it.

### AST/empty-namespace proof (executed, both copies)
    self-test 558-559:  free names=[]  value_in_EMPTY_namespace=True
    pytest  (line 210): free names=[]  value_in_EMPTY_namespace=True
Zero free names, constant True with `__builtins__` stripped. No program state can
falsify either. (Both expressions asserted present VERBATIM in their source first.)

### The pytest oracle behaves identically (executed)
    CONTROL unmutated                                  : PASSED
    shipped fixture + MUT-A startswith                 : AssertionError -> KILLED
    shipped fixture + MUT-B containment                : AssertionError -> KILLED
    drift fixture(4.1/4.2) only                        : PASSED -> SURVIVED
    drift + MUT-A                                      : PASSED -> SURVIVED
    drift + MUT-B                                      : PASSED -> SURVIVED
So on BOTH oracles the cycle-11 state is restorable with every check green.

### The other new guards ARE real (so this is WARN-shaped, not sole-coverage)
- MUT-C (reverse sweep: `if not step_id.startswith(row_step)`) -> rc=1, kills ONLY
  `reverse direction: extension query does not sweep its prefix`. That check adds
  independent signal (MUT-A does not kill it). NOT vacuous.
- MUT-D (filter deleted) -> rc=1, 6 checks red.
- Shipped matrix M23/M24 reproduce KILLED under my own run.
=> the filter AXIS has genuine behavioural coverage; only the meta-guard on the
   FIXTURE is vacuous. Per qa.md 4c that is WARN, not BLOCK.

## Criteria 1-8, independently driven

- **C1 LOCALISED -- MET.** I re-derived the pre-step state myself:
  `git show 'd1c4a79d~1:handoff/verdict_ledger.jsonl'` -> total 35, 86.74 rows 0,
  distinct step_ids 10, max date 2026-08-11, 86.21 rows 5. Every figure in
  contract §2 / live_check §1 reproduces EXACTLY. The 86.21 positive control is
  what licenses reading 86.74's zero as measured. Cause = NEVER-WRITTEN, and the
  step correctly declined to re-scope (WIP records + critique prose are not a
  machine-readable sequence; `qa_wip.py` itself says a WIP is never a verdict).
- **C2 population rule -- MET with a residual (see QA-C12-C).** The C2 section
  states "Population = EVERY non-blank line in handoff/verdict_ledger.jsonl" and
  quotes the enumeration command. Both commit-anchored figures reproduce under my
  own run: `d1c4a79d~1`=35, `d1c4a79d`=43.
- **C3 cross-process -- MET, driven by ME.** 3 separate `python` writer processes
  wrote to a tmp ledger; a 4th separate invocation read back
  `["CONDITIONAL","CONDITIONAL","CONDITIONAL"]`, rc=0.
- **C4 3rd-CONDITIONAL DRIVEN -- MET, driven by ME.** Extracted the SHIPPED
  `enforceEscalation` from `.claude/workflows/qa-verdict.js` by regex and ran it
  unmodified under node against my ledger-emitted sequence:
    1 prior C + CONDITIONAL -> n=1 auto_fail=false   (anti-vacuity control)
    2 prior C + CONDITIONAL -> n=2 auto_fail=true
    3 prior C + CONDITIONAL -> n=3 auto_fail=true
    [C,C,PASS] + CONDITIONAL -> n=0 auto_fail=false  (reset control)
- **C5 86.79/86.45 interaction -- MET.** Resolved in writing at
  experiment_results §C5. Verified rather than read: the writer never references
  `records_retained` (grep count 0), and NO_VERDICT `continue`s in the consumer.
- **C6 drop must not clear; absence != zero -- MET, driven by ME.**
    [C,C,NO_VERDICT] + CONDITIONAL -> n=2  auto_fail=true   (drop does NOT reset)
    absent / null sequence          -> n=null auto_fail=null  status=not_supplied
    ["NOPE"]                        -> n=null auto_fail=null  status=unparseable
  Fails closed; never 0.
- **C7 verdict semantics UNCHANGED under every flag combination -- MET, driven by
  ME.** 4 verdicts x 9 sequences x 6 opts = 216 combinations: the verdict object
  is never mutated, and the escalation object never carries a `verdict`/`ok` key.
- **C8 mutation-test EVERY new guard -- NOT MET.** Control observed GREEN first
  (yes) and byte-identical restore (yes: sha before == after == 0cc08f20..., and
  my own mutations were in-memory; repo shas unchanged at the end). But two NEW
  guards shipped this cycle admit NO failing mutation at all -- see QA-C12-A.

## QA-C12-B -- the cycle-12 completeness claim does not reproduce
Claim (experiment_results:832-834): "Class sweep across all six artifacts of
86.85+86.71: every remaining match is a quotation INSIDE a correction block."
My own sweep of the six files finds `experiment_results_86.85.md:266` leading with
the BOLD assertion **"Only one consumer is proven."** -- an active lead sentence,
not a quotation, contradicted by its own body two lines later ("attempt_budget.py
(86.71) went LIVE mid-drain"). The live_check sibling (§9 item 3) DID replace its
lead ("**Two consumers are live**"); the experiment_results copy did not. A
correction that ACCOMPANIES its stale headline instead of REPLACING it is the same
class QA-C10-1 and QA-C11-B were raised for.
(§9 items 1-2 re-verification CONFIRMED true by me: no automatic caller of
`verdict_ledger_write` exists at the verdict seam -- the only importers are
attempt_gate.py (a READER of emit_sequence), the tests, the matrix and the coverage
checker; and all 8 86.74 rows carry a `backfilled ... reconstructed from` note.)

## QA-C12-C -- one bare, stale, unanchored ledger-row count survives
`experiment_results_86.85.md:29` (files-touched table): "Now 45 in the working
tree, the +2 being this step's own cycle-1 and cycle-2 FAIL rows." No population
rule beside it, no enumeration command, and stale: the tree now holds 68 rows of
which 11 are 86.85's. NOTE/WARN-shaped; C2's dedicated section discharges the
substance.

## Harness compliance (5 items) -- CLEAN
- research gate: brief_status COMPLETE, gate_passed true, 8 read in full, 23 URLs,
  recency_scan true. Brief first committed 2026-08-14 (9034ddfb), contract first
  committed 2026-08-15 15:44 (d1c4a79d) -> research BEFORE contract.
- experiment_results present (45,090 bytes) and carries the cycle-12 GENERATE.
- log-last: masterplan 86.85 status=pending; no cycle-12 row in harness_log.
- no verdict-shopping: evidence CHANGED since cycle 11 -- writer sha
  9ade917c -> 0cc08f20, self-test 32 -> 34 checks, matrix 22 -> 24 cells,
  live_check §9 items 3-4 replaced. Verified by `git show 2a6cd4b6`.
- ruff F821/F401/F811 over the DERIVED cycle-12 commit scope (5 .py files,
  non-empty set asserted): "All checks passed!", exit 0.

## Deterministic reproduction summary
immutable cmd exit=0 | self-test 34 ok / exit 0 | pytest 38 passed / exit 0 |
matrix 24 cells 24 killed 0 survived, control GREEN first, sha UNCHANGED True,
guards 21 covered 21 | ruff exit 0. ALL artifact claims reproduce.

## Environment notes
- HEAD moved DURING this evaluation: ca800b50 -> 3f814080 (86.84 cycle 11). Diff
  ca800b50..HEAD touches NO 86.85 artifact or source -- only the shared
  handoff/verdict_ledger.jsonl (86.84's own rows). 86.85 shas identical at start
  and end; `git status --short` on the three sources is empty.
- MY OWN probe defect, recorded: a first `for c in ...; git show "$c:handoff/..."`
  loop returned rows=0 for EVERY commit. zsh consumed `$c:handoff` as a parameter
  modifier. Corrected to `${c}:handoff/...` and the figures then reproduced. A red
  check that indicts its own probe is as misleading as a green one that cannot fail.
- No UI claims in this step -> qa.md §1c not applicable. §1d discharged by driving
  the writer CLI live in a temp dir across four separate processes.

## VERDICT (returned via StructuredOutput): CONDITIONAL
Capped on criterion 8 (QA-C12-A), with QA-C12-B/C as WARN-level prose findings.
Criteria 1,3,4,5,6,7 MET and independently driven; criterion 2 MET with a residual.

COMPLETED: 2026-08-17T12:20:56Z
