STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 90.1
WRITTEN: 2026-08-21T06:44:20Z
COMPLETED: 2026-08-21T06:56:21Z

# Q/A write-first record -- step 90.1, cycle 5

Spawned via Workflow rail. Prior verdicts reported by Main (ADVISORY):
FAIL (c1), CONDITIONAL (c2), CONDITIONAL (c3), no c4 spawn.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command exit code + git status + lint + smoke
C. Criteria 1-6 MET/NOT MET with cited evidence
D. Independent mutation cells (fixture/harness shapes)

## Findings (appended as established)

### Attempt / verdict evidence
- qa_wip.py 90.1 --spawned-at 2026-08-21T06:44:20Z: source_present=true,
  attempt_number=4 (INCLUSIVE, is_lower_bound=true, status=ok), prior_attempts=3,
  records_retained=4 (GAUGE), records_pruned_known=null.
- verdict_history_86_21.py --step 90.1 --evidence-only: status=no_rows_for_step,
  verdicts=(none). CROSS-CHECK: prior_attempts(3) > ledger rows(0) => LEDGER IS
  STALE for this step. sequence: UNKNOWN from the authoritative source.
  Main's advisory disclosure (FAIL, CONDITIONAL, CONDITIONAL) matches the count 3.

### B. Deterministic
- IMMUTABLE COMMAND, run bare (no pipe):
  `python3 scripts/harness/attempt_gate.py --self-test` -> exit 0
  `python3 scripts/qa/mutation_matrix_90_1.py --verify`  -> exit 0
  Matrix: KILLED 15 | SURVIVED 0 (excl N0) | ERROR 0 | null survived: True.
  Control green across all 25 named checks before any cell.
- verdict_ledger.jsonl sha256 BEFORE == AFTER whole run:
  fcfe56ad9788f0bc248253aea49e086812ab951c4145ecc5eac2b92c982e3eb2 (criterion 6, my own derivation)
- 90.1 success_criteria + verification.command hash CONSTANT across all 11 commits
  in which step 90.1 exists (sha256[:16]=f98626019b331382; cmd=e5ae167bc5ca).
  NO criteria erosion.
- masterplan status 90.1 = pending (NOT flipped). harness_log has ONE row:
  "Cycle 194 -- 2026-08-20 -- phase=90.1 result=CONDITIONAL (loop stopped on
  operator instruction)" -- records the prior cycle, does NOT claim PASS.

### BLOCKING FINDING -- criterion 5 clause 3, FOURTH relocation (Q/A-authored cell QA1)
The cycle-4 fix scores ERROR by scanning drive stderr for the literal
"Traceback (most recent call last)" and then typing the LAST line against
UNRESOLVABLE_ERRORS. `handle_hook` has its own blanket `except Exception`
(attempt_gate.py:468) that catches a NameError raised anywhere inside its try
block, prints a ONE-LINE "[attempt-gate] INTERNAL ERROR -- ... failing OPEN"
message, and returns 0. No traceback is ever emitted, so the discriminator
cannot see it.

Cell QA1 (authored by this Q/A, run through the matrix's own `run_cell`):
  anchor (count=1 in attempt_gate.py):
    "        rows = read_ledger()\n        decision, state = decide(sid, rows)"
  mutation: read_ledger() -> read_ledger_v2()
  SCORE: KILLED   (must be ERROR)
  _drive_traceback(obs) -> None
  every drive stderr, verbatim:
    [attempt-gate] INTERNAL ERROR -- NameError: name 'read_ledger_v2' is not
    defined -- failing OPEN (the launch proceeds UNCOUNTED; ...)
  traceback_present = False on ALL of below/at/unknown/over/under/nested.
This is the same class as QX2 (handle_hook -> handle_hook_v2), one seam inward:
a call-site rename instead of a definition rename. NameError is in the module's
OWN UNRESOLVABLE_ERRORS tuple, so by the shipped code's own definition this
mutant "cannot run" and must score ERROR. It scores KILLED.

CLASS, not a one-off. Three independent call sites, all anchor_count=1:
  QX2repro  def handle_hook  -> def handle_hook_v2        -> ERROR  (fix works)
  QA1       read_ledger()    -> read_ledger_v2()          -> KILLED (must be ERROR)
  QA1b      extract_step_id_claim() -> ..._claim_v2()     -> KILLED (must be ERROR)
  QA1c      extract_step_id()       -> extract_step_id_v2()-> KILLED (must be ERROR)
Boundary is exactly handle_hook's own `except Exception` (attempt_gate.py:468):
name errors OUTSIDE it emit a traceback and ARE caught; name errors INSIDE it
are swallowed into a one-line message and are MISSED. That region is the whole
executable body of the gate's hook path, so the guard is vacuous for the
dominant sub-class. The stated discriminator ("a mutant that cannot run dies
with a name-resolution exception") is strictly WIDER than the implemented one
("...dies with an UNHANDLED TRACEBACK whose last line is a name-resolution
exception"). The gap between the two IS the defect.

### Criteria independently reproduced (MET)
- C1 MET. My own walk of the live ledger via attempt_outcomes.resolve_row over
  all 106 attempt rows: {CONDITIONAL 50, NO_VERDICT 25, PASS 14, FAIL 11,
  UNKNOWN 6}. Vocabulary is EXACTLY the 5 permitted values. All 6 UNKNOWNs are
  outcome_reason=no_run_record: the 5 synthetic 999.2 pipetest rows + THIS
  spawn's own in-flight row (2026-08-21T06:44:15Z, run record not yet written).
  `--backfill --dry-run` exits 0, prints per-value counts + the UNKNOWN count,
  and md5 of the ledger is UNCHANGED across two dry runs. Population drift
  (criterion says 92; ledger held 93/89 at cycle time, holds 110/106 now) is
  DISCLOSED by Main, not quietly reconciled.
- C2 MET, and the sha256 guard PROVEN NON-VACUOUS by a Q/A-authored cell.
  QA2: write_escalation also writes
  escalation_attempt_budget_<step_id.rsplit('.',1)[0]>.md (a "roll a sub-id up
  to the parent" slip). Result: exactly ONE check fails --
  "a NON-exhaustion denial leaves a pre-existing exhaustion escalation
  BYTE-IDENTICAL (c2, sha256 before == after)"; sha_before 34280ec1d146 !=
  sha_after c0a410e05ddd. Sole attribution, no other check fired. The guard is
  live.
- C3 MET, and stronger than the shipped fixture. My own drive of the real hook:
    0 rows                    -> ALLOW
    1 row 1,199,999           -> ALLOW
    1 row 1,200,000           -> DENY   (rule is >=)
    1 row 1,200,001           -> DENY   [the criterion's named fixture]
    3 rows summing 1,199,997  -> ALLOW
    3 rows summing 1,200,000  -> DENY   (SUMMED, not max-of-row)
    3 rows summing 1,200,003  -> DENY
  Escalation body now prints "tokens used : 1,200,001 / 1,200,000" -- the
  constant-0 defect is genuinely closed.
  NOTE (not a criterion miss): a TOKEN-ceiling denial still prints the ATTEMPT
  wording on stderr ("step 9.1 has used 1/5 attempts") and offers
  `--operator-extend --by 1`, which raises max_attempts only and therefore
  cannot lift a token denial. The FILE body is honest; the stderr remedy is not.
- C4 MET. Driven against the real module, not a fixture:
    '86.118'->'86.118' ADMITTED; '86.118.1'->None DENIED; '86.1180'->None
    DENIED; '999.99'->None DENIED. Also '999.2' DENIED, '086.118' DENIED,
    no-step_id-at-all still allowed+uncounted (the deliberate escape hatch).
    Self-test ids exempted BY CONSTRUCTION via ATTEMPT_GATE_MASTERPLAN pointing
    at a synthetic plan -- explicit exemption, not a silent pass.
- C6 MET. handoff/verdict_ledger.jsonl sha256
  fcfe56ad9788f0bc248253aea49e086812ab951c4145ecc5eac2b92c982e3eb2 identical
  before and after the full immutable command AND after every mutation cell I
  ran. Only VERDICT_LEDGER write in the changed files is attempt_gate.py:597,
  inside _self_test, behind the cycle-4 containment guard.
  CONTAINMENT GUARD RED-FIRST, with a null control (my first attempt was
  CONFOUNDED -- a flat tempdir copy makes REPO=Path(__file__).parents[2] resolve
  to the tempdir root, so the NULL scored rc=1 too; measuring relocation, not
  the guard). Re-run nested with a null control, discriminating on the guard's
  own message rather than rc: NULL -> containment_line False; MUTANT (redirect
  line deleted) -> containment_line True, "VERDICT_LEDGER still points INSIDE
  the repo". Guard is real.

### Harness compliance (5 items)
1. Research gate BEFORE contract: PASS. brief_status COMPLETE, gate_passed
   true, external_sources_read_in_full 10 (>=5), 26 unique URLs (>=10),
   "Recency scan (2024-2026) -- performed" present. mtimes: brief 21:12 <
   contract 21:15 < artifacts 21:45-22:33 (local tz).
2. Contract BEFORE generate: PASS, and all SIX immutable criteria are present
   VERBATIM in contract_90.1.md (string-equality checked against masterplan).
3. experiment_results present: PASS (32,802 bytes, CYCLE 4 section current).
4. Log-last: masterplan 90.1 status=pending (NOT flipped). harness_log carries
   ONE row, "Cycle 194 ... phase=90.1 result=CONDITIONAL (loop stopped on
   operator instruction)" -- records the PRIOR cycle honestly, claims no PASS.
   NOTE not violation.
5. No verdict-shopping: PASS. Evidence CHANGED since cycle 3 -- commit
   a252b025 (attempt_gate.py +16, mutation_matrix_90_1.py +82,
   experiment_results +88, evaluator_critique +84).
Criteria erosion: NONE. success_criteria+command hash constant across all 11
commits in which 90.1 exists.

### Lint / syntax
- Scope DERIVED from git (git diff --name-only 3bf0b0fe^..HEAD -- '*.py'),
  non-empty (4 files), passed via xargs (not an unquoted zsh var).
  uvx ruff check --select F821,F401,F811 -> "All checks passed!", exit 0.
- ast.parse OK on all four subject files.
- No frontend/** and no backend/** in the 90.1 diff -> gates 1b/1c/1d do not
  bind. Step makes no UI claim.
- No unintended production change: git status shows only append-only audit
  streams + away_ops runtime files + this Q/A's own WIP record.
- Final tree recheck at return time: HEAD still f41e59a7; subject md5s identical
  to the matrix's own post-run report (attempt_gate 3c34102b34fb505dc69acb6b8a874827,
  attempt_budget 5511ac7e6f105b6b0716d4b80812a170,
  attempt_outcomes 81ebe68b498c63cbc424bf1f01ae02d1); the four pre-existing
  escalation_attempt_budget_*.md files untouched (mtimes 17-18 Aug). Nothing I
  ran mutated the repo.

### Precision on the C5 miss -- the kill is UNATTRIBUTABLE, not merely mislabelled
QA1b (extract_step_id_claim -> ..._v2, the FIRST call inside handle_hook's try,
so nothing downstream executes at all) fails 7 of 25 checks and is scored KILLED:
  - a below-ceiling launch for a REAL step is ALLOWED and COUNTED
  - an at-ceiling launch is DENIED (exit 2) and writes the attempt_budget escalation
  - a launch claiming an id ABSENT from the plan of record is DENIED (c4)
  - the unknown-id denial names the rejected claim and says the launch cost nothing
  - the unknown-id denial writes its OWN reason-named artifact (c2/c4)
  - and it still wrote its own record rather than staying silent
  - ONE attempt over DEFAULT_MAX_TOKENS is DENIED on the token ceiling (c3)
None of those guards was defeated; the gate never reached any of them. A broken
build green-washes criteria 2, 3 AND 4's checks simultaneously. That is exactly
the harm clause 3 exists to prevent.

### WARN -- an asserted fix that was never made, and it has PROPAGATED
experiment_results_90.1.md CYCLE 4 states: "The docstring is corrected, the decoy
moved to 386s, and M11 re-pointed at the measured threshold".
  - decoy moved to 386s: TRUE (mutation_matrix_90_1.py:229 plants 386_000).
  - M11 re-pointed: TRUE (cell M11 sets DEFAULT_TOLERANCE_S = 386).
  - "The docstring is corrected": FALSE.
    scripts/harness/attempt_outcomes.py:34-36 still reads verbatim:
      "Ambiguity first appears at 900s, which is why the default tolerance is
       30s: 30x headroom over the observed worst case and still an order of
       magnitude short of ambiguity."
    `git show --stat a252b025 -- scripts/harness/attempt_outcomes.py` is EMPTY --
    the cycle-4 commit never touched that file. Last touched by 1fc7b2e6 (cycle 2).
  - It propagated: the newly filed masterplan step 90.10's audit_basis repeats
    "90.1 cycle 4 corrected the docstring and re-pointed M11 at 386s".
  - Also stale: mutation_matrix_90_1.py:218-228 still says the decoy is "placed
    just past the DOCUMENTED ambiguity threshold (the module docstring: ambiguity
    first appears at 900s)" and "Moved to 950s", while the code plants 386_000.
  Product behaviour is unaffected -- but the stale 900s figure survives in
  PRODUCTION SOURCE, which is precisely the borrow-a-quoted-number failure 90.10
  was filed to stop.

### NOTE -- cycle-4 tolerance-sweep figures already drifted (~8h), threshold holds
My re-derivation on today's ledger vs Main's cycle-4 table:
  tol=385 ambiguous=0 (Main 0)      summed 22,942,987 (Main 21,059,736)
  tol=386 ambiguous=1 (Main 1)      summed 22,665,588 (Main 20,782,337)  <- threshold REPRODUCES
  tol=899 ambiguous=11 (Main 6)     summed 20,610,218 (Main 19,692,711)
The load-bearing claim (first ambiguity at 386s) reproduces exactly. The summed
figures and the 899s ambiguity count are corpus-dated and have already moved --
which is the recurrence 90.10 is filed for, so this is corroboration, not a
contradiction.

### NOTE -- token-denial stderr prescribes a remedy that cannot work
A DENY on the TOKEN ceiling prints "step 9.1 has used 1/5 attempts" and offers
`--operator-extend <sid> --by 1`. That raises max_attempts only; max_tokens is
fixed, so the next launch is denied identically. The escalation FILE body is
honest ("tokens used : 1,200,001 / 1,200,000"), so no criterion is violated.

### Verdict reasoning (worst-of-N lenses)
correctness   : product correct; matrix scoring wrong for a class -> CONDITIONAL
reproduce     : C1,C2,C3,C4,C6 reproduce; C5 clause 3 falsified by 3 executed
                counterexamples; "docstring is corrected" falsified -> FAIL
scope-honesty : disclosure otherwise excellent (population drift, UNKNOWN
                provenance, blast radius, a self-introduced M14 defect owned in
                writing); one asserted fix false and propagated -> CONDITIONAL
min(lenses) = FAIL. Criterion 5 states a UNIVERSAL ("never counts as a kill")
and I hold three executed counterexamples; a matrix result licenses only "these
N mutations were killed", never a global claim.

