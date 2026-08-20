STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 90.1
WRITTEN: 2026-08-20T19:27:24Z

# Q/A write-first record -- step 90.1 (cycle 1)

Spawn: Workflow rail, agentType qa. Evidence commit claimed: 3bf0b0fe.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable verification command, git status/diff scope, lint, syntax, runtime smoke
C. LLM judgment vs 6 immutable criteria + the 6 specific attack targets Main named

## Findings (appended as established)

### A. Harness compliance (all 5 clean)
- research_brief_90.1.md mtime 21:12:47 < contract_90.1.md 21:15:34 < attempt_outcomes.py
  21:17:38 < attempt_gate.py 21:21:17 < mutation_matrix_90_1.py 21:23:43 <
  experiment_results_90.1.md 21:26:07 < live_check_90.1.md 21:26:30 (local CEST). Order OK.
- log-last: `grep -F "phase=90.1" handoff/harness_log.md` -> 0 hits (positive control
  phase=86.116 -> 2 hits). masterplan 90.1 status=pending. Correct.
- verdict_history_86_21.py --step 90.1 --evidence-only -> status `no_rows_for_step`,
  verdicts (none). qa_wip.py --spawned-at 2026-08-20T19:27:24Z -> source_present true,
  attempt_number 1, prior_attempts 0, attempt_number_status ok, prior_records [].
  Cross-check prior_attempts(0) > ledger rows(0)? No. No staleness flag. Cycle 1 confirmed:
  no verdict-shopping possible.
- experiment_results_90.1.md + live_check_90.1.md present; no evaluator_critique_90.1.md
  yet (correct for cycle 1).

### B. Immutable verification command
`python3 scripts/harness/attempt_gate.py --self-test && python3 scripts/qa/mutation_matrix_90_1.py --verify`
EXIT=0. Self-test 32 checks all ok. Matrix: CONTROL GREEN observed first (21 checks),
KILLED 10 | SURVIVED 0 (excl N0) | ERROR 0 | N0 null mutant SURVIVED | real tree md5
unchanged. Reproduces the artifact exactly.

### B2. BLOCKING FINDING -- the backfill is NOT re-runnable (criterion 1)
Ran, verbatim, in my own shell:
```
$ python3 scripts/harness/attempt_outcomes.py --backfill --dry-run
AssertionError: backfill would MUTATE an existing field on row ts='2026-08-20T19:27:19Z'
step_id='90.1': ... {'outcome': 'UNKNOWN', 'outcome_reason': 'no_run_record',
'total_tokens': 0 ...} != {... 'outcome': None, 'outcome_reason': 'unresolved_at_launch',
'total_tokens': None ...}. Refusing to write; the ledger is append-only and enrichment
may only ADD keys.
exit=1
```
MECHANISM (two halves of the SAME commit are mutually incompatible):
- attempt_gate.py:451-462 now writes launch rows with the keys PRESENT and null:
  `"outcome": None, "outcome_reason": "unresolved_at_launch", "total_tokens": None,
  "run_id": None`, with a comment stating "the record is completed later by
  attempt_outcomes.py --backfill".
- attempt_outcomes.backfill():306-313 enforces additive-only by projecting the merged row
  onto the ORIGINAL key set and raising if ANY value differs. Since the key is already
  present, completing it is a MUTATION, so the guard aborts the WHOLE ledger write.
So the documented completion mechanism is the exact mechanism that now crashes.
NOT hypothetical: the first launch after the commit (19:27:19Z -- THIS Q/A spawn's own
attempt row) broke it, ~27 min after the artifact's capture. A second null row already
exists (19:27:57Z, step 90.2 research gate). Ledger is now 95 rows / 91 attempt.
Criterion 1 says "a RE-RUNNABLE backfill reconstructs both ... and prints the per-value
counts" -- it now prints no counts and exits 1.
Scope of damage: the LIVE gate is NOT broken -- resolved_rows() uses `not r.get("outcome")`
so a null row is re-resolved lazily in memory each read (verified by reading
attempt_outcomes.py:341 + attempt_gate.py:242-264). Damage is (a) the backfill CLI is dead,
(b) persisted rows can never be completed, (c) criterion 1's re-runnability clause is false.
Failure is fail-closed (raises before the write at :320), so no ledger corruption.
WHY THE MATRIX MISSED IT -- FIXTURE BLINDNESS (qa.md 4c shape #5). Every seeded row in
mutation_matrix_90_1.py::observations() and in _self_test uses the OLD row shape (bare
ts/type/step_id, or an already-non-null outcome). NONE uses the shape the production gate
now writes. Proven in memory:
  production shape -> projection != parsed: True   (backfill ABORTS)
  fixture shape    -> projection != parsed: False  (every drive stays green forever)

### B3. BLOCKING FINDING -- the membership check DENIES 10 real pending steps (criterion 4)
`masterplan_step_ids()` (attempt_outcomes.py:349-370) walks ONLY `phases[].steps[]`. The
real plan of record also holds steps under `phases[].subphases[]`. Measured:
  shallow walk (phases->steps) : 1347 ids
  dotted ids anywhere in file  : 1282
  dotted ids the walk MISSES   : 14
13 of the 14 are real steps under `subphases`; 10 of those are status=pending AND
harness_required=true: 38.13, 46.0, 46.1, 46.2, 46.3, 46.4, 46.5, 46.6, 46.7, 46.8.
PROVEN BY EXECUTION through the real hook (ledger+escalation dir redirected to a tempdir,
REAL masterplan):
  step_id=46.0    exit=2  "[attempt-gate] DENIED: ... '46.0', which is not a step in .claude/masterplan.json"
  step_id=38.13   exit=2  "[attempt-gate] DENIED: ... '38.13', which is not a step in .claude/masterplan.json"
  step_id=86.118  exit=0  (control -- the probe discriminates)
Criterion 4 says "rejects any id ABSENT from .claude/masterplan.json". 46.0 is PRESENT in
that file; the parser just does not look where it lives. So the gate now blocks 10 real
pending steps and the denial text asserts a falsehood about the plan of record.
Direction is fail-CLOSED, which is the opposite of the disclosed "membership degrades open".
ROOT CAUSE per qa.md 4b: no KNOWN-MEMBER RECALL TEST. Main's cells used one member
(86.118) and three non-members. A recall run over the plan's own members finds 14 misses.

### B4. WARN -- 90.1 turns a sibling step's checker permanently RED (consumer-contract-break)
`python3 scripts/qa/mutation_matrix_86_71.py --verify` now prints
"CONTROL IS RED -- the matrix is meaningless" with 5 failing checks. Root cause proven:
that matrix drives the hook with step_id "77.7"; '77.7' in masterplan -> False; the file
contains ZERO occurrences of ATTEMPT_GATE_MASTERPLAN, so it uses the REAL plan; and the
pre-90.1 gate had no membership check ('masterplan_step_ids' in pre-image -> False).
Census of every attempt_gate consumer that supplies a step id: exactly ONE is broken
(mutation_matrix_86_71.py); attempt_gate.py and mutation_matrix_90_1.py both set a
synthetic plan. Mitigation: 86.71's IMMUTABLE command is an ast.parse, not this matrix, so
no masterplan-immutable command is broken. The masterplan notes told Main to fix the
fixture; Main fixed the self-test fixture and did not sweep the other one, and the
"blast radius" measurement covered launch RECORDS, not checker FIXTURES.

### B5. WARN -- M10 is mislabelled; the direction it CLAIMS to test survives
M10's stated mutation is "the join reverts to `timestamp` semantics by WIDENING the
tolerance past the measured ambiguity threshold". The code actually does
`DEFAULT_TOLERANCE_S = 30` -> `0` (NARROWING). Neither named property is what runs.
I supplied both named mutants through the author's own run_cell harness:
  MX1 (start = d.get("startTime") -> d.get("timestamp"))  -> KILLED   (field half IS guarded)
  MX2 (DEFAULT_TOLERANCE_S = 30 -> 86400)                 -> SURVIVED (widening is UNGUARDED)
MX2 is NOT an equivalent mutant. Behavioural differential on the real ledger:
  tol=   30s  graded=66 ambiguous=0   summed_tokens=19,742,415
  tol=  300s  graded=66 ambiguous=0   summed_tokens=19,742,415
  tol= 3600s  graded= 6 ambiguous=68  summed_tokens= 3,847,645
  tol=86400s  graded= 2 ambiguous=77  summed_tokens= 1,865,277
i.e. widening collapses token accounting to ~9% and re-opens the very inertness
criterion 3 exists to close. Direction is fail-safe (under-count allows more), so WARN.

### B6. NOTE -- the containment check added as the fix for the disclosed leak is a proxy
attempt_gate.py:728-733, named "the self-test wrote every escalation into its OWN temp dir
-- nothing leaked into handoff/current/", asserts
`ESCALATION_DIR != old_e and all(p.parent == ESCALATION_DIR for p in ESCALATION_DIR.iterdir())`.
The second clause is a TAUTOLOGY (proven: iterdir() yields only children, so it is True for
any directory). The check never looks at handoff/current/ at all. It does catch the exact
historical bug (via the rebinding proxy) but not the property it names. The FIX ITSELF is
real: I confirmed the stray escalation_unknown_step_id_9.9.md is absent, was never
committed, and the four real exhaustion escalations are byte-identical after my own
self-test + matrix run. So target #5 is answered: Main FIXED it, did not merely disclose it.

### B7. WARN -- criterion 5 clause 3 falsified: a mutant that cannot build scores KILLED
Criterion 5: "a mutant that fails to run scores ERROR and never counts as a kill."
Two Q/A-supplied cells through the author's own run_cell:
  MXE1  anchor absent                       -> ERROR  ("anchor appears 0 times")  correct
  MXE2  injected `((((` -> file is a SyntaxError -> KILLED  (blanket check failures)
run_cell's ERROR path covers (a) anchor-count mismatch and (b) an exception raised inside
observations(). It does NOT cover a mutant that breaks the SUBPROCESS: subprocess.run does
not raise on non-zero exit, so the drives return rc!=0, the checks fail, and the harness
credits a KILL. BOUNDED: I parsed all 11 shipped cells after applying their replacement --
every one yields a parseable mutant, so none of the reported 10 kills is a build-failure
artifact. The mechanism is defective; this matrix's own cells do not exploit it.

## Criterion-by-criterion
1. NOT MET  -- fields+counts+UNKNOWN all verified (0 original fields changed vs the .bak,
   exactly 4 keys added, order preserved, UNKNOWN=5 all 999.2). The explicit word
   "RE-RUNNABLE" is falsified by execution: exit 1, AssertionError (B2).
2. MET      -- reason-named path, forged fallback deleted, sha256 before==after on a
   planted victim (matrix control) AND on the four real files; I re-derived all four
   hashes after my own runs -- unchanged.
3. MET      -- ENFORCE chosen and shown by execution. My own in-process drive:
   1,199,999 -> exhausted False/CONTINUE; 1,200,000 -> True/ESCALATE; 1,200,001 ->
   True/ESCALATE. Independent re-derivation of "no live decision changed" at the
   decide() level across 28 live ids: NONE. (`exhausted` alone does flip for 75.11.4,
   masked by CLOSED_PASS -- Main disclosed exactly that.)
4. NOT MET  -- the four named cells reproduce exactly on the real module, and the
   self-test exemption IS sound by construction (membership genuinely runs against a
   synthetic plan; 9.9/9.1.1/9.10 genuinely absent from it; env restored in finally;
   ATTEMPT_GATE_MASTERPLAN unset in production/settings.json). But the criterion says
   "absent from .claude/masterplan.json" and the walk denies 14 ids that are PRESENT
   (B3) -- 10 of them pending + harness_required.
5. MET with WARNs -- control GREEN first; M1 and M2 (the two NAMED mutants) KILLED;
   N0 survived; real tree md5 unchanged. WARN B5 (M10 mislabelled; the widening
   direction it claims to test SURVIVES with a 10x token-accounting differential) and
   WARN B7 (clause 3 falsified by MXE2).
6. MET      -- verdict_ledger.jsonl sha256 fcfe56ad...3eb2 identical before my runs and
   after the self-test + full matrix + my 4 extra cells. Only VERDICT_LEDGER write is
   attempt_gate.py:577, inside _self_test, with the global rebound to a tempdir at :532.

## Other checks
- ruff F821,F401,F811 over a DERIVED 8-file scope (git diff 3bf0b0fe~1..3bf0b0fe + worktree
  + untracked, xargs-quoted): "All checks passed!" exit=0.
- ast.parse OK on all 6 changed .py files. Backend import smoke OK (charts, settings,
  claude_code_client).
- Research gate: brief_status COMPLETE, gate_passed true, 10 read-in-full (floor 5),
  25 URLs (floor 10), recency scan present. All 6 criteria verbatim in the contract.
- Commit 3bf0b0fe swept in pre-existing uncommitted work (backend/api/charts.py,
  backend/config/settings.py, backend/agents/claude_code_client.py, 2 new test files,
  many handoff artifacts) plus handoff/audit/attempt_budget_audit.jsonl.bak. All predate
  the 90.1 work (present in the session-start git status). NOTE only.
- Claim that did not reproduce (direction conservative, against Main's own thesis): the
  commit message says the timestamp join "resolves 9". My re-derivation gives 1 with ISO
  parsing (0 if the field is treated as numeric). Not in any criterion or artifact.

COMPLETED: 2026-08-20T19:40:23Z

