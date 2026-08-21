STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 90.2
WRITTEN: 2026-08-21T08:07:25Z

# Q/A write-first record -- step 90.2 (cycle 4 evidence)

## Prior-attempt / prior-verdict EVIDENCE (gathered, not a trigger)
- `qa_wip.py 90.2 --spawned-at 2026-08-21T08:07:25Z`: source_present=true,
  attempt_number=4 (status ok, is_lower_bound=true), prior_attempts=3,
  records_retained=4 (GAUGE, not a counter), records_pruned_known=null.
- `verdict_history_86_21.py --step 90.2 --evidence-only`:
  status=`no_rows_for_step`, verdicts=(none).
- CROSS-CHECK: prior_attempts (3) > ledger verdict count (0) => LEDGER IS STALE
  for this step. sequence: UNKNOWN. Not guessed.
- `grep phase=90.2 handoff/harness_log.md` -> 0 rows (secondary source).
- `evaluator_critique_90.2.md` carries three Main-transcribed cycle verdicts
  (Cycle 1/2/3 headers). ADVISORY ONLY -- Main is the constrained party.

## A. HARNESS COMPLIANCE -- CLEAN (5/5)
1. research gate: brief envelope brief_status=COMPLETE, gate_passed=true,
   external_sources_read_in_full=7 (>=5), urls_collected=17 (>=10),
   recency_scan_performed=true. Contract cites run wf_05a76fdf-b16.
2. order (mtime): brief 08-20T21:39 < contract 21:52 < code (08-21 09:08 /
   09:30 / 09:48 / 10:05) < experiment_results 10:06:41 < live_check 10:06:55.
3. experiment_results_90.2.md present; live_check_90.2.md regenerated cycle-4.
4. log-last: no harness_log row; masterplan 90.2 status="pending".
5. no-verdict-shopping: evidence CHANGED (commits ca656466, f4f103c6,
   91626e61, 049544b0). HEAD aebd1602 unchanged for the whole evaluation.
   Working tree: only handoff/audit/*.jsonl + this WIP file.

## B. DETERMINISTIC
- IMMUTABLE COMMAND run bare, exit read directly: **EXIT=0**.
  "checks run: 87 (floor 74) / failed: 0"; 19 cells, N0 SURVIVED, QX ERROR,
  all others KILLED as expected; ledger sha256 byte-identical.
- Lint gates 1a/1b/1d NOT triggered: `git diff --name-only ca656466~1 HEAD`
  contains ZERO `*.py` and ZERO `frontend/**` and ZERO `backend/**`.
  JS syntax covered by `node --check` inside the immutable command.
- `--replay` re-run by me: pinned startsWith=441 (exact 436, +5 variants),
  parseable 398, with_verdict 397, non-PASS 288 -> queue_residual 41 /
  remediate 247 under BOTH matchers, disagreement 0/0, FAILs->queue_residual 0.

## C. CRITERION-BY-CRITERION
1 MET (with WARN). 2 MET. 3 MET. 4 MET. 5 MET. **6 NOT MET.**

### C1 -- MET, plus WARN S1 (one seam short)
Driven, not grepped: carve-out mutant `k !== 'route'` KILLED by the behavioural
drive; flatten-at-merge KILLED; `verdict:'PASS'` inside the routing KILLED by
J + I + the driven guard.
WARN S1: `leakedS` is computed on `merged`, but the script's final statement is
`return { ...merged, verdict_unmodified: untouched }` (qa-verdict.js:1080).
SURVIVORS (exit 0, zero failing checks):
  `return { ...merged, ...severity_routing, verdict_unmodified: untouched }`
  `return { ...merged, route: severity_routing.route, verdict_unmodified: ... }`
Caller fields become top-level siblings of ok/verdict in the object Main
transcribes VERBATIM -- the 86.78 QA-F shape criterion 1 names. Inherited: the
escalation and research_routing filters share the seam. FIX: run the three leak
filters against the object actually RETURNED, or fold verdict_unmodified into
`merged` before the guards.

### C2 -- MET
Structural `verdict.verdict === 'CONDITIONAL' && allResidual`. Section A drives
FAIL-all-WARN -> remediate, PASS -> remediate, same entries CONDITIONAL ->
queue_residual (discriminates). My own KC control (guard removed) -> exit 1,
6 failing checks. Replay: 0 FAILs routed to queue_residual, both corpora.

### C3 -- MET
24 real returns (>=20), 6 PASS / 6 FAIL / 12 CONDITIONAL. I independently
verified AUTHENTICITY: all 24 run_ids resolve in the real run-record corpus and
every fixture key is byte-identical to the record's parsed `result` --
**24/24 matched, 0 missing, 0 key mismatches**. Section F asserts by string
equality + input non-mutation; non-vacuous (my C3a mutant killed at "6/24 by
string equality", "18 mutated").

### C4 -- MET
41/247 reproduce EXACTLY at the pin under both matchers with identical run
sets; run ids printed; strict 32 explicitly NOT reproduced under four measured
definitions (41 / 26 / 11 / 4) and NOT edited to match; mixed WARN+untagged ->
remediate asserted and M2 kills the inversion. live_check carries the 41/247
table, the strict table, and the FAIL-immunity line (3 occurrences).

### C5 -- MET
4 of 4 of my own gate mutants KILLED by the specifically-named assertion:
  G1 well-formedness filter removed -> "a residual that grades NOTHING"
  G2 parent-as-own-residual        -> "the parent cannot be its own residual"
  G3 loose id match                -> "90.10 does NOT satisfy 90.1"
  G4 fail-open on bad JSON         -> "an unparseable plan REFUSES"
G0 (null) SURVIVED. NOT-WIRED disclosed plainly at source and in the results.

### C6 -- NOT MET (blocking)
Control-green-first half HOLDS (N0 SURVIVED, M1 KILLED same run, QX ERROR, no
NO-OP). Clause 1 exemplar (M2) KILLED. **Clause 2 falsified by execution:**

SURVIVOR 1 (fires on REAL fixture data):
  `derived_severities: derived,`
  -> `derived_severities: derived.length >= 4 ? derived.slice(0, -1) : derived,`
  exit 0, 87 checks, 0 failures. Differential on the checker's OWN 24 returns:
   wf_fc420eba-820 (FAIL, 4 entries) 4 -> 3, dropped verbatim entry
     "WARN illusory-guard: test_nightly_default_documented_off OR-escape-hatch
      satisfiable by a comment"; route remediate -> remediate (UNCHANGED, so
     every route assertion is structurally blind)
   wf_82381b7e-58c (FAIL, 5 entries) 5 -> 4, route unchanged.
SURVIVOR 2 (same site, branch-gated): `comparable ? derived.slice(0,-1)
  : derived` and the drop-first form -- both exit 0.
SURVIVOR 3 (the cycle-4 completeness claim itself): a NEW array-valued return
  key that is an array only on the comparable branch --
  `findings_digest: comparable ? entries.slice(0, -1) : null` -- SURVIVES E1b's
  set-equality check, because `arrayKeys` is computed from ONE probe return.
  The results claim "a fifth array field fails the checker until it is covered"
  is therefore an overgeneralization.
ROOT CAUSE: cycle 4 generalised along the FIELD dimension but coverage is still
bound to ONE probe input shape (2 findings / 3 detail rows / non-comparable).
Any drop conditioned on a shape outside that probe survives. FIX: parameterise
E1b over a FAMILY of probe shapes -- at minimum >=4 findings, an index-aligned
(comparable) input, and an emitted-but-mismatched input -- and assert the
array-key SET and per-array content on each.

KILL-MECHANISM HONESTY: two other mutants of mine exited non-zero only because
my replacement text collided with M11 / M15 / M16 `apply` strings and made
those cells NO-OP. That is a harness self-check artifact, NOT detection of the
drop, and I do not count it as coverage.

## D. NOTES (do not cap)
- N1 stale prose: `## 1. What was built` still says "66 checks over a floor of
  55, a 13-cell mutation matrix" and `### Criterion 6` says "11 cells" with an
  11-row table, while the current state is 87 / 74 / 19. The CYCLE 4 addendum
  states the current numbers correctly; the base sections are cycle-1 text left
  unlabelled. (Also internally inconsistent at cycle 1: "13-cell" vs 11 rows.)
- N2 the CYCLE 4 prose says the probe "makes the three arrays DIFFERENT
  lengths"; measured lengths are 2 / 2 / 3 -- TWO distinct values, and the
  checker's own assertion only compares two of the three
  (`new Set([derived.length, emitted.length]).size === 2`).
- N3 disclosed and accepted: `violation_details` prose severity is never scored;
  the immediate-negator rule fires on zero pinned entries; the close gate is not
  wired into any close path.

## HARNESS USED FOR MY OWN MUTATION WORK
Shadow repo in `mkdtemp`: mutated copy of qa-verdict.js (or residual_close_gate)
plus copies of the REAL checker, gate, fixtures, masterplan and ledger, then the
SHIPPED checker run end to end. Controls run FIRST every batch: N0 unchanged ->
exit 0; KC (verdict guard removed) -> exit 1 with 6 failing checks. Nothing in
the repository was written or modified.

COMPLETED: 2026-08-21T08:18:47Z
