STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 90.2
WRITTEN: 2026-08-21T07:32:30Z

# Q/A write-first record -- step 90.2 (cycle 2)

Spawn: Workflow rail, agentType qa. Read qa.md in full (STEP 0 done).
Commit under review: ca656466. HEAD at start AND at end: 0ddda824 (unchanged).

## SEQUENCE EVIDENCE
- qa_wip.py 90.2 --spawned-at 2026-08-21T07:32:30Z -> source_present=true,
  attempt_number=2 (status ok, is_lower_bound=false), prior_attempts=1,
  prior_records=[verdict_wip_90.2__20260821T071422Z.md], records_retained=2 (gauge).
- verdict_history_86_21.py --step 90.2 --evidence-only -> status=no_rows_for_step,
  verdicts=(none).
- CROSS-CHECK: prior_attempts (1) > ledger verdict count (0) -> THE LEDGER IS STALE
  for this step. sequence: UNKNOWN from the ledger. Main's cycle-1 disclosure (FAIL)
  is advisory only.
- harness_log: grep -cF "phase=90.2" = 0. masterplan status=pending. Log-last honoured.

## A. HARNESS COMPLIANCE (5 items) -- CLEAN
1. research gate: brief_status=COMPLETE, external_sources_read_in_full=7 (>=5),
   urls_collected=17 (>=10), recency_scan_performed=true, gate_passed=true;
   contract cites wf_05a76fdf-b16 and the brief path.
2. mtime order: brief 08-20 21:39 < contract 21:52 < code 08-21 09:29/09:30 <
   experiment_results 09:31:16 < evaluator_critique 09:32:10.
3. experiment_results + live_check present; all 6 criteria appear VERBATIM in the
   contract (string-equality checked against masterplan.json).
4. log-last honoured (see above).
5. no-verdict-shopping: evidence CHANGED (ca656466 touches qa-verdict.js +29,
   residual_close_gate.mjs +11, experiment_results, live_check, evaluator_critique).

## B. DETERMINISTIC
- IMMUTABLE COMMAND exit 0. 66 checks (floor 55), 0 failed, 13 cells as expected,
  N0 SURVIVED, QX ERROR, verdict_ledger sha256 cddc78f43062bdc8 unchanged.
- git status: only append-only handoff/audit/*.jsonl + my own WIP. No unintended change.
- Diff scope (git diff --name-only b09b937b..ca656466): ZERO *.py, zero frontend/**,
  zero backend/**. So gates 1a / 1b / 1d / 1c are N/A, not skipped.
  node --check exit 0 on all three changed JS/MJS files.

## C. INDEPENDENT RE-DERIVATION (my own code, not the author's)
441 / 436 exact / variants {writefirst-82-5:3, -82-7:2} / 398 parseable / 397 verdicts
/ CONDITIONAL 221, PASS 109, FAIL 67 / 288 non-PASS / 906 violated_criteria entries /
969 violation_details rows over with_verdict, 0 with a `severity` key. ALL REPRODUCE.
My own case-sensitive token-anywhere matcher: queue_residual 41 / remediate 247, and
the SYMMETRIC DIFFERENCE against the author's 41 run ids is 0 -- identical MEMBERSHIP,
not merely equal cardinality. FAILs all-WARN/NOTE: 0 of 67.
Verified claims: "3 of the 41 carry detail rows with no matching tagged entry, all
three read SEVERITY NOTE" -> EXACTLY 3 (wf_1afa11f6-75a, wf_2dd1efc9-d0c,
wf_28cf4dbb-9aa) and all three constraints contain "SEVERITY NOTE".
Fixtures: 24/24 resolve to real run records and match verbatim on
verdict + violated_criteria + violation_details; mix PASS 6 / FAIL 6 / CONDITIONAL 12.
RETIRED candidate findings: my stricter delimited rule gave 39 -- the 2 differing runs
carry `WARN(scope-honesty)` entry-initial tags MY follow-set wrongly excluded; the
author's rule is right. My first pass used a case-INSENSITIVE matcher and got 34
because "evidence block"/"block comment" match BLOCK -- my control was the defective one.

## D. MY OWN MUTATION CELLS (isolated /tmp tree, control GREEN first, restored GREEN)
  QA1-FIX allResidual over derivedOnly ....... KILLED (correctly -- it would ignore a
          judge-emitted BLOCK; candidate "the checker pins the hazard" RETIRED)
  QA3  drop LAST finding from derived ........ KILLED
  QA4  drop from derived_severities .......... KILLED
  QA5  drop from governing_severities ........ **SURVIVED** -> F1
  QA6  guard weakened to !== 'FAIL' .......... KILLED
  QA7  next_action on both routes ............ KILLED
  QA8  severity_source never ABSENT .......... KILLED
  QA9  derived index off-by-one .............. KILLED
  QA10 derivation_is_authoritative -> true ... KILLED
  QB1  routing mutates the verdict in place .. KILLED
  QB2  leak guard throw -> console.warn ...... KILLED
  QB3  leak guard filter can never fire ...... KILLED
  QB4  leak guard scans the WRONG object ..... KILLED
  QB5  severity_routing actually FLATTENED ... KILLED
  QB6  BLOCK stops dominating ................ KILLED
  QB7  negator widened to 45-char proximity .. SURVIVED (differential: moves 1 real
        run, wf_7fa0e5d6-c50; 1 of 906 entries reclassified -> NOT equivalent) -> F5
  QB8  delimiter `after` 45 -> 400 ........... SURVIVED (differential: 0 runs,
        0 of 906 entries -> equivalent mutant on the observed population; NOT a finding)

## E. FINDINGS
### F1 [WARN] criterion 6, second clause -- a silent finding-drop SURVIVES on the
sibling return field. `governing_severities` is in the return and is the array the
route is computed from (and the AUTHORITATIVE one once 86.98 lands). Mutating only
the returned object literal (`governing_severities: governing.slice(1)`) leaves the
route unchanged and passes ALL 66 checks, rc=0. Section E (checker :261-267) asserts
cardinality / index-alignment / class-content on `derived_severities` ONLY. M3
(:437-439) mutates the SOURCE array `derived`, which feeds BOTH fields, so the single
cell appears to cover two sites while only one is guarded.
FIX: mirror the three section-E assertions onto `governing_severities`, and add a
matrix cell that truncates it in the return literal.

### F2 [WARN] latent + undisclosed: `allResidual` is computed over `governing`
WITHOUT requiring `comparable`. Driven against the REAL extracted function:
  - CONDITIONAL + violated_criteria = [2 UNTAGGED blockers] + violation_details =
    ONE row with severity NOTE -> route = queue_residual,
    disagreement_status = not_index_comparable, derived = [UNTAGGED, UNTAGGED].
  - CONDITIONAL + violated_criteria = [] + one NOTE detail row -> queue_residual,
    which is exactly what checker section C and mutant M7 forbid on the other branch.
`emitted` is indexed by `violation_details`; `derived` by `violated_criteria`. The code
already computes `comparable` and does not use it to gate `allResidual`.
UNREACHABLE TODAY: VERDICT_SCHEMA (:439) has violation_details items
additionalProperties:false with no `severity` key -- verified. FAIL-immunity DOES hold
on this branch (FAIL/PASS -> remediate, driven). But this is the branch the step's own
notes say must "satisfy rather than pre-empt" 86.98 criterion 5, and the checker grades
it as working machinery (section D). The bypass is disclosed NOWHERE.
FIX: require `comparable` before `governing` may drive `allResidual`, or fall back
per-index to the derived class when `emitted[i]` is absent.

### F3 [NOTE] the N5 correction replaced one site, not both.
experiment_results_90.2.md:186 "0 of **969** (was quoted as 0 of 978)"; :260 still
"**0 of 978** violation_details rows carry a `severity` key". N5 is reported as
"FIXED at source". 969 is the figure that reproduces (measured). The in-code comment
carries 969 correctly; only the artifact's §"Explicitly NOT done" is stale.

### F4 [NOTE] the two artifacts disagree on the unpinned LIVE row.
experiment_results §criterion-4 table: LIVE 451/408/407/295/41/254.
live_check §2 verbatim replay: LIVE 452/409/408/296/41/255 -- which is what
`--replay` prints now. Inherent drift on an unpinned row; the PINNED figures agree
and are the load-bearing ones.

### F5 [NOTE] the rejected design is unguarded. M8 kills REMOVING the negator rule;
nothing pins its NARROWNESS. Widening IMMEDIATE_NEGATOR to a 45-char proximity window
-- the design the artifact says was measured and discarded -- survives all 66 checks
while reclassifying 1 real entry and moving 1 real run out of queue_residual. The
prior Q/A's NOTE ("unmeasured magic number") is upgraded: it is behaviour-changing,
not equivalent.

### F6 [NOTE] third independent tally of the tag-form table. Author 41/88/29/20/7
(12 bare); cycle-1 Q/A 41/91/37/1/5 (2 bare); my count of SEVERITY_TOKEN occurrences
over non-PASS violated_criteria is 177 against the artifact's ~197. The table is
already labelled ILLUSTRATIVE and nothing shipped depends on it (N4 acknowledged).

## F. CRITERION MAPPING
1 MET   -- sibling never merged; guard DRIVEN, and 4 independent neutering shapes
           (no-throw / can-never-fire / wrong-object / actual flattening) all KILLED.
2 MET   -- structural verdict guard verified on BOTH branches; 0 of 67 real FAILs
           all-WARN/NOTE (my own derivation); M1 + my QA6 both killed.
3 MET   -- 24 real returns, verbatim-verified by me against the run records, all three
           verdict values, string equality; QB1 (in-place verdict mutation) KILLED.
4 MET   -- 41/247 reproduce EXACTLY on the derived population with ZERO symmetric
           difference against my independent matcher; mixing rule enforced (M2 killed);
           "32 vs 41" stated, not resolved (4 strict definitions, none give 32).
5 MET   -- residual_close_gate.mjs refuses on absent / unparseable / toothless residual,
           fail-closed, driven against the REAL masterplan; NOT-WIRED disclosed at source.
6 MET WITH A NAMED GAP -- control GREEN first (verified in my own isolated tree), both
           NAMED mutants killed and independently reproduced; but F1 is a surviving
           mutant of exactly the shape the second clause names, on the sibling field.

## G. VERDICT DIRECTION
CONDITIONAL. Both cycle-1 blockers are genuinely fixed and I attacked them rather than
re-reading them. Two WARN-level findings (F1, F2), each with a named one-line fix.
No unintended production change; harness compliance clean.

COMPLETED: 2026-08-21T07:52:11Z
