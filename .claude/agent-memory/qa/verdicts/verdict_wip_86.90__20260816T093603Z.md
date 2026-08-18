STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.90
WRITTEN: 2026-08-16T09:36:03Z

# Q/A cycle-4 evaluation of step 86.90 (prompt render / [object Object])

## Plan
- A. harness-compliance audit (5 items)
- B. deterministic: immutable command, git scope, lint, rerunnable checkers
- C. mutation / claim audit against the 7 immutable criteria
- Specific hunts from the tasking prompt: (A) criteria CONTAINER hole closed or moved?
  nested array / array-containing-object / array-like object with length.
  (B) ABSENT path untouched. (C) is [3b] control discriminating -- would it go red if
  requireArgArray refused everything? (D) any figure that fails to reproduce (5th cycle).

## Findings log

### Prior-attempt evidence
- qa_wip.py 86.90 --spawned-at 2026-08-16T09:36:03Z: source_present=true,
  attempt_number=4 (status ok, is_lower_bound=true), prior_attempts=3,
  records_retained=4 (GAUGE, not a counter).
- verdict_history_86_21.py --step 86.90 --evidence-only: status=no_rows_for_step,
  verdicts=(none). attempt_number(4) > ledger count(0) => LEDGER IS STALE.
  Main's advisory disclosure says [CONDITIONAL, CONDITIONAL, CONDITIONAL].
  Ledger cannot corroborate. sequence: ADVISORY-ONLY (ledger stale).

### B. Deterministic
- IMMUTABLE COMMAND: bash -c 'source .venv/bin/activate && node --check
  .claude/workflows/qa-verdict.js && echo parses'  -> "parses", exit=0.
- node scripts/qa/verify_prompt_render_86_90.mjs -> exit 0, "ALL GREEN: 95 passed,
  0 failed". 6 mutation cells, all KILLED, each with its CONTROL clean first.
- node scripts/qa/verify_research_gate_workflow.mjs -> exit 0, 124 passed 0 failed.
- node scripts/qa/verify_escalation_86_78.mjs -> exit 0, failed: 0.
- node scripts/qa/verify_rail_retry.mjs -> exit 0, 38 passed 0 failed.
- node scripts/qa/verify_workflow_args_boundary.mjs -> exit 1, "FAILED: 84 passed,
  3 failed" -- MATCHES Main's disclosure (filed 86.92). Pre-existing to re-verify.

### CONTAINER SHAPE MATRIX (my own shapes, driven against the shipped script)
| shape | threw | spawns | placeholder | note |
|---|---|---|---|---|
| array of strings (control) | no | 1 | false | renders 1./2. -- OK |
| nested array element | no | 1 | false | renders as JSON fence -- lossless |
| array containing an OBJECT | no | 1 | false | renders as JSON fence -- lossless |
| array-like object {0:..,length:1} | YES names args.criteria | 0 | - | CLOSED |
| empty array [] | no | 1 | TRUE | placeholder on a PRESENT field (see below) |
| SPARSE array (hole) | no | 1 | false | hole SILENTLY skipped by .map, blank line |
| element undefined / null / '' | YES names args.criteria[i] | 0 | - | loud |
| array w/ non-index own prop (a.meta) | no | 1 | false | prop SILENTLY dropped |
| Array subclass | no | 1 | false | passes Array.isArray |
| Proxy over array | no | 1 | false | passes (Main disclosed, unfixed by decision) |
| boolean / string / number | YES names args.criteria | 0 | - | CLOSED |
### MY OWN MUTATION MATRIX on the cycle-4 guard (hunt item C)
| mutant | [3b] CONTROL-1 array | [3b] CONTROL-2 absent | [3b] wrong-shape |
|---|---|---|---|
| SHIPPED baseline | GREEN | GREEN | GREEN |
| requireArgArray REFUSES EVERYTHING | **RED** | **RED** | GREEN |
| wrong shape -> a DIFFERENT silent fallback | GREEN | GREEN | **RED** |
| reverted to the original silent discard | GREEN | GREEN | **RED** |
=> the [3b] CONTROLS ARE DISCRIMINATING (a blanket-refusal guard turns both RED),
   and criterion 5's "a different silent fallback does not close it" is ENFORCED
   by execution, not asserted. Anchor uniqueness verified = 1.

### REGRESSION PROBES (hunt item B -- the ABSENT path must be untouched)
- args NOT BOUND (blind): no throw, 0 spawns, returns {dry_run:true, verdict:null,
  ok:false} -- phase-86.17 invariant PRESERVED by cycle 4.
- args = null: 0 spawns, verdict null, ok false.
- args as a JSON STRING (the majority historical path) + criteria list: renders,
  1 spawn, evidence as a JSON fence.
- args as a JSON STRING + criteria string: THROWS, 0 spawns.
- a THROW returns NOTHING (return value is null) -- no path from the new throw to a
  verdict object. Criterion 7 holds on the new control-flow outcome.

### HISTORICAL-CALLER CENSUS (my own derivation, 595 run records this slug)
qa-verdict run records = 395; criteria shape: list=386, ABSENT=1, args-unreadable=8,
STRING=0, OBJECT=0. => no historical caller can be broken by requireArgArray, and
the single ABSENT caller keeps the documented placeholder. (Main's spawn-prompt
figure "384 of 385" is not in the graded artifacts; my 386+1 is consistent with it
under corpus growth.)

### BLAST-RADIUS RE-DERIVATION (criterion 4)
Independent method: parse the FIRST USER MESSAGE of every agent transcript
(1,396 prompts) and require the coerced field to be its OWN prompt line.
- runIds with a coerced field: 23. Author's table: 22.
- author \ mine = {} (EMPTY -- I missed none of theirs)
- mine \ author = {wf_4588d8a7-e70} = this step's OWN deliberate pre-fix probe,
  correctly excluded from "production spawns".
- "6 of them also lost extra": my count = 6. EXACT MATCH.
- OBJECTIVE / INTERNAL SCOPE numerators: 0 and 0. MATCHES criterion 3's claim.
=> the load-bearing census REPRODUCES by symmetric difference, not merely by count.

### FIGURES THAT DO **NOT** REPRODUCE (claim audit)
- "0 of 75 spawns carrying OBJECTIVE:" -> I measure 71 (line-start) / 72 (substring).
- "0 of 72 carrying INTERNAL SCOPE:" -> I measure 59 by BOTH operationalisations.
  Corpus growth can only RAISE my number, so 72 -> 59 is not explained by growth.
- "args recovered ... a real object on 31 records" -> I measure 62 real-object
  records of 595 (413 JSON strings, consistent with the claimed 409 under growth).
- "507 prompts inspected" -> I inspect 1,396 first-user messages; plausibly a
  per-RUN vs per-AGENT-FILE unit difference, but the unit is not stated.
  NUMERATORS all reproduce; only these auxiliary DENOMINATORS do not.

### CYCLE-4 SPECIFIC DEFECT FOUND (the fifth instance Main asked me to hunt)
experiment_results_86.90.md:196 heading reads "### Mutation matrix (6 cells, all
KILLED)" over a table enumerating only **5** rows (M1..M5). The 6th cell added
THIS cycle (container-guard-reverted-to-silent-discard) is described at :565 but was
never added to the matrix it belongs to. Live run emits exactly 6 ": KILLED" lines,
so the NUMBER is right and the ENUMERATION is stale -- W2's regeneration audit
checks COUNTS, not the list the count heads, so it printed "STALE COUNTS REMAINING:
none" while the table under the count is one row short.

### ASSERTION-COUNT CLAIMS -- ALL REPRODUCE
Ran each historical checker blob against the CURRENT shipped scripts:
cycle1 a21a5889 = 53, cycle2 98c5b6ab = 78, cycle3 468c7908 = 83, cycle4 = 95.
Claimed "53 at cycle 1, 78 at cycle 2, 83 at cycle 3" and "83 -> 95": ALL EXACT.

=> the JSON-REACHABLE container hole IS closed (string/object/number/boolean/
   array-like). Residual gaps are NOT JSON-reachable (sparse hole; non-index own
   property) -- same class as the exotic A1/A2/A4/A6/A7 the author DID fix for the
   prose fields, so it is an ASYMMETRY between the container and prose paths.

### A. HARNESS COMPLIANCE -- CLEAN
1. research-gate-before-contract: research_brief_86.90.md envelope brief_status
   COMPLETE, gate_passed true, external_sources_read_in_full 12 (>=5),
   urls_collected 45 (>=10), recency_scan_performed true, audit_class true with
   coverage.dry true (6 rounds, 2 dry). PASS.
2. contract-before-generate: mtimes research 09:59:05 < contract 10:01:10 <
   experiment_results/live_check 11:35:04 < evaluator_critique 11:35:36. PASS.
3. experiment_results present (34,025 bytes) + live_check present. PASS.
4. log-last: `grep -F "phase=86.90" handoff/harness_log.md` -> 0 rows; masterplan
   86.90 status = pending. Correctly NOT logged and NOT flipped. PASS.
5. no-verdict-shopping: evidence CHANGED between cycles 3 and 4 -- qa-verdict.js,
   research-gate.js, verify_prompt_render_86_90.mjs and both artifacts all in
   commit 0ecccafe; checker 83 -> 95 with a new cell. Documented cycle-2 flow. PASS.
6. queued-is-real (the cycle-2 D2 rule): 86.92/86.93/86.94/86.95 all EXIST as
   pending masterplan steps. PASS.

### SCOPE / LINT
- graded commit 0ecccafe touches 10 files: 2 workflow scripts, 2 checkers, 6
  handoff artifacts. `git diff --name-only 0ecccafe^ 0ecccafe | grep -E
  "^(backend|frontend)/"` -> NONE. No trading/production code.
- ruff F821,F401,F811 over the commit's only .py (verify_changelog_flip_86_91.py,
  86.91's file): "All checks passed!" exit=0. Non-empty file set asserted first.
- 86.90 subject files: `git diff --stat HEAD -- <the 3 files>` EMPTY (tree ==
  commit). Unrelated uncommitted edits exist in backend/api/sovereign_api.py and
  5 frontend files; they predate this spawn and are NOT in the graded commit.
- args_boundary checker RED at 84/3 -- matches Main's disclosure; filed as 86.92.
- No UI claims -> 1c N/A. No backend/** in the commit -> 1d N/A.

### THREE ARTIFACT DEFECTS FOUND IN THE CYCLE-4 EDIT ITSELF
E1 experiment_results:196 "Mutation matrix (6 cells, all KILLED)" over 5 rows.
E2 experiment_results:488 "ALL GREEN: 95 passed ... (REGENERATED cycle 3)" --
   cycle-4 number under a cycle-3 regeneration label.
E3 live_check:281 "(REGENERATED cycle 4)" but the cycle-4 diff to that file is
   ONE line; its [5] body has 4 KILLED / 0 CONTROL vs a real run's 6 / 6, and
   `grep -c container-guard-reverted-to-silent-discard live_check_86.90.md` = 0.
All three were CREATED BY the cycle-4 edit (they did not exist at cycle 3): the
5-row table became wrong when the heading went 5->6; both markers became wrong
when the totals went 83->95. W2's audit is pointed at NUMBERS, not at captures.

### CRITERIA DISPOSITION (all by MY execution, not by reading)
1 REPRODUCED  : MET  -- historical receipt wf_b1747d75-eec found by my own
                first-user-message scan; probe wf_4588d8a7-e70 likewise;
                section [1] regenerates from blob 75831f4c, 2 of 2 scripts.
2 LAYER       : MET  -- pre-fix blob reproduces, current does not; fix is in the
                prompt-template field boundary only.
3 research-gate: MET -- vulnerable pre-fix and clean now BY EXECUTION (I ran it);
                numerator 0/0 reproduces. Denominators do not (V4).
4 BLAST RADIUS: MET  -- symmetric difference author\mine = EMPTY; my only extra is
                their own probe run. "6 also lost extra" = 6 exact. 86.86 resolved
                explicitly (re-graded PASS on the fixed rail, wf_a09930e2-3d7);
                other three PASSes named and filed as 86.93.
5 FAIL LOUDLY : MET for the reachable surface incl. the CONTAINER; residual
                non-JSON-reachable asymmetry recorded as V5 (NOTE).
6 GUARD+MUTATION: MET -- 6 cells, each control clean FIRST; plus my own 3
                independent mutants proving the [3b] controls discriminate and
                that a DIFFERENT silent fallback is caught.
7 SEMANTICS   : MET -- blind run still {verdict:null, ok:false}; a throw returns
                NOTHING; VERDICT_SCHEMA / enforceEscalation / enforceGate untouched.

### VERDICT REASONING
7/7 criteria MET, harness compliance clean, no unintended production change, no
product defect survived an adversarial matrix. The three defects above are
artifact-transcription defects in the UNDERSTATING direction over a re-runnable
command -- WARN, not BLOCK, per qa.md 4c's wiring analogue (a genuine behavioural
guard exists; only its transcript is wrong). But they are NEW, introduced by this
cycle's own edit, inside the two artifacts whose cycle-4 remedy was precisely
"counts are now derived, never typed" -- the remedy stopped one seam short of the
captures. qa.md 4b is unconditional: an edited capture in a verbatim-labelled
block is a finding regardless of whether the command passed.
=> CONDITIONAL. Not FAIL: no criterion is missed. Not PASS: the operator-facing
live_check gate artifact misdescribes this cycle's own mutation evidence.

COMPLETED: 2026-08-16T09:49:42Z

