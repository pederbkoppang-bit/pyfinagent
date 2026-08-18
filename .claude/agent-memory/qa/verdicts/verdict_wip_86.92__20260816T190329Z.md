STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.92
WRITTEN: 2026-08-16T19:03:29Z

## Q/A write-first record (crash-survival). NOT a verdict.

Spawn: Workflow rail, step 86.92, EVALUATE.
qa_wip.py --spawned-at 2026-08-16T19:03:29Z -> attempt_number=1,
attempt_number_status=ok, attempt_number_is_lower_bound=false, source_present=true,
prior_attempts=0, prior_records=[].
verdict_history_86_21.py --evidence-only -> status=no_rows_for_step, verdicts=(none).
Cross-check: prior_attempts(0) == ledger rows(0) -> no staleness signal.
sequence: EMPTY (no prior verdicts recorded; the tool's own detail calls absence
weak evidence because nothing writes the ledger automatically yet).

### A. HARNESS COMPLIANCE (5 items) -- CLEAN
1. research-gate-before-contract: PASSED and ENFORCED, not self-reported.
   Run record /Users/ford/.claude/projects/-Users-ford--openclaw-workspace-pyfinagent/
   04c4aab2-53ec-484b-8e2f-ba3d31780808/workflows/wf_2ee79ffe-d4f.json:
   status=completed, scriptPath launch (rail R7), start 20:47:33, end 20:56:34 local,
   540,445 ms, 190,482 tokens, 2 agents. result.gate_passed=true,
   self_report_disagreed=false, violations=[], sources_floor_ok 7>=5,
   urls_floor_ok 22>=10, recency_scan_ok, brief_status_in_brief COMPLETE,
   urls_collected_corroborated 22<=22, all_7_claimed_sources_present_in_brief.
   NOTE: the COMMITTED brief (687109bb, 20:54:47) still carries
   brief_status=INCOMPLETE/gate_passed=false; the working-tree brief (mtime
   20:55:34) carries COMPLETE/true. The commit caught the brief 47s mid-flight.
   Not a mid-EVALUATE edit -- nothing changed after 20:55:34.
2. contract-before-generate: gate 20:56:34 < contract 20:57:37 < .mjs 21:00:07
   < masterplan 21:01:08 < experiment_results 21:02:05 < live_check 21:02:40.
   ORDER OK. (Reproduction/diagnosis ran 20:47 -- read-only, and the contract
   discloses it as "DONE before this contract". No production file moved before
   the contract.)
3. experiment_results_86.92.md present and substantive (223 lines).
4. log-last: grep -cF "phase=86.92" handoff/harness_log.md = 0; masterplan 86.92
   status=pending. NOT yet logged / flipped. COMPLIANT.
5. no-verdict-shopping: attempt 1, no prior verdict. N/A.

### B. DETERMINISTIC
- IMMUTABLE CMD: `node --check scripts/qa/verify_workflow_args_boundary.mjs && echo parses`
  -> "parses", exit=0. (Author-disclosed as parse-only and unable to fail on this
  defect -- CONFIRMED: green at every historical commit I replayed.)
- FULL CHECKER in-repo: exit=0, "ALL GREEN: 95 passed, 0 failed" (95 ok lines, 0 FAIL).
- 86.23 compound command `verify_research_gate_workflow.mjs && verify_workflow_args_boundary.mjs`
  -> exit 0 (leg1 0, leg2 0). Claim "exit 1 -> 0" CONFIRMED (whole compound re-run).
- Sibling gates: 124/0, 95/0, 38/0, ALL CHECKS PASS. All four reproduce exactly.
- SCOPE: commit b46f0e17 = 5 files, exactly ONE production file. Deleted lines in
  the production diff = 4, all replaced by equivalent-or-stronger form. check()
  calls 23 -> 31 (+8 = exactly the 8 new assertions claimed). No assertion removed.
  `git diff --stat HEAD -- .claude/workflows/ .claude/agents/qa.md scripts/qa/` EMPTY.
  No production module imports the checker (grep: only comments + masterplan).
- LINT: derived scope `git diff --name-only HEAD -- '*.py'` = backend/api/sovereign_api.py
  (1 file, non-empty set asserted). `uvx ruff check --select F821,F401,F811` ->
  "All checks passed!", exit 0. That file's mtime is 2026-08-14 -- pre-existing,
  not authored by this step. Same for the 5 modified frontend/*.tsx (all 08-14),
  so gate 1b is not triggered by this step's diff.
- No .py / frontend / backend file in the step's own commit -> 1a/1b/1d N/A for
  the graded change; no UI claims -> 1c N/A.

### C. INDEPENDENT RE-DERIVATION (I re-ran everything; nothing below is read off
    the author's artifacts)

C1. "enforceGate is pure; brief_path is inert" -- REPRODUCED.
  Stale 4-field literal driven against brief_path=research_brief_86.17.md vs
  /nonexistent/path/does_not_exist_zzz.md: path-normalised violations arrays
  IDENTICAL (true); both gate_passed=false. FILED CAUSE IS FALSE as a mechanism.
  BRIEF_VERIFICATION_SCHEMA.required = 9 fields; old literal supplied 4. Correct.
  Pre-fix literal really is at :179 and :319 (verified in the b1469a06 blob).

C2. BISECT -- REPRODUCED by execution in per-commit scratchpad mirrors. Section [1]
  degrades to a gitShow artifact outside a git repo, so I discount exactly that line:
    089726f9 (08-10 08:27, parent of cad38647): 86 passed/1 failed = artifact only. GREEN.
    cad38647 (phase-86.6, 08:51): 83/4 -> 3 REAL failures; [3] message leads with RECENCY.
    d3bb1dfb (phase-86.37, 17:34):  same 3 REAL; message now leads with brief_status
                                    -- which is WHY the red looked like an 86.37 problem.
    b1469a06 (pre-fix HEAD):        same 3 REAL.  (in-repo: 84 passed/3 failed)
  RED onset = cad38647 (phase-86.6), NOT phase-86.37 as filed. CONFIRMED.
  `git log -S` on BOTH new schema fields returns cad38647 only.
  Window 2026-08-10 08:51 -> 2026-08-16 21:02 = 6d12h11m. "~6d12h" CONFIRMED.

C3. Cell [4] was NON-DISCRIMINATING -- REPRODUCED, 2x2:
    guard=present fixture=stale   -> false (4 viol)
    guard=absent  fixture=stale   -> false (3 viol)   <- DEAD
    guard=present fixture=healthy -> false (1 viol)
    guard=absent  fixture=healthy -> true  (0 viol)   <- KILLS
  Exactly the matrix claimed.

C4. Historical rot replay -- REPRODUCED EXACTLY: deleting the 3 fields shrinks the
  fixture by 155 bytes (byte count matches to the byte), the canary NAMES all three
  (brief_status_in_brief, distinct_urls_in_brief, recency_section_present) in BOTH
  the declared and consumed assertions, and the three original 2026-08-10 failures
  reappear. My mirror: 88 passed/7 failed; author's worktree: 89/6 -- delta is
  exactly the [1] gitShow artifact. Consistent, not contradictory.

C5. 13 KILLED cells all green (reproduced). 11 pre-existing + 2 new canary cells.
  95 = 87 (pre-rot total, confirmed by my 089726f9 replay) + 8 new. Arithmetic holds.

C6. Blast radius INDEPENDENTLY re-derived: the only masterplan steps whose
  verification references this checker are 86.17 (done), 86.23 (pending),
  86.92, 86.101. masterplan at a212dfe9 -> 86.17 pending; at 089726f9 (08:27) ->
  86.17 ALREADY done; at cad38647 (08:51) -> done. So 86.17 closed ~24 min BEFORE
  the break. NO step closed inside the red window on this gate. Author's claim
  and its careful distinction ("retrospective auditability, not a wrongly-admitted
  step") are correct.

C7. -1 sentinel: research-gate.js:632 IS exactly
  `const n = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : -1)`;
  :738 `const briefUrls = n(verification.distinct_urls_in_brief)` renders it at :740.
  86.101 filed, status pending, 5 criteria, masterplan diff `20 0` pure addition.

### MY OWN MUTATION MATRIX AGAINST THE NEW GUARDS
Control at HEAD in the mirror = 94 passed / 1 failed (the [1] artifact only).
  M1 drop brief_status_in_brief from fixture      -> KILLED (named)
  M2 drop recency_section_present                 -> KILLED (named)
  M3 drop distinct_urls_in_brief                  -> KILLED (named)
  M4 move the START boundary marker               -> KILLED (anchored check fires)
  M6 verificationFieldsRead returns [] anchored   -> KILLED (consumed canary fires)
  M7 enforceGate reads an undeclared field        -> KILLED (both consumed and the
     "reads no field the schema does not require" cell fire, naming __undeclaredSneak__)
  M9 schema.required emptied of the 3 new fields  -> KILLED
  M10 `delete crippled.brief_status_in_brief` inert -> KILLED (both canary-KILLED cells)
  M5 NEUTER THE COMMENT-STRIPPER                  -> **SURVIVED**  (FINDING 1)

### FINDING 1 (WARN) -- the comment-stripper POSITIVE CONTROL cannot fail
scripts/qa/verify_workflow_args_boundary.mjs, section [3]:
    const poisoned = src.replace('function enforceGate',
      '// verification.__bogusProseOnlyField__ appears only in prose here\nfunction enforceGate')
The poison is inserted IMMEDIATELY BEFORE the only occurrence of the slice anchor,
so verificationFieldsRead slices FROM the anchor and the comment is OUTSIDE the
scanned region. MEASURED:
    occurrences of 'function enforceGate' in research-gate.js : 1
    index of injected comment                                 : 33473
    index of slice START in poisoned                          : 33524
    comment INSIDE the slice?                                 : false
    poisoned, stripper ON  -> false ; poisoned, stripper OFF -> false
So `stripped` is false unconditionally and `naive && !stripped` is true
unconditionally. Proof by mutation: with BOTH strip operations replaced by an inert
`.replace(/__NEVER_MATCHES_AAA__/g,'')` the checker still prints
    ok   [3] fixture canary CONTROL: a comment-only field IS present in the raw source
    ok   [3] fixture canary CONTROL: the stripper rejects a comment-only field
(94 passed / 1 failed = the [1] artifact only).
Also: on today's source fields(stripped) == fields(unstripped) (identical 7-element
arrays), so the stripper has no current effect either.
The code does not implement what its own comment describes: the comment says "the
un-stripped scan sees it, the stripped scan rejects it" (a scan-vs-scan
differential); the code tests a raw-string regex against a region scan.
REMEDY, measured to discriminate: inject the same comment INSIDE the region (e.g.
before `const selfReported = ...`) -> stripper ON false / OFF true.
FALSE CLAIMS this refutes:
  - source comment: "The positive control ... proves the stripping is live rather
    than decorative ... A control that cannot fail is not a control."
  - experiment_results_86.92.md:68-69: "If the injection ever stops landing, the
    control fails rather than passing vacuously."
SEVERITY WARN, not BLOCK: 1 vacuous control alongside 8 genuine guards I executed
and killed (qa.md 4c verdict wiring). NOT a duplicate of queued step 86.23, whose
two vacuity gaps are in section [5] (asymmetric CONTROL, throw-scored-as-kill).

### FINDING 2 (WARN) -- trimmed output inside blocks declared verbatim
live_check_86.92.md:7-8 declares "Every block below is verbatim tool output from
this session unless labelled otherwise." The block at :250-259 shows 2 of 6 FAIL
lines then the "FAILED: 89 passed, 6 failed" summary, with 4 lines elided and no
ellipsis; the following prose says "the two canary failures" + "three original
failures" = 5, against a summary of 6 (the 6th is the differential cell, disclosed
only in the sibling artifact). experiment_results_86.92.md:111-115 likewise
truncates 5 of 11 KILLED lines mid-sentence. I regenerated every elided line and
all are TRUTHFUL -- this is disclosure completeness, not fabrication.

### FINDING 3 (NOTE) -- provenance slip, contradicted by the shipped code
contract_86.92.md:64 and live_check_86.92.md:81 attribute the added schema fields
to "86.28/86.37". `git log -S "distinct_urls_in_brief"` and
`git log -S "recency_section_present"` on .claude/workflows/research-gate.js each
return cad38647 = phase-86.6 ONLY. The shipped source comment in the checker has it
right ("phase-86.6, cad38647, 2026-08-10 08:51"). Load-bearing bisect statement is
correct everywhere it matters; this is an internal inconsistency in a supporting
clause. Does not degrade the verdict on its own.

### FINDING 4 (NOTE) -- stale born-inert markers on two graded artifacts
live_check_86.92.md:3 still reads "STATUS: IN PROGRESS" although section E is
written and final. The committed research_brief_86.92.md carries
brief_status=INCOMPLETE / gate_passed=false while the working-tree copy (which is
what the gate actually read, and what the run record corroborates) is COMPLETE/true.
Cosmetic; no gate consumed the stale copies.

### CRITERION MAP
1 MET   - reproduced at b1469a06 (84/3); cause localised by execution: which
          assertions ([3] healthy-run, [3] no-regression, [4] drop-blind-violation),
          which fixture (:179/:319 literal), which enforceGate rules (brief_status
          ABSENT branch, recency over-claim, distinct-URL over-claim). Filed cause
          falsified by a nonexistent-path control I reproduced byte-identically.
2 MET   - sentinel deliberate + documented (:632) AND the rendering filed as 86.101
          (pending, 5 criteria, pure +20 masterplan addition). Both disjuncts answered.
3 MET   - enforceGate untouched (git diff empty, commit stat, no check() removed,
          23 -> 31). Fixture replaced, not the rule.
4 MET   - fixture is synthetic, owned by the checker, DERIVED from schema.required;
          durability proven by execution (M1/M2/M3 + the 155-byte historical replay
          naming all three fields).
5 MET (with FINDING 1 attached) - exit 0, 95/0; all 13 KILLED cells green; the
          previously non-discriminating [4] cell now discriminates (2x2 reproduced).
          No pre-existing mutant survives. BUT one NEWLY-ADDED control cannot fail,
          which is the failure mode this criterion's own sentence is about.
6 MET   - duration from git history of both files (6d12h11m, bisected by execution);
          no step closed in the window on this gate -- independently re-derived from
          masterplan blobs at a212dfe9/089726f9/cad38647.
7 MET   - qa-verdict.js / research-gate.js / qa.md untouched; no production module
          imports the checker; nothing can convert a non-PASS into a PASS.

VERDICT DIRECTION: CONDITIONAL. Every criterion is met on its letter and the
product change is correct and fully reproduced by me. The cap is FINDING 1: a
newly-added guard that cannot fail, plus two artifact sentences asserting the
opposite. Per qa.md 4c a vacuous guard alongside genuine behavioural guards is a
WARN-level finding with a named fix -- not a FAIL.

COMPLETED: 2026-08-16T19:16:27Z (read from `date -u`, not narrated)
