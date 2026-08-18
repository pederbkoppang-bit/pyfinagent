STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.92
WRITTEN: 2026-08-16T19:24:29Z

# Q/A cycle 2 for step 86.92 (verify_workflow_args_boundary.mjs gate RED)

Prior: cycle 1 = CONDITIONAL (wf_1afa11f6-75a). Main claims 45b74291 on top of
b46f0e17 fixes: (1) illusory comment-stripper control, (2) scope-honesty
elisions, (3) provenance 86.28->86.6, (4) born-inert live_check header.

## A. HARNESS COMPLIANCE (5 items) -- CLEAN
1. research-gate-before-contract: brief_status COMPLETE, gate_passed true,
   external_sources_read_in_full 7 (>=5), urls_collected 22 (>=10), recency-scan
   section @line 81. Contract cites run wf_2ee79ffe-d4f + names 7 sources +
   discloses 3 unfetchable (Wiley 402, xunitpatterns ECONNREFUSED, GTB). MET.
2. contract-before-generate: research brief committed 687109bb; contract created
   in b46f0e17. Current mtimes: research 20:55:34 < live_check 21:22:09 <
   contract 21:22:52; checker 21:19:08 is BEFORE the contract mtime -- because
   cycle 2 AMENDED the contract after the final code edit, which is the
   documented cycle-2 flow, not a violation. Single-commit ordering ambiguity
   noted (memory: project_contract_order_mtime_fallback). MET.
3. experiment_results present + substantive. MET.
4. log-last: grep -Fc 'phase=86.92' harness_log.md = 0; masterplan status =
   pending. MET.
5. no-verdict-shopping: evidence CHANGED (45b74291 rewrote 76 lines of the
   checker + 4 artifacts) AND the change is CAUSALLY responsible -- see D.0. MET.

## B. DETERMINISTIC
- IMMUTABLE CMD `bash -c 'node --check ... && echo parses'` -> `parses`, EXIT=0.
- Checker: EXIT=0, `ALL GREEN: 96 passed, 0 failed`; 96 ok-lines, 0 FAIL-lines.
  Reproduced in THREE environments (repo, $S/mroot, $S/m2) -- all 96/0.
- `git log 687109bb~1..HEAD -- .claude/workflows/` = EMPTY. research-gate.js /
  qa-verdict.js last touched 0ecccafe; qa.md last touched 85127353. Rail R5 held.
- Step diff (b46f0e17~1..HEAD) touches NO *.py and NO frontend/**.
- ruff F821,F401,F811 on the derived py scope (backend/api/sovereign_api.py,
  an UNCOMMITTED file dated 2026-08-14, i.e. 2 days BEFORE this step's
  20:47-21:24 window): "All checks passed!", exit 0. The 6 uncommitted
  backend/frontend edits ALL have 2026-08-14 mtimes => not attributable to 86.92.
  NO unintended production change from this step.

## C. INDEPENDENT RE-DERIVATION OF EVERY HEADLINE CLAIM (all reproduced)
- Pre-fix RED: b1469a06 blob -> `FAILED: 84 passed, 3 failed`, the same 3 cells.
- Purity falsification, driven directly (my own harness, 5 variants):
    stale fixture + REAL 86.17 path        -> false | 3 viol
    stale fixture + NONEXISTENT path       -> false | 3 viol
    stale fixture + brief_path undefined   -> false | 3 viol
    HEALTHY fixture + REAL path            -> true  | 0 viol
    HEALTHY fixture + NONEXISTENT path     -> true  | 0 viol
  enforceGate body: 0 fs/readFile/require/import. The filed cause (stale on-disk
  brief) is FALSE; Main's falsification stands.
- schema.required = 9 fields; enforceGate READS 7; pre-fix literal at :179/:319
  supplied exactly 4 (`brief_exists, brief_non_empty, char_count, urls_missing`).
  Contract's "4 of the 9 / 4 of the 7" is exact.
- Provenance per field (git log -S): recency_section_present cad38647 (86.6),
  distinct_urls_in_brief cad38647 (86.6), brief_status_in_brief d3bb1dfb (86.37).
  Cycle-1's "86.28" correctly retired. research-gate.js:715 really does label the
  block `phase-86.28` in-source -- a live inconsistency, correctly recorded.
- `const n = (v) => (... ? v : -1)` IS at research-gate.js:632. Sentinel is the
  fail-closed coercion. 86.101 present: pending, P3, 5 criteria; masterplan diff
  in b46f0e17 = 20 insertions / 0 deletions (pure addition).
- BISECT reproduced INDEPENDENTLY (checker+workflows extracted at each ref):
    a212dfe9 2026-08-09 22:36:45 -> ALL GREEN: 87 passed, 0 failed
    089726f9 2026-08-10 08:27:34 -> ALL GREEN: 87 passed, 0 failed
    cad38647 2026-08-10 08:51:11 -> FAILED: 84 passed, 3 failed
    d3bb1dfb 2026-08-10 17:34:06 -> FAILED: 84 passed, 3 failed
  Break = cad38647 (phase-86.6). Filed 86.37 attribution is wrong; Main corrected it.
- "no step closed in the red window on this gate": DERIVED from masterplan (not
  typed) -- the only steps referencing the checker anywhere are 86.17 (done),
  86.23, 86.101, 86.92 (pending). 86.17 flipped pending->done between 6763f10f
  (2026-08-09 22:11:01, pending) and eddde6a9 (2026-08-09 22:46:35, done) -- ~10h
  BEFORE the break. Main's "No" reproduces.
- Sibling gates re-run by me: 124/0, 95/0, 38/0, ALL CHECKS PASS. 86.23 chained
  command exit=0. All match.
- 13 KILLED lines in the artifacts are BYTE-IDENTICAL to my fresh run's grep.
- check() counts: 23 -> 31 -> 32 total occurrences (22 -> 30 -> 31 call sites).
  Main's counting rule is stated and reproduces; reconciles with cycle-1's 23->31.
- F1 geometry: region 33457..44477, `const selfReported` @36913. Exact.

## D. MY OWN ADVERSARIAL MUTATION MATRIX (13 cells, run out-of-repo)
D.0 THE DECISIVE DIFFERENTIAL (kills the sycophancy hypothesis):
    same both-strip-ops-inert mutant vs cycle-1 checker (b46f0e17) -> SURVIVED rc=0
    same mutant vs cycle-2 checker (HEAD)                          -> KILLED  rc=1
    => the cycle-2 edit is CAUSALLY what closed the cycle-1 finding.
M1 both-strip-ops inert            KILLED (1 cell: the stripper control)
M2 block-comment-strip only inert  SURVIVED  <-- see NOTE-1
M3 line-comment-strip only inert   KILLED
M4 poison anchored at slice START  KILLED (3 cells, incl. the containment
   (reproduces cycle-1's exact defect) assertion "poison IS visible w/o stripping")
M5 region slicing inert (whole file) SURVIVED <-- equivalent (see NOTE-2)
M6 EG_END marker moved             KILLED (4 cells, anchored fires)
M7 fixture drops a field           KILLED (5 cells, canary names it)
M8 schema GROWS a required field   KILLED (exactly 1 cell, naming it) <-- criterion 4
M9 region anchored but TRUNCATED   KILLED (consumed canary: "reads 0 field(s)")
M11 enforceGate blind guard removed KILLED (3 cells) <-- criterion 7 direction
M12b future rule tightens a VALUE
    on an existing field           KILLED (3 cells, old prose signature) <-- NOTE-3
M13 future rule reads an UNDECLARED
    verification field             KILLED (2 cells, both naming it)
F1a block-comment poison, stripper INTACT -> green (correct)
F1a + block-strip neutered (2-factor)     -> RED, 2 cells naming __docOnlyField__

NOTE-1 (block-comment strip). The positive control injects a LINE comment, so it
exercises only one of the stripper's two operations. Measured equivalence TODAY:
all four scan variants (both-strip / line-only / block-only / no-strip) and even
a WHOLE-FILE scan return the identical 7-field set, and there are 0 block
comments inside the enforceGate region -- so M2 is an EQUIVALENT MUTANT, not a
live hole. REACHABILITY established by the 2-factor experiment above: once any
`/* ... verification.X ... */` lands in enforceGate the block-strip becomes
load-bearing and its failure mode is a FALSE RED. One-line fix: inject the poison
as a block comment too (or assert each strip op separately). NOTE, not WARN: the
control genuinely discriminates (kills M1/M3/M4) and no criterion's sole coverage
depends on the uncontrolled half.

NOTE-2 (region slicing). Whole-file scan == region scan today, so M5 is
equivalent. The DANGEROUS direction (anchored but truncated) IS caught -- M9.

NOTE-3 (durability bound). Criterion 4 offers a disjunction and Main took the
second branch (synthetic fixture owned by the checker), proven by M8/M13. Residual
worth stating: a future rule that TIGHTENS a value on an already-present field
still surfaces as the old 3-cell prose signature rather than a named canary --
though the message now names the fixture's own field/value
(`brief too short: char_count=9000`) instead of a brief on disk, so the diagnosis
is materially better than the 2026-08-10 case. Not a criterion miss.

NOTE-4 (cross-source date). harness_log Cycle 197 headers 86.17 as
`2026-08-10 result=PASS`; the masterplan flip to `done` was 2026-08-09 22:46.
Main used the flip date and is CORRECT; a reader cross-checking the log alone
would get a different answer. Log is secondary per qa.md.

## E. CRITERION MAP (all 7 MET, each by execution)
1 MET  84/3 reproduced; cause localised by driving enforceGate (5 variants).
2 MET  sentinel at :632 verified deliberate + fail-closed; rendering defect FILED
       as 86.101 (present, pending, 5 criteria, +20/-0).
3 MET  zero bytes changed in research-gate.js / qa-verdict.js / qa.md; and
       positively -- M11 shows the gate still refuses a blind perfect envelope.
4 MET  via the criterion's 2nd disjunct; M8 = 1 named cell, M13 = 2 named cells,
       M7/M9/M6 all fire. Bound in NOTE-3.
5 MET  EXIT 0, 96/0; 13 KILLED cells reproduce byte-identically AND I re-ran the
       underlying mutations myself rather than reading the cells.
6 MET  bisect reproduced row-for-row; ~6d12h; derived-scope answer "no step closed
       in the window" reproduces (86.17 closed ~10h before the break).
7 MET  nothing that computes a verdict was touched; no rule relaxed; the fixture
       change moves the FIXTURE, not the gate.

## F. VERDICT-SEQUENCE EVIDENCE (reported, not aggregated)
qa_wip.py 86.92 --spawned-at 2026-08-16T19:24:29Z:
  source_present: True | attempt_number: 2 | prior_attempts: 1
  attempt_number_status: ok | attempt_number_is_lower_bound: False
  records_retained: 2 (GAUGE, not a counter) | records_pruned_known: None
verdict_history_86_21.py --step 86.92 --evidence-only:
  status: no_rows_for_step | verdicts: (none)
CROSS-CHECK: attempt_number (2) > ledger rows (0) => the LEDGER IS STALE for this
step; sequence from the ledger is UNKNOWN. Main's advisory disclosure (cycle 1 =
CONDITIONAL, wf_1afa11f6-75a) is corroborated by the one prior WIP record on disk
(verdict_wip_86.92__20260816T190329Z.md) but is not ledger-confirmed.

## G. OUTCOME
PASS. Every criterion met with executed evidence; harness compliance clean; no
unintended production change; the cycle-1 blocker is causally fixed (D.0).
Three NOTE-level residuals recorded above; none caps the verdict.

COMPLETED: 2026-08-16T19:47:11Z
