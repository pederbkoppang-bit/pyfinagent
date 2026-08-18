STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.37
WRITTEN: 2026-08-17T14:34:37Z

# Q/A write-first record -- step 86.37 (cycle 6 re-evaluation)

Spawned via Workflow rail. Read .claude/agents/qa.md in full at 14:34Z.

## Task framing supplied by Main
Cycle 6 re-evaluation. Predecessor (cycle 5) graded all six criteria MET,
compliance CLEAN, named exactly ONE clearing action: replace the sentence at
live_check_86.37.md:105-107 so the named site carries the 86.81/6b4df8f9/8b520f6c
attribution instead of the retired 86.28 one. Main claims that landed in 26d840c9.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable verification command exit code; git scope check; lint
C. Re-derive the cycle-6 claim independently (grep, git show, diff)
D. Mutation / vacuity spot-checks on the guards the criteria rest on
E. Criterion-by-criterion MET/NOT MET

## Findings log (append-only)

### F1 -- prior-attempt / prior-verdict EVIDENCE (gathered, not a trigger)
`python scripts/qa/qa_wip.py 86.37 --spawned-at 2026-08-17T14:34:37Z`:
  source_present=True; records_retained=4 (GAUGE, incl. my own record);
  attempt_number=4; prior_attempts=3; attempt_number_status='ok';
  attempt_number_is_lower_bound=True; records_pruned_known=None; is_verdict=False.
  prior_records: verdict_wip_86.37__20260817T142211Z.md,
  verdict_wip_86.37__20260817T140430Z.md, verdict_wip_86.37.md (unstamped, legacy).

`python scripts/qa/verdict_history_86_21.py --step 86.37 --evidence-only`:
  status = ok; detail = "4 verdict(s) from the ledger";
  verdicts = FAIL -> CONDITIONAL -> CONDITIONAL -> CONDITIONAL.

CROSS-CHECK: attempt_number (4) is NOT > ledger count (4), so the qa.md staleness
rule does not fire. BUT Main frames this as CYCLE 6 and evaluator_critique_86.37.md
is said to carry cycles 1-5 -- so the ledger's 4 rows do not cover 5 graded cycles.
I report the sequence and this gap as observed; I compute no aggregate over it.

### F2 -- immutable criteria are UNAMENDED
Walked `.claude/masterplan.json` at 30 successive commits touching it: step 86.37
`verification.success_criteria` is len=6 with md5 6ecc875595d9 at EVERY one
(26d840c9 .. c627a810). No criteria erosion, no amendment.
NOTE: the spawn prompt lists SEVEN paraphrased criteria; the masterplan's SIX
govern (per my instruction). Prompt item 5 ("the rules/research-gate.md and
researcher.md contradiction ... reconciled, with the reconciliation quoted") has
NO counterpart in the masterplan criteria -- it is a cycle-1 finding carried
forward, not an immutable criterion. Prompt item 7 == verification.live_check.

### F3 -- IMMUTABLE VERIFICATION COMMAND: EXIT=0
`bash -c 'node --check .claude/workflows/research-gate.js && node scripts/qa/verify_research_gate_workflow.mjs'`
=> "ALL GREEN: 124 passed, 0 failed", EXIT=0. Re-run by me, not quoted.
Sections present: [1] control, [2] floors, [3] empty/errored, [4] artifact
cross-check, [5] stage-2 fail-closed, [6]/[6b]/[6c]/[6d] anti-self-grade + tier +
BEHAVIOURAL spawn recorder, [7] 9 enforceGate mutants, [7b] 5 driver mutants,
[8] structural incl. born-inert marker + stage-1 wrap proximity pin.

### F4 -- the CYCLE-6 CLAIM re-derived INDEPENDENTLY (the one thing that changed)
Claim at live_check_86.37.md sec.6: "the +3 are the stage-1 RETRY assertions
added 2026-08-14 by phase-86.81's work, commits 6b4df8f9/8b520f6c; 86.28's
checker commits predate the cycle-3 baseline and were already inside the 121."

Re-derived HERMETICALLY (scratchpad mini-repos; tracked tree never written):
extracted BOTH research-gate.js and verify_research_gate_workflow.mjs at each
commit that touched the checker, ran each pair in its own dir.
  23270f29 (cycle-3 baseline, 2026-08-10T18:03) -> 121 passed, rc=0
  6b4df8f9 (2026-08-14T12:15)                   -> 124 passed, rc=0
  8b520f6c (2026-08-14T13:49)                   -> 124 passed, rc=0
  HEAD                                          -> 124 passed, rc=0
SYMMETRIC DIFFERENCE of check NAMES (not cardinality -- per the 75.5 instance-#10
rule that equal counts can cover different members):
  23270f29 -> 6b4df8f9: +3 / -0, and the 3 are EXACTLY
     "a SINGLE stochastic drop is RETRIED, not surfaced as a dropped run"
     "...and the retried run reports NO rail_dropped (the drop was recovered)"
     "...and the recovered run PASSES the gate, so a retry is a real recovery not a downgrade"
  6b4df8f9 -> 8b520f6c: +0 / -0 (member-wise identical)
  8b520f6c -> HEAD:     +0 / -0
`git log -- scripts/qa/verify_research_gate_workflow.mjs` shows the ONLY commits
after 23270f29 are 6b4df8f9 and 8b520f6c. All five 86.28 checker commits
(d2e987f1 a6c3c3f3 49793961 294a9a09 d638a3ec, all 2026-08-10 morning) predate
23270f29 (18:03 same day) -- so they WERE inside the 121, as claimed.
=> The corrected attribution REPRODUCES. It also fixes the exact defect my
memory `feedback_carried_forward_residuals_go_stale` names (a +3 credited to a
commit made BEFORE the baseline): the credited commits are now both AFTER it.
NOTE-level precision only: all 3 checks landed in 6b4df8f9; 8b520f6c changed the
checker (61,656 -> 62,061 bytes) but added ZERO check names. The artifact says
"phase-86.81's work, commits 6b4df8f9/8b520f6c" -- a work-pair attribution, not
a per-commit split -- so this is a refinement, not a falsification.
grep -c on live_check_86.37.md: 6b4df8f9=1, 8b520f6c=1, 86.81=1, 86.28=2.
TRAP HIT AND RECORDED: my first reconstruction loop used `$C:path` unbraced; zsh
applied the `:s` history modifier and git received a mangled ref, so ALL FOUR
trees (incl. the HEAD control) reported rc=1/0-checks. The control is what
exposed it. Re-run with `"${C}:path"`. Same family as
reference_zsh_no_word_splitting.

### F5 -- HARNESS COMPLIANCE (5 items)
1. RESEARCH GATE: no `research_brief_86.37.md` exists anywhere under handoff/.
   The contract §1 DISCLOSES this: the gate was REUSED (`research_brief_86.31.md`
   -- exists; envelope tier=complex, external_sources_read_in_full=12,
   urls_collected=64, recency_scan_performed=true, gate_passed=true) because the
   rail being repaired IS the rail that runs the gate. OPERATOR-RATIFIED:
   `handoff/current/operator_asks_2026-08-11.md:81-84` -- "ANSWERED 2026-08-17
   (attended session, AskUserQuestion): 'Ratify the reuse (Recommended)'".
   => COMPLIANT by operator ruling, not by my discretion. Verified at source.
2. CONTRACT-BEFORE-GENERATE: contract_86.37.md mtime 2026-08-10T17:25:58 local;
   first code commit d3bb1dfb 2026-08-10T17:34:06. Contract precedes code by 8m.
3. experiment_results_86.37.md present (20,147 B), live_check_86.37.md present.
4. LOG-LAST: masterplan 86.37 status = "pending" (NOT flipped). OK.
5. NO VERDICT-SHOPPING: evidence CHANGED between cycle 5 and 6 -- commit
   26d840c9 modified live_check_86.37.md (17 lines), experiment_results (+20),
   evaluator_critique (+70). The changed file is the one the cycle-5 finding named.

### F6 -- MY OWN MUTATION MATRIX (independent of the author's)
Hermetic mini-repos under scratchpad/mut*, HEAD sources copied in; tracked tree
never written. CONTROL FIRST.
  C0-CONTROL                          rc=0 124 ok / 0 failed  (baseline holds)
  Q1 rail_dropped field DELETED       KILLED  117/7 (named: "the dropped run reports rail_dropped.dropped === true")
  Q2 drop reports dropped:false       KILLED  119/5 (same named assertion)
  Q3 catch FABRICATES a floor-passing envelope (F5 salvage anti-pattern)
                                      KILLED  123/1 -- BUT only by the SOURCE-SCAN
                                      "the drop path does NOT assign gate_passed
                                      anywhere in its catch block". The behavioural
                                      drop assertion stayed GREEN because the
                                      recompute-from-disk still refused it, i.e. the
                                      mutant was a smell, not a defect. Kill mechanism
                                      NAMED rather than assumed (vacuity shape #11).
  Q4 STAGE1_MAX_ATTEMPTS 3->1         KILLED  121/3 (retry assertions are load-bearing)
  Q5 STAGE1_MAX_ATTEMPTS 3->50        SURVIVED 124/0  -- see F8, NOTE-level
  Q6 marker check fail-open           KILLED  122/2 ("brief_status=INCOMPLETE FAILS
                                      the gate..." + "...the violation NAMES the marker")
  Q7 gate_passed = railDropped ? true : enforced   KILLED 119/5 BEHAVIOURALLY
                                      ("a DROPPED run returns gate_passed === false
                                      even with a PERFECT stage-2 verification")
  Q8 "a recovered COMPLETE brief counts as a pass" (the subtle F5 fallacy)
                                      KILLED  119/5 by the SAME behavioural assertion
  Q9 FIXTURE mutant -- the checker's drop stub no longer THROWS
                                      KILLED  122/2  => the fixture CAN represent the
                                      failure; vacuity shape #5 cleared on the drop path.
  Q10 HARNESS mutant -- check() forced always-true   SURVIVED 124/0. EXPECTED and NOT a
                                      finding: an assertion harness cannot detect its own
                                      neutering. Recorded as a methodological control.

### F7 -- CRITERION 1/2/3 REPRODUCED BY ME, END TO END
Drove the REAL workflow source (wrapped as `__drive(args, phase, log, agent)`;
this executes the shipped file, it is not a re-implementation) with a stage-1
agent stub that THROWS, and a brief on disk carrying 12 read-in-full URLs, a
snippet table, a recency section and `brief_status: COMPLETE`, plus a stage-2
stub returning a PERFECT verification (the input most favourable to a wrong pass):
  d3bb1dfb~1 (PRE-FIX) : THREW -- NO RETURN VALUE ("agent({schema}): subagent
                          completed without calling StructuredOutput"), 1 spawn.
  HEAD (POST-FIX)      : RESOLVED, 4 spawns (3 stage-1 retries + stage-2),
                          gate_passed=false,
                          rail_dropped={dropped:true, error:"agent({schema}) ..."},
                          violations=["empty_or_errored_return"],
                          brief_verification populated with all 6 keys
                          (brief_exists, brief_bytes, urls_found_in_brief,
                           recency_section_present, distinct_urls_in_brief,
                           brief_status_in_brief).
=> criteria 1, 2 and 3 independently reproduced, not read.

### F8 -- CRITERION 6 both halves reproduced by me
  U1-UNWRAP-STAGE1-TRYCATCH (revert the try/catch to the bare pre-fix await)
     -> 108 ok / 16 FAILED. NAMED assertions red, first among them
     "a stage-1 DROP does not kill the workflow -- the driver RESOLVES (kills
     QA-RETHROW)". Matches the author's reported M1 108/16 exactly.
  Drop path forced to gate_passed:true -> Q7/Q8 above, KILLED behaviourally.
=> criterion 6's two named halves are both reproduced independently.

### F9 -- the DISCLOSED residual (M9/M10) verified, INCLUDING its direction of harm
Deleting the born-inert TEACHING from the stage-1 prompt (752 B, the STEP 0b
block) OR from the stage-2 prompt (602 B, the brief_status_in_brief instruction)
leaves the checker 124 ok / 0 failed -- SURVIVED, exactly as the artifact
discloses. So the prompt-teaching half of criterion 4 is not pinned by a test.
DIRECTION OF HARM MEASURED (drove the real workflow, all three marker states):
                   HEAD            teaching-deleted
  COMPLETE      -> gate_passed=true    gate_passed=true
  INCOMPLETE    -> false, violation NAMES the marker    (identical)
  ABSENT        -> false, "carries NO brief_status marker" (identical)
So deleting the teaching cannot manufacture a PASS: a researcher that stops
writing the marker produces ABSENT, and ABSENT is a loud named FAILURE on every
run. The gap is that the teaching is unpinned, not that the gate can be fooled.
=> queue-class (evidence-quality), NOT a criterion miss. Correctly disclosed.
PROBE HYGIENE: my first two direction-of-harm probes came back RED for the WRONG
reason (I omitted `brief_non_empty` from the stage-2 stub, so every case hit "is
EMPTY"). Fixed the probe, not the reading -- a red that indicts the probe is not
a finding (feedback_a_red_check_may_indict_the_probe).

### F10 -- FINDING: live_check section 3 is a SPLICED capture (queue-class)
live_check_86.37.md:12 claims "Every block below is regenerated from the shipped
tree." Section 3's fenced block pairs the header
  `[5] stage-2 verification is itself load-bearing -- absent verification FAILS CLOSED`
with six ok-lines that belong to section `[8] structural`, and OMITS a seventh
member of that group ("ABSENT is reported DISTINCTLY from INCOMPLETE, not folded
into it"). MEASURED: at EVERY tree in this step's history the born-inert block
sits under `[8]`, never `[5]` --
  d3bb1dfb 110 ok, 133060b0 117 ok, 23270f29 121 ok, HEAD 124 ok -> all `[8]`.
No run of this checker produces that header/body pairing, so the block was
assembled, not regenerated. This is the same class the artifact itself confesses
at :86-88 ("a number in a block labelled verbatim has to reproduce").
MATERIALITY: nil for the criterion. All 7 marker assertions exist and pass at
HEAD (I ran them), and I additionally re-derived the marker semantics through
the real driver in F9. The defect is provenance/labelling in a cycle-3-era
block, not a false behavioural claim.

### F11 -- Q5 SURVIVOR, out of this step's scope (report, do not cap)
STAGE1_MAX_ATTEMPTS 3 -> 50 survives at 124/0: nothing pins the retry bound.
The constant is phase-86.81's (commit 6b4df8f9), NOT 86.37's, and no 86.37
criterion mentions it. Shape is `unbounded-llm-loop` (OWASP LLM10 denial-of-
wallet) at WARN in the code-review skill -- each extra attempt is a real
`agent()` spawn on the shared weekly Max pool. Queue-class observation.

### F12 -- code-review heuristics (5 dimensions evaluated)
26d840c9 touches NO production code: masterplan.json (adds NEW pending steps
86.109/86.110, disclosed in the commit message as the concurrent intraday
session's filings; 86.37's own criteria md5 unchanged -- see F2),
attempt_budget_audit.jsonl(+1), verdict_ledger.jsonl(+1), and the three 86.37
handoff artifacts. No secret, no exec/eval, no trading-path change, no dep pin.
Python lint gate: derived scope `git diff --name-only HEAD -- '*.py'` =
backend/api/sovereign_api.py (NON-EMPTY, asserted before reading the exit code);
`uvx ruff check --select F821,F401,F811` -> "All checks passed!", exit 0.
That file and the 5 modified frontend files are UNCOMMITTED work belonging to a
CONCURRENT session's Red-Line/sovereign intraday change, not to 86.37; 26d840c9
does not touch them. Frontend eslint/tsc gate NOT APPLICABLE to 86.37's diff
(it touches no frontend/**); backend smoke NOT APPLICABLE (no backend/**);
live-UI gate NOT APPLICABLE (the step makes no UI claim).

### F13 -- ledger discrepancy from F1 RESOLVED
`grep -F '"86.37"' handoff/verdict_ledger.jsonl` -> 4 rows, cycles 1, 2, 4, 5
(FAIL, CONDITIONAL, CONDITIONAL, CONDITIONAL). Cycle 3 has NO ledger row because
it ended PARKED -- `handoff/harness_log.md` carries exactly one 86.37 row:
"## Cycle 1204 -- 2026-08-10 -- phase=86.37 result=PARKED". So the ledger is
self-consistent with a 5-cycle history, not stale. Under the corrected staleness
rule shipped 3 minutes into this evaluation (2dbe09d4: compare `prior_attempts`,
not `attempt_number`), prior_attempts 3 > 4 rows is FALSE -- no staleness signal.
The WIP-record shortfall is expected: cycles 1-2 (2026-08-10) predate the stamped
write-first convention for this step.

### F14 -- criterion-by-criterion (masterplan's SIX, which govern)
1 REPRODUCE FIRST                      MET  (F7: prefix THREW / postfix RESOLVED, my drive)
2 drop => gate_passed:false ALWAYS     MET  (F7 floor-satisfying brief + PERFECT stage-2
                                             still false; Q7/Q8 killed behaviourally)
3 recovery report, DISTINCT field      MET  (rail_dropped + brief_verification 6 keys;
                                             Q1/Q2 killed)
4 born-inert envelope, marker semantics MET (researcher.md:332-339 + research-gate.js
                                             STEP 0b; marker gated BEFORE any count;
                                             7 assertions green; Q6 killed; F9 real-driver
                                             sweep COMPLETE/INCOMPLETE/ABSENT)
5 floors + anti-trust UNCHANGED        MET  (FLOOR_SOURCES=5 / FLOOR_URLS=10 at :433-434;
                                             124/0 exit 0 incl. over-claim + self-grade
                                             override + claimed-URL cross-check)
6 MUTATION-TESTED, both named halves   MET  (U1 unwrap 108/16 named red; Q7/Q8 killed)
live_check field (5 named items)       MET in substance; residual F10.

### F15 -- HEAD RECHECK at end of evaluation
HEAD = 61a72837 (07e33d18 phase-86.79) -- unchanged during my run. Two 86.79
commits (2dbe09d4, 07e33d18) landed at 14:37Z, ~3 min after my spawn; neither
touches 86.37's scope (one edits qa.md's staleness rule -- see F13 -- the other a
86.79 handoff file). research-gate.js md5 e26dc258bc862beead7f4a336c978480 and
live_check md5 9935b72ae0e68ffcbd1f80700d94c5dc, `grep -c 6b4df8f9` = 1, all
identical at the start and the end of my evaluation.

### VERDICT REACHED: PASS, with three named queue-class residuals (F9, F10, F11).
All six immutable criteria independently REPRODUCED as MET (not read). Harness
compliance clean on all five, gate reuse operator-ratified at source. No
production code touched by 26d840c9. The residuals are evidence-quality only:
F10 UNDERSTATES the evidence (a mislabelled section header + one omitted PASSING
assertion), F9 is disclosed and measured fail-closed, F11 belongs to 86.81.

COMPLETED: 2026-08-17T14:48:37Z
(self-correction: I first typed an INVENTED 15:02:11Z here without reading the
clock -- `feedback_never_narrate_a_clock_you_did_not_read`. Replaced with the
value `date -u +%Y-%m-%dT%H:%M:%SZ` actually returned.)
