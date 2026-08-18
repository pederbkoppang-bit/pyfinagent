STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.71
WRITTEN: 2026-08-17T11:15:33Z

# Q/A write-first record -- step 86.71 (cycle 3)

## Attempt / sequence evidence (gathered, not applied)
- `qa_wip.py 86.71 --spawned-at 2026-08-17T11:15:33Z`: attempt_number=3
  (status ok, is_lower_bound=true), prior_attempts=2, source_present=TRUE,
  records_retained=3 (gauge), records_pruned_known=null.
- `verdict_history_86_21.py --step 86.71 --evidence-only`: status=ok,
  sequence = **FAIL -> CONDITIONAL** (2 verdicts).
- CROSS-CHECK: prior_attempts (2) == ledger verdict count (2). Ledger NOT stale.
- THIRD corroboration, not one of the two documented sources: the live
  attempt-gate hook wrote a PRODUCTION row for MY OWN launch --
  2026-08-17T11:15:28Z, step_id 86.71, workflow qa-verdict.js,
  attempt_number_inclusive=3, session e6b8ec06 -- five seconds before this
  file's WRITTEN stamp. Three independent counters agree on 3.
- No aggregate computed. Whatever follows from the sequence is the caller's.

## Harness compliance (5 items) -- CLEAN
1. Research gate: research_brief_86.71.md envelope brief_status=COMPLETE,
   external_sources_read_in_full=9, urls_collected=30, snippet_only=21,
   recency_scan_performed=true, gate_passed=true; "## Recency scan (last 2
   years, 2024-2026)" present at :102.
2. Order: research 09:47:50Z < contract 10:07:27Z < code (attempt_budget
   10:44:21Z, attempt_gate 11:08:19Z, matrix 11:07:44Z) < artifacts 11:15:07Z.
3. experiment_results_86.71.md present (Cycles 1-3); live_check sections 1-9.
4. Log-last: `grep -cF "phase=86.71" handoff/harness_log.md` = 0; masterplan
   status=pending. Not yet logged, not yet flipped.
5. Not verdict-shopping: cycle-2 verdict transcribed 11:06:48Z; attempt_gate.py
   11:08:19Z, matrix 11:07:44Z, artifacts 11:15:07Z -- ALL after. Diff vs HEAD
   = attempt_gate +36, mutation_matrix_86_71 +147, experiment_results +53,
   live_check +177. Substantive, not cosmetic.
Contract carries all 8 criteria VERBATIM (byte-compared against masterplan).

## Deterministic
- IMMUTABLE COMMAND: prints `parses`, **exit=0**.
- ruff F821,F401,F811 over DERIVED scope (git diff --name-only HEAD '*.py'
  UNION git ls-files --others --exclude-standard '*.py'; 8 files, non-empty
  asserted, xargs -0): **All checks passed! exit=0**.
- pytest backend/tests/test_phase_86_32_attempt_budget.py: 15 passed, exit=0.
  `test_exhaustion_cannot_auto_pass` present at :102.
- mutation_matrix_86_32.py: exit=0.
- mutation_matrix_86_71.py --verify (my run): CONTROL green 9 checks,
  relocated-unmutated SURVIVES, null-mutant SURVIVES, 7/7 KILLED, 0 survivors,
  0 errors, md5 36758fd2c4779ae667d00abf228aaed7. exit=0.
- VERBATIM-CAPTURE CHECK: live_check section 9's fenced block vs freshly
  regenerated stdout -- 25 lines vs 25 lines, **0 mismatches, IDENTICAL**.
- Self-test: 15 checks green, exit=0. Production ledger 14 -> 14 lines: the
  disclosed cycle-3 def-time-default pollution bug is genuinely FIXED.
- Gates n/a: 1b (graded change has no frontend/**; the dirty frontend files are
  a peer session's, disclosed), 1c (no UI claim), 1d (no backend/** source in
  the graded change; the gate itself was exercised live instead).
- No unintended production change: HEAD cadab378 unchanged start->finish;
  attempt_gate md5 36758fd2..., matrix df1e216c..., audit ledger
  1a82aae4... / 14 lines byte-unchanged after everything I ran. The 6 other
  dirty .py files reference attempt_budget/attempt_gate **0 times** each.

## Criteria, each independently DRIVEN
C1 MET. Ran live_check section 7's script verbatim: runs=494 (no step_id 99),
   repeats 329 = **66.6%**, qa 311/400 = **77.8%**, researcher 18/93 =
   **19.4%**, max qa **9 on 36.8**. It IS runnable from the artifact now (the
   cycle-2 blocker) and reproduces within corpus growth. Disagreement with the
   filed 58.4% is reported, not adopted. NOTE below on section 1's staleness.
C2 MET. Author's control string `attempt_budget` -> hits
   backend/tests/test_phase_86_32_attempt_budget.py (non-zero); runtime
   surfaces empty. My SECOND, independent control string `DEFAULT_MAX_ATTEMPTS`
   -> 4 files; `from attempt_budget import` -> only attempt_gate.py (the new
   wiring) and scripts/qa/verify_counter_86_79.py (a QA verification script,
   outside the author's stated scope -- NOTE, not a runtime caller).
C3 MET, driven by me: 14 SEPARATE OS processes on a temp ledger -- 7 launches,
   each followed by a separate `--status` process: 1/5, 2/5, 3/5, 4/5, 5/5
   CONTINUE/allow, then rc=2 DENY at #6 and #7. Persistence is a file.
C4 MET, driven by me on the REAL production ledger: at-ceiling 999.2 -> exit
   **2**, deny message + escalation file (redirected to temp), and the
   production ledger md5 UNCHANGED (a deny does not append). Below-ceiling
   86.71 = 3/5 CONTINUE/allow and 86.85 = 3/5 CONTINUE/allow -- unaffected.
   jq confirms the PreToolUse/Workflow registration in .claude/settings.json.
   Strongest evidence: MY OWN launch is a production row.
C5 MET, exhaustively by me: 4,368 (non-PASS sequence x flag) cells over lengths
   1..6 -> close_kind value set is EXACTLY {CONTINUE, ESCALATE}; zero paths to
   any CLOSED_*/PASS without a PASS; every at/over-ceiling non-PASS history
   escalates. Positive control: a PASS reaches CLOSED_PASS / CLOSED_COMPLETE,
   so the probe discriminates. Escalation body reads "THIS IS NOT A PASS AND
   NOT A FAIL"; its only "PASS" token is inside that negation.
C6 MET. Every production ledger row is a `qa-verdict.js` launch; the Q/A rail is
   what is being counted and bounded (mine is 3/5).
C7 MET. No .env in the diff or status. The only settings.json change is commit
   192ef652: +11 lines, one PreToolUse/Workflow hook block, no flag promoted.
   ASK-1 recorded in contract_86.71.md:88.
C8 **NOT FULLY MET** -- see F1.

## FINDING F1 (mutation-proven, WARN) -- 1 of the 3 NEW self-test checks is a TAUTOLOGY
`scripts/harness/attempt_gate.py:366-370`:

    check("operator extension WITHOUT --reason is refused",
          cmd_extend("9.4", 1, "   ") == 2)
    before_rows = len(read_ledger(led))          # read AFTER the call
    check("refused extension appends NO row",
          len(read_ledger(led)) == before_rows)  # ledger compared to ITSELF

`before_rows` is captured AFTER `cmd_extend` already ran, so the assertion is
`len(x) == len(x)`. Vacuity shape #4 (tautology true by construction).

MUTATIONS EXECUTED (in-memory exec; tree md5 36758fd2... unchanged throughout;
CONTROL observed GREEN first each time):
  - CONTROL unmutated self-test: rc=0, 0 failing checks.
  - M-A (blank-reason path APPENDS a row but STILL returns 2 -- exactly the
    defect the check names): self-test rc=0, failing_checks=[] -> **SURVIVED**,
    while printing `ok    refused extension appends NO row`.
  - M-B (the matrix's own G7: `if not reason.strip():` -> `if False:`):
    self-test rc=1, FAIL "operator extension WITHOUT --reason is refused" -> the
    FIRST new check is REAL.
  - M-C (accepted extension writes no row): rc=1, FAIL "operator extension WITH
    a reason appends its labelled row" -> the THIRD new check is REAL.
  So: 2 of the 3 new checks are genuine; the middle one cannot fail.

SEVERITY = WARN, not BLOCK, because the behaviour IS covered elsewhere. I ran
M-A through the MATRIX's own checks: relocated-unmutated control SURVIVES;
M-A FAILS 'an operator extension WITHOUT --reason is REFUSED and appends no
row' (extend probe refused_rc=2, rows_after_refusal=1, expected 0) -> KILLED.
`_extend_probe` is a real behavioural guard. qa.md 4c: vacuous guard alongside a
genuine behavioural guard = WARN with a named fix.
NAMED FIX: move `before_rows = len(read_ledger(led))` ABOVE the `cmd_extend`
call (one line).

Bearing on C8: the cycle-3 artifact presents its criterion-8 evidence as "the
7-cell matrix PLUS the self-test", i.e. it claims the self-test as C8 evidence,
but no artifact reverts each new self-test check and shows it red -- and one of
them provably cannot go red.

## FINDING F2 (NOTE/WARN) -- the new ERROR-on-import guard is MARKER-based, not outcome-based
`scripts/qa/mutation_matrix_86_71.py:265-267` scores a cell ERROR when the
mutant's stderr contains ModuleNotFoundError / ImportError / SyntaxError.
I probed it (control green first, tree md5 unchanged):
  - Z1 `from attempt_budget_DOES_NOT_EXIST import ...` -> **ERROR** (correct)
  - Z2 `def _now(( -> str:`                            -> **ERROR** (correct)
  - a NameError raised AT IMPORT TIME (`_UNDEFINED_AT_IMPORT_TIME` at module
    level) -> guard does NOT fire -> scored **KILLED (by 'below-ceiling launch
    is ALLOWED')** -- the same "a mutant that never ran scores as a kill" class
    cycle 1 found, closed for three markers and still open for others.
  - `--verify` correctly returns rc=1 when any cell is ERROR.
Not firing today: all 7 real cells import cleanly and each has an attributable
kill (I printed the FULL failure list for G4 = exactly 1 check, the corrupt
probe; G7 = exactly the 2 extend checks -- no mis-attribution, vacuity shape
#11 clear). experiment_results says "cell-level import breakage now scores
ERROR, never a kill", which is broader than the implementation; live_check
section 8 states the marker list correctly.
NAMED FIX: score ERROR whenever the relocated-unmutated control passes but the
mutant's drive shows a traceback / the gate logic never ran, rather than
matching three exception names.

## FINDING F3 (NOTE) -- live_check section 1's headline figures are a superseded capture
Section 1 displays 481 runs / 320 repeats / qa 302/390 = 77.4% / researcher
18/90 = 20.0%. The artifact's own stated command (section 7) now yields
494 / 329 / qa 311/400 = 77.8% / researcher 18/93 = 19.4%, and Main's own
summary quotes a third set (492 / 327 / 309/398 / 18/93). All three are the same
command at different corpus sizes -- consistent with growth, not fabricated --
but section 1 carries no measurement timestamp and nothing reconciles it with
section 8's "reproduces 66.4-66.5%" (now 66.6%). A reader taking section 1 as
"the re-derived figures" cannot reproduce them.

## FINDING F4 (NOTE) -- synthetic residue in a git-tracked production audit stream
handoff/audit/attempt_budget_audit.jsonl carries 5 `session_id: pipetest` rows
for the synthetic step 999.2 and 1 `operator_extension` row for step 9.4. The
9.4 row IS disclosed (live_check section 9); the five 999.2 rows are narrated in
section 3 but never called out as residue LEFT in the production stream. Carried
from cycle 2, unfixed, no criterion owns it. Append-only, so removal is not the
fix -- a labelled marker row or a note in experiment_results is.

## Carried notes (unfixed, no criterion owns them, not blocking)
- The counter keys on the literal step-id string: 999.2 -> 5/5 deny while
  999.20 / 999.2.0 -> 0/5 allow.
- The PASS exception is permanent per step: a step carrying a PASS is
  un-budgeted thereafter (can only ADMIT more evaluation, never less).
- The Agent-tool fallback path is NOT gated. Now disclosed at the source
  (attempt_gate.py docstring) and in live_check section 7(3), so it is a stated
  residual rather than a silent hole -- but it is the documented next move after
  exactly the drops this budget bounds.
- History is not backfilled: counting starts 2026-08-17.

## Writes
Only this file, inside .claude/agent-memory/qa/verdicts/. Every drive used
ATTEMPT_GATE_LEDGER / ATTEMPT_GATE_VERDICT_LEDGER / ATTEMPT_GATE_ESCALATION_DIR
overrides into Python tempfile directories; every mutant was exec'd from memory
or from a temp copy. No write was blocked.

COMPLETED: 2026-08-17T11:29:41Z
