STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.21
WRITTEN: 2026-08-11T07:39:15Z

CYCLE 5. Prior: c1 COND, c2 COND, c3 FAIL (reset), c4 COND -> ONE consecutive CONDITIONAL stands.

## A. HARNESS COMPLIANCE -- CLEAN (5/5)
1. Research gate: research_brief_86.21.md envelope gate_passed=true, external_sources_read_in_full=7
   (>=5), urls_collected=26 (>=10), recency_scan_performed=true, tier=moderate. Brief is
   byte-identical to HEAD (git status clean) though mtime moved to 07:32:49Z -- mtime-only touch.
2. Contract-before-generate: research_brief + contract first committed dc621419 2026-08-10
   00:01:20 +0200; verdict_history_86_21.py first committed 7897cb8c 00:06:37 +0200.
   Contract < code. (Research and contract share one commit, so git cannot separate them --
   the envelope + citation carry that half.)
3. experiment_results_86.21.md present (mtime 07:38:06Z = commit 5b7966e8).
4. LOG-is-last: masterplan 86.21 status=pending, retry_count=0 (< max_retries=3). harness_log
   carries exactly ONE row for the step: ":32340 ## Cycle 199 -- 2026-08-10 -- phase=86.21
   result=FAIL". Zero result=CONDITIONAL rows -> 3rd-CONDITIONAL escalation NOT armed.
   Cross-checked against the disclosed history (COND,COND,FAIL,COND -> consecutive=1). Agree.
5. No verdict-shopping: evidence CHANGED. 5b7966e8 = 6 files, +405/-57, touching BOTH scripts
   and all four handoff artifacts. Documented cycle-2 flow.

## B. DETERMINISTIC
- IMMUTABLE CMD `bash -c 'grep -c "^## Cycle" handoff/harness_log.md && ls handoff/current/
  evaluator_critique_*.md | head -3'` -> 1214 + three filenames, exit=0.
  ON THE RECORD SINCE CYCLE 1: this command cannot go red for any defect this step could have.
  Its exit 0 is evidence of nothing. Everything below is my own re-derivation.
- No unintended production change. git diff --name-only HEAD = my own WIP file + two
  hook-appended audit JSONLs. Untracked = my WIP, a peer session's verdict_wip_86.38, and
  handoff/current/research_brief_86.33.md (a PEER SESSION's file, not attributable to 86.21).
  Zero backend/**, zero frontend/**, zero production module. Gates 1b/1c/1d do not bind.
- Derived .py scope (git the authority: diff HEAD U ls-files --others U 5b7966e8 file list;
  empty-set guard armed; files passed as separate array args, no zsh word-split): 2 files.
  ruff --select F821,F401,F811 -> "All checks passed!", exit=0.
  RUFF LIVENESS PROBE: `echo 'x = undefined_name_zzz' | uvx ruff check --select F821 -` -> F821
  raised, exit=1. Tool proven alive; the clean result is a real zero.
- ast.parse OK on both files.
- REPRODUCED self-test: 20 case lines, SELF-TEST PASSED, rc=0.
- REPRODUCED matrix: 16 cells, ALL 16 KILLED, rc=0, md5 142f6befbd7fc96689f568cb16b98820,
  integrity unchanged.

## C. ATTACK (a) -- IS THE NEW SELF-CHECK GENUINE, OR CIRCULARITY ONE LEVEL DOWN?
Harness: in-memory copies of mutation_matrix_86_21.py, TARGET re-pinned to the real
verdict_history file, plus an INJECTED guaranteed-survivor cell ZZ (a pure comment rewrite,
cannot change behaviour). Repo never written; both md5s verified unchanged after every run.

CONTROL (unmutated matrix + ZZ): rc=1, "16 of 17 killed. SURVIVORS: ZZ".
  -> the probe discriminates. A real survivor is reported SURVIVED.

SC1 -- RESTORE THE CYCLE-3 DEFECT (`return "broken", (f"mutant failed to LOAD` ->
       `return "killed", ...`):
  rc=5. "[broken-scoring self-check] uncompilable mutant -> 'killed' (WRONG: a mutant that
  never ran was scored as a result)" ... "REFUSING TO SCORE -- the BROKEN path could not be
  verified."
  ==> THE CYCLE-5 FIX IS REAL. Confirmed independently. Claim (1) in the spawn prompt holds.

SC2 -- ALWAYS-KILLED SCORING (`return ("killed" if r != 0 else "survived"), ...` ->
       `return "killed", ...`):
  rc=0. Both self-check cells report "correct". ZZ -- a mutant that CANNOT change behaviour --
  is reported "KILLED  | ZZ". Summary: "ALL 17 MUTANTS KILLED -- every guard IN THIS MATRIX
  can fail."
  ==> ***A SCORING DEFECT THE SELF-CHECK MISSES, EXECUTED AND CONFIRMED.***
  verify_broken_scoring() pins the "broken" outcome and the "killed" outcome and pins NO
  "survived" outcome. Its own docstring states the reasoning -- "a version that simply returns
  'broken' for everything cannot pass" -- i.e. it is armed against always-BROKEN, which fails
  CLOSED (rc=5, harmless), and blind to always-KILLED, which fails OPEN (rc=0, false green over
  the entire matrix). That is the identical one-sided-guard signature the cycle-2 Q/A named for
  `c >= 2` vs `c >= 1`, recurring inside the self-check for the self-check.
  FIX (one line, same shape as the two existing cells): a third cell driving score_cell with a
  COMPILABLE guard-irrelevant mutant (a comment rewrite) asserting outcome == "survived".

## D. ATTACK (b) -- ARE A3/A5-A10 A DEFENSIBLE SCOPE CALL? -> YES, DEFENSIBLE.
Built 8 print-layer mutants of _report myself and scored them through the production score_cell.
ALL 8 SURVIVE (self-test rc=0): B1 ordinal, B2 status line, B3 verdicts line, B4 detail line,
B5 ledger_empty NOTE, B6 ledger_missing NOTE, B7 CAUSE-branch numbers, B8 printed contrast.
Per survivor_needs_behavioural_differential I measured rather than filed on sight:
 - B1 (ordinal `c + 1` -> `c`) on a genuinely ARMED step (2 consecutive CONDITIONALs):
     control "auto-FAIL armed : True  (a further CONDITIONAL would be the 3rd)"
     B1      "auto-FAIL armed : True  (a further CONDITIONAL would be the 2nd)"
   The load-bearing boolean stays TRUE and correct; only the parenthetical is wrong.
 - B8 (contrast hardcoded 0) on 36.17: prints "prescribes: 0 row(s)" and then, one line below,
   "DISAGREEMENT: ledger says 0, harness_log grep says 3" -- the CORRECT value still prints.
   Self-contradicting screen, not a silent falsification. My initial hypothesis that B8
   falsifies criterion 1's contrast is WRONG in its strong form; recording that.
 - B5/B6 remove prose while `consecutive : NOT KNOWABLE`, `armed : UNKNOWN` and exit=1 survive.
JUDGMENT: the two the author DID guard (A1/A2) are exactly the two that falsify the armed flag
itself. That is correct prioritisation, not an under-fix. THE GAP IS DISCLOSURE, not scope:
experiment_results_86.21.md never states that 7 print-layer mutants were knowingly left
unguarded or why. (It IS on the record in evaluator_critique_86.21.md:391, transcribed verbatim.)

## E. ATTACK (c) -- IS "ADVISORY" ADEQUATE FOR CRITERION 4? -> YES, AND VERIFIED.
The module (:40-48) and §5 state plainly that a Main-supplied count is ADVISORY, not
authoritative, name the residual (a writer Main does not control), and record it NOT DONE in §8.
Auditability claim CHECKED rather than accepted: handoff/verdict_ledger.jsonl IS git-tracked
(git ls-files hit) and NOT gitignored (git check-ignore exit=1) -- so it survives the *.log
gitignore trap that has defeated this project before. git log shows 2 commits touching it.
This is the strictly weaker claim, honestly made and not dressed up. ADEQUATE.

MEASURED CAVEAT (the most material residual in the step):
  $ python scripts/qa/verdict_history_86_21.py --step 86.21
  status : ok | detail : 3 verdict(s) | verdicts : CONDITIONAL -> CONDITIONAL -> FAIL
  consecutive : 0 | auto-FAIL armed : False  (a further CONDITIONAL would be the 1st)   exit=0
  TRUE history is FOUR verdicts -- cycle 4's CONDITIONAL (wf_982cd319-493) was never appended;
  ledger mtime is 2026-08-09T22:35:42Z, two days stale. So consecutive is really 1 and a further
  CONDITIONAL would be the 2nd. A well-formed-but-STALE ledger is a FIFTH failure mode with no
  status: not missing, not empty, not corrupt, not no-rows. It reports `ok` at exit 0 and
  UNDER-COUNTS -- the fail-OPEN direction, on this very step, at evaluation time.
  NOT a criterion miss: criterion 5 says "missing or unreadable", criterion 6 says "corrupt or
  empty"; staleness is outside both words. §8 discloses the MECHANISM generically. Nothing
  states it has ALREADY happened.

## F. CLAIM AUDIT (qa.md 4b) -- THREE CAPTURES IN experiment_results_86.21.md DO NOT HOLD
1. §2, the CRITERION-1 EVIDENCE BLOCK, IS THE WRONG CAPTURE. Prose: "the reproduction is built
   on phase-86.20's real mid-flight state, replayable by anyone:" -- the fence that follows is
   the MUTATION MATRIX output, byte-identical to §7's block. The prose after it ("Two recorded
   verdicts, status still pending, and the grep the rule prescribes returns ZERO") cites numbers
   that appear nowhere in the block. `grep -c "688ac349\|7145f566" experiment_results_86.21.md`
   = 0: the actual reproduction is absent from this file entirely.
   I RE-RAN IT MYSELF from live_check §2 and it HOLDS: 688ac349 -> grep 0 / 1 critique header /
   masterplan 86.20 = pending; 7145f566 -> grep 0 / 2 headers / pending. Criterion 1 is
   substantively MET; its experiment_results evidence is a duplicated block.
2. §6's SELF-TEST capture is STALE: 15 case lines vs 20 emitted today. Missing (vi-b) (vi-c)
   (vi-e) (vi-f) (vi-d) -- i.e. the cycle-4 AND cycle-5 additions. This is the block that
   carries criterion 5.
3. §10 pastes `$ ... | grep -cE '^   \('` -> `18`; the command returns `20`. Three lines below,
   the SAME section's prose says "cycle 5 measures 20 cases / 16 cells". The section whose whole
   subject is "state the rule and count under it" contradicts itself.
   Cycle 5's sweep was a TOKEN sweep (9ece5e79 / "ALL 11 MUTANTS" / "15 self-test cases"), so it
   could not catch a stale COUNT or a wrong BLOCK. Third cycle of this class.
Direction of all three: UNDER-claiming. No inflated number found. Checked and NOT findings:
live_check §4/§5's "11 rows" describe the cycle-2 measurement and the seeding act (ledger is 14
rows now: 36.17 x6, 86.20 x3, 86.17 x2, 86.21 x3) -- dated captures, correctly scoped.
4. STRUCTURAL: §9 = cycle 2, §10 = cycle 3, and there is NO cycle-4 or cycle-5 section. Every
   prior cycle got its own narrative; these two did not.

## G. CRITERION MAP -- ALL SIX MET
 C1 MET  -- reproduction re-derived by me at both commits (grep 0, headers 1 then 2, status
            pending) and reproduced LIVE on 86.21 (0 CONDITIONAL rows, 4 verdicts, pending).
            Evidence present in live_check §2; experiment_results §2 carries the wrong block.
 C2 MET  -- separate append-only ledger; harness_log.md absent from 5b7966e8's file list;
            reason stated at verdict_history_86_21.py:16-31 and §3, incl. why
            evaluator_critique_<id>.md was rejected (17+ filename shapes, depth-1 vs depth-2).
 C3 MET  -- self-test (i) loads exactly CONDITIONAL,FAIL,FAIL,CONDITIONAL,CONDITIONAL ->
            consecutive=2, armed=True; reset-on-FAIL exercised. Live 6-row history -> 0 with the
            sixth PASS DISCLOSED rather than trimmed to match the criterion's wording.
 C4 MET  -- ADVISORY not authoritative, residual named, NOT-DONE recorded; auditability claim
            verified by me (tracked + not gitignored).
 C5 MET  -- four statuses, direction stated per status, asserted through the real entry point
            with exit codes 1/1/1/0/0. Residual: staleness has no status (outside the wording).
 C6 MET  -- corrupt/empty/missing/blank-field/no-step_id all NOTICE; 16/16 killed with control
            green and md5 stable, reproduced by me; and SC1 independently confirms the cycle-5
            scoring fix. Residual: the scoring self-check is one-sided (SC2).

## H. CODE-REVIEW HEURISTICS
No security, no trading-domain findings. Two read-only scripts under scripts/qa/ + one JSONL;
no kill-switch / stop-loss / perf-metrics / execution / signal path; no secrets; nothing under
backend/** or frontend/**, so 1b/1c/1d do not trigger and NO Playwright capture was needed (no
UI claims). importlib/exec_module operate on the repo's own file in a tempdir with no external
input -- not command-injection per the negation list. #17 illusory-guard fires at WARN (a
coverage gap alongside genuine behavioural guards -- the matrix DOES discriminate today, proven
by my ZZ control), which caps at CONDITIONAL rather than BLOCK.

## I. VERDICT SHAPE
worst-of-lenses: correctness = PASS-level (SC1 proves the fix; arithmetic correct);
does-it-reproduce = CONDITIONAL (F1/F2/F3); scope-honesty = CONDITIONAL (F4 + the undisclosed
7-survivor residual + the undisclosed live staleness). min() = CONDITIONAL.
3rd-CONDITIONAL check: 0 result=CONDITIONAL rows for 86.21 in harness_log; disclosed history
gives consecutive=1. NOT armed. CONDITIONAL is legitimate and is what I return.
NOT downgraded to FAIL: every non-reproducing figure UNDER-claims, the substantive criterion-1
evidence exists and I re-ran it, and the central thing I was asked to attack (the cycle-5
scoring fix) genuinely works.

COMPLETED: 2026-08-11T07:48:15Z
