STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.85
WRITTEN: 2026-08-15T13:45:10Z

# Q/A write-first record -- step 86.85 (verdict ledger writer)

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable verification command + git scope + lint + tests
C. LLM judgment against 8 immutable criteria

## Findings (appended as established)

## A. Harness compliance (in progress)
- research_brief_86.85.md mtime 2026-08-14T21:34:19 (local) < contract_86.85.md 2026-08-15T15:37:58
  < verdict_ledger_write.py 15:39:31 < mutation_matrix 15:41:11 < ledger 15:41:52
  < experiment_results 15:43:27 < live_check 15:44:13. ORDER OK.
- masterplan 86.85 status = pending (not yet flipped). LOG-LAST pending check TBD.
- Work is COMMITTED as d1c4a79d (6 files, +1213).

## B. Deterministic
- IMMUTABLE COMMAND: `parses`, exit=0. GREEN.
- RUFF LINT GATE (qa.md 1a) over `git diff --name-only d1c4a79d~1 d1c4a79d -- '*.py'`
  = {scripts/qa/mutation_matrix_86_85.py, scripts/qa/verdict_ledger_write.py}:
    F401 [*] `shutil` imported but unused
      --> scripts/qa/mutation_matrix_86_85.py:22:8
    Found 1 error.  exit=1
  *** FINDING L1: lint gate RED on a file introduced by this step. qa.md 1a: "Non-zero
      exit = FAIL". Dead import in the mutation-matrix harness (not production code). ***
- Ledger enumeration reproduced independently (population = every non-blank line of
  handoff/verdict_ledger.jsonl):
    total rows 43; recorded_by {main:43}; verdict {CONDITIONAL 23, PASS 8, NO_VERDICT 7,
    FAIL 5}; distinct step_ids 11; 86.74 rows 8; rows with recorded_at 29; max date 2026-08-15.
  MATCHES experiment_results C2 EXACTLY except max date (theirs quoted 2026-08-11 as the
  BEFORE state, which is correct as a before-state).

## B (cont). Independent reproduction
- writer --self-test: 11/11 ok, exit 0.
- AUTHOR MUTATION MATRIX re-run BY ME: control GREEN first, M1..M5 all KILLED,
  sha256 of target unchanged, ledger md5 1d2150f36d187c8f0a69bdfd3a44b62d before AND after.
  REPRODUCES exactly.
- REAL enforceEscalation extracted BYTE-EXACTLY from .claude/workflows/qa-verdict.js
  (2225 bytes, 52 lines, asserted `fn in src`) and driven:
    CONTROL 1 prior C + current C   n=1 auto_fail=false
    DRIVEN  2 prior C + current C   n=2 auto_fail=true
    CONTROL 2 prior C + current PASS n=2 auto_fail=false
    CONTROL 2 prior C + current FAIL n=2 auto_fail=false
    [C,C,NV] + C                    n=2 auto_fail=true   (drop does NOT reset)
    absent sequence                 n=null auto_fail=null status=not_supplied
    86.74 real priors [NV,NV,C,C,P,C,C] + C  n=2 auto_fail=true
  EVERY author claim REPRODUCES on the real shipped function.
- LIVE reader on the REAL ledger: `verdict_history_86_21.py --step 86.74 --evidence-only`
  -> status=ok, 8 verdicts, NV->NV->C->C->PASS->C->C->C.  (was no_rows_for_step before)
- CROSS-PROCESS (mine, non-palindromic): proc1 writes C, proc2 writes PASS,
  proc3 (separate invocation) --emit-sequence -> ["CONDITIONAL","PASS"]. ORDER PRESERVED.
- CRITERION 7 EXHAUSTIVE: 128 flag combinations (4 verdicts x 32 flag sets);
  96 writes round-trip the exact verdict through a SEPARATE read process; 32 unkeyed
  combos correctly refused exit=3. 0 verdict-altering findings.
- RECONSTRUCTION AUDIT (all reproduce):
  * 10 WIP files for 86.74 -> `ls | grep -c 86.74` = 10. TRUE
  * harness_log phase=86.74 rows: Cycle 190 NO-VERDICT(x2), 191 C, 192 C, 193 PASS,
    195 C, 196 FAIL. NO Cycle 194 for 86.74 -> "cycle 5 never logged" TRUE.
  * "cycle numbers not unique": Cycle 193/194/195 exist for phase=36.17 (2026-08-09)
    AND 193/195 for phase=86.74 (2026-08-14). TRUE.
  * "3 narrative cycle sections": evaluator_critique_86.74.md has `## Cycle 5/6/7`
    at lines 321/459/573. TRUE.
  Every backfilled ledger row maps 1:1 to a named source. FAITHFUL.

## *** INDEPENDENT MUTATION CELLS (mine, not the author's) ***
- QA-M1  reverse emit_sequence output order (return out[::-1])      -> **SURVIVED**
    self-test exit 0, ALL 11 checks green INCLUDING the one literally named
    "sequence is oldest->newest". Root cause: the fixture appends three rows all
    CONDITIONAL (verdict_ledger_write.py:315-318), so the asserted list
    ["CONDITIONAL","CONDITIONAL","CONDITIONAL"] is PALINDROMIC -> reversal is
    unobservable. Vacuity shapes #4 (true by construction) + #5 (fixture cannot
    represent the failure).
    BEHAVIOURAL DIFFERENTIAL (driven on the real enforceEscalation):
      oldest->newest [PASS,C,C] + current C -> n=2 would_auto_fail=TRUE
      newest->oldest [C,C,PASS] + current C -> n=0 would_auto_fail=FALSE
    i.e. an order regression in the ONE function that feeds args.verdict_sequence
    SILENTLY SUPPRESSES the auto-FAIL, and no guard can see it.
- QA-M3  neuter check() failure recording                            -> SURVIVED (by
    construction; establishes the self-test is the matrix's sole oracle)
- QA-M4  existing_keys() returns empty (dedup bypass at the other seam) -> KILLED
- QA-M5  emit_sequence drops NO_VERDICT rows                          -> KILLED
- QA-M6  recorded_at collapsed to date only                           -> KILLED
- LAUNDERING PROBE: temp ledger with 4 rows for step 77.7 = [CONDITIONAL, "COND",
  CONDITIONAL, ""] -> `--emit-sequence` prints ["CONDITIONAL","CONDITIONAL"],
  exit 0, EMPTY stderr. The 2 unrecognisable rows vanish silently.
  Contrast, driven on the real consumer: passing those tokens raw yields
  status=unparseable / n=null (FAILS CLOSED). So emit_sequence's silent filter
  BYPASSES the consumer's fail-closed branch and manufactures a confident number.
  Internally inconsistent with read_rows(), which is deliberately LOUD (exit 4) about
  a corrupt LINE for exactly the "would under-count" reason.

## CLAIM AUDIT -- before-state reproduced from git (d1c4a79d~1)
  10814 bytes / 35 rows / recorded_by {main:35} / verdict {C 18, PASS 7, FAIL 5, NV 5}
  / step_ids 10 / 86.74 rows 0 / max date 2026-08-11.  ALL REPRODUCE EXACTLY.
- settings.local.json sha256 NOW = 8f03f1949599866fe3875266557ff23818d1d1dc5e1cf7a4eef337e68124d966
  == the before AND after value claimed in live_check section 5. Restore verified
  independently. 0 probe hooks remain. File is gitignored (git check-ignore confirms).
- *** FINDING N1 (count does not reproduce): verdict_ledger_write.py:132 and
  contract_86.85.md section 1 F4 both state run_id "is present on 33 of the 35
  pre-existing rows". MEASURED on d1c4a79d~1: non-empty run_id = 35/35;
  wf_-prefixed = 35/35. No predicate yields 33. No population rule stated, no
  enumeration command quoted -> direct criterion-2 hit. ***

## A (final). Harness compliance 5/5 CLEAN
1. research gate: brief_status COMPLETE, gate_passed true, 8 read-in-full, 23 URLs,
   recency scan true; brief 08-14 21:34 < contract 08-15 15:37.
2. contract before generate: verified by mtime chain; 8/8 immutable criteria present
   VERBATIM in the contract (whitespace-normalised comparison against masterplan.json).
3. experiment_results_86.85.md present.
4. LOG-LAST: zero `phase=86.85` rows in harness_log (control: phase=86.74 -> 7 rows);
   masterplan 86.85 status still `pending`.
5. NO VERDICT-SHOPPING: qa_wip source_present=true, attempt_number=1, prior_attempts=0,
   status=ok; ledger --step 86.85 -> no_rows_for_step. FIRST attempt. Auto count (0
   priors) does not exceed ledger count (0) -> no staleness signal.
   sequence for 86.85: NONE (no_rows_for_step, correctly NOT read as 0 verdicts).

## SCOPE / UNINTENDED CHANGES
- d1c4a79d touched 6 files; NONE production. Lint scope authority for a COMMITTED step
  = the commit range (git diff --name-only HEAD -- '*.py' returns only the PRE-EXISTING
  backend/api/sovereign_api.py; untracked .py = 0).
- 6 uncommitted production files (backend/api/sovereign_api.py + 5 frontend) have
  mtimes 2026-08-14 13:12-13:35, PREDATING this step's 15:37+ window, and
  sovereign_api.py lints CLEAN -> the F401 is unambiguously this step's new file.
- NOTE for Main: auto-commit-and-push does `git add -A` on the status flip and WILL
  sweep those 6 pre-existing production files into the 86.85 commit.

## CRITERION MAP
C1 MET | C2 NOT MET (N1) | C3 MET | C4 MET | C5 MET | C6 MET | C7 MET (exhaustive)
C8 NOT MET (QA-M1 vacuous "sequence is oldest->newest" guard; control-green and
   byte-identical restore both reproduce, but not EVERY new guard is mutation-resistant)
L1 lint gate RED (qa.md 1a: non-zero exit = FAIL)
W1 emit_sequence launders out-of-vocabulary tokens, bypassing the consumer's
   fail-closed branch

## VERDICT DIRECTION: FAIL
Grounds: (1) criterion 8 -- a NEW guard proven vacuous by execution with a material
behavioural differential on the real consumer (armed -> suppressed); (2) criterion 2 --
a ledger-row count with no population rule, no command, and which does not reproduce;
(3) qa.md 1a deterministic lint gate RED on a file this step introduced.
NOT a quality judgement on the rest: every other quantified claim I tested reproduced
exactly, the localisation is genuinely rigorous, and the honest-limits sections are
accurate rather than defensive.

COMPLETED: 2026-08-15T13:54:45Z
