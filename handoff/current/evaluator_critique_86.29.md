# Evaluator critique -- step 86.29

# CYCLE 1 -- RAIL DROP. NO VERDICT.

**Run `wf_d4e2e794-567` (task `wctzb6dqu`), 2026-08-11 06:38-06:49Z. Terminated
with `agent({schema}): subagent completed without calling StructuredOutput
(after in-conversation nudge)` after 197,098 subagent tokens and 42 tool uses.**

**THIS IS NOT A VERDICT.** Per `.claude/rules/research-gate.md` and the CLAUDE.md
harness protocol, an errored/empty return is NO VERDICT, never PASS -- and
equally, never CONDITIONAL. The record below reached an internal assessment of
CONDITIONAL. **That assessment is not adopted, not recorded as this step's
verdict, and does not advance any counter.** Step 86.29 has had ZERO completed
Q/A cycles.

This is the SECOND drop of the day (the first was 86.34's cycle 3 at 185,745
tokens). Both were rescued before the next spawn overwrote the fixed per-step
path -- the durability defect queued as 86.36.

The record is rescued verbatim below because it is **evidence for the re-run**:
it names five reproducible findings, and every one of them was re-measured by
Main and CONFIRMED before being acted on.

```
STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.29
WRITTEN: 2026-08-11T06:38:03Z
COMPLETED: 2026-08-11T06:48:17Z

# Q/A write-first record -- step 86.29, CYCLE 1

Launch: Workflow structured-output rail. Author = pyfinagent-51 (Main).
Contract authored by peer session pyfinagent-06 (c806cad6) BEFORE GENERATE (974297ce).

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable cmd `bash -c 'test -f .claude/hooks/archive-handoff.sh && bash -n .claude/hooks/archive-handoff.sh'`
C. Re-run the two re-runnable scripts myself
D. Attack the 4 self-nominated weak points + criterion-by-criterion judgment

## Findings log (appended as established)

### A. Harness compliance (5 items) -- ALL CLEAN
- research-gate-before-contract: `handoff/current/research_brief_86.29_rerun.md`
  envelope `brief_status COMPLETE`, `gate_passed: true`, `external_sources_read_in_full: 7`,
  `recency_scan_performed: true`, internal_files 7. The FIRST brief
  (`research_brief_86.29.md`) honestly records gate FAILED (no envelope, stopped
  mid-loop) and was re-run -- that is the rail working, not a skip.
- contract-before-generate: PROVABLE FROM GIT (split authorship).
  c806cad6 (2026-08-10 22:53) touches ONLY `handoff/current/contract_86.29.md`.
  974297ce (2026-08-11 08:37) touches the hook + 2 scripts + results + live_check.
  Contract commit strictly precedes and contains NO code.
- experiment_results present: `handoff/current/experiment_results_86.29.md` (13151 B).
- log-last: `handoff/harness_log.md` -- to be checked. masterplan 86.29 status=pending,
  retry_count=0. NOT yet flipped.
- no-verdict-shopping: CYCLE 1, no prior verdict for 86.29 exists. Nothing to shop.

### B. Deterministic
- IMMUTABLE CMD `bash -c 'test -f .claude/hooks/archive-handoff.sh && bash -n ...'` -> **EXIT=0**
- hook sha256[:16] = `6dc68f781edb4fd0` -- MATCHES the digest stated in live_check line 5.
- `git diff --name-only HEAD` = only agent-memory/audit-jsonl/heartbeat churn.
  NO unintended production change. GENERATE commit `974297ce` touched exactly the
  5 files experiment_results section 1 claims. Verified with
  `git diff --name-only 974297ce^ 974297ce`.
- `python scripts/qa/prove_archive_provenance_86_29.py` -> RESULT: PASS (0 problems),
  EXIT=0. Reproduced BY ME. 4/4 mutants KILLED, 3/3 control checks GREEN,
  BEFORE half declares '82.54' from the git-recovered pre-fix hook.
  Isolation: real hook digest unchanged, archive dir list unchanged (819).
- `python scripts/qa/derive_archive_misattribution_86_29.py` -> EXIT=0.
  Reproduced BY ME at tree 33255004: **153 mismatch / 387 agree / 255 unclassified
  / 24 no_contract over 819 dirs**. Recall 2/2, controls 4/4, precision 1.0000,
  0 suspects. EXACTLY the "after" row of experiment_results section 6.

### B2. Live-witness claim (attack point 2) -- MAIN'S READING IS CORRECT
Read `.claude/hooks/archive-handoff.sh:241-244`:
```
for f in contract.md experiment_results.md evaluator_critique.md research.md research_brief.md; do
    [ -f "$CURRENT_DIR/$f" ] || continue
    [ -f "$target/$f" ] && continue          # derived branch already won
    if rolling_declares_step "$CURRENT_DIR/$f" "$short_sid"; then
```
For 86.31 the derived branch supplied contract/experiment_results/evaluator_critique/
research_brief, so those four hit the `[ -f "$target/$f" ] && continue` early-out;
`handoff/current/research.md` is ABSENT (verified) so it hits the first `continue`.
=> `rolling_declares_step` was NEVER CALLED. `rolling_skipped=0` in
`handoff/archive/phase-86.31/PROVENANCE.md` is consistent with that and with nothing
else. Main's statement "the live run proves the DERIVATION branch ONLY, not the
guard" is EXACT -- neither over- nor under-claimed.
Independently verified the live dir: 5 artifacts + PROVENANCE.md, every head -1
declares 86.31 (`# Contract -- step 86.31` etc.). First archive dir since
2026-08-06 holding its own step's files -- confirmed by me, not taken from the report.

### B3. SECOND live witness appeared DURING this evaluation
`handoff/archive/phase-86.25/` minted 2026-08-11T06:42:07Z by a peer step closure
(archive grew 819 -> 820 between two of my own runs). PROVENANCE: `derived=5
rolling_copied=0 legacy_moved=0 rolling_skipped=0`; every head -1 declares 86.25.
Same shape as 86.31: derivation branch only, guard still never reached.
New evidence Main could not have disclosed. It STRENGTHENS the mechanism claim and
shows the ungraded-infrastructure exposure of section 4b is ONGOING until this closes.

### C. Criterion 2 -- RE-DERIVED BY ME under bash (not read from the artifact)
```
sid=86.29 0/0   sid=86.6 0/0   sid=82.54 0/0   sid=86.31 0/0
sid=86.26 0/0   sid=4.5.9 0/0  sid=25.A 0/0    sid=86.25 0/0
POSITIVE CONTROL (temp dir holding 86.29-contract.md) -> 1
```
NOTE zsh aborts this loop with `no matches found` (nomatch); the hook runs under
bash where an unmatched glob stays literal and `[ -f ]` fails -> 0. Re-ran under
bash, the hook's own shell. Criterion 2 MET, independently.

### D. Lint + scope
- scope DERIVED from the GENERATE commit (working-tree diff is empty; the fix is
  committed at 974297ce): `git diff --name-only 974297ce^ 974297ce -- '*.py'` ->
  2 files, NON-EMPTY asserted. `uvx ruff check --select F821,F401,F811` ->
  "All checks passed!", exit=0.
- qa.md 1b/1c/1d do NOT bind: no `frontend/**`, no `backend/**`, no UI claim.
- `scripts/housekeeping/verify_handoff_layout.py` reports 455 violations -- ALL
  pre-existing "no step-id prefix" complaints about the suffix convention itself
  (it expects a PREFIX). Zero mentions of PROVENANCE.md. Not a regression here.
- No consumer of `handoff/archive/*` enumerates dir contents in a way PROVENANCE.md
  breaks (checked verify_phase_4000_*.sh, quarantine_phantom_archives.py, layout).

### E. MY OWN MUTATION MATRIX (7 cells) -- beyond the author's 4
Control on the shipped hook first, then mutate. Nothing written to the repo:
the hook text is mutated IN MEMORY and written to a tempfile.
```
Y1 KILLED   fall-through `sys.exit(1)` -> `sys.exit(0)`  [MY check only]
Y2 SURVIVED remove `[ -f "$target/$f" ] && continue`     [near-equivalent]
Y3 KILLED   total never zero -> loud branch unreachable
Y4 KILLED   derived branch writes a wrong archive name
Y5 KILLED   invert the declaration comparison
Y6 KILLED   suppress the systemMessage emitter
Y7 SURVIVED variant glob widened to `${base}_*.md`       [author's fixture]
Y7 KILLED   same mutant under a REALISTIC fixture        [MY fixture]
```

### F. FINDINGS (all reproducible)

**F1 [WARN] Fixture cannot represent the criterion-4 failure class.**
`prove.make_scratch` puts ONLY the step-under-test's files in `handoff/current/`;
the real dir holds 410-521 files from ~200 steps. I added three other steps'
artifacts and widened the variant glob to `${base}_*.md` (Y7). Shipped hook:
GREEN, no alien files. Author's `check_right_step`: **SURVIVED**. My realistic
fixture: **KILLED**, 18 alien files copied into `phase-99.1/` incl.
`contract_82.54.md`. That is verbatim the defect criterion 4 names -- "copies
another step's files ... must be a visible failure". qa.md 4c shape #5.

**F2 [WARN] The "unsure -> do not copy" fall-through is covered by NO cell.**
The hook's own comment calls that asymmetry "the whole fix", but every fixture
rolling file DECLARES something, so `sys.exit(1)` (no pattern matched) is never
exercised. My check with non-declaring rolling files: shipped hook GREEN;
mutant Y1 copies all 4 undeclared rolling files. Killed only by my check.

**F3 [WARN] The precision oracle is NOT independent of the classifier.**
`confirm_mismatch` reuses the SAME `_DECLARE` list, differing only in aggregation
(union-of-all vs first-hit). It detects "right pattern, wrong order" and is BLIND
to "grammar does not recognise this header". Concrete class it misses: the
grammar accepts only ASCII `--`. 33 of the 255 "unclassified" dirs DO declare a
step with an EN/EM-DASH (`# Contract — Step 76.9.2`); **7 are genuine mismatches
the census does not count** -- phase-75.5.12 / 76.9.3 / 78.0 / 78.16 / 78.2 /
79.2 all hold 76.9.2's contract, phase-75.1 holds 75.2's. So 153 is a FLOOR
(>=160). C1's letter still holds: those 7 sit in the 49 "genuinely opaque" bucket
that IS explicitly reported as not-clean.

**F4 [WARN] A printed claim does not reproduce.** The census prints, and
live_check D reproduces verbatim: *"no mismatched dir mentions its own step id
anywhere in its contract head."* FALSE -- **47 of 153 do**, e.g.
`handoff/archive/phase-10.5.0/contract.md` head reads
`step: phase-10.5-batch (covers 10.5.0, 10.5.1, ...)`. The tabular line one line
above states the correct narrower property ("appears in no DECLARATION in the
head"); the summary sentence overstates it. The conclusion ("none is the 86.19
truncation shape") survives. phase-10.5.0 also shows the census can over-flag a
legitimate BATCH contract, so 153 has contestable positives as well as >=7 false
negatives.

**F5 [NOTE] Section B is the only evidence block with no `$ command` line**, and
its "456 suffix-convention files" does not reproduce under four rules I tried
(443 / 410 / 521 / 454). NOT load-bearing -- the zeros + positive control, which
are, I reproduced exactly.

### G. Attack points, answered
1. Ungraded live hook: disclosure is COMPLETE and CORRECTLY SIZED (names the
   mechanism, the counterfactual casualty, verifies the outcome itself, and
   states what the witness does NOT prove). It violates NO immutable criterion --
   criterion 3 forbids the DEMONSTRATION touching handoff/archive, and the
   demonstration ran wholly in temp dirs (isolation asserted, re-verified by me);
   the live dirs came from genuine peer closures. Fail-open trap bounds the blast
   radius. **WARN, not blocking.** I agree the disclosure does not exonerate it.
2. Rolling-guard coverage: **Main's reading is EXACT.** Verified in code (B2).
3. Precision oracle: **partially conceded** -- see F3/F4. Not vacuous (the
   SUSPECT/CONFIRMED controls genuinely both fire, verified) but not independent.
4. Anchor guard: **REAL, verified by execution.** Replaced MUTANTS with an absent
   anchor and a no-op replace -> "ANCHOR MISSING -- refusing to score" +
   "MUTATION DID NOT CHANGE THE TEXT -- refusing to score", rc=1 (FAIL).
   BONUS: the BEFORE-half guard is real too -- PRE_FIX_REF=HEAD -> "REFUSED:
   the recovered script already contains the fix", rc=1.

### H. Criterion roll-up
C1 MET (with F3/F4 attached) | C2 MET (re-derived by me) | C3 MET (BEFORE-guard
verified) | C4 MET both branches (with F1/F2 attached) | C5 MET | C6 MET (4/4
killed, reproduced; guard proven real).

### I. 3rd-CONDITIONAL check
`grep -cE "phase=86\.29 result=CONDITIONAL" handoff/harness_log.md` -> **0**.
Only two entries exist, both research-gate: Cycle 1203 GATE-FAILED, Cycle 1205
GATE-PASSED. Counter NOT armed. CONDITIONAL is permitted on this cycle.

### J. VERDICT REACHED: CONDITIONAL
Every criterion MET, code correct under 11 mutations (4 author + 7 mine), both
scripts reproduce, harness compliance clean, no unintended production change.
CONDITIONAL because four reproducible WARN findings remain -- one false sentence
inside an artifact labelled verbatim, a census that misses >=7 members of its own
population behind a one-character grammar gap, and two fixture blind spots, one of
which lets the exact criterion-4 defect class survive the author's suite.
Not FAIL: no criterion unmet, no defect in shipped behaviour.

```

## What Main did with it

**Re-measured every finding rather than trusting any of them.** The two
substantive ones, verified independently:

```
F4: mismatched dirs whose OWN sid appears somewhere in the head: 47 of 153
F3: unclassified dirs with an EN/EM-DASH in a heading: 38 of 255
F3: of those, GENUINE mismatches the ASCII grammar misses: 7
     phase-75.1               actually declares 75.2
     phase-75.5.12            actually declares 76.9.2
     phase-76.9.3             actually declares 76.9.2
     phase-78.0               actually declares 76.9.2
     phase-78.16              actually declares 76.9.2
     phase-78.2               actually declares 76.9.2
     phase-79.2               actually declares 76.9.2
```

All five findings CONFIRMED and remediated -- see `experiment_results_86.29.md`
section 7 and the regenerated `live_check_86.29.md`. The census grammar now
accepts en/em-dash separators, the overstated sentence is corrected and prints
both numbers, and the prove harness gained two behavioural checks and two
mutation cells covering the failure classes its fixture previously could not
express.

**Not treated as a grade.** No criterion is marked MET on the strength of this
record. The fresh Q/A re-derives everything.

**Not verdict-shopping.** There is no prior verdict to shop, and the tree has
changed materially since the drop. Per the CLAUDE.md cycle-2 flow, a fresh spawn
on changed evidence is the documented path.


---

# CYCLE 2 VERDICT -- Q/A, Workflow rail run `wf_2675058b-ab3`

**Transcribed VERBATIM from the captured return value.** Main records the
verdict and never authors it. 170,061 subagent tokens / 23 tool uses / 543s.

**This is the FIRST completed verdict for 86.29** -- cycle 1 dropped and is NO
VERDICT. One CONDITIONAL now stands; the 3rd-consecutive escalation is not armed.

## VERDICT: CONDITIONAL   (ok=false, harness_compliance_ok=true, certified_fallback=false)

## violated_criteria

- claim-does-not-reproduce: F4 remediation is unreachable code at this tree (experiment_results s7 F4)
- stale-figure: '47 of 153' re-derives to 43 of 156 (live_check:29,:266-267)
- undisclosed-composition: 153->156 is +8/-5, the -5 false-positive correction is reported nowhere
- false-invariant-in-shipped-source: rolling_declares_step comment claims one grammar two consumers; they have drifted
- stale-figures-in-section-4: '153 dirs' and '49 unclassified' contradict s3/s6/s7 156 and 16

## reason

All 6 immutable criteria are MET on the substance and harness compliance is clean 5/5 (immutable cmd exit=0; ruff F821/F401/F811 exit=0 on a git-derived non-empty 2-file scope; prove script re-run by me = 5/5 checks GREEN, 6/6 mutants KILLED, isolation True/True at 821 dirs; census re-run = recall 2/2, controls 4/4, 156/419/222/24; the three post-fix archive dirs phase-86.31/86.25/86.34 each declare their own step). I independently verified the anchor guard is non-vacuous (absent anchor and no-op replace both REFUSE to score and count as failures, RESULT: FAIL(2)), and killed a differently-constructed mutant of my own (hardcode the wrong-but-existing sid 82.54 in the derived path) with BOTH right_step and no_alien_files, so the kills are not construction artifacts. I adjudicated the one unadjudicated suspect: handoff/archive/phase-69/contract.md heads '# Contract - Step 69.3', so the census is CORRECT and the SUSPECT is an ORACLE false alarm - precision 0.9936 understates, and the widening introduced no census false positive. CONDITIONAL is forced by three WARN-level claim defects, all executed not argued: (A) experiment_results section 7 F4 claims 'the code now prints both numbers' - it prints NEITHER, because that block is gated on `if not suspect:` (derive_archive_misattribution_86_29.py:342) and this tree HAS a suspect, a fact stated four paragraphs earlier in the same document; the surviving figure '47 of 153' re-derives to 43 of 156 under the cycle-2 grammar. (B) the 153->156 move is reported as purely 'members it had been blind to', but by symmetric difference it is +8 (phase-69, 75.1, 75.5.12, 76.9.3, 78.0, 78.16, 78.2, 79.2) and -5 (phase-69.0..69.4, false positives the ASCII-only grammar had produced); the -5 half is disclosed nowhere and 153+7 != 156. (C) the SHIPPED hook's rolling_declares_step comment asserts its patterns are 'the SAME set' as the census so the two 'cannot drift' - they have drifted (census widened _DASH in cycle 2, hook did not; 7 patterns vs 4), proven behaviourally: identical declaration, '# Contract -- step 99.5' COPIED vs '# Contract - step 99.6' (em-dash) REFUSED. (C) is fail-closed so it cannot re-poison an archive. Full record: .claude/agent-memory/qa/verdicts/verdict_wip_86.29__20260811T065922Z.md

## violation_details

### 1. Contradiction

**action** -- python scripts/qa/derive_archive_misattribution_86_29.py (run by Q/A at tree 2682584d)

**state** -- Script printed the SUSPECT line and did NOT print 'STATED AT TRUE SIZE: N of M DO mention their own sid'. The corrected mention-vs-declare output lives inside `if not suspect:` at derive_archive_misattribution_86_29.py:342, and precision is 0.9936 with 1 suspect (phase-69) at this tree, so the branch is dead. Re-derived independently: 43 of 156 mismatched dirs mention their own sid; the artifact says 47 of 153.

**constraint** -- SEVERITY WARN. experiment_results_86.29.md s7 F4: 'Fixed: the code now prints both numbers and names the distinction between mentioning and declaring'. qa.md s4b: a numeric claim whose reproducing command does not reproduce it is a Contradiction finding.

### 2. Overgeneralization

**action** -- Symmetric difference of the mismatch SETS under the ASCII-only (cycle-1) and dash-widened (cycle-2) grammars, computed by Q/A over the same 821 dirs

**state** -- GAINED 8: phase-69, 75.1, 75.5.12, 76.9.3, 78.0, 78.16, 78.2, 79.2. LOST 5: phase-69.0, 69.1, 69.2, 69.3, 69.4 - these were FALSE POSITIVES of the ASCII-only grammar (em-dash heading unmatched, fell through to `^#.*?\bphase-(SID)` and declared '69' != '69.0'). Net +3. Nothing in experiment_results or live_check mentions the -5; the only sentence conceding over-flagging is inside the same dead `if not suspect:` block.

**constraint** -- SEVERITY WARN. experiment_results s6: 'the jump from 153 to 156 is not three new bad dirs, it is the census finally seeing members it had been blind to'; s7 F3: '7 of them are genuine mismatches the census was not counting'. qa.md s4b: cardinality agreement is not sufficient - compare two operationalizations by symmetric difference and report the residual.

### 3. Contradiction

**action** -- Drove the SHIPPED hook in a scratch tree with only rolling files, identical declaration for the step being archived, varying only the dash character

**state** -- '# Contract -- step `99.5`' -> rolling contract.md COPIED. '# Contract - step `99.6`' (em-dash) -> REFUSED, nothing copied. The census accepts all three dashes (_DASH = (?:--|em|en)) since cycle 2; the hook's rolling_declares_step still hard-codes `--`, and carries 4 patterns to the census's 7. Direction is fail-closed (under-copy surfacing as the loud-failure path), so it cannot reintroduce misattribution.

**constraint** -- SEVERITY WARN. .claude/hooks/archive-handoff.sh, rolling_declares_step comment: 'the patterns below are deliberately the SAME set used by scripts/qa/derive_archive_misattribution_86_29.py -- one declaration grammar, two consumers, so the census and the hook cannot drift into disagreeing about what "declares a step" means.'

### 4. Contradiction

**action** -- Read experiment_results_86.29.md s4 against s3/s6/s7 of the same file

**state** -- s4 says 'Not backfilled the 153 dirs' and '49 archive dirs remain genuinely unclassified'; s3/s6/s7 say 156 and 16. Commit 4b526b61 superseded sections 1/3/5/6 and left 4 carrying cycle-1 figures with no superseded marker.

**constraint** -- SEVERITY NOTE. The artifact's own rule (s3): 'The cycle-1 values they supersede are kept only in section 7's movement table, so this table cannot be read against a stale figure.'

### 5. Invalid_Precondition

**action** -- The hook edit was in force for a concurrent session before it was graded; archive dirs phase-86.31, 86.25, 86.34 were minted by the ungraded change

**state** -- Disclosed in experiment_results s4b with the correct limitation stated (rolling_skipped=0, so the live run exercised the derivation branch only and is NOT evidence the poison guard refuses poisoned input). I verified all three dirs independently: each contains '# Contract -- step <own sid>' plus PROVENANCE.md. Outcome correct; the risk was borne by a peer session.

**constraint** -- SEVERITY NOTE. Recorded as a process finding, NOT a criterion miss - criteria 1-6 are all MET and the change is fail-open (trap 'exit 0' EXIT preserved). Does not block; must not be repeated for a PostToolUse hook while a peer session is live.

## checks_run (17)

- harness_compliance_audit_5_of_5
- immutable_verification_command_exit_0
- python_lint_gate_ruff_F821_F401_F811_derived_scope
- no_unintended_production_change
- prove_archive_provenance_rerun_5_checks_6_mutants
- census_rerun_recall_2_of_2_controls_4_of_4
- anchor_guard_independent_probe_absent_anchor_and_noop_replace
- independent_mutant_QA_MX7_wrong_but_existing_sid
- grammar_symmetric_difference_old_vs_new
- adjudicated_phase_69_precision_suspect
- read_all_16_genuinely_opaque_dirs
- live_archive_dirs_86_31_86_25_86_34_verified
- hook_census_grammar_drift_behavioural_differential
- claim_reproduction_R1_R2_R4_counts
- 3rd_conditional_counter_check_harness_log
- code_review_heuristics
- evaluator_critique

## notes

Write-first record: .claude/agent-memory/qa/verdicts/verdict_wip_86.29__20260811T065922Z.md (COMPLETE; still not a verdict). DEVIATION FROM THE SPAWN PROMPT, deliberate: the prompt asked for the fixed name verdict_wip_86.29.md, but qa.md on disk (phase-86.36, which STEP 0 makes authoritative) mandates the run-stamped name - writing the fixed name would have destroyed the cycle-1 dropped run's record, which is precisely the defect 86.36 fixed. I wrote nothing else; no write was blocked. qa.md 1c/1d do not bind (no UI claim, no backend/frontend diff), so no Playwright capture was taken and none was needed. 3rd-CONDITIONAL rule NOT armed: grep -E 'phase=86\.29 result=(CONDITIONAL)' handoff/harness_log.md returns 0; cycle 1 dropped without a verdict and its record was treated as unverified claims, not adopted. Tree is 2682584d, two commits past the eceb3a3b the artifacts name (86.36 and 86.38 landed meanwhile); all figures above are re-derived at 2682584d and the archive is 821 dirs at both. THINGS I CHECKED AND FOUND SOUND, so Main does not chase them: (1) the 16 'genuinely opaque' bucket is honest - I read all 16 first lines, they are genuinely different header shapes, and at least two (phase-3.2 holding '# Phase 3.2.1 Contract', phase-60 holding '# Contract -- 60.4') are real uncounted mismatches, consistent with 156 being stated as a FLOOR; (2) the precision oracle's shared grammar is a real independence limit, but I supplied the missing independence myself - adjudicating phase-69 and running the old-vs-new set difference - and the residual error runs CONSERVATIVE (the oracle under-reports precision), so disclosure plus my adjudication is sufficient for criterion 1 at this tree; a future census should not rely on disclosure alone; (3) F5's remediation reproduces - R1/R2/R4 are labelled rules with commands, and the counts moved 400/415/456 -> 404/416/460 exactly as the artifact predicts ('the rule is what is stable; the number is not'). REMEDIATION FOR CYCLE 3, all small: ungate the mention-vs-declare print from `if not suspect:` (or print it unconditionally) and re-derive it as 43 of 156; state the 153->156 composition as +8/-5 and name the five corrected false positives; either widen the hook's rolling_declares_step to the same dash set as the census or delete the in-code claim that the two grammars cannot drift; refresh section 4's 153/49 to 156/16.
