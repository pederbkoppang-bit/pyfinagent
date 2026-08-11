STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.29
WRITTEN: 2026-08-11T06:59:22Z

NOTE ON FILENAME: qa.md on disk (phase-86.36) mandates the run-stamped name
`verdict_wip_<sid>__<STAMP>.md`. The spawn prompt asked for the OLD fixed name
`verdict_wip_86.29.md`. I followed the ON-DISK qa.md (STEP 0 says qa.md is the
single source of truth) -- using the fixed name would have DESTROYED the
cycle-1 dropped run's record, which is the exact defect 86.36 fixed.

## Cycle 2 evaluation of step 86.29 (archive-handoff step-attribution)

### A. HARNESS COMPLIANCE -- CLEAN (5/5)
1. research-gate-before-contract: research_brief_86.29.md (2026-08-10T17:19Z) +
   _rerun.md (18:40Z); rerun envelope brief_status COMPLETE,
   external_sources_read_in_full 7, recency_scan_performed true, gate_passed true.
   contract_86.29.md mtime 22:52Z -> research BEFORE contract. harness_log shows
   GATE-FAILED (C1203) then GATE-PASSED (C1205).
2. contract-before-generate: contract committed alone at c806cad6 (141 lines,
   1 file); `git show --name-only c806cad6 | grep -c archive-handoff.sh` = 0.
   GENERATE is 974297ce. Order holds.
3. experiment_results_86.29.md present (19457 B), cycle-2 revised (4b526b61).
4. log-last: `grep -E "phase=86\.29 result="` -> only GATE-FAILED/GATE-PASSED,
   no step result line; masterplan 86.29 status=pending. Not yet flipped.
5. no-verdict-shopping: cycle 1 DROPPED (no verdict). Evidence CHANGED between
   spawns (eceb3a3b + 4b526b61). Prior CONDITIONAL count for 86.29 = 0, so the
   3rd-CONDITIONAL rule is NOT armed.

### B. DETERMINISTIC
- Immutable cmd `test -f .claude/hooks/archive-handoff.sh && bash -n ...` EXIT=0.
- Lint gate, scope DERIVED from the step's three commits
  (974297ce eceb3a3b 4b526b61), non-empty (2 files), xargs -0 (no word-split):
  `uvx ruff check --select F821,F401,F811` -> "All checks passed!" exit=0.
- No unintended production change: no backend/**, no frontend/**. git status
  shows only researcher memory + audit jsonl + my own WIP.
- prove_archive_provenance_86_29.py re-run by me: 5/5 checks GREEN,
  6/6 mutants KILLED, isolation True/True (hook sha 6dc68f781edb4fd0,
  821 dirs unchanged). RESULT: PASS (0 problems). REPRODUCES.
- derive_archive_misattribution_86_29.py re-run: recall 2/2, controls 4/4,
  156 mismatch / 419 agree / 222 unclassified / 24 no_contract over 821,
  precision 0.9936 (1 SUSPECT: phase-69). REPRODUCES.

### C. INDEPENDENT PROBES I RAN (not the author's)
- ANCHOR-GUARD PROBE (attack 4): monkeypatched MUTANTS with (a) an absent
  anchor and (b) a no-op replace. Output: "MX ANCHOR MISSING -- refusing to
  score" and "MY MUTATION DID NOT CHANGE THE TEXT -- refusing to score",
  RESULT: FAIL (2 problems). The guard is REAL and counts as failure, not KILL.
- QA-MX7, my own differently-constructed mutant: replace
  `${base}_${short_sid}.md` with `${base}_82.54.md` (a WRONG-but-EXISTING sid --
  the faithful form of criterion 4's "copies another step's files"). KILLED by
  BOTH right_step ("declares '82.54', expected '99.1'") and no_alien_files
  ("5 alien file(s) archived"). So the M1/M5 kills are not construction artifacts.
- OLD-vs-NEW GRAMMAR SYMMETRIC DIFFERENCE (attack 1): rebuilt _DECLARE with
  ASCII `--` only and diffed the mismatch SETS (not counts).
    GAINED (8): phase-69, 75.1, 75.5.12, 76.9.3, 78.0, 78.16, 78.2, 79.2
    LOST   (5): phase-69.0, 69.1, 69.2, 69.3, 69.4
  Net +3 => 153 -> 156. The 5 LOST were FALSE POSITIVES of the ASCII-only
  grammar (em-dash heading unmatched -> fell through to `^#.*?\bphase-(SID)`
  -> declared '69' != '69.0').
- ADJUDICATED phase-69 (the suspect Main left open): `handoff/archive/phase-69/
  contract.md` heads `# Contract — Step 69.3`. Dir 69 holds 69.3's contract ->
  the census's "mismatch" is CORRECT; the oracle's SUSPECT is an ORACLE false
  alarm. 0.9936 therefore UNDERSTATES precision. No census false positive found
  from the widening.
- READ ALL 16 "genuinely opaque" dirs: honest bucket, genuinely different header
  shapes. At least two are visible REAL mismatches still uncounted (phase-3.2
  holds `# Phase 3.2.1 Contract`, phase-60 holds `# Contract -- 60.4`), which is
  consistent with 156 being stated as a FLOOR.
- LIVE corroboration verified by me: handoff/archive/phase-86.31 / 86.25 / 86.34
  each contain `# Contract -- step <own sid>` + PROVENANCE.md.
- F5 remediation reproduces: R1/R2/R4 are labelled rules with commands. Numbers
  moved 400/415/456 -> 404/416/460, exactly as the artifact predicts.

### D. CRITERIA
1 MET   recall gate runs FIRST and returns 1 on failure (verified in source and
        in execution order); controls 4/4; 610 -> 222 broken into 206 harness
        per-cycle + 16 opaque WITH reasons; I re-derived and adjudicated.
2 MET   8 sids x 2 globs = 0 with a positive control returning 1, under bash.
3 MET   pre-fix hook recovered by `git show c806cad6:` and EXECUTED -> 82.54;
        after -> 99.1; scratch tree only; isolation asserted and re-verified.
4 MET   derivation + loud-failure + no_alien_files, all mutation-backed
        (M1/M4/M5 + my QA-MX7).
5 MET   "NOT backfilled" stated plainly with reasons.
6 MET   6/6 killed, anchor guard independently proven non-vacuous.

### E. FINDINGS (all WARN unless noted)
F-A (Contradiction) experiment_results §7 F4 claims "the code now prints both
    numbers and names the distinction between mentioning and declaring".
    EXECUTED: it prints NEITHER. The block is gated on `if not suspect:`
    (derive_archive_misattribution_86_29.py:342) and at the tree it names there
    IS one suspect (phase-69) -- stated four paragraphs earlier in the SAME
    document. The F4 remediation is unreachable code at this tree. The surviving
    quoted figure "47 of 153" (live_check:29,:266-267) is a cycle-1 ASCII-grammar
    value; re-derived at this tree under the cycle-2 grammar it is 43 of 156.
F-B (Contradiction) The 153->156 move is reported as purely "the census finally
    seeing members it had been blind to" (§6) / "7 of them are genuine
    mismatches" (§7 F3). Measured: +8 / -5. The -5 half -- the ASCII-only
    grammar ALSO produced false positives (69.0-69.4) -- appears nowhere. The
    only sentence conceding over-flagging sits inside the same dead block.
    153+7 != 156 is the cardinality-vs-membership trap.
F-C (Contradiction, in the SHIPPED hook) `rolling_declares_step`'s comment
    claims its patterns are "the SAME set used by
    scripts/qa/derive_archive_misattribution_86_29.py -- one declaration
    grammar, two consumers, so the census and the hook cannot drift into
    disagreeing about what 'declares a step' means." They HAVE drifted: cycle 2
    widened `_DASH` in the census only (hook keeps ASCII `--`; census 7 patterns
    vs hook 4). PROVEN BEHAVIOURALLY in a scratch tree, same declaration for the
    step being archived: `# Contract -- step \`99.5\`` -> rolling COPIED;
    `# Contract — step \`99.6\`` -> REFUSED. Direction is fail-closed (under-copy
    surfacing as the loud-failure path), so no misattribution risk.
F-D (NOTE) experiment_results §4 still carries cycle-1 figures ("the 153 dirs",
    "49 archive dirs remain genuinely unclassified") while §3/§6/§7 carry
    156/16. 4b526b61 superseded sections 1/3/5/6, not 4.
F-E (process, NOT a criterion miss) The hook edit was in force for a concurrent
    session before it was graded; 3 real archive dirs were minted by it. Honestly
    disclosed in §4b WITH the correct limitation (rolling_skipped=0 -> the
    derivation branch only; the poison was never offered). Outcome verified
    correct by me. The risk was borne by a peer session. Not exonerated, but not
    a criterion violation.

VERDICT RETURNED: CONDITIONAL (0 prior CONDITIONALs, counter not armed).

COMPLETED: 2026-08-11T07:05:54Z
