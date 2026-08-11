STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.44
WRITTEN: 2026-08-11T15:34:18Z

# Q/A cycle 2 (attempt 2 of 5) -- step 86.44

Remediation commit 431401dc on top of fe9a6dad. Cycle-1 verdict was CONDITIONAL
with six findings.

## HARNESS COMPLIANCE -- 5/5 CLEAN

- research_brief_86.44.md envelope: brief_status COMPLETE, 8 full reads (floor 5),
  16 URLs (floor 10), recency_scan true, gate_passed true.
- contract-before-generate: brief 16:56:19 < contract 16:59:16 < fe9a6dad 17:13:05.
- experiment_results_86.44.md present (17:33:18, i.e. after 431401dc 17:33:47? -- mtime
  is the working-copy write; content matches the commit).
- LOG-LAST: `grep -cE '^## Cycle .* phase=86\.44' handoff/harness_log.md` = 0; naive
  `grep -c 'phase=86\.44'` = 0; masterplan status=pending, retry_count=0. Not logged,
  not flipped. Correct.
- No verdict-shopping: evidence CHANGED (431401dc, 5 files, +170/-27).
- masterplan diff ea5b1cd5^..HEAD touches ONLY "id": "86.55" (the filed defect);
  86.44's criteria block untouched.

## DETERMINISTIC

- IMMUTABLE COMMAND `bash -c 'test -f handoff/harness_log.md && grep -c "^## Cycle" ...'`
  -> **1224**, exit=0.
- LINT GATE (derived scope, xargs -0, non-empty guard passed, 4 files:
  backend/api/backtest.py, scripts/harness/run_harness.py, scripts/qa/mutation_matrix_86_44.py,
  scripts/qa/prove_cycle_number_toctou_86_44.py) -> "All checks passed!", exit=0.
  Cycle-1 finding (a) F401 `subprocess` is FIXED.
- AST OK on all 4 .py.
- backend.api.backtest import OK; run_harness exec_module OK.
- scoped pytest `-k "harness or backtest or cycle"`: **162 passed, 1 skipped**, 3289 deselected.
- git status: no unintended production change; tree clean after my mutation runs.

## CENSUS RE-DERIVED INDEPENDENTLY at tree 915d2cb0 -- EVERY FIGURE REPRODUCES

total 1224 | numeric 1064 | non-numeric 160 | placeholder N/N+k 58 | EMPTY 54 |
paren 36 | step-id 10 | other 2 | token '1' = 481 (39.3% of 1224) | distinct
duplicated integers 141 | headers in dup groups 969 | 481/969 = 49.6%.
Placeholder k: 58 distinct, **k=23 absent** -> the runbook's 58-not-59 correction
(finding b) is CORRECT and re-derived.

## MUTATION MATRIX -- I RAN IT MYSELF, TWICE

CONTROL GREEN first on all three checks (d1 72/72 vs a 1064-cycle seed; d2 1224/1224;
d3 0 literals across 2 pinned sources). M1/M2/M3/M4 ALL KILLED, restore byte-identical
on every cell, POST-RESTORE control all True, exit 0. M3 names per-step-protocol.md,
M4 names CLAUDE.md -> the D3 guard is genuinely NON-VACUOUS ON BOTH PINNED MEMBERS.

## FINDINGS (all fixable; verdict CONDITIONAL, the 2nd -- not the 3rd)

1. INCOMPLETE REMEDIATION. Cycle-1 finding (e) named "section 5 AND section 6 reason 3".
   Only sec 5 got a correction block. `git diff fe9a6dad 431401dc -- experiment_results`
   does NOT touch "evidence of D4". Sec 6 reason 3 still reads "141 duplicated integers
   are evidence of D4" -- the retracted wording, in the section carrying the criterion-4
   DECISION. Correction sits BESIDE the uncorrected claim.

2. NEW, MEASURED OVER-ATTRIBUTION in sec 1: "The 481 have one mechanical cause:
   run_harness.py:953 passes the loop index". FALSE for >=63 of 481 (13.1%). 62 of
   those 63 carry `phase=` in the HEADER LINE; the run_harness entry template
   provably never writes phase= there (extracted the f-string: header is
   `## Cycle {cycle} -- {ts}` only; "phase=" in template -> False). Those are manual
   protocol-format entries restarting per-step numbering at 1. Correct split:
   >=418/481 (86.9%) run_harness-shaped, >=62 manual. Sec 5's correction endorses
   sec 1 ("which my own sec 1 already attributed correctly") -- so it inherits the error.

3. D3 CLASS ASSERTED AS "TWO-MEMBER", NOT DERIVED. At HEAD, three MORE live tracked
   occurrences of `## Cycle N -- YYYY-MM-DD` outside the pinned population:
   .claude/hooks/lib/harness_log_gate.py:22 (live hook docstring, same bare-N-beside-
   <step_id> shape); docs/audits/phase-24-2026-05-12/24.0-charter-findings.md:92;
   tests/_phase_24_helpers.py:197,207 (comment literally "format from CLAUDE.md" --
   direct evidence of propagation). Guard cannot fail on 3 of its 5 live members.
   BOUNDED SEVERITY: none is a "copy this block when you append" instruction, and the
   guard's own message honestly says "across 2 pinned sources".

4. sec 9 "Files changed" STALE: still says mutation_matrix "NEW -- 3 cells" (sec 8 and
   the script say 4), and OMITS CLAUDE.md -- the file whose omission was cycle-1's
   sharpest finding -- and pending_restart_2026-08-11.md.

5. NOTE: run_harness.py:953 is the `def` line, not the loop (:1123) or call sites
   (:1149/:1196). Unremediated from cycle 1. I first measured :1117/:1143/:1190 --
   contaminated by my own M1 mutant window; corrected on a clean tree.

## ANSWERS TO THE THREE GRADE-HARDEST QUESTIONS

(i) The guard is non-vacuous on both pinned members -- EXECUTED, not argued. But the
    boundary MOVED rather than closed: 5 live members, 2 pinned. Finding 3.
(ii) SUFFICIENT and independently verified live: endpoint still 1064, pid 66306 still
    the :8000 listener (started 2026-08-10 21:33:01, ~20h before fe9a6dad).
    pending_restart file is thorough (restart verb, post-restart verification, the
    orphaned-server/EADDRINUSE trap). Nothing else asserted as live that is not; sec 8's
    "1224 of 1224" is the in-memory matrix control, correctly labelled, and I reproduced
    it by fresh import. Residual: sec 10 still omits the caveat, but sec 4 carries it.
(iii) YES. Every COUNT in secs 1 and 5 reproduces exactly. What does not reproduce is
    the ATTRIBUTION -- finding 2 -- plus findings 1 and 4.

## HAZARD WORTH NAMING

The matrix transiently writes CLAUDE.md, the runbook, run_harness.py and backtest.py.
Restores are byte-identical and I verified the tree clean, but a peer session's
`git add -A` inside the ~2-minute mutant window would commit a mutated CLAUDE.md.

COMPLETED: 2026-08-11T15:52:41Z
