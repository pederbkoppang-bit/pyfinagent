STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.47
WRITTEN: 2026-08-18T02:05:42Z

# Q/A write-first record -- step 86.47 (drought census)

Spawned via Workflow rail. qa.md read in full at 02:05:42Z.

## Plan
A. Harness-compliance audit (5 items)
B. Deterministic: immutable command, git status/diff scope, lint, runtime smoke
C. LLM judgment vs 6 immutable criteria + mutation matrix on the 42 invariants

## Findings (appended as established)

### Attempt evidence (gathered, NOT a trigger)
- qa_wip.py 86.47 --spawned-at 2026-08-18T02:05:42Z: source_present=true,
  attempt_number=4 (status ok, is_lower_bound=true), prior_attempts=3,
  records_retained=4 (GAUGE), records_pruned_known=null.
- verdict_history_86_21.py --step 86.47 --evidence-only: status=ok,
  "3 verdict(s) from the ledger", verdicts = FAIL -> CONDITIONAL -> CONDITIONAL.
- CROSS-CHECK: prior_attempts (3) == ledger rows (3) -> ledger NOT stale.
- Main's advisory disclosure said "attempt 5 of 5"; qa_wip says attempt_number=4
  (lower bound). Discrepancy noted; Main is the constrained party, advisory only.

### B. Deterministic
- IMMUTABLE COMMAND: bash -c 'source .venv/bin/activate && python -c
  "import ast;ast.parse(open(\"backend/services/autonomous_loop.py\").read());
  print(\"parsed\")"'  -> stdout "parsed", EXIT=0. REPRODUCED.
- HEAD = 3739f034.
- scripts/qa/drought_census_86_47.py exists, UNTRACKED (32,147 bytes, mtime
  18 aug 04:04).
- WORKING-TREE PRODUCTION DIFF PRESENT (NOT authored by 86.47 per Main):
  backend/services/autonomous_loop.py (+18/-?) carries an uncommitted
  "phase-86 UI bugfix" changing the persisted `summary` field to prefer
  full_report.final_synthesis.final_summary. Also backend/api/sovereign_api.py
  and 9 frontend files. NEEDS SCOPE ADJUDICATION.

### BQ RE-DERIVATION (independent, via google-cloud-bigquery in .venv) -- 100% REPRODUCED
- TRADE_DAYS: 26 days, list matches byte-for-byte, last = 2026-08-13. OK
- weekday split: 2026-04-26 Sun / 2026-05-16 Sat / 2026-05-17 Sun; 79 weekdays
  in [2026-04-26..2026-08-13]; 23/79 = 0.2911. OK
- RJD: paper_trades BUY 19/34, SELL 0/32; analysis_results 18/580 (3.10%). OK
- judge coverage (JSON_QUERY): 382/526, 256/275, 13/13. OK
- JSON_VALUE CONTROL: 0/526 -- independently confirms the cycle-1 mechanism. OK
- SILENCE WINDOW: all 13 rows reproduce incl. decisions + pcts; all path=full. OK
- BUY CROSSING: 8 rows; 7 lite NO_RISK_ASSESSMENT; 1 full 2026-08-13 DELL
  REJECT pct=0. OK
- PATH COVERAGE: 288/580 all-time; 288/288 from 2026-06-11. OK
- FUNNEL: all 8 cells reproduce exactly. OK
- SYNTH_ERROR: 'Failed to parse final report.', full, 219, 2026-06-11..2026-08-13. OK
- DAILY: all 11 rows reproduce exactly; NO 2026-08-18 row yet, so AS_OF is honest. OK
- POSITIONS: DELL/NTAP both Technology. OK
- REJECT_THEN_TRADE: 4 rows; 3 carry literal 'REJECT', 4th carries ''. OK
- risk_intervention_log (pyfinagent_data): 0 rows. OK
- STATED REASON: 8/8 window REJECTs mention sector/concentration; texts quote
  "100.0% Technology across 2 positions against a 60.0% threshold". OK
- 53-minute claim: 2026-08-13 DELL analysis 18:38:03Z, trade 19:31:19Z = 53m16s. OK

### MUTATION MATRIX (17 cells, scratchpad copy, md5-identical control)
control: exit 0 both modes, "OK: all 42 invariants hold".
KILLED (4): M4 weekend-day swap, M5 window-gets-a-BUY (control),
            M11 synth_error_size (control), M12 position sector split (control).
SURVIVED (13): M1 funnel B_post ok full BUYs 1->8; M2 crossing lite->full;
  M3 reject-row loses REJECT; M6 delete a _check; M7 path coverage 288->579;
  M8 healthy-null BUYs 100->80; M9 RJD 18->57; M10 AS_OF; M13 judge cov 382->300;
  M14 approved pct 2->44; M15 crossing date -> 2029; M16 window ticker;
  M17 money-path qty.
CONCLUSION-INVERTING survivors (differential confirmed by diffing stdout):
 * M1: p_post 0.0291->0.0545, need_post 102->54, and "post-break, synthesis ok"
   P(0 in 13) 0.1762 -> 0.0311 (crosses 0.05) while line 466 still prints the
   UNCONDITIONAL prose "Under every post-break null it is not [surprising]".
   Also breaks the untested tie FUNNEL B_post BUY total (8) == len(BUY_CROSSING).
 * M7: "PATH COVERAGE 579/580 = 99.8%" printed while the next two lines still
   assert "the pre-break baseline cell is path-UNKNOWN". Guard is a bare
   `pc < pt` -- the SAME bounds-only class cycle 4 flagged on jp>0.
 * M2: "the other 7 are LITE with NO risk_assessment at all" false (6);
   guard quantifies only over rows ALREADY labelled lite.
 * M3: "Three carry 'REJECT' in the TRADE ROW ITSELF" false (2); only the
   ROW COUNT is pinned.
 * M15: the 2026-08-10 CRWD+HPE pair criterion 4 names by DATE is unpinned.

### THE BLOCKING FINDING -- criterion 2's "with what stated reason" is MISATTRIBUTED
scripts/qa/drought_census_86_47.py:544-545 prints, for criterion 2's CORRECTED
(non-empty) population:
    "REFUSED: 1 at 0%, on the same sector-concentration ground as section 4."
_reached contains exactly one row: 2026-08-13 DELL. Its recorded judge reasoning
(BQ, re-read by me in full, 3,260 chars) opens:
    "DECISION DRIVER -- LIVE GATE VETO (verified, not narrative). I ran the
     composite veto chain directly (mcp pyfinagent-risk evaluate_candidate) ...
     Result: vetoed=true, reason=projected_dd_over_cap, projected_dd 22.5% vs a
     10% cap [INTERNAL risk-gate]."
and later labels concentration explicitly as
    "CORROBORATING DOWNSIDE (independent of the gate): (1) Concentration --".
So the judge's own text names a DIFFERENT gate as the driver and says the
concentration point is NOT the gate. The census asserts sameness.
- `projected_dd` / `projected_dd_over_cap` / that 10% cap appear NOWHERE in the
  census or any 86.47 artifact (grep over handoff/current/*86.47* + the script:
  only an unrelated research-brief blog row mentions "drawdown").
- NO printed query selects `reasoning`: `--sql | grep -c reasoning` = 0. So the
  "stated reason" half of criterion 2 has no reproducing predicate at all, and
  criterion 2 says a count without its predicate is a rejected outcome.
- Section 4's own sector claim IS correct: 8/8 window REJECTs mention
  sector/concentration and quote "100.0% Technology ... against a 60.0%
  threshold". Only 6b's cross-attribution to that ground is wrong.

### SUPPORTING CLAIM-AUDIT FINDINGS
- live_check_86.47.md:278 cites shipped code
  `if _p0 < 0.05 and n_an >= need_healthy: ... else: ...`.
  `grep -n "_p0" scripts/qa/drought_census_86_47.py` returns NOTHING (exit 1).
  The deliverable has exactly ONE conclusion-conditional (`if n_an >=
  need_healthy:` at :567) plus the inline sparsity ternary at :474.
- live_check:277 "every conclusion that depends on a computed value is
  conditional on it" -- REFUTED BY EXECUTION (M1): census:466 prints the
  unconditional "Under every post-break null it is not [surprising]" while the
  mutant's own table shows P=0.0311 for a post-break null. Exit 0.
- The census docstring :28-30 "Every constant is now guarded" -- refuted by a
  known-member recall test: 13 of 17 constant mutations survive at exit 0.
- live_check:294-305 presents an 8-cell matrix whose CONTROL line reads "all 42
  invariants hold", but 42 is the CYCLE-4 count; the cells are cycle-3's
  ("Re-probed after the fix" under "*Cycle 3* fixes the class"). The four
  guard families added in cycle 4 (judge majority, lite/full rates, the
  contrast, the 5 BUY-crossing guards) have NO mutation evidence in any artifact.

### HARNESS COMPLIANCE (5 items)
1. research-gate-before-contract: CLEAN. research_brief_86.47.md
   brief_status COMPLETE, gate_passed true, sources_read_in_full 14 (floor 5),
   urls_collected 48 (floor 10), recency_scan_performed true, audit_class
   coverage dry=true (K=2). mtime 01:04:55Z < contract 01:47:34Z.
2. contract-before-generate: CLEAN. contract 01:47:34Z < census 02:04:24Z <
   live_check 02:04:44Z < experiment_results 02:05:08Z (mtimes converted from
   local CEST).
3. experiment_results present: YES.
4. log-last: CLEAN. masterplan 86.47 status=pending, retry_count 0/3; no
   "phase=86.47 result=" header in harness_log.md (only a forward-pointer
   mention at :33915).
5. no-verdict-shopping: CLEAN. Prior spawn WRITTEN 01:50:49Z, its verdict
   recorded 02:02:51Z; census/live_check/experiment_results all rewritten
   02:04-02:05Z. Evidence CHANGED.
WARN: evaluator_critique_86.47.md (mtime 01:24Z) contains only Cycle 1 and one
ledger row, while handoff/verdict_ledger.jsonl carries 3 rows for 86.47
(wf_acfe2459-948 FAIL 01:24:53Z, wf_775cfbb1-5ee CONDITIONAL 01:47:17Z,
wf_89107a13-3d6 CONDITIONAL 02:02:51Z). Two returned verdicts are not
transcribed into the five-file artifact.

### OTHER DETERMINISTIC RESULTS
- ruff F821,F401,F811 over DERIVED scope (git diff HEAD + git ls-files --others,
  3 files, non-empty guard asserted, xargs -0): "All checks passed!" exit 0.
- import backend.services.autonomous_loop OK; import backend.api.sovereign_api OK.
- curl :8000/api/health -> http=200 {"status":"ok","version":"6.93.236"}.
- --sql prints 11 query blocks, exit 0.
- No UI claims in this step -> gate 1c not applicable; no browser capture taken.
- Criterion 5 scope: the dirty production files predate this step
  (autonomous_loop.py 2026-08-17 21:42 local, sovereign_api.py 15:54,
  frontend 22:19; this step's window opens 03:04 local). live_check sec.10
  discloses this; experiment_results:12 still says the unqualified
  "No production file was modified." NOTE only.

### CRITERION VERDICTS
C1 MET  C2 NOT MET  C3 MET  C4 MET  C5 MET  C6 MET
VERDICT: FAIL (criterion 2's stated-reason element is factually wrong on the
population the criterion names, and has no reproducing query).

### STABILITY RECHECK AT CLOSE
HEAD unchanged 3739f034; census md5 2c26032e0f5bbbb8d09461511980ce78 unchanged;
experiment_results/live_check byte sizes unchanged. Tree stable across the eval.

COMPLETED: 2026-08-18T02:17:59Z
