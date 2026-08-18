STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.74
WRITTEN: 2026-08-15T13:16:13Z

# Q/A write-first record -- step 86.74 (falsy-zero risk-verdict inversion)

## Prior-attempt evidence (gathered, NOT applied as a trigger)
- `qa_wip.py 86.74 --spawned-at 2026-08-15T13:16:13Z`: `attempt_number: 10`,
  `prior_attempts: 9`, `attempt_number_status: ok`,
  `attempt_number_is_lower_bound: true`, `source_present: true`,
  `records_retained: 10` (gauge, NOT the counter), `records_pruned_known: null`.
- `verdict_history_86_21.py --step 86.74 --evidence-only`: `status:
  no_rows_for_step`; `verdicts: (none)`.
- CROSS-CHECK: attempt_number (10) > ledger count (0) => **ledger is STALE**;
  `sequence: UNKNOWN` from the authoritative source. harness_log (secondary)
  carries phase=86.74 rows for Cycles 190 NO-VERDICT, 191 C, 192 C, 193 PASS,
  195 C -- **no Cycle 194 row for 86.74 exists** (the only "Cycle 194" is
  2026-08-09 phase=36.17), so Main's own commit-message sequence "191 C, 192 C,
  193 PASS, 194 C, 195 C" is not fully corroborated by the log. Main's
  disclosure is ADVISORY ONLY. Graded on merits.

## A. Harness compliance -- CLEAN (5/5)
- research_brief_86.74.md (14 aug 12:24) < contract_86.74.md (14 aug 16:19) <
  experiment_results_86.74.md (15 aug 15:14). Order OK.
- experiment_results / live_check / evaluator_critique all present.
- masterplan 86.74 status = `pending`. Not flipped. LOG-LAST OK.
- Changed evidence since prior verdicts (9c431237, a400a987, 77e4ae08; results
  file mtime 2 min before this spawn). NOT verdict-shopping.

## B. Deterministic
- IMMUTABLE CMD -> **41 passed, exit 0** (bare exit capture).
- Lint, scope DERIVED (`git diff --name-only 9d14291e^ HEAD -- '*.py'` U
  `f97dc2f5..HEAD`), 10 files, non-empty asserted, xargs:
  `uvx ruff check --select F821,F401,F811` -> All checks passed!, exit 0.
- Dirty tree = `backend/api/sovereign_api.py` (+`1y` red-line window) + 5
  frontend files, mtimes 14 aug 13:24-13:28, identical to the session-start
  snapshot. Different workstream; NOT an unintended 86.74 change.
- 9c431237 / 77e4ae08 verified docs-only. a400a987 touched settings.py:
  DESCRIPTION STRINGS ONLY, both `Field(False, ...)` defaults unchanged.
- Adjacent-suite regression check, run by ME by swapping the pre-86.74
  `portfolio_manager.py` in-memory under 4 suites: failure set at HEAD == failure
  set pre-86.74, **6 == 6, symmetric difference EMPTY both ways**. No regression.

## C. Criteria (all numbers re-derived by me)
- **C1 MET.** `_resolve_position_pct` uses `is not None` on BOTH sources; helper
  reads no flag. Driven through real `decide_trades`: nested REJECT/0% -> [] with
  flag OFF **and** ON.
- **C2 MET.** Main path: nested/flat REJECT/0% -> no order (both flags);
  APPROVE_REDUCED/3% -> $719.93 = exactly 3% of NAV 23997.71 (verdict-sized, not
  defaulted); ABSENT -> $2399.77 = the legitimate 10% default; UNPARSEABLE ->
  no order. Swap path: my own TRUE-ORPHAN mutant (floor moved BETWEEN the SELL
  and BUY appends, so the SELL orphans) is **KILLED** rc=1 by
  `test_swap_path_zero_pct_emits_no_SELL_specifically`.
- **C3 MET.** My own AST sweep for `BoolOp(Or)` with right operand
  `Constant 10.0` / `Name|Attribute DEFAULT_POSITION_PCT`: **[] (zero sites)**.
  `DEFAULT_POSITION_PCT` referenced only at :1019 (def), :1053 (ABSENT arm),
  :350 (log arg). The in-source comment states the residual reachability caveat
  honestly rather than overclaiming "true by construction".
- **C4 MET.** BigQuery, my own query: 2026-07-20..08-13 = **129 rows, 0/0/0**
  (exact reproduction, day-by-day); 2026-08-14 = **6 rows, 6/6/6 = 100%**. No
  2026-08-15 rows yet (cycle runs 18:00Z), so n=6/one cycle as disclosed.
- **C5 MET.** `risk_debate.py:357` `f"Risk debate complete: ticker={ticker}, "`.
- **C6 NOT MET (partial).** Seam PROVEN by me: real `extract_all_signals` on
  DELL's nested REJECT/0% returns agents `['Trader','RiskJudge']`; the control
  (no verdict at all) returns `['Trader']` only -- discriminating, not a
  tautology. BUT the criterion demands the contribution appear **in
  signals_log.factors_json for a gated buy**, and my BQ query shows NO such
  post-fix row: 2026-08-14 has 1 signals_log row, factors_json len **19**, no
  RiskJudge. Reference records reproduce exactly -- DELL 2026-08-13 len **517**
  RiskJudge **False**; NTAP 2026-07-31 len **1232** RiskJudge **True**.
  Structural note: the "including a 0% REJECT" half is now UNSATISFIABLE
  end-to-end *because the fix works* (a 0% REJECT can no longer produce a buy,
  so it can never produce a buy's signals_log row). The non-zero half awaits the
  first post-fix cycle that places a buy. Disclosed by Main in §6 item 5.
- **C7 MET as worded** (Main self-grades PARTIAL; I disagree, in Main's
  disfavour-of-itself direction). My independently written BQ query reproduces
  every bucket: 34 BUYs, `rows_after_join == distinct_buys == 34` (no fan-out),
  `no_row_within_2s = 14`, `joined_truncated = 19`, `joined_fs_present = 1`
  (DELL), `pt_verdict_pop = 19`, `both_absent = 14`; verdict distribution
  `'' 15 / APPROVE_REDUCED 15 / REJECT 3 / APPROVE_HEDGED 1` (15 empty = 14 +
  DELL). Rule stated, positive control (DELL) detected, buckets sum to the
  population. The criterion asks for the count + rule + a controlled zero; all
  are delivered, and the 14 with no verdict in EITHER source fall outside its
  "while a completed risk verdict existed" predicate.
- **C8 MET.** AST: **9 -> 38** test functions, **17 -> 62** asserts; `grep -c
  'assert '` = 64, inflated by 2. Baseline recomputed by me from
  `git show 9d14291e^:<path>`.
- **C9 MET.** MY OWN in-memory matrix (sha256 `042cd8e5eca44783` identical
  before/after; no tree write, so no restore to get wrong). CONTROL GREEN FIRST
  (rc=0). M1 `if raw is None`->`if not raw` KILLED; M2 sizing seam ->
  `or 10.0` KILLED; **MQ1 (mine, absent from Main's matrix)** re-gate the
  nested-first resolution behind the flag KILLED; **MQ2** SIZE-with-None ->
  DEFAULT KILLED; **MQ3** UNPARSEABLE -> DEFAULT KILLED; **MS1** delete the swap
  $50 floor KILLED; **MS3** TRUE ORPHAN KILLED. One EQUIVALENT mutant (MS2,
  floor re-applied immediately before the SELL with nothing in between) survived
  and is correctly NOT a finding -- behaviourally identical to shipped code.
- **C10 MET.** No threshold/gate/cap weakened; every change strictly tightening.
  Live check: `GET /api/paper-trading/portfolio` still shows DELL qty 4.806437,
  cost_basis 2392.26 -- not liquidated, not resized. Diff touches no executor.

## WARN -- residual falsy-zero of THIS EXACT CLASS, undisclosed and unqueued
`backend/services/autonomous_loop.py:3091-3094` (Claude lite judge) and
`:3337-3340` (Gemini lite judge):
```python
"recommended_position_pct": float(
    risk_dict.get("recommended_position_pct")
    or _LITE_RISK_DEFAULT["recommended_position_pct"]   # = 3.0
),
```
MEASURED, not argued: judge says `0.0` -> value written into `risk_assessment`
is **3.0** -> `decide_trades` emits **BUY $719.93**, where the true `0.0`
produces **[]**. The zero is destroyed UPSTREAM of the fixed helper, so 86.74's
fix cannot reach it. Pre-existing (phase-25.A, 9c5eb8ad, 2026-05-12); NOT
introduced here; named by no criterion (they scope the helper, the 10%-NAV
default and `decide_trades`). Grep over all five 86.74 artifacts for
`_LITE_RISK_DEFAULT|lite risk|3091|3337`: **zero hits** -- not disclosed, and
not in D1-D5 of queued_defects_from_86.74.md. Live-harm bound: prod .env has
`paper_risk_judge_reject_binding=True`, so a lite REJECT is blocked on the
DECISION leg; exposure needs a non-REJECT decision with pct 0.0, or that env
line absent (its `Field` default is False) -- i.e. the protection is
env-config-dependent, which is the fragility criterion 1 exists to remove.

COMPLETED: 2026-08-15T13:47:05Z
