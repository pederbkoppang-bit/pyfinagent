STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.58
WRITTEN: 2026-08-13T20:14:06Z

# Q/A write-first record -- step 86.58 (dead signal_downgrade rule)

Workflow rail, EVALUATE. qa.md read in full at runtime.

## Attempt count (qa.md 3rd-CONDITIONAL rule)
`python scripts/qa/qa_wip.py 86.58` -> records_retained=1 (my OWN file), prior_records=[].
=> prior attempts = 0. **Attempt 1.** Prior-verdict sequence: (none).
harness_log cross-check: `grep -F 86.58` finds only the filing mention at :34336, no
`result=` row. Ledger and log agree: attempt 1.

## A. HARNESS COMPLIANCE -- CLEAN
- Research gate: brief 31,129 chars, envelope `brief_status: COMPLETE`,
  `gate_passed: true`, sources_read_in_full=6 (>=5), urls=38 (>=10),
  `recency_scan_performed: true`, recency section at :255. PASS.
- Criteria verbatim: all 6 masterplan criteria are byte-present in contract_86.58.md
  (programmatic `x in contract` check, 6/6 True).
- Log-last: masterplan 86.58 status=pending, no harness_log result row. Respected.
- No verdict-shopping: attempt 1.
- Protocol order: DISCLOSED breach (criteria 1-2 ran pre-contract). Disclosure is
  honest and even names that an mtime check would pass (mtimes: brief 21:22 <
  contract 21:24 < script 21:25 < results 21:39, all "clean"). NOT harmless -- it is
  the direct cause of Finding 2.

## B. DETERMINISTIC
- Immutable command (from masterplan, not from the artifact): `parses`, **exit=0**.
- Driven proof re-run BY ME: exit=0, output byte-equivalent to the pasted block.
  Control cell B green; A and C dead. REPRODUCES.
- `git show --stat 9740c64f` = masterplan.json, contract_86.58.md,
  scripts/qa/drive_86_58_dead_downgrade.py. **No backend/ file touched.**
- `git status --short`: only hook-written handoff jsonl + untracked away_ops +
  my WIP. No unintended production change.
- Lint (scope DERIVED from the commit, non-empty asserted):
  `uvx ruff check --select F821,F401,F811 scripts/qa/drive_86_58_dead_downgrade.py`
  -> "All checks passed!" exit=0.
- `pytest backend/tests/test_phase_61_2_decision_integrity.py -q` -> **33 passed**.

## FINDING 1 (BLOCKING) -- criterion 3 measured a PROXY, and the answer INVERTS
The script ASSERTS both flags are False and aborts otherwise (drive_*.py:60). The
"flag-ON blast radius" was inferred by HAND-SETTING pos.recommendation='BUY'
(cell B) -- the assumed post-fix value. The production flag-read was never run.

Driven with the flags actually ON (`Settings().model_copy(update={...})`):
```
UNRECOGNISED token = '__UNRECOGNISED__'
flags ON? True True
resolve('new_buy_signal', flagON) = '__UNRECOGNISED__'
resolve('BUY',            flagON) = 'BUY'
F CONTROL flagsON pos=BUY            fresh=HOLD -> [('NTAP','SELL','signal_downgrade')]  GREEN
E         flagsON pos=new_buy_signal fresh=HOLD -> []
G         flagsON pos=swap_buy       fresh=HOLD -> []
H         flagsON pos=Strong Buy     fresh=HOLD -> [('NTAP','SELL','signal_downgrade')]
```
DISCRIMINATION CONTROL (proves the probe reads flag state, not a fixed answer):
  'Strong Buy' + HOLD, flags OFF -> []   ;   flags ON -> [(...,'signal_downgrade')]

Mechanism: flag-ON `_resolve_rec` returns `canon if canon is not None else
_UNRECOGNISED_REC`; 'new_buy_signal' -> None -> `__UNRECOGNISED__`, member of NONE
of _BUY_RECS/_SELL_RECS/_DOWNGRADE_RECS. And `_pos_rec` is written ONLY by
execute_buy (paper_trader.py:488,:512); the partial-sell path at :676 PRESERVES the
stored value. Flipping the flags does not rewrite existing rows.

Refuted claims (experiment_results_86.58.md:162-168):
  "Blast radius: 1 of 1 currently-held positions (100%)"
  "Promoting the 06-8 flags today would sell the book's only position"
Measured: **0 of 2 currently-held rows become candidates at promotion time.**
Exposure begins at the next execute_buy that rewrites the field. The fix is NOT
inert -- production does pass analysis_recommendation (portfolio_manager.py:578,:918
-> autonomous_loop.py:251,:1768) -- it is simply not retroactive.
This propagates into criterion 4: the operator recommendation's DIRECTION is
defensible, its stated JUSTIFICATION is measurably wrong.

## FINDING 2 (WARN) -- criterion 2 counts stale in the published artifact
Main's own SQL, re-run verbatim: {'rec':'new_buy_signal','n':2,'in_closed_set':0}.
  NTAP qty 5.346643 entry 2026-07-31T18:47:37Z
  DELL qty 4.806437 entry **2026-08-13T19:31:19Z** (= 21:31 local)
Artifact (written 21:39 local) says "TOTAL 1 rows", "tickers=1", "book holds 1
position" -- stale by 8 minutes at publication. Method and query are CORRECT and the
qualitative finding is STRENGTHENED (2 of 2 = 100% off-vocab, 0 in closed set).

## FINDING 3 (NOTE) -- the pasted verification command does not run
experiment_results:214 drops the closing `"`; run as printed:
`bash: -c: line 0: unexpected EOF while looking for matching '"'`.
The masterplan's stored command is correct and I ran it: exit 0. Substance fine,
"verbatim" block is not.

## FINDING 4 (NOTE) -- live_check_86.58.md absent
Required by masterplan verification.live_check, incl. "flag values read from the
RUNNING process" -- which the artifact honestly discloses it cannot obtain (flags
absent from GET /api/settings/). Open item for closure.

## FINDING 5 (NOTE) -- NTAP re-eval table stale by one row
Re-derived from financial_reports.analysis_results: 10 rows since 2026-07-24; all 9
of Main's rows reproduce EXACTLY. New row 2026-08-13 'Hold' 6.2 empty_summary=True.
"7 of 9 empty-summary HOLD" and "5 carry 0.0" both reproduce on the 9-row snapshot.
81.2% figure traces to q1_binding_constraint_86.59.md:76 (uncited inline).

## CRITERION VERDICTS
1 MET  -- driven, control green, reproduced by me.
2 MET-with-stale-counts (WARN) -- query stated, method right, cardinality wrong.
3 NOT MET -- flag-ON path never executed; published number refuted by driving it.
4 MET-in-form / justification wrong -- no flag promoted (verified), but the recorded
  blast radius is the refuted one.
5 N/A ACCEPTED -- criterion is conditional ("any guard added"); no guard added. I
  verified the write seam at paper_trader.py:452 is genuinely unguarded and reached
  only from execute_buy, so deferring it to the operator-gated flag decision + 86.63
  is a defensible reading, not an evasion.
6 MET  -- zero backend/ diff; the UNRECOGNISED line fired twice in MY own run.

## SEPARATION OF DUTIES
Main edited qa.md today (86.75). Checked: neither deleted clause bears on this
verdict (no prior verdict exists to anchor to; the removed quant rubric is
irrelevant to a harness/measurement step), and the counter reads attempt 1 under
BOTH the ledger and the log. Operator review of that edit should still stand.

VERDICT RETURNED: FAIL (criterion 3).
COMPLETED: 2026-08-13T20:26:40Z
