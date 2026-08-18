STATUS: COMPLETE -- write-first record, still NOT a verdict
STEP: 86.74
WRITTEN: 2026-08-14T15:26:33Z
COMPLETED: 2026-08-14T15:47:00Z

Drop-recovery spawn (Agent-tool fallback). 4 prior attempts, 0 returned verdicts.

## Harness compliance 5/5 (run by me)
- research_brief_86.74.md: brief_status COMPLETE, gate_passed true,
  external_sources_read_in_full 7 (>=5). ON DISK, verified.
- contract_86.74.md mtime 16:19 < generate commit 9d14291e 16:36:20 +0200.
  Research brief 12:24 < contract. Order holds.
- experiment_results_86.74.md present, 20,203 B.
- evaluator_critique_86.74.md present, records NO VERDICT (2 rail drops). This
  spawn supplies the verdict.
- harness_log.md Cycle 190 phase=86.74 result=NO-VERDICT. Masterplan 86.74 still
  `pending` -- no premature flip.

## Deterministic (run by me)
- Immutable cmd: `37 passed, 1 warning in 1.96s`, exit 0. MATCHES artifact header.
- Lint F821/F401/F811 over `git diff --name-only 8b520f6c..HEAD -- '*.py'`
  = 6 files (non-empty asserted): "All checks passed!" exit=0.
- Mutation matrix re-run BY ME: control GREEN, M1..M6 all KILLED, restore
  byte-identical (my own md5 before/after on 4 subjects: all 4 match).
- Counts re-derived by AST: 34 test functions, 55 asserts, 3 parametrized x2
  = 37 collected. Both the "34" and the "37" in the artifact reproduce and
  measure different things. No transcription drift.

## Spot-checks (3 load-bearing claims)
1. NOTHING LOOSENED -- CONFIRMED. All 4 `or 10.0` sites (531, 824, 877, 902)
   now call `_sizing_pct`. Legacy max output 10.0; new max output 10.0, and
   10.0 is now reachable ONLY from ABSENT. 0.0/UNPARSEABLE/unrecognised-state
   all return 0.0 (below the $50 floor -> no order). Every input either
   unchanged or strictly less likely to buy. risk_debate.py = ticker in log
   only; autonomous_loop.py = additive column write.
2. THE TWO REWRITTEN TESTS ARE AN INVERSION -- CONFIRMED. `TestFlagOffLegacy`
   asserted the DEFECT (`abs(amount - NAV*0.10) < 0.5`, `_buy is not None` on
   a REJECT). Replaced by `TestRejectBindsInBothFlagStates`, parametrized over
   BOTH flag states, asserting `_buy(orders) is None` on REJECT/0%. Strictly
   stronger: forbids the buy the old tests required.
3. MUTATION MATRIX 6/6 WITH GREEN CONTROL -- CONFIRMED independently.

## My own finding (WARN, not blocking)
`test_every_sizing_site_routes_through_the_single_seam` matches only
`ast.Constant == 10.0`. I verified the evasion: a reintroduction written
`cand.get("position_pct") or DEFAULT_POSITION_PCT` is NOT detected (Name node,
scan returns []), while the literal form IS. Sites 824/877/902 live in
`_compute_swap_candidates`, which NO test drives -- so for 3 of 4 sizing sites
that source scan is the sole regression guard. Vacuity shape #2. Current tree
is correct; this is regression-resistance, not a live defect. C3's behavioural
core (`_sizing_pct` units + test_default_is_reachable_from_ABSENT_AND_NOTHING_
ELSE) is genuine and non-vacuous, so C3 is MET.

## C4 / C7 ruling (independent)
C4 PARTIAL: write exists, mutation-tested (M3 KILLED), baseline 0-of-129
measured, root cause located. Post-fix BQ share NOT measured. Backend pid 27945
started 13:30:35 CEST, BEFORE both commits (16:36:20, 17:05:05) -- committed but
NOT in force. The measurement is temporally impossible without a restart the
operator's standing batch-to-session-end instruction forbids. Not defective, not
covered -> Missing_Assumption, caps at CONDITIONAL. NOT FAIL: the criterion is
substantially addressed and the gap is stated, not proxied.
C7 PARTIAL: population derived from the table (34 BUYs), enumeration rule
stated, positive control passes (DELL detected True), 1 confirmed, 33
UNDETERMINED and explicitly NOT claimed as a measured zero. The criterion's
anti-false-zero demand is MET; its completeness is not. Missing_Assumption,
caps at CONDITIONAL.
Main self-reported both as partial. Graded on the evidence, not on the framing:
I re-derived both gaps and reached the same place independently.

## Sequence
qa_wip.py: attempt_number 5 (lower bound, status ok, source_present true),
prior_attempts 4. verdict_history_86_21.py --evidence-only: no_rows_for_step.
qa_wip (4 prior) > ledger (0) -> LEDGER IS STALE; sequence unreliable from it.
From the WIP records: 4 prior attempts, ZERO returned verdicts (all rail drops,
NO_VERDICT). No prior CONDITIONAL and no prior FAIL exists for this step.

VERDICT COMPUTED: CONDITIONAL (worst-of-3-lenses: correctness PASS,
reproduces PASS, scope-honesty PASS -> capped by C4+C7 completeness).
