# Contract -- step 82.3

**Step id:** 82.3 | **Priority:** P1 | depends_on: ['82.0', '82.15', '82.2']

## RETROSPECTIVE ARTIFACT -- read this first

**This contract was written AFTER the 56 runs completed.** The 82.3 Q/A found
that no contract for this step existed anywhere: the rolling
`handoff/current/contract.md` was overwritten by step 82.4's, so 82.3's three
immutable criteria lived only in `.claude/masterplan.json`.

That missing file is the **structural root** of the cycle-1 criteria-erosion
breach. When I spawned the Q/A I summarised the criteria from memory instead of
reading them from a contract -- because there was no contract to read. Copying
them programmatically into the *spawn prompt* repaired the symptom; this file
repairs the cause. The rolling-filename collision is itself the hazard: two
steps in flight, one `contract.md`.

**What was NOT retrofitted:** the criteria below are copied programmatically
from `masterplan.json` (unchanged since the phase was installed -- the Q/A
verified they are byte-identical to HEAD). The hypothesis and plan describe what
was actually done, not a reconstruction designed to match the outcome. The
pre-registration that matters -- the RANKING RULE -- was genuinely fixed before
any number existed and the Q/A verified it by mtime arithmetic: `contract.md`
(then holding 82.4's ranking section) was written 89 seconds after pass A
launched and ~17 minutes before run 1 finished.

## Research gate

`handoff/current/research_brief_82.3.md` -- gate_passed: true (7 sources read in
full, ~28 URLs, recency scan). Its load-bearing findings, each verified by Main:

- `compute_pbo` returns **0.0 silently** when `T < S*2` (=32), and 0.0 **passes**
  a `<= 0.5` gate. Per-window returns give T~27 -> a fabricated pass on every
  strategy. The T-axis must be DAILY NAV returns.
- One matrix **per strategy**, columns = K configs of the same model
  (Bailey et al. Algorithm 2.3). A pooled cross-strategy matrix answers a
  different question.
- `total_return_pct` is **already net of commission** -- do not double-count.
- Measured runtime **20.3 +/- 0.3 min/run** (the `<30s` figure in
  `.claude/rules/backend-backtest.md` is stale by ~40x; queued as 82.20).
- Unequal holding periods are VALID but BIASED: the purge is strategy-blind
  (`backend/backtest/backtest_engine.py:665`), so `reversion_sigma` gets a 135d
  purge against a 15d label horizon. **A win is clean; a loss is confounded.**

## Hypothesis

The three phase-82.2 candidates can be measured against the `triple_barrier`
incumbent on one sample with all four gate metrics, and the result will show
whether any clears `DSR >= 0.95` and `PBO <= 0.5`.

## Immutable success criteria (copied programmatically from .claude/masterplan.json)

1. one backtest result row per candidate exists under backend/backtest/experiments/results/ and is appended to quant_results.tsv
2. each result row reports DSR, PBO, turnover and net-of-cost return against the triple_barrier incumbent computed on the same sample window
3. a test asserts each candidate's result file exists, parses, and carries all four metrics as numbers rather than nulls

**Verification command:** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_3_candidate_backtests.py -q`

**live_check:** verbatim optimizer/backtest console output for each candidate showing the sample window and the DSR/PBO/turnover/net-of-cost figures

## What was run

`scripts/harness/run_82_3_candidate_backtests.py`, K=8 factorial per strategy
(max_depth x min_samples_leaf x learning_rate), two passes:

| pass | window | strategies | runs | wall clock |
|---|---|---|---|---|
| A | 2018-01-01..2025-12-31 | triple_barrier, stretch_regime, reversion_sigma | 24 | 8h34m |
| B | 2024-07-01..2025-12-31 | + qarp | 32 | 1h23m |

`qarp` is absent from pass A because `historical_fundamentals` has zero rows
before 2024-06-30 (queued as 82.21). A 2.8-minute smoke run caught that before
the sweep launched, rather than after ~11 hours of zeros.

## Out of scope

No live-funnel change. No re-run after the gate repair -- that is 82.24, and
today's artifacts are its pre-repair baseline.
