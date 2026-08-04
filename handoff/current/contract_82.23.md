# Contract -- step 82.23

**Step id:** 82.23 | **Priority:** P0 | depends_on: ['82.22']
PLAN phase, before GENERATE.

## Research gate

`handoff/current/research_brief_82.22_82.23.md` -- **gate_passed: true**
(8 sources read in full, 33 URLs, recency scan, 14 internal files). ONE gate
feeding both contracts: 82.22 and 82.23 are one coupled surface
(`optimizer_best.json` + the promotion gate).

**Integrity note from the gate, recorded because it bears on trust:** the
researcher's first WebFetch over the DSR paper returned two fluent quotes that
appear NOWHERE in the PDF. It caught this itself by re-extracting locally with
pdfplumber, named the fabricated strings in the brief, and re-verified every
remaining quote. No contract cites the fabricated text.

## Hypothesis

The promotion gate is **sound but starved**. It is not fail-open.

## CORRECTION carried into this contract

An earlier framing said "PBO <= 0.5 was never computed, so the gate ran on one
term", implying silent promotion. Measured, that is wrong in the failure
direction:

| gate | threshold | on missing PBO | live? |
|---|---|---|---|
| `backend/autoresearch/gate.py:22` | **0.20** | **fail-CLOSED** (`missing_dsr_or_pbo`) | YES via `backend/autoresearch/friday_promotion.py:59` |
| `backend/services/promotion_gate.py:37` | 0.5 | fail-OPEN (defaults 0.0) | NO -- zero callers |
| `backend/backtest/analytics.py:198` | 0.5 | doc only | n/a |

`backend/autoresearch/friday_promotion.py:108` additionally defaults a missing
PBO to **1.0** -- the worst value. So a missing PBO never promoted anything; it
BLOCKED promotion. The real gap is that the backtest lane emits no PBO at all,
so its candidates are dropped as `missing_dsr_or_pbo`.

## The architectural constraint that shapes this step

**PBO CANNOT live in `generate_report`.** Its signature
(`backend/backtest/analytics.py:649`) takes ONE `BacktestResult`; all call sites
pass a single run; and `compute_pbo` returns **0.0 -- the best possible value --
when N < 2** (`backend/backtest/analytics.py:207`). Adding it there would emit a
hard-coded PASS on every run. PBO must be produced where N configurations
exist: the sweep level.

**And the optimizer's own trials cannot be reused as that matrix.** Bailey
Algorithm 2.3 is explicit that for a guided search the columns must be "the
final outcome of each guided search ... and not the intermediate steps".
`QuantOptimizer.run_loop` IS a guided search (keep/discard against a running
best, `backend/backtest/quant_optimizer.py:293`), so stacking its ten
nav_histories yields a number wrong in an unquantified direction.

## Plan

1. A sweep-level PBO producer over INDEPENDENT configurations (the phase-82.3
   runner's factorial shape), reusing `compute_pbo` rather than re-implementing.
2. Emit `pbo` + `pbo_n_trials` + `pbo_matrix_shape` so the gate never receives a
   bare number without its N.
3. **Refuse rather than emit a silent 0.0**: N<2 or T<S*2 must produce an
   explicit absence, which the live gate already treats as fail-closed.
4. Reconcile the three PBO thresholds, or document why they legitimately differ.
5. No change to the live gate's fail-closed behaviour.

## Immutable success criteria (copied programmatically from .claude/masterplan.json)

1. generate_report emits a pbo field for a run whose daily NAV series and K>=2 configurations are available, asserted on a fixture
2. a fixture whose T is below S*2 causes the reported pbo to be ABSENT or explicitly flagged, never the silent 0.0 that would pass the <=0.5 gate -- asserted by a test that fails if 0.0 is emitted for an under-length matrix
3. the promotion gate refuses a candidate whose pbo exceeds the threshold, asserted on a fixture carrying the measured incumbent value 0.7486
4. a trial-diversity number (mean pairwise column correlation) accompanies every reported pbo, asserted present on the same fixture

**Verification command:** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_23_pbo_in_gate.py -q`

## Out of scope

No live-funnel change. No re-run of 82.3 (that is 82.24). The K-floor
(`_DEFAULT_K = 8` vs the paper's `N >> 10`) and the `num_trials` reset are
queued as their own steps rather than folded in.
