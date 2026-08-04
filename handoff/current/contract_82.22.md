# Contract -- step 82.22

**Step id:** 82.22 | **Priority:** P0 | depends_on: None
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

The mis-attribution is mechanical and reproducible: `_save_best_params`
(`backend/backtest/quant_optimizer.py:720`) writes `"run_id": self._run_id` --
the CURRENT run -- alongside `self.best_sharpe` / `self.best_dsr`, which
`_load_previous_best` (`:737`) may have inherited from a PRIOR run without ever
recording where they came from. When the current run beats nothing (`kept=0`),
the prior metrics are re-stamped with the new run's identity. That is exactly
`52eb3ffe`'s 1.17046/0.95258 appearing as `60617e0b` on 2026-07-24.

## Two findings that change the fix

**(a) 1.1705 is not "correct but mis-labelled" -- it is STALE.** Six of run
60617e0b's ten trials returned the identical Sharpe **0.6455483636** (measured:
`Counter` over the ten artifacts). That is the incumbent params re-measured
under current code and data. So the honest present-day figure is ~0.646, and
the fix must not preserve 1.17 as though it were merely mis-addressed.

**(b) `_load_previous_best` resets `self.num_trials = 1`** (`:759`, `:789`) on
every warm start, so a carried-forward DSR is under-deflated relative to the
true cumulative search. Bailey/Lopez de Prado call the undisclosed trial count
the single most important missing field in a backtest. Queue separately.

## Blast radius -- MEASURED, and narrower than assumed

15 consumers read this file. **All are `dict.get`-based, so ADDING keys is safe
for every one**; renaming `params`/`sharpe`/`dsr`/`run_id` would break named
consumers. Correction to an earlier assumption:
`backend/services/paper_go_live_gate.py::compute_gate` does **NOT** read DSR
from this file -- it takes live paper DSR from `compute_metrics_v2` (`:136`).
The file reaches that gate indirectly via `backend/services/perf_metrics.py`
-> `compute_sharpe_gap` tier 1 -> the `sr_gap_le_30pct` boolean, and directly
via `backend/autoresearch/rotation_runner.py` (the incumbent DSR bar every
challenger must beat). Those two are the load-bearing paths.

## Plan

1. ADDITIVE schema only: `metrics_run_id`, `metrics_source_artifact`,
   `warm_started_from`, `num_trials`, `schema_version`. Every existing key
   keeps its meaning and position.
2. `_save_best_params` records which run ACTUALLY produced the persisted
   metrics, not merely which run was executing.
3. A consistency check that FAILS on the current on-disk state.
4. **Absence must not read as freshness**: a file lacking `metrics_run_id` is
   treated as unknown-provenance, never as self-attributed.
5. No renames. No consumer changes.

## Immutable success criteria (copied programmatically from .claude/masterplan.json)

1. a check asserts the sharpe/dsr recorded in optimizer_best.json are reproducible from a saved result artifact whose run_id matches the recorded run_id, and FAILS on the current on-disk state
2. when the best is warm-started from a prior run, the source run id is persisted as a distinct field from the current run id, asserted on a fixture
3. a fixture where kept=0 asserts the file cannot claim a best that no experiment in that run produced
4. the deflation math is asserted UNCHANGED: a fixture reproduces monotonically decreasing DSR as num_trials rises, so a fix to provenance cannot silently alter the statistic

**Verification command:** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_22_optimizer_best_provenance.py -q`

## Out of scope

No live-funnel change. No re-run of 82.3 (that is 82.24). The K-floor
(`_DEFAULT_K = 8` vs the paper's `N >> 10`) and the `num_trials` reset are
queued as their own steps rather than folded in.
