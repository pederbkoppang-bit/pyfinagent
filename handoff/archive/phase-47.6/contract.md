# Contract — phase-47.6: Dynamic strategy rotation (per-strategy DSR selector + anti-churn)

**Cycle:** 7 of the production-ready+money push (priority 5, the NORTH STAR). FREE of LLM spend.
**Step:** 47.6 | **Phase:** phase-47 | **Status:** in-progress | **Harness:** required | **Tier:** complex.

## Research-gate summary (PASSED)
Researcher `acb97aa2cc1ee7618`, tier=complex, `gate_passed: true`. 6 sources in full, 19 URLs, recency
scan, 11 internal files. Brief: `research_brief_phase_47_6_strategy_rotation.md`.

KEY FINDING: the rotation machinery largely EXISTS (quant_optimizer rotates `strategy` as a search
param; `analytics.compute_deflated_sharpe`; `gate.PromotionGate` DSR>=0.95/PBO<=0.20;
`friday_promotion.py` weekly DSR-desc ranking + `promoted_strategies` BQ write; `autonomous_loop.py:46
load_promoted_params` consumes it). The MISSING piece is the incumbent-vs-challengers SELECTION with
anti-churn hysteresis. Today the live strategy is STATIC (triple_barrier, Sharpe 1.17).

## Hypothesis
A pure `select_best_strategy()` that reuses PromotionGate (gate), mirrors friday_promotion's
DSR-desc/PBO-asc ranking, and adds an anti-churn min-improvement (Delta-DSR) vs the incumbent gives the
system the north-star "shift to highest earner" behavior WITHOUT whipsaw, and feeds the live loop via
the existing promoted_strategies path. Unit-testable with synthetic DSR dicts (no live backtest needed
for the committed deliverable).

DESIGN REFINEMENT (vs the brief's placeholder): gate-passers have DSR in [0.95, 1.0], so the max
possible Delta-DSR is 0.05 -- a 0.05 min_improvement would make switching impossible. Default set to
**0.01** (one DSR-point), band-appropriate + parameterized.

## Immutable success criteria (verbatim from masterplan.json phase-47.6)
1. NEW backend/autoresearch/strategy_selector.py::select_best_strategy is a PURE function that
   gate-filters per-strategy candidates by DSR>=0.95 & PBO<=0.20 (REUSING PromotionGate, no
   metric-formula duplication), ranks passers DSR-desc/PBO-asc, and switches from the incumbent ONLY
   when the top challenger's DSR exceeds the incumbent's by >= min_improvement (anti-churn hysteresis);
   favors incumbent on tie; handles first-selection (no incumbent) and no-passer (retain incumbent)
2. NEW tests/autoresearch/test_strategy_selector.py has >=6 behavioral cases (top-DSR pick, DSR-gate
   veto, PBO veto, anti-churn below-min-improvement retains incumbent, incumbent-tie, first-selection)
   and all pass
3. reuses existing PromotionGate (DSR/PBO not re-implemented); ast.parse clean; pytest green

## Plan steps
1. NEW `backend/autoresearch/strategy_selector.py::select_best_strategy(per_strategy, incumbent, *,
   gate, min_improvement=0.01, num_trials=5)` -- pure; reuses PromotionGate; DSR-desc/PBO-asc ranking;
   anti-churn hysteresis; first-selection + no-passer + incumbent-tie handling.
2. NEW `tests/autoresearch/test_strategy_selector.py` -- 8 behavioral cases.
3. Verify: ast.parse + `pytest tests/autoresearch/test_strategy_selector.py`. Fresh Q/A.

## Blast radius
ADDITIVE: one new pure module + its test. No change to the live loop, gate, or promotion path this
cycle (the selector is callable but not yet wired into a cron). No LLM spend, no trade execution.

## Deferred (documented, NOT this step)
- Live per-strategy DSR population via 5 quant-only walk-forward backtests (the selector's real inputs).
- Weekly cron scheduling of the selection sweep + writing the choice to promoted_strategies.
- Real-capital activation (stays paper-only).
- effective-N clustering of the correlated strategies (v1 uses plain num_trials -> over-deflates = safe).

## References
- `research_brief_phase_47_6_strategy_rotation.md` (gate); `roadmap_master.md` workstream 6
- `backend/autoresearch/gate.py::PromotionGate`; `backend/autoresearch/friday_promotion.py` (ranking pattern)
- `backend/backtest/analytics.py:239` compute_deflated_sharpe; `backend/services/autonomous_loop.py:46` load_promoted_params
- Bailey & Lopez de Prado 2014 (DSR best-of-N); jump-model 2024 (switch-penalty turnover reduction)
