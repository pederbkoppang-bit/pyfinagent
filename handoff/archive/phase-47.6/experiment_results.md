# Experiment Results — phase-47.6: Dynamic strategy rotation (per-strategy DSR selector + anti-churn)

**Cycle:** 7 (priority 5, the NORTH STAR). FREE of LLM spend. **Result:** ready for Q/A.

## What was built (ADDITIVE: 1 pure module + 1 test)
1. NEW `backend/autoresearch/strategy_selector.py::select_best_strategy(per_strategy, incumbent, *,
   gate=PromotionGate(), min_improvement=0.01, num_trials=5)` -- the SELECTION layer for the north-star
   "shift capital to whichever strategy makes the most money":
   - Gate-filters candidates by DSR>=0.95 & PBO<=0.20 by **reusing `gate.PromotionGate`** (no metric
     re-implementation -- 5 PromotionGate refs, 0 DSR/PBO formula reimpl, grep-confirmed).
   - Ranks passers DSR-desc / PBO-asc (mirrors friday_promotion).
   - **Anti-churn hysteresis**: switches off the incumbent ONLY when the top challenger's DSR exceeds
     it by >= min_improvement; otherwise retains the incumbent (avoids whipsaw between near-tied
     strategies -- jump-model 2024: switch-penalty cut turnover 141%->44% net-positive).
   - Edge handling: first-selection (no incumbent), no-passer (retain incumbent, never go to cash),
     incumbent-is-top (retain). Pure; fail-open; ASCII-only.
2. NEW `tests/autoresearch/test_strategy_selector.py` -- 8 behavioral cases (synthetic DSR dicts, no
   live backtest): first-selection picks top DSR; DSR-gate veto; PBO veto; anti-churn below-threshold
   retains incumbent; switch on sufficient improvement; incumbent-is-top; no-passer retains incumbent;
   weak-incumbent-replaced-by-large-margin.

## Design refinement vs the research brief (disclosed)
The brief's placeholder `min_improvement=0.05` is unreachable: gate-passers have DSR in [0.95, 1.0], so
the max Delta-DSR between two passers is 0.05. Set the default to **0.01** (one DSR-point) -- band-
appropriate, and parameterized so the operator can tune. Documented in the module docstring.

## Verbatim verification output
```
$ python -c "import ast; ast.parse(open('backend/autoresearch/strategy_selector.py').read())"  -> ast OK
$ python -m pytest tests/autoresearch/test_strategy_selector.py -q                              -> 8 passed in 0.01s
$ grep -c PromotionGate strategy_selector.py        -> 5 (reused)
$ grep -c <dsr/pbo formula reimpl patterns>         -> 0 (no metric duplication)
```

## Success-criteria mapping (masterplan phase-47.6)
1. pure select_best_strategy: gate-filter via PromotionGate, DSR-desc/PBO-asc rank, anti-churn switch,
   incumbent-tie + first-selection + no-passer handling -- **MET**.
2. test_strategy_selector.py >=6 behavioral cases all pass -- **MET** (8 passed).
3. reuses PromotionGate (no DSR/PBO reimpl); ast clean; pytest green -- **MET** (5 refs, 0 reimpl).

## Scope honesty / deferred (documented)
This ships the SELECTION logic (the missing north-star piece) -- pure + unit-tested. EXPLICITLY
DEFERRED (NOT claimed): live per-strategy DSR population via 5 quant-only walk-forward backtests (the
selector's real inputs); the weekly cron that drives the sweep + writes the choice to
promoted_strategies; real-capital activation (stays paper-only); effective-N clustering of the
correlated strategies (v1 plain num_trials over-deflates = safe direction). The selector is callable +
integration-ready (the loop already reads promoted_strategies via load_promoted_params), but this cycle
does NOT wire the cron or run the backtests.

## Files
backend/autoresearch/strategy_selector.py (new), tests/autoresearch/test_strategy_selector.py (new),
.claude/masterplan.json (phase-47.6), handoff/current/{contract.md, research_brief_phase_47_6_strategy_rotation.md}.
