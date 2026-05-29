# Live Check — phase-47.6: Dynamic strategy rotation (selector)

Deterministic pure-function step (no live system needed). Verbatim output, 2026-05-29.

## Immutable command (ast + pytest) -- exit 0
```
$ python -c "import ast; ast.parse(open('backend/autoresearch/strategy_selector.py').read())"  -> ast OK
$ python -m pytest tests/autoresearch/test_strategy_selector.py -q
........                                                                 [100%]
8 passed in 0.01s
```

## Reuse / no-duplication check
```
$ grep -c PromotionGate backend/autoresearch/strategy_selector.py        -> 5   (reused)
$ grep -c <dsr/pbo/deflation formula reimpl patterns>                    -> 0   (no metric duplication)
```

## Anti-churn correctness (Q/A-traced)
incumbent dsr=0.97, challenger dsr=0.975, min_improvement=0.01 -> delta 0.005 < 0.01 -> RETAIN incumbent
(reason=below_min_improvement). Switch fires only at delta >= 0.01. No-passer -> retain incumbent (never
cash). Selection gates+ranks on DSR (deflated, best-of-N correct), not raw Sharpe.

DEFERRED (NOT live this cycle): per-strategy DSR population via 5 quant-only backtests, the weekly cron,
real-capital activation. The selector is integration-ready via the existing promoted_strategies path.
