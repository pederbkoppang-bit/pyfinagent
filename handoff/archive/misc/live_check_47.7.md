# Live Check — phase-47.7: Learn-loop correctness fix

Deterministic step (no live system needed for the committed deliverable). Verbatim output, 2026-05-29.

## Immutable command (ast + pytest) -- exit 0
```
$ python -c "import ast; ast.parse(open('backend/services/autonomous_loop.py').read())"  -> ast OK
$ python -m pytest backend/tests/test_phase_35_1_learn_loop_writer.py -q                 -> 5 passed in 2.18s
```

## Field fix targets the real field (Q/A-independently-confirmed)
`paper_trader.py:364` writes the SELL row with `realized_pnl_pct` (NO `return_pct` key). The fix
(autonomous_loop.py sell-close fallback) reads `realized_pnl_pct` first, `return_pct` fallback, then 0.0.

## Mutation guard (genuine, not tautology)
De-masked test mock = `{realized_pnl_pct: 17.89}` (NO return_pct key):
```
PRE-FIX  read return_pct        -> 0.0    (test assert ==17.89 -> FAILS)
POST-FIX read realized_pnl_pct  -> 17.89  (test assert ==17.89 -> PASSES)
```
Q/A independently reproduced this end-to-end through the real `_learn_from_closed_trades`.

## Operator-gate respected
`settings.py:32 paper_learn_loop_enabled: bool = Field(False, ...)` -- UNTOUCHED (its description says
"operator flips to true"). No Anthropic reflection spend enabled. `test_field_default_off` passes.

## Deferred (NOT live this cycle)
LIVE outcome_tracking row from a sell-close needs the operator to flip the flag + a cycle to run (the
reflection fan-out uses claude-sonnet-4-6 = Anthropic-metered = operator-gated). save_outcome append-only
dedup + DoD-6 probe's non-existent cycle_id column are flagged follow-ups.
