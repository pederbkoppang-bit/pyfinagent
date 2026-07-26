# live_check — phase-80.40

**Required (masterplan, verbatim):** *Verbatim curl of `/api/paper-trading/performance` showing
`max_drawdown_pct` present and numeric, plus a Playwright capture of `/paper-trading/positions`
showing the kill-switch row rendering a REAL verdict instead of NO DATA.*

## §A. Method

Same isolated rig as `live_check_36.7.md` (`:8001`/`:3100`, real unmodified data, operator's
`:8000`/`:3000` never touched — see that file for the full method disclosure, not repeated here).

## §B. Criterion 1 — `max_drawdown_pct` present and numeric

```
$ curl -s http://localhost:8001/api/paper-trading/performance
{
  "nav": 23838.16,
  "sharpe_ratio": 3.44,
  "max_drawdown_pct": -5.31,
  "pnl_pct": 19.19,
  ...
}
```

Present alongside `sharpe_ratio` (proving this is not a null-`perf` case — the object arrives and
this field now arrives with it), numeric, negative (the correct sign convention — see contract for
the verification against `analytics.py`).

## §C. Criterion — Playwright capture, real verdict

`handoff/current/captures_36.7_80.40/36.7_80.40_ARMED_real_verdict.png`:

> `Max drawdown (-15%)` → **`SAFE`**
> `Drawdown` → **`-5.3% / -15%`**

Both are real, measured verdicts derived from `-5.31`, not the `NO DATA` that `80.36`'s
presence-check correctly renders when the field is absent. This is the exact criterion:
*"rendering a REAL verdict instead of NO DATA."*

## §D. `80.36`'s behaviour independently reconfirmed unchanged

- **Field present → real verdict:** §C, above.
- **`perf` absent → still `NO DATA`:** covered by the existing `80.36` test suite
  (`cockpit-helpers.risk.test.tsx`), re-run in this cycle: **6 passed**, no change to any assertion.

## §E. Teardown

Shared with `36.7` — see `live_check_36.7.md` §G. Confirmed once for both steps: rig torn down,
tsconfig/next-env restored, operator's `:8000` (200, pid 70791) and `:3000` (302) untouched
throughout.

## §F. Criteria summary

| # | Criterion | Status |
|---|---|---|
| 1 | `max_drawdown_pct` present and numeric | **MET** — §B |
| 2 | computed in `perf_metrics.py` | **MET** — verified by reading the file; no inline computation in `paper_trading.py` |
| 3 | test pins a known drawdown, fails against a `0`-stub | **MET** — workflow-reported and reproduced |
| 4 | -15%/10% threshold reconciliation | **MET** — see contract §Criterion 4; label-only fix, no value changed |
| 5 | `80.36` behaviour unchanged | **MET** — §D |
| 6 | mutation-tested | **MET**, plus Main's own `R7` addition |
