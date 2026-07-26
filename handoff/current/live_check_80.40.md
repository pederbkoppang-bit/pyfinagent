# live_check — phase-80.40

**Required (masterplan, verbatim):** *Verbatim curl of `/api/paper-trading/performance` showing
`max_drawdown_pct` present and numeric, plus a Playwright capture of `/paper-trading/positions`
showing the kill-switch row rendering a REAL verdict instead of NO DATA.*

## §A. Method

Same isolated rig as `live_check_36.7.md` (`:8001`/`:3100`, real unmodified data, operator's
`:8000`/`:3000` never touched — see that file for the full method disclosure, not repeated here).

> ## RESOLVED 2026-07-26 on the operator's OWN `:8000` (restart authorized as standing practice)
> ```
> $ curl -s http://localhost:8000/api/paper-trading/performance
>   sharpe_ratio: 3.44
>   max_drawdown_pct: -5.31
> ```
> Present, numeric, negative, on the live instance after `launchctl kickstart` (pid 70791 → 76381).
> The rig evidence below stands as the pre-restart proof; this is the criterion met literally.

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

## §E. Teardown — **PRE-RESTART reading, superseded; not a contradiction of the RESOLVED block**

> **Annotation added cycle 3 (post-Q/A).** Everything in this section was measured **before** the
> operator-authorized restart recorded at the top of this file. `pid 70791` is therefore the
> pre-restart pid; the backend is now launchd pid `76381` (measured with
> `launchctl print gui/$(id -u)/com.pyfinagent.backend` — note that `lsof -ti tcp:8000` is NOT a
> reliable way to read it, it returns browser connection processes). "Untouched throughout" was true
> of the rig work described here. The later restart was a separate, authorized action, and it is what
> made criterion 1 satisfiable on the operator's own instance. Both statements are true of their own
> moment; this annotation exists so a reader of the archive cannot mistake them for a conflict.

Shared with `36.7` — see `live_check_36.7.md` §G. Confirmed once for both steps: rig torn down,
tsconfig/next-env restored, operator's `:8000` (200, pid 70791 at that time) and `:3000` (302)
untouched throughout.

## §F. Criteria summary

| # | Criterion | Status |
|---|---|---|
| 1 | `max_drawdown_pct` present and numeric | **MET** — §B |
| 2 | computed in `perf_metrics.py` | **MET** — verified by reading the file; no inline computation in `paper_trading.py` |
| 3 | test pins a known drawdown, fails against a `0`-stub | **MET** — mutation **run by Main**, verbatim `AssertionError: assert 0.0 == -30.0` recorded in `experiment_results_80.40.md` §"Criterion 3 — the 0-stub mutation, RUN BY MAIN". (Cycle-3 correction: this cell previously said "workflow-reported and reproduced", which is a second-hand claim and is exactly what the criterion's "record that failing output verbatim" clause forbids.) |
| 4 | -15%/10% threshold reconciliation | **MET** — see contract §Criterion 4; label-only fix, no value changed |
| 5 | `80.36` behaviour unchanged | **MET** — §D |
| 6 | mutation-tested | **MET**, plus Main's own `R7` addition |
