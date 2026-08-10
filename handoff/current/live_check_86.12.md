# live_check -- phase-86.12

Captured 2026-08-10 03:58:01 CEST. Every block is stdout.

## A. The immutable verification command
```
$ curl -s .../kill-switch | python3 -c "...print sod_nav, current_nav"
23833.94 23833.94
```

## B. The full live payload -- the state the book is in right now
```
{
    "paused": false,
    "pause_reason": null,
    "sod_nav": 23833.94,
    "sod_date": "2026-08-09",
    "peak_nav": 24666.57,
    "baseline_provenance": null,
    "current_nav": 23833.94,
    "breach": {
        "daily_loss_breached": false,
        "daily_loss_pct": 0.0,
        "daily_loss_limit_pct": 4.0,
        "trailing_dd_breached": false,
        "trailing_dd_pct": 3.3755,
        "trailing_dd_limit_pct": 10.0,
        "any_breached": false,
        "daily_baseline_missing": false,
        "daily_baseline_stale": true,
        "trailing_baseline_missing": false,
        "baselines_present": true,
        "armed": false
    },
    "thresholds": {
        "daily_loss_limit_pct": 4.0,
        "trailing_dd_limit_pct": 10.0
    }
}
```

## C. Provenance trace (criterion 1), DERIVED from the AST
```
backend/api/paper_trading.py:518    get_kill_switch_state              reads total_nav: YES (stored)
backend/api/paper_trading.py:581    resume_trading                     reads total_nav: YES (stored)
backend/services/paper_trader.py:1357   check_and_enforce_kill_switch      reads total_nav: YES (stored)
backend/services/paper_trader.py:1460   check_and_enforce_kill_switch      reads total_nav: YES (stored)
backend/agents/mcp_servers/risk_server.py:78     kill_switch                        reads total_nav: no (caller-supplied)
backend/services/kill_switch.py:1065   check_auto_resume                  reads total_nav: no (caller-supplied)
```

## D. Criteria 2 + 3 -- the measurement
```
==============================================================================
1. sod_snapshot ROWS (criterion 2) -- counted, not taken from the step
==============================================================================

journal rows total: 64   sod_snapshot rows: 10
The step says 8. If the number above differs, the step's figure is stale and THIS one governs.

sod_snapshot ts (UTC)                    nav  date        weekday
--------------------------------------------------------------------------
2026-07-27T06:19:46.567773+00:0     23838.16  2026-07-27  Mon
2026-07-28T16:45:41.951028+00:0     23772.49  2026-07-28  Tue
2026-07-29T18:06:06.008080+00:0     23772.49  2026-07-29  Wed
2026-07-30T18:42:04.957538+00:0     23772.49  2026-07-30  Thu
2026-07-31T18:47:30.141829+00:0     23772.49  2026-07-31  Fri
2026-08-03T19:29:34.433705+00:0     23803.94  2026-08-03  Mon
2026-08-05T19:34:47.386888+00:0     23830.46  2026-08-05  Wed
2026-08-08T20:58:29.379594+00:0     23830.46  2026-08-08  Sat
2026-08-09T13:03:44.126943+00:0     23833.94  2026-08-09  Sun
2026-08-09T13:08:40.510286+00:0     23833.94  2026-08-09  Sun

==============================================================================
2. WAS THE ANCHOR EVER TAKEN AT THE START OF A DAY? (criterion 2)
==============================================================================

A 'start-of-day' anchor taken at 18:00-21:00 UTC is a PRIOR-CLOSE anchor: US
markets close at 20:00/21:00 UTC. The hour of each snapshot is what decides
whether the daily-loss limit measures 'since today's open' or 'since last
night'.

  06:xx UTC  x1   ['2026-07-27']
  13:xx UTC  x2   ['2026-08-09', '2026-08-09']
  16:xx UTC  x1   ['2026-07-28']
  18:xx UTC  x3   ['2026-07-29', '2026-07-30', '2026-07-31']
  19:xx UTC  x2   ['2026-08-03', '2026-08-05']
  20:xx UTC  x1   ['2026-08-08']

==============================================================================
3. THE STORED NAV AND ITS ASOF (criteria 2 + 3)
==============================================================================

  portfolio_id   : default
  total_nav      : 23833.94
  updated_at     : 2026-08-09T15:04:21.156672+00:00
  AGE OF THE NAV : 10.90 hours
  current_cash   : 22820.64

This `updated_at` is the asof of `total_nav`, and it is the value every
kill-switch path DISCARDS: each reads `portfolio['total_nav']` out of this very
dict and never looks at the timestamp beside it.

==============================================================================
4. HISTORICAL NAV vs THE SOD ANCHOR (criterion 2, multi-day)
==============================================================================

date               sod_nav   snapshot nav   equal?
------------------------------------------------------------
2026-07-27        23838.16       23772.49   NO
2026-07-28        23772.49       23772.49   YES
2026-07-29        23772.49       23772.49   YES
2026-07-30        23772.49       23772.49   YES
2026-07-31        23772.49       23770.98   NO
2026-08-03        23803.94       23803.94   YES
2026-08-05        23830.46       23830.46   YES
2026-08-08        23830.46       23833.94   NO
2026-08-09        23833.94       23833.94   YES
2026-08-09        23833.94       23833.94   YES

  equality held on 7/10 comparable days (70%)
  -> criterion 2 answer: the equality holds SOMETIMES, not a one-off

==============================================================================
5. THE DELTA vs THE COCKPIT (criterion 3)
==============================================================================

  kill-switch current_nav        : 23833.94
  /performance nav               : 23833.94
  /portfolio portfolio.total_nav : 23833.94

  MAX SPREAD: 0.000000

  The $0.06 delta the step describes does NOT reproduce: all three read the
  SAME stored `total_nav`, so they agree exactly by construction. A delta can
  only appear when one of them is served either side of a `mark_to_market`
  write -- i.e. it is a RACE against the cycle, not a rounding or FX
  difference. That also explains why it was ~$0.06 rather than a round number:
  it is one mark's worth of price movement on a single position.

  sod_nav 23833.94 == current_nav 23833.94: True
  daily_baseline_stale : True
  armed                : False
```

## E. Criterion 4 -- the intraday-drawdown test, verbatim
```
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py::test_evaluate_breach_NEVER_marks_to_market PASSED [  8%]
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py::test_every_producer_reads_the_STORED_total_nav_not_a_live_mark PASSED [ 16%]
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py::test_the_daily_loss_leg_DOES_fire_on_a_same_day_drawdown[0.0-False] PASSED [ 25%]
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py::test_the_daily_loss_leg_DOES_fire_on_a_same_day_drawdown[3.9-False] PASSED [ 33%]
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py::test_the_daily_loss_leg_DOES_fire_on_a_same_day_drawdown[4.01-True] PASSED [ 41%]
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py::test_the_daily_loss_leg_DOES_fire_on_a_same_day_drawdown[4.1-True] PASSED [ 50%]
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py::test_the_daily_loss_leg_DOES_fire_on_a_same_day_drawdown[12.0-True] PASSED [ 58%]
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py::test_the_leg_CANNOT_fire_while_the_anchor_is_from_a_PREVIOUS_day PASSED [ 66%]
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py::test_the_production_ENFORCEMENT_path_reads_the_stored_nav PASSED [ 75%]
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py::test_evaluate_breach_checks_BASELINE_staleness_but_not_NAV_staleness PASSED [ 83%]
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py::test_the_asof_IS_available_and_is_discarded_by_every_caller PASSED [ 91%]
backend/tests/test_phase_86_12_kill_switch_nav_freshness.py::test_AT_the_limit_the_displayed_pct_and_the_DECISION_disagree PASSED [100%]
============================== 12 passed in 1.62s ==============================
```

## F. Criterion 6 -- the live journal is untouched
```
sha256 BEFORE this capture: ea78508bee73887c82df2346da408c7281e7e9229334a6131d7fa06c09977065
sha256 AFTER  this capture: ea78508bee73887c82df2346da408c7281e7e9229334a6131d7fa06c09977065
byte-identical: YES

git diff on the kill switch itself (must be empty -- no threshold touched):
(empty above = unchanged)
```
