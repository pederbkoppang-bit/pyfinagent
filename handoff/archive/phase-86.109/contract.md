# Contract -- step 86.109

**Step:** 86.109 -- the data-freshness alarm has no weekday/trading-day
awareness, so every Friday-close-to-Monday-morning gap is misclassified `red`
and pages Slack while the system is healthy. **P1.**

## Research-gate summary (what the gate CHANGED about the plan)

Gate **PASSED** on the reconciliation run (`wf_8a25910d-384`; 40 sources read in
full, 134 URLs against 135 distinct in the brief, audit-class dry after 23
rounds; brief `research_brief_86.109.md`, 112,187 chars). The first run
(`wf_f8489214-adf`) was enforced **FAILED** on a one-URL over-claim; the
reconciliation found the claim double-counted two URLs behind parenthetical
suffixes and lowered it rather than inventing one. Recorded because a gate that
failed and was then honestly reconciled is stronger evidence than one that
passed first time.

**Five findings change the plan, and the first two change what the fix IS.**

**1. The filed premise "no weekday awareness" is right, but weekend
suppression cannot fix it. MEASURED:** 1,149 `Data freshness critical` log
lines, **991 delivered to Slack**, ~17.5/day. Of the pages, **30.0% land
Sat/Sun and 38.4% land on MONDAY** -- Monday is the *largest single bucket*
(165 of 430). A Sat/Sun mute does not touch it. The filing's "58 since
2026-08-15" has moved by ~17x; this is a live, growing alarm storm.

**2. THE CENTRAL FINDING, and it cuts against criterion 2's literal wording:
the calendar belongs on NOTIFICATION, never on DETECTION.** Three independent
vendors agree verbatim -- Grafana ("suppresses notifications but does not
interrupt alert evaluation"), PagerDuty ("incidents are created regardless ...
but *how responders are notified* varies"), Alertmanager (a muted route "will
not send any notifications" yet "otherwise acts normally"). Putting a
trading-day term inside `_band()` would make a Friday-dead writer
indistinguishable from an idle weekend -- the AWS "no dogs barking"
anti-pattern -- and would directly contradict **criterion 3**, which requires a
genuinely stale weekday source to still classify red.

**3. An HTTP GET must not page, and that is the cheapest and largest win.**
Three HTTP call sites reach `_fire_freshness_alarm`
(`paper_trading.py:498`, `observability_api.py:36` and `:55`) and **none is
gated**; `freshness_cron.py:146-151` is the only gated caller.
`CycleHealthStrip.tsx:105` polls every 30s, so an open dashboard tab is the
page trigger, with `AlertDeduper`'s 1-hour P1 window the only brake. RFC 9110
§9.2.1 (safe methods are "essentially read-only") and Azure's health-endpoint
guidance ("you don't want every request to the dashboard to trigger a health
check") both name this. **This fix needs zero calendar work.**

**4. Criterion 2's premise IS TRUE and my own pre-gate grep was WRONG.** I
reported one production consumer of `is_trading_day`; the gate found the 51.3
digest application at `backend/slack_bot/scheduler.py:365-375`
(`_is_us_trading_day_now`, used at `:565`/`:610`), hidden behind a wrapper name
plus a function-local import -- `is_trading_day` spans 7 files, not 2. Recorded
because I raised a false alarm and the gate corrected it.

**5. `cycle_health.py` already holds three calendar notions, and
`is_weekday_et` (`:262`) is holiday-BLIND.** A trading-day term added to
`_band()` would be a drifting *fourth*. Criterion 2's "not a second,
independently-drifting definition" is therefore an argument for touching
`_band()` **less**, not more.

**One measured correction to the literature's own carve-out:** the "never
suppress CRITICAL" rule is inapplicable here because `freshness_cron.py:162-166`
already emits `severity="P1"` for this alarm, so noise and a dead writer share
a severity. And the strongest cited objection to a weekend mute is that **it
never expires**, not that it hides failures.

## Hypothesis

The alarm storm has two independent causes and only one of them is the
calendar. The dominant, cheapest cause is that three ungated HTTP read paths
page on a 30s dashboard poll. The second is that a pure age/interval ratio is
the wrong question across a non-trading gap. Fixing the read paths removes the
storm's mechanism; gating the *notification* leg on the existing trading-day
mechanism removes the false urgency, while detection stays calendar-blind so a
Friday-dead writer is still caught.

## Immutable success criteria (copied verbatim from `.claude/masterplan.json`)

1. the false positive is REPRODUCED by execution, not asserted: call the CURRENT _band()/compute_freshness with synthetic ages representing a genuine Friday-close-to-Monday-morning gap on an otherwise-healthy weekday cadence (e.g. last write ~64h ago, interval 86400s), and show the unmodified code returns band='red' purely from elapsed calendar time -- not the 3.79/3.32/2.36/2.78 ratios copied from this filing, which will have moved by the time this step runs
2. the fix reuses the SAME trading-day mechanism step 51.3 already applied to the digest jobs (backend/backtest/markets.py::is_trading_day or equivalent), applied to _band()/compute_freshness/_fire_freshness_alarm -- not a second, independently-drifting definition of 'trading day'
3. a CONTROL proves the fix is not a blanket suppression: a genuinely stale WEEKDAY source (writer job broken mid-week, e.g. a Tuesday gap exceeding the interval with no trading-day boundary crossed) still classifies red after the fix, demonstrated alongside the weekend-false-positive cell
4. the un-gated HTTP call sites are addressed and the choice is STATED: either GET /api/paper-trading/freshness (+ its two aliases in observability_api.py) adopt freshness_cron.py's existing state-transition (newly-red-only) gate before calling _fire_freshness_alarm, or an equivalent mechanism is added so a 30s dashboard poll cannot re-trigger a Slack page outside the 1-hour AlertDeduper window -- state which was chosen and why, not both left unexamined
5. the regression guard is mutation-tested with the control observed GREEN FIRST: removing the trading-day-awareness must turn the guard RED, reproduced by actually removing it, not asserted
6. verdict semantics and other steps' status are UNCHANGED: nothing here may flip an unrelated step or alter a prior verdict

**Immutable verification command:**
`bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/services/cycle_health.py\").read())" && echo parses'`

**Immutable live_check:** `live_check_86.109.md` carrying the reproduced false
positive, the post-fix classification, the weekday control, the mutation cell,
and the chosen HTTP-route fix with before/after evidence.

## Plan

**P1 -- criterion 1, reproduced by execution.** Call the CURRENT unmodified
`_band()` / `compute_freshness` with synthetic ages for a Friday-15:00 write
read on Monday-08:00 (~64h, interval 86400) and show `red`. Use synthetic
inputs, **not** the filing's ratios -- they have already moved.

**P2 -- criterion 4 FIRST, because it is the storm's mechanism and needs no
calendar.** Make the three HTTP read paths stop paging: they compute and return
freshness exactly as today but do not call `_fire_freshness_alarm`. State the
choice and why (RFC 9110 + Azure: a dashboard poll must not be a page trigger;
this is stronger than adopting the cron's state-transition gate at three more
sites, because it removes the side effect rather than rate-limiting it). The
cron keeps its existing gate and remains the only notifier.

**P3 -- criterion 2, on the NOTIFICATION leg, with the deviation declared.**
Gate the cron's `notify()` on the existing trading-day mechanism, reusing
`markets.is_trading_day` through the same path the digest jobs use -- no new
definition. **DECLARED DEVIATION for the evaluator to judge:** criterion 2
names `_band()/compute_freshness/_fire_freshness_alarm`. The fix lands at
`_fire_freshness_alarm` (one of the three) and NOT inside `_band()`, because
(a) three vendors and the AWS anti-pattern agree the calendar belongs on
notification not detection, (b) criterion 3 *requires* a stale weekday source
to still classify red, which a calendar-aware `_band()` puts at risk, and
(c) `cycle_health.py` already carries three calendar notions and a fourth would
be exactly the "independently-drifting definition" criterion 2 forbids.
Preserve `is_trading_day`'s **fail-open** polarity: a missing calendar library
must never suppress a page.

**P4 -- criterion 3's control, alongside the weekend cell.** A Tuesday->Wednesday
gap exceeding the interval, no trading-day boundary crossed, must still be red
AND still notify after the fix. This is the cell that proves the change is not a
blanket mute.

**P5 -- the Monday problem, stated not silently dropped.** 38.4% of pages land
on Monday, which neither P2 nor P3 fully addresses: Monday IS a trading day, so
a trading-day gate does not mute it, and the ratio is legitimately >2.0 until
the Monday cycle runs. Report what the fix does and does not remove, with the
Monday bucket measured before and after. If the residue needs an
expected-next-write primitive (the brief's finding 4 -- `_CYCLE_COMPLETED_STALE_SEC`
already exists for the cycle clock but not for table bands), that is **filed as
its own step, not smuggled in here.**

**P6 -- criterion 5 mutations** with the control observed GREEN first, a
byte-identical restore, each cell scored, UNSCORABLE if its control was not
green. Cells must include: remove the trading-day gate (must go red) and
re-enable alarm emission on an HTTP path (must go red).

## Scope honesty -- what this step does NOT do

- **It does not make `_band()` calendar-aware.** See P3's declared deviation.
- **It does not silence anything unconditionally.** Detection is unchanged;
  only the notification leg and the read-path side effect change.
- **It does not build the expected-next-write primitive.** Filed if the Monday
  residue warrants it.
- **It promotes no flag and writes no `.env`.**
- **It changes no verdict semantics and flips no other step** (criterion 6).

## References

`research_brief_86.109.md` (findings C1.1-C1.10, the A8 measurement, the A4/A7
call-site enumeration, the A5 existing gate, the criterion-2 premise check);
`backend/services/cycle_health.py`, `backend/services/observability/alerting.py`,
`backend/slack_bot/scheduler.py` (the 51.3 precedent).
