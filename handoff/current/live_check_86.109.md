# live_check -- step 86.109 (2026-08-17; exits unpiped)

## 1. Immutable verification command

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/services/cycle_health.py\").read())" && echo parses'
parses
EXIT=0
```

## 2. Criterion 1 -- the false positive, REPRODUCED BY EXECUTION

Synthetic ages, not the filing's ratios (which have moved). `_band` is
**unmodified by this step** -- printed from the live module so that is checked,
not claimed:

```
$ python -c '<print inspect.getsource(_band), then call it>'
def _band(age_sec, interval_sec) -> str:
    if age_sec is None or interval_sec <= 0: return "unknown"
    ratio = age_sec / interval_sec
    if ratio >= CRITICAL_RATIO: return "red"
    if ratio >= WARN_RATIO:     return "amber"
    return "green"

  Fri 15:00 -> Mon 08:00 (healthy weekend)   age= 65h ratio=2.71 band=red
  Fri 15:00 -> Sun 15:00 (healthy weekend)   age= 48h ratio=2.00 band=red
  normal weekday cadence                     age= 20h ratio=0.83 band=green
  Tue -> Thu, writer DEAD mid-week           age= 50h ratio=2.08 band=red

CRITICAL_RATIO=2.0  WARN_RATIO=1.5
```

**The red is arithmetically correct and the question is wrong.** A 24h-interval
source is at ratio 2.0 by Sunday afternoon on a perfectly healthy Friday close.
Note the last row: a genuinely dead mid-week writer produces the *same* band
from the *same* function -- which is exactly why the calendar must not go here
(criterion 3).

## 3. What the alarm is actually doing, MEASURED

```
$ <count "Data freshness critical" across 7 rotated .gz + live backend.log>
total lines: 1149
live backend.log: 30,004,359 bytes, mtime 2026-08-17T21:55:29Z, read 21:55:31Z
```

1,149 reproduces the research gate's figure exactly. The filing's "58 since
2026-08-15" was an early reading of a still-growing storm.

**Attribution, with a bound rather than a split:**

```
rotated ON OR BEFORE 2026-08-04 (the cron did not exist yet):  867
rotated after, up to 2026-08-14:                               204
live backend.log (post 2026-08-14):                             78
TOTAL:                                                        1149

>= 867 of 1149 (75.5%) predate the cron entirely (added 2026-08-05,
commit b7c69bb9) and are therefore NECESSARILY from the level-triggered
HTTP read path.
```

**The remaining 282 cannot be attributed, and I am not going to pretend they
can.** My first attempt at this was a probe that could not have failed:
`grep detected_by` over the corpus returned **0 of 1,149**, which looks like
"no cron pages at all" and proves nothing -- `alerting.py` logs `source=` and
`title=` but **not** `details`, so `detected_by` never reaches a log line, and
both paths emit `source="cycle_health"` with an identical title. The date bound
above is the part that is actually derivable.

## 4. Criterion 4 -- the un-gated HTTP call sites, and the choice

**Chosen: the three read paths stop emitting the alarm entirely
(`emit_alarm=False`). NOT: extend the cron's state-transition gate to them.**

Why this and not that:

- **A read path should not have the side effect at all.** RFC 9110 §9.2.1 --
  safe methods are "essentially read-only". Azure's health-endpoint guidance
  states the remedy verbatim: *"if the health status is reported through a
  dashboard, you don't want every request to the dashboard to trigger a health
  check. Instead, periodically check the system health, and cache the status."*
  A transition gate would rate-limit the side effect; removing it removes the
  mechanism.
- **The storm is a poll, not a state change.** `CycleHealthStrip.tsx:105` polls
  every 30s, so an open tab is the trigger. `freshness_cron.py`'s own module
  docstring already records the phase-66 precedent: *"~120 pages/hour the
  moment a dashboard tab was open against a red table."*
- **Something else already owns notification.** `freshness_cron.py` has the
  state-transition gate, passes `emit_alarm=False` itself, and is now the sole
  notifier. Two notifiers with two gates is how they drift.

**Detection is untouched.** The three handlers compute and return the identical
payload; only the side effect is removed.

Proven by DRIVING the three real handlers, not by grepping:

```
$ pytest -q backend/tests/test_phase_86_109_freshness_calendar.py backend/tests/test_phase_82_10_freshness_paging.py
33 passed

test_paper_trading_freshness_route_does_not_page        -> emit_alarm forwarded as False
test_observability_freshness_aliases_do_not_page[...]   -> both aliases, False
test_compute_freshness_still_pages_when_asked           -> 1 real alarm when asked, 0 when not
```

That last one is the anti-vacuity control -- **and an earlier version of it was
not one.** It never called `compute_freshness` at all: it built a `_BQ` class it
did not use, saved and restored `_fire_freshness_alarm` without ever triggering
it, and its only live assertion duplicated a phase-82.10 test. A Q/A made the
alarm branch inert (`if False and emit_alarm ...`) and it SURVIVED all 17 tests,
so this file's claim that it guarded the alarm path was false.

It now forces an all-red payload through the REAL `compute_freshness` and counts
REAL `_fire_freshness_alarm` invocations: **1** with `emit_alarm=True`, **0**
with `emit_alarm=False`, and the returned payload identical in both -- which is
also the proof that only the side effect was removed.

### A prior step's test forbade this change, and it caught me

`test_phase_82_10_freshness_paging.py::test_http_call_sites_were_not_edited_to_pass_emit_alarm`
asserted `"emit_alarm" not in src` for both API modules. The suite went red the
moment this fix landed -- **the guard worked.**

Reading it decided the disposition rather than overriding it. Its docstring was
*"Scope honesty: this step must not change dashboard behaviour."* That is
phase-82.10 pinning **its own** scope -- it added the cron and recorded that it
had not touched the dashboard. It was never a standing policy that a dashboard
GET should page, and step 86.109's criterion 4 authorises exactly these call
sites.

So the assertion is **inverted in place, not deleted**, with the supersession
written at the site of the original claim, and it now points at this step's
behavioural drivers as the real guard (a source scan is not one).

**And the inverted form was itself vacuous on its first draft.** `"emit_alarm=False"
in src` is satisfied by the explanatory comment the SAME diff added: a Q/A
mutated every real call site to `emit_alarm=True` and the assertion still passed
for both files. It now strips comment lines before asserting, and that fix is
proven rather than claimed -- mutating every non-comment occurrence to `True`
turns it RED (`1 failed, rc=1`), with a byte-identical restore.

## 5. Criterion 2 -- ONE trading-day definition, and a declared deviation

**My own pre-gate grep was WRONG and the research gate corrected it.** I
reported that `is_trading_day` had a single production consumer and that
criterion 2's premise might be false. The gate found the phase-51.3 digest
application at `backend/slack_bot/scheduler.py:365-375`
(`_is_us_trading_day_now`, used at `:565`/`:610`), hidden behind a wrapper name
plus a function-local import. **The premise is TRUE.** Recorded because I
raised a false alarm.

The body of that wrapper is now `backend/backtest/markets.py::is_us_trading_day_now`,
and the digest wrapper **delegates** to it. One definition, two consumers,
digest behaviour unchanged. `test_trading_day_helper_is_shared_not_duplicated`
pins the delegation by substituting the shared helper and asserting the digest
wrapper calls it; mutation cell **N8** re-grows a parallel definition and is
KILLED.

**DECLARED DEVIATION, for the evaluator to judge -- and it is LARGER than a
prior revision of this sentence admitted.** Criterion 2 says the fix is
"applied to `_band()`/`compute_freshness`/`_fire_freshness_alarm`". This file
previously said it "lands at the notifier -- one of the three named". **That
was false, and false in the direction that made the deviation look smaller.**
Measured:

```
$ git diff --stat HEAD -- backend/services/cycle_health.py
(empty)
```

`cycle_health.py` is **unmodified**, so `_band()`, `compute_freshness()` and
`_fire_freshness_alarm()` are **all three** untouched. The calendar gate lands
in `backend/services/freshness_cron.py::run_freshness_check` -- a **FOURTH
site**, none of the three named, and the functional successor of
`_fire_freshness_alarm` for the cron path. The reasons the fix belongs there
rather than in any of the three:

1. Three independent implementations agree the calendar belongs on the routing
   leg. Grafana: a mute timing *"suppresses notifications but does not
   interrupt alert evaluation."* PagerDuty: *"incidents are created regardless
   ... but how responders are notified varies."* Alertmanager: a muted route
   *"will not send any notifications"* yet *"otherwise acts normally."*
2. **Criterion 3 requires** a genuinely stale weekday source to still classify
   red. A calendar-aware `_band()` puts that at risk and makes a Friday-dead
   writer indistinguishable from an idle weekend -- AWS's "no dogs barking"
   anti-pattern.
3. `cycle_health.py` already carries three calendar notions, one of which
   (`is_weekday_et`, `:262`) is holiday-**blind**. A fourth inside `_band()`
   would be precisely the "independently-drifting definition" criterion 2
   forbids. The criterion is an argument for touching `_band()` **less**.

## 6. Criterion 3 -- the control, alongside the weekend cell

Both driven through the real `run_freshness_check` using its own injectable
`bq`/`settings`/`notify`:

| Cell | Result |
|---|---|
| newly red, **non**-trading day | `pages == []` -- withheld |
| newly red, **trading** day | `pages == ["freshness_critical_paper_trades"]` -- **still pages** |
| steady-state red on a weekday, three ticks | pages **once** -- the transition gate still works |
| calendar library raises | **still pages** -- fail-open polarity preserved |

The second row is criterion 3: the change is not a blanket suppression.

### The deferral bug this step wrote and then caught

The first draft committed `_last_red_sources = red_now` **before** the gate.
That absorbs a weekend red into "already known", so on Monday `newly_red` is
empty and **the source never pages at all** -- a weekend mute that silently
becomes permanent. The comment even claimed *"they will page on the next
session"*, which the code did not deliver.

Fixed by holding withheld sources out of the baseline
(`_last_red_sources = red_now - newly_red`), and pinned two ways:
`test_weekend_suppression_is_DEFERRED_not_dropped` (withheld Saturday source
pages on Monday) and `test_a_source_that_recovers_over_the_weekend_never_pages`
(the converse -- deferral must not resurrect a page for something that healed).
Mutation cell **N3** restores the bug and is KILLED.

## 7. Criterion 5 -- mutation matrix, control GREEN first

```
$ python scripts/qa/mutation_86_109.py
CONTROL rc=0  collected=17

N1 KILLED   removing the trading-day gate is caught
N2 KILLED   an INVERTED gate -- muting weekdays, paging weekends -- is caught
N3 KILLED   absorbing a withheld source into the baseline -- a permanent silent mute
N4 KILLED   a read path that pages again is caught
N5 KILLED   an observability alias that pages again is caught
N6 KILLED   a calendar failure that SUPPRESSES a page (inverted polarity) is caught
N7 KILLED   a helper that ignores the calendar and refuses is caught
N8 KILLED   re-growing a parallel trading-day definition is caught

N9  KILLED   the calendar reaching _band -- invisible Mon-Fri to the old tautology
N10 KILLED   an alarm path made completely inert -- the old "control" could not see it
N11 KILLED   a byte-pin satisfiable by the comment the same diff added

KILLED=11/11  SURVIVORS=none  UNSCORABLE=none
RESTORE VERIFIED: every cell re-hashed to its pre-mutation SHA-256.
```

**N9-N11 exist because a Q/A executed three mutants this matrix did not contain
and all three SURVIVED.** Every one attacked a GUARD rather than the product --
the shipped behaviour was correct in each case, and the assertions meant to
protect it could not fail. N11 also took two attempts here: mutating the guard's
comment-strip only WEAKENS it and leaves the suite green (a non-discriminating
mutant, scored SURVIVED on its first run and recorded as such), so the cell now
mutates the CODE site while leaving the explanatory comment standing -- which is
precisely the shape that defeated the first draft.

The matrix now runs BOTH test files, because a cell whose named test is not even
collected scores UNSCORABLE rather than KILLED.

Scoring rule is the strict one: control green FIRST, pytest exit **1** (exit 5
is not a kill), the mutant must collect the SAME count as the control (a mutant
that cannot build is not a killed mutant), and the **named** test must be the
failing one.

## 8. Lint and regression, over a DERIVED scope

```
$ { git diff --name-only HEAD -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u
backend/api/observability_api.py
backend/api/paper_trading.py
backend/api/sovereign_api.py            <- PEER session, not this step
backend/backtest/markets.py
backend/services/autonomous_loop.py     <- PEER session, not this step
backend/services/freshness_cron.py
backend/slack_bot/scheduler.py
backend/tests/test_phase_82_10_freshness_paging.py
backend/tests/test_phase_86_109_freshness_calendar.py
scripts/qa/mutation_86_109.py
  (10 files. A prior revision of this block said "9" -- a typed count beside a
   derivation, which is the exact thing a derived scope exists to avoid. The
   output is pasted now instead of summarised.)

$ uvx ruff check --select F821,F401,F811 --no-cache --output-format=concise $(cat scope)
backend/backtest/markets.py:9:20: F401 [*] `typing.Optional` imported but unused
Found 1 error.
RUFF_EXIT=1

$ git show HEAD:backend/backtest/markets.py > /tmp/markets_head.py && uvx ruff check ... /tmp/markets_head.py
/tmp/markets_head.py:9:20: F401 [*] `typing.Optional` imported but unused   <- PRE-EXISTING
$ grep -c Optional  HEAD copy: 1   worktree: 1                              <- no use added or removed
```

**PRE-EXISTING, not introduced here**, and a second instance of the class
already filed as step **86.113**. Not fixed here, for the same reason 86.113
exists: a step should not quietly repair unrelated lines under its own name.

```
$ python -m pytest backend/tests/ -q -p no:cacheprovider -k "freshness or cycle_health or observab or paper_trading or scheduler or markets or calendar or 82_10 or 51_3"
218 passed, 7 skipped, 3421 deselected                <- AFTER the 82.10 disposition
  -> BEFORE it, this selection was "1 failed, 217 passed": the failure was
     test_phase_82_10::test_http_call_sites_were_not_edited_to_pass_emit_alarm,
     i.e. this step's own change, correctly caught. See §4.
```

## 9. Criterion 6 -- nothing else changed

No masterplan step was flipped and no prior verdict altered by this step. The
only masterplan write in this session's earlier work was the FILING of new
pending steps (86.112-86.114 by step 86.108), not a status change.

## 10. What this step does NOT fix -- the Monday residue

**38.4% of the DATABLE pages landed on a MONDAY** -- and that percentage has a
denominator that must travel with it. Only **430 of the 1,149** lines carry a
date at all; the other 719 are the older plain-text format (`17:27:19 W
[alerting] ...`), which carries a time but no date and genuinely cannot be
weekdayed:

```
total 1149 | DATABLE 430 (37.4%) | undatable 719
  Mon 165  38.4%      Tue  38  8.8%      Wed  27  6.3%      Thu 35 8.1%
  Fri  36   8.4%      Sat  46 10.7%      Sun  83 19.3%
  Sat+Sun = 129 (30.0% of datable) -- the share the calendar gate addresses
```

Monday *is* a trading day, so the calendar gate does not mute it. Two things are true and both belong here:

- The read-path fix (§4) removes those pages **regardless of weekday**, because
  it removes the poll-driven mechanism rather than filtering by date. Since
  ≥75.5% of the corpus is necessarily read-path, the Monday bucket is expected
  to collapse with it.
- **That expectation is not yet measured.** An "after" rate needs the fix
  running across a weekend, which this session cannot produce. Stated as an
  open measurement, not claimed as a result.

If a residue survives, the brief's finding 4 names the right primitive -- an
**expected-next-write** clock rather than an age ratio, which the repo already
has for the cycle clock (`_CYCLE_COMPLETED_STALE_SEC`) but not for table bands.
That is **filed as its own step if needed, not smuggled in here.**

## 12. Cycle 2 -- closing the first CONDITIONAL

Verdict `wf_7d82fcae-55e` found **all 6 criteria MET and ZERO product defects**,
and judged both declared deviations SOUND on the merits. It capped the verdict
on six evidence-side findings. All six are closed, and three of them were guards
I had described as strong that it falsified by execution:

1. **`test_band_has_no_day_of_week_term_after_the_fix` was a TAUTOLOGY.**
   `first = _band(...); second = _band(...)` in the same instant is true of
   every implementation, including a calendar-aware one. A `_band` that returns
   "green" on weekends SURVIVED all 17 tests -- on a Monday, invisibly. Replaced
   by two guards that fire on any day: a structural scan of `_band`'s own source
   for any calendar reference, and a behavioural check that freezes the clock to
   each of the seven weekdays and asserts the answer never moves. Cell **N9**.
2. **The stated anti-vacuity control never called `compute_freshness`.** Its
   `_BQ` class and its `_fire_freshness_alarm` save/restore were dead code, and
   an inert alarm branch SURVIVED. Rewritten to force an all-red payload through
   the real function and count real alarm invocations: 1 when asked, 0 when not,
   identical payload both times. Cell **N10**.
3. **The inverted 82.10 byte-pin was satisfied by the comment my own diff
   added.** With every real call site mutated to `emit_alarm=True` it still
   passed. Now strips comment lines first, and the fix is proven: mutating every
   non-comment occurrence turns it RED. Cell **N11**.
4. **The deviation was described as landing at "one of the three named"
   functions.** `git diff HEAD -- backend/services/cycle_health.py` is EMPTY, so
   all three are untouched and the gate lands at a FOURTH site. Corrected in §5,
   including that the error made the deviation look smaller than it is.
5. **"9 files" in §8 did not reproduce** -- the derivation yields 10. The actual
   output is pasted now instead of a typed count.
6. **The 38.4%-Monday figure had an undisclosed denominator.** It is 430 datable
   lines, not the 1,149 the section is about; 719 lines are the older plain-text
   format carrying a time but no date. Full weekday table now in §10.

An expectation of mine was also wrong and the code corrected it: the new
seven-day band check first expected 30h to be `amber`, but 30/24 = 1.25 is below
`WARN_RATIO` 1.5 and is green. The age was changed to 40h so all three bands are
covered.

## 11. NOT YET IN FORCE -- pending restart

The running backend is **pid 41635**, started 2026-08-17T13:57:16Z, before
these edits. It still holds the old handlers, so the three read paths on the
LIVE process continue to page until the restart. Per the batched-restart rule
the restart is deferred to session end, together with step 86.108's two new
routes.

No flag promoted, no `.env` written.
