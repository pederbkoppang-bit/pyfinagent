# experiment_results -- step 86.109

**GENERATE complete for all six criteria.** Verbatim evidence lives in
`handoff/current/live_check_86.109.md`; this file is the build record.

## What was built

### New files

| File | Purpose |
|---|---|
| `backend/tests/test_phase_86_109_freshness_calendar.py` | 17 tests. Drives `_band`, the three real HTTP handlers, and the real `run_freshness_check` through its own injectable `bq`/`settings`/`notify`. |
| `scripts/qa/mutation_86_109.py` | 11-cell mutation matrix, run over BOTH test files, strict scoring (control green first; exit 5 is not a kill; same collect count; the NAMED test must fail). |

### Modified files

| File | Change |
|---|---|
| `backend/backtest/markets.py` | **New `is_us_trading_day_now()`** -- the single ET-today trading-day definition, extracted from the phase-51.3 digest wrapper so a second consumer could reuse it rather than grow a parallel copy. Fail-open preserved. |
| `backend/slack_bot/scheduler.py` | `_is_us_trading_day_now` now **delegates** to the shared helper. Behaviour unchanged (same ET "today", same polarity); the wrapper is kept because the digest call sites name it. |
| `backend/services/freshness_cron.py` | The notifier is gated on the trading day, and withheld sources are **DEFERRED** (held out of the baseline) rather than absorbed. Detection and the state machine are untouched. Loud WARNING on every withheld page. |
| `backend/api/paper_trading.py` | `/freshness` passes `emit_alarm=False` -- a dashboard GET must not page. |
| `backend/api/observability_api.py` | Both aliases likewise. |
| `backend/tests/test_phase_82_10_freshness_paging.py` | A phase-82.10 **scope guard** asserted `"emit_alarm" not in src` for both API modules and went red on this change. Read rather than overridden: its docstring is *"this step must not change dashboard behaviour"*, i.e. 82.10 pinning its OWN scope, not a standing policy. Inverted in place with the supersession recorded at the site of the original claim. |

## Verbatim verification output

```
$ bash -c 'source .venv/bin/activate && python -c "import ast; ast.parse(open(\"backend/services/cycle_health.py\").read())" && echo parses'
parses
EXIT=0

$ pytest backend/tests/test_phase_86_109_freshness_calendar.py -q
17 passed

$ pytest backend/tests/test_phase_82_10_freshness_paging.py backend/tests/test_phase_86_109_freshness_calendar.py -q
33 passed

$ python scripts/qa/mutation_86_109.py
CONTROL rc=0  collected=33
KILLED=11/11  SURVIVORS=none  UNSCORABLE=none
RESTORE VERIFIED: every cell re-hashed to its pre-mutation SHA-256.

$ uvx ruff check --select F821,F401,F811 --no-cache --output-format=concise <10-file DERIVED scope>
backend/backtest/markets.py:9:20: F401 [*] `typing.Optional` imported but unused
Found 1 error.
RUFF_EXIT=1
  -> PRE-EXISTING (reproduces on the HEAD copy; Optional count is 1 in both).
     Second instance of the class already filed as step 86.113. Not fixed here.

$ pytest backend/tests/ -q -p no:cacheprovider -k "freshness or cycle_health or observab or paper_trading or scheduler or markets or calendar or 82_10 or 51_3"
218 passed, 7 skipped, 3421 deselected               <- AFTER the 82.10 disposition
  -> BEFORE it: "1 failed, 217 passed" -- the failure WAS this step's own
     change, caught by 82.10's scope guard.
```

## The measurement, and its honest limit

```
"Data freshness critical" lines, 7 rotated .gz + live backend.log:  1149
  rotated ON OR BEFORE 2026-08-04 (the cron did not exist yet):      867
  rotated after, to 2026-08-14:                                      204
  live backend.log:                                                   78
```

**≥ 867 of 1149 (75.5%) necessarily came from the HTTP read path**, because
they predate `freshness_cron.py` entirely (added 2026-08-05, `b7c69bb9`).

The remaining 282 **cannot** be attributed. My first probe for this was one
that could not have failed: `grep detected_by` returned 0 of 1,149, which reads
like "no cron pages" and proves nothing -- `alerting.py` logs `source=` and
`title=` but not `details`, so `detected_by` never reaches a log line, and both
paths emit `source="cycle_health"` with an identical title.

## The three findings that shaped the work

1. **The calendar belongs on NOTIFICATION, never on DETECTION**, per three
   independent implementations (Grafana / PagerDuty / Alertmanager) and AWS's
   "no dogs barking" anti-pattern. Criterion 2 names `_band()` as a possible
   site; putting it there would break criterion 3 and would add a fourth
   calendar notion to a module that already has three, one of them
   holiday-blind. Declared as a deviation.
2. **A prior step's test forbade the criterion-4 fix, and it was right to
   fire.** The disposition came from reading its docstring, which showed it was
   82.10's own scope pin rather than a policy. Inverted in place, not deleted.
3. **The first draft of the gate had a real bug**: it committed the baseline
   before the gate, absorbing a weekend red into "already known" so it would
   never page -- a weekend mute that silently becomes permanent, while the
   comment claimed the opposite. Caught by writing the deferral test before the
   matrix. Cell N3 restores it and is KILLED.

## Scope honesty -- what this step does NOT do

- **It does not make `_band()` calendar-aware** (declared deviation, §5 of the
  live_check).
- **It does not measure an "after" rate.** That needs the fix running across a
  weekend. The Monday residue (38.4% of pages) is expected to collapse with the
  read-path fix because that fix is weekday-independent, but **expected is not
  measured** and is stated as an open measurement.
- **It does not build the expected-next-write primitive.** Filed if a residue
  survives.
- **It promotes no flag and writes no `.env`.**
- **It flips no step and alters no prior verdict** (criterion 6).

## NOT YET IN FORCE

Backend **pid 41635** started 2026-08-17T13:57:16Z, before these edits, so the
three read paths on the LIVE process still page until the restart. Batched to
session end together with step 86.108's two new routes.

## Cycle 2 -- response to the CONDITIONAL (`wf_7d82fcae-55e`)

Verdict: all 6 criteria MET, **zero product defects**, both declared deviations
judged SOUND. Six evidence-side findings, all closed. Three were guards this
step had described as strong and the evaluator falsified by execution: a
tautological `_band` check that a calendar-aware mutant survived; an
"anti-vacuity control" that never called `compute_freshness`; and an inverted
byte-pin satisfied by the comment the same diff introduced. Each is now
replaced by something that fires, with cells **N9**, **N10** and **N11**
reproducing the exact surviving mutants. Matrix is **11/11** and now runs both
test files, because a cell whose named test is not collected scores UNSCORABLE
rather than KILLED.

The other three were disclosure errors, all corrected in the live_check: the
deviation lands at a FOURTH site (`cycle_health.py` is byte-unmodified), the
derived scope is 10 files not 9, and the 38.4%-Monday figure has a denominator
of 430 datable lines rather than the full 1,149.
