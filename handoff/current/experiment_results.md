# experiment_results -- step 82.15

**GENERATE phase.** Contract: `handoff/current/contract.md`.
Research: `handoff/current/research_brief_82.15.md` (gate_passed=true, 7 sources
read in full, 24+ URLs).

## What was built

The `realtime_start` column 82.0 added had **zero consumers**. Four sites in
`backend/backtest/cache.py` now implement a point-in-time read, gated by
`macro_point_in_time_enabled` (default **True**):

| # | site | change |
|---|---|---|
| 1 | preload SELECT | now fetches `realtime_start` (it did not, so the fast path could not filter on it) |
| 2 | `_macro_full` row build | carries `realtime_start` into the cache |
| 3 | fast-path lookup | no longer `break`s on a date-only match |
| 4 | BQ fallback WHERE | vintage predicate added, with `DATE(@cutoff)` |

Plus `MACRO_PUBLICATION_LAG_DAYS` (per-series) and `_effective_vintage()`.

## The design decision, and why the obvious fix is wrong

A strict `realtime_start <= cutoff` returns **0 of 4729 rows** at cutoffs
2020-01-01, 2023-01-01 AND 2025-06-01 -- because 82.0 backfilled our WRITE date
(2026-03-22..25), which says nothing about 2019. That would blank all six macro
features across the entire 2018-2025 backtest window, and do it SILENTLY:
`historical_data.py` guards on `if macro:`, so the keys are never set rather
than set to None.

So availability is derived as:

```
effective_vintage = MIN(realtime_start, obs_date + MACRO_PUBLICATION_LAG_DAYS[series])
```

MIN never blanks the sample. Where `realtime_start` is a TRUE vintage it governs and the row is correctly visible from its real publication date. Where the stamp is an 82.0 backfill artifact (our write date) the rule degrades to the lag estimate, which is conservative ONLY IF that lag upper-bounds the real release delay -- where it underestimates a delayed release the row is admitted early (see LIMITS). For rows
ingested from now on the true stamp governs; for historical rows the per-series
lag supplies a defensible estimate instead of a meaningless write date. That is
also the documented treatment of the NULL population (criterion 3).

## Verification command output (verbatim)

```
$ source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_15_macro_point_in_time.py -q
.............                                                            [100%]
15 passed in 0.83s
```

## MEASURED effect on live data -- 23 of 28 lookups changed

Observation date returned at each cutoff, filter OFF vs ON:

```
cutoff      series      OFF (look-ahead)   ON (known then)   delta
2019-06-28  GDP               2019-04-01        2019-01-01     90d
2019-06-28  CPIAUCSL          2019-06-01        2019-05-01     31d
2019-06-28  FEDFUNDS/UMCSENT/UNRATE   (same 31d shift)
2019-06-28  DGS10/T10Y2Y      2019-06-28        2019-06-27      1d
2023-01-03  GDP               2023-01-01        2022-07-01    184d
2023-01-03  CPIAUCSL/FEDFUNDS/UMCSENT/UNRATE  (same 61d shift)
2025-06-02  GDP               2025-04-01        2025-01-01     90d
2026-07-31  GDP               2026-04-01        2026-01-01     90d
```

**23 of 28 series x cutoff combinations changed.** Each is a value the backtest
previously read before it existed. The worst case is GDP at cutoff 2023-01-03,
which was reading an observation not published for another **184 days**.

And the sample SURVIVES: all 7 series still resolve at 2018-06-01, 2021-03-15,
2024-09-30 and 2026-07-31 (`preload_macro` -> 4729).

## LIMITS -- do not overstate this

1. **This fixes PUBLICATION LAG only, NOT REVISIONS.** Ingest dedupes on
   `(series_id, date)`, so a revised value can never sit beside its original;
   revisions are structurally uncapturable without an ALFRED backfill.
   **82.15 must not be reported as "look-ahead fixed".** Queued as 82.18.
2. **Sharpe will fall** on macro-conditioned strategies. Sources bracket
   100-500bp of return / 15-25% of Sharpe for GDP-dependent strategies. That is
   the point: the previous figures were partly reading the future.
3. **Comparability.** No figure produced before this step is comparable with one
   after. 82.3 has NOT run, so nothing existing breaks -- but ALL of 82.3
   (incumbent AND candidates) must run in ONE flag state.
4. The per-series lags are conservative estimates from release-calendar
   practice, not per-observation truth. Only an ALFRED backfill gives truth.

## Flag decision (Main's call, disclosed)

`macro_point_in_time_enabled` defaults **True**, against this project's
default-OFF convention. That convention is kept for the MONEY path; this is the
research lane, and defaulting OFF would mean knowingly shipping look-ahead into
the very evidence 82.3 exists to produce. The flag makes the effect measurable
ON vs OFF and revertible without a code change.

## Scope honesty

- No live-funnel change: `backend/services`, `backend/tools`, `backend/agents`
  untouched. The live regime path reads FRED directly, not this table.
- Regression, MEASURED this cycle rather than carried forward: 82.0 = 16, 82.2 = 22,
  82.15 = 13 -> **51 green** (`51 passed in 9.10s`). An earlier line here said 49,
  which was 16+22+11 -- the cycle-1 size of this very suite, invalidated by the two
  tests cycle 2 added. A count that this cycle's own change falsifies must be
  re-derived, not carried.
- `sortino.py:108` was deliberately NOT touched: it queries
  `pyfinagent_data.historical_macro`, which 404s (Main hit this independently at
  the start of the session). It is dead code and belongs to 82.8 -- "fixing" its
  vintage would imply it runs.


## CYCLE 2 -- disposition of the cycle-1 CONDITIONAL

Cycle-1 verdict: **CONDITIONAL**. Critically, the Q/A confirmed **the production
code is CORRECT on both paths** -- it dry-ran the generated SQL against live
BigQuery (valid, 168,506 bytes ON) and got agreement with the fast path on all
7 series at 2023-01-03, and it independently reproduced every headline number
(23/28 changed, GDP@2023-01-03 184d, 7/7 series resolve, 0/4729 under a strict
filter). The block was the GUARDS, not the behaviour. All three findings fixed.

**F1 [BLOCKING] -- criterion 2's fallback half was an illusory guard. FIXED.**
Its sole coverage was two substring scans (`"realtime_start" in sql`,
`"DATE(@cutoff)" in sql`), and three mutants survived them: inverting the
comparison (`<=` -> `>=`), swapping `LEAST` for `GREATEST`, and zeroing the lag
CASE. The Q/A executed the inverted mutant against live BigQuery and it returned
`{GDP: 2023-01-01, CPIAUCSL: 2023-01-01, ...}` -- precisely the look-ahead rows
this step removes. Criterion 2 explicitly demands the halves be "asserted
separately so fixing one and not the other cannot pass", so that clause was
materially defeated. The test now pins the predicate SHAPE (`) <= DATE(@cutoff)`,
`LEAST(` present and `GREATEST(` absent) and generates the expected lag CASE
from `cache.MACRO_PUBLICATION_LAG_DAYS`, asserting every `WHEN '<sid>' THEN
<lag>` fragment -- which also pins both read paths to ONE source of truth.

**F2 -- criterion 1's kill mechanism was mis-attributed. FIXED.** Dropping
`realtime_start` from the preload SELECT (M10) and from the stored row (M11)
BOTH left the cycle-1 suite green, because the fixture's GDP row is excluded by
the 125-day lag regardless of its stamp. So preload sites 1 and 2 had ZERO
coverage and the demonstrated exclusion was attributable to the lag rather than
the vintage. Added `test_preload_select_requests_the_vintage_column` and
`test_preloaded_rows_carry_the_vintage`.

**F3 -- a claim I had backwards, in four places including a test docstring.
FIXED.** I wrote "whichever is EARLIER wins, so the rule can only ever be
conservative". MIN selects the earlier availability date, which makes a row
visible SOONER -- **anti**-conservative with respect to look-ahead. The rule is
still correct (a true vintage should govern; discarding a far-future backfill
stamp is what keeps the sample alive) but it is conservative only where the lag
upper-bounds the real release delay. The Q/A's counter-case: the Oct-2013
shutdown pushed the Employment Situation to ~52 days against a `UNRATE` lag of
40, so such a row would be admitted ~12 days early. Reworded in
`experiment_results.md`, `contract.md`, the `cache.py` header and the test
docstring, each now pointing at 82.18 as the only real close.

Also recorded from the Q/A: under MIN semantics `realtime_start` alone can never
CAUSE an exclusion (effective <= derived always), so criterion 1 holds on the
demanded fixture but not as a general property. That is a true statement about
the design and is disclosed rather than papered over.

### Mutation re-proof (in-tree, restored from backup, 0 MUTANT markers left)

```
M1  invert `) <= DATE(@cutoff)` -> `>=`      1 failed   (was: 11 passed)
M2  LEAST -> GREATEST                        1 failed   (was: 11 passed)
M3  zero the lag CASE                        1 failed   (was: 11 passed)
M10 drop realtime_start from preload SELECT  1 failed   (was: 11 passed)
M11 drop realtime_start from stored rows     1 failed   (was: 11 passed)
RESTORED                                     13 passed
```

Suite is now 13 tests (was 11).


## CYCLE 3 -- disposition of the cycle-2 CONDITIONAL

Cycle 2 confirmed all 4 criteria MET and independently reproduced the whole
mutation matrix (M1/M2/M3/M10/M11 all killed, control green). Four findings,
all closed.

**F-a -- a count invalidated by this very cycle. FIXED, and MEASURED.**
`experiment_results.md` said "49 green in total", which is 16+22+**11** -- the
CYCLE-1 size of this suite, falsified by the two tests cycle 2 added. Measured
now: 82.0 = 16, 82.2 = 22, 82.15 = 15 -> **`53 passed in 9.15s`**.

And I got it wrong AGAIN on the first attempt: this paragraph originally said
`52`, arithmetic done in my head while correcting a miscount. That is the FOURTH
unverified count in this phase and the third written INSIDE the text fixing the
previous one. The number above is now the output of the command, pasted, not
computed. The rule that actually works is: run the command, paste its output,
never add the columns yourself.

**F-b -- the F3 rewording swept 3 of 4 declared locations. FIXED.**
`contract.md:22` still read "using a conservative effective vintage" --
unqualified -- while the correction sat ten lines below at `:32`. So the
past-tense claim "Reworded in experiment_results.md, contract.md, the cache.py
header and the test docstring" was only partly true. Line 22 now carries the
conditional form explicitly.

**F-c -- the production flag read was exercised by ZERO tests. FIXED.**
Every test monkeypatches `_pit_enabled`, so replacing its body with
`return False` left 13/13 green while the entire step was inert. Added
`test_production_flag_read_is_exercised_and_defaults_on`, the one assertion that
touches the unpatched path. Mutation M6 now kills it. Lesson: mutate the FLAG
READ, not only the guard it gates.

**F-d -- the caller's fail-closed branch was uncovered. FIXED, after my own
first attempt failed.** The helper's `None` return was covered but
`cached_macro`'s handling of it was not, so failing OPEN passed 13/13. My first
fixture used `date="not-a-date"` -- and mutation M5 STILL survived, because
`"not-a-date" > "2023-02-01"` lexicographically, so the row was dropped by the
date check and never reached the vintage branch. **The test passed for the wrong
reason.** Fixture corrected to `"2022-99-99"` (sorts below the cutoff, still
unparseable), with an inline assertion pinning that property. M5 now kills it.

### Mutation matrix after cycle 3 (in-tree, restored, 0 MUTANT markers)

```
M1  invert `) <= DATE(@cutoff)`              1 failed
M2  LEAST -> GREATEST                        1 failed
M3  zero the lag CASE                        1 failed
M5  fail-OPEN on unparseable date            1 failed   (survived my first fixture)
M6  _pit_enabled -> return False             1 failed
M10 drop realtime_start from preload SELECT  1 failed
M11 drop realtime_start from stored rows     1 failed
CONTROL                                      15 passed
```

Suite is 15 tests (11 -> 13 -> 15 across the three cycles).
