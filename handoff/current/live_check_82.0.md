# live_check -- step 82.0

Captured 2026-08-03 by Main against the live system.

## 1. The required observable: preload_macro() returns > 0

**CORRECTED IN CYCLE-2 after a Q/A FAIL.** The cycle-1 version of this file
claimed preload_macro "returned 0 before this step (that was the freeze)".
That is FALSE and does not reproduce. `historical_macro.date` is a STRING
column, and BOTH the pre-fix phase-25.D7 gate and the cycle-1 per-series
rewrite tested `isinstance(rd, datetime.date)` -- false for every production
row -- so the staleness branch never executed. Pre-fix `preload_macro`
returned **4412** (it cached the stale table); it now returns **4729**.

The honest delta is **4412 -> 4729 (+317 rows)**, and the real defect was
never a refusal or a hang: the guard was VACUOUS and silently fed backtests
212-day-old macro features. Verbatim post-fix:

```
$ python -c "... cache.init_cache(...); print(cache.preload_macro())"
preload_macro() -> 4729
series cached: ['CPIAUCSL', 'DGS10', 'FEDFUNDS', 'GDP', 'T10Y2Y', 'UMCSENT', 'UNRATE']
```

All 7 ingested series are cached. The cycle-1 claim that this avoided a
per-cutoff-date BQ fallback / ~40-minute hang is WITHDRAWN -- the cache was
already populating pre-fix (with stale data), so no hang was occurring on
this path. What changed is the DATA (now current) and the GATE (now real).

Gate non-vacuity, executed against the production STRING date type:

```
A) live preload_macro() -> 4729 | series: 7
B) stale-GDP, production STRING dates -> preload_macro() -> 0 (0 = gate fires)
WARNING preload_macro: stale data, refusing to cache -- 1 of 2 series past
        their per-series SLA: GDP(newest=2018-05-17 age=3000d limit=225d)
```

Direction B is the check the cycle-1 fixture could not perform: it passed
`datetime.date`, a type the production query never returns, so the guard was
green for every possible production state including a fully dead table.

## 2. MAX(date) advanced past 2025-12-31, per series

```
series         n       newest  age_d nullVint  newest_vintage
CPIAUCSL     101   2026-06-01     63        0  2026-08-03
DGS10       2144   2026-07-30      4        0  2026-08-03
FEDFUNDS     102   2026-06-01     63        0  2026-08-03
GDP           34   2026-04-01    124        0  2026-08-03
T10Y2Y      2145   2026-07-31      3        0  2026-08-03
UMCSENT      102   2026-06-01     63        0  2026-08-03
UNRATE       101   2026-06-01     63        0  2026-08-03

TOTAL rows=4729  last_ingest=2026-08-03 07:01:30.964909+00:00
```

Pre-step the table held 4412 rows, `MAX(date) = 2025-12-31`,
`MAX(ingested_at) = 2026-03-25`.

Every series is inside its per-series SLA
(`cache.MACRO_SERIES_MAX_AGE_DAYS`): CPIAUCSL 63<=80, DGS10 4<=5,
FEDFUNDS 63<=70, GDP 124<=225, T10Y2Y 3<=5, UMCSENT 63<=70, UNRATE 63<=75.

> **ANNOTATION -- dated capture.** The daily bounds quoted above (`DGS10 4<=5`,
> `T10Y2Y 3<=5`) are the values in force when this evidence was captured. They
> were later widened to **12** days in cycle 3, after the Q/A observed that a
> 5-day bound left only ONE day of live headroom against a cron that had never
> been seen firing, and that FRED daily series skip weekends and holidays. The
> conclusion is unchanged -- 4<=12 and 3<=12 -- so the capture still shows every
> series inside its SLA. Annotated rather than rewritten: this is a dated record
> of a live run. Section 5 likewise records `15 passed`, the suite size at
> capture time; it is **16** after the cycle-3 regression pins were added.

**This validates the SLA thresholds against real data.** GDP's newest row is
124 days old while being fully current -- FRED dates quarterly series to the
quarter start, so a flat 35-day bound would have condemned a healthy table
had it ever executed.

CORRECTED: these per-series ages were computed by an ad-hoc SQL query, NOT by
`preload_macro`. In cycle-1 `preload_macro` evaluated ZERO series, so the
statement "both failure directions are now closed" was unsupported at the
time it was written. After the cycle-2 date-coercion fix the gate does
evaluate all 7 series, and direction B above demonstrates the dead-series
case actually refusing.

## 3. The end-date coupling is severed, observed live

The backfill's FRED request carried `observation_end=2026-08-03` (today),
NOT `settings.backtest_end_date` ("2025-12-31"):

```
INFO HTTP Request: GET https://api.stlouisfed.org/fred/series/observations
     ?series_id=DGS10&api_key=<REDACTED>&file_type=json
     &observation_start=2018-01-01&observation_end=2026-08-03&sort_order=asc
     "HTTP/1.1 200 OK"
INFO Ingested 317 macro rows (observation_end=2026-08-03, outcome=ok)
INFO macro_ingest_daily: inserted 317 rows
```

317 rows inserted. Pre-fix this call requested `observation_end=2025-12-31`
and inserted zero rows while reporting success.

## 4. Migration applied (versioned, idempotent)

```
$ python scripts/migrations/add_macro_realtime_start.py
[migration] add realtime_start column: OK
[migration] backfill vintage from ingested_at: OK
[migration] verifying ...
  total_rows=4412 null_vintage=0 earliest=2026-03-22 latest=2026-03-25
```

Pre-existing rows carry the conservative vintage `DATE(ingested_at)`
(2026-03-22..25); rows written by this step carry a true first-observation
vintage of 2026-08-03. Post-backfill null_vintage is 0 across all 4729 rows.

## 5. Test suite

```
$ python -m pytest backend/tests/test_phase_82_0_macro_ingestion.py -q
15 passed in 0.81s
```

Mutation-tested: reverting `preload_macro`'s per-series gate to the pre-fix
global-max logic makes `test_per_series_sla_catches_dead_gdp_behind_a_live_daily_series`
FAIL. cache.py was restored from backup and re-verified (0 MUTANT markers).

CYCLE-2 NOTE ON THAT MUTATION: it was MIS-SCOPED. It only proved
discrimination inside the date-typed fixture space, which production never
produces -- so it could not detect that the guard was vacuous. The lesson is
that a mutation matrix inherits its fixture's blind spots; mutating the code
does not test the fixture. Two regression pins were added in cycle-2:
`test_gate_is_not_vacuous_on_the_production_date_type` and
`test_unparseable_dates_fail_closed`.

## Honest limitations

- `test_app_startup_registers_the_macro_cron` is a SOURCE SCAN of
  `backend/main.py`, not a behavioural check. It proves the registration call
  is present in the file, not that a booted app registered the job. The
  behavioural half is covered by
  `test_macro_ingest_cron_is_registered` against a stub scheduler. Booting
  the app to assert on a live APScheduler jobstore was not done.
- The scheduled job has NOT yet been observed firing on its cron trigger --
  that requires a backend restart plus waiting for 08:10 ET. The registration
  is verified; the first autonomous run is not.
- The conservative vintage backfill (`DATE(ingested_at)`) is NOT the true
  publication vintage for pre-2026-08-03 rows. It is provably not earlier
  than the truth, so it cannot manufacture look-ahead, but it will make
  point-in-time backtests over the historical span slightly pessimistic.
