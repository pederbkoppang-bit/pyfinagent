# Contract -- step 82.15

**Step id:** 82.15 (phase-82) | **Priority:** P0 | PLAN phase, before GENERATE.

## Research gate

`handoff/current/research_brief_82.15.md` -- **gate_passed: true** (moderate,
7 sources read in full, 24+ URLs, recency scan performed).

Main re-verified the load-bearing claims:

| claim | verified |
|---|---|
| a naive strict filter returns ZERO rows historically | **CONFIRMED by measurement.** `realtime_start` MIN is 2026-03-22; `realtime_start <= cutoff` yields 0/4729 rows at 2020-01-01, 2023-01-01 AND 2025-06-01. The backtest window is 2018-01-01..2025-12-31, so the naive fix blanks all six macro features across the ENTIRE sample |
| 0 NULL `realtime_start` | CONFIRMED (migration completed) -- so there is no live NULL population, though criterion 3 still requires the NULL branch be decided and tested on a FIXTURE |
| four code sites, not two | CONFIRMED: `cache.py:255` (preload SELECT omits `realtime_start`), `:351` (stores only value/date), `:481-485` (the DESC-list `break` fires on the first DATE match and would skip a row failing the vintage test), `:494-515` (fallback WHERE; `date` is STRING but `realtime_start` is DATE, so the bound STRING `@cutoff` needs `DATE(@cutoff)`) |
| `sortino.py:108` targets a table that 404s | CONFIRMED -- Main hit exactly this 404 independently at the start of the session. Dead code; **do not** "fix" the vintage there believing it runs. Belongs to 82.8 |

## Hypothesis

Publication-lag look-ahead can be removed WITHOUT destroying the sample by
using an effective vintage that is conservative ONLY WHERE the per-series lag
upper-bounds the real release delay (see the correction below -- an earlier
draft claimed unconditional conservatism, which is false):

```
effective_vintage(series, obs_date, realtime_start)
    = MIN(realtime_start, obs_date + MACRO_PUBLICATION_LAG_DAYS[series])
```

For pre-migration rows `realtime_start` is our write date (2026-03-22..25),
which carries no information about 2018-2025; the lag term supplies a
defensible per-series availability date instead. For rows ingested from now on,
MIN never blanks the sample. Where `realtime_start` is a TRUE vintage it governs and the row is correctly visible from its real publication date. Where the stamp is an 82.0 backfill artifact (our write date) the rule degrades to the lag estimate, which is conservative ONLY IF that lag upper-bounds the real release delay -- where it underestimates a delayed release the row is admitted early (see LIMITS).

## Costs, stated up front (the research brief's, carried verbatim in intent)

1. **This fixes PUBLICATION LAG ONLY, not REVISIONS.** The ingest dedupes on
   `(series_id, date)` (`data_ingestion.py`), so a revised value can never be
   stored beside its original -- revisions are structurally uncapturable.
   **82.15 must NOT be reported as "look-ahead fixed".**
2. **Sharpe will fall** on macro-conditioned strategies (sources bracket
   100-500bp of return / 15-25% of Sharpe for GDP-dependent strategies).
3. **Comparability.** Any figure produced before this step is not comparable
   with one produced after. 82.3 has NOT been run, so nothing existing breaks --
   but ALL of 82.3 (incumbent AND candidates) must run in ONE flag state.
4. The complete fix is a one-off ALFRED backfill of true vintages
   (`realtime_start`/`realtime_end`, the pattern already used at
   `backend/tools/fred_releases.py:55-56`). Queue as its own step.

## Flag decision (Main's call, disclosed)

Ships behind `macro_point_in_time_enabled`, **default True**. This project's
convention is default-OFF for behaviour changes, and that convention is kept
for the MONEY path -- but this touches the research lane only (backtest feature
construction), and defaulting OFF would mean knowingly shipping look-ahead into
the very evidence 82.3 exists to produce. The flag exists so the effect can be
measured ON vs OFF and reverted without a code change.

## Immutable success criteria (verbatim from .claude/masterplan.json)

1. cached_macro excludes rows whose realtime_start is after the requested cutoff, asserted on a fixture containing a row dated before the cutoff but stamped with a vintage after it
2. the same exclusion holds on BOTH the preloaded fast path and the per-cutoff BQ fallback, asserted separately so fixing one and not the other cannot pass
3. a fixture with NULL realtime_start (pre-migration rows) resolves per the documented decision rather than being silently dropped or silently included, and the chosen behaviour is stated in the test name
4. a regression fixture asserts the pre-fix behaviour differed: the same query returns a row under the old date-only filter and does not under the new one, so the guard cannot pass vacuously

**Verification command:** `source .venv/bin/activate && python -m pytest backend/tests/test_phase_82_15_macro_point_in_time.py -q`

## Plan

1. `MACRO_PUBLICATION_LAG_DAYS` beside `MACRO_SERIES_MAX_AGE_DAYS`
   (`cache.py:23-52`), with each lag justified per series.
2. `_effective_vintage()` helper.
3. Fix ALL FOUR sites: fetch `realtime_start` in the preload SELECT, carry it
   into `_macro_full`, restructure the fast-path loop so it does not `break` on
   a date-only match, and add the predicate to the fallback WHERE with
   `DATE(@cutoff)`.
4. Tests incl. the NULL-vintage fixture branch and a pre-fix regression pin.
5. Fresh Q/A.

## Out of scope

The ALFRED backfill (own step). `sortino.py` (82.8, dead code). No live-funnel
change.
