---
name: vacuous-bq-guards-82-12
description: Step 82.12 sweep -- the literal vacuous-isinstance sweep yields ZERO defects (82.0 closed the only one); the real find is a nonexistent-column P1; BQ dry_run is the $0 oracle; measured BQ->Python type map
metadata:
  type: project
---

Measured 2026-08-05 for masterplan step 82.12 (vacuous type-assumption guards over
BigQuery STRING columns). Brief: `handoff/current/research_brief_82.12.md`.

**The literal sweep is EMPTY -- say so, don't pad.** 534 non-test `isinstance` calls in
`backend/`; 8 are date-type tests; 5 are live code; **all 5 are CORRECT**. The 82.0 fix
(`backend/backtest/cache.py` `preload_macro`) closed the only vacuous instance, and it is
complete across all 5 production shapes (str/date/datetime/None/`""`/BQ `Row`). A step
framed as "find more of the same" has zero expected yield -- re-frame it as "build the
schema oracle and prove the surface clean".

**Why:** an audit whose honest answer is "nothing found" invites either a wasted-cycle
verdict or manufactured false positives. Both are worse than the honest report.

**How to apply:** when a defect-class sweep comes back empty, lead with the measurement
that proves emptiness, then pivot the value to the standing detector.

## Measured facts (re-derive line numbers; they move)

- **BQ -> Python mapping, measured on the live table (not inferred):** STRING -> `str`,
  DATE -> `datetime.date`, TIMESTAMP -> `datetime.datetime` (tz-aware UTC), FLOAT ->
  `float`. Same table (`financial_reports.historical_macro`) carries all four.
- **Live schema shape:** 4 datasets, 33 tables, **477 columns** -- 189 STRING, 182 FLOAT,
  48 INTEGER, 31 TIMESTAMP, 13 DATE, 6 BOOLEAN, 5 JSON, 3 RECORD. Every STRING-typed date
  column lives in `financial_reports` (a us-central1-local legacy); `pyfinagent_data`/
  `_pms`/`_hdw` use native DATE/TIMESTAMP.
- **In SQL this defect is LOUD, only Python makes it silent.** `STRING >= CURRENT_DATE()`
  -> `400 No matching signature for operator >=`. STRING columns have an EMPTY "coerce to"
  set (coercion is literal/parameter-only). That asymmetry is why the bug survived.
- **`SAFE.TIMESTAMP(ts)` on a native TIMESTAMP -> `400 SAFE with function timestamp is
  not supported`, but plain `TIMESTAMP(ts)` on a TIMESTAMP is ACCEPTED.** The
  `cycle_health.py` comment about this is accurate -- do not "fix" it.
- **No type checker catches this.** pyright default = 0 diagnostics; pyright strict +
  `reportUnnecessaryIsInstance` = 5 diagnostics, ALL about the untyped dict, **none on
  any isinstance line** -- including `isinstance(v, date)` where `v: str`. The decisive
  fact lives in the BQ schema, not the source. See [[measure-dont-assert]].
- `pyrightconfig.json` exists at root (pins `.venv312`/py3.12 while the project runs 3.14
  in `.venv`) and sets no `typeCheckingMode`. There is no ruff/pre-commit config.

## The real find: same root cause, different symptom

`backend/slack_bot/jobs/_production_fns.py` `make_ledger_fetch_fn` selects
`paper_trades.timestamp` and `realized_pnl` -- **neither column exists** (real ones are
`created_at` STRING and `realized_pnl_pct` FLOAT). Reproduced live: `400 Unrecognized
name: timestamp`. Swallowed by `except Exception -> logger.warning -> return []`, so
`nightly_outcome_rebuild` has been running on ZERO trades.

**Why it matters:** 82.12's immutable criteria scope to "declared STRING consumed as
dates/numbers", which would EXCLUDE this. Wrong-NAME and wrong-TYPE are the same root
cause (unverified assumption about a BQ column + fail-open except). Widen the hypothesis
in the contract prose; queue the fix as its own step.

## Traps that produce cry-wolf

- **Lexical ISO-date STRING comparison is CORRECT by construction** (`'2026-01-01'` zero-
  padded => lexical order == chronological). Does NOT extend to numbers-in-strings
  (`'9' > '10'`). A naive "STRING column in a date range" grep flags a wall of correct code.
- **`datetime` subclasses `date`**, so `isinstance(ts, date)` is TRUE for a TIMESTAMP
  column. The correct plain-date test is `isinstance(d, date) and not isinstance(d, datetime)`.
- **INERT != VACUOUS.** An isinstance branch that never runs in prod but sits above a
  correct `str` fallthrough is inert, not a defect. Only call it vacuous when a SAFETY
  DECISION depends on the dead branch.
- `paper_positions.stop_advanced_at_R` is a **timestamp string**, not an R-multiple
  (`_at_R` = "at the R threshold"). Name-based scoping mis-reads it.
- `cycle_health._STRING_DATE_TIMESTAMP_COLS` is a hand-written 2-entry set but is **6/6
  correct today** against its actual call sites. Latent drift hazard, not a live defect.

## Instrument recommendation

- **`QueryJobConfig(dry_run=True)` is the oracle** -- BigQuery parses and type-checks
  against the live schema for **$0**. Verified it catches nonexistent columns AND
  STRING-vs-DATE operator mismatches. Do not hand-roll SQL regex parsing.
- **A NAME regex is itself a hand-written list** -- it satisfies "derived from live
  schemas" in letter while violating it in spirit, and it under-covers. Derive two-sided
  (schema oracle x consumer sites) and join.
- **Assert the instrument scanned something.** My first scanner reported "0 unknown
  identifiers" because of a cwd-reset relative path plus a non-greedy regex that captured
  the PROJECT name instead of the table. A checker that scans nothing is indistinguishable
  from a clean codebase. See [[operations-that-cannot-fail-loudly]].
- Branch coverage (`--cov-branch`) is the cheapest standing detector: a vacuous guard
  shows as a **partial branch** (the `if` line covered, one destination never visited).
  Line coverage shows nothing.
- Fixture precedent to copy: `backend/tests/test_phase_82_0_macro_ingestion.py` builds
  rows with the PRODUCTION type and then **asserts the fixture's own type** before using
  it. Read the expected type FROM the oracle, not a hardcoded `"STRING"`.
- No repo helper returns realistic BQ `Row` objects; every fixture is a hand-built dict,
  so `cache.py`'s explicit non-dict-Row branch is untested.
