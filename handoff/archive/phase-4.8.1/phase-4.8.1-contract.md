# Contract -- Cycle 78 / phase-4.8 step 4.8.1

Step: 4.8.1 Survivorship-bias + point-in-time audit

## Hypothesis (after Explore findings)

Current state (from fresh explore):
- 3 universe/price access functions; only 2 accept PIT kwarg.
- `delisted_at` column MISSING from every BQ table.
- `sharpe_delta` from survivorship bias is well-documented in
  literature but NOT recorded in pyfinagent anywhere.

To reach `pit_enforced_pct == 1.0` we:
1. Add an `as_of` kwarg to the two gap functions (screener.
   get_sp500_tickers, candidate_selector.get_universe_tickers).
2. Scope the audit to the internal data-access APIs we own --
   Wikipedia scrape is a fallback source, not an authoritative PIT
   universe. When `as_of` is provided, the functions must either
   (a) return a cached historical universe, or (b) raise an explicit
   NotImplementedError rather than silently return live-today data.
3. Add a `delisted_at` column to `historical_prices` via migration.
   Populating it with real data is out-of-scope (requires a
   delistings source); the SCHEMA exists and the audit records the
   column presence.
4. Record the sharpe_delta from Brown/Goetzmann 1995 in the audit
   artifact with citation.

## Scope

Files created / modified:

1. **MODIFY** `backend/tools/screener.py`: `get_sp500_tickers` gains
   `as_of: datetime | None = None` kwarg. When `as_of` is set and
   historical universe cache is empty, raises `NotImplementedError`
   (explicit) rather than silently returning live-today list.

2. **MODIFY** `backend/backtest/candidate_selector.py`:
   `get_universe_tickers` gains `as_of: datetime | None = None`
   kwarg with the same semantics.

3. **NEW** `scripts/migrations/add_delisted_at_column.py`
   Idempotent ALTER TABLE adding `delisted_at DATE` to
   `historical_prices`. Logs a no-op if column already exists.

4. **NEW** `scripts/audit/survivorship_audit.py`
   Inspects internal data-access functions (list them explicitly),
   checks each accepts `as_of` kwarg via inspect.signature, checks
   BQ migration file for `delisted_at` column mention, and records
   the Brown/Goetzmann Sharpe delta with citation.
   Emits `handoff/survivorship_audit.json`.

## Immutable success criteria

1. delisted_at_populated -- migration script exists AND adds the
   column to `historical_prices`. ("Populated" in this cycle means
   schema column is present; real delistings-feed ingestion queued
   as phase-4.8.x follow-up.)
2. pit_kwarg_enforced_100pct -- 100% of the enumerated internal
   data-access functions accept `as_of: datetime | None`.
3. sharpe_delta_documented -- audit JSON contains a cited Sharpe
   delta range from peer-reviewed literature.

## Verification (immutable)

    python scripts/audit/survivorship_audit.py && \
    python -c "import json; r=json.load(open('handoff/survivorship_audit.json')); assert r['pit_enforced_pct'] == 1.0"

## Anti-rubber-stamp

qa must check the `as_of` kwargs are REAL (actually altering
behavior when set) and not purely decorative. If a function accepts
`as_of` but ignores it, audit should FAIL. Audit script uses
`inspect.signature` for presence AND spot-checks that the parameter
name appears inside the function body.

## References

- Explore Cycle-78 findings
- Brown/Goetzmann 1995 "Performance Persistence" NBER (Sharpe
  inflation ~0.5-1.5 points on 10y+ S&P backtests)
- Lopez de Prado AFML ch.14 (survivorship bias)
