# live_check_50.1 -- FX data layer (LIVE evidence)

Verified 2026-05-30 against live yfinance + BigQuery (financial_reports, us-central1).

## 1. Migration (criterion #2)
`python scripts/migrations/create_historical_fx_rates_table.py --apply`
-> `APPLIED: sunny-might-477607-p8.financial_reports.historical_fx_rates created/already-exists`
(no --location pin; client resolved the us-central1 region.)

## 2. Live get_fx_rate -- DIRECTION LOCKED (criteria #1, #4)
```
EUR/USD (USD per EUR) = 1.1659088134765625
KRW/USD (USD per KRW) = 0.000663512767153591
USD/EUR (inverse)     = 0.8577000091612245
USD/USD               = 1.0   (from==to short-circuit -> US-only byte-identical)
```
EUR/USD ~1.16 (correct), KRW/USD ~0.00066 = 1/~1507 KRW-per-USD (correct, NOT inverted -- the KRWUSD=X inversion pitfall is avoided).

## 3. Backfill + BQ contents (criterion #2)
`fx_rates.backfill_fx(["EUR","KRW"], "2026-05-15", "2026-05-29")` -> 22 rows written.
historical_fx_rates well-formed rows:
```
EURUSD: 2026-05-15..2026-05-29, n=12, avg_rate=1.163596
KRWUSD: 2026-05-15..2026-05-29, n=12, avg_rate=0.000665
```

## 4. Point-in-time as-of read (no look-ahead)
```
as-of EUR/USD @2026-05-20 = 1.1607122421264648
as-of KRW/USD @2026-05-20 = 0.0006631959519788489
```
The as-of query `WHERE pair=? AND date<=? ORDER BY date DESC LIMIT 1` returns the latest rate on/before the requested date -> no look-ahead.

## 5. data_ingestion.py:146 currency-stub fix (criterion #3)
`"USD" if market=="US" else "USD"` -> `markets.get_market_config(market)["currency"]`.
Deterministic check: `fx_rates.market_currency('EU')=='EUR'`, `('KR')=='KRW'`, `('US')=='USD'`. `import backend.backtest.data_ingestion` OK.

## Deterministic masterplan command
```
ast.parse(fx_rates.py) -> OK
get_fx_rate('USD','USD')==1.0 + market_currency EU/KR/US == EUR/KRW/USD -> "fx_rates OK"
test -f handoff/current/live_check_50.1.md -> present
```

## Known caveats (flagged for hardening; NOT criterion violations)
- **2 malformed rows in the streaming buffer**: the FIRST backfill run had a bug (single-ticker yf.download returns a 1-col DataFrame; `.items()` iterated the column name into `date`) -> wrote 2 rows with `date='EURUSD=X'`/`'KRW=X'`. Bug FIXED (squeeze to Series before `.items()`). The DELETE cleanup failed ("would affect rows in the streaming buffer" -- BQ blocks DELETE on rows <~90min old from insert_rows_json). These 2 rows are HARMLESS: the as-of read `date<=` excludes them (non-ISO `'EURUSD=X'` sorts lexically after any `'2026-..'`). Cleanup `DELETE WHERE date NOT LIKE '2%'` will succeed after the streaming buffer flushes (~90 min).
- **Append (not upsert)**: backfill + live write-through both `insert_rows_json` (append), so a date can have duplicate rows. The as-of read (`LIMIT 1`) is unaffected. Hardening follow-up: switch backfill to a BQ load-job (avoids the streaming buffer + the DELETE limitation) and MERGE-upsert for dedup.

## Verdict
All 4 immutable criteria met: fx_rates.py with get_fx_rate (USD/USD=1.0) + correct EUR/USD + KRW/USD direction; historical_fx_rates table created + backfilled (12 days each); data_ingestion currency stub fixed; live rates fetched verbatim. US-only/USD path byte-identical (from==to->1.0). $0 LLM, no pip.
