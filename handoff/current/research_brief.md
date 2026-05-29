# Research Brief — phase-50.1: FX Data Layer

**Step:** phase-50.1 — `backend/services/fx_rates.py` + `historical_fx_rates` BQ table (EUR/USD + KRW/USD from yfinance) + fix `data_ingestion.py:146` currency stub.
**Tier:** moderate
**Builds on:** `handoff/current/research_brief_multimarket.md` (already gate-passed — the broad multi-market brief). THIS gate = FX-layer implementation specifics + exact internal integration points.
**Status:** IN PROGRESS (WRITE-FIRST, appended incrementally)

---

## Internal code inventory (Q1-Q6, file:line anchors)

### Q1 — BQ table creation pattern + which dataset

**Pattern (two idioms, both idempotent):**
1. **Standalone migration script** (`scripts/migrations/*.py`) — the CLAUDE.md-mandated, version-controlled, re-runnable path. Two sub-shapes:
   - **`google.cloud.bigquery.Table` + `create_table` with a Python `SchemaField` list**, guarded by `client.get_table(ref)` in a try/except (create only on miss). Canonical example: `scripts/migrations/migrate_backtest_data.py:88-95` (the file that created `historical_prices`/`historical_fundamentals`/`historical_macro`). This is the closest mirror because `historical_fx_rates` is a sibling historical table.
   - **`CREATE TABLE IF NOT EXISTS` DDL string + `client.query(SQL).result()`**, with a `--apply`/dry-run flag and `--verbose`. Canonical example: `scripts/migrations/create_data_source_events_table.py:43-98` (PARTITION BY DATE + CLUSTER BY, OPTIONS descriptions, `argparse`, dry-run default). This is the NEWER, cleaner idiom (phase-25.B7) and is preferable for a NEW table because it self-documents partition/cluster and defaults to dry-run.
2. **Auto-create on first ingest** (`data_ingestion.py:39-49` `_ensure_tables_exist`) — imports `ALL_TABLES` from `migrate_backtest_data` and creates any missing table at ingest time. **This means: if `historical_fx_rates` is added to `migrate_backtest_data.ALL_TABLES`, it auto-creates on the next ingest run** — but a dedicated migration is still the version-controlled source of truth.

**Recommended shape for `historical_fx_rates`:** a NEW `scripts/migrations/create_historical_fx_rates_table.py` mirroring `create_data_source_events_table.py` (DDL string, `--apply` dry-run default, PARTITION BY DATE(date) — but note date is stored as STRING in the sibling tables; see schema note below — CLUSTER BY pair). PLUS add the table to `migrate_backtest_data.ALL_TABLES` so `_ensure_tables_exist` auto-creates it (defense-in-depth, matches how prices/macro work).

**Dataset: `financial_reports`** (NOT `pyfinagent_data`). Rationale with anchors:
- The sibling historical tables (`historical_prices`, `historical_fundamentals`, `historical_macro`) live in **`financial_reports`** — `migrate_backtest_data.py:24` (`DATASET = "financial_reports"`), and `data_ingestion.py:34` (`self.dataset = settings.bq_dataset_reports`), `settings.py:43` (`bq_dataset_reports: str = "financial_reports"`).
- `historical_fx_rates` is read alongside `historical_prices` for backtest NAV conversion (50.5), so co-locating in `financial_reports` avoids a cross-dataset join and matches the `DataIngestionService._table()` helper (`data_ingestion.py:36-37`) that all the other historical tables use. **Location note: `financial_reports` is `us-central1`, not `US`** (per auto-memory + CLAUDE.md BQ table) — the migration's BQ client must not pin `--location US`; the `bigquery.Client(project=...)` default (no location) resolves it correctly, as `migrate_backtest_data.py` does.
- `data_source_events` is the lone counter-example in `pyfinagent_data` (`create_data_source_events_table.py:39`), but that table is consumed by the analysis-pipeline provenance metric, not the backtest replay path. FX belongs with the historical replay tables.

**Schema note (mirror the siblings exactly):** `historical_prices` stores `date` as **STRING** (`migrate_backtest_data.py:31`, `mode="REQUIRED"`), NOT a DATE/TIMESTAMP. To keep joins on `(ticker,date)`↔`(pair,date)` trivial and avoid a type-mismatch, `historical_fx_rates.date` should also be a **STRING** "YYYY-MM-DD". If PARTITION BY is desired it must be on a DATE/TIMESTAMP column, so either (a) skip partitioning (the table is tiny — ~1000 rows/pair/3yr, like `historical_macro` which is unpartitioned) or (b) add a separate `DATE` column for partitioning. Given the table is ~3-4K rows total, **unpartitioned (mirroring `historical_macro`) is the right call** — `historical_macro` at `migrate_backtest_data.py:61-68` is the exact size-class precedent (252-ish rows, no partition).

### Q2 — yfinance usage in-repo + retry/rate-limit/error pattern

**No shared yfinance wrapper exists.** 20+ call sites each inline `import yfinance as yf` (grep: `data_ingestion.py:11`, `screener.py:11`, `tools/yfinance_tool.py:7`, `paper_trader.py` via `_get_live_price:1127`, `regime_detector.py:102`, `_production_fns.py:62`, etc.). Two call shapes:
- **Bulk `yf.download(batch, ...)`** — `data_ingestion.py:104-108` and `screener.py:110-111`. Args: `group_by="ticker", auto_adjust=True, threads=True, progress=False`; `start=/end=` (ingestion) or `period=` (screener). FX pairs should use this shape for the backfill.
- **Single `yf.Ticker(t).history(period=...)`** — `paper_trader.py:_get_live_price:1127-1130` (`period="1d"`, `Close.iloc[-1]`). FX live-mark should mirror this for the daily rate.

**Error/retry pattern = try/except + skip/return-empty; there is NO exponential-backoff retry around yfinance** despite the backend-tools.md claim ("automatic retry and exponential backoff" applies to AlphaVantage/FRED-keyed APIs, NOT yfinance). Concretely:
- `data_ingestion.py:103-114` — `try: yf.download(...) except Exception as e: logger.error(...); continue` (skip the batch). Then `if data is None or data.empty: continue`.
- `screener.py:109-117` — same: `try/except → logger.error → return []`; `if data.empty: return []`.
- `_get_live_price` (`paper_trader.py:1126-1133`) — `try/except → logger.debug → return None`.
- Per-ticker inner loop wraps each ticker in its own try/except and `logger.warning` on failure (`data_ingestion.py:120,154-155`).

**Batch size constant**: `_YF_BATCH = 50` (`data_ingestion.py:24`). For FX (only 2-3 pairs) a single `yf.download([...])` call suffices — no batching needed.

**Implication for `fx_rates.py`:** reuse the existing idiom (try/except → log → return None/empty + per-pair inner guard). FX is low-cardinality (2-3 pairs) and called once/day live + once at backfill, so the documented yfinance rate-limit degradation (multimarket brief finding: 2024-2026 IP bans) is low-risk here. The FRED fallback (Q-external) is the robustness hedge, mirroring `data_ingestion.ingest_macro` (`:284-324`) httpx pattern.

### Q3 — the `data_ingestion.py:146` currency stub

**Exact bug (`data_ingestion.py:146`):**
```python
"currency": "USD" if market == "US" else "USD",  # TODO: lookup from MARKET_CONFIG
```
Both branches return `"USD"` — non-US rows get the wrong currency.

**Context (what writes it / the market arg):**
- This is inside `ingest_prices` (`:93`), in the per-row dict appended at `:142-153`.
- `market` is derived at `:137-141`: defaults to `"US"`, and if the ticker is namespaced (`"DE:BAS"`), it splits on `:` → `market="DE", clean_ticker="BAS"` (`:140-141`). So `market` is the market CODE from `markets.py`.

**The fix:** read currency from `markets.MARKET_CONFIG`. The map already exists at `markets.py:21-52` with the `get_market_config()` accessor at `markets.py:75-78`:
```python
from backend.backtest import markets
...
"currency": markets.get_market_config(market)["currency"],
```
`get_market_config` is robust: it uppercases and falls back to US for unknown markets (`markets.py:77-78`), so a malformed namespace can't crash the ingest. Mapping confirmed: US→USD, EU→EUR, KR→KRW (`markets.py:24,42,48`). **CAVEAT:** the multimarket brief uses market code `DE` for Germany in places, but `markets.py` keys Germany under **`EU`** (XETRA), with no `DE` key — `get_market_config("DE")` would fall back to USD. The namespace→market convention must use `EU` (the markets.py key), or a `DE`→`EU` alias must be added in 50.3. For 50.1's scope (just the currency lookup) the fix is correct as written; the `DE`-vs-`EU` code reconciliation is a 50.3 universe-mapper concern, worth flagging in the contract.

### Q4 — FX consumers (who calls fx_rates) + their date types

**TWO consumers with DIFFERENT date semantics — the API must serve both:**

| Consumer | Function (file:line) | Mark source | Date type | FX need |
|----------|----------------------|-------------|-----------|---------|
| **50.2 paper_trader** | `mark_to_market()` `paper_trader.py:432-508` (NAV at `:480`, pnl at `:446`); also `execute_buy:221,245,253,275`, `execute_sell`, `_get_live_price:1124-1133` | LIVE mark via `_get_live_price(ticker)` — `yf.Ticker(t).history(period="1d")`, **no date arg** | "today" (live daily mark) | `get_fx_rate("EUR","USD")` for today's rate — a **live/cached** lookup |
| **50.5 backtest_trader** | `mark_to_market(date: str, prices: dict)` `backtest_trader.py:188-201`; NAV in `_compute_nav(prices)` `:233-238`; `close_all_positions(date, prices)` `:203`; snapshots keyed by `date` (`DailySnapshot.date`) | Historical close from cache, keyed by **`date: str`** ISO "YYYY-MM-DD" | **point-in-time** as of `date` | `get_fx_rate("EUR","USD", date)` for the rate AS OF that backtest day — a **BQ historical** lookup, point-in-time correct |

**Date type CONCLUSION:** both use **`str` ISO "YYYY-MM-DD"** (backtest `date` is a str; paper_trader has no date but "today" is naturally `date.today().isoformat()`). So the API signature should be:
```python
def get_fx_rate(from_ccy: str, to_ccy: str, date: str | None = None) -> float
```
where `date=None` ⇒ latest/live rate (paper_trader path), and `date="2024-03-15"` ⇒ point-in-time historical (backtest path). Accepting `str` matches both consumers with zero conversion. (If a `datetime`/`date` is ever passed, coerce with `.isoformat()[:10]` — but the two real consumers both speak `str`.)

**Critical correctness note:** the NAV math is currency-blind today (`backtest_trader.py:235` `pos.quantity * prices.get(...)`; `paper_trader.py:444` `pos["quantity"] * live_price`). 50.2/50.5 will wrap the per-position term with `* get_fx_rate(pos_currency, base_currency, date)`. So `fx_rates.py` must return `1.0` for same-currency (USD→USD) cheaply so the US-only path stays byte-identical (multimarket brief criterion: `["US"]` behavior unchanged). **Add an explicit `if from_ccy == to_ccy: return 1.0` short-circuit** before any lookup.

### Q5 — caching pattern + recommendation (BQ vs in-memory TTL vs both)

**Existing patterns:**
- **`backend/services/api_cache.py`** — thread-safe in-memory TTL cache, module-level singleton (`get_api_cache()` at `:109`), `get`/`set(key, value, ttl_seconds)`/`invalidate(glob)`/`stats`. Per-endpoint TTL registry `ENDPOINT_TTLS` at `:115-141` (e.g. `paper:status=60s`, `settings:models=3600s`, `paper:ticker_meta=86400s` 24h). This is the right tool for the LIVE daily FX mark.
- **`backend/backtest/cache.py`** — BQ bulk-preload cache (`preload_prices`/`preload_fundamentals` = 2 queries for an entire backtest; CLAUDE.md "always call `cache.preload_macro()`"). This is the right tool for the HISTORICAL FX series in a backtest (bulk-load the whole FX history once, then in-memory dict lookups during the day-loop).

**RECOMMENDATION: (c) BOTH — split by consumer, mirroring how prices already work:**
1. **BQ table `historical_fx_rates`** = the point-in-time historical source for backtests (50.5) and the durable backfill. Backtest preloads the FX series into memory once (mirror `cache.preload_prices`), then does dict lookups in the day-loop — NOT a BQ query per day (the per-day-query anti-pattern is exactly what `cache.py` exists to avoid).
2. **In-memory TTL cache for the LIVE daily mark** (50.2) via `api_cache` with a 24h-ish TTL (FX marks once/day; `paper:ticker_meta` at 86400s is the precedent). Key e.g. `fx:EUR:USD:2026-05-30`. On miss → yfinance live → (FRED fallback) → set. Optionally also persist the live mark into `historical_fx_rates` so today's rate becomes tomorrow's history (write-through), giving the backtest a continuous series without a separate backfill job.

This is the SAME split prices already use: `historical_prices` (BQ, backtest) + `_get_live_price` (live, paper). FX should not invent a new pattern.

**What `markets.py` exposes for currency already:** `MARKET_CONFIG[market]["currency"]` (`markets.py:21-52`) and `get_market_config(market)["currency"]` (`:75-78`). It does NOT expose any FX/conversion — only the static market→currency string. `fx_rates.py` is the missing layer that turns those currency codes into a rate. So `fx_rates.py` should import `markets` for the currency CODES but owns all rate logic.

### Q6 — markets.py currency map (reuse, don't duplicate)

`backend/backtest/markets.py:21-52` `MARKET_CONFIG` — the single source of truth for market→{exchange, currency, timezone, description}:
- US → {XNYS, **USD**, America/New_York}
- NO → {XOSL, NOK, Europe/Oslo}
- CA → {XTSE, CAD, America/Toronto}
- EU → {XETR, **EUR**, Europe/Berlin}  ← Germany/XETRA is the `EU` key
- KR → {XKRX, **KRW**, Asia/Seoul}

Accessors: `parse_namespaced_ticker(t)` (`:55-72`), `get_market_config(market)` (`:75-78`), `get_trading_calendar(market)` (`:81-102`, uses `exchange_calendars` — **confirmed installed, v4.13.2**), `is_trading_day(date, market)` (`:105-120`).

**`fx_rates.py` should expose a thin `market_currency(market) -> str` convenience that delegates to `markets.get_market_config(market)["currency"]`** so callers (paper_trader, backtest_trader) can go market→currency→rate without re-importing markets everywhere. Do NOT duplicate the currency strings.

**Confirmed installed deps (`.venv`):** `exchange_calendars 4.13.2`, `yfinance 1.2.0`, `pandas 3.0.1`. No new pip deps needed for 50.1 (yfinance + httpx + google-cloud-bigquery all present).

## External research

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://blog.quantinsti.com/download-forex-price-data-yfinance-library-python/ | 2026-05-30 | blog (quant practitioner) | WebFetch full | yfinance FX ticker = `EURUSD=X`/`GBPUSD=X` with `=X` suffix; download via `yf.download(ticker, ...)` (same as stocks), daily + intraday via `interval`. Confirms the `=X` forex convention and that `yf.download` is the right method. |
| https://fred.stlouisfed.org/series/DEXUSEU/ (metadata via FRED API) | 2026-05-30 | official (Federal Reserve H.10) | FRED API (page 403'd WebFetch; pulled `/fred/series` JSON) | **Verbatim units: "U.S. Dollars to One Euro"** → USD per 1 EUR (≈1.16). Daily, source Board of Governors, **start 1999-01-04**, "noon buying rates NYC". Direction MATCHES yfinance `EURUSD=X`. |
| https://fred.stlouisfed.org/series/DEXKOUS (metadata via FRED API) | 2026-05-30 | official (Federal Reserve H.10) | FRED API JSON | **Verbatim units: "South Korean Won to One U.S. Dollar"** → KRW per 1 USD (won-per-dollar). Daily, **start 1981-04-13** (deepest history). Direction MATCHES yfinance `KRW=X`/`USDKRW=X`. |
| https://insights.glassnode.com/why-use-point-in-time-data/ | 2026-05-30 | industry (data vendor) | WebFetch full | Look-ahead bias = backtest using info unavailable at decision time. "A value you see today for January 15 2024 may not be the value published on January 15 2024." Rule: **PiT data is append-only + immutable; each point reflects only what was known when first computed.** As-of querying + validate revised-vs-PiT to expose the gap. |
| https://www.ecb.europa.eu/.../euro_reference_exchange_rates/...index.en.html | 2026-05-30 | official (ECB) | WebFetch full | ECB euro reference rates: **foreign-ccy per 1 EUR** (e.g. USD 1.1644/EUR), fixed ~16:00 CET (concertation ~14:10), **business days only, NOT on weekends/TARGET holidays** (the weekend-gap problem, confirmed by an authority). Free for info; "using for transaction purposes strongly discouraged." |
| https://www.kantox.com/glossary/wmreuters-benchmark-rates | 2026-05-30 | industry (FX treasury vendor) | WebFetch full | WM/Reuters fix = daily 4pm London; **volume-weighted median of trades+order-book in a 5-min window** (mid, not simple bid-offer midpoint). "Widely embedded in fund valuations, custodian reports" — the institutional **mark-to-market / NAV** FX standard. Covers 150+ ISO-4217 pairs. |
| https://sharpely.in/blog/bias-free-backtesting-explained... | 2026-05-30 | industry (backtest platform) | WebFetch full | PiT controls: "data added only after officially reported"; decisions on rebalance day but **executed at next-day prices**; delisted names sold at last available price; historical index membership. Corroborates glassnode PiT + the forward-fill/last-available convention. |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://finance.yahoo.com/quote/KRW=X/ | data (Yahoo) | Search snippet confirms `KRW=X` page title = "USD/KRW" (won-per-dollar); direction question answered without full fetch. |
| https://finance.yahoo.com/quote/USDKRW=X/ | data (Yahoo) | Snippet: `USDKRW=X` = same USD/KRW pair (alternate ticker). |
| https://finance.yahoo.com/quote/KRWUSD=X/ | data (Yahoo) | Snippet: `KRWUSD=X` = the INVERSE (KRW/USD). Confirms which ticker to AVOID for the won-per-dollar convention. |
| https://www.lseg.com/content/dam/ftse-russell/.../wmr-fx-methodology.pdf | doc (LSEG/WMR) | PDF; the Kantox glossary (read in full) already gives the mid-rate + 4pm-fix method authoritatively. |
| https://www.cmegroup.com/articles/case-study/...wm-refinitiv-400-pm-fixing-rate.html | industry | Snippet: asset-manager exposure to the 4pm fix; reinforces WM/R as the valuation standard. |
| https://analystprep.com/study-notes/cfa-level-2/problems-in-backtesting/ | edu (CFA) | Snippet: backtest biases taxonomy (look-ahead/survivorship); glassnode + sharpely full reads cover it. |
| https://mikeharrisny.medium.com/look-ahead-bias-in-backtests-and-how-to-detect-it... | blog | Snippet: detection of look-ahead bias; corroborates PiT. |
| https://arxiv.org/pdf/2601.13770 | paper (preprint) | "Look-Ahead-Bench" 2026 PiT-LLM benchmark; recency signal that PiT is an active 2026 research topic; binary PDF, not needed in full for the FX-storage decision. |
| https://aws.amazon.com/marketplace/pp/prodview-4ztvijzvvllpa | vendor | Snippet: DEXUSEU redistributed on AWS Marketplace; confirms the series is a recognized canonical feed. |
| https://blog.quantinsti.com/... (recency variant) | blog | Same domain as the read-in-full source. |

**URLs collected (unique):** 16+ (7 read-in-full + 9 snippet-only above; additional search hits not listed).

### Search-query variants run (3-variant discipline)

1. **Current-year frontier:** the recency scan below used 2026-scoped FX/PiT queries (FRED `DEXKOUS` end date 2026-05-22; "Look-Ahead-Bench" arXiv 2601 = 2026; WM/Reuters methodology v30).
2. **Last-2-year window:** WM/Reuters reform 2013-2015 + current methodology, glassnode/sharpely PiT pieces (2024-2026 vintage).
3. **Year-less canonical:** "point-in-time historical FX rates backtest look-ahead bias" (→ glassnode + sharpely + CFA), "FX quote convention base currency mid rate" (→ ECB + WM/Reuters), "yfinance currency pair ticker EURUSD=X KRW=X direction" (→ quantinsti + Yahoo pages). The year-less queries surfaced the canonical PiT and FX-convention authorities.

---

## Recency scan (2024-2026)

Searched the last-2-year window on FX sourcing + point-in-time backtest correctness. **Findings (all COMPLEMENT, none supersede):**
1. **FRED FX series are live as of 2026-05-22** (DEXUSEU/DEXKOUS both updated through last week) — the free FRED fallback is current and unbroken; deepest history (DEXKOUS to 1981, DEXUSEU to 1999) far exceeds the project's 2018/2022 backtest start, so FRED can fully backfill if yfinance FX history is short.
2. **yfinance reliability degradation (2024-2026)** carried from the multimarket brief — applies to FX too (rate-limits/IP-bans), which RAISES the weight on the FRED fallback for FX specifically. FX is only 2-3 pairs/day so the live-mark risk is low, but the backfill (one bulk call) is where a yfinance failure would bite → FRED backfill is the hedge.
3. **"Look-Ahead-Bench" (arXiv 2601.13770, 2026)** — a 2026 benchmark formalizing PiT look-ahead bias in finance LLMs; confirms PiT correctness is an active 2026 concern, not settled folklore. No change to the storage design (append-only, as-of query), just firmer backing.
4. **WM/Reuters methodology v30** (current LSEG ground-rules) — the 4pm-London mid fix remains THE institutional NAV/valuation FX standard; no 2024-2026 change that affects a paper-trading daily mark (we don't need intraday-fix precision; a daily mid close is sufficient).
**No 2024-2026 finding contradicts the design below.**

### Consensus vs debate (external)
- **Consensus:** (a) `EURUSD=X` and `KRW=X` are the correct yfinance tickers and their direction matches the FRED H.10 series exactly (USD/EUR and KRW/USD); (b) store historical FX point-in-time (append-only, immutable, as-of query) to avoid look-ahead bias; (c) use the **mid rate** for portfolio valuation/NAV (WM/Reuters mid is the institutional standard); (d) FX has weekend/holiday gaps that must be forward-filled (last-available) for valuation — confirmed by ECB (no weekend rates) + sharpely (delisted/missing → last available price).
- **Debate / caution:** none material for paper-trading scope. The only nuance is intraday-fix precision (WM/R 4pm vs a daily close) — irrelevant for a once-daily paper mark; a daily mid close is the right granularity.

### Pitfalls (from literature)
1. **Direction inversion** — `KRWUSD=X` (inverse) vs `KRW=X`/`USDKRW=X` (won-per-dollar). Picking the wrong ticker silently inverts every KRW valuation. MITIGATION: store the pair with an explicit `base`/`quote` convention + assert magnitude (KRW/USD ≈ 1300, EUR rate ≈ 1.16) in a sanity check.
2. **Weekend/holiday FX gaps** — FX doesn't trade Sat/Sun and skips currency-specific holidays; a backtest day or a Monday mark may have no fresh rate. MITIGATION: forward-fill the last available rate (the standard valuation convention; ECB publishes none on weekends, sharpely uses "last available price").
3. **Look-ahead via revised data** — less acute for FX than fundamentals (FX spot is rarely revised), but the append-only/as-of discipline still applies: a backtest as of date D must read the rate stored for D (or the last ≤ D), never a future rate.
4. **Bid/ask vs mid** — using a bid or ask instead of mid biases NAV; use mid for valuation (WM/Reuters). yfinance FX close is effectively a mid-ish daily close — acceptable for paper; document it.

## SYNTHESIS — the deliverable (concrete enough to write contract + code)

### (a) Q1-Q6 answers — see "Internal code inventory" above (each with file:line).

### (b) Recommended `fx_rates.py` API + storage design

**Ticker / direction decision (locked):**
- **EUR/USD:** yfinance `EURUSD=X` → USD per 1 EUR (≈1.16). FRED fallback `DEXUSEU` ("U.S. Dollars to One Euro") — **same direction**, no inversion.
- **KRW/USD:** yfinance `KRW=X` (≡ `USDKRW=X`) → KRW per 1 USD (≈1300). FRED fallback `DEXKOUS` ("South Korean Won to One U.S. Dollar") — **same direction**. **Do NOT use `KRWUSD=X`** (that's the inverse).
- Store one canonical convention in the table: a `pair` like `"EURUSD"` meaning "units of QUOTE(USD) per 1 BASE(EUR)" — i.e. `pair = BASE+QUOTE`, value = quote-per-base. Then `EURUSD`=1.16 (USD per EUR), `USDKRW`=1300 (KRW per USD). `get_fx_rate(from, to)` derives the right pair + inverts if needed.

**Conversion semantics (the function contract):**
```python
def get_fx_rate(from_ccy: str, to_ccy: str, date: str | None = None) -> float:
    """Units of `to_ccy` per 1 `from_ccy`, as of `date` (ISO 'YYYY-MM-DD').
    date=None -> latest/live mark. Returns 1.0 when from_ccy == to_ccy."""
```
- `get_fx_rate("EUR","USD")` → 1.16 (1 EUR = 1.16 USD). To value a €100 position in USD base: `100 * get_fx_rate("EUR","USD") = $116`. This matches the multimarket brief's `market_value_base = local_value * fx_rate` (50.2.2).
- **`if from_ccy == to_ccy: return 1.0`** short-circuit FIRST (keeps the US-only path byte-identical — multimarket criterion).
- Inversion: store `USDKRW`=1300; `get_fx_rate("KRW","USD")` returns `1/1300`. Provide both directions via one stored pair + reciprocal.
- Convenience: `market_currency(market) -> str` delegating to `markets.get_market_config(market)["currency"]` (don't duplicate the map — Q6).

**Storage = BOTH (c), mirroring how prices already split BQ-historical + live):**
1. **BQ table `historical_fx_rates`** in `financial_reports` (sibling of `historical_prices`; Q1). Schema (mirror `historical_macro` — unpartitioned, tiny):
   ```
   pair         STRING   REQUIRED   -- "EURUSD" | "USDKRW" (BASE+QUOTE, value = quote per base)
   date         STRING   REQUIRED   -- "YYYY-MM-DD" (STRING to match historical_prices.date)
   rate         FLOAT64  NULLABLE   -- mid daily close, quote-per-base
   source       STRING   NULLABLE   -- "yfinance" | "fred" (provenance; mirrors data_source_events spirit)
   ingested_at  TIMESTAMP NULLABLE
   ```
   Uniqueness on `(pair, date)` (same as `historical_prices` `(ticker,date)`). NO partition (size-class = `historical_macro`).
2. **Live daily mark** via `api_cache` (Q5): key `fx:{pair}:{date}` (e.g. `fx:EURUSD:2026-05-30`), TTL ~24h (precedent `paper:ticker_meta`=86400s). On miss → yfinance `yf.Ticker("EURUSD=X").history(period="1d")` (mirror `_get_live_price`) → FRED fallback → set. **Write-through:** persist the live mark into `historical_fx_rates` so today's rate becomes tomorrow's history → continuous backtest series without a separate daily backfill job.
3. **Backtest path:** bulk-preload the FX series once per backtest (mirror `cache.preload_prices`), then in-memory dict lookups in the day-loop — never a BQ query per day.

**Fetch with FRED fallback (mirror existing idioms):**
- Primary: `yf.download(["EURUSD=X","KRW=X"], start=, end=, ...)` (backfill) / `yf.Ticker(...).history(period="1d")` (live) — try/except → log → return None, per `data_ingestion.py:103-114` + `_get_live_price:1126-1133`. No exponential backoff exists for yfinance in-repo; match that.
- Fallback: FRED `DEXUSEU`/`DEXKOUS` via `httpx` exactly like `data_ingestion.ingest_macro` (`:286-296`) — `FRED_BASE` + `series_id` + `api_key` + `observation_start/end`. FRED key already in `backend/.env` (confirmed present). FRED gives daily history to 1999/1981 → covers any backtest start.
- **Forward-fill on read:** `get_fx_rate(..., date)` returns the rate for `date`, else the **last available rate ≤ date** (weekend/holiday gap handling — ECB publishes no weekend rates; sharpely "last available price"). Implement as `WHERE pair=? AND date<=? ORDER BY date DESC LIMIT 1` (point-in-time, as-of query — glassnode immutability rule) or a forward-fill over the preloaded dict.

**Migration:** NEW `scripts/migrations/create_historical_fx_rates_table.py` mirroring `create_data_source_events_table.py` (DDL `CREATE TABLE IF NOT EXISTS`, `--apply` dry-run default, `--verbose`, OPTIONS descriptions). ALSO add `("historical_fx_rates", REF, SCHEMA)` to `migrate_backtest_data.ALL_TABLES` so `_ensure_tables_exist` auto-creates it (defense-in-depth; matches prices/fundamentals/macro). BQ client uses default location (no `--location` pin) so `financial_reports`/us-central1 resolves correctly.

### (c) The exact `data_ingestion.py:146` fix

Replace (`data_ingestion.py:146`):
```python
"currency": "USD" if market == "US" else "USD",  # TODO: lookup from MARKET_CONFIG
```
with (add `from backend.backtest import markets` at module top — `markets.py` is a sibling module, no circular-import risk since `markets.py` imports only `exchange_calendars`/`logging`):
```python
"currency": markets.get_market_config(market)["currency"],
```
- `market` here is the namespace code from `:137-141` (US default; `"DE:BAS"`→`"DE"`).
- `get_market_config` (`markets.py:75-78`) uppercases + falls back to US for unknown → can't crash.
- Mapping: US→USD, EU→EUR, KR→KRW (`markets.py:24,42,48`).
- **FLAG for the contract (DE-vs-EU):** `markets.py` keys Germany as **`EU`** (XETRA), with NO `DE` key — so a `"DE:..."` namespace falls back to USD. For 50.1 the line-146 fix is correct as written (it reads whatever the market code maps to); the namespace-code reconciliation (use `EU`, or add a `DE`→`EU` alias) is a **50.3 universe-mapper** concern, not 50.1. Note it so the contract scopes 50.1 to the lookup only.

### (d) External-source-backed FX-handling best practices (applied)
1. **Point-in-time / as-of, append-only storage** (glassnode + sharpely + arXiv 2601.13770) → `historical_fx_rates` is append-only; `get_fx_rate(..., date)` reads the rate AS OF `date` (`WHERE date<=? ORDER BY date DESC LIMIT 1`), never a future rate. A backtest on 2024-03-15 sees only ≤2024-03-15 FX. **Direct attack on look-ahead bias** — the project's core backtest-correctness doctrine (backend-backtest.md "No future leakage").
2. **Forward-fill weekend/holiday gaps with last-available rate** (ECB: no weekend/TARGET-holiday rates; sharpely: last available price) → the as-of query naturally forward-fills (last ≤ date). A Monday or a German-holiday mark reuses Friday's rate rather than NULL.
3. **Mid rate for valuation/NAV** (WM/Reuters via Kantox: volume-weighted median mid is THE fund-NAV standard) → use yfinance/FRED daily close as the mid for the paper daily mark (granularity sufficient for once-daily paper; document that we approximate the WM/R 4pm mid with a daily close). NEVER bid or ask.

### (e) Application mapping (external → internal file:line)
- glassnode/sharpely PiT → `fx_rates.get_fx_rate(date)` as-of query feeding `backtest_trader.mark_to_market(date, prices)` (`backtest_trader.py:188`) where NAV is currency-blind today (`_compute_nav:233-238`).
- WM/Reuters mid → `fx_rates` daily-close mark feeding `paper_trader.mark_to_market` (`paper_trader.py:432`, NAV `:480`) — wraps the `* live_price` term (`:444`) with `* fx_rate` in 50.2.
- FRED DEXUSEU/DEXKOUS fallback → mirror `data_ingestion.ingest_macro` httpx (`:286-296`); key in `backend/.env`.
- markets.py currency map → both the `:146` fix AND `fx_rates.market_currency()` delegate to `markets.get_market_config` (`markets.py:75-78`).

---

## Research Gate Checklist

Hard blockers — all satisfied:
- [x] >=5 authoritative external sources READ IN FULL (7: quantinsti yfinance, FRED DEXUSEU API-meta, FRED DEXKOUS API-meta, glassnode PiT, ECB reference rates, Kantox WM/Reuters, sharpely PiT). Source hierarchy honored (2 official Fed/ECB, 3 industry, 2 practitioner blog).
- [x] 10+ unique URLs total (16+ incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported (FRED live to 2026-05-22; arXiv 2601 PiT 2026; yfinance degradation)
- [x] Full pages/series-metadata read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (Q1-Q6)

Soft checks:
- [x] Internal exploration covered: data_ingestion, markets, migrations (2 idioms), screener+all yfinance sites, fred_data, api_cache, backtest/cache, paper_trader (NAV/cost/pnl/_get_live_price), backtest_trader (NAV/mark_to_market), settings, bigquery_client dataset routing
- [x] Contradictions/consensus noted (ticker direction inversion risk; intraday-fix vs daily-close granularity resolved to daily-close-sufficient)
- [x] All claims cited per-claim with file:line or URL

## Research Gate JSON envelope

```json
{
  "tier": "moderate",
  "external_sources_read_in_full": 7,
  "snippet_only_sources": 9,
  "urls_collected": 16,
  "recency_scan_performed": true,
  "internal_files_inspected": 11,
  "gate_passed": true
}
```
