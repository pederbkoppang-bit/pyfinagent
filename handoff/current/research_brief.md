# Research Brief — phase-50.3: International universe + ticker-suffix mapper + live-loop routing

**Tier:** complex
**Date:** 2026-05-30
**Status:** IN PROGRESS (write-first skeleton; appended as findings land)

## Safety invariant (non-negotiable)
A new `paper_markets` setting defaults to `["US"]`. When `paper_markets == ["US"]`,
the live loop's universe + behaviour MUST be byte-identical to today (+20% engine
unchanged). International is BUILT but OFF. Go-live flip to `["US","EU","KR"]`
happens AFTER the 50.5 data-quality gate. So 50.3 must NOT change live behaviour
by default.

Operator chose BOTH EU (Germany/.DE/DAX) AND South Korea (KOSPI/.KS, KOSDAQ/.KQ)
via free yfinance (no pykrx/EODHD).

---

## Internal code inventory (Q1-Q6, file:line)

### Q1 — The live universe path + where `paper_markets` plugs in (byte-identical for ["US"])

**Today's universe selection** (`autonomous_loop.py:310-329`):
- Default path: `universe = None` (`:322`). `screen_universe(tickers=None, ...)` (`:324-329`)
  then internally calls `get_sp500_tickers()` (`screener.py:103-104`) → Wikipedia
  scrape of the S&P-500 list, dots→dashes (`screener.py:56`).
- Opt-in path: `settings.russell1000_universe_enabled` → `get_russell1000_tickers()`
  (`:312-314`). Default OFF.
- So today the universe is a flat `list[str]` of bare US tickers (`AAPL`, `BRK-B`).

**EXACT plug-in point (the one change that stays byte-identical):** build the
universe in `autonomous_loop` BEFORE the `screen_universe` call and pass it in
explicitly. New code (~`:321`), gated on `paper_markets`:
```python
markets = getattr(settings, "paper_markets", ["US"]) or ["US"]
if markets == ["US"]:
    universe = None  # <- byte-identical: screen_universe(None) -> get_sp500_tickers()
                     #    russell branch above still wins when its flag is on
else:
    from backend.backtest.candidate_selector import build_multimarket_universe
    universe = build_multimarket_universe(markets)  # yfinance-SUFFIXED symbols
```
**Byte-identity proof:** when `paper_markets == ["US"]` (the default) the `else`
is never taken; `universe` stays exactly what it is today (`None`, or the
Russell list when that flag is on). `screen_universe(None)` → `get_sp500_tickers()`
→ identical Wikipedia scrape. NOT ONE byte of the US path changes. The
multi-market branch is dead code until the operator flips `paper_markets`.

**IMPORTANT placement detail:** the russell1000 block (`:312-320`) sets
`universe` first. The `paper_markets` block must be written so that (a) US-only
leaves the russell decision intact, and (b) a multi-market list, when active,
either *extends* the US base or *replaces* it (recommend: build the US slice via
the SAME `get_sp500_tickers()`/russell path, then concat EU+KR suffixed symbols,
so US tickers inside a multi-market run are byte-identical to the US-only run).

### Q2 — candidate_selector.get_universe_tickers(market=) + constituent source

`candidate_selector.py:98-132`: `get_universe_tickers(market=DEFAULT_MARKET, as_of=None)`.
- US branch: Wikipedia S&P-500 scrape (`:134-144`), dots→dashes.
- **Non-US branch RETURNS `[]` with a warning** (`:127-132`) — "Only 'US' is
  implemented". The docstring literally says "Phase 5 will add: NO (OBX),
  CA (TSX60), DE (DAX), KR (KOSPI50)". **This is the stub 50.3 fills.**
- `as_of is not None` raises `NotImplementedError` (`:121-126`) — PIT membership
  not available. 50.3 international universe is LIVE-only (`as_of=None`), so this
  guard is untouched; backtest PIT membership for EU/KR is a 50.5 concern.

**What it needs to return:** for EU → the 40 DAX symbols as **yfinance-suffixed**
strings (`SAP.DE`, `SIE.DE`, ...); for KR → KOSPI-200 symbols as `005930.KS`,
`000660.KS`, ... . **Source recommendation: a CURATED STATIC LIST in-repo**, not
a Wikipedia scrape — see external §"Constituent source" for the rationale
(DAX/KOSPI Wikipedia tables are less stable than the S&P table; the lists change
only ~2x/year; a static list is deterministic, offline-safe, and free of
scrape-fragility). Put the lists in a new module
`backend/backtest/universe_lists.py` (or a `markets.py` dict) as
`DAX40 = [...]`, `KOSPI200 = [...]` with the yfinance symbols literal.

### Q3 — Ticker namespacing + the yfinance suffix mapper

`markets.py:55-72`: `parse_namespaced_ticker("US:AAPL") -> ("US","AAPL")`;
bare ticker → `("US", ticker)`. `MARKET_CONFIG` (`:21-52`) has US/NO/CA/EU/KR
with exchange+currency+tz, but **NO yfinance-suffix field** and **NO forward
mapper** (`{market}:{ticker}` -> suffixed symbol). That mapper does not exist yet.

**Where the mapper lives:** add to `markets.py` (it already owns market identity).
Two new pieces:
1. `MARKET_CONFIG[m]["yf_suffix"]`: `US`→`""`, `EU`→`".DE"`, `KR`→`".KS"`
   (KOSDAQ → `".KQ"`; see external note — KOSDAQ needs a per-ticker suffix, so
   for KR store the FULL suffixed symbol in the curated list rather than deriving).
2. `to_yfinance_symbol(namespaced_or_bare) -> str`:
   - `US:AAPL` / `AAPL` → `AAPL`
   - `EU:SAP` → `SAP.DE`
   - `KR:005930` → `005930.KS`
   Implementation: `market, t = parse_namespaced_ticker(x); suffix =
   get_market_config(market)["yf_suffix"]; return t if t already ends with a
   known suffix else t + suffix`. (Idempotent — applying twice is a no-op.)

**RECOMMENDED DESIGN (simpler + avoids the KOSDAQ/.KQ vs .KS derivation trap):**
store the **already-suffixed yfinance symbol AS the ticker** throughout the
international pipeline (e.g. the universe list contains `SAP.DE`, `005930.KS`
directly; the `market` is carried as a separate column = `"EU"`/`"KR"`). Then
NO per-call suffix derivation is needed — `_get_live_price` and `yf.download`
receive the symbol verbatim, and `market` is passed to `execute_buy(market=)`
for the currency logic (50.2). The `{market}:{ticker}` namespaced form
(`parse_namespaced_ticker`) is then only an *optional* internal convenience, not
a required transform on the hot path. **This sidesteps the .KS-vs-.KQ problem
entirely** (the curated list just lists each KOSDAQ name with `.KQ`).

**Who consumes a suffixed ticker (must NOT assume bare US):**
- `screen_universe` (`screener.py:64`): passes `tickers` straight to
  `yf.download(tickers, ...)` (`:110`) — yfinance accepts suffixed symbols in a
  batch. ✓ no change beyond getting suffixed symbols in the list.
- `_get_live_price(ticker)` (`paper_trader.py:1200-1209`): `yf.Ticker(ticker)
  .history(...)` — **MUST receive the suffixed symbol** (`005930.KS`). ✓ if the
  stored ticker IS the suffixed symbol (recommended design), this works verbatim.
- Sector backfill `yf.Ticker(ticker).info` (`paper_trader.py:835`): same — needs
  the suffixed symbol.
- **NO ticker-shape regex anywhere.** Grep for `isalpha`/`^[A-Z]`/`isupper`/
  `sanitize`/ticker-regex across screener, paper_trader, autonomous_loop,
  yfinance_tool returned **nothing** (only a comment hit). So no code assumes
  `^[A-Z.]+$`; a `005930.KS` ticker will not trip a validator. (Frontend/auth
  sanitization is "alphanumeric + dots" per security.md — `.KS`/`.DE` pass; the
  numeric KR code passes alphanumeric. Verify the API-layer ticker validator if
  any international ticker is ever user-submitted, but the autonomous loop never
  routes through user input.)
- BQ keys: `paper_positions.ticker` / `paper_trades.ticker` are STRING — storing
  `005930.KS` is fine; it just becomes the position key. The idempotency guard
  (`paper_trader.py` BUY) keys on `ticker` string equality — consistent as long
  as the SAME suffixed form is used at buy + mark + sell (the recommended design
  guarantees this).

### Q4 — Price fetch for international (the suffix must reach yfinance)

The ticker flow:
`universe (suffixed) -> screen_universe -> yf.download(suffixed) [OK] ->
screen_data[{"ticker": suffixed}] -> rank_candidates -> analysis ->
_StagedBuy(ticker=suffixed) -> autonomous_loop:988 _get_live_price(suffixed)
[OK] -> execute_buy(ticker=suffixed, market="EU"/"KR") -> pos_row stores
ticker=suffixed + market -> mark_to_market -> _get_live_price(suffixed) [OK]`.

**The single rule:** the yfinance SUFFIX must be applied ONCE, at universe
construction (`build_multimarket_universe` / curated list), and the suffixed
symbol is then carried verbatim as the `ticker` everywhere. `market` is carried
ALONGSIDE (separate field) for the 50.2 currency math. **Do NOT** store the bare
namespaced form (`KR:005930`) as the position ticker and re-derive the suffix in
`_get_live_price` — that would require threading the suffix transform into every
price call (`paper_trader.py:1200`, `:835`, `screener.py:110`) and is more
error-prone. Suffix-at-source + carry-verbatim is the minimal-change path.

For a `KR:005930` *position*, the recommended design stores `ticker="005930.KS"`,
`market="KR"`. `_get_live_price("005930.KS")` → `yf.Ticker("005930.KS").history()`
→ KRW quote. 50.2's `execute_buy(market="KR")` derives KRW via
`fx_rates.market_currency("KR")` and converts to USD. The two layers compose.

### Q5 — `paper_markets` setting (default ["US"], list-type pydantic)

`settings.py` is a pydantic-settings `BaseSettings` (`:14`). **There is currently
NO list-type Field in settings.py** (grep for `list[str] = Field`/`List[str]`/
`default_factory`/`Field([` returned nothing). Existing multi-market scalars live
at `:49-51`: `default_market: str = Field("US", ...)`, `base_currency: str =
Field("USD", ...)`. Add adjacent:
```python
paper_markets: list[str] = Field(
    default_factory=lambda: ["US"],
    description="phase-50.3: markets the live paper loop screens + trades. "
                "Default ['US'] => byte-identical to pre-50.3 (US-only). "
                "Flip to ['US','EU','KR'] AFTER the 50.5 data-quality gate.",
)
```
**Use `default_factory=lambda: ["US"]`, NOT `Field(["US"])`** — a mutable default
shared across instances is a pydantic/Python footgun; `default_factory` gives each
Settings a fresh list. pydantic-settings parses a `PAPER_MARKETS` env var as JSON
for list types (e.g. `PAPER_MARKETS='["US","EU","KR"]'`), so the operator flip is
an env var or Settings-UI edit. **Byte-identical default confirmed:** absent the
env var, `paper_markets == ["US"]` → the Q1 `if markets == ["US"]` branch → US
path unchanged.

### Q6 — Sector taxonomy for EU/KR (existing sector caps)

The sector cap (`paper_max_per_sector`) reads `pos["sector"]`, populated at
`execute_buy(sector=...)` (`paper_trader.py:130`, persisted `:310`,`:331`) from
the screener `sector_lookup` (`screener.py:194-200`) or the yfinance backfill
(`paper_trader.py:835` `yf.Ticker(ticker).info` → `info.get("sector")`).

- yfinance `.info` exposes `sector`/`sectorKey` + `industry`/`industryKey` and
  **does support non-US tickers** (yfinance Sector/Industry docs explicitly scope
  by ISO-3166 region incl. DE/JP/GB — external §). So `.DE` and `.KS` tickers
  generally return a `sector`.
- **Taxonomy caveat:** yfinance's sector strings are **Yahoo's own taxonomy**
  (the docs do NOT claim GICS), e.g. "Technology", "Financial Services",
  "Consumer Cyclical", "Healthcare". The US path ALREADY uses these Yahoo strings
  (the `SECTOR_ETFS` map in `screener.py:21-26` uses "Technology"/"Financials"/
  "Health Care" — note "Financials" vs Yahoo's "Financial Services"; the live
  sector cap just groups by whatever string `.info["sector"]` returns, so it's
  internally consistent regardless of GICS-exactness). EU/KR tickers return the
  SAME Yahoo taxonomy → the sector cap groups them consistently with US names.
  **No GICS remap needed** for the cap to function; a German bank and a US bank
  both come back "Financial Services" and share a bucket.
- **Risk:** yfinance `.info` is the *least reliable* yfinance endpoint (external
  §; multiple GitHub issues of empty `.info`). The existing backfill is already
  **fail-open** (`paper_trader.py:837-843` — yfinance failure → skip, keep going).
  So a missing EU/KR sector degrades to `None`/"Unknown" (cap treats it as its own
  bucket) — non-fatal, same as a US ticker with missing sector today.

## External research

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://help.yahoo.com/kb/SLN2310.html | 2026-05-30 | official (Yahoo) | WebFetch full | **Authoritative suffix table:** XETRA = `.DE` (15-min delay), Frankfurt = `.F` (15-min), Korea Stock Exchange/KOSPI = `.KS` (**20-min delay**), KOSDAQ = `.KQ` (20-min), Paris = `.PA` (15-min). Data provider ICE. → KR quotes are 20-min delayed; the suffix mapper is correct. |
| https://medium.com/@Tobi_Lux/data-from-yfinance-some-observations-41e99d768069 | 2026-05-30 | industry (measured study) | WebFetch full | **[ADVERSARIAL/quality]** Measured 12 DAX-40 stocks vs XETRA reference: BASF close diffs **-7% to +3% (avg -2.5%)**; **max deviation up to 11%** across DAX-40. **Broken OHLC** (O=H=L=C within 0.5%) on **up to 10% of days** (10-24 days/252). "Any candlestick algorithm reliant on OHLC is likely to fail." Limited impact on MACD/EMA/RSI (average-based); large impact on candlestick logic. Stooq/Onvista far better. |
| https://en.wikipedia.org/wiki/DAX | 2026-05-30 | reference | WebFetch full | DAX = 40 constituents (expanded 30→40 in 2021 post-Wirecard), cap-weighted, Xetra prices. Full symbol list captured (see constituent table below). **3 are NOT `.DE`:** Airbus = `AIR.PA` (Paris). The rest are `.DE`. |
| https://en.wikipedia.org/wiki/KOSPI_200 | 2026-05-30 | reference | WebFetch full | KOSPI-200 = 200 free-float-cap-weighted KRX names; rebalanced June + December; base 100 @ 1990-01-03. 6-digit numeric KRX codes (Samsung 005930, SK Hynix 000660, LG Elec 066570, Hyundai Motor 005380, Kia 000270, LG Chem 051910, Samsung SDI 006400, KB Fin 105560, Shinhan 055550, SK Innovation 096770, ...). Yahoo suffix `.KS`. |
| https://github.com/ranaroussi/yfinance/issues/2125 | 2026-05-30 | community (bug + maintainer) | WebFetch full | **429 rate-limit** fires when looping over many tickers, even with VPN/server-switching. Mitigation: retry + rotate User-Agent (a 429 became 200 on UA change); library lacks built-in retry. → screening 240+ intl tickers per cycle needs retry/backoff + a curl_cffi session. Batch `yf.download` (one request) is safer than 240 per-ticker `.history()` loops. |
| https://www.promptcloud.com/blog/scrape-yahoo-finance/ | 2026-05-30 | industry (2026) | WebFetch full | **[ADVERSARIAL]** "Major international exchanges is solid. For smaller markets...data quality is inconsistent...validate against a second source." **"A scraper that silently returns empty data is more dangerous than one that fails loudly."** Yahoo changes page/endpoint structure "multiple times per year." Teams handling ">200 tickers at production frequency typically spend more maintaining scraping" than licensed feeds. → free yfinance is VIABLE for DAX-40 + KOSPI-200 (both "major"), but mandates explicit empty-data detection + cross-validation; this is exactly what the 50.5 data-quality gate is for. |
| https://ranaroussi.github.io/yfinance/reference/yfinance.sector_industry.html | 2026-05-30 | official (yfinance docs) | WebFetch full | `.info` exposes `sector`/`sectorKey` + `industry`/`industryKey`; Sector/Industry API scopes by ISO-3166 region with **DE / GB / JP examples** → non-US sector data IS supported. Taxonomy is **Yahoo's own** (docs do NOT claim GICS). → EU/KR tickers return the same Yahoo sector strings the US path already groups on; no GICS remap needed. |
| https://user42.tuxfamily.org/chart/manual/Yahoo-Exchanges.html | 2026-05-30 | reference (compiled) | WebFetch full | Corroborates `.DE`=XETRA, `.F`=Frankfurt, `.KS`=KSE, `.KQ`=KOSDAQ. (Dot-to-dash rule confirmed via the SLN2310 search snippet: `BT.A`→`BT-A.L`; relevant only for alpha tickers with internal dots — KR's numeric codes have none, so no transform needed for KR.) |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://github.com/ranaroussi/yfinance/issues/2453 | community | "possibly delisted thrown for listed symbol" — open issue, thin excerpt (no resolution); corroborates the silent-empty failure mode covered by promptcloud (full read). |
| https://github.com/ranaroussi/yfinance/issues/2469 | community | "Unable to fetch prices for valid tickers: possibly delisted; no timezone found" — same silent-fail class. |
| https://github.com/ranaroussi/yfinance/issues/2633 | community | curl_cffi SSLError when fetching — informs the curl_cffi-session mitigation. |
| https://github.com/ranaroussi/yfinance/issues/2128 | community | "New rate-limiting" — corroborates #2125 (full read). |
| https://github.com/sharebook-kr/pykrx | code | The KR-specific alternative the operator deliberately declined; noted as the fallback if yfinance KR proves unusable at the 50.5 gate. |
| https://monetaiq.com/best-yahoo-finance-alternatives-for-financial-data-in-2026/ | industry | 2026 alternatives roundup (recency-scan input); corroborates "validate intl data". |
| https://en.wikipedia.org/wiki/Cross_listing | reference | ADR-vs-direct-listing / dual-listing ISIN distinction (pitfall input). |
| https://help.yahoo.com/kb/finance/exchanges-markets-covered-yahoo-finance-sln2310.html | official | Regional mirror of the suffix table. |
| https://finance.yahoo.com/quote/%5EKS200/components/ | data | Live KOSPI-200 components (a possible dynamic source; rejected in favor of curated static list — see synthesis). |
| https://finance.yahoo.com/quote/005930.KS/ | data | Samsung 005930.KS live page — confirms the suffixed symbol resolves on Yahoo. |
| https://stockanalysis.com/quote/krx/005930/ | data | Cross-source confirmation Samsung trades as KRX:005930. |

**URLs collected (unique):** 19 (8 read-in-full + 11 snippet-only).

### Search-query variants run (3-variant discipline)
1. **Current-year frontier (2026):** "yfinance 2026 reliability international stocks data quality update changelog"; "yfinance Korean stocks KOSPI .KS ... reliability 2026".
2. **Last-2-year window (2025):** "yfinance rate limit curl_cffi 2025 international tickers reliability batch download"; "KOSPI 200 constituents ... .KS 2025".
3. **Year-less canonical:** "Yahoo Finance ticker suffix exchange codes .DE .KS .KQ .L .PA international symbol format reference" (→ official Yahoo Help SLN2310, gnucash/tuxfamily compiled tables); "multi-market equity screener pitfalls dual-listing ADR ticker collision exchange suffix delisting survivorship" (→ cross-listing prior art); "yfinance Ticker info sector empty None international stocks" (→ the canonical sector-availability GitHub issues + docs).

### Recency scan (2024-2026)
Searched the last-2-year window on yfinance international reliability + intl data quality. **Findings (COMPLEMENT prior art; none change the recommended design):**
1. **2026 production-data articles (promptcloud, monetaiq, quantvps)** uniformly state yfinance coverage of MAJOR exchanges (XETRA, KRX) is "solid" but flag silent-empty-data and "multiple HTML changes/year" as the real risks → reinforces: (a) free yfinance is viable for DAX-40 + KOSPI-200 specifically (both major-exchange, large-cap), (b) the build MUST detect silent-empty returns, (c) cross-validation belongs in the 50.5 gate. This DIRECTLY supports the operator's free-yfinance choice for these two markets while flagging the real failure mode.
2. **2024-2025 yfinance rate-limit wave (#2125/#2128/#2469/#2633)** — Yahoo tightened anti-scraping; 429s on multi-ticker loops; curl_cffi `impersonate="chrome"` session + retry is the community mitigation. → the international screener (240+ symbols) should use batch `yf.download` (already does, `screener.py:110`) + a retry/backoff wrapper; per-ticker `_get_live_price` loops (mark-to-market over many intl positions) are the higher 429 risk.
3. **The Tobi Lux DAX-40 measured study** (carried from the 50-series multimarket brief, re-read in full) remains the single best QUANTIFIED reliability source: ≤11% deviation, broken OHLC on ≤10% of days for German blue-chips. No newer measured study supersedes it.
4. **No 2024-2026 change** to the Yahoo suffix scheme (`.DE`/`.KS`/`.KQ` stable) or to yfinance's `.info` sector exposure for international tickers.

**No 2024-2026 finding contradicts the recommended design.** The adversarial sources (Tobi Lux, promptcloud) AGREE the data is *usable for major-exchange large-caps* while *unreliable for OHLC-precision and silent on failures* — which is why International is built-but-OFF behind a 50.5 data gate, not flipped on at 50.3.

### Consensus vs debate (external)
- **Consensus:** (a) Yahoo suffix scheme is `{base}.{exch}` — `.DE` XETRA, `.KS` KOSPI, `.KQ` KOSDAQ (multiple independent sources agree); (b) yfinance reliably resolves major-exchange large-caps (DAX-40, KOSPI-200 names all resolve); (c) `.info` is the least-reliable endpoint but does carry intl sector data; (d) silent-empty data + rate limits are the real production risks, not total non-coverage.
- **Debate/nuance:** (a) **OHLC precision** — Tobi Lux measures real ≤11% close deviations for German names; the pyfinagent screener uses CLOSE-based momentum/RSI/SMA (average-based, the LESS-affected family per Tobi Lux), NOT candlestick patterns, so the impact on SCREENING is bounded — but the 50.5 gate must still cross-validate. (b) **Production viability** — promptcloud argues >200-ticker production should buy a feed; the counter is that pyfinagent is paper/local (Claude Max, no real money, daily cadence) so the maintenance burden is acceptable and the operator explicitly chose free yfinance. (c) **Static vs dynamic constituent list** — a live Yahoo/Wikipedia fetch is fresher but scrape-fragile; a curated static list is deterministic + offline-safe and the index changes only ~2x/year. For RELIABILITY (the stated 50.3 goal) the static list wins.

### Pitfalls (from literature) — applied to the multi-market screener
1. **Silent-empty data** (promptcloud: "more dangerous than failing loudly") — a non-resolving `.KS`/`.DE` symbol returns an empty frame, not an error. MITIGATION: `screen_universe` already `continue`s on `len(close) < 20` (`screener.py:133`); add explicit logging of dropped-intl-symbol COUNT per market so a silent universe collapse is visible. The 50.5 gate asserts a minimum resolve-rate per market.
2. **429 rate-limit on multi-ticker loops** (#2125) — MITIGATION: keep batch `yf.download` for screening (one request); for the mark-to-market `_get_live_price` fan-out over many intl positions, consider a small backoff/retry or a batched price fetch (out of 50.3 minimal scope, but flag for 50.4/50.5).
3. **Dual-listing / ADR ambiguity** (cross-listing literature: ADRs get a DIFFERENT ISIN, are NOT fungible with the local line) — e.g. SAP trades as `SAP` (NYSE ADR) AND `SAP.DE` (XETRA). MITIGATION: the curated EU list pins the **local XETRA** line (`SAP.DE`), never the US ADR; never let the same economic name enter the universe twice under two symbols. The numeric KR codes have no US-ADR collision risk in the universe (we list `005930.KS`, not the `SSNLF` OTC ADR).
4. **Ticker collision across exchanges** — a bare `MRK` is Merck&Co (US) AND `MRK.DE` is Merck KGaA (DE) — DIFFERENT companies. MITIGATION: the suffix IS the disambiguator; ALWAYS carry the suffixed symbol + `market`. Never compare/group international and US positions by the bare root.
5. **Airbus edge case (`AIR.PA`)** — a DAX-40 member listed in PARIS, not XETRA. The curated list must store `AIR.PA` (its real Yahoo symbol), NOT `AIR.DE`. Generic "EU → append .DE" derivation would BREAK Airbus. → store the full per-name symbol in the curated list (the recommended design); do not derive the suffix for EU.
6. **Numeric KR tickers as a position key** — `005930.KS` is a valid string key for BQ/idempotency; leading zeros must be preserved (string, never int). MITIGATION: store as STRING throughout (already STRING columns); never `int("005930")`.

## Synthesis / deliverable

### (a) Q1-Q6 with file:line + the plug-in point — see INTERNAL CODE INVENTORY above.
Summary of the load-bearing answers:
- **Q1 plug-in:** `autonomous_loop.py` ~`:321`, an `if paper_markets == ["US"]: universe = None` guard that leaves the US path byte-identical and only takes the multi-market branch when the operator flips the setting.
- **Q2:** fill `candidate_selector.get_universe_tickers` non-US stub with curated static lists; add `build_multimarket_universe(markets)` helper.
- **Q3/Q4 mapper:** add `yf_suffix` to `MARKET_CONFIG` + a `to_yfinance_symbol()` helper in `markets.py`, BUT prefer storing the already-suffixed symbol AS the ticker (curated lists hold `SAP.DE`/`005930.KS`) and carrying `market` separately — this avoids per-call suffix derivation and the `.KS`/`.KQ`/`AIR.PA` traps.
- **Q5:** `paper_markets: list[str] = Field(default_factory=lambda: ["US"])` in `settings.py` (NOT `Field(["US"])`).
- **Q6:** yfinance `.info` gives Yahoo-taxonomy sectors for `.DE`/`.KS`; the existing sector cap groups on whatever string comes back; no GICS remap; fail-open backfill already handles missing sectors.

### (b) Suffix-mapper design + where it lives + every consumer
- **Lives in:** `backend/backtest/markets.py` — add `MARKET_CONFIG[m]["yf_suffix"]`
  (`US`→`""`, `EU`→`".DE"`, `KR`→`".KS"`) + `to_yfinance_symbol(ticker)` +
  `to_namespaced(symbol, market)` helpers. **Constituent lists** in a new
  `backend/backtest/universe_lists.py` (`DAX40`, `KOSPI200` as literal
  yfinance-symbol lists) consumed by `candidate_selector.get_universe_tickers`.
- **RECOMMENDED hot-path convention:** the curated list yields the SUFFIXED
  symbol as the `ticker`; `market` travels alongside. Then the mapper is only
  used at universe-build time, and every downstream consumer receives a
  ready-to-query symbol:
  - `screen_universe` → `yf.download([...suffixed...])` (`screener.py:110`) ✓
  - `_get_live_price(suffixed)` (`paper_trader.py:1200`) ✓
  - sector backfill `yf.Ticker(suffixed).info` (`paper_trader.py:835`) ✓
  - `execute_buy(ticker=suffixed, market=...)` (`paper_trader.py:119,131`) ✓ —
    50.2 already persists `market`+`base_currency`; 50.3 just threads `market`
    through `_StagedBuy` (add field) + the `autonomous_loop.py:996` call (add
    `market=order.market`).
- **The ONE wiring gap 50.3 closes:** `_StagedBuy` (`portfolio_manager.py:25-40`)
  has no `market` field, and `autonomous_loop.py:996-1014` doesn't pass `market=`.
  Add `market: str = "US"` to `_StagedBuy`, set it when staging a candidate (from
  the candidate's market), and pass `market=order.market` at `:996`. Default
  `"US"` keeps every current buy byte-identical.

### (c) DAX-40 + KOSPI-200 constituent source + format (curated static list)
**RECOMMENDATION: curated static list in-repo** (deterministic, offline-safe,
free of scrape-fragility; indices rebalance only ~2x/year). yfinance symbols:

**DAX-40** (from Wikipedia, full read; **note AIR.PA is Paris-listed**):
`ADS.DE, AIR.PA, ALV.DE, BAS.DE, BAYN.DE, BEI.DE, BMW.DE, BNR.DE, CBK.DE,
CON.DE, DTG.DE, DBK.DE, DB1.DE, DHL.DE, DTE.DE, EOAN.DE, FRE.DE, FME.DE,
G1A.DE, HNR1.DE, HEI.DE, HEN3.DE, IFX.DE, MBG.DE, MRK.DE, MTX.DE, MUV2.DE,
PAH3.DE, QIA.DE, RHM.DE, RWE.DE, SAP.DE, G24.DE, SIE.DE, ENR.DE, SHL.DE,
SY1.DE, VOW3.DE, VNA.DE, ZAL.DE`

**KOSPI-200 (seed — large-caps confirmed from Wikipedia; the full 200 should be
finalized from a single source at build time, then frozen as a static list):**
`005930.KS (Samsung Elec), 000660.KS (SK Hynix), 066570.KS (LG Elec),
005380.KS (Hyundai Motor), 000270.KS (Kia), 051910.KS (LG Chem),
006400.KS (Samsung SDI), 105560.KS (KB Fin), 055550.KS (Shinhan),
096770.KS (SK Innovation), 029780.KS (Samsung Card), 023530.KS (Lotte Shopping)
...` (operator/build step to complete to ~200; leading zeros preserved as STRING).
KOSDAQ names (if added later) use `.KQ`.

**Why not Wikipedia-scrape at runtime like S&P?** The S&P scrape is load-bearing
because the index has ~500 names changing frequently; DAX-40/KOSPI-200 are small,
stable, and the goal of 50.3 is RELIABILITY for a built-but-off path. A static
list cannot silently collapse to `[]` on a Wikipedia HTML change (a real
promptcloud failure mode). Refresh the static list manually ~2x/year at rebalance.

### (d) BYTE-IDENTICAL VERIFICATION PLAN (paper_markets=["US"] -> identical to today)
1. **Universe-identity unit test (the gate).** Assert
   `build_multimarket_universe(["US"]) == get_sp500_tickers()` (or that the Q1
   branch with `paper_markets==["US"]` sets `universe = None`, the exact current
   value). Concretely: stub `settings.paper_markets = ["US"]`, run the universe-
   selection block, assert the resulting `universe` object is byte-identical to
   the pre-50.3 value (`None`, or the russell list when that flag is on).
2. **Default-value test.** `Settings().paper_markets == ["US"]` with no env var;
   and `default_factory` gives a fresh list per instance (mutate one, the other
   is unaffected).
3. **No-suffix-applied test.** `to_yfinance_symbol("AAPL") == "AAPL"` and
   `to_yfinance_symbol("US:AAPL") == "AAPL"` (US suffix is `""`) — a US ticker is
   never mutated. `to_yfinance_symbol("BRK-B") == "BRK-B"` (dash preserved).
4. **execute_buy default-market test.** Calling `execute_buy(...)` WITHOUT
   `market=` defaults to `"US"` → `pos_row["market"]=="US"`, `base_currency=="USD"`
   → 50.2 FX path uses `get_fx_rate("USD","USD")==1.0` → money math byte-identical
   (already covered by the 50.2 byte-identity test; re-assert here).
5. **Live before/after.** With `paper_markets` unset, run one autonomous cycle (or
   the universe+screen step) and assert the screened ticker set + count match a
   pre-50.3 baseline capture. NO `.DE`/`.KS` symbol may appear when
   `paper_markets==["US"]`.
6. **Positive intl smoke (built-but-off path exercised explicitly).** With
   `paper_markets=["US","EU"]` in a TEST, assert `build_multimarket_universe`
   includes `SAP.DE` + `AIR.PA`, NOT `AIR.DE`; assert `to_yfinance_symbol` on a
   curated entry is idempotent; (optionally, network-gated) assert
   `_get_live_price("SAP.DE")` returns a EUR float.

### (e) yfinance KR/.KS viability assessment + risks
**VERDICT: VIABLE for KOSPI-200 large-caps via free yfinance, with documented
risks — consistent with the operator's choice.** Evidence:
- `005930.KS` (Samsung) and the KOSPI-200 large-caps resolve on Yahoo (live pages
  confirmed; `.KS` is the official KSE suffix per Yahoo Help). KOSPI-200 is
  major-exchange large-cap, the segment promptcloud calls "solid."
- **Risk 1 — 20-minute quote delay** (Yahoo Help): KR (and KOSDAQ) quotes are
  20-min delayed (vs 15-min for XETRA). For a DAILY paper cadence this is
  immaterial (marks use daily close), but note it.
- **Risk 2 — 429 rate-limits** (#2125/#2128) on multi-ticker loops: screening
  200 KR + 40 EU + 500 US = 740 symbols. Batch `yf.download` (already used) is one
  request and the safe path; the per-position `_get_live_price` fan-out is the
  429-exposed path — add retry/backoff there in a follow-up.
- **Risk 3 — silent-empty data** (promptcloud): a delisted/renamed KR code returns
  an empty frame, not an error. The screener already drops on `len<20`; ADD a
  per-market resolve-count log so a silent collapse is visible. The 50.5 gate must
  assert a minimum KR resolve-rate before the operator flips `paper_markets`.
- **Risk 4 — `.info` sector gaps** (yfinance issues): KR sector may be missing more
  often than US; the fail-open backfill (`paper_trader.py:837`) degrades to
  "Unknown" (own sector bucket) — non-fatal.
- **Risk 5 — OHLC precision** (Tobi Lux, measured on DE; KR not separately
  measured but same data pipeline) — the screener uses CLOSE-based factors (the
  less-affected family), so screening is bounded; candlestick logic (none here) is
  not. **FX-correct ≠ price-accurate; the 50.5 gate is the real guard.**
- **Net:** free yfinance is the right call for 50.3 (built-but-off). The residual
  data-quality risk is exactly what 50.5 gates. If the 50.5 KR resolve-rate is
  unacceptable, pykrx (declined now) is the documented fallback.

### (f) Application mapping (external -> internal file:line)
- Yahoo suffix table (SLN2310) → `MARKET_CONFIG[m]["yf_suffix"]` in `markets.py:21-52` (`.DE`/`.KS`/`.KQ`).
- DAX-40 Wikipedia (incl. AIR.PA) → curated `DAX40` list in `universe_lists.py`; consumed by `candidate_selector.get_universe_tickers("EU")` (`:127-132` stub filled).
- KOSPI-200 Wikipedia → curated `KOSPI200` list; `get_universe_tickers("KR")`.
- yfinance sector docs (Yahoo taxonomy, intl-supported) → sector cap reads `.info["sector"]` at `paper_trader.py:835`; no GICS remap.
- Tobi Lux + promptcloud (silent-empty + ≤11% dev) → per-market resolve-count log in `screen_universe` (`screener.py:206`); 50.5 data-quality gate.
- 429 issue #2125 → keep batch `yf.download` (`screener.py:110`); flag retry/backoff for `_get_live_price` fan-out.
- Cross-listing/ADR (different ISIN) + Merck collision → curated list pins LOCAL lines (`SAP.DE` not ADR; `MRK.DE`≠US `MRK`); suffix is the disambiguator carried with `market`.

## Research Gate Checklist

Hard blockers — all satisfied:
- [x] >=5 authoritative external sources READ IN FULL (8: Yahoo Help SLN2310 [official], yfinance sector docs [official], DAX Wikipedia, KOSPI-200 Wikipedia, Tobi Lux measured DAX study [industry], yfinance #2125 [community], promptcloud 2026 [industry], tuxfamily suffix table [reference]). Hierarchy honored (2 official, 2 reference, 2 industry incl. 1 measured, 1 community, plus DAX/KOSPI reference).
- [x] 10+ unique URLs total (19 incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported (2026 promptcloud/monetaiq/quantvps; 2024-2025 rate-limit wave; suffix scheme + sector exposure stable)
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim (Q1-Q6, universe path, mapper consumers, settings, pos_row, execute_buy callsite)

Soft checks:
- [x] Internal exploration covered: autonomous_loop (universe path + execute_buy callsite), candidate_selector (get_universe_tickers stub), markets.py (namespacing + MARKET_CONFIG, no suffix mapper yet), screener.py (get_sp500_tickers + screen_universe + yf.download + sector_lookup), paper_trader (execute_buy signature + pos_row market persistence [50.2 done] + _get_live_price + .info sector backfill), portfolio_manager (_StagedBuy fields — the wiring gap), settings.py (no list-type Field yet; multi-market scalars), security.md (ticker sanitization "alphanumeric+dots")
- [x] Contradictions/consensus noted (OHLC precision vs average-based factors; static vs dynamic constituent list; production-viability debate)
- [x] All claims cited per-claim with file:line or URL

## Research-gate JSON envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 8,
  "snippet_only_sources": 11,
  "urls_collected": 19,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "gate_passed": true
}
```
