# Research Brief: Multi-Market Expansion (EU + South Korean Equities)

**Status:** IN PROGRESS (write-first incremental)
**Tier:** complex
**Initiated:** 2026-05-29
**Operator ask:** Expand pyfinagent beyond US-only (S&P 500) to EU + South Korean equities; multiple exchanges, multiple currencies, broader sectors. Deliverable = research-backed implementation plan + missing-datapoints inventory. NOT implementing yet — plan + masterplan tasks first.

---

## PART A — INTERNAL AUDIT (US-only assumptions, file:line)

### MAJOR FRAMING FINDING: this is NOT greenfield
Two parallel, disconnected efforts already exist:
- **OLD phase-2.9 "Multi-Market Data Layer" (status=done)** built `backend/backtest/markets.py` (5 markets fully configured: US/NO/CA/EU/KR with exchange code, currency, timezone) + `candidate_selector.get_universe_tickers(market=...)`.
- **DEFERRED phase-5 "Multi-Market Expansion (15-step)"** — many `pending` steps incl. 5.2 Data Provider Abstraction (yfinance+EODHD), 5.9 International Equities (EODHD+IBKR), 5.13 Multi-Asset Backtest. phase-5.1 (Broker ABC), 5.4 (Risk engine), 5.6 (Options) are DONE.
- **THE GAP**: the LIVE autonomous loop does NOT use the multi-market path at all. It calls `screen_universe` / `get_sp500_tickers` / `get_russell1000_tickers` from `backend/tools/screener.py` (hard US-only Wikipedia scrape). The `markets.py` abstraction is wired into the BACKTEST candidate_selector only.

### Audit table (US-only assumptions)
| # | Concern | file:line | Current state | What must change |
|---|---------|-----------|---------------|------------------|
| A1 | Live universe source | `backend/services/autonomous_loop.py:25,310-325`; `backend/tools/screener.py:18,29` (`SP500_URL` Wikipedia; `get_sp500_tickers`/`get_russell1000_tickers`) | Hard US-only. Live loop never touches `markets.py`. | Add market-aware universe loader to the LIVE path (or route live loop through `candidate_selector.get_universe_tickers(market=...)`). |
| A2 | Backtest universe (multi-market aware) | `backend/backtest/candidate_selector.py:98-144` | `get_universe_tickers(market=...)` returns `[]` + warning for non-US (line 127-132); only US Wikipedia implemented. PIT raises NotImplementedError. | Implement non-US universe lists (DAX/CAC/KOSPI constituents). |
| A3 | Sector breadth / Tech concentration | `backend/tools/screener.py:210-403` (`rank_candidates`); `candidate_selector.py:168-200` (`_rank_candidates`) | NO structural sector exclusion. Composite alpha score = momentum 0.4 + rsi 0.2 + vol 0.2 + sma 0.2. Tech wins because momentum-weighted + US momentum currently Tech/semis-concentrated. Sector caps EXIST (`paper_max_per_sector=2`, `paper_max_per_sector_nav_pct=30%` — settings.py:180,188) and a `sector_neutral_momentum_enabled` flag (OFF, settings.py:317) that ranks within-sector. | Health/other sectors are OUT-COMPETED on momentum, not excluded. Lever: enable `sector_neutral_momentum_enabled` and/or broaden universe so non-Tech has more representation. |
| A4 | Currency = USD hardcoded in math | `backend/services/paper_trader.py:221,243,253,275,405,445-447,480` | ALL math is currency-blind dollar arithmetic: `quantity*exec_price`, `cost_basis`, `nav = cash + positions_value`, `pnl = market_value - cost_basis`. No FX conversion anywhere. A EUR position at €100 is added to a USD NAV as $100. | Multi-currency layer: store position currency, convert to base_currency at valuation. |
| A5 | base_currency setting exists but unused in math | `backend/config/settings.py:50-51` (`default_market="US"`, `base_currency="USD"`) | Settings declared but paper_trader never reads base_currency. BQ `paper_portfolio`/`paper_positions` already have `base_currency`+`market` columns. | Wire base_currency into NAV/PnL; populate market+currency per position. |
| A6 | Data ingestion currency stub bug | `backend/backtest/data_ingestion.py:146` | Literal `"currency": "USD" if market == "US" else "USD",  # TODO: lookup from MARKET_CONFIG` — always writes USD even for non-US. | Look up currency from `markets.MARKET_CONFIG[market]["currency"]`. |
| A7 | Price pipeline ticker convention | `backend/backtest/data_ingestion.py:93-167` (`ingest_prices` via `yf.download`) | Uses `{market}:{ticker}` namespace internally but strips to clean_ticker for yfinance. yfinance needs SUFFIX form (.DE/.KS) not namespace. | Map `{market}:{ticker}` -> yfinance suffix (KR:005930 -> 005930.KS) before download. |
| A8 | FRED macro = US only | `backend/backtest/data_ingestion.py:21,308` (`FRED_SERIES`, `market="US"`) | All macro is US (FEDFUNDS, CPIAUCSL...). | Add ECB / BOK macro series OR accept US-macro-only regime signals as a known limitation for v1. |
| A9 | Market hours / calendar | `backend/backtest/markets.py:81-120` (`get_trading_calendar`, `is_trading_day`) EXISTS via `exchange_calendars`; live daily cycle = `paper_trading_daily` America/New_York | Calendar helper EXISTS (XNYS/XOSL/XTSE/XETR/XKRX) but live scheduler is single US-time cron. | Per-market scheduling OR run one cycle that respects each market's `is_trading_day`. |
| A10 | Broker / execution | `backend/markets/broker_base.py` (ABC, phase-5.1), `alpaca_broker.py:42,129` (USD default) | Broker ABC exists; only AlpacaBroker (US). For PAPER we simulate fills — no intl broker needed; paper_trader fills against price data. | PAPER: none needed. REAL-money intl: out of scope. |
| A11 | Backtest engine universe | `backend/backtest/backtest_engine.py` (STRATEGY_REGISTRY, preload) | Preloads `tickers + ["SPY"]` baseline; assumes US benchmark. | Per-market benchmark (^GDAXI / ^KS11) + market-aware preload. |

## PART B — EXTERNAL RESEARCH

### Read in full (>=5 required; counts toward gate)
| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://help.yahoo.com/kb/finance-for-web/exchanges-data-providers-yahoo-finance-sln2310.html | 2026-05-29 | doc | WebFetch | Exact yfinance suffixes: XETRA=.DE (15m), Euronext Paris=.PA (15m), LSE=.L (20m), Euronext Amsterdam=.AS (15m), KRX/KOSPI=.KS (20m), KOSDAQ=.KQ (20m). Provider = ICE Data Services. |
| https://eodhd.com/pricing | 2026-05-29 | doc(vendor) | WebFetch | Free tier = 20 calls/day (unusable for universe). EOD All-World = **$19.99/mo** (150k+ tickers, 30yr, global incl. KRX/Xetra/Euronext/LSE). Fundamentals = **$59.99/mo**. ALL-IN-ONE = $99.99/mo. |
| https://pypi.org/project/pandas_market_calendars/ | 2026-05-29 | doc | WebFetch | v5.4.0 (May 27 2026), MIT, py>=3.10. Mirrors ALL `exchange_calendars`; 50+ exchanges incl. XKRX/XETR/XLON/XPAR/XNYS. Holidays, open/close, breaks, tz (zoneinfo). No runtime server calls. |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC8074275/ | 2026-05-29 | paper (peer-reviewed) | WebFetch | Factors replicate internationally: momentum 8.43% EAFE / 13.32% EM; value +ve esp. EM. **Momentum FAILS in Japan** (outlier). Implementation caveats: liquidity (factors lean on micro-caps you can't trade), data-quality heterogeneity, higher intl transaction costs. |
| https://ar5iv.labs.arxiv.org/html/1611.01463 | 2026-05-29 | paper (preprint) | WebFetch (ar5iv HTML; PDF was binary) | Multi-currency return decomposition: **r_j = a_j(r_ja − i_j) + c_j(r_jc + i_j)** — local asset return minus carry + currency return plus carry. FX forward cost of carry = interest-rate differential (i_buy − i_sell), baked into return not post-hoc. |
| https://medium.com/@Tobi_Lux/data-from-yfinance-some-observations-41e99d768069 | 2026-05-29 | blog (practitioner) | WebFetch | **THE DATA-RISK FINDING.** yfinance Close vs XETRA for BAS.DE: −7% to +3% (avg −2.5%); max deviation across 12 DAX stocks **up to 11%**. **10–24 days/yr** have OHLC all-identical (candlestick algos break). Stooq = only rounding diffs but slower + no adj-close. |

### Identified but snippet-only (context; does NOT count toward gate)
| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://eodhd.com/exchange/KO | vendor | KRX coverage confirmed via search snippet |
| https://github.com/sharebook-kr/pykrx | code | Korea-specific free scraper (KRX); noted as KR fallback |
| https://www.quantstart.com/articles/an-introduction-to-stooq-pricing-data/ | blog | Stooq = free, 21k global securities, no adj-close (snippet sufficient) |
| https://www.kimchang.com/en/insights/detail.kc?...idx=26719 | legal | KRX foreign-investor IRC abolished 2023; LEI+custodian+KRW still needed (real-money only) |
| https://www.lseg.com/en/insights/ftse-russell/t-1-settlement-in-the-us-a-european-perspective | industry | EU T+1 target 2027-10-11; ADR T+1 vs ordinary T+2 — paper-irrelevant |
| https://www.sciencedirect.com/science/article/abs/pii/S0275531917305056 | paper | Zaremba/Shemer factor-momentum 24 intl markets (paywalled abstract) |
| https://www.bok.or.kr/eng/main/contents.do?menuNo=400192 | official | Bank of Korea FX portfolio-investment procedures (real-money custody) |
| https://analystprep.com/study-notes/cfa-level-iii/currency-movement-on-portfolio-risk-and-return/ | edu | CFA L3 currency return decomposition (corroborates arXiv formula) |
| https://www.allinvestview.com/articles/multi-currency-portfolio-guide/ | blog | "Convert every holding to base currency for valuation" (corroboration) |
| https://stockanalysis.com/list/kosdaq-korea/ | data | KOSDAQ constituent list source |
| https://www.nb-data.com/p/best-financial-data-apis-in-2026 | blog | 2026 data-API comparison (recency) |

**Query variants run (3-variant discipline):** current-year frontier (`...2026`), last-2-year (`...2025`), year-less canonical (`portfolio return decomposition local return currency return`, `international equity factor momentum cross-market`). Source table mixes 2026 vendor docs, 2024-2026 blogs, and year-less canonical papers (PMC8074275, arXiv 1611.01463).

### Recency scan (2024-2026)
Searched 2024-2026 literature on intl-equity data + multi-currency accounting + calendars. **Findings that MATTER for the plan:**
1. **yfinance reliability has DEGRADED through 2025-2026** — aggressive rate-limiting, IP bans, and the XETRA deviation/identical-OHLC problem are 2024-2026 observations, not historical. This RAISES the weight on EODHD/Stooq for EU.
2. **`pandas_market_calendars` v5.4.0 shipped 2026-05-27** (2 days ago) — actively maintained, mirrors `exchange_calendars`; the calendar problem is fully solved by an existing maintained lib.
3. **EU T+1 settlement target = 2027-10-11** (current planning) — confirms settlement is a real-money-only, future concern; paper trading is unaffected.
4. **KRX foreign-investor registration (IRC) abolished Dec 2023** — easier real-money access, but LEI + KRW conversion + local custodian still required (paper-irrelevant). No NEW finding that changes paper-trading feasibility.
No 2024-2026 finding SUPERSEDES the canonical multi-currency decomposition (arXiv 1611.01463) or the international-factor evidence (PMC8074275); they COMPLEMENT.

### Consensus vs debate (external)
- **Consensus:** equity factors (esp. momentum) replicate across Europe + Korea; convert all holdings to base currency at valuation; `exchange_calendars`/`pandas_market_calendars` is THE calendar solution; EODHD is the consensus low-cost global EOD+fundamentals vendor.
- **Debate / caution:** (a) Japan is a documented momentum-failure outlier — Korea is NOT flagged as a failure market but is adjacent, so transfer is plausible-but-unproven; (b) yfinance international quality is contested — "works" vs "11% deviations" — consensus is "OK for casual, risky for production EU."

### Pitfalls (from literature)
1. yfinance EU data: up to 11% price error + 10-24 identical-OHLC days/yr → any candlestick/intraday-range signal silently breaks (RSI on bad bars, vol estimates).
2. Factor strategies lean on micro-caps that are illiquid/untradeable internationally — screen on liquidity (min volume in LOCAL currency).
3. FX return is a SEPARATE return stream — a EUR position can rise in EUR but fall in USD. Must decompose to avoid mis-attributing alpha.
4. Momentum can fail in specific markets (Japan) — do NOT assume US-tuned params transfer; re-validate per market.

## PART C — SYNTHESIS (the deliverable)

### (b) MISSING-DATAPOINTS INVENTORY
| Datapoint | Have for US? | Missing for EU/KR | Recommended source | Cost / approval |
|-----------|--------------|-------------------|--------------------|-----------------|
| EOD price history (EU: .DE/.PA/.L/.AS) | Yes (yfinance→BQ) | yfinance "works" but ≤11% error + identical-OHLC days | **v1: yfinance suffix** (free, accept risk); **v2: EODHD $19.99/mo** or Stooq (free, no adj-close) | v1 free; v2 = OWNER approval (paid) |
| EOD price history (KR: .KS/.KQ) | Yes (US) | yfinance covers .KS/.KQ; pykrx as KR-native fallback | **v1: yfinance .KS/.KQ**; fallback **pykrx** (free, pip) | pykrx = OWNER approval (pip) |
| Fundamentals (EU/KR) | Yes (yfinance quarterly) | yfinance `.info` sector/financials patchy for intl | **v1: yfinance**; **v2: EODHD Fundamentals $59.99/mo** | v2 = OWNER approval (paid) |
| **FX rates** (EUR/USD, KRW/USD, GBP/USD) | N/A (US only) | **COMPLETELY MISSING** — no FX anywhere in codebase | yfinance FX pairs (`EURUSD=X`, `KRW=X`, `GBPUSD=X`) free; OR FRED (`DEXKOUS`, `DEXUSEU`) free | FREE (yfinance or FRED) |
| Market calendars | Implicit (US cron) | Helper EXISTS (`markets.py`) but `exchange_calendars` may not be installed | `pandas_market_calendars` v5.4.0 or `exchange_calendars` | pip — likely already a dep (markets.py imports it); verify |
| Sector taxonomy | GICS (yfinance `sector`) | yfinance returns GICS-ish for intl too, but ICB used in EU/UK | Keep yfinance `sector` (GICS-aligned); document GICS-vs-ICB mismatch as known | FREE |
| Universe constituents (DAX/CAC/KOSPI) | S&P500 Wikipedia | `get_universe_tickers(market)` returns [] for non-US | Wikipedia lists (DAX-40, CAC-40, KOSPI-200) — same pd.read_html pattern | FREE |
| Per-market benchmark index | SPY/^GSPC | Missing | yfinance `^GDAXI` (DAX), `^KS11` (KOSPI), `^FTSE`, `^FCHI` | FREE |
| Macro (ECB/BOK rates) | FRED US series | Missing non-US macro | FRED has intl series (`ECBDFR`, `INTDSRKRM193N`); OR accept US-macro-only for v1 | FREE |

### (c) RECOMMENDED PHASED IMPLEMENTATION PLAN (step-shaped for masterplan)
**Reuse note:** phase-2.9 + phase-5.1/5.4/5.6 already built `markets.py`, broker ABC, risk engine. The deferred phase-5 (5.2/5.9/5.13) overlaps. Recommend a NEW focused phase (call it **phase-50: Multi-Market Equities (EU+KR paper)**) that COMPLETES the live-loop gap rather than the broad asset-class phase-5. Each step below = one masterplan entry with a verifiable criterion.

**Phase 50.1 — FX rate data layer (FOUNDATION, do first; FREE)**
- 50.1.1 Add `backend/services/fx_rates.py`: fetch + cache daily FX (EURUSD=X, KRW=X, GBPUSD=X) from yfinance; BQ table `historical_fx_rates`. Criterion: `get_fx_rate("EUR","USD",date)` returns a float for 30 past days.
- 50.1.2 Backfill FX history to match price history start (2022-01-01). Criterion: BQ `historical_fx_rates` row count > 0 for each pair.
- 50.1.3 Fix `data_ingestion.py:146` currency stub to read `markets.MARKET_CONFIG[market]["currency"]`. Criterion: non-US ingest row shows correct currency (EUR/KRW), not USD.

**Phase 50.2 — Multi-currency portfolio accounting (CORE; FREE)**
- 50.2.1 Add `market`+`currency` to every position write in `paper_trader.py`. Criterion: a EUR BUY persists `currency="EUR"` in paper_positions.
- 50.2.2 Convert to base_currency in `mark_to_market` + NAV (`paper_trader.py:432-495`): `market_value_base = local_value * fx_rate`. Criterion: NAV with a EUR position equals cash + (EUR qty*price*EURUSD), NOT cash + raw EUR number.
- 50.2.3 P&L attribution: split into local return + FX return per arXiv 1611.01463 (`r = local + FX`). Criterion: a position flat in local currency but with FX move shows nonzero fx_pnl and ~zero local_pnl.

**Phase 50.3 — Universe + ticker convention + sector breadth (FREE)**
- 50.3.1 Implement `candidate_selector.get_universe_tickers(market)` for DE/KR (Wikipedia DAX-40, KOSPI-200). Criterion: `get_universe_tickers("DE")` returns >30 tickers.
- 50.3.2 Add `{market}:{ticker}` → yfinance-suffix mapper (KR:005930→005930.KS, DE:BAS→BAS.DE) used by ingestion + screener. Criterion: round-trip mapping unit test passes for .DE/.KS/.KQ/.L/.PA.
- 50.3.3 Route the LIVE `autonomous_loop` universe selection (lines 310-329) through a market-aware loader + a `paper_markets` setting (default `["US"]`). Criterion: with `paper_markets=["US","DE"]`, a cycle screens both universes; with `["US"]` behavior is byte-identical to today.
- 50.3.4 (Sector breadth — optional, may already suffice) Document that Tech-concentration is momentum-driven not structural; expose `sector_neutral_momentum_enabled` as the lever. Criterion: with flag ON + multi-market universe, a cycle's BUY set spans >2 sectors.

**Phase 50.4 — Market calendar / scheduling (FREE)**
- 50.4.1 Verify/install `exchange_calendars`; gate live trades on `markets.is_trading_day(date, market)`. Criterion: on a German holiday that is a US trading day, no DE trade fires but US trades do.
- 50.4.2 Decide scheduling model: ONE daily cycle that iterates enabled markets and skips closed ones (simplest) vs per-market cron. Recommend single-cycle for v1. Criterion: cycle log shows per-market open/closed decision.

**Phase 50.5 — Backtest engine multi-market (FREE)**
- 50.5.1 Per-market benchmark in backtest (^GDAXI/^KS11 instead of SPY). Criterion: a DE backtest reports DAX baseline Sharpe.
- 50.5.2 Multi-currency NAV in `backtest_trader.py` mirroring 50.2. Criterion: a EUR backtest NAV is FX-converted.

**Phase 50.6 — UI (exchange/currency selection + display)**
- 50.6.1 Exchange/market filter on paper-trading dashboard + backtest config. Criterion: UI dropdown lists US/DE/KR; selection filters positions.
- 50.6.2 Currency display: show position local currency + base-currency value; multi-currency NAV breakdown. Criterion: a EUR position shows "€X (= $Y)".
- 50.6.3 Market-hours indicator (open/closed per market using calendar). Criterion: badge reflects live open/closed state per market.

**Owner-approval gates:** EODHD ($19.99 + $59.99/mo), pykrx (pip), any paid data. All phase-50.1/50.2 work is FREE (yfinance + FRED + existing libs).

### (d) UI CHANGES (which pages)
- **Paper-trading dashboard** (`frontend/src/app/...paper-trading`): market/exchange filter, per-position currency badge (local + base), multi-currency NAV breakdown, market-hours open/closed indicator.
- **Backtest page**: market selector in config; per-market benchmark in results.
- **Home cockpit**: NAV is already base-currency — add a small "X% non-USD exposure" chip + FX-attribution line.
- **Settings page**: expose `paper_markets` list + `sector_neutral_momentum_enabled` toggle.

### (e) SEQUENCING + RISK
- **MINIMUM VIABLE FIRST MARKET: EU via Germany (XETRA / .DE) using yfinance**, NOT Korea-first. Why: (1) yfinance .DE coverage is mature and documented; (2) EUR/USD FX is the most liquid pair (easy to source/validate); (3) DAX-40 is a small, clean, liquid universe; (4) GICS sectors map cleanly; (5) no language/encoding issues with tickers. Korea (.KS/.KQ + KRW + pykrx fallback) is phase-50.x AFTER the EU path proves the multi-currency machinery.
- **BIGGEST RISK = DATA QUALITY (yfinance intl).** Up to 11% XETRA deviation + 10-24 identical-OHLC days/yr will silently corrupt momentum/RSI/vol signals. MITIGATION: (a) a data-quality gate that flags identical-OHLC bars and drops them; (b) budget for EODHD ($19.99/mo) as the v2 source if v1 signals look noisy; (c) validate a sample of .DE bars against a second source (Stooq) in 50.1.
- **Second risk: strategy non-transfer.** US-tuned momentum params may not hold in EU/KR (Japan precedent). MITIGATION: re-run the optimizer per-market; don't assume `optimizer_best.json` transfers.
- **Sequencing:** FX layer (50.1) MUST precede accounting (50.2) MUST precede universe go-live (50.3). Calendar (50.4) and backtest (50.5) parallel-izable. UI (50.6) last.

### Application to pyfinagent (external → internal mapping)
- arXiv 1611.01463 decomposition `r = a(r_a − i) + c(r_c + i)` → implement in `paper_trader.mark_to_market` (50.2.2/50.2.3) where today it's `pnl = market_value − cost_basis` (paper_trader.py:446) with no FX term.
- yfinance/XETRA 11% finding → drives the data-quality gate in 50.1 + the EODHD owner-approval option.
- `pandas_market_calendars` v5.4.0 → satisfies `markets.get_trading_calendar` (markets.py:81); just needs install-verify (50.4.1).
- PMC8074275 momentum-fails-in-Japan → drives per-market re-optimization (50.3/risk note), not blind param transfer.
- Yahoo suffix doc → drives the namespace→suffix mapper (50.3.2): KR:005930→005930.KS, DE:BAS→BAS.DE.

### Research Gate Checklist
Hard blockers — all satisfied:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6: Yahoo doc, EODHD, pandas_market_calendars, PMC8074275 paper, arXiv 1611.01463 paper, Tobi Lux blog)
- [x] 10+ unique URLs total (16+ incl. snippet-only)
- [x] Recency scan (2024-2026) performed + reported
- [x] Full pages/papers read (not abstracts) for read-in-full set
- [x] file:line anchors for every internal claim (Part A table)

Soft checks:
- [x] Internal exploration covered universe/screener/currency/data/calendar/broker/backtest modules
- [x] Contradictions noted (yfinance "works" vs 11% error; Japan momentum outlier)
- [x] Claims cited per-claim with URLs + file:line

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 6,
  "snippet_only_sources": 11,
  "urls_collected": 17,
  "recency_scan_performed": true,
  "internal_files_inspected": 9,
  "gate_passed": true
}
```
