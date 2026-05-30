---
name: project-multimarket-universe-wiring
description: phase-50.3 international universe wiring — exact byte-identical plug-in point, suffix-mapper design, the ONE wiring gap (_StagedBuy.market), DAX-40/KOSPI-200 source, yfinance KR viability + risks
metadata:
  type: project
---

phase-50.3 (researched 2026-05-30): build international universe (EU/.DE/DAX-40 +
KR/.KS/KOSPI-200) via free yfinance, BUILT-but-OFF behind `paper_markets` default `["US"]`.

**Byte-identical plug-in point (Q1):** `autonomous_loop.py` ~line 321. Guard
`if paper_markets == ["US"]: universe = None` leaves the US path byte-identical
(`screen_universe(None)` -> `get_sp500_tickers()` Wikipedia scrape unchanged). The
russell1000 block (`:312-320`) sets `universe` first — the paper_markets block must
preserve that decision for US-only. Multi-market branch is dead code until operator flips setting.

**The ONE wiring gap 50.3 closes:** `_StagedBuy` (portfolio_manager.py:25-40) has NO
`market` field, and `autonomous_loop.py:996-1014` execute_buy call doesn't pass `market=`.
Add `market: str = "US"` to _StagedBuy + pass `market=order.market` at :996. Default "US"
keeps every current buy byte-identical. NOTE: 50.2 ALREADY persists `market`+`base_currency`
in BOTH pos_row branches (paper_trader.py:311-312, :332-333) and added `market="US"` param to
execute_buy signature (:131) — so the position-WRITE side is done; only the candidate->buy
THREADING is missing.

**Suffix-mapper design (Q3/Q4) — RECOMMENDED:** store the already-suffixed yfinance symbol
AS the ticker (curated lists hold `SAP.DE`, `005930.KS` directly); carry `market` as a
separate field. Then NO per-call suffix derivation — `_get_live_price(ticker)`
(paper_trader.py:1200), `yf.Ticker(ticker).info` (:835), `yf.download` (screener.py:110) all
receive the ready symbol verbatim. This sidesteps the `.KS`-vs-`.KQ` (KOSDAQ) and `AIR.PA`
(Airbus is Paris-listed, NOT .DE!) traps that a naive "EU->append .DE" derivation hits.
Add `MARKET_CONFIG[m]["yf_suffix"]` + `to_yfinance_symbol()` to markets.py for build-time use.

**NO ticker-shape assumptions:** grep for isalpha/`^[A-Z]`/regex/sanitize across screener,
paper_trader, autonomous_loop returned NOTHING. A numeric `005930.KS` ticker trips no
validator. BQ keys are STRING; preserve KR leading zeros (never int()).

**Constituent source (Q2):** curated STATIC list in-repo (NOT runtime Wikipedia scrape like
S&P). DAX-40/KOSPI-200 are small + stable (rebalance ~2x/yr); static list can't silently
collapse to [] on an HTML change. Fill the `candidate_selector.get_universe_tickers` non-US
stub (candidate_selector.py:127-132 currently returns [] with warning). Full DAX-40 .DE list
+ KOSPI-200 large-cap seed captured in the 50.3 brief.

**`paper_markets` setting (Q5):** `list[str] = Field(default_factory=lambda: ["US"])` —
NOT `Field(["US"])` (mutable-default footgun). settings.py has NO existing list-type Field;
multi-market scalars at :49-51 (default_market, base_currency).

**Sector (Q6):** yfinance `.info` gives Yahoo-OWN-taxonomy sectors (NOT GICS) for .DE/.KS;
the existing US sector cap already groups on these Yahoo strings, so no remap needed;
fail-open backfill (paper_trader.py:837) handles missing intl sectors -> "Unknown" bucket.

**yfinance KR viability:** VIABLE for KOSPI-200 large-caps (operator's free-yfinance choice
confirmed OK). Risks: 20-min quote delay for .KS/.KQ (Yahoo Help SLN2310, vs 15-min XETRA);
429 rate-limits on multi-ticker loops (#2125 — batch yf.download safe, per-position
_get_live_price fan-out exposed); silent-empty data (promptcloud 2026 — add per-market
resolve-count log); ≤11% OHLC deviation measured on DAX-40 (Tobi Lux) but screener uses
CLOSE-based factors (less affected). 50.5 data-quality gate is the real guard. pykrx is the
documented fallback if KR resolve-rate fails at 50.5.

**Pitfalls:** Airbus=AIR.PA (Paris!); Merck US `MRK` != Merck KGaA `MRK.DE` (suffix
disambiguates); SAP ADR `SAP` != XETRA `SAP.DE` (curated list pins LOCAL line, never ADR).

See [[project_multimarket_scaffolding_disconnected]] for the broader 50-series context.
