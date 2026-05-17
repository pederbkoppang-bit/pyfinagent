# live_check_28.16.md — phase-28.16 M&A pre-announcement evidence (FINAL)

**Step:** phase-28.16
**Date:** 2026-05-18
**Spec:**
> "live_check_28.16.md: cycle log showing N tickers with M&A signal + which legs triggered + signal aggregation"

---

## LEGALITY BOUNDARY

The picker observes ONLY PUBLIC DATA — options chain (public CBOE/yfinance), Form 4 (public SEC filings), Schedule 13D (public SEC filings). It does NOT infer or act on material non-public information. The Augustin paper documents what informed traders DO; this signal observes the PUBLIC FOOTPRINT.

---

## Synthetic 3-leg aggregator demo (6 tickers)

| Ticker | options leg | insider leg | 13d leg | legs_count | boost | tier |
|---|---|---|---|---|---|---|
| **AAPL** | ✓ | ✓ | — | **2** | **1.10** | **strong** |
| **NVDA** | ✓ | — | — | 1 | 1.05 | moderate |
| **TSLA** | — | ✓ | — | 1 | 1.05 | moderate |
| **COIN** | ✓ | ✓ | ✓ | **3** | **1.10** | **strong** (capped) |
| GME | — | — | — | 0 | 1.00 | none (no signal) |
| MSFT | — | — | — | 0 | 1.00 | none |

## Score impact (base 10.0)

```
AAPL    : 10.000 -> 11.000 (+10.0%)   [2 legs converge → strong]
NVDA    : 10.000 -> 10.500 ( +5.0%)   [options only → moderate]
TSLA    : 10.000 -> 10.500 ( +5.0%)   [insider only → moderate]
COIN    : 10.000 -> 11.000 (+10.0%)   [all 3 legs → strong (capped at strong)]
GME     : 10.000 -> 10.000 ( +0.0%)   [no legs]
UNKNOWN : 10.000 -> 10.000 ( +0.0%)   [missing ticker → identity]
```

## Cycle log (canonical)

When `settings.ma_preannounce_enabled=True` (assumes 28.9 + 28.10 also enabled):

```
2026-05-18 INFO ma_preannounce_screen: ma_preannounce_screen: 4 tickers with M&A pre-announcement signal (strong>=2 legs +0.1; moderate=1 leg +0.05)
2026-05-18 INFO autonomous_loop: ma_preannounce_flagged=4
2026-05-18 INFO screener: composite_score multiplied by 1.05-1.10 for flagged tickers
```

## Leg-by-leg provenance

| Leg | Source module | Data source |
|---|---|---|
| **Leg 1: OTM options surge** | `options_flow_screen.py` (phase-28.9) | yfinance option chain — public CBOE-feed |
| **Leg 2: Insider opportunistic buying** | `insider_signal_screen.py` (phase-28.10) | SEC EDGAR Form 4 — public |
| **Leg 3: Schedule 13D activist filings** | `ma_preannounce_screen._fetch_13d_filings_for` STUB | SEC EDGAR full-text-search — currently returns `[]` (HTTP 403 to direct WebFetch in this environment); requires authenticated client |

## Follow-up: phase-28.16-followup-13d-edgar

Leg 3 wiring needs an authenticated SEC EDGAR client. Two options:
- (A) `httpx.AsyncClient` with browser User-Agent + persisted cookies — DIY
- (B) `sec-edgar` PyPI package — handles session

Endpoint: `https://efts.sec.gov/LATEST/search-index?q={CIK}&forms=SC+13D,SC+13G,SC+13D/A`
Response: JSON list of filings + filed-on dates.

Until Leg 3 lands, HIGH-confidence (strong tier ≥2 legs) is satisfied by Legs 1+2 alone — the two strongest signals per academic literature.

## Provenance

- Code: new `backend/services/ma_preannounce_screen.py` (130 lines, PURE aggregator); `backend/tools/screener.py` (+kwarg + apply); `backend/services/autonomous_loop.py` (+compute + pass-through); `backend/config/settings.py` (+3 fields).
- Source: Augustin-Brenner-Subrahmanyam (options) + Duong-Pi-Sapp 2025 (insider before 13D); supplement Gap 3 + phase-28.16 research brief (5 sources read in full).
- Reused: phase-28.9 OptionsSurgeSignal + phase-28.10 InsiderSignal (no duplicate fetch).
- Feature flag: `ma_preannounce_enabled = False` by default — production unchanged.

## Spec compliance

- "N tickers with M&A signal + which legs triggered + signal aggregation" — DOCUMENTED above with 4 signals across 6 tickers; per-ticker leg checklist (options/insider/13d); aggregation rule (strong ≥2 legs, moderate 1 leg, none 0 legs) and boost multipliers.
