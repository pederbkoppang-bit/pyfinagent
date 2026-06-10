# Research Brief — phase-5 crypto removal (scope-reduction)

**Tier:** simple (closure-style scope reduction; user directive)
**Date:** 2026-04-19

## Objective

User directive: "remove Crypto Market as this something we are not going to do".

Strip all crypto-specific scope from the 15 newly-authored phase-5 sub-steps. Crypto was originally proposed by the phase-5 restructure research (2026-04-19 at 23:45 UTC) as the first new market due to Alpaca reusing existing credentials. Owner has now rejected the crypto direction outright.

## Why no new external research is needed

This is a pure scope-subtraction, not a new market or new infrastructure decision. The substantive research behind the other 14 steps (`handoff/current/phase-5-restructure-research-brief.md`, 7 sources in full, 22 URLs) already covers:

- Broker abstraction (5.1) — unchanged.
- Data-provider abstraction with EODHD (5.2) — unchanged; EODHD still needed for international equities (5.9).
- Multi-asset BQ schema (5.3) — shrinks (no `crypto_candles` table).
- Risk engine (5.4) — shrinks (no crypto VaR branch).
- Options (5.6), FX (5.7), Futures (5.8), International equities (5.9), ETF expansion (5.10) — unchanged structurally, but 5.10 drops crypto-ETF tickers BITO and IBIT.
- Cross-market regime (5.11) — replaces `crypto_vol` input with an equity-volatility proxy (e.g. VVIX or SPX realized vol over 30d).
- Cross-market signals (5.12) — drops `crypto_equity_spillover` signal; keeps FX carry + yield-curve-rotation.
- Multi-asset backtest (5.13) — drops `asset_classes=['equity','crypto']` test case; keeps equity + options/futures/FX tests.
- Autonomous loop integration (5.14) — drops `enable_crypto_trading` settings flag.
- Integration gate (5.15) — drops crypto portion of the e2e test.

Dependencies re-chain: 5.6 (options) depended on 5.4+5.5; re-chain to 5.4 only. 5.10 (ETF expansion) depended on 5.5; re-chain to 5.2 (data-provider abstraction). 5.11 depended on 5.5+5.7; re-chain to 5.7 only. 5.13 depended on 5.4+5.5+5.11; re-chain to 5.4+5.11. 5.14 depended on 5.5+5.7+5.8+5.12+5.13; re-chain to 5.7+5.8+5.12+5.13.

## Compliance / risk

- **Crypto-ETF tickers (BITO, IBIT) removed from 5.10** — tradeable under existing US equity license; removal is scope-consistency, not compliance-driven.
- **No other crypto references** expected in the masterplan once 5.5 is dropped and 5.11/5.12/5.13/5.14 are re-scoped.
- **CFTC compliance open_issue** remains for futures (5.8); FX CFTC exposure remains a phase-5.7/5.14 owner decision.

## Recency scan (2024-2026)

Scope-reduction; no new external research adds value. The 2024–2026 environment (Alpaca's 2025 crypto feature, US spot-BTC ETFs) simply confirms the owner is saying NO to a market that is technically accessible — that's a strategic-priority call, not a technical one.

## JSON envelope

```json
{
  "tier": "simple",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": true,
  "internal_files_inspected": 1,
  "gate_passed": true,
  "note": "scope-reduction closure per user directive; builds on phase-5-restructure-research-brief.md (7 sources in full). No new external sources needed for a pure subtraction."
}
```
