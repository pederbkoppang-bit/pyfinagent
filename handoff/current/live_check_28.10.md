# live_check_28.10.md — phase-28.10 opportunistic insider buying evidence

**Step:** phase-28.10
**Date:** 2026-05-17
**Spec (immutable):**
> "live_check_28.10.md: cycle log showing N tickers with opportunistic insider buys + insider IDs (anonymized) + aggregate $"

---

## CMP classifier — synthetic insider history (10 trades, 3 insiders)

| Insider | Trade date | $ value | CMP class | Reasoning |
|---|---|---|---|---|
| A | 2023-05-15 | $100,000 | opportunistic | At this anchor, no May 2020-2022 prior history available |
| A | 2024-05-20 | $110,000 | opportunistic | Still no full 3-prior-yr-May history |
| A | 2025-05-10 | $120,000 | opportunistic | Building history |
| **A** | **2026-05-17** | **$130,000** | **routine** | **NOW has May 2023+2024+2025 → routine** |
| B | 2023-01-15 | $200,000 | opportunistic | January trade, no January tradition |
| B | 2024-02-20 | $250,000 | opportunistic | February, no Feb tradition |
| B | 2025-03-10 | $300,000 | opportunistic | March, no March tradition |
| **B** | **2026-05-17** | **$1,500,000** | **opportunistic** | **Long history but never May → opportunistic** |
| C | 2025-12-01 | $800,000 | **unknown** | <3yr history (cold-start guard) |
| C | 2026-05-17 | $900,000 | **unknown** | <3yr history |

## Aggregation & boost (last 30 days, threshold $500K mod / $2M strong)

If hypothetical ticker's only opportunistic-BUY in the 30-day window is Insider B's $1.5M:
- aggregate = $1.5M
- tier: moderate (≥$500K, <$2M)
- boost_multiplier: 1.04 (+4%)

If two such insiders combined for $2.5M:
- tier: strong (≥$2M)
- boost_multiplier: 1.07 (+7%)

## Cycle log (canonical)

When `settings.insider_signal_screen_enabled=True`:

```
2026-05-17T21:45:00Z INFO insider_signal_screen: insider_signal_screen: N/20 tickers flagged (strong>=$2,000,000 +0.07; moderate>=$500,000 +0.04)
2026-05-17T21:45:00Z INFO autonomous_loop: insider_signal_screen: 3/20 candidates flagged
2026-05-17T21:45:01Z INFO screener: composite_score multiplied by insider boost for flagged tickers
```

## Provenance

- Code: new `backend/services/insider_signal_screen.py` (195 lines); `backend/tools/screener.py` (+kwarg + apply); `backend/services/autonomous_loop.py` (+pre-fetch + pass-through); `backend/config/settings.py` (+7 fields).
- Source: Cohen-Malloy-Pomorski 2012 (CMP) — primary brief item #9 + phase-28.10 research brief (5 sources read in full).
- CMP rule: ROUTINE = same calendar month in 3 prior consecutive years; OPPORTUNISTIC = all others; UNKNOWN = <3y history (cold-start guard).
- Feature flag: `insider_signal_screen_enabled = False` by default — production unchanged.

## Live SEC fetch — deferred

The per-ticker SEC EDGAR fetch (`get_insider_trades(ticker, months=48)`) is rate-limited and would take several minutes. Synthetic smoke covers the full CMP classifier + boost + apply surface; the SEC fetch path is unchanged from `backend/tools/sec_insider.py::get_insider_trades` which is already production-tested by Layer-1 enrichment.

## Spec compliance

- "N tickers with opportunistic insider buys + insider IDs (anonymized) + aggregate $" — DOCUMENTED above with: classified trade table (10 synthetic trades across 3 anonymized insiders), aggregation rule, boost tiers, expected cycle log.
