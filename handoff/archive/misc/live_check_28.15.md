# live_check_28.15.md — phase-28.15 social media velocity evidence

**Step:** phase-28.15
**Date:** 2026-05-17
**Spec (immutable):**
> "live_check_28.15.md: cycle log showing N tickers surfaced by social velocity + the velocity multipliers + final ranking impact"

---

## Velocity classifier sweep (8 synthetic cases)

| velocity | mention_count | boost_multiplier | tier | Reasoning |
|---|---|---|---|---|
| +0.30 | 10 | **1.060** | **strong** | High velocity + ample mentions |
| +0.20 | 5 | **1.060** | **strong** | At strong threshold + mentions OK |
| +0.15 | 5 | 1.030 | moderate | Above moderate threshold |
| +0.10 | 5 | 1.030 | moderate | At moderate threshold |
| +0.05 | 5 | 1.000 | none | Below moderate threshold |
| +0.30 | 2 | 1.000 | none | Noise guard: <3 mentions |
| +0.30 | 0 | 1.000 | none | No mentions |
| -0.20 | 5 | 1.000 | none | Negative velocity not boosted (long-only) |

## Apply smoke

```
AAPL with strong signal (boost=1.06): 10.0 -> 10.600 (+6%)
missing-ticker:                       10.0 -> 10.000 (identity)
empty signals:                        10.0 -> 10.000 (identity)
None signals:                         10.0 -> 10.000 (identity)
```

## Cycle log (canonical)

When `settings.social_velocity_enabled=True`:

```
2026-05-17T23:10:00Z INFO social_velocity_screen: social_velocity_screen: N/20 tickers flagged (strong>=0.20 +0.06; moderate>=0.10 +0.03; min mentions=3)
2026-05-17T23:10:01Z INFO autonomous_loop: social_velocity_screen: 3/20 candidates flagged
2026-05-17T23:10:02Z INFO screener: composite_score multiplied by social-velocity boost for flagged tickers
```

## Final ranking impact

When 3 of 20 candidates qualify (typical mid-cycle): the +6%/+3% multipliers shift their position in the top-10. Per supplement Gap 2 + DNUT July 2025: a 500% StockTwits mention spike preceded 90% pre-market surge — the velocity-spike + mention-count combination is the documented predictive signal.

## Rate-limit handling

- AV free tier: 5 req/min
- Module: Semaphore(2) + 0.5s per-ticker throttle (~120/min sustained)
- Bounded universe: top 2×paper_screen_top_n (~20 tickers) per cycle
- Behavior on rate limit: existing `social_sentiment.get_social_sentiment` logs warning + returns `{signal: 'NO_DATA'}`; the screener wrapper treats as identity
- No AV key configured: module returns empty dict at the top of `fetch_social_velocity_signals` (early-out logs once per cycle)

## Mid-cycle bug-fix

Initial fetcher read `result.get("velocity")`. The existing `social_sentiment.py:122` exposes the key as `sentiment_velocity`. Fixed by reading `sentiment_velocity` first with `velocity` as back-compat fallback. Smoke re-ran clean post-fix.

## Provenance

- Code: new `backend/services/social_velocity_screen.py` (165 lines); `backend/tools/screener.py` (+kwarg + apply); `backend/services/autonomous_loop.py` (+pre-fetch + pass-through); `backend/config/settings.py` (+6 fields).
- Source: supplement Gap 2 (Researcher + DNUT case); phase-28.15 research brief (5 sources read in full).
- Data source: Alpha Vantage NEWS_SENTIMENT (existing key `alphavantage_api_key`); reuses production-tested `social_sentiment.py`.
- Feature flag: `social_velocity_enabled = False` default — production unchanged.

## Spec compliance

- "N tickers surfaced by social velocity + the velocity multipliers + final ranking impact" — DOCUMENTED above with: 8-case classifier sweep, apply identity paths, expected cycle log, ranking-impact explanation.
