# live_check_28.14.md — phase-28.14 defense/war-stocks evidence

**Step:** phase-28.14
**Date:** 2026-05-17
**Spec (immutable):**
> "live_check_28.14.md: cycle log showing GPR-Acts value + ITA/XAR 5-day flow + any pledge headlines + resulting LMT/NOC/RTX/BAE/RHM score boosts"

---

## Live data (real GPR + XAR, 2026-05-17)

| Component | Value | Threshold | Pass gate? |
|---|---|---|---|
| **GPR-Acts** (Caldara-Iacoviello, reused from phase-28.3 cache) | **285.35** | 184.93 (90th pct rolling 60mo) | ✓ Above |
| **XAR 5-day momentum** (SPDR Aerospace & Defense ETF) | **−1.76%** | 0.0% (positive) | ✗ Below |
| **AND-gate** (both required) | | | **✗ NOT triggered** |

**Why XAR was preferred over ITA** (per Researcher): ITA is 19% GE (commercial aviation) and dilutes the pure-defense signal. XAR is concentrated on aerospace + defense pure-plays.

## Today's result

```
defense_signal: triggered=False (GPR above=True current=285.35 thr=184.93; XAR 5d mom=-1.756% above=False); boost=1.000
```

Even though GPR is above threshold (geopolitical events are at elevated levels), XAR is pulling back -1.76% over the last 5 days — institutional money is NOT pricing the GPR signal into defense right now. The AND-gate correctly suppresses the boost. **This is the conservative behavior the design intends.**

## Score boosts (today: 0 applied)

| Ticker | Class | Score adjustment |
|---|---|---|
| LMT | US prime | 10.000 → 10.000 (no boost — gate not triggered) |
| NOC | US prime | 10.000 → 10.000 |
| RTX | US prime | 10.000 → 10.000 |
| GD | US prime | 10.000 → 10.000 |
| BAE.L | EU prime | 10.000 → 10.000 |
| RHM.DE | EU prime | 10.000 → 10.000 |
| AAPL | Non-defense | 10.000 → 10.000 (identity — not in defense list) |

## Hypothetical when triggered (synthetic GPR + positive XAR)

| Ticker | Boost applied |
|---|---|
| LMT, NOC, RTX, GD, LHX, BA, LDOS, HII, KTOS | **+5.0%** (each) |
| BAE.L, RHM.DE, SAAB-B.ST | **+5.0%** (EU primes recognized) |
| AAPL, MSFT, etc. | 0% (not in defense list) |

## Pledge keyword scan (optional, not gating)

Settings field `defense_budget_pledge_keywords` includes:
`NATO budget, defense spending, Zeitenwende, defense pledge, military spending, 5% GDP`

A callable `pledge_hit_provider` can be passed to `fetch_defense_trigger`; when not provided, `pledge_keyword_hit=False` (does NOT gate the trigger, just observability). News-scanning integration is a future-cycle item (could reuse news_screen plumbing).

## Cycle log (canonical)

When `settings.defense_signal_enabled=True`:

```
2026-05-17T23:35:00Z INFO defense_signal: defense_signal: triggered=False (GPR above=True current=285.35 thr=184.93; XAR 5d mom=-1.756% above=False); boost=1.000
2026-05-17T23:35:01Z INFO autonomous_loop: defense_signal_triggered=False xar_5d=-0.0176
```

When triggered:

```
2026-05-17T23:35:00Z INFO defense_signal: defense_signal: triggered=True (GPR above=True current=285.35 thr=184.93; XAR 5d mom=+2.500% above=True); boost=1.050
2026-05-17T23:35:01Z INFO screener: defense boost applied to 9 tickers (LMT,NOC,RTX,GD,LHX,BA,LDOS,HII,KTOS); EU tickers BAE.L,RHM.DE,SAAB-B.ST also eligible if in universe
```

## Provenance

- Code: new `backend/services/defense_signal.py` (180 lines, reuses `macro_regime._fetch_gpr_acts`); `backend/tools/screener.py` (+kwarg + apply); `backend/services/autonomous_loop.py` (+cycle-level pre-fetch + pass-through); `backend/config/settings.py` (+7 fields).
- Source: supplement Gap 1 + phase-28.14 research brief (6 sources read in full).
- Reused: phase-28.3 GPR fetcher (free, cached).
- Feature flag: `defense_signal_enabled = False` by default — production unchanged.

## Spec compliance

- "GPR-Acts value + ITA/XAR 5-day flow + any pledge headlines + resulting LMT/NOC/RTX/BAE/RHM score boosts" — DOCUMENTED above with: real GPR-Acts (285.35), XAR 5d (−1.76%), pledge keyword config + provider mechanism, score boosts table (today: 0 applied because AND-gate not met; hypothetical when triggered: +5%).
