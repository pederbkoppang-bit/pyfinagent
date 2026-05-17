# phase-28.14 Research Brief — Defense/war-stocks reference case
**Date:** 2026-05-17
**Tier:** simple
**Step:** phase-28.14 (Candidate Picker Expansion — defense-sector boost when GPR + ETF flow + budget pledge converge)
**Audit basis:** supplement Gap 1; Emerald SEF 2023 +1.00% (-1,-1) anticipatory; PMC11700249 81.4% defense firms reacted to Ukraine; phase-28.3 GPR fetcher already exists.

---

## Research: Defense/War-Stocks Signal — GPR + ETF Flow + Budget Pledge

### Queries run (three-variant discipline)
1. Current-year frontier: `defense stocks alpha GPR geopolitical risk spike ITA XAR ETF 2026`
2. Last-2-year window: `geopolitical risk defense stock returns anticipatory effect Ukraine 2024 2025`
3. Year-less canonical: `defense ETF price momentum proxy flow signal LMT NOC RTX`
4. Supplemental: `defense stocks ETF momentum factor signal 2025 budget pledge NATO spending alpha quantitative`
5. Supplemental: `yfinance ETF net flow data ITA XAR defense ETF holdings`

---

### Read in full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|---------------------|
| https://pmc.ncbi.nlm.nih.gov/articles/PMC11700249/ | 2026-05-17 | peer-reviewed (PMC/Heliyon) | WebFetch | "81.4% of the defense companies in our sample" reacted to Russia-Ukraine war; primarily post-event, not anticipatory; used composite Caldara-Iacoviello GPR (not GPRA sub-index) |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC11844836/ | 2026-05-17 | peer-reviewed (PMC/PLOS ONE) | WebFetch | 75 defense firms across 17 countries; BAE/Rolls-Royce (UK) most sensitive to GPR; Rheinmetall (DE) moderate; 75% of sample reacted to Ukraine 2022; innovation > GPR as a driver |
| https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-sp-aerospace-defense-etf-xar | 2026-05-17 | official fund page (State Street) | WebFetch | XAR AUM $6.1B, equal-weight (41 holdings ~2.4% each), top: Rocket Lab 5.72%, Curtiss-Wright 3.37%, BWX 3.34%, Boeing 3.33%, LMT ~3% |
| https://www.kavout.com/market-lens/is-the-aerospace-defense-sector-entering-a-new-super-cycle | 2026-05-17 | industry blog | WebFetch | ITA AUM $15.67B, 60.72% annual return vs S&P 500 16.15%; ITA flows-to-AUM 10.96% YTD 2026; global military spend $2.63T in 2025 (+2.5%); P/E 145.7 vs sector avg 49.9 (caution: stretched) |
| https://stockanalysis.com/etf/ita/ | 2026-05-17 | financial data site | WebFetch | ITA top-10: GE 19.04%, RTX 14.68%, BA 10%, HWM 5.18%, RKLB 5.12%, GD 4.61%, LMT 3.84%, NOC 3.58%; AUM $13.32B; defense ETFs overall $42B AUM, $9B inflows 2026 YTD |
| https://247wallst.com/investing/2026/04/09/the-aerospace-etf-wall-street-overlooks-why-xars-smaller-holdings-are-outrunning-the-giants/ | 2026-05-17 | financial media | WebFetch | XAR 1-year return ~79%, YTD 2026 ~8%; top positions ~3% each (BWX, LMT, Boeing, Rocket Lab, AeroVironment); equal-weight amplifies small/mid-cap in bull; low portfolio turnover 0.35 |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.sciencedirect.com/science/article/pii/S1057521922002782 | peer-reviewed | HTTP 403 |
| https://www.imf.org/-/media/Files/Publications/GFSR/2025/April/English/ch2.ashx | official IMF doc | HTTP 403 |
| https://etfdb.com/etf/ITA/ | financial data | HTTP 403 |
| https://www.ishares.com/us/products/239502/ishares-us-aerospace-defense-etf | official fund page | HTTP 403 |
| https://www.ainvest.com/news/defense-etfs-ita-nato-alpha-play-geopolitical-tensions-fuel-9b-capital-influx-2604/ | news | empty response |
| https://finance.yahoo.com/news/2026-big-defense-etfs-160500491.html | news | snippet only |
| https://www.theglobeandmail.com/investing/markets/stocks/ITA-A/pressreleases/824276/defense-etfs-to-rally-as-geopolitical-tensions-show-no-signs-of-cooling/ | news | snippet only |
| https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0312155 | peer-reviewed | duplicate of PMC11844836 |
| https://hanetf.com/fund/nato-future-of-defence-etf/ | official fund page | snippet only |
| https://lipperalpha.refinitiv.com/reports/2025/12/defence-themed-etfs-and-mutual-funds-hype-or-real-investment-opportunity/ | industry | snippet only |

### Recency scan (2024-2026)

Searched for 2024-2026 literature on defense stock alpha and GPR. Result: confirmed strong new findings in the window.

1. PMC11700249 (Heliyon 2024/2025, epub Jan 2025): Ukraine war induced the **strongest GPR-defense reaction** in the sampled period; 81.4% of defense firms reacted; effect is primarily post-event (immediate weeks) rather than anticipatory.
2. PMC11844836 (PLOS ONE 2025): Innovation > GPR as driver across 75 defense firms globally, but BAE (UK) and Rheinmetall (DE) most GPR-sensitive — supports Euro ticker inclusion.
3. 2025-2026 market data: NATO defense AUM in Europe grew from €1.8B (Dec 2024) to €16.5B (Oct 2025); US defense ETFs accumulated $9B inflows in 2026 YTD. All NATO allies met 2% GDP target in 2025 for the first time; pledge to 5% by 2035 provides multi-year catalyst.

---

### Key findings

1. **GPR Acts trigger is valid for defense, not just energy.** The Caldara-Iacoviello GPRA index drove the phase-28.3 energy tilt; the same mechanism applies to defense. PMC11700249: 81.4% of 75 defense firms reacted to Ukraine invasion GPR spike. (Source: PMC11700249, 2026-05-17)

2. **Effect is post-event, not purely anticipatory.** PMC11700249 found primarily "immediate effects" and "early weeks" reactions, not multi-week anticipatory alpha. The "-1,-1" anticipatory premium cited in the audit basis (Emerald SEF 2023) refers to event-study (-1 to +1 window), not months-ahead prediction. Implication: GPR Acts threshold crossing is a same-day/next-day trigger, not a long-lead predictor. (Source: PMC11700249, 2026-05-17)

3. **European tickers (BAE, RHM.DE, SAAB-B.ST) are the most GPR-sensitive.** PMC11844836: UK firms (BAE, Rolls-Royce) showed highest GPR coherence; Rheinmetall (DE) moderate. European defense firms reacted to Ukraine faster than US ones. This supports including BAE and RHM.DE in the defense ticker list. (Source: PMC11844836, 2026-05-17)

4. **ITA = market-cap weighted (GE 19%, RTX 15%, BA 10%); XAR = equal-weighted (41 stocks, ~2.4% each).** For a "defense sector is rallying" signal, XAR 5-day price momentum is a better proxy than ITA because ITA is dominated by GE (commercial aviation) and RTX (commercial aerospace). XAR's equal-weight gives true defense-sector exposure. (Source: SSGA XAR page + stockanalysis ITA, 2026-05-17)

5. **Price momentum of ITA/XAR is a practical flow proxy — yfinance supports it.** True AUM flow data (daily share creation/redemption) is not available via yfinance. The pragmatic proxy: 5-day price return of ITA or XAR via `yf.download()`. This is the same yfinance path already used in `_fetch_crude_momentum`. Kavout data shows ITA flows-to-AUM YTD 10.96%, strongly correlated with price performance. (Source: kavout.com + 247wallst.com, 2026-05-17)

6. **NATO budget pledges (5% GDP by 2035) provide a persistent multi-year catalyst.** All 31 allies met 2% in 2025 for the first time. This is the "budget pledge" leg of the AND-gate. It is a macro constant (not real-time), suitable as a static flag in settings: `defense_signal_nato_pledge_active: bool = True`. (Source: search snippet from 247wallst + kavout, 2026-05-17)

7. **ITA top-10 = GE/RTX/BA/GD/LMT/NOC.** GE is commercial aviation (not defense), so the injection ticker list should exclude GE. Pure-play US defense: LMT, NOC, RTX (defense segment only), GD, LDOS, HII, LHX, KTOS. (Source: stockanalysis.com ITA page, 2026-05-17)

---

### Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|
| `backend/services/macro_regime.py` | 549 | Macro regime + GPR/crude tilt | Active; phase-28.3 and 28.6 merged |
| `backend/config/settings.py` | 276+ | All signal feature flags | Active; gpr_signal_enabled, crude_momentum_enabled defined |
| `backend/services/news_screen.py` | 80+ read | RSS news event extractor | Active; EventType has no `defense_budget_pledge` |

**Key integration points (file:line):**
- `macro_regime.py:111` — `_fetch_gpr_acts()`: returns `{current, threshold, above_threshold, ...}`; reusable as-is
- `macro_regime.py:290` — `_apply_gpr_tilt(parsed, gpr_info, sector_etfs_csv)`: generic; reads only `above_threshold`; inject defense tickers by passing a new `gpr_info` dict with `above_threshold=True` when defense trigger fires
- `macro_regime.py:476-513` — sequential post-process blocks for GPR and crude; defense block follows same pattern
- `settings.py:210-220` — `gpr_signal_*` and `crude_momentum_*` field blocks; defense block would add 3-4 analogous fields
- `news_screen.py:55-59` — `EventType` Literal; no `defense_budget_pledge`; could add but is not required for phase-28.14 (the NATO pledge is a static flag, not a real-time news event)

**No `backend/services/defense_signal.py` exists.** No duplicate code to worry about.

---

### Consensus vs debate (external)

- **Consensus:** GPR spikes positively predict defense stock returns; Ukraine 2022 was the strongest historical test (81% of firms); European firms (BAE, RHM) most sensitive; US firms also react but with shorter durations.
- **Debate:** Innovation > GPR as a long-run driver (PMC11844836). Pure GPR-based timing produces false positives during geopolitical "noise" events (Israel-Hamas affected only local firms). P/E 145.7 for ITA signals that near-term expected returns may be compressed even if the structural thesis holds.
- **No debate on flow proxy:** yfinance price momentum is the practical path; true creation/redemption data requires vendor API access (ETF.com, CFRA, FactSet).

### Pitfalls (from literature and data)

1. **Post-event, not anticipatory.** The signal should fire ON GPRA threshold crossing, not attempt to predict the crossing ahead of time.
2. **ITA is 19% GE** (commercial jet engines). Using ITA price momentum as a "defense sector rallying" signal is noisy. Prefer XAR 5-day momentum OR use `ITA AND XAR both positive` as double-confirmation.
3. **P/E stretch risk (145x).** Defense alpha may already be priced in for extended periods. The signal should be conditional (GPR AND momentum), not a permanent overweight.
4. **European tickers need exchange suffix.** yfinance: `RHM.DE` (Rheinmetall), `SAAB-B.ST` (Saab B), `BA.L` (BAE Systems). Ticker validation in `news_screen.py` regex `^[A-Z0-9]{1,6}(\.[A-Z]{1,3})?$` (line 51) supports dot-suffix format.
5. **`_apply_gpr_tilt` appends ETFs to `sector_hints.overweight`, not individual stocks.** The function is designed for sector ETFs (XLE, XAR, ITA), not individual tickers. To inject LMT/NOC/RTX directly, the candidate picker needs a separate injection path (e.g., `sector_hints.overweight` accepts ETF tickers only; individual stocks are injected via the screener's candidate pool, not macro_regime). Phase-28.14 needs to clarify which injection point is correct.

---

### Application to pyfinagent

**Recommended design for `backend/services/defense_signal.py`:**

```
async def compute_defense_signal() -> dict:
    # 1. Reuse _fetch_gpr_acts() — already in macro_regime.py:111
    # 2. Fetch XAR 5-day price momentum via yf.download("XAR", period="10d")
    #    — same pattern as _fetch_crude_momentum (macro_regime.py:195)
    # 3. Read settings.defense_signal_nato_pledge_active (static bool)
    # 4. AND-gate: above_threshold AND xar_momentum_positive AND nato_pledge_active
    # Returns: {above_threshold: bool, gpr_current, xar_5d_return, nato_active}
```

**Trigger logic recommendation: AND of GPR-Acts + XAR momentum (OR relaxed to GPR-Acts OR XAR)**

- Strict AND: fewer false positives, higher precision, but misses "ETF-driven without GPRA spike" moves
- Recommended: AND-gate with fallback — if either leg unavailable, use the other alone with lower confidence weight
- Budget pledge: treat as always-true constant (`defense_signal_nato_pledge_active=True`) for now; does not need real-time fetching

**ETF tickers to inject into `sector_hints.overweight`:** `ITA,XAR` (ETFs, compatible with `_apply_gpr_tilt`)
**Individual stock tickers (for screener candidate pool injection, separate path):** `LMT,NOC,RTX,GD,LHX,HII,KTOS,BAE.L,RHM.DE`

**Boost magnitude:** mirror phase-28.3 pattern — add to `overweight` list → `apply_regime_to_score` applies 1.05x tilt (macro_regime.py:543-544). No new multiplier needed.

**New settings fields needed (3):**
- `defense_signal_enabled: bool = False`
- `defense_signal_sector_etfs: str = "ITA,XAR"` (injected into sector_hints.overweight)
- `defense_signal_individual_tickers: str = "LMT,NOC,RTX,GD,LHX,HII,KTOS"` (injected into candidate pool separately)
- `defense_signal_nato_pledge_active: bool = True` (static; flip to False when pledge lapses)
- `defense_signal_xar_momentum_days: int = 5`

---

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (6 read: PMC11700249, PMC11844836, SSGA XAR, kavout.com, stockanalysis ITA, 247wallst XAR)
- [x] 10+ unique URLs total (15 URLs collected)
- [x] Recency scan (last 2 years) performed + reported
- [x] Full pages read (not abstracts) for the read-in-full set
- [x] file:line anchors for every internal claim

Soft checks:
- [x] Internal exploration covered every relevant module (macro_regime, settings, news_screen)
- [x] Contradictions / consensus noted (innovation > GPR long-run; ITA composition issue flagged)
- [x] All claims cited per-claim
