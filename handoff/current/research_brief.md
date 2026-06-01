# research_brief -- phase-51.2: sector diversification (research-recommended money lever)

**Tier:** complex | **Date:** 2026-06-01 | **Researcher:** Layer-3 Harness
**North-star:** maximize live money at lowest cost; amplify the WORKING US momentum
engine's BREADTH inside `screener.rank_candidates` (the live-orders path), NOT
winner-take-all rotation (architecturally disconnected + money-losing per the
rotation-research verdict at `handoff/.../research_rotation_element2_verdict.md`).
**Constraint:** the working US momentum core must NOT regress -- every change is
config-gated, default-OFF, backtest-proven before any live enable. $0 LLM.

Status: COMPLETE.

---

## PART A -- INTERNAL CODE AUDIT (pinned to file:line)

### A0 -- THE HEADLINE (one-paragraph)

Both levers ALREADY EXIST and are ALREADY WIRED into the live path. The live
`rank_candidates` call (`backend/services/autonomous_loop.py:621`) ALREADY passes
`sector_neutral=settings.sector_neutral_momentum_enabled` (`:629`) and
`multidim_momentum=settings.multidim_momentum_enabled` (`:632`). The
`rank_candidates` function ALREADY implements both paths (`screener.py:415-445`
sector-neutral within-sector percentile; `:401-413`+`:464-523` multidim z-blend).
**The ONE thing that's broken is data timing:** the within-sector percentile path
groups candidates by `s.get("sector")` (`screener.py:425`), but `screen_universe`
is called at `:369` WITHOUT a `sector_lookup`, and sector enrichment only happens
at `:659-676` -- AFTER `rank_candidates` has already returned. So at rank time every
candidate has `sector = None`, every candidate falls into the `_UNKNOWN_` group
(`screener.py:425`), and the sector-neutral path silently collapses to a single
global percentile pool (`:431-433`) -- a no-op that re-ranks identically to raw
momentum. **The fix is a one-line-ish data-timing change: build the ticker->sector
map BEFORE `rank_candidates` and pass it as `sector_lookup=` to `screen_universe`
(or attach sectors to `screen_data` before ranking).** No new scoring code is
needed; the scoring code is correct and tested -- it just never receives sectors.

### A1 -- `screener.rank_candidates`: the three relevant code paths

**Signature** (`screener.py:222-246`): `rank_candidates(screen_data, top_n=10,
strategy="momentum", ..., sector_neutral=False, sector_neutral_min_group_size=3,
..., multidim_momentum=False, multidim_weights=None, ...)`. Both levers are
first-class params, default-OFF.

**(1) Base momentum score** (`screener.py:268-282`, the path the caller cited):
```
score = mom_1m*0.40 + mom_3m*0.35 + mom_6m*0.25
if rsi > 80: score *= 0.7   elif rsi < 20: score *= 0.8
if vol > 0.6: score *= 0.85
```
No sector term. Pure cross-sectional price momentum. This is what runs live today
(both flags OFF). Per the diagnostic, non-tech sectors are OUT-COMPETED here (a
high-momentum tech name simply scores higher than a steady industrial), not excluded.

**(2) sector_neutral within-sector percentile** (`screener.py:415-445`). When
`sector_neutral=True` and `scored` is non-empty:
- Group `scored` by `key = (s.get("sector") or "").strip() or "_UNKNOWN_"` (`:424-425`).
- Any group with `key == "_UNKNOWN_"` OR `len(members) < sector_neutral_min_group_size`
  (default 3) is pulled into a `global_pool` and deleted from the per-sector map
  (`:430-433`).
- Each remaining per-sector group + the global_pool gets `composite_score` REPLACED
  by `pandas.Series.rank(method="average", pct=True)` -- a within-group percentile in
  [0,1] (`:435-445`). Original preserved on `composite_score_raw`.
- **Effect when sectors ARE present:** the top momentum name in EVERY sector gets
  ~1.0, so the final top_n is spread across sectors instead of dominated by the one
  hottest sector. This is exactly the breadth lever phase-51.2 wants.
- **Effect TODAY (sectors absent at rank time):** every candidate -> `_UNKNOWN_` ->
  global_pool -> ONE global percentile pool. Percentile-ranking a single pool by the
  raw composite preserves the raw ordering exactly (monotone transform) -> `top_n`
  is byte-identical to flags-OFF. **THIS is why it no-ops.** Confirmed: the no-op is
  not a bug in the percentile math; it's that `s.get("sector")` is `None` for all.

**(3) multidim_momentum z-blend** (`screener.py:401-413` call site + `:464-523`
`_apply_multidim_momentum`). When `multidim_momentum=True`: replaces `composite_score`
with a 4-component cross-sectional z-blend: `w_price*z(price) + w_high*z(52w_high) +
w_sue*z(SUE) + w_sector*z(sector_boost)` (default weights 0.35/0.25/0.20/0.20). The
"sector" component is `sector_momentum_ranks[sector].boost_multiplier - 1.0`
(`:499-506`) -- i.e. it needs BOTH a `sector` field on the stock AND a populated
`sector_momentum_ranks` dict (the phase-28.12 sector-ETF momentum overlay). When the
stock has no sector OR the ranks dict is None, `z_sector` is all-zeros (`:505-506`),
so the sector component drops out and multidim reduces to a 3-component (price +
52w-high + SUE) blend. **multidim is NOT primarily a sector-diversification lever** --
it's a momentum-quality refinement (anchoring + earnings surprise). Its sector
component is a sector-MOMENTUM TILT (overweight hot sectors), the OPPOSITE of
sector-NEUTRALITY (equal-weight across sectors). See A4.

### A2 -- The wiring gap: WHERE candidates get a sector, and WHEN

**`screen_universe` CAN attach sectors** (`screener.py:64-72` signature has
`sector_lookup: Optional[dict] = None`; `:203-213` attaches `row["sector"] =
meta.get("sector")` when the lookup is provided). **But the live caller does NOT
pass it:** `autonomous_loop.py:369-374` calls `screen_universe(tickers=universe,
period="6mo", short_interest_lookup=..., short_interest_threshold=...)` -- NO
`sector_lookup=`. So `screen_data` rows have no `sector` key.

**Sector enrichment DOES happen -- but too late:** `autonomous_loop.py:659-676`,
AFTER `rank_candidates` returns at `:621-651`. It calls
`_fetch_ticker_meta(top_tickers, settings, bq)` (only the top-N survivors) and writes
`c["sector"]` onto the already-ranked candidates so the DOWNSTREAM sector cap in
`decide_trades` works. The comment at `:657-658` literally says "Without this
enrichment, decide_trades sees `sector=None`" -- it was added for the position-sizing
sector CAP, not for ranking. Ranking already happened.

**`_fetch_ticker_meta` cost/source** (`backend/api/paper_trading.py:1058-1175`):
- **BQ-FIRST**, yfinance fallback. Step 1 (`:1079-1142`): ONE BigQuery query that
  UNIONs `paper_positions` (priority 1) + `analysis_results` (priority 2), returns the
  highest-priority non-null `{company_name, sector}` per ticker. One round-trip for
  the whole batch.
- Step 2 (`:1144-1173`): for tickers still missing a sector, parallel yfinance
  `.info` via `ThreadPoolExecutor(max_workers=5)`. Comment `:1149`: ~14 tickers ~3s.
- Result cached 24h (`ttl_sec=86400`, `:1175`). **Cost to build the map: $0** (BQ +
  yfinance, no LLM). The expensive leg is yfinance `.info` per missing ticker.

**Cost of moving enrichment BEFORE ranking -- the ONE real consideration:** today
enrichment runs on the **top_n survivors only** (~10-30 tickers). To give candidates
a sector AT RANK TIME, you must resolve sectors for the **full screened set**
(`screen_data`, which is the whole S&P 500 universe that passed price/volume
filters -- potentially ~400-500 tickers), because ranking happens before the top-N
cut. If those sectors aren't already in BQ (`paper_positions`/`analysis_results`),
each miss is a yfinance `.info` call. **Mitigation: the BQ leg covers most S&P 500
names cheaply in one query; only the residual misses hit yfinance, and the 24h cache
amortizes after cycle 1.** A cleaner option is a CHEAP STATIC sector map (S&P 500
GICS sectors change rarely) loaded once -- see A2-recommendation below. Either way
the cost is $0-LLM and bounded; the open question is purely yfinance latency on a
cold cache for the full universe.

**A2 minimal-wiring recommendation (the EXACT change):** Build a ticker->sector
lookup for the FULL `universe` BEFORE `screen_universe` at `autonomous_loop.py:369`,
and pass it as `sector_lookup=`. Concretely:
```
# autonomous_loop.py, just before :369
sector_lookup = await asyncio.to_thread(_build_universe_sector_map, universe, settings, bq)
screen_data = screen_universe(
    tickers=universe, period="6mo",
    sector_lookup=sector_lookup,            # <-- THE NEW ARG (screener.py:69 already accepts it)
    short_interest_lookup=short_interest_lookup or None,
    short_interest_threshold=getattr(settings, "short_interest_threshold", 0.10),
)
```
where `_build_universe_sector_map` is `_fetch_ticker_meta` reduced to `{ticker:
sector}` (it already returns sector; just project the field). Then `screen_data`
rows carry `sector` (`screener.py:206-213`), so when `sector_neutral=True` the
percentile groups by real GICS sectors instead of all-`_UNKNOWN_`. **The post-rank
enrichment at `:659-676` stays as-is** (it now mostly hits cache and still serves the
downstream sector cap). **This is the entire wiring change** -- no `rank_candidates`
signature change, no new scoring code, gated by the EXISTING
`sector_neutral_momentum_enabled` flag (default OFF). For the live-money EU/KR
universe (phase-50), `_fetch_ticker_meta`/BQ may not have intl sectors; a static
DAX-40/KOSPI-200 GICS map (mirroring the curated ticker lists) is the robust source.

### A3 -- How a BACKTEST measures sector-neutral ON vs OFF (the hard part)

**The live screener (`screener.rank_candidates`) is DIFFERENT CODE from the backtest
engine's candidate selector.** Confirmed:
- The backtest engine selects candidates via `CandidateSelector.screen_at_date(...)`
  (`backtest_engine.py:402,464`) -> `CandidateSelector._rank_candidates`
  (`candidate_selector.py:175-206`). That ranker uses a COMPLETELY DIFFERENT formula:
  `mom_6m/100*0.4 + rsi_meanrev*0.2 + inverse_vol*0.2 + sma*0.2` (`:198-204`). It has
  **NO `sector_neutral` param, NO `multidim` param, and never reads a `sector`
  field.** The backtest engine then trains a `GradientBoostingClassifier` on label
  functions -- a totally separate selection mechanism from the live screener.
- Therefore **the existing backtest engine CANNOT exercise the live screener's
  sector-neutral path.** Running `engine.run_backtest()` with
  `sector_neutral_momentum_enabled=True` would change NOTHING in the backtest, because
  the engine never calls `screener.rank_candidates`. A standard walk-forward backtest
  is the WRONG instrument here.
- The feature-ablation harness (`scripts/ablation/run_ablation.py`) is ALSO the wrong
  instrument -- it ablates `_NUMERIC_FEATURES` in the backtest engine's feature
  matrix (`:70,233,253`), not the live screener.

**=> The ON-vs-OFF comparison needs a SCREENER-LEVEL backtest/harness, not the ML
backtest engine.** The cheapest VALID measurement (see A3-design below) is a
**point-in-time replay of `screen_universe` + `rank_candidates` itself** over a set
of historical dates, computed two ways (flag OFF vs ON), scoring the realized
forward return of each method's top_n. This is a NEW, small harness -- but it is $0
(yfinance prices only, no LLM, no BQ-heavy ML) and it is the ONLY artifact that
actually proves the live lever's Sharpe/return/sector-spread tradeoff without
flipping a live flag.

**A3 cheapest-valid measurement design (the artifact masterplan-51.2 criterion #2
should require):** A `scripts/ablation/sector_neutral_replay.py` (mirrors the
existing ablation runner's shape -- TSV out, verdict gate) that:
1. Picks K historical rebalance dates (e.g. monthly, 2023-01..2025-12, ~36 dates) on
   the S&P 500 universe.
2. At each date `t`: download trailing-6mo prices ending at `t` (yfinance, the SAME
   data `screen_universe` uses), build `screen_data`, attach REAL sectors via a static
   GICS map (or `_fetch_ticker_meta`). Run `rank_candidates(..., sector_neutral=False)`
   -> top_n_OFF; run `rank_candidates(..., sector_neutral=True)` -> top_n_ON. (Reuse the
   PRODUCTION `rank_candidates` so the harness measures the real code, not a reimpl.)
3. Compute each basket's realized forward return over the holding horizon (e.g. 21
   trading days fwd, equal-weight), the basket's number of distinct GICS sectors
   (the breadth metric), and turnover vs the prior period's basket.
4. Aggregate across dates: mean/Sharpe of the forward-return series for OFF vs ON,
   mean sector-count (spread), mean turnover. Log to
   `sector_neutral_replay_results.tsv` with a verdict.
- **Acceptance/gate (mirror `run_ablation.py:187-192`):** ON is "keep" if it raises
  sector spread materially (e.g. +>=2 distinct sectors in the top_n) AND does NOT cut
  the forward-return Sharpe by more than a small floor (e.g. delta_sharpe >= -0.05);
  "discard" if it hurts Sharpe with no breadth gain. This is exactly the
  Sharpe/turnover/spread tradeoff the literature predicts (Part B), measured on the
  REAL live code, at $0, WITHOUT touching the live flag.
- **Why this is the cheapest valid design:** it reuses production `rank_candidates`
  (no reimplementation risk), uses only free yfinance price data (no LLM, no
  ML-training cost), and is a pure cross-sectional replay (~36 dates x one yfinance
  batch each -- minutes, not the ~5-10min-per-iteration ML backtest). A full ML
  walk-forward would be both more expensive AND invalid (wrong code path).
- NOTE on rigor: this replay is a SIGNAL-LEVEL test (does the basket's forward return
  hold up; does breadth rise), not a full portfolio simulation (no position sizing,
  no sector cap, no commissions). That is the correct scope for proving a RANKING
  change. If a fuller proof is wanted later, the replay baskets can be fed through
  `BacktestTrader` -- but that is gold-plating for the ON-vs-OFF ranking question.

### A4 -- sector_neutral vs multidim_momentum: which is the better FIRST lever?

**RECOMMENDATION: sector_neutral within-sector percentile is the better first lever.**
Justification (simpler, lower-regression-risk, directly on-thesis):
1. **It is the lever that matches the diagnostic.** Finding 4 is "non-tech sectors are
   OUT-COMPETED by pure momentum." Sector-neutral percentile DIRECTLY fixes this: it
   ranks within each sector, so the best industrial competes against industrials, not
   against NVDA. multidim's sector component is a sector-MOMENTUM TILT (overweight hot
   sectors via `boost_multiplier`, `screener.py:499-506`) -- it would AMPLIFY
   concentration in the hot sector, the OPPOSITE of breadth. multidim is a
   momentum-quality upgrade, not a diversification lever.
2. **Lower regression risk / simpler to reason about.** sector_neutral is a single
   monotone re-grouping with a clean fallback (`<min_group_size` -> global pool,
   `screener.py:430-433`); when sectors are absent it provably no-ops (identity). It
   has ONE new dependency: sectors at rank time (the A2 wiring). multidim replaces the
   composite with a 4-way z-blend that depends on `pct_to_52w_high`, `pead`
   `surprise_score`, AND `sector_momentum_ranks` -- three signal sources, more moving
   parts, more ways to silently degrade, and it changes the score for EVERY candidate
   (not just regrouping), a larger behavioral delta against the working engine.
3. **It is research-supported for THIS goal.** Part B: industry/sector-relative
   (sector-neutral) momentum has long literature support (Moskowitz-Grinblatt 1999;
   Asness-Moskowitz-Pedersen 2013) and the 2026 low-correlation-breadth consensus
   (AQR/UBP). multidim's components (52w-high George-Hwang; SUE) are momentum
   ENHANCERS, not breadth/diversification levers.
4. **Both are already gated and wired** -- so choosing sector_neutral first costs
   nothing in optionality; multidim remains available as a follow-on momentum-quality
   experiment.

**One nuance:** sector_neutral and multidim are NOT mutually exclusive long-term
(you can sector-neutralize THEN z-blend), but for a first, low-regression,
on-thesis money lever, ship sector_neutral alone, prove it on the A3 replay, then
consider multidim separately.

---

## PART B -- EXTERNAL RESEARCH

### Read in full (>=5 required; 9 read; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key finding |
|-----|----------|------|-------------|-------------|
| https://quantpedia.com/should-factor-investors-neutralize-the-sector-exposure/ | 2026-06-01 | practitioner summary of peer-reviewed (Harvey et al.) | WebFetch (full) | **THE LOAD-BEARING / ADVERSARIAL FINDING.** "Keeping the across [sector] component produces better long-short factors in only 20% of the trials, while doing so delivers better long-only factors in 78% of the trials." => "the long-only investor is more likely to benefit from investing in the factor as it stands [NOT sector-neutralized]." Decision rule: neutralize only if "the ratio of the Sharpe ratios across and within components is less than their correlation." |
| https://people.duke.edu/~charvey/Research/Published_Papers/P165_Is_sector_neutrality.pdf | 2026-06-01 | peer-reviewed (Harvey, Duke; primary PDF) | WebFetch (full, binary-extracted summary) | Sector neutralization is a NUANCED trade-off (can enhance OR diminish). Momentum "demonstrates strong sector effects that make neutralization especially consequential." "higher turnover and transaction costs represent a significant practical drawback." Long-short benefits from neutralizing; long-only often does not. |
| https://blogs.cfainstitute.org/.../momentum-investing-a-stronger-more-resilient-framework... (-> rpc.cfainstitute.org) | 2026-06-01 | official/industry (CFA Institute, Dec 2025) | WebFetch (full, via redirect) | Multidim composite (price + 10 alt signals) "delivers higher average returns, stronger t-statistics, and substantially improved drawdown" vs price momentum since 1927. **Median Sharpe 0.61, range 0.38-0.94.** Vol-scaling -> "annualized returns of nearly 18% ... drawdowns cut nearly in half." Price momentum max DD as large as **-88%**. This is the SAME source the project already cites for sector_neutral + multidim (`settings.py:324,334`). |
| https://www.aqr.com/Insights/Research/Journal-Article/Do-Industries-Explain-Momentum | 2026-06-01 | peer-reviewed (Moskowitz-Grinblatt 1999, AQR host) | WebFetch (full) | "industry momentum ... captures these [stock-momentum] profits almost entirely" except 12-month. Individual-stock momentum is "significantly less profitable once we control for industry momentum." Momentum yields "up to 12 percent abnormal return per dollar long." => industries are a DOMINANT driver of momentum; sector structure matters a lot to momentum. |
| https://www.evidenceinvestor.com/post/factor-momentum-and-stock-momentum | 2026-06-01 | practitioner summary of peer-reviewed (Ehsani-Linnainmaa, JoF 2022) | WebFetch (full) | "Factor momentum explains all forms of individual stock momentum." "Industry momentum stems from factor momentum." "Factor returns are persistent at monthly time scales, while stock returns mean revert." Equal-risk factor avg Sharpe **0.96**. => industry momentum is real but is itself a manifestation of factor momentum. |
| https://arxiv.org/html/2503.09647 | 2026-06-01 | peer-reviewed preprint (q-fin, 2025) | WebFetch (full, arXiv HTML) | Sector-aware allocation Sharpe **2.51 / +8.79%** vs cross-momentum (sector-blind) **-0.61 / -1.39%**. BUT **HUGE CAVEAT: backtest window is only Jan-Jun 2019** (6 months) -> not generalizable across cycles. Costs modeled at 10bps commission + 10bps impact. Sector-aware "more effective at capturing market opportunities while managing volatility." |
| https://quantpedia.com/strategies/sector-momentum-rotational-system | 2026-06-01 | practitioner (Quantpedia, replicated) | WebFetch (full) | Sector-ETF momentum rotation (top-3 of 11 SPDRs, monthly): Sharpe **0.54**, CAGR 13.94%, **max DD -46.29%**, "overperformance nearly 4% against simple buy and hold." Monthly rebalance = HIGH turnover. This is the phase-28.12 sector-momentum overlay's basis -- a sector TILT, not sector-neutrality. |
| https://www.quantseeker.com/p/popular-investing-research-in-2025 | 2026-06-01 | practitioner (2025 research recap) | WebFetch (full) | "Momentum at Long Holding Periods" (Calluzo-Moneta-Topaloglu): rankings persist -> "longer holding periods with lower turnover ... improved implementability for real portfolios." Scaled factor portfolios "Sharpe ratios up to 2." (Recency: 2025 momentum frontier = lower turnover + composite signals.) |
| https://am.gs.com/en-us/advisors/insights/article/2026/technology-2026-ai-dispersion-diversification | 2026-06-01 | official/industry (Goldman Sachs AM, Jan 2026) | WebFetch (full) | "Since the end of 3Q25, dispersion among the Magnificent 7 has widened to **52.3%**" (Jan 13 2026). Advocates "active diversification across industries." Mega-caps DIVERGING strategically. (2026 recency: concentration unwinding -> breadth more relevant; but GS frames it as WITHIN-AI dispersion.) |

### Identified but snippet-only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|-------------------------|
| https://www.nber.org/system/files/working_papers/w25551/w25551.pdf | peer-reviewed (Ehsani-Linnainmaa NBER w25551) | binary PDF, no text extracted; HTML/summary read via evidenceinvestor instead; AEA HTML mirror exists at aeaweb.org/conference/2020/.../RHhbnykd |
| https://onlinelibrary.wiley.com/doi/full/10.1002/for.3232 | peer-reviewed (Mamais 2025, momentum shifts across sectors) | Wiley HTTP 402 paywall; snippet confirms momentum performance VARIES across sectors + time (the premise sector-neutral exploits) |
| https://alphaarchitect.com/factor-investing-and-sector-neutrality/ | practitioner summary (Harvey et al.) | HTTP 403; the QuantPedia summary of the SAME paper was read in full instead |
| https://www.morningstar.com/portfolios/these-diversification-strategies-are-winning-2026 | industry (2026 diversification) | HTTP 403 on both URL forms; WebSearch snippet carries the load-bearing stat (US most concentrated in 10 largest names since 1932; 2026 mega-cap rotation in-the-red) |
| https://www.ishares.com/.../spring-2026-investment-outlook-inflation-ai-markets | industry (BlackRock 2026) | HTTP 403; value-sector diversification recommendation captured via WebSearch snippet |
| https://www.ainvest.com/news/...sector-rotation-selective-ai-exposure-2601/ | industry (2026 sector rotation) | JS-rendered, returned empty body on 3 attempts; WebSearch snippet used for recency context |
| https://quantstreet.substack.com/p/industry-momentum | practitioner (Mamaysky industry momentum) | HTTP 404 (substack slug moved); core industry-momentum result covered by AQR Moskowitz-Grinblatt read |
| https://arxiv.org/html/2511.12490v1 | peer-reviewed preprint (drift-regime cross-sectional factor, 2025) | identified in search; off-core (drift regimes, not sector-neutral); not load-bearing for this step |
| https://quantpedia.com/strategies/momentum-factor-effect-in-stocks | practitioner (Quantpedia) | read in full actually (Sharpe ~0.5, CAGR 8.3%, **max DD -87.41%**, Barroso-Santa-Clara vol-management "nearly doubles Sharpe"); listed here as it corroborates the CFA crash-risk numbers rather than adding sector-neutral evidence |

### Search-query variants run (3-variant discipline)
1. **Current-year frontier (2026):** "sector diversification breadth concentrated AI tech-led market low correlation durable 2026 systematic equity" (-> Goldman Sachs 2026, Morningstar 2026, iShares Spring 2026, AInvest 2026).
2. **Last-2-year window (2025):** "industry-neutral momentum strategy improves Sharpe reduces turnover 2025 quant equity" (-> CFA Institute Dec 2025, QuantSeeker 2025 recap, Mamais 2025 Wiley) -- see Recency scan.
3. **Year-less canonical:** "Moskowitz Grinblatt do industries explain momentum"; "sector-neutral momentum versus cross-sectional momentum risk-adjusted return Sharpe industry-relative"; "sector neutralization momentum factor reduces concentration turnover cost Sharpe Asness 2013" (-> Moskowitz-Grinblatt 1999, Ehsani-Linnainmaa, **Harvey et al. Duke sector-neutrality** -- the year-less query is what surfaced the decisive long-only finding).

### Recency scan (2024-2026) -- PERFORMED
Searched the last-2-year window on (a) industry/sector-neutral momentum 2025, (b) sector diversification / breadth in concentrated AI-led markets 2026, (c) momentum turnover reduction. **Findings (COMPLEMENT the canonical Moskowitz-Grinblatt / Harvey prior art; the 2026 macro context REINFORCES breadth but does not change the long-only caveat):**
1. **CFA Institute (Dec 2025)** is the freshest rigorous source and is ALREADY the project's cited basis for both `sector_neutral_momentum_enabled` and `multidim_momentum_enabled` (`settings.py:324,334`). CONFIRMED its claims: multidim composite > price-only (higher returns, stronger t-stats, better drawdown); median Sharpe 0.61 (range 0.38-0.94); vol-scaling halves drawdown. It does NOT, however, quantify sector-NEUTRAL specifically -- the project's citation slightly over-reaches (CFA supports the multidim composite and vol-scaling explicitly; the sector-neutral-specific evidence is Harvey et al., which is more equivocal for long-only).
2. **2025 momentum frontier = lower turnover + composite signals** (QuantSeeker 2025: "Momentum at Long Holding Periods" -> longer holds, lower turnover, better implementability). This is RELEVANT: sector-neutralization INCREASES turnover (Harvey et al.), cutting against the 2025 implementability trend. A sector-neutral lever should be paired with turnover awareness.
3. **2026 macro context strongly favors breadth** (Goldman Sachs: Mag-7 dispersion 52.3%; Morningstar: most concentrated since 1932, 2026 mega-cap rotation in-the-red; iShares: value/non-US sector diversification). This REINFORCES the strategic case for spreading the momentum book across sectors RIGHT NOW -- the concentration is unwinding, so a sector-blind momentum book that piled into mega-cap tech is most exposed exactly when leadership rotates.
4. **No 2024-2026 source overturns** Moskowitz-Grinblatt (industries drive momentum) or Harvey et al. (sector-neutralization is conditional, momentum-sensitive, turnover-costly, and long-only-equivocal). They sharpen it: the 2026 concentration context raises the EXPECTED benefit of breadth, while the long-only caveat and turnover cost remain the binding constraints.

### Key findings (per-claim, cited)
1. **Industries are a DOMINANT driver of momentum -> sector structure is highly material to a momentum book.** "industry momentum ... captures these [stock-momentum] profits almost entirely" (Source: Moskowitz-Grinblatt 1999, https://www.aqr.com/Insights/Research/Journal-Article/Do-Industries-Explain-Momentum, accessed 2026-06-01). A pure cross-sectional momentum screen is implicitly a SECTOR BET; making the sector dimension explicit (neutral or tilt) is therefore a first-order, not cosmetic, change.
2. **Sector-neutralizing momentum is a CONDITIONAL win, and for a LONG-ONLY book it is more often NOT beneficial.** "Keeping the across [sector] component produces better long-short factors in only 20% of the trials, while doing so delivers better long-only factors in 78% of the trials" (Source: Harvey et al. via QuantPedia, https://quantpedia.com/should-factor-investors-neutralize-the-sector-exposure/, accessed 2026-06-01). pyfinagent's screener is **LONG-ONLY** -> the literature's base rate says keeping the across-sector (i.e. NOT fully neutralizing) book is better ~78% of the time. **This is the single most important caveat in this brief.** The exact decision rule: neutralize only if "the ratio of the Sharpe ratios across and within components is less than their correlation" (Source: Harvey et al. primary PDF, https://people.duke.edu/~charvey/Research/Published_Papers/P165_Is_sector_neutrality.pdf).
3. **Sector neutralization increases turnover and transaction costs -- a material practical drawback.** "higher turnover and transaction costs represent a significant practical drawback" (Source: Harvey et al. Duke PDF, same URL). This compounds with the 2025 implementability trend toward LOWER turnover (Source: QuantSeeker 2025, https://www.quantseeker.com/p/popular-investing-research-in-2025).
4. **Where sector-aware allocation DID help, the uplift was large but the evidence is thin/short-window.** Sector-aware Sharpe 2.51 vs sector-blind -0.61 (Source: arXiv 2503.09647, https://arxiv.org/html/2503.09647) -- BUT on a 6-month 2019 window only (not generalizable). Sector-ETF momentum rotation Sharpe 0.54, +~4% vs buy-and-hold, but max DD -46% and high turnover (Source: Quantpedia, https://quantpedia.com/strategies/sector-momentum-rotational-system). The realistic, durable uplift is single-digit, turnover-sensitive -- consistent with the rotation brief's conclusion.
5. **The multidim composite (the OTHER lever) has stronger, longer-horizon evidence than sector-neutral -- but it is a momentum-QUALITY lever, not a breadth lever.** "delivers higher average returns, stronger t-statistics, and substantially improved drawdown" since 1927; median Sharpe 0.61 (0.38-0.94); vol-scaling halves drawdown (Source: CFA Institute Dec 2025, rpc.cfainstitute.org). Its sector COMPONENT is a sector-momentum TILT (overweight hot sectors), which AMPLIFIES concentration -- the opposite of breadth.
6. **The 2026 macro backdrop raises the value of breadth specifically now.** US "more concentrated ... than since 1932"; 2026 mega-cap rotation "in-the-red"; Mag-7 dispersion 52.3% (Sources: Morningstar 2026 snippet; Goldman Sachs 2026, https://am.gs.com/en-us/advisors/insights/article/2026/technology-2026-ai-dispersion-diversification). A sector-blind momentum book is maximally exposed to mega-cap tech right as leadership rotates -> breadth has elevated EXPECTED value, even if the long-only base rate (finding 2) argues for a PARTIAL tilt rather than full neutralization.

### Consensus vs debate (external)
- **Consensus:** (a) industries/sectors are a dominant component of momentum (Moskowitz-Grinblatt; Ehsani-Linnainmaa) -- the sector dimension is first-order; (b) momentum carries severe crash risk (-87% to -88% max DD) that vol-scaling roughly halves (CFA, Quantpedia/Barroso-Santa-Clara); (c) a multi-signal momentum composite beats price-only momentum (CFA); (d) 2026 macro favors breadth as concentration unwinds (GS, Morningstar, iShares).
- **Debate / the binding nuance:** **does sector-NEUTRALIZING help a LONG-ONLY momentum book?** Harvey et al. say it is CONDITIONAL and, for long-only, beneficial in only ~22% of trials (keeping the across-sector book wins 78%). Sector neutralization also raises turnover. So the literature does NOT give a clean "sector-neutralize and Sharpe goes up" for pyfinagent's long-only setting -- it says "it depends, and the default lean for long-only is to keep some across-sector exposure." This is the precise reason the A3 backtest/replay is NON-OPTIONAL: the sign of the effect for THIS book on THIS universe must be measured, not assumed.

### Pitfalls (from literature) -> applied to phase-51.2
1. **Do NOT assume sector-neutral raises Sharpe -- for a long-only book the base rate is the opposite.** (Harvey et al.) The project's `settings.py:324` description ("Improves Sharpe ... per CFA Institute Dec 2025") is OPTIMISTIC and not precisely supported for the long-only sector-NEUTRAL case (CFA supports the composite + vol-scaling, not sector-neutral-for-long-only). MEASURE before believing the description.
2. **Full sector-neutralization is the aggressive version; a PARTIAL sector tilt/cap is the lower-regression first move.** Harvey's long-only result (keep across-sector 78% of the time) argues for a soft constraint (e.g. cap per-sector weight, or blend within-sector percentile with the raw score) rather than the existing hard within-sector-percentile REPLACEMENT (`screener.py:438-440` overwrites composite entirely). A soft tilt preserves most of the working momentum signal.
3. **Turnover is a real cost the existing code does not measure.** Sector-neutral regrouping changes which names make top_n; the A3 replay MUST report turnover (it does -- design step 3) so the gate can reject a breadth gain that is eaten by churn.
4. **Don't conflate sector-NEUTRAL with sector-MOMENTUM (multidim's sector leg / phase-28.12).** They pull OPPOSITE directions (equal-weight across sectors vs overweight hot sectors). Shipping both naively could cancel or double-count. Pick sector-NEUTRAL for breadth (this step); keep multidim/sector-momentum as separate, independently-gated experiments.
5. **Short backtest windows lie.** The arXiv 2.51 Sharpe is a 6-month 2019 result. The A3 replay must span multiple regimes (2023-2025 minimum) and report the dispersion across dates, not a single aggregate, to avoid the IS-peak trap the rotation brief documented (Bailey/Borwein PBO).

---

## SYNTHESIS -- application to pyfinagent (the actionable answer)

### S1 -- Is sector-div the right lever? YES as a NEXT money lever, with a precise caveat.
Sector diversification IS the right near-term lever to amplify the working momentum
engine's breadth, for three reasons the evidence supports: (a) the sector dimension
is first-order to momentum (Moskowitz-Grinblatt), (b) the 2026 concentration unwind
raises the expected value of breadth right now (GS/Morningstar), and (c) it operates
INSIDE the live `rank_candidates` path so it affects live orders with zero
architectural bridge (unlike rotation -- the rotation verdict's whole point). **The
caveat that must be carried into the contract:** for a LONG-ONLY book, the literature
base rate (Harvey et al.) says FULL sector-neutralization helps only ~22% of the time;
the safer first move is a PARTIAL/soft sector tilt, and the sign of the effect MUST be
measured on pyfinagent's own universe via the A3 replay before any live enable. This
is NOT a reason to skip the lever -- it is a reason to (i) prefer the soft version and
(ii) make the A3 backtest the gate.

### S2 -- sector_neutral over multidim (confirmed). See A4.
sector_neutral within-sector percentile directly fixes "non-tech out-competed";
multidim is a momentum-quality upgrade whose sector leg is a TILT (amplifies
concentration). Ship sector_neutral first.

### S3 -- The exact wiring (file:line). See A2.
ONE change: build a ticker->sector map for the full `universe` and pass
`sector_lookup=` into `screen_universe` at `autonomous_loop.py:369`. The scoring code
(`screener.py:415-445`) is already correct and gated by
`sector_neutral_momentum_enabled` (default OFF); it only ever lacked sectors at rank
time.

### S4 -- The cheapest valid measurement (file:line). See A3.
A NEW `scripts/ablation/sector_neutral_replay.py` that replays the PRODUCTION
`screen_universe`+`rank_candidates` over ~36 historical monthly dates (2023-2025),
flag OFF vs ON, scoring forward-return Sharpe + sector-spread + turnover. The ML
backtest engine and the feature-ablation harness CANNOT measure this (different code
path -- `candidate_selector._rank_candidates`, no sector_neutral param). $0, free
yfinance data, real production code, no live change.

### S5 -- Expected Sharpe/turnover tradeoff from the literature.
- **Sharpe:** ambiguous sign for long-only sector-NEUTRAL (Harvey: ~22% win rate for
  full neutralization long-only); where sector-AWARE allocation helped, single-digit
  to large but on thin/short windows; realistic durable expectation is a SMALL Sharpe
  change with a MEANINGFUL reduction in sector concentration (the breadth is the
  reliable win; the Sharpe is the thing to protect, not assume).
- **Turnover:** EXPECTED TO RISE (Harvey: "higher turnover and transaction costs ...
  significant practical drawback"). The replay must quantify it; the soft/partial
  version limits it.
- **Drawdown/crash:** breadth + (optionally) vol-scaling reduces the -87%/-88%
  momentum crash risk (CFA; Barroso-Santa-Clara) -- the strongest reliable benefit.

### S6 -- Reason sector-div might NOT be the best lever (the honest redirect-within-redirect).
The Harvey long-only finding is a genuine yellow flag: a LONG-ONLY momentum book is in
the ~78% majority where keeping across-sector exposure beats neutralizing. If the A3
replay shows ON cuts Sharpe without a commensurate concentration/crash benefit, the
correct call is to NOT enable it live (or to ship only the soft/partial tilt). Two
alternatives the evidence ranks comparably or higher for money-per-risk:
(a) **vol-scaling the momentum book** (CFA/Barroso-Santa-Clara: nearly DOUBLES Sharpe,
halves drawdown -- a LARGER and better-evidenced effect than sector-neutral, and also
a live-path change), and (b) the **multidim composite** (CFA: better Sharpe/drawdown
since 1927, longer evidence base than sector-neutral). Recommend sector_neutral as the
phase-51.2 lever BECAUSE it most directly answers diagnostic finding 4 and the 2026
breadth case -- but flag in the contract that vol-scaling is the higher-EV adjacent
lever if the sector-neutral replay disappoints.

---

## GATE ENVELOPE

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 9,
  "snippet_only_sources": 9,
  "urls_collected": 18,
  "recency_scan_performed": true,
  "internal_files_inspected": 7,
  "gate_passed": true
}
```

`gate_passed: true` -- 9 sources read in full (floor 5), recency scan performed
(2024-2026, reported), 3-variant query discipline visible, 18 unique URLs, internal
audit pinned to file:line across 7 files (screener.py, services/autonomous_loop.py,
config/settings.py, backtest/candidate_selector.py, backtest/backtest_engine.py,
api/paper_trading.py, scripts/ablation/run_ablation.py).

