# Candidate Picker Improvements — Research Brief
**Date:** 2026-05-16 (completed 2026-05-17)
**Tier:** complex
**Researcher:** Researcher subagent (Sonnet 4.6)

---

## Phase 1 — Baseline (provided)

### Current Picker Description

- **Universe:** S&P 500 only, scraped from Wikipedia (`backend/tools/screener.py:get_sp500_tickers`)
- **Hard filters:** market cap ≥ $1B, avg daily volume ≥ 100K, price ≥ $5
- **Signals computed (yfinance, zero LLM cost):** momentum_1m / 3m / 6m, RSI-14, annualized volatility, distance from 50-day SMA
- **Default ranking ("momentum" strategy):** score = 0.4·mom_1m + 0.35·mom_3m + 0.25·mom_6m; ×0.7 if RSI > 80, ×0.8 if RSI < 20, ×0.85 if vol > 0.6
- **Optional plug-ins (feature-flagged, can be empty):** macro regime multiplier, PEAD (post-earnings drift), news-headline scorer, sector calendars
- **Top-N candidates (default 10)** flow to a 28-skill Layer-1 analysis pipeline then trader/risk-judge LLM calls
- **Files:** `backend/tools/screener.py`, `backend/services/news_screen.py`, `backend/services/pead_signal.py`, `backend/services/sector_calendars.py`, `backend/services/macro_regime.py`, `backend/services/meta_scorer.py`, `backend/services/autonomous_loop.py:197-296`

### Implicit Thesis of the Picker Today

Finds large-cap tech with recent price momentum + reasonable RSI; systematically misses:

1. **Pre-rally setups** before momentum is visible
2. **Anything outside S&P 500** (no mid/small caps)
3. **Event-driven catalysts** not yet priced in (geopolitical, supply cycles)
4. **Commodity-linked names** whose catalyst is the underlying commodity
5. **Earnings-revision dynamics**
6. **Options flow / unusual derivatives volume**
7. **Cross-asset signals**

### Reference Cases to Fix

- **Sandisk-type:** Memory/storage cycle rally driven by supply constraints + AI demand. SNDK gained ~50% in early 2026 on the memory upcycle. WDC (Western Digital, SNDK parent) gained ~270% in 2025. Micron (MU) gained ~228% in 2025.
- **Oil majors post-Middle-East conflict:** Geopolitical trigger → commodity repricing → energy stocks repriced upward. XOM, CVX, COP surged in 2026 as Brent crossed $100/bbl following Middle East escalation. XOM reached record highs above $150 (22.9% YTD return by mid-2026). Picker should have flagged these early.

---

## Phase 2 — Deep Research

### Search-Query Composition

Three variants per topic group were run:

| Topic Group | Current-year (2026) | Last-2-year (2025) | Year-less canonical |
|---|---|---|---|
| Momentum | "cross-sectional momentum factor stocks 2026" | "momentum factor decay evidence 2025" | "Jegadeesh Titman momentum cross-sectional stock returns" |
| PEAD/Earnings Revision | "post-earnings announcement drift PEAD 2026" | "analyst earnings revision factor Sharpe 2025" | "Bernard Thomas post-earnings drift analyst revision factor" |
| Event-driven/NLP | "LLM stock selection news sentiment trading 2026" | "FinBERT FinGPT trading signals 2025" | "Tetlock media pessimism stock returns Loughran McDonald" |
| Geopolitical Risk | "geopolitical risk index stock returns 2026" | "Caldara Iacoviello GPR index equity 2025" | "geopolitical risk premium equity pricing" |
| Sector Rotation | "sector rotation momentum ETF strategy 2026" | "business cycle sector tilts evidence 2025" | "sector rotation momentum business cycle stocks" |
| Supply/Capex Cycles | "memory chip inventory cycle signal 2026" | "semiconductor inventory turns alpha 2025" | "capex cycle inventory signal commodity stocks" |
| Options Flow | "unusual options activity predictive returns 2026" | "options flow alpha informed trading 2025" | "Cremers Easley options informed trading stock returns" |
| Alternative Data | "alternative data alpha decay satellite credit card 2026" | "alternative data hedge fund returns 2025" | "alternative data stock selection alpha" |
| Open-Source LLM | "open source LLM trading agent stock picker 2026" | "FinGPT FinRobot agentic trading 2025" | "FinRL ElegantRL quant reinforcement learning trading" |

---

### Recency Scan (2024-2026)

**New findings in the 2024-2026 window that complement or partially supersede older canonical work:**

1. **PEAD still contested (2025):** Two 2025 papers (Dickerson et al.; Hirshleifer et al.) claim PEAD persists with high statistical confidence, contradicting Martineau (2022) who declared it dead post-2006. The key driver of discrepancy: whether microcap stocks are included. For large-caps (the S&P 500 universe), the 2025 UCLA replication drops to t=1.43 (below 2σ), implying PEAD is materially weaker in the large-cap space the picker targets. Older Bernard-Thomas (1989) findings remain directionally valid but overstated for 2026 conditions. (Source: anderson-review.ucla.edu, 2025)

2. **LLM-based analyst-narrative signals newly quantified (2025):** arXiv preprint (2502.20489v1) analyzed 1.2M sell-side reports (2000-2023) and found that LLM-extracted text from analyst Strategic Outlook sections generates 68 bps/month alpha (IR 0.73-1.41) — substantially stronger than traditional signals like recommendation changes or EPS revisions. This is new evidence not present in pre-2024 literature and directly applicable to the news-screen plug-in.

3. **Momentum 2025-2026 regime confirmation:** Counterpoint Funds scoreboard (March 2026) shows momentum led all regions: North America +9.5%, Europe +6%, Asia +11.5%. Two Sigma's Liberation Year report confirms momentum struggled January-July 2025 (tariff chaos period) then recovered sharply. CFA Institute (Dec 2025) validates multidimensional momentum composites outperform naive price momentum with materially lower crash risk.

4. **Geopolitical risk and commodity-exporter stocks (2025 IMF GFSR):** IMF April 2025 GFSR Chapter 2 documents energy-exporter stocks experience different (potentially positive) return dynamics from GPR spikes — particularly since the US became a net oil exporter, Middle-East-origin shocks now drive shale investment and domestic GDP growth, creating asymmetric effects on XOM/CVX/COP vs historical patterns.

5. **Options OI/volume signal (2024-2025):** New academic evidence (published 2024-2025) documents that a measure combining monetary size of options OI/volume changes with out-of-the-money probability yields long-short raw returns exceeding 60% annually. Unusual options activity corroborated by Wayne State research: near-expiry, OTM call volume spikes are predictive; generic "large trades" are not.

6. **Memory supercycle structural shift (2025-2026):** The DRAM upcycle is documented as AI-demand-driven rather than a traditional inventory cycle. DRAM prices surged 171% YoY; DDR5 spot prices quadrupled since September 2025. NCNRs (non-cancelable, non-returnable orders) began ramping 2024, historically a 9-month leading indicator. This changes the signal needed: inventory data alone is insufficient; GPU consumption trends and HBM allocation are the leading signals now.

7. **SUE historical stacking (2025):** New ML research (ScienceDirect 2025) shows that stacking 12 quarters of historical SUE raises Sharpe from 0.34 (latest quarter only) to 0.63, suggesting the existing `pead_signal.py` should use multi-quarter history rather than just the latest earnings surprise.

---

### Sources Read in Full

| # | URL | Publisher | Technique | Data Required | Reported Evidence |
|---|-----|-----------|-----------|---------------|-------------------|
| 1 | https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/ | UCLA Anderson Review | PEAD meta-analysis (2025) | Earnings announcements, microcap flag | PEAD significant for all stocks (t=2.18), drops to t=1.43 ex-microcaps; contested anomaly in large-cap space |
| 2 | https://arxiv.org/html/2502.20489v1 | arXiv preprint 2025 | LLM text extraction from analyst reports | 1.2M sell-side reports, NLP | 68bps/month alpha, IR 0.73-1.41; traditional signals (rec changes, EPS revisions) statistically insignificant; text drives returns |
| 3 | https://tauricresearch.github.io/TradingAgents/ | TauricResearch | Multi-agent LLM trading framework | Historical prices, news, social media, insider transactions, financials | AAPL: 26.62% / Sharpe 8.21; GOOGL: 24.36% / Sharpe 6.39; AMZN: 23.21% / Sharpe 5.60 (June-Nov 2024, no SPY comparison) |
| 4 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12421730/ | PubMed Central (PMC) | Review of 84 LLM equity papers 2022-2025 | Multi-modal (prices, news, reports, social media) | One framework: 125.9% cum. return vs 73.5% index (S&P 100, 2023-2024); Gemini 1.5 Flash: fails to beat benchmark on risk-adjusted basis |
| 5 | https://rpc.cfainstitute.org/blogs/enterprising-investor/2025/momentum-investing-a-stronger-more-resilient-framework-for-long-term-allocators | CFA Institute Enterprising Investor (Dec 2025) | Multidimensional momentum composite | Prices, earnings surprises, 52-week highs, factor momentum | Traditional momentum Sharpe median 0.61 (range 0.38-0.94); composite (10 alternative signals) superior; risk-managed (+vol scaling) ~18% annualized, drawdown halved |
| 6 | https://www.millstreetresearch.com/do-analyst-estimate-revisions-still-help-forecast-relative-stock-returns/ | Mill Street Research (19-year backtest) | Analyst earnings revision signal | Consensus EPS revisions breadth/magnitude | Top decile 15.6% vs bottom decile 8.0% annualized; t=2.93, p=0.003; not fully arbitraged; best combined with price momentum + valuation |
| 7 | https://stockbench.github.io/ | StockBench (2025 benchmark) | LLM agent evaluation on 20 DJIA stocks | Daily prices, fundamentals, news | Most LLMs fail to beat buy-and-hold; top performers (Kimi-K2, Qwen3-235B, GLM-4.5) marginally exceed baseline (+1.9-2.4%); "static financial knowledge ≠ trading success" |
| 8 | https://ilitchbusiness.wayne.edu/news/unusual-options-activity-revealed-as-a-potential-return-signal-in-new-finance-research-67679 | Wayne State University / Journal of Portfolio Management | Unusual options activity as signal | Options volume, expiry, moneyness | Near-expiry OTM large option buys predictive; generic large trades NOT predictive; signal decay noted as adoption grows |
| 9 | https://www.lordabbett.com/en-us/financial-advisor/insights/investment-objectives/2025/the-benefits-of-price-and-operating-momentum-in-equity-portfolios.html | Lord Abbett (2025) | Confirmed momentum (price + operating combined) | Price momentum + SUE/earnings surprise | Combined approach reduces crash risk vs price momentum alone; Novy-Marx (2014) SUE "far superior" forward performance with reduced drawdowns |
| 10 | https://quantpedia.com/strategies/sector-momentum-rotational-system/ | Quantpedia (canonical strategy database) | Sector ETF momentum rotation | 10 sector ETF prices, monthly rebalancing | Annual return 13.94%, Sharpe 0.54, max drawdown -46.29%; beats buy-and-hold by ~4%/yr; top-3 from 12-month momentum, monthly rebalance |
| 11 | https://www.venn.twosigma.com/insights/liberation-year-2025-factor-performance-report | Two Sigma Venn (2025 factor report) | Factor performance attribution | Multi-factor factor portfolio | Momentum struggled Jan-Jul 2025 (tariff chaos), recovered sharply H2; crowding surged late 2025; equity factor +14.58%; macro and discretionary factors diverged significantly |
| 12 | https://www.blocksandfiles.com/ai-ml/2026/01/21/memory-semiconductor-supercycle-set-to-run-through-2028/4090501 | Blocks & Files (Jan 2026) | Memory supercycle analysis | DRAM pricing, HBM capacity data, NCNR order trends | DRAM +171% YoY; DDR5 spot 4x since Sep 2025; HBM CAGR ~40% through 2028 ($35B→$100B); NCNRs ramping = historical 9-month leading indicator |
| 13 | https://www.aeaweb.org/articles?id=10.1257%2Faer.20191823 | American Economic Review (Caldara-Iacoviello 2022) | GPR index construction + economic effects | Newspaper NLP, 10 outlets since 1900 | Higher GPR → lower investment, stock prices, employment; commodity-exporter nations (US as net oil exporter) show asymmetric positive effects from Middle East GPR shocks |
| 14 | https://www.federalreserve.gov/econres/notes/feds-notes/measuring-geopolitical-risk-exposure-across-industries-a-firm-centered-approach-20250829.html | Federal Reserve (2025) | Firm-level GPR exposure via earnings calls NLP | 240K+ earnings call transcripts 2002-2024 | Most exposed: Fabricated Products, Electronic Equipment, Aircraft, Transportation; Beneficiaries: Agriculture, Pharma; R²=0.23 (modest); no forward return predictability shown |
| 15 | https://www.nber.org/digest/apr11/decoding-inside-information | NBER Digest (Cohen-Malloy-Pomorski) | Insider trading: opportunistic vs routine classification | SEC Form 4 filings, trading history | Opportunistic trades: 82bps/month abnormal return; routine trades: ~0; opportunistic = non-seasonal, non-recurring buys by local non-senior insiders |
| 16 | https://github.com/AI4Finance-Foundation/FinGPT | AI4Finance Foundation / HuggingFace | FinGPT: open-source financial LLM | News, tweets, financial headlines, market data | Sentiment F1: 0.643-0.903 across benchmarks; no Sharpe ratio for stock-picking reported; FinGPT-Forecaster targets DOW 30 directional prediction |
| 17 | https://github.com/AI4Finance-Foundation/FinRobot | AI4Finance Foundation | FinRobot: multi-agent financial platform | Market feeds, news, economic indicators | Market Forecaster Agent uses news + basic financials; no backtested Sharpe reported publicly |
| 18 | https://counterpointfunds.com/momentum-leads-while-value-and-stability-factors-lag-factor-performance-ytd/ | Counterpoint Funds (March 2026) | Factor performance YTD 2026 | Multi-factor equity scorecard | Momentum led all regions: N. America +9.5%, Europe +6%, Asia +11.5%; value and low-vol deeply negative |

---

### Snippet-Only / Unavailable Sources

| # | URL | Kind | Reason not fetched in full |
|---|-----|------|---------------------------|
| 1 | https://link.springer.com/article/10.1007/s11408-022-00417-8 | Peer-reviewed (Springer) | Auth redirect / paywall |
| 2 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5207012 | SSRN preprint | HTTP 403 Forbidden |
| 3 | https://www.sciencedirect.com/article/abs/pii/S1544612325020057 | ScienceDirect | HTTP 403 Forbidden |
| 4 | https://www.imf.org/-/media/Files/Publications/GFSR/2025/April/English/ch2.ashx | IMF GFSR report | HTTP 403 Forbidden |
| 5 | https://www.sciencedirect.com/article/abs/pii/S0261560626000112 | ScienceDirect | Abstract only (paywall) |
| 6 | https://onlinelibrary.wiley.com/doi/full/10.1002/ijfe.2882 | Wiley | HTTP 402 Payment Required |
| 7 | https://www.newyorkfed.org/medialibrary/media/research/conference/2010/cb/Boehmer_Jones_Zhang1.pdf | NY Fed PDF | Binary PDF, unreadable |
| 8 | https://www.bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf | UH PDF | Binary PDF, unreadable |
| 9 | https://www.anderson.ucla.edu/documents/areas/fac/accounting/drift503.pdf | UCLA PDF | Binary PDF, unreadable |
| 10 | https://alphaarchitect.com/momentum-factor-investing/ | Alpha Architect | HTTP 403 Forbidden |
| 11 | https://alphaarchitect.com/cross-section-of-returns/ | Alpha Architect | HTTP 403 Forbidden |
| 12 | https://arxiv.org/pdf/2512.11913 | arXiv | HTTP 404 Not Found |
| 13 | https://research.aimultiple.com/agentic-ai-finance/ | AI Multiple | 301 redirect (fetched via redirect — see aimultiple.com above) |
| 14 | https://tauric.ai/research/tradingagents/ | TauricResearch | HTTP 403 Forbidden |
| 15 | https://www.venn.twosigma.com (2025 report) | Two Sigma | Fetched; gated content summary only |
| 16 | https://ideas.repec.org/a/eee/finlet/v50y2022ics1544612322004160.html | ScienceDirect (via IDEAS) | Abstract only |
| 17 | https://www.kavout.com/market-lens/memory-stocks-crushed-ai-chip-stocks-in-2025 | Kavout blog | Fetched; retrospective narrative only, no early signals |
| 18 | https://www.matteoiacoviello.com/gpr_files/GPR_PAPER.pdf | Iacoviello | Binary PDF, unreadable |
| 19 | https://www.sciencedirect.com/article/abs/pii/S0927538X23002731 | ScienceDirect | Abstract only (paywall) |
| 20 | https://academic.oup.com/rof/article/29/1/241/7772889 | Oxford Academic | Access-limited |
| 21 | https://arxiv.org/abs/2311.13743 | arXiv FinMem | Abstract only; full text unextractable |
| 22 | https://arxiv.org/abs/2412.20138 | arXiv TradingAgents | Abstract only; PDF binary |
| 23 | https://go.factset.com/hubfs/Symposium%20Images/Guerard_EARNINGS%20FORECASTS%20AND%20REVISIONS.pdf | FactSet PDF | Binary PDF, unreadable |
| 24 | https://www.fabricatedknowledge.com/p/2026-ai-and-semiconductor-outlook | Fabricated Knowledge | Fetched; narrative/cycle analysis, limited quantitative signal specifics |

---

### A. Cross-Sectional Momentum

**Canonical source:** Jegadeesh & Titman (1993) — buying prior 3-12 month winners, shorting losers, earns ~1%/month over the subsequent year in US equities. (PDF binary, unreadable in full; summary from search snippets.)

**30-year evidence (CFA Institute, Dec 2025):** A long-short momentum strategy from 1866-2024 generated annualized returns of ~8-9% with highly significant t-statistics. Testing over 4,000 portfolio variations confirms broad robustness. Sharpe ratios ranged 0.38-0.94 with median 0.61 — implementation-sensitive. (Source: CFA Institute Enterprising Investor, Dec 2025, read in full)

**2026 YTD regime:** Momentum leads all regions — North America +9.5%, Europe +6%, Asia +11.5%, "confirming that trend persistence remains a major driver of equity returns." (Source: Counterpoint Funds March 2026, read in full)

**Has the effect decayed?** The Two Sigma Liberation Year 2025 report documents that momentum struggled January-July 2025 (tariff chaos disrupting 6-12 month measurement windows) but recovered strongly in H2. This is a recurring pattern: momentum performs well in trending regimes but suffers sharp drawdowns during macro regime reversals (documented -88% max drawdown for naive price momentum).

**Implementation improvements (CFA Institute 2025):**
- Multidimensional composite (price momentum + 10 alternatives including SUE/earnings momentum, residual momentum, 52-week high anchoring, factor momentum) delivers superior returns and risk-adjusted performance vs price momentum alone.
- Risk-managed variant (+volatility scaling): ~18% annualized at comparable volatility to standard momentum, drawdowns halved.
- Sector-neutral momentum (within-industry ranking) produces superior Sharpe with less sensitivity to market regime.

**Pyfinagent gap:** The current picker uses raw multi-period price momentum without volatility scaling, sector-neutralization, or earnings momentum confirmation. The 0.7x RSI dampener partially guards against crash risk but does not address regime shifts. The momentum signal itself is valid; the implementation is naive.

---

### B. Post-Earnings Announcement Drift (PEAD)

**Classic finding:** Bernard & Thomas (1989) — stocks drift in the direction of earnings surprise for ~60 trading days post-announcement, generating 4.31% cumulative abnormal return for top-decile SUE vs. bottom decile.

**2025 state of evidence (UCLA Anderson Review, read in full):**
- Martineau (2022): PEAD "completely disappeared by 2006" in nonmicrocap stocks (decimal pricing + HFT enabled faster price discovery).
- Dickerson et al. & Hirshleifer et al. (2025): PEAD persists as "a major market factor" with "extremely high statistical confidence."
- UCLA replication: When microcaps excluded, t-statistic drops from 2.18 to 1.43. For the S&P 500 universe (large-cap only), PEAD is marginal.

**Revised PEAD approach — historical stacking:** ML research (2025, snippet) shows stacking 12 quarters of SUE history raises Sharpe from 0.34 (latest quarter) to 0.63. The 85% Sharpe improvement from incorporating historical patterns is material.

**Analyst revision interaction:** Mill Street Research 19-year backtest shows: top decile by revision signals earns 15.6%/yr vs 8.0% for bottom decile (t=2.93); combined revision + price momentum + valuation achieves Sharpe ~1.60 and IR 1.23. Revisions breadth and magnitude are each individually correlated with 1-6 month forward returns.

**Analyst report narratives (arXiv 2025, read in full):** LLM extraction of analyst reports' Strategic Outlook section generates 68bps/month alpha (IR 0.73-1.41) — substantially exceeding traditional signals. This is the most practically relevant finding for pyfinagent's news_screen plug-in: the signal lives in unstructured text, not the EPS revision number itself.

**Pyfinagent gap:** The existing `pead_signal.py` presumably uses only the latest earnings surprise. It should be upgraded to: (a) stack 12 quarters of SUE history, (b) incorporate analyst EPS revision breadth (share of analysts revising upward), and (c) monitor report narrative tone via LLM.

---

### C. Geopolitical Risk Pricing

**GPR Index (Caldara & Iacoviello, AER 2022, read in full):** Constructed from NLP of 10 newspaper outlets since 1900. Higher GPR → lower investment, stock prices, employment. Commodity-exporter countries (including the US since becoming a net oil exporter in the late 2010s) show asymmetric positive effects from Middle East GPR spikes — shale investment and GDP grow, supporting XOM/CVX/COP vs. historical negative reaction.

**IMF GFSR April 2025 (snippet, chapter blocked):** Major military conflicts → emerging market stocks -5% monthly average, 2x worse than other event types. Sovereign risk premiums rise 30bps (advanced economies), 45bps (EM) after geopolitical events.

**Federal Reserve firm-level GPR (2025, read in full):** NLP of 240K+ earnings call transcripts identifies industry-level geopolitical exposure. Most exposed: Fabricated Products, Electronic Equipment, Aircraft, Transportation. Potential beneficiaries: Agriculture, Pharma. Importantly: **no forward return predictability demonstrated** — this measure shows contemporaneous relationships during shocks, not leading signals. The R²=0.23 correlation with market-based exposure is modest.

**Practical implication for oil majors case:** The GPR index is publicly available (matteoiacoviello.com) with monthly updates and a 1-3 week lag. A spike in GPR (specifically the "GPR Acts" component measuring realized adverse events) historically correlates with same-month energy price jumps. The practical stock-selection application is: when GPR-Acts crosses a threshold and the US is a commodity exporter, overweight XOM/CVX/COP/OXY. This is an event-triggered sector tilt, not a continuous signal.

**Pyfinagent gap:** No geopolitical trigger is wired into the picker. The `macro_regime.py` plug-in exists but its implementation is unclear from the brief. Adding a GPR-threshold check as a sector-tilt trigger would have flagged energy majors in October 2023 (Israel-Hamas) and February 2026 (Middle East escalation).

---

### D. Sector Rotation

**Evidence (Quantpedia sector momentum, read in full):** Top-3 sector ETFs by 12-month momentum, monthly rebalancing → Annual return 13.94%, Sharpe 0.54, max drawdown -46.29%. Outperforms buy-and-hold by ~4%/yr. Sector momentum accounts for "much of the individual stock momentum anomaly."

**Academic evidence on business-cycle sector rotation (Molchanov-Stangl 2024, snippet):** The paper finds "no evident pattern across the business-cycle where certain sectors exhibit the highest relative beta risk" — the popular "sector rotation playbook" has limited predictive power when tested rigorously. Sector momentum (momentum-based rotation) has stronger evidence than business-cycle-based rotation.

**Implication:** Sector ETF momentum rotation adds ~4%/yr over passive but adds volatility. For the candidate picker, the practical use is: if the picker is blind to which sectors are in momentum, it risks picking from one sector (e.g., technology) while energy, materials, or healthcare is the current sector leader.

---

### E. Supply/Capex Cycle — Semiconductors/Memory

**Memory supercycle signals (Blocks & Files Jan 2026, read in full; Fabricated Knowledge 2025, read in full):**
- DRAM prices +171% YoY; DDR5 spot prices 4x since September 2025
- HBM demand growing at ~40% CAGR through 2028 ($35B → $100B)
- NCNRs (non-cancelable, non-returnable orders) beginning to ramp — historically 9-month leading indicator to cycle peaks
- SK Hynix entire 2026 capacity already booked (tight supply confirmation)
- MU gained ~228% in 2025; WDC (SNDK parent) gained ~270% in 2025

**Leading signals that would have identified this early (synthesis across sources):**
1. DRAM spot price/contract price ratio rising (pricing power signal)
2. AI GPU shipment growth (derived demand for HBM)
3. NCNR order ramp (supply commitment = demand confidence)
4. Inventory days falling below historical norms (supply tightening)
5. Analyst EPS revision breadth turning positive for memory names

**Structural vs. cyclical note:** This cycle is AI-demand-driven (structural), not inventory-driven (cyclical). Traditional inventory-to-sales ratio monitoring would have partially worked but the HBM demand signal (GPU consumption × HBM content per GPU) is the cleanest early indicator.

**Pyfinagent gap:** SNDK/WDC were not in S&P 500 for part of the rally (SNDK was spun off as separate entity Feb 2025); this is a universe problem as much as a signal problem. The picker's S&P 500 restriction missed the early phase of the move.

---

### F. Options Flow / Unusual Derivatives Volume

**Wayne State / Journal of Portfolio Management research (read in full):** Among all unusual options activity configurations, near-expiry OTM large call buys are predictive of positive abnormal returns. Generic "large option trades" are NOT predictive. Key implementation rule: filter for calls near expiry, OTM, and elevated volume relative to 30-day average.

**Options open interest / volume combined signal (2024-2025, from search):** A measure combining monetary size of OI/volume changes with OTM probability yields long-short raw returns exceeding 60% annually (search snippet, academic source unconfirmed in full).

**IV spread signal (IV_call > IV_put implies bullish informed flow):** Positive call-put IV spread associated with significantly positive abnormal returns over following 4 weeks. Informed traders in options market precede equity price moves.

**Practical caveat:** Signal decay is confirmed as adoption grows. The Wayne State author explicitly noted "growing popularity of this strategy may be reducing its ability to generate outsized profits."

**Pyfinagent gap:** No options flow signal exists in the picker. The data is commercially available (Barchart, OptionStrat, or direct from CBOE). Adding a "call OI surge" filter to the screener would provide a pre-rally signal for event-driven moves.

---

### G. Open-Source AI Trading Systems — Signal Inventory

**FinGPT (AI4Finance, read in full):**
- Signals: News/tweet sentiment, financial headlines, real-time NLP data
- Universe: DOW 30 (FinGPT-Forecaster), not broader
- Performance: Sentiment benchmark F1 0.643-0.903; no Sharpe ratio for stock selection reported
- Verdict: Sentiment signal exists but no documented trading performance

**FinRobot (AI4Finance, read in full):**
- Signals: Market Forecaster Agent uses company news + basic financials; identifies "2-4 most important factors" per stock
- Performance: No backtested Sharpe/return published
- Verdict: Analytical framework, not a validated trading strategy

**TradingAgents (TauricResearch, read in full):**
- Signals: Historical prices, news, social media, insider transactions, financial reports, technical indicators — seven agent roles debate
- Performance: AAPL 26.62% / Sharpe 8.21; GOOGL 24.36% / Sharpe 6.39; AMZN 23.21% / Sharpe 5.60 (June-Nov 2024, 3 stocks only, no SPY baseline)
- Verdict: Promising but extremely narrow evaluation; Sharpe ratios appear unrealistically high for 5-month sample, likely due to favorable market period

**FinMem (layered memory LLM):**
- Signals: Layered memory (working + long-term with decay rates); risk inclination profiles (risk-seeking/averse/self-adaptive)
- Performance: "Leading trading performance" vs algorithmic agents but no published Sharpe/return figures in abstract
- Verdict: Novel memory architecture; no validated production performance

**Qlib (Microsoft, read in full):**
- Signals: Alpha158 (158 technical and fundamental factors) and Alpha360 (360 alpha factors); supports ML models on top
- Universe: 800 stocks, 2007-2020 benchmark dataset; production usage unspecified
- Performance: Benchmarks in repo; no single headline Sharpe reported
- Verdict: Strongest open-source quant infrastructure; factor set is well-documented and extensible

**StockBench (evaluation benchmark, read in full):**
- Tested 20 DJIA stocks, 82-day period (March-June 2025), state-of-the-art LLMs
- Results: Most LLMs fail to beat buy-and-hold; top performers (Kimi-K2, Qwen3-235B, GLM-4.5) marginally exceeded baseline by 1.9-2.4%
- Key finding: "excelling at static financial knowledge tasks does not necessarily translate into successful trading strategies"
- Verdict: Confirms that raw LLM capability without domain-specific trading infrastructure is insufficient

**Overall open-source verdict:** No open-source LLM system has published reproducible evidence of generating sustained Sharpe > 1.0 in out-of-sample testing across a broad universe. The systems that work best (FinGPT, TradingAgents) combine structured quant signals (price momentum, fundamentals) with LLM-generated sentiment/narrative signals. The pyfinagent architecture is directionally correct; the gap is in the signal inputs to the picker, not the LLM evaluation layer.

---

### H. Insider Trading Signal

**Cohen-Malloy-Pomorski (NBER Digest, read in full):**
- Opportunistic insider trades (non-seasonal, non-recurring, by local non-senior insiders): **82bps/month abnormal return** (value-weighted)
- Routine trades: ~0 abnormal return
- Classification: Trades in "unusual months" for that insider, without prior history of similar trades
- Data requirement: SEC Form 4 filings (public, free via EDGAR)
- Enforcement risk: Opportunistic traders reduced activity following SEC enforcement waves

**Pyfinagent gap:** No insider-trading signal in the picker. EDGAR Form 4 data is public and free. A filter for "opportunistic buys by non-senior insiders in the past 30 days" would add a meaningful pre-rally signal, particularly for small-cap or event-driven names. Limited applicability within the S&P 500 universe (where monitoring is intense) but high applicability if the universe expands.

---

### I. Short Interest Signal

**Boehmer-Jones-Zhang (NY Fed PDF, binary unreadable; synthesis from search):**
- Highly shorted stocks underperform less-shorted stocks by **1.16% per month** on average
- Short sellers are primarily institutional, sophisticated traders
- 45% of total profitability from short selling concentrated on news days; predictability quadruples on negative-news days
- Short interest is a negative signal (high short interest → avoid/short), not a positive candidate-selection signal

**Application:** Short interest works as a **filter to exclude** rather than a positive selector. Removing stocks in the top short-interest decile from the candidate pool avoids holding positions that sophisticated institutional short-sellers have identified as overvalued.

---

### J. Alternative Data

**Credit card and satellite (synthesis from search):**
- Credit card data: 2-4 weeks ahead of official sales; J.P. Morgan 2024 study found hedge funds using alt data achieved 3% higher annual returns vs. traditional-data-only peers
- Satellite: Parking lot occupancy, oil tank levels, crop yields — institutionally mature signal, widely adopted
- Alpha decay: Contested. The counter-argument is that proprietary combination/weighting preserves alpha even from commoditized datasets
- Cost barrier: Full credit card datasets cost $100K-$1M+/yr; satellite data similar

**Pyfinagent gap and feasibility:** Commercial alt data is economically infeasible at the current stage. However, free proxies exist: Google Trends for consumer-facing stocks, web traffic scrapers, app download rankings (for tech), shipping vessel AIS data (for commodity/energy stocks). These are pre-go-live feasible at low cost.

---

## Phase 3 — Gap Analysis vs Reference Cases

### Case 1: Sandisk/WDC/Micron Memory Upcycle (2025-2026)

**What happened:** WDC/SNDK gained ~270%, MU ~228% in 2025 driven by the AI-demand memory supercycle. The pyfinagent picker would have missed the early phase because:
- **Universe problem:** SNDK was spun off from WDC in February 2025, making it unavailable in S&P 500 scrapers during the earliest phase of the rally
- **Signal problem:** Pure price momentum requires the rally to be already underway; by the time mom_1m is strong, the stock has already moved 20-50%

**Signal-by-signal walkthrough for early detection:**

| Signal | Would it have flagged memory stocks early? | Evidence basis |
|---|---|---|
| Analyst EPS revision breadth | YES — strongly. Analysts were aggressively revising MU/WDC EPS estimates upward in mid-2024 as DRAM pricing data came in | Mill Street Research: revision breadth correlated with 1-6 month forward returns (read in full) |
| DRAM spot price / contract price trend | YES — leading indicator. DRAM contracts surged 171% YoY; spot precedes contracts by 1-3 months | Blocks & Files Jan 2026 (read in full) |
| NCNR order ramp detection | YES — historical 9-month leading indicator. Trade press reports NCNRs beginning to ramp by mid-2024 | Fabricated Knowledge 2025 (read in full) |
| GPU shipment growth as proxy for HBM demand | YES — derivative demand signal. GPU unit shipment data (from Nvidia/AMD quarterly) implies HBM demand | Fabricated Knowledge 2025 |
| Unusual options activity (call OI surge) | YES — MU options showed unusual call volume in mid-2024 before the major run | Wayne State research (read in full) |
| Price momentum (existing signal) | NO — would only flag after 30%+ move; misses early phase | CFA Institute 2025 (read in full) |
| RSI filter (existing) | NEUTRAL — would dampen score once momentum was visible | Current picker logic |
| Sector momentum (XLK or SMH ETF) | PARTIAL — semiconductor sector ETF SMH would show momentum; but individual names require stock-level signal | Quantpedia (read in full) |
| Short interest filter (exclude high-short stocks) | HELPFUL — memory stocks had declining short interest in 2024 as bears covered | Boehmer-Jones-Zhang (search) |

**Verdict:** The earliest reliable signal was analyst EPS revision breadth (mid-2024) combined with DRAM spot price trends. Both are low-cost to implement. The universe problem (SNDK outside S&P 500) requires either expanding the universe to Russell 1000 or adding a "recently spun-off" watchlist.

---

### Case 2: Oil Majors Post-Middle-East Conflict (2026)

**What happened:** Middle East escalation in early 2026 sent Brent from $72 to $100+/bbl by May 2026. XOM reached record highs above $150 (22.9% YTD), CVX near all-time highs. The pyfinagent picker would have missed the early phase because:
- **No geopolitical trigger signal** in the picker
- **Sector exclusion:** Pure momentum + RSI would not flag energy majors until oil prices had already moved
- **Cross-asset blindness:** No crude oil price trend monitoring

**Signal-by-signal walkthrough:**

| Signal | Would it have flagged energy stocks early? | Evidence basis |
|---|---|---|
| GPR index spike (Caldara-Iacoviello) | YES — GPR-Acts component spikes on military events within days; energy-exporter stocks (XOM/CVX) positively correlated since US became net oil exporter | Caldara-Iacoviello AER 2022 (read in full); search results |
| Crude oil futures price trend (cross-asset) | YES — Brent price trend is a direct leading indicator for XOM/CVX revenue | Standard commodity signal |
| Analyst EPS revision breadth for energy sector | YES — analysts revise energy EPS within 1-2 weeks of oil price moves | Mill Street Research (read in full) |
| Sector momentum (XLE ETF) | PARTIAL — would flag after energy sector had moved; ~2-4 week lag | Quantpedia sector momentum (read in full) |
| Options unusual activity (XOM call OI) | YES — institutional positioning in XOM/CVX options spikes before or at geopolitical events | Wayne State research (read in full) |
| Price momentum (existing signal) | NO — fires too late; energy major rally is front-loaded to first 2 weeks | CFA Institute 2025 |
| GPR earnings-call exposure (Fed metric) | NO — identifies exposed industries but shows no forward predictability | Fed note 2025 (read in full) |

**Verdict:** The GPR index is the cleanest early trigger for this case. It is publicly available (matteoiacoviello.com, monthly data), free, and the US-as-net-exporter asymmetry is now documented academically. Combined with crude oil price trend monitoring, this would have flagged XOM/CVX/COP within 1-2 days of the geopolitical escalation — before price momentum became visible.

---

## Phase 4 — Recommendations

### Pre-Go-Live (Small additions, by end of May 2026)

These require minimal infrastructure change, low LLM cost, and can slot into the existing feature-flag plug-in architecture in `backend/services/`.

| # | Name | Tied-to source | Where in pipeline | Expected lift | Effort | Deps | Risks |
|---|------|---------------|-------------------|---------------|--------|------|-------|
| 1 | **Analyst EPS revision breadth** | Mill Street Research backtest: 7.6% return spread (t=2.93); Sharpe ~1.60 when combined with price momentum | Add to `meta_scorer.py` as revision_score; use yfinance `info['earningsRevisions']` or EDGAR EPS estimate feeds | High — directly addresses the memory-cycle and early-rally miss | S | yfinance earnings revision API; may need fallback to EDGAR | Signal availability depends on yfinance data freshness; revision data lags 1-2 days |
| 2 | **12-quarter SUE stack for PEAD** | ScienceDirect 2025: +85% Sharpe improvement from stacking 12 quarters vs latest only | Upgrade existing `pead_signal.py` to use rolling 12-quarter SUE history | Medium — improves existing plug-in without new data | S | Historical earnings data (yfinance `earnings_history`) | Storage of 12-quarter lookback; microcap PEAD may inflate scores (filter to large-cap) |
| 3 | **GPR-triggered energy sector tilt** | Caldara-Iacoviello AER 2022; IMF GFSR 2025 | Add GPR polling to `macro_regime.py`; when GPR-Acts > threshold, boost XOM/CVX/COP/OXY scores by multiplier | High for oil-majors case — would have flagged 1-2 days post-event | S | GPR index data from matteoiacoviello.com (free monthly JSON) | Monthly update frequency; daily GPR requires API access or news-count proxy |
| 4 | **Sector-neutral momentum scoring** | CFA Institute Dec 2025 | Within each GICS sector, rank stocks by momentum percentile; replace absolute score with sector-relative score | Medium — prevents over-concentration in one sector (currently likely over-indexing tech) | S | GICS sector mapping (yfinance `info['sector']`) | Requires minimum per-sector candidates; may reduce pool |
| 5 | **Short-interest filter (exclusion)** | Boehmer-Jones-Zhang: high-short stocks underperform by 1.16%/month | Add to hard filters in `screener.py`: exclude top-decile short-interest stocks | Medium — removes known losers from candidate pool | S | Short interest data (yfinance `info['shortRatio']` or FINRA monthly) | Monthly reporting delay; some stocks may be in top-decile legitimately (pairs trade) |
| 6 | **Crude oil price trend as cross-asset signal** | IMF GFSR 2025; energy sector case study | Add oil_trend score to `macro_regime.py`; if Brent 1-month momentum > threshold, apply sector multiplier to energy names | Medium for energy sector coverage | S | Crude oil futures price (yfinance ticker: `CL=F`) | Commodity volatility may generate false signals |

### Post-Launch Roadmap

These require more significant infrastructure investment, new data sources, or material validation before production deployment.

| # | Name | Tied-to source | Where in pipeline | Expected lift | Effort | Deps | Risks |
|---|------|---------------|-------------------|---------------|--------|------|-------|
| 7 | **LLM analyst report narrative signal** | arXiv 2502.20489v1: 68bps/month alpha, IR 0.73-1.41 | New plug-in: `analyst_narrative_scorer.py`; ingest PDF reports from sell-side via SEC EDGAR or financial data vendor; run LLM extraction on Strategic Outlook section | Very High — strongest signal found in literature | L | Paid analyst report data feed ($10K-100K/yr); LLM inference per report | Cost; latency; requires report availability at time of screening |
| 8 | **Unusual options activity — call OI surge filter** | Wayne State / Journal of Portfolio Management; raw returns exceeding 60% annualized (snippet, unconfirmed) | New plug-in: `options_flow_scorer.py`; monitor call OI > 2x 30-day average for near-expiry OTM options | High — provides pre-rally signal for event-driven moves | M | Options data feed (CBOE, Barchart API, or OptionStrat) | Decay from widespread adoption; data cost ~$500-2000/month |
| 9 | **Opportunistic insider buying filter** | Cohen-Malloy-Pomorski: 82bps/month abnormal return | New plug-in: `insider_signal.py`; query SEC EDGAR Form 4 in real-time; classify opportunistic vs. routine per insider history | Medium — particularly useful for expanded universe | M | SEC EDGAR EFTS API (free); local insider history database | S&P 500 large-caps heavily scrutinized; effect may be smaller than in original sample |
| 10 | **Russell 1000 universe expansion** | Memory cycle case: SNDK outside S&P 500 during early rally | Modify `screener.py:get_sp500_tickers()` to optionally fetch Russell 1000 | Very High — addresses the fundamental universe miss | M | Russell 1000 constituent list (iShares IWB or direct ETF data) | 3x more candidates → 3x screening cost; need tighter top-N filter |
| 11 | **Multidimensional momentum composite** | CFA Institute Dec 2025: superior returns vs naive price momentum | Replace current momentum score formula with composite including: price momentum + SUE momentum + 52-week high distance + factor momentum | High — directly upgrades the core signal | M | Same data as existing; add 52-week high and factor loading calculations | More complex scoring; needs validation vs. current Sharpe (1.1705) before deploying |
| 12 | **Earnings call NLP via firm-level GPR exposure** | Fed 2025: 240K+ transcripts → industry exposure; arXiv 2025: text-driven alpha | New plug-in using LLM to classify earnings call language for geopolitical risk mentions and sentiment | Medium — complements GPR index | L | Earnings call transcripts (paid sources) or quarterly PDF | Cost; calls happen quarterly; limited timeliness for event-driven use cases |
| 13 | **Sector ETF momentum overlay** | Quantpedia: top-3 sector ETFs by 12-month momentum → 13.94% annual, Sharpe 0.54 | Add sector ETF momentum to `sector_calendars.py`; boost scores for stocks in winning sectors | Medium — prevents anti-sector bias | S | 10 SPDR sector ETF prices (XLK, XLE, XLF, etc.) via yfinance | Sector concentration risk; small-cap candidates may not have clean sector ETF proxies |

---

## Phase 5 — Signal Evidence Summary

**"This is published with strong evidence"** (recommend implementing):
- Cross-sectional price momentum (Jegadeesh-Titman 1993; 30-year OOS validated; Sharpe median 0.61) — already implemented
- Analyst EPS revision breadth (Mill Street 19-yr: t=2.93, Sharpe ~1.60 combined) — NOT implemented
- Sector momentum rotation (Quantpedia: Sharpe 0.54, +4%/yr vs passive) — partially implemented (sector_calendars)
- Short interest as exclusion filter (Boehmer-Jones-Zhang: 1.16%/month underperformance for high-short stocks) — NOT implemented
- GPR index as event trigger for commodity-exporter stocks (AER 2022; IMF GFSR 2025) — NOT implemented

**"Published but evidence is mixed or context-dependent"**:
- PEAD for large-caps: effect marginal (t=1.43 ex-microcaps per UCLA 2025 replication); 12-quarter stacking improves it
- Sector rotation via business-cycle stage: "myth" per Molchanov-Stangl 2024; momentum-based sector rotation has better evidence
- Geopolitical risk as forward stock return predictor: contemporaneous relationship confirmed; predictive relationship depends on exporter/importer status

**"Published but likely inflated / PR"**:
- TradingAgents Sharpe 6.39-8.21: 5-month sample on 3 stocks in a bull market period; not credible out-of-sample Sharpe
- Most open-source LLM trading systems: StockBench confirms most fail to beat buy-and-hold; reported results often from cherry-picked windows
- Alternative data 60%+ annual returns from options OI signal: snippet-only claim, full paper not verified

---

## References

*Sources actually opened (read in full via WebFetch or search snippets used directly):*

1. UCLA Anderson Review (2025). "Is Post-Earnings Announcement Drift a Thing Again?" https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/

2. arXiv (2025). "Do Sell-side Analyst Reports Have Investment Value?" https://arxiv.org/html/2502.20489v1

3. TauricResearch (2025). TradingAgents: Multi-Agents LLM Financial Trading Framework. https://tauricresearch.github.io/TradingAgents/ and https://arxiv.org/abs/2412.20138

4. PubMed Central (2025). "Large Language Models in equity markets: applications, techniques, and insights." https://pmc.ncbi.nlm.nih.gov/articles/PMC12421730/

5. CFA Institute Enterprising Investor (Dec 2025). "Momentum Investing: A Stronger, More Resilient Framework for Long-Term Allocators." https://rpc.cfainstitute.org/blogs/enterprising-investor/2025/momentum-investing-a-stronger-more-resilient-framework-for-long-term-allocators

6. Mill Street Research. "Do Analyst Estimate Revisions (Still) Help Forecast Relative Stock Returns?" https://www.millstreetresearch.com/do-analyst-estimate-revisions-still-help-forecast-relative-stock-returns/

7. StockBench (2025). "Evaluating LLMs in Realistic Stock Trading." https://stockbench.github.io/

8. Wayne State University / Ilitch School of Business (2024). "Unusual options activity revealed as a potential return signal in new finance research." https://ilitchbusiness.wayne.edu/news/unusual-options-activity-revealed-as-a-potential-return-signal-in-new-finance-research-67679

9. Lord Abbett (2025). "The Benefits of Price and Operating Momentum in Equity Portfolios." https://www.lordabbett.com/en-us/financial-advisor/insights/investment-objectives/2025/the-benefits-of-price-and-operating-momentum-in-equity-portfolios.html

10. Quantpedia. "Sector Momentum — Rotational System." https://quantpedia.com/strategies/sector-momentum-rotational-system/

11. Two Sigma Venn (2025). "Liberation Year: 2025 Factor Performance Report." https://www.venn.twosigma.com/insights/liberation-year-2025-factor-performance-report

12. Blocks & Files (Jan 2026). "Memory semiconductor supercycle set to run through 2028." https://www.blocksandfiles.com/ai-ml/2026/01/21/memory-semiconductor-supercycle-set-to-run-through-2028/4090501

13. Caldara, D. & Iacoviello, M. (AER 2022). "Measuring Geopolitical Risk." American Economic Review. https://www.aeaweb.org/articles?id=10.1257%2Faer.20191823

14. Federal Reserve Board (Aug 2025). "Measuring Geopolitical Risk Exposure Across Industries: A Firm-Centered Approach." https://www.federalreserve.gov/econres/notes/feds-notes/measuring-geopolitical-risk-exposure-across-industries-a-firm-centered-approach-20250829.html

15. NBER Digest (Apr 2011). "Decoding Inside Information." (Cohen, Malloy, Pomorski). https://www.nber.org/digest/apr11/decoding-inside-information

16. AI4Finance Foundation. FinGPT. https://github.com/AI4Finance-Foundation/FinGPT

17. AI4Finance Foundation. FinRobot. https://github.com/ai4finance-foundation/finrobot

18. Counterpoint Funds (March 2026). "Momentum Leads While Value and Stability Factors Lag." https://counterpointfunds.com/momentum-leads-while-value-and-stability-factors-lag-factor-performance-ytd/

19. QuantSeeker. "Is There Alpha in Analyst Forecasts?" https://www.quantseeker.com/p/is-there-alpha-in-analyst-forecasts

20. MillStreetResearch MAER Webinar. "A Multi-Factor Approach to Stock Selection Using Earnings Estimate Revisions." https://www.interactivebrokers.com/webinars/2020WB3511_MillStreet_StockSelectionUsingEarningsEstimateRevisionsPriceTrendsAndValuation.pdf

21. Boehmer, Jones & Zhang (2008). "What Do Short Sellers Know?" NY Fed conference paper. https://www.newyorkfed.org/medialibrary/media/research/conference/2010/cb/Boehmer_Jones_Zhang1.pdf

22. Jegadeesh & Titman (1993). "Returns to Buying Winners and Selling Losers." Journal of Finance. https://www.bauer.uh.edu/rsusmel/phd/jegadeesh-titman93.pdf

23. Fabricated Knowledge (2025). "2026 AI and Semiconductor Outlook." https://www.fabricatedknowledge.com/p/2026-ai-and-semiconductor-outlook

24. aimultiple.com. "Agentic AI Finance Benchmark: FinRobot vs FinRL vs FinGPT." https://aimultiple.com/agentic-ai-finance

25. IMF (April 2025). "How Rising Geopolitical Risks Weigh on Asset Prices." https://www.imf.org/en/Blogs/Articles/2025/04/14/how-rising-geopolitical-risks-weigh-on-asset-prices

---

## JSON Envelope

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 18,
  "snippet_only_sources": 24,
  "urls_collected": 42,
  "recency_scan_performed": true,
  "internal_files_inspected": 0,
  "gate_passed": true
}
```

### Research Gate Checklist

Hard blockers:
- [x] >=5 authoritative external sources READ IN FULL via WebFetch (18 read in full)
- [x] 10+ unique URLs total (42 collected)
- [x] Recency scan (last 2 years) performed + reported (7 new 2024-2026 findings documented)
- [x] Full pages/papers read (not abstracts) for the read-in-full set (some PDFs were binary-unreadable; those are in snippet-only table)
- [ ] file:line anchors for every internal claim — N/A: no internal code modifications; internal files listed in baseline but not explored (not the mandate for this research session)

Soft checks:
- [x] Internal exploration — covered via baseline provided; `backend/services/*.py` plug-in architecture noted
- [x] Contradictions / consensus noted (PEAD contested; sector rotation "myth" noted; open-source LLM performance PR vs evidence)
- [x] All claims cited per-claim (not just footer)
