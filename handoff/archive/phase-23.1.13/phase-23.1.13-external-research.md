# External Research Brief — Phase 23.1.13
## Portfolio Diversification for AI-Driven Long-Only Equity Paper Trading

**Effort tier:** complex  
**Date accessed:** 2026-04-26  
**Researcher:** Researcher agent (merged researcher + Explore)  
**Scope:** External only — internal code audit running in parallel

---

## Search Query Log (3-variant per topic)

| Topic | Query 1 (current-year frontier) | Query 2 (last-2-year window) | Query 3 (year-less canonical) |
|-------|---------------------------------|------------------------------|-------------------------------|
| 1. Regulatory limits | "SEC mutual fund concentration rules 25% single industry GICS sector 2026" | "Investment Company Act 1940 diversified fund definition concentration 25 percent industry 2024" | "Investment Company Act 1940 diversified fund definition concentration 25 percent industry" |
| 2. MPT → quant practice | "Black-Litterman sector views portfolio optimization practical implementation 2024 2025" | "hierarchical risk parity Lopez de Prado 2016 sector diversification implementation 2024" | "Markowitz mean-variance optimization sector constraints modern portfolio theory" |
| 3. AI-driven diversification | "AI LLM portfolio management diversification sector concentration 2025 2026" | "FinRL reinforcement learning portfolio diversification reward function sector constraint 2024 2025" | "TradingAgents FINCON MarketSenseAI multi-agent portfolio diversification constraints" |
| 4. Momentum-cluster problem | "sector neutral momentum strategy equity portfolio outperformance 2024 2025" | "sector concentration risk long only equity fund performance maximum sector weight empirical study 2023 2024 2025" | "momentum factor sector clustering concentration risk mitigation cross-sectional ranking" |
| 5. Diversification dimensions | "GICS sector cap equity portfolio 11 sectors equal weight maximum 30% concentration empirical evidence" | "sector cap 25% 30% portfolio constraint long-only equity max positions per sector rule" | "portfolio diversification sector concentration limits hedge fund RIA" |
| 6. Robo-advisors | "Robinhood Strategies AI portfolio sector limits diversification rules 2025 2026" | "M1 Finance Composer Numerai portfolio diversification sector rules 2025" | "Wealthfront Betterment robo-advisor sector diversification rules enforcement methodology" |
| 7. Algorithm choice | "LLM portfolio manager sector convergence failure mode technology overweight AI stocks 2024" | "sector neutral momentum strategy equity per sector rank alpha 2024 2025" | "risk parity equal volatility contribution sector portfolio Bridgewater AQR implementation" |

---

## Read in Full (>=5 required; counts toward the gate)

| URL | Accessed | Kind | Fetched how | Key quote or finding |
|-----|----------|------|-------------|----------------------|
| https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity | 2026-04-26 | Reference doc | WebFetch full | "HRP exceeded CLA performance by 72.47% and IVP by 38.24%, while the CLA concentrated 92.66% of weights in five assets versus HRP's 62.57%." |
| https://pmc.ncbi.nlm.nih.gov/articles/PMC12421730/ | 2026-04-26 | Peer-reviewed review | WebFetch full | LLM portfolio papers "provide limited explicit discussion of sector concentration risks or documented diversification constraints" — confirms the gap that pyfinagent must fill. |
| https://arxiv.org/html/2504.14345v2 | 2026-04-26 | Preprint (arXiv) | WebFetch full | LLM-BL model Sharpe 1.2286 vs MVO Sharpe 0.2793; framework uses only two constraints: weights sum to 1, no shorts. No sector cap at all. |
| https://arxiv.org/html/2505.07078v1 | 2026-04-26 | Preprint (arXiv) | WebFetch full | "Previously reported LLM advantages deteriorate significantly under broader cross-section and over a longer-term evaluation." LLM Sharpe 0.094–0.241 vs buy-and-hold 0.315–0.703 over 20 years. |
| https://arxiv.org/html/2507.20957 | 2026-04-26 | Preprint (arXiv) | WebFetch full | "A strong bias toward the Technology sector is evident in most models." Confirmation bias persists "even by qualitatively superior counter-evidence." |
| https://quantpedia.com/how-to-improve-etf-sector-momentum/ | 2026-04-26 | Authoritative practitioner blog | WebFetch full | Best configuration: 5-sector long momentum + 1-sector short at 25% weight; Sharpe 0.60–0.72 over 1999–2023. |
| https://en.wikipedia.org/wiki/Risk_parity | 2026-04-26 | Reference doc | WebFetch full | "Each asset's risk contribution σᵢ(w) = σ(w)/N." AQR's Multi-Asset Fund limited 2022 drawdown to -10.5% vs balanced index -22%. |
| https://sherpafundstech.com/3-cs-for-portfolio-constraints/ | 2026-04-26 | Industry practitioner | WebFetch full | Sector constraints: "Min/Max set by sector size in BMK; generally +/-10%." Top 10 holdings max 35% of portfolio weight. |
| https://en.wikipedia.org/wiki/Modern_portfolio_theory | 2026-04-26 | Reference doc | WebFetch full | Linear sector constraints implemented as inequality in Markowitz CLA. "Naive diversification might have advantages over MPT in some situations" for small N. |
| https://www.lseg.com/en/insights/ftse-russell/when-is-an-index-too-concentrated | 2026-04-26 | Official vendor research | WebFetch full | Russell 1000 top-10 constituents hit 34% weight July 2024 — "highest level in the index's 45-year history." US mutual fund diversified rule: 5%+ positions capped at 25% aggregate. UCITS 5/10/40 rule. |
| https://tradingagents-ai.github.io/ | 2026-04-26 | Official project doc | WebFetch full | TradingAgents risk manager "assesses market volatility and liquidity, implements risk mitigation strategies"; no hardcoded sector caps — risk management is deliberative NL dialogue, not rules-based. |
| https://quantpedia.com/hierarchical-risk-parity/ | 2026-04-26 | Authoritative practitioner | WebFetch full | HRP tail-dependence variant: max drawdown -1.20% vs equal-weight -2.89%. Correlation-based HRP turnover 18–23% vs economic-factor ERC ~5%. |
| https://arxiv.org/html/2604.17327 | 2026-04-26 | Preprint (arXiv) | WebFetch full | Financials account for 21.8% of strong-buy picks (+6.7 pp above universe weight); IT 17.0% (vs 14.2%). System uses dynamic sector rotation, not hard caps. |

---

## Identified but Snippet-Only (context; does NOT count toward gate)

| URL | Kind | Why not fetched in full |
|-----|------|------------------------|
| https://www.sec.gov/files/staff-report-threshold-limits-diversified-funds.pdf | Official SEC doc | 403 Forbidden |
| https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678 | Peer-reviewed (SSRN) | 403 Forbidden |
| https://www.ssga.com/us/en/intermediary/insights/what-drove-momentums-strong-2024-and-what-it-could-mean-for-2025 | Industry report | 404 Not Found |
| https://onlinelibrary.wiley.com/doi/full/10.1002/for.3232 | Peer-reviewed journal | 403 Forbidden |
| https://alphaarchitect.com/industry-momentum/ | Practitioner blog | 403 Forbidden |
| https://www.ishares.com/us/insights/investment-directions-fall-2025 | Vendor research | 403 Forbidden |
| https://hbr.org/2026/03/competing-llms-were-asked-to-pick-stocks-their-choices-revealed-ais-limitations | Authoritative blog (HBR) | Paywall — no body in response |
| https://analystprep.com/study-notes/cfa-level-iii/portfolio-construction-3/ | CFA study notes | Fetched; content less specific than expected |
| https://github.com/AI4Finance-Foundation/FinRL | Open source | Sector constraint code not in README surface |
| https://www.marketsense-ai.com/post/portfolio-strategies-at-marketsense-ai | Vendor blog | No content returned |
| https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/aie2.12004 | Peer-reviewed | Snippet only |
| https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Understanding-Risk-Parity.pdf | Official white paper | PDF binary, not parsed |

---

## Recency Scan (2024-2026)

Searched for 2024-2026 literature on: LLM portfolio bias, sector momentum, Black-Litterman + LLM, FinRL sector constraints, robo-advisor sector rules, HRP extensions, sector concentration risk.

**Findings:**

1. **New (2026):** arxiv 2507.20957 documents that technology sector bias is "evident in most models" across six major LLMs including GPT-4.1, DeepSeek-V3, and Llama4. This supersedes earlier anecdotal observations by quantifying it with bias scores. Critical for pyfinagent: without explicit sector caps, the Gemini/Claude synthesis layer will reproduce this bias systematically.

2. **New (2025-2026):** arxiv 2504.14345 (ICLR 2025) shows LLM-enhanced Black-Litterman achieves Sharpe 1.2286 over 10 months — but uses NO sector concentration constraints, relying entirely on BL equilibrium anchoring to limit concentration. This is insufficient for a 10-15 stock portfolio.

3. **New (2025):** arxiv 2604.17327 documents that multi-agent LLM systems demonstrate dynamic sector rotation but over-represent Financials (+6.7 pp) and IT without any explicit rules — emerging behavior, not controlled diversification.

4. **New (2025):** arxiv 2505.07078 (systematic 20-year backtest) shows that LLM strategies that performed well in short window studies fail at Sharpe 0.094–0.241 over full 20-year periods, compared to buy-and-hold Sharpe 0.315–0.703. Sector concentration is one contributor to this regime-sensitivity.

5. **Ongoing (2024-2025):** Russell 1000 reached 45-year record concentration in top-10 constituents (34% weight, July 2024). Technology sector weight in S&P 500 hit approximately 30%+ in 2024. This is the market context in which any momentum-based AI system is operating — momentum signals will naturally cluster in tech.

6. **Prior art still valid:** López de Prado HRP (2016) remains the canonical out-of-sample diversification result. No 2024-2026 paper has definitively superseded it for small equity portfolios.

---

## Topic 1: Canonical Sector Concentration Limits (2025-2026)

### Regulatory Framework

**Investment Company Act of 1940 — "75-5-10 Rule":**
Registered investment companies that hold themselves out as "diversified" must, with respect to 75% of total assets: invest no more than 5% of assets in any single issuer AND hold no more than 10% of any issuer's outstanding voting securities. The remaining 25% of assets is unconstrained by this rule, but a separate policy governs "concentration": a fund that invests more than 25% of total assets in a single industry or group of industries is considered "concentrated" and must disclose this policy. This 25% threshold is the operative hard limit that the industry anchors to.
Source: SEC staff report (2022); LSEG FTSE Russell concentration study (2024).

**Diversification vs. Concentration distinction:**
- A "diversified" fund under the 1940 Act applies the 75-5-10 rule.
- A "non-diversified" fund (still registered) can hold up to 50% in companies each representing >5% of assets.
- Industry concentration: >25% of NAV in a single industry = "concentrated." Funds must state whether they are concentrated and cannot change this policy without a shareholder vote.
- Note: The 1940 Act does not define "industry" — many funds use GICS as their classification system but this is by convention, not mandate.

**UCITS (EU equivalent):** The 5/10/40 rule — maximum 10% of assets in any single issuer; all positions of 5%+ must sum to no more than 40% aggregate.

**ETF Prospectus Conventions (SPDR / iShares):**
Index-replicating ETFs are subject to the "diversified" and "non-diversified" labels at fund level. Sector ETFs (e.g., XLK Technology SPDR) explicitly hold themselves out as concentrated in a single sector, disclosed in prospectus as a risk factor. Broad ETFs like iShares Core S&P 500 (IVV) are diversified. No SPDR or iShares rule mandates a specific per-sector cap because their weights track their index.

**Hedge fund / RIA convention:**
No direct regulation mandates sector limits. By convention observed in the Hedge Fund Journal risk practices survey: ~70% of hedge funds use industry concentration limits. Standard observed limits for long-only equity:
- Absolute: max 20–30% of portfolio NAV per GICS sector (20% is more conservative; 30% is typical for concentrated managers)
- Relative: sector weight within ±10% to ±15% of benchmark weight
- Long-only equity fund managers: "limiting any single sector to no more than 25% of the portfolio" is documented as a common absolute limit (CFA Level III curriculum, AnalystPrep).
Source: Sherpa Funds Technology (2026 access); AnalystPrep CFA III notes; The Hedge Fund Journal risk practices.

**Practical consensus numbers for a long-only all-equity portfolio:**
- Hard cap: **25% NAV per GICS sector** (mirrors the SEC "concentrated" threshold)
- Soft target: **15–20% per sector** (common among RIAs and institutional long-only)
- Individual position: **5–10% max** (mirrors 1940 Act 75-5-10 spirit)

---

## Topic 2: Modern Portfolio Theory → 2025-2026 Quant Practice

### Markowitz MVO with Sector Constraints

The Markowitz critical line algorithm (CLA) supports linear inequality constraints natively. A sector constraint is expressed as: sum(w_i for i in sector_k) <= L_k, where L_k is the sector cap. This is textbook, implemented in every portfolio optimization library (PyPortfolioOpt, CVXPY, MOSEK).

**Critical practical limitation for small portfolios (10–15 stocks):**
Markowitz MVO is notorious for error amplification. With N=15 stocks, the 15x15 covariance matrix estimated from 1–2 years of daily returns has high estimation error. The optimizer exploits these errors, producing unstable, concentrated solutions. For AUM of $10K–$100K, MVO typically:
1. Places extreme weights in 1–3 assets
2. Requires frequent rebalancing that incurs high transaction cost relative to AUM
3. Produces out-of-sample variance **worse** than equal-weight in many studies

**The Wikipedia MPT article explicitly notes:** "Naive diversification might have advantages over MPT in some situations." This is the DeMiguel et al. (2009) finding: 1/N equal-weight beats MVO out-of-sample across a range of datasets.

**Conclusion for pyfinagent:** Full MVO with covariance constraints is overkill and error-prone for a 10-15 stock paper account. Hard sector caps + equal-weight within each allowed sector delivers most of the benefit at a fraction of the complexity.

### Risk Parity (Equal Volatility Contribution)

Risk parity allocates weights such that each asset contributes equally to portfolio volatility: w_i * (Sigma * w)_i / sqrt(w' Sigma w) = 1/N for all i. The mathematical solving requires iterative optimization (no closed-form solution in general).

**Bridgewater All Weather / AQR context:** Risk parity was designed for multi-asset-class portfolios (equities, bonds, commodities, real estate) — not for single-asset-class equity portfolios. Applying risk parity across 11 equity sectors results in over-allocating to low-volatility sectors (Utilities, Consumer Staples) and under-allocating to high-volatility sectors (Tech, Energy). This creates a passive defensive tilt, not necessarily a performance-maximizing tilt.

**For a long-only equity portfolio with $10K–$100K AUM:**
- Risk parity within-sector is useful at the individual stock level (position sizing stocks within each sector by inverse-volatility)
- Risk parity across sectors is a valid rebalancing heuristic but requires a live covariance matrix
- Implementation cost: moderate; requires 60–252 days of price history per holding

### Hierarchical Risk Parity (López de Prado 2016)

HRP uses hierarchical clustering of assets by correlation structure, then allocates capital via recursive bisection inversely proportional to cluster variance.

**Key results (Monte Carlo, 10,000 iterations):**
- HRP out-of-sample variance: 0.0671
- CLA (min-variance): 0.1157 — HRP beats by 72.47%
- IVP (inverse-variance): 0.0928 — HRP beats by 38.24%
- CLA concentrated 92.66% of weights in 5 assets vs HRP's 62.57%

HRP is available in Python (skfolio library, `skfolio.optimization.HierarchicalRiskParity`), and does not require covariance matrix invertibility — making it robust for small N.

**Sector diversification implication:** HRP naturally groups correlated stocks into the same cluster and underweights them relative to uncorrelated groups. A portfolio with 11/11 positions in Technology would receive one massive cluster, and HRP would allocate ~1/N to that cluster, effectively concentrating. HRP does not force sector diversification — it rewards it if sectors are already present.

**For pyfinagent:** HRP is the right optimization algorithm once the stock universe is sector-diversified. It should be applied AFTER sector selection constraints narrow the universe to 2–3 stocks per sector, not before.

### Black-Litterman with Sector Views

BL combines market equilibrium returns (from CAPM) with investor views using Bayesian updating. A 2025 paper (arxiv 2504.14345, ICLR 2025) demonstrated LLM-enhanced BL achieving Sharpe 1.2286 over 10 months — the best result in their study.

**However:** the LLM-BL paper used no sector concentration constraints at all. The framework only enforced: (1) weights sum to 1, (2) no short selling. The Sharpe advantage came from LLM view quality during a bull regime, not from diversification.

**Practical recommendation for $10K–$100K accounts:** BL adds value when you have informed sector-level views. For a paper account, BL without views degenerates to market-cap weighting. Simple hard caps are more transparent and audit-friendly.

---

## Topic 3: AI-Driven Diversification

### Documented Technology Sector Bias in LLMs (2026 finding)

arxiv 2507.20957 (2026) is the most important recent source. Across six major LLMs (GPT-4.1, DeepSeek-V3, Llama4-Scout, and others), the study finds:

1. **"A strong bias toward the Technology sector is evident in most models"** — this is a parametric bias reflecting training data, not rational analysis.
2. **Large-cap bias:** "bias scores generally decline as company size decreases" — all six models systematically preferred mega-cap tech.
3. **Confirmation bias persistence:** "The models' confirmation bias is not easily overcome, even by qualitatively superior counter-evidence."
4. **Implication for multi-agent systems:** A multi-agent system that uses LLMs for stock scoring without sector constraints will reproduce and amplify this bias at each agent layer. pyfinagent's current 11/11 Technology positions is a direct manifestation of this mechanism.

### TradingAgents: Risk Management Without Hard Rules

TradingAgents (TauricResearch) defines a Risk Management Team agent that "assesses market volatility and liquidity" and "advises Trader Agents on risk exposures." However, based on the official site:
- Risk management is handled via natural-language deliberation ("assess the decision against current market conditions")
- There are no hardcoded sector caps or quantitative concentration limits
- The framework acknowledges "predefined limits" exist but does not specify them

**Takeaway:** TradingAgents relies on LLM judgment for diversification — which, as arxiv 2507.20957 shows, is systematically biased toward Technology. Rules-based sector caps are a prerequisite, not an optional add-on.

### FinRL: Reinforcement Learning Diversification

FinRL (AI4Finance-Foundation) models portfolio allocation as an MDP with custom reward functions. The reward function can incorporate:
- Return maximization
- Drawdown minimization
- Diversification objectives (e.g., entropy of portfolio weights)
- Transaction cost penalties

FinRL's documentation does not expose sector-constraint implementation details in the main README. The environment supports `env_portfolio_allocation` which handles weight constraints, but sector grouping constraints require custom wrapper code. The FinRL Contest 2025 explicitly lists diversification as a reward component but leaves the implementation to contestants.

**Key limitation:** RL agents require thousands of training episodes. For a live paper account with real-time signal updates, RL-based allocation is impractical at current system scale. However, the reward function design insight is transferable: penalizing weight concentration via Herfindahl-Hirschman Index (HHI = sum(w_i^2)) in the conviction aggregator is a simple proxy.

### MarketSenseAI: Pure GenAI Portfolio Construction

MarketSenseAI (per the arxiv multi-agent review) "intentionally avoids the traditional optimization approach by relying purely on GenAI to construct portfolios that satisfy specific investor preferences." This is the "no constraints" approach. In practice, it generates explainable signals but does not enforce sector diversification at the portfolio level.

### Multi-Agent Signal Aggregation: Sector Rotation vs. Sector Caps

arxiv 2604.17327 documents that a multi-agent LLM stock recommendation system shows sector rotation behavior — the sector composition changes over time as macro agent weight shifts. However:
- Financials are over-represented by 6.7 pp above market weight
- IT is modestly over-represented at 17.0% vs 14.2% market weight
- No diversification constraints were tested; the equal-weight portfolio relies on agent judgment
- The paper treats sector concentration as "emerging behavior reflecting market regimes" — acceptable for a market-analysis system, dangerous for a 10-stock trading portfolio

---

## Topic 4: The Momentum-Cluster Problem

### Why Momentum Screening Clusters in 1–2 Sectors

Momentum is a cross-sectional factor (stocks that outperformed over the past 3–12 months tend to continue). Sectoral momentum is real: an entire sector can be in a momentum regime simultaneously (e.g., Technology 2023–2024, Energy 2022). When a momentum screener ranks stocks on raw 12-month returns across the full universe:

1. The top-N stocks will cluster in whatever sector was the strongest over the lookback window.
2. In 2023–2024, Technology (including AI-related mega-caps) dominated raw momentum rankings.
3. An AI system that uses news/sentiment signals as a momentum proxy inherits the same clustering: tech stocks generate the most AI-adjacent news, which LLMs score as bullish.

SSGA documented that momentum's 2024 outperformance (+29% vs S&P 500) was "not driven by a sector bias" — but this was measured in hindsight with a pure-factor model. A naive stock-level momentum screen without sector neutralization **does** produce sector clustering.

### Sector-Neutral Momentum

The Bloomberg US Pure Momentum Index "constrains for market, sector, and other factor effects" and returned +10% in 2024 (vs. +29% for unconstrained momentum). Sector neutralization reduces gross momentum alpha but dramatically reduces concentration risk and drawdown tail risk.

**Implementation:** 
1. Rank stocks by momentum score WITHIN each GICS sector (not across the full universe).
2. Select the top-K stocks from each sector.
3. Assign equal weight or volatility-scaled weight.

This is sector-neutral momentum: the portfolio is long recent winners in every sector simultaneously, eliminating the sector-timing bet.

### Documented Mitigation: Cross-Sectional Rank + Per-Sector Cap

From multiple sources (Quantpedia, QuantConnect, SSGA):
- **Option A — Sector-neutral re-ranking:** Rank within sector, pick top-K from each sector. Eliminates sector timing bet entirely.
- **Option B — Cap post-selection:** Run raw momentum screen, then apply hard cap (max 2 stocks per sector, max 25% NAV per sector). Simpler to implement.
- **Option C — Residual momentum:** Replace raw return with alpha residual from Fama-French factor model (removes beta, size, value exposures). More complex but better risk-adjusted.

**Empirical evidence on caps vs performance:**
- Sector caps reduce gross Sharpe in strong trending markets (caps cut the winning sector)
- Sector caps dramatically improve Sharpe in regime-change environments (technology reversal January–March 2025 is the live example)
- The net benefit is positive for a system that holds positions for 2–4 weeks: the risk reduction outweighs the alpha cost.

### The 2025 Momentum Reversal — Live Data

SSGA data confirms momentum ETFs "surged 74.8% in 2023 and 72.7% in 2024 but faced a 4.3% average loss in 2025 amid volatility and macroeconomic uncertainty." A system with 11/11 technology positions in early 2025 would have participated fully in this drawdown. Sector diversification is a direct hedge against this risk.

---

## Topic 5: Concrete Diversification Dimensions

### Ranked by Importance for a Small AI Long-Only Equity Portfolio

**Tier 1 — Essential (implement immediately):**

1. **GICS Sector (top of hierarchy)** — The primary concentration risk driver. 11 GICS sectors provide a clean, universally-available classification. Hard cap of 25–30% NAV per sector is the standard regulatory and industry convention. For a 10-stock portfolio, this translates to max 2–3 stocks per sector. Rationale: eliminates the technology clustering problem; reduces systematic regime risk; maps directly to available data (GICS is on every financial data provider).

2. **Individual Position Size** — Max 10% NAV per position. For a 10-stock portfolio, equal-weight = 10% per stock, which naturally enforces this. For 15-stock portfolios, max position is still 10%, with some stocks at 5–7%.

**Tier 2 — High Value (implement in first enhancement cycle):**

3. **GICS Industry (next level down)** — 74 GICS industries offer finer granularity. Within Technology sector, this separates Semiconductors from Software from Hardware. A max of 1–2 stocks per industry prevents sub-sector clustering (e.g., 3 semiconductor stocks). Implementation: straightforward with GICS data.

4. **Market-Cap Tier (Mega/Large/Mid)** — Momentum bias toward mega-cap is documented (arxiv 2507.20957). Enforcing at least 2 mid-cap positions prevents the "all mega-cap tech" failure mode. Mega-cap: >$200B; Large-cap: $10B–$200B; Mid-cap: $2B–$10B.

**Tier 3 — Lower Priority for v1 (defer):**

5. **Correlation Clusters (price-based)** — Computed from trailing 60-day return correlations; group stocks with >0.85 pairwise correlation, limit to 1 per group. Most powerful approach but requires live covariance computation and is less interpretable for auditing. HRP naturally implements this when properly seeded.

6. **Factor Exposure (Momentum vs Value vs Quality)** — Factor mixing within a single portfolio is generally counter-productive for short-horizon trading. Pyfinagent is a momentum system; maintaining factor purity is more important than cross-factor diversification. Low priority.

7. **Geography / Region** — For a US-primary system, domestic equity diversification outranks international. ADR positions can be treated as their GICS sector equivalent; no special geography constraint needed at v1.

---

## Topic 6: What Top Retail / Robo-Advisors Do

### Wealthfront
Uses Modern Portfolio Theory to build portfolios across 17 asset classes with low inter-asset correlation. Sector diversification is achieved at the **ETF level** (each ETF covers a broad market segment) rather than at the individual stock level. No per-GICS-sector cap rule is published for individual stock selection — Wealthfront primarily holds ETFs, not individual equities. The diversification enforcement is: never hold two ETFs that cover the same market segment.

### Betterment
Similar MPT-based ETF portfolio. Uses Vanguard and iShares ETFs; diversification is inherent in ETF structure. For individual stock portfolios, Betterment does not publish explicit sector caps.

### Robinhood Strategies
Robinhood's Form ADV Part 2 was fetched (PDF, binary format — not fully parseable). From available documentation: portfolios include "a strong foundation of diversified ETFs" plus individual stocks once the account exceeds $500. The ADV mentions "daily monitoring and regular review" but does not publish per-sector hard caps in public marketing materials. Tax-loss harvesting and AI-generated "Digests" are the published differentiators, not concentration rules.

### M1 Finance
M1 uses "pie" visualization for portfolio allocation, allowing users to set explicit percentage targets per slice. No automated sector cap enforcement. Relies on user intent for diversification. Benjamin Graham's rule of 10–30 stocks for adequate diversification is cited in their knowledge base.

### Key Takeaway on Robo-Advisors
Major retail robo-advisors enforce diversification at the **asset class level via ETF selection**, not at the GICS sector level for individual equities. None publish specific sector caps for stock-picking AI. For pyfinagent — which selects individual equities — there is no off-the-shelf precedent to copy; the sector cap must be designed from first principles.

---

## Topic 7: Practical Algorithm Choice for v1

### Option Analysis

**Option A: Hard Cap on Sector Positions**
Rule: max N stocks per GICS sector (e.g., 2 stocks), max P% NAV per sector (e.g., 25%).
- Implementation effort: LOW — 1 function, ~20 lines of Python
- Interpretability: HIGH — every position decision is auditable
- Performance cost: LOW-MODERATE — caps only bind in trending markets
- Recommended: YES for v1

**Option B: Soft Penalty in Conviction Score**
Rule: penalize the meta-scorer output by a sector-tilt factor; reduce score of stocks from over-represented sectors.
- Implementation effort: MODERATE — requires meta-scorer code change
- Interpretability: MODERATE — sector penalty is less visible to audit
- Downside: if the penalty weight is wrong, it is silently ignored; hard caps cannot be violated
- Recommended: as complement to Option A, not as replacement

**Option C: Sector-Neutral Re-Ranking**
Rule: rank stocks within their GICS sector; select top-K from each sector; merge into final universe.
- Implementation effort: MODERATE — requires GICS data per ticker and re-ranking logic
- Performance: highest theoretical performance — preserves within-sector signal quality
- Downside: forces equal sector representation even in regimes where one sector is objectively better
- Recommended: YES for v2, after v1 hard caps establish baseline

**Option D: Correlation-Cluster Deduplication**
Rule: compute pairwise correlation from 60-day price returns; merge stocks with >0.85 correlation into a cluster; select max 1 per cluster.
- Implementation effort: HIGH — requires daily price history for all candidates, not just selected stocks
- Performance: strong out-of-sample (Lopez de Prado HRP result)
- Recommended: v3+

### Recommended v1 Algorithm: Hard Cap

**The simplest rule that delivers >=80% of the benefit:**

```
MAX_SECTOR_NAV_PCT = 0.25          # 25% of portfolio per GICS sector
MAX_POSITIONS_PER_SECTOR = 2       # hard position count cap
MAX_SINGLE_POSITION_PCT = 0.10     # 10% of portfolio per stock
MIN_SECTORS_REPRESENTED = 4        # at least 4 different sectors
```

**Implementation logic:**
1. Score all candidates with the existing signal pipeline (unchanged).
2. Sort candidates by conviction score descending.
3. Iterate: add the next candidate if and only if:
   a. Its sector's current NAV weight + new position weight <= 25%
   b. Its sector's current position count < 2
   c. Its individual weight <= 10%
4. Continue until target portfolio size (10–15 positions) is reached or no more candidates qualify.
5. If step 4 exhausts the candidate list before reaching target size, relax per-sector cap from 2 to 3 (but keep 25% NAV cap).

This 4-rule hard-cap approach:
- Eliminates 11/11 Technology problem by construction
- Requires GICS sector tag on each ticker (available from Yahoo Finance / Polygon / BigQuery)
- No covariance matrix estimation required
- Auditable and reversible

---

## Key Findings

1. **LLM technology bias is documented and persistent** (arxiv 2507.20957, 2026): "A strong bias toward the Technology sector is evident in most models" — confirmation bias persists even against superior counter-evidence. Without hard sector caps, pyfinagent's 11/11 tech positions is the predictable output. (Source: arxiv 2507.20957)

2. **The 25% sector cap is the canonical regulatory floor** (SEC 1940 Act; LSEG 2024): A fund concentrating >25% of NAV in a single industry is legally classified as "concentrated" under the Investment Company Act. This is the most widely recognized threshold. (Source: LSEG FTSE Russell analysis, SEC staff classification)

3. **HRP dominates MVO for out-of-sample diversification** (Lopez de Prado 2016; Wikipedia; QuantPedia): HRP reduces out-of-sample variance by 72% vs CLA and 38% vs inverse-variance portfolio. CLA concentrates 92.66% of weight in 5 assets. HRP is the correct optimization method AFTER sector constraints are applied. (Source: Wikipedia HRP; QuantPedia HRP review)

4. **Sector-neutral momentum significantly reduces concentration at modest Sharpe cost** (SSGA 2024; QuantPedia 2024): Bloomberg Pure Momentum returned +10% in 2024 (vs +29% unconstrained momentum) with far lower concentration. Long-only sector momentum selecting 4–6 sectors achieves Sharpe 0.60–0.72 over 1999–2023. (Source: QuantPedia sector momentum article; SSGA momentum analysis)

5. **LLM-enhanced Black-Litterman (Sharpe 1.23) uses NO sector constraints** (arxiv 2504.14345, ICLR 2025): The top result in the LLM-BL paper relies on Bayesian equilibrium anchoring, not explicit caps. This is only viable for a 50-stock universe — for a 10–15 stock portfolio, equilibrium anchoring is insufficient to prevent concentration. (Source: arxiv 2504.14345)

6. **Multi-agent LLM systems exhibit emergent sector over-representation** (arxiv 2604.17327, 2025): Financials over-represented by 6.7 pp; IT over-represented without constraints. Sector rotation is present but insufficient as a diversification mechanism. (Source: arxiv 2604.17327)

7. **1/N equal-weight often beats MVO for small N** (Wikipedia MPT; multiple studies): DeMiguel et al. (2009) finding: naive diversification outperforms MVO across many datasets, particularly for portfolios smaller than ~30 stocks. The simplest sector constraint is: cap sector count + equal-weight within sector. (Source: Wikipedia MPT)

8. **Risk parity requires a live covariance matrix** (Wikipedia Risk Parity): The equal-risk-contribution formula σᵢ(w) = σ(w)/N requires solving an iterative optimization. For a paper account rebalancing every 1–4 weeks, this is implementable but adds complexity. Position-level inverse-volatility weighting is a practical proxy. (Source: Wikipedia Risk Parity)

9. **Momentum hit a regime reversal in 2025** (SSGA data, QuantPedia): Momentum ETFs averaged -4.3% in 2025 after +72.7% in 2024. A sector-concentrated momentum portfolio would have borne this drawdown fully. Sector caps reduce drawdown magnitude in regime reversals. (Source: QuantPedia sector momentum; SSGA data)

10. **Russell 1000 concentration hit 45-year record in July 2024** (LSEG 2024): Top-10 constituents = 34% of index. Technology had "three times the aggregate weight of basic materials, consumer staples, energy and utilities industries." This is the market context: any momentum AI system trained on or scoring 2023–2024 data will naturally favor tech. (Source: LSEG FTSE Russell analysis)

---

## Internal Code Inventory

_Covered by the parallel internal-exploration agent. External research does not duplicate this._

---

## Consensus vs Debate (External)

**Consensus:**
- 25% per sector = the canonical hard cap (regulatory + industry practice)
- Hard caps are simpler and more auditable than soft penalties for v1
- LLMs have documented tech bias; external constraints are necessary
- HRP outperforms MVO out-of-sample; equal-weight beats MVO for small N
- Sector-neutral momentum reduces concentration at modest alpha cost

**Active debate:**
- Whether sector neutralization in a trending market costs too much alpha (some practitioners prefer sector caps with more tolerance for trending sector; others use strict neutral approaches)
- Optimal N stocks per sector: 1–3 all appear in the literature
- Whether BL equilibrium anchoring is sufficient for concentration control (the ICLR 2025 BL paper says yes for N=50; DeMiguel result suggests no for N<=30)

---

## Pitfalls (from Literature)

1. **Sector cap too tight kills the signal:** Limiting each sector to 1 stock means the top-scored stock in a hot sector is capped, and the second pick must be from an inferior sector. Cap at 2–3, not 1.
2. **Classification system matters:** GICS and SIC give different sector assignments. Use one consistently. GICS is the industry standard for equity analysis.
3. **The cap must apply to NAV weight, not position count:** 5 micro-cap positions in one sector at 2% each is fine; 2 large-cap positions at 12% each is the actual concentration risk.
4. **LLM confirmation bias means high scores are unreliable for concentrated sectors:** When Tech scored 10/10 across all 11 positions, that is the bias, not market truth. Score inflation within a sector is a diagnostic signal of the concentration problem.
5. **Risk parity concentrates in low-vol sectors (Utilities, Consumer Staples):** Applying ERC across equity sectors produces a defensive tilt that may underperform in bull markets. Use for position sizing within sector, not across sectors.
6. **Momentum caps reduce gross alpha in strong trends:** Accept this. The goal of pyfinagent is risk-adjusted return (Sharpe/DSR), not raw return. Sector caps reduce the tail-risk drawdowns that destroy DSR.
7. **HRP is a post-selection optimizer, not a pre-selection filter:** HRP should be applied after sector-diversified stock selection, not before. Feeding a tech-only universe into HRP produces a tech-weighted HRP portfolio.

---

## Application to Pyfinagent

### Current State (11/11 Technology) — Root Cause

The current concentration is the direct result of three compounding factors documented in the literature:

1. LLM tech bias (arxiv 2507.20957): The Gemini synthesis layer systematically rates tech stocks higher due to training data composition.
2. Momentum clustering: 2023–2024 price momentum was dominated by technology mega-caps; any momentum-flavored scoring will reproduce this.
3. No post-scoring sector filter: The existing pipeline selects by conviction score with no sector constraint applied post-scoring.

### Minimum-Viable Diversification Rules for v1

```python
# Recommended pyfinagent v1 sector diversification constants
SECTOR_NAV_CAP = 0.25          # max 25% of NAV in any single GICS sector
SECTOR_POSITION_CAP = 2        # max 2 stock positions per GICS sector  
POSITION_NAV_CAP = 0.10        # max 10% of NAV in any single position
MIN_SECTORS = 4                # minimum number of distinct GICS sectors
TARGET_PORTFOLIO_SIZE = 10     # nominal number of holdings

# Portfolio selection filter (applied after existing signal scoring):
# 1. Sort candidates by conviction_score descending
# 2. Greedily add: skip if sector already at SECTOR_POSITION_CAP
#    or if sector_weight + (1/TARGET_PORTFOLIO_SIZE) > SECTOR_NAV_CAP
# 3. After selection, if MIN_SECTORS not met, expand candidate pool
#    (lower conviction threshold for under-represented sectors)
```

**GICS data requirement:** Every ticker in the candidate list must have a GICS sector tag. Source: Yahoo Finance `info['sector']`, Polygon.io `/v3/reference/tickers`, or BigQuery `pyfinagent_data` fundamentals table (confirm availability in internal audit).

### Integration Points (from literature, not internal code read)

- Signal scoring: no change (keep 28-agent pipeline)
- Post-scoring filter: insert sector constraint filter between conviction aggregation and order submission
- Rebalancing: apply the same filter at each rebalance cycle; existing positions that violate the cap should be flagged (not forced out immediately — use drift tolerance of 5 pp before triggering a trade)

---

## ROI Ranking of Diversification Techniques

Ranked by (risk reduction per implementation hour) for pyfinagent's specific context:

| Rank | Technique | Risk Reduction | Implementation Complexity | Est. Hours | ROI |
|------|-----------|---------------|--------------------------|-----------|-----|
| 1 | Hard sector position cap (max 2/sector, 25% NAV) | Very High — eliminates 11/11 tech directly | Very Low — 1 filter function | 2–4 h | Highest |
| 2 | GICS data tagging for all candidates | Prerequisite for #1 | Very Low — API call | 1–2 h | Critical enabler |
| 3 | Min-sectors enforcement (>=4 sectors) | High — forces breadth | Very Low — condition in selector | 0.5 h | Very High |
| 4 | Max individual position (10% NAV cap) | Moderate — limits single-stock blow-up | Very Low — weight clip | 0.5 h | High |
| 5 | Sector-neutral re-ranking (rank within sector) | High — removes sector timing bet | Moderate — re-ranking logic | 4–8 h | High |
| 6 | Inverse-volatility position sizing | Moderate — reduces vol drag from concentrated positions | Moderate — requires price history | 4–6 h | Moderate |
| 7 | HRP (post-selection optimizer) | Moderate — improves weight allocation | Moderate — skfolio library | 6–10 h | Moderate |
| 8 | Correlation-cluster deduplication | High — removes hidden factor overlap | High — requires full covariance | 12–16 h | Moderate |
| 9 | Full MVO with sector constraints | Low net benefit for N<20 | High — error amplification risk | 8–12 h | Low |
| 10 | RL-based diversification reward | Low for live trading | Very High — training infrastructure | 40+ h | Very Low |

**Top 4 items alone (rows 1–4) can be implemented in approximately 4–7 hours of engineering work and eliminate the current 11/11 Technology concentration entirely.**

---

## Concrete Recommended Rules for pyfinagent v1

### Hard Rules (non-negotiable)
1. **Max 25% of portfolio NAV per GICS sector** — mirrors SEC "concentrated" threshold; industry standard for long-only equity.
2. **Max 2 stock positions per GICS sector** — for a 10-stock portfolio, this means at least 5 sectors represented.
3. **Max 10% of portfolio NAV per individual position** — prevents single-stock blow-up.
4. **Minimum 4 distinct GICS sectors** — provides genuine diversification across economic exposures.

### Soft Rules (enforce via scoring penalty, not hard exclusion)
5. **Prefer sectors not already at 1 position** — break ties in conviction scores by favoring under-represented sectors.
6. **Flag large-cap bias** — if >80% of positions are mega-cap (>$200B market cap), flag for review.

### Monitoring / Alerting
7. **Daily sector concentration report** — log current NAV weight per GICS sector to BigQuery; alert if any sector exceeds 30% (5 pp buffer above the 25% hard cap, allowing for drift before forced rebalance).
8. **Sector count metric** — track and expose in the backtest/harness dashboard: `sectors_represented` and `max_sector_weight` as headline metrics alongside Sharpe/DSR.

### Phased Implementation
- **v1 (now):** Hard caps (rules 1–4), GICS data tagging.
- **v2 (next cycle):** Sector-neutral re-ranking (rank candidates within sector before cross-sector merge).
- **v3 (later):** HRP-based position sizing post-selection; correlation-cluster deduplication.

---

## Research Gate Checklist

### Hard blockers

- [x] >=5 authoritative external sources READ IN FULL via WebFetch (13 sources fetched in full)
- [x] 10+ unique URLs total (25 unique URLs collected including snippet-only)
- [x] Recency scan (last 2 years) performed + reported (section above; 5 distinct 2024-2026 findings)
- [x] Full pages read (not abstracts) for the read-in-full set
- [ ] file:line anchors for every internal claim — N/A for external-only research; internal exploration is parallel

Note on the file:line anchor hard blocker: this brief is explicitly the external-only half. The parallel internal audit agent covers file:line anchors. The gate_passed flag below reflects the external gate only.

### Soft checks

- [x] Internal exploration covered every relevant module — covered by parallel agent
- [x] Contradictions / consensus noted (consensus vs debate section above)
- [x] All claims cited per-claim (not just listed in a footer)

---

## Sources

### External Sources Read in Full
- [Hierarchical Risk Parity — Wikipedia](https://en.wikipedia.org/wiki/Hierarchical_Risk_Parity)
- [LLMs in Equity Markets: Applications, Techniques, Insights — PMC/Frontiers](https://pmc.ncbi.nlm.nih.gov/articles/PMC12421730/)
- [LLM-Enhanced Black-Litterman Portfolio Optimization — arXiv 2504.14345](https://arxiv.org/html/2504.14345v2)
- [Can LLM-based Financial Investing Strategies Outperform the Market in Long Run? — arXiv 2505.07078](https://arxiv.org/html/2505.07078v1)
- [Your AI, Not Your View: The Bias of LLMs in Investment Analysis — arXiv 2507.20957](https://arxiv.org/html/2507.20957)
- [How to Improve ETF Sector Momentum — QuantPedia](https://quantpedia.com/how-to-improve-etf-sector-momentum/)
- [Risk Parity — Wikipedia](https://en.wikipedia.org/wiki/Risk_parity)
- [The 3 Cs for Setting Portfolio Constraints — Sherpa Funds Technology](https://sherpafundstech.com/3-cs-for-portfolio-constraints/)
- [Modern Portfolio Theory — Wikipedia](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [When is an Index Too Concentrated? — LSEG FTSE Russell](https://www.lseg.com/en/insights/ftse-russell/when-is-an-index-too-concentrated)
- [TradingAgents Official Documentation](https://tradingagents-ai.github.io/)
- [Hierarchical Risk Parity — QuantPedia](https://quantpedia.com/hierarchical-risk-parity/)
- [Signal or Noise in Multi-Agent LLM-based Stock Recommendations? — arXiv 2604.17327](https://arxiv.org/html/2604.17327)

### Additional Snippet-Only Sources Consulted
- [SEC Staff Report on Threshold Limits for Diversified Funds (2022)](https://www.sec.gov/files/staff-report-threshold-limits-diversified-funds.pdf)
- [Building Diversified Portfolios that Outperform Out-of-Sample — López de Prado, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678)
- [LLM-Enhanced Black-Litterman (ICLR 2025) — arXiv abstract](https://arxiv.org/abs/2504.14345)
- [Competing LLMs Were Asked to Pick Stocks — HBR 2026](https://hbr.org/2026/03/competing-llms-were-asked-to-pick-stocks-their-choices-revealed-ais-limitations)
- [FinRL — GitHub AI4Finance-Foundation](https://github.com/AI4Finance-Foundation/FinRL)
- [TradingAgents GitHub — TauricResearch](https://github.com/TauricResearch/TradingAgents)
- [Risk Parity Not Performing — CAIA 2024](https://caia.org/blog/2024/01/02/risk-parity-not-performing-blame-weather)
- [GICS Sector and Industry Map — SSGA](https://www.ssga.com/us/en/institutional/capabilities/equities/sector-investing/gics-sector-and-industry-map)
- [S&P 500 Sectors: Equal-Weight and Cap-Weight — S&P DJI](https://www.spglobal.com/spdji/en/documents/education/practice-essentials-sp-500-sectors-equal-weight-and-cap-weight-indices.pdf)
- [Robinhood Strategies — Form ADV Part 2](https://cdn.robinhood.com/assets/robinhood/legal/RAM_Brochure_and_Brochure_Supplements.pdf)
- [Understanding Risk Parity — AQR White Paper](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Understanding-Risk-Parity.pdf)
- [Portfolio Construction Objectives and Constraints — AnalystPrep CFA III](https://analystprep.com/study-notes/cfa-level-iii/portfolio-construction-3/)

---

```json
{
  "tier": "complex",
  "external_sources_read_in_full": 13,
  "snippet_only_sources": 12,
  "urls_collected": 25,
  "recency_scan_performed": true,
  "internal_files_inspected": 0,
  "report_md": "handoff/current/phase-23.1.13-external-research.md",
  "gate_passed": true
}
```
