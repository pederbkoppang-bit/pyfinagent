# Phase 5.5 Literature Review: Modern Alternative Data Sources for Quant Trading

**Prepared for:** pyfinagent trading signal system
**Date:** 2026-04-17
**Scope:** 2024-2026 citations; 10 alt-data categories

---

## Context

pyfinagent is an autonomous AI-powered trading signal system targeting May 2026 go-live. Current best
metrics are Sharpe 1.1705 and DSR 0.9984. The system runs a Planner -> Generator -> Evaluator harness
with a FastAPI + BigQuery + Gemini + Claude stack. The objective of this review is to survey the current
state of alternative data research across ten categories and produce actionable recommendations for which
signals to add, keep, or replace in the existing pipeline.

Alternative data -- non-traditional data sources beyond price, volume, and fundamentals -- now represent
a multi-billion-dollar industry. As of 2024, more than 70% of the best-performing quantitative hedge funds
use at least one commercial alt-data product (RavenPack, 2025). The academic literature confirms alpha
decay is accelerating; the half-life of a factor discovered in 2000 is roughly five years (Harvey et al.,
2016). This makes fresh, hard-to-replicate data sources the primary moat for systematic strategies.

---

## Taxonomy

| Tier | Category | Cost | Decay Rate | Replication Barrier |
|------|----------|------|-----------|---------------------|
| 1 | Institutional filings (13F/Form 4) | Free (SEC) | Slow (90-day lag) | Low |
| 1 | Economic nowcasting (Fed) | Free (API) | Moderate | Low |
| 2 | News/sentiment (NLP) | Low-Med | Medium | Medium |
| 2 | Short interest / FINRA | Free/Low | Fast | Low |
| 2 | Retail social (WSB/Stocktwits) | Low | Very fast | Low |
| 2 | Search interest (Trends/Wikipedia) | Free | Fast | Low |
| 3 | Options flow (unusual activity) | Med | Very fast | Medium |
| 3 | Patent activity (Google/USPTO) | Free | Very slow | Low |
| 4 | Consumer spending (transaction) | High | Medium | High |
| 4 | Satellite/geospatial | Very High | Slow | Very High |

---

## Signal Classes

### 1. News and Sentiment (NLP)

Natural language processing applied to financial news is now the most mature alt-data category. The
progression from bag-of-words models through FinBERT to instruction-tuned LLMs is well-documented.

**FinBERT** (Araci, 2019; arXiv:1908.10063) remains the standard baseline. Fine-tuned on financial
phrases corpus and Reuters news, it classifies sentiment as positive, negative, or neutral at sentence
level. Accuracy on the FPB benchmark is consistently around 86%. https://arxiv.org/abs/1908.10063 FULL READ

**FinGPT** (Yang et al., 2023, updated November 2025; arXiv:2306.06031) is the leading open-source
financial LLM. It uses low-rank adaptation (LoRA) fine-tuning on top of general-purpose LLMs, enabling
practitioners to fine-tune a 7B model on consumer hardware for under $300. On sentiment tasks it
outperforms zero-shot ChatGPT and FinBERT. https://arxiv.org/abs/2306.06031 FULL READ

**FinGPT Dissemination-Aware** (Liang et al., December 2024; arXiv:2412.10823) addresses a key gap:
prior models ignored how far a news item spread. By clustering news by dissemination reach and
incorporating that as context, they achieved an 8% improvement in short-term stock movement
prediction over vanilla LLM approaches. https://arxiv.org/abs/2412.10823 FULL READ

**Hybrid LLM-Transformer** (arXiv:2601.02878, January 2026): formalizes LLM-as-signal-generator
feeding a Transformer with noise-robust gating. Demonstrated 5.28% RMSE reduction (p=0.003) over
a pure Transformer baseline. Useful architecture pattern for pyfinagent's existing Gemini pipeline.
https://arxiv.org/abs/2601.02878 FULL READ

**RavenPack** is the dominant commercial platform. Its 2025 backtests show that adding premium FT
journalism to broad news coverage improves annualized equity performance by approximately 120 bps
and raises Information Ratio from 0.59 to 0.73 (2-day holding period).
https://www.ravenpack.com/research/sentiment-driven-stock-selection FULL READ
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5560198

### 2. Institutional Filings (13F / 13D / Form 4)

SEC EDGAR provides bulk download of all institutional filings at no cost. Form 13F (quarterly long
positions, filed within 45 days of quarter-end), Form 13D/13G (5%+ ownership disclosures), and
Form 4 (insider transaction within 2 business days) are the three primary signals.

Key resources:
- SEC EDGAR Form 13F data sets (updated quarterly since March 2024): https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets FULL READ
- edgartools Python library for parsing all filing types: https://edgartools.readthedocs.io/ FULL READ
- WhaleWisdom: aggregates 13F filings, provides backtester for cloning top funds: https://whalewisdom.com/
- QuiverQuant: congressional STOCK Act trades, 13F, Form 4, with a Python API: https://github.com/Quiver-Quantitative/python-api FULL READ
- QuantConnect integration for US Congress Trading data: https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/quiver-quantitative/us-congress-trading

The 45-90 day lag on 13F data is a well-documented limitation; the alpha from following filing-disclosed
positions is concentrated in small-caps (Brunnermeier & Nagel, 2004) where institutional activity is
less efficiently priced. Congressional trades via STOCK Act have shown stronger short-term alpha due
to the 45-day (was 30-day) reporting deadline -- faster than 13F.

### 3. Short Interest and Options Flow

**FINRA Short Interest**: FINRA publishes bi-monthly short interest for all exchange-listed and OTC
equities under Rule 4560. Data is free, available via interactive API (5 rolling years), and as
historical file downloads. The API supports automated ingestion.
https://www.finra.org/finra-data/browse-catalog/equity-short-interest FULL READ
https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files

**Daily short sale volume** (also FINRA): published on a T+1 basis, distinct from bi-monthly short
interest snapshots. More timely but noisier.

**Unusual options activity**: detects informed positioning before announcements. Academic research
supports that unusual call/put volume predicts earnings surprises (Pan & Poteshman, 2006). Commercial
providers include Unusual Whales and Market Chameleon. No direct free API; scraping or subscription
required.

### 4. Retail Social Sentiment

**WallStreetBets / Reddit**: Social media attention and retail investor behavior research (Journal of
Financial Analysis, 2024) finds that high-WSB-attention trades are associated with negative subsequent
returns and higher volatility, suggesting retail meme attention is a contrarian signal rather than
momentum signal at the peak.
https://www.sciencedirect.com/article/pii/S1057521924006537 FULL READ

ACM Transactions on Social Computing (2024) found WSB has statistically significant Granger-causal
relationships with GME and AMC returns at specific lags, but limited generalizability across the
broader equity universe. https://dl.acm.org/doi/10.1145/3660760

**Stocktwits**: 10M+ users, explicit bullish/bearish tagging per post. Research in Digital Finance
(Springer, 2023) shows sentiment polarity is positively correlated with contemporaneous returns but
has weak predictive power for next-day returns in isolation; predictive power emerges when conditioned
on volume spikes. FinBERT applied to Stocktwits outperforms raw sentiment tags.
https://link.springer.com/article/10.1007/s42521-023-00102-z
https://pmc.ncbi.nlm.nih.gov/articles/PMC10280432/

**X (Twitter) firehose**: full firehose access now requires Enterprise API subscription at $42,000/month
as of 2023 price changes. The academic/research API tier was deprecated. For pyfinagent budget
constraints this is not viable without Peder approval.

### 5. Search Interest and Wikipedia Pageviews

**Google Trends / pytrends**: pytrends (GeneralMills) is an unofficial API wrapper with 200,000+
monthly PyPI downloads. As of 2024, Google changed endpoint URLs, causing intermittent failures
(GitHub issue #638). Google launched an official Trends API in alpha in 2025, but quotas are limited.
https://github.com/GeneralMills/pytrends FULL READ
https://github.com/GeneralMills/pytrends/issues/638
https://developers.google.com/search/blog/2025/07/trends-api

**Wikipedia pageviews**: Freely available since May 2015 at the Wikimedia REST API. Provides daily
page view counts by article and project. Researchers have used company article view spikes as an
investor attention proxy. Considered more stable than pytrends for automated ingestion.
https://franz101.substack.com/p/google-trends-api-alternative-wikipedia

### 6. Economic Nowcasting

**NY Fed Staff Nowcast**: Updates weekly, tracking real GDP growth estimates for the current and next
quarter. Incorporates a Dynamic Factor Model across dozens of macro releases (payrolls, PMI, retail
sales). The research methodology is published (Staff Report SR-830, Bok et al.).
https://www.newyorkfed.org/research/policy/nowcast
https://www.newyorkfed.org/research/staff_reports/sr830.html FULL READ

**NY Fed Markets Data APIs**: REST endpoints for SOFR, repo rates, Treasury auctions, and more,
available without authentication. https://markets.newyorkfed.org/static/docs/markets-api.html

**FRED API** (St. Louis Fed): 840,000 economic time series from 119 sources. Free with API key.
fredapi Python wrapper provides Pandas-native access. The best free macro data backbone for any
systematic strategy. https://fred.stlouisfed.org/
https://github.com/mortada/fredapi

### 7. Satellite and Geospatial

**Orbital Insight**: Processes satellite imagery for parking lot counts, oil tank fill levels, crop
yields, and ship traffic. Used by hedge funds to nowcast retail earnings and commodity supply.

**SpaceKnow**: Provides AI-powered satellite analytics for industrial activity monitoring. Typical
per-symbol cost is in the $10,000-$50,000/year range.

Academic validation: A Nature Humanities and Social Sciences Communications paper (2023) found that
container port imagery from satellite data predicts global stock returns with a Sharpe ratio improvement
of approximately 0.3 vs. price-only models.
https://www.nature.com/articles/s41599-023-01891-9 FULL READ

**Legality**: The SEC has not issued specific guidance prohibiting satellite-derived data. The mosaic
theory (codified under Reg FD) permits use of public non-material information. Satellite imagery of
publicly visible infrastructure (parking lots, ports) is generally considered public observation.
However, imagery providers themselves have ToS restricting redistribution. The Neudata regulatory
review (2024) notes increasing scrutiny on what constitutes "public" data when the collection method
involves commercial surveillance capabilities.
https://www.neudata.co/blog/the-nascent-regulatory-landscape-of-web-scraping FULL READ

Cost barrier: Orbital Insight and SpaceKnow are out of budget range for pyfinagent without explicit
Peder approval.

### 8. Patent Activity

**Google Patents Public Data** on BigQuery: A collection of USPTO, EPO, and international patent
filings accessible via BigQuery SQL. The project `patents-public-data` contains tables for
publications, claims, citations, and classifications. Free tier: 1 TB/month.
https://github.com/google/patents-public-data FULL READ
https://cloud.google.com/blog/topics/public-datasets/google-patents-public-datasets-connecting-public-paid-and-private-patent-data

**PatentsView** (USPTO): Structured relational patent data with assignee disambiguation, available
as bulk CSV downloads. https://patentsview.org/external-resources-patentsview-users

Patent filing velocity per company predicts innovation output and has been linked to long-run stock
outperformance in technology sectors (Hirshleifer et al., 2013). The signal decays slowly due to
filing-to-publication lag (18 months average), but is very low cost to ingest for pyfinagent given
existing BigQuery infrastructure.

### 9. Alt-Credit and Consumer Spending

**Second Measure** (acquired by Bloomberg, 2020): Transaction-level consumer spending data derived
from credit/debit panel. The Bloomberg Second Measure product tracks daily revenue estimates for
thousands of public and private companies. Access requires Bloomberg Terminal subscription.
https://www.bloomberg.com/professional/insights/data/alternative-data-insights-consumer-spending-growth-in-2025/

**Earnest Analytics** (acquired by Consumer Edge, April 2025): Maintained an Orion Consumer Spend
product tracking 2,500+ national brands. Claimed 90% accuracy vs. actuals for earnings surprises.
Post-acquisition, branding is Consumer Edge.
https://www.alleywatch.com/2025/04/consumer-edge-acquires-earnest-analytics-alternative-data-transaction-healthcare-intelligence/
https://www.earnestanalytics.com/insights/june-2023-earnest-analytics-spend-index

**Cost flag**: Both products are high-cost enterprise subscriptions. Peder explicit approval required
before any engagement.

### 10. AI Frontier: Time Series Foundation Models

A new class of zero-shot forecasting models emerged in 2024-2025, enabling inference on financial
time series without task-specific training.

**Chronos** (Amazon, arXiv:2403.07815, March 2024): T5-based probabilistic framework, tokenizes
time series as discrete bins. Zero-shot performance on 42 benchmarks matches or exceeds task-specific
models. Model sizes from 20M to 710M parameters. Chronos-2 (arXiv:2510.15821, October 2025) extends
to multivariate and covariate-informed tasks.
https://arxiv.org/html/2403.07815v1 FULL READ
https://github.com/amazon-science/chronos-forecasting FULL READ
https://arxiv.org/html/2510.15821v1

**TimesFM** (Google, 2024): Decoder-only foundation model pre-trained on 100 billion time points
including Google Trends and Wikipedia pageviews. Handles sequences up to 2048 steps via patching.
Released as open weights.

**Moirai 2.0** (Salesforce, arXiv:2511.11698, November 2025): Decoder-only, trained on 36M series.
2x faster and 30x smaller than Moirai 1.0-Large. Uses quantile loss and multi-token prediction.
Ranks among top pretrained models on the Gift-Eval benchmark.
https://arxiv.org/html/2511.11698v1 FULL READ
https://www.salesforce.com/blog/time-series-morai-moe/

---

## Providers

| Provider | Category | Cost | API Available | Free Tier |
|----------|----------|------|---------------|-----------|
| SEC EDGAR | Filings | Free | Yes (bulk JSONL) | Yes |
| FINRA | Short interest | Free | Yes (REST) | Yes |
| FRED (St. Louis Fed) | Macro | Free | Yes | Yes (key) |
| NY Fed Nowcast | Nowcasting | Free | Download only | Yes |
| Google Patents BQ | Patents | Free (1TB/mo) | BigQuery SQL | Yes |
| Wikipedia Pageviews | Attention | Free | Yes (REST) | Yes |
| QuiverQuant | Congress/13F/social | $12.50/mo | Yes (Python) | Limited |
| WhaleWisdom | 13F tracker | Tiered | Yes | 8 qtrs |
| FinBERT (HuggingFace) | NLP model | Free | Library | Yes |
| FinGPT (open source) | NLP model | Free (compute) | Library | Yes |
| RavenPack | News sentiment | Enterprise | Yes | No |
| Stocktwits | Retail social | Tiered | Yes | Yes |
| pytrends | Search interest | Free* | Library | Yes* |
| Chronos / Moirai | TS foundation | Free (weights) | Library | Yes |
| Orbital Insight | Satellite | Very High | Yes | No |
| Second Measure | Consumer spend | Enterprise | Via Bloomberg | No |
| Consumer Edge (ex-Earnest) | Consumer spend | Enterprise | Yes | No |

*pytrends reliability degraded in 2024 due to endpoint changes; use official Trends API alpha or
Wikipedia pageviews as fallback.

---

## Evidence on Alpha

| Signal | Evidence | Horizon | Sharpe Improvement | Source |
|--------|----------|---------|-------------------|--------|
| News sentiment (RavenPack) | 120 bps ann., IR 0.59->0.73 | 2-day | +0.14 IR | RavenPack 2025 |
| Earnings call NLP | IR 1.2 (mid/large cap) | 2 weeks | Strong | RavenPack SSRN 5559999 |
| Insider Form 4 buys | Documented, decay 1-2 years | 1 month | Moderate | Multiple |
| Congressional trades | $100M+ backtested | 45 days | Significant | QuiverQuant |
| Short interest | Classic short-squeeze factor | 1 week | Low standalone | FINRA |
| Satellite (parking lots) | 4-5% in 3 days near earnings | Event | Strong | RS Metrics study |
| Container port imagery | Sharpe +0.3 vs price-only | Global macro | +0.3 Sharpe | Nature HSSC 2023 |
| WSB social | Granger-causes GME/AMC at lags | 1-5 days | Contrarian | ACM Trans 2024 |
| FinGPT dissemination | 8% prediction improvement | Short-term | Moderate | arXiv 2412.10823 |
| Consumer spend (Earnest) | 90% earnings surprise accuracy | Event | High | Consumer Edge |
| Patent velocity | Long-run tech outperformance | 12+ months | Low near-term | Hirshleifer 2013 |
| Google Trends | Investor attention proxy | 1-4 weeks | Moderate | Multiple |

---

## Risks and ToS

**Mosaic Theory and Legality**: Combining multiple public non-material signals is legally protected
under the mosaic theory (CFA Institute standards; SEC Reg FD). The key bright lines are: (1) the
data source must be lawfully obtained; (2) it must not constitute material non-public information
(MNPI). SEC v. Dorozhko (2009) established that using hacked data to trade violates Section 10(b)
even without a fiduciary duty breach.
https://journals.library.columbia.edu/index.php/CBLR/announcement/view/198

**Web Scraping ToS**: Many providers (Reddit, X/Twitter, Stocktwits) have ToS that restrict
automated scraping. Reddit's 2023 API pricing changes effectively ended free high-volume access.
The Neudata 2024 regulatory review notes increasing EU Digital Markets Act scrutiny of data
aggregation practices. https://www.neudata.co/blog/the-nascent-regulatory-landscape-of-web-scraping

**Data staleness (13F)**: The 45-day filing deadline means 13F-based signals reflect positions
that may have fully changed. Backtest results using point-in-time 13F data are often
look-ahead-biased in naive implementations.

**Model overfitting on sentiment**: Harvey et al. (2016, t-stat >= 3.0 threshold) applies equally
to alt-data factors. RavenPack's own research acknowledges that many sentiment signals have
in-sample IR that does not persist out-of-sample when the factor is published (backtest overfitting).

**Satellite legality**: No specific SEC prohibition, but the GDPR in the EU and state privacy laws
in the US are increasingly applied to commercial surveillance data. Providers like Orbital Insight
have their own redistribution restrictions in ToS.

**Budget / approval gates**: Orbital Insight, Second Measure, Consumer Edge, and RavenPack
Enterprise all require Peder explicit approval due to cost exceeding the project's $230-350/mo
compute budget.

---

## Recommendations for pyfinagent

The following table maps each alt-data class to current provider status and recommended action.
Priority 1 = immediate value, Priority 3 = long-term / budget-gated.

| # | Alt-Data Class | Current Status | Provider Candidate | Action | Priority |
|---|---------------|----------------|--------------------|--------|----------|
| 1 | News/NLP sentiment | Gemini agents with news (unstructured) | FinBERT or FinGPT LoRA on open news | Replace Gemini free-text with FinBERT scored sentiment feature; lower cost, structured output | 1 |
| 2 | Economic macro/nowcasting | FRED data partially used | FRED API + NY Fed Nowcast JSON download | Keep FRED; ADD NY Fed weekly nowcast as leading macro feature | 1 |
| 3 | Insider Form 4 filings | Not currently ingested | SEC EDGAR bulk JSONL + edgartools | Add insider net buy/sell ratio as a feature; free and low-latency | 1 |
| 4 | FINRA short interest | Not currently ingested | FINRA REST API (free) | Add days-to-cover ratio; bi-monthly lag acceptable for swing signals | 1 |
| 5 | Wikipedia pageviews | Not currently ingested | Wikimedia REST API (free) | Add as attention proxy; more reliable than pytrends for automation | 2 |
| 6 | Congressional trades | Not currently ingested | QuiverQuant ($12.50/mo) | Add; low cost, documented alpha, easy Python API | 2 |
| 7 | Patent activity | Not currently ingested | Google Patents BQ (free 1TB/mo) | Add patent filing velocity for tech sector; reuses existing BQ infra | 2 |
| 8 | TS foundation models | Not used; Gemini for NLP | Chronos (Amazon) or Moirai 2.0 (Salesforce) | Evaluate Chronos zero-shot on price features as ensemble component | 2 |
| 9 | Retail social (Stocktwits/WSB) | Not currently ingested | Stocktwits public API or Pushshift Reddit archives | Add as contrarian signal on extreme sentiment; use FinBERT for scoring | 3 |
| 10 | Consumer spending (transaction) | Not ingested | Consumer Edge (ex-Earnest) | FLAG: requires Peder approval; high cost; high alpha for event-driven | 3 |
| 11 | Satellite/geospatial | Not ingested | Orbital Insight / SpaceKnow | FLAG: requires Peder approval; out of budget; highest replication barrier | 3 |
| 12 | 13F institutional holdings | Not currently ingested | WhaleWisdom API or SEC EDGAR direct | Add as contrarian/confirmation signal; free via EDGAR; 90-day lag noted | 2 |

**Immediate wins (add within current budget):** Items 1-5 are all free or near-free and have
documented alpha in the literature. The FinBERT-over-news replacement (item 1) is the highest
priority because it improves signal quality on data pyfinagent already ingests.

**Second wave (sub-$50/month):** Items 6-8 expand coverage into filings and model diversity
with minimal incremental cost.

**Budget-gated (Peder approval required):** Items 10-11 require explicit approval per CLAUDE.md
cost controls before any API trial or data purchase.

---

## References

| # | Citation | URL | Access Date |
|---|----------|-----|-------------|
| 1 | Araci D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063 | https://arxiv.org/abs/1908.10063 | 2026-04-17 |
| 2 | Yang H. et al. (2023, updated 2025). FinGPT: Open-Source Financial Large Language Models. arXiv:2306.06031 | https://arxiv.org/abs/2306.06031 | 2026-04-17 |
| 3 | Liang Z. et al. (2024). FinGPT: Enhancing Sentiment-Based Stock Movement Prediction. arXiv:2412.10823 | https://arxiv.org/abs/2412.10823 | 2026-04-17 |
| 4 | Zhou & Mehra (2026). Hybrid LLM-Transformer for Stock Price Prediction. arXiv:2601.02878 | https://arxiv.org/abs/2601.02878 | 2026-04-17 |
| 5 | Das et al. (2025). Moirai 2.0: Time Series Foundation Models. arXiv:2511.11698 | https://arxiv.org/html/2511.11698v1 | 2026-04-17 |
| 6 | Ansari A. et al. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815 | https://arxiv.org/html/2403.07815v1 | 2026-04-17 |
| 7 | Chronos-2 (2025). From Univariate to Universal Forecasting. arXiv:2510.15821 | https://arxiv.org/html/2510.15821v1 | 2026-04-17 |
| 8 | RavenPack (2025). Sentiment-Driven Stock Selection. | https://www.ravenpack.com/research/sentiment-driven-stock-selection | 2026-04-17 |
| 9 | Hafez P. et al. (2025). Harnessing News Sentiment for FX Futures. SSRN 5560198 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5560198 | 2026-04-17 |
| 10 | Hafez P. et al. (2025). Introducing Earnings Intelligence Factors. SSRN 5559999 | https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5559999 | 2026-04-17 |
| 11 | SEC EDGAR Form 13F Data Sets. | https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets | 2026-04-17 |
| 12 | edgartools Python library. | https://edgartools.readthedocs.io/ | 2026-04-17 |
| 13 | QuiverQuant Python API. | https://github.com/Quiver-Quantitative/python-api | 2026-04-17 |
| 14 | QuantConnect US Congress Trading Docs. | https://www.quantconnect.com/docs/v2/writing-algorithms/datasets/quiver-quantitative/us-congress-trading | 2026-04-17 |
| 15 | FINRA Equity Short Interest. | https://www.finra.org/finra-data/browse-catalog/equity-short-interest | 2026-04-17 |
| 16 | FINRA Daily Short Sale Volume. | https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files | 2026-04-17 |
| 17 | Swagel et al. (2024). Social media attention and retail investor behavior. Journal of Financial Analysis. ScienceDirect. | https://www.sciencedirect.com/article/pii/S1057521924006537 | 2026-04-17 |
| 18 | Kolasinski et al. (2024). WallStreetBets: Collective Intelligence. ACM Trans Social Computing. | https://dl.acm.org/doi/10.1145/3660760 | 2026-04-17 |
| 19 | StockTwits classified sentiment and stock returns. Digital Finance, Springer 2023. | https://link.springer.com/article/10.1007/s42521-023-00102-z | 2026-04-17 |
| 20 | Stock price prediction using Stocktwits and FinBERT. PMC. | https://pmc.ncbi.nlm.nih.gov/articles/PMC10280432/ | 2026-04-17 |
| 21 | NY Fed Staff Nowcast. | https://www.newyorkfed.org/research/policy/nowcast | 2026-04-17 |
| 22 | Bok et al. Macroeconomic Nowcasting with Big Data. NY Fed SR-830. | https://www.newyorkfed.org/research/staff_reports/sr830.html | 2026-04-17 |
| 23 | NY Fed Markets Data APIs. | https://markets.newyorkfed.org/static/docs/markets-api.html | 2026-04-17 |
| 24 | FRED Economic Data. | https://fred.stlouisfed.org/ | 2026-04-17 |
| 25 | mortada/fredapi Python wrapper. | https://github.com/mortada/fredapi | 2026-04-17 |
| 26 | Google Patents Public Data on BigQuery. | https://github.com/google/patents-public-data | 2026-04-17 |
| 27 | Google Patents Public Datasets blog. | https://cloud.google.com/blog/topics/public-datasets/google-patents-public-datasets-connecting-public-paid-and-private-patent-data | 2026-04-17 |
| 28 | PatentsView external resources. | https://patentsview.org/external-resources-patentsview-users | 2026-04-17 |
| 29 | Container port satellite data and stock returns. Nature HSSC 2023. | https://www.nature.com/articles/s41599-023-01891-9 | 2026-04-17 |
| 30 | Neudata: regulatory landscape of web scraping. | https://www.neudata.co/blog/the-nascent-regulatory-landscape-of-web-scraping | 2026-04-17 |
| 31 | SEC v. Dorozhko commentary. Columbia Business Law Review. | https://journals.library.columbia.edu/index.php/CBLR/announcement/view/198 | 2026-04-17 |
| 32 | pytrends GitHub (GeneralMills). | https://github.com/GeneralMills/pytrends | 2026-04-17 |
| 33 | pytrends endpoint change issue #638. | https://github.com/GeneralMills/pytrends/issues/638 | 2026-04-17 |
| 34 | Google official Trends API alpha (2025). | https://developers.google.com/search/blog/2025/07/trends-api | 2026-04-17 |
| 35 | Wikipedia pageviews as Google Trends alternative. | https://franz101.substack.com/p/google-trends-api-alternative-wikipedia | 2026-04-17 |
| 36 | Bloomberg Second Measure consumer spending 2025. | https://www.bloomberg.com/professional/insights/data/alternative-data-insights-consumer-spending-growth-in-2025/ | 2026-04-17 |
| 37 | Consumer Edge acquires Earnest Analytics (April 2025). AlleyWatch. | https://www.alleywatch.com/2025/04/consumer-edge-acquires-earnest-analytics-alternative-data-transaction-healthcare-intelligence/ | 2026-04-17 |
| 38 | Chronos forecasting GitHub (Amazon Science). | https://github.com/amazon-science/chronos-forecasting | 2026-04-17 |
| 39 | WhaleWisdom 13F tracker. | https://whalewisdom.com/ | 2026-04-17 |
| 40 | Salesforce Moirai-MoE blog. | https://www.salesforce.com/blog/time-series-morai-moe/ | 2026-04-17 |
