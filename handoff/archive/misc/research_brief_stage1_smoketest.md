# Research Brief: phase-31.0.1 Stage 1 Smoketest (`screen_universe`)

**Step:** phase-31.0.1 -- Smoketest Stage 1: `screen_universe(["AAPL","MSFT","NVDA","JPM"])` returns 4 enriched candidate dicts with sector + composite_score populated.

**Tier:** deep | **Effort:** max | **Date:** 2026-05-19
**Methodology:** Pass-1 broad scan + Pass-2 adversarial + cross-domain triangulation
**Floor:** 20+ sources read in full, 40+ URLs collected, >=1 [ADVERSARIAL] tag

## Objective

Validate that `screen_universe` correctly enriches the 4-ticker synthetic basket (AAPL, MSFT, NVDA, JPM) with `sector` and a scorable composite metric, ready to feed Stage 2 of the 13-stage end-to-end smoketest. Cite quantitative-screening best practice and survivorship-bias mitigation literature. Audit the actual code at `backend/tools/screener.py` for the signature and return shape.

## Three-variant search composition

Mandatory per `.claude/rules/research-gate.md`: each topic queries 2026 / 2025 / year-less canonical.

Topic 1: "quantitative stock screening best practice"
Topic 2: "GICS sector classification methodology yfinance"
Topic 3: "multi-factor composite score RSI momentum volatility 2026"
Topic 4: "survivorship bias S&P 500 universe construction"
Topic 5: "smoke test trading pipeline basket size synthetic"

---

## Pass 1: Broad scan (read in full)

| # | URL | Accessed | Kind | Fetched | Key quote / finding |
|---|-----|----------|------|---------|---------------------|
| 1 | https://www.quant-investing.com/blog/how-to-implement-a-multi-factor-quantitative-investment-strategy | 2026-05-19 | Industry blog | WebFetch full | Multi-factor composite = value-composite (5 ratios: P/B, P/S, EV/EBIT, EV/EBITDA, EV/FCF) + momentum (3-12mo) + smaller weights to yield/quality/low-vol. Intra-sector ranking improves risk-adj returns. Workflow: export -> composite -> intra-sector rank -> expected return -> constraints -> ~30-35 names. |
| 2 | https://arxiv.org/html/2508.18592v1 | 2026-05-19 | Peer-reviewed (arXiv) | WebFetch full | Entropy Weight Method + IC_Mean dynamic weighting. After LASSO factor screening (37 of 50 retained), Sharpe rose from 24.28% to 61.17%. Empirical proof that factor quality matters more than factor count. CNE5 Barra extension. |
| 3 | https://www.analyticalplatform.com/the-hidden-impact-of-survivorship-bias-on-backtesting-results-of-investment-strategies/ | 2026-05-19 | Industry/quant blog | WebFetch full | Survivorship bias inflates SP500 returns by 1.45pp/yr; for 20-smallest small-cap strategy, by 26.84pp/yr (5x growth distortion). Mitigation: use point-in-time historical constituent lists at each rebalance, not today's snapshot. |
| 4 | https://www.fidelity.com/learning-center/trading-investing/markets-sectors/global-industry-classification-standard | 2026-05-19 | Official broker/edu | WebFetch full | GICS 4-tier hierarchy: 11 sectors / 25 industry groups / 74 industries / 163 subindustries. Companies classified by primary revenue + earnings + market perception. 8-digit code (Sector=25 -> ... -> Subindustry=25201040). Annual review + significant-event triggers. |
| 5 | https://dagster.io/blog/smoke-test-data-pipeline | 2026-05-19 | Practitioner blog (Dagster) | WebFetch full | Smoke-test discipline: run ALL transforms on empty/synthetic data to catch "stupid mistakes" in 5s vs 5min (60x faster dev). Stub I/O at source (custom resource); never clobber prod data. No specific basket size given -- emphasis on schema-shape validation, not data volume. |
| 6 | https://blankcapitalresearch.com/learn/jegadeesh-titman-momentum | 2026-05-19 | Quant research blog | WebFetch full | Jegadeesh-Titman 1993 canonical: 12-month formation, skip last 1 month (reversal noise), hold 3-6 months. 12/3 cell = 1.01% monthly return; long-short top-vs-bottom decile ~12% annualized excess return 1965-1989. 25% weight in canonical composites. |
| 7 | https://pmc.ncbi.nlm.nih.gov/articles/PMC8531567/ | 2026-05-19 | Peer-reviewed (PMC) | WebFetch full | European equity factor scoring: 3 factors (value 4 metrics, profitability 3 metrics, momentum 12-1). Iterative (hierarchical) best -- Sharpe 0.94, 16.54% return, Jensen alpha 5.65%. Mixed strategies Sharpe 0.80. Notes factor decay post-2002. |
| 8 | https://microsoft.github.io/code-with-engineering-playbook/automated-testing/smoke-testing/ | 2026-05-19 | Official engineering playbook | WebFetch full | "Smoke tests cover only the most critical application path... not... to actually test behavior, keeping execution time and complexity to minimum." Subset of integration / e2e tests; gate-style use. No prescribed basket size -- emphasis on speed + critical-path coverage. |
| 9 | https://abstracta.us/blog/testing-strategy/smoke-testing-in-software-testing/ | 2026-05-19 | Industry blog | WebFetch full | "Smoke testing checks basic functionality of the application, while regression testing validates recent changes... haven't introduced new defects." Small number of test cases, rapid execution, gating mechanism. Use after every build. |
| 10 | https://newrelic.com/blog/how-to-relic/smoke-testing-with-synthetic-monitors | 2026-05-19 | Vendor engineering blog | WebFetch full | Critical-path strategy: core user journeys (login, checkout, API). Should take "just a few minutes"; run "immediately after deployment"; staging + production; from 3+ locations. No prescribed data basket size -- prescriptions are on speed + path coverage. |
| 11 | https://riazarbi.github.io/quant/backtesting-sp500-constituent-history/ | 2026-05-19 | Quant blog | WebFetch full | Two PIT-correct sources for SP500: (1) iShares historical holdings query-param hack (back to 2007); (2) Wikipedia revisions parsed with pywikibot. Wikipedia revisions "guaranteed to avoid survivorship and look ahead bias." Implementation: Python pywikibot + wikitextparser. |
| 12 | https://teddykoker.com/2019/05/creating-a-survivorship-bias-free-sp-500-dataset-with-python/ | 2026-05-19 | Practitioner blog | WebFetch full | iShares IVV monthly holdings scrape back to 2006 (BeautifulSoup) + Quandl WIKI prices + Yahoo Finance fallback. Manual ticker-rename table for symbol changes. Master tickers list + individual CSVs. |
| 13 | https://eodhd.com/financial-academy/financial-faq/survivorship-bias-free-financial-analysis | 2026-05-19 | Vendor documentation | WebFetch full | EODHD HistoricalTickerComponents filter for SP500 PIT membership. Delisted-ticker reuse warning ('_old' suffix). TWTR-2022 cited as ~$44B example that "can't be ignored." No bulk magnitude numbers. |
| 14 | https://mrkaiwu.github.io/ml_multifactor.html | 2026-05-19 | Quant practitioner | WebFetch full | ML ensemble multifactor (LightGBM + NN + LSTM × regression / classification / ranking). 18-config ensemble achieves 46.51% annualized return. Author withholds factor names. Cross-validates that ML composites beat single-factor models. |
| 15 | https://martinfowler.com/articles/practical-test-pyramid.html | 2026-05-19 | Industry standard reference | WebFetch full | "Write lots of small and fast unit tests. Write some more coarse-grained tests and very few high-level tests that test your application from end to end." End-to-end "notoriously flaky" -- focus on "high-value interactions users will have with your application." "Push tests as far down the pyramid as you can." |
| 16 | https://quantitativepy.substack.com/p/building-a-crypto-trading-bot-from-2bb | 2026-05-19 | Practitioner blog | WebFetch full | PaperTradingAdapter shares ExchangeBase interface with prod adapter -- swap with config flag. State via dict (balances/positions/orders). Validates: duplicate-order prevention, fee calc, slippage, risk limits, position tracking, error paths. No specific basket size prescribed. |
| 17 | https://arxiv.org/html/2507.15876v1 | 2026-05-19 | Peer-reviewed (arXiv) | WebFetch full | Bayesian-graphical CTA replication. Short-term trend factors (10/20/40/60d) + long-term (500d). MKT+STT post-cost Sharpe 0.49 vs SG CTA Trend benchmark 0.03 (2010-2025). Execution 2bp + roll 10-20bp + mgmt 0.5%. |
| 18 | https://simplified-zone.com/master-backtesting-optimise-your-backtesting-portfolio/ | 2026-05-19 | Practitioner blog | WebFetch full | 8-name demo basket (AAPL, MSFT, GOOGL, AMZN, JPM, V, PG, JNJ) for walk-forward MPT backtest -- direct analog for smoketest sizing. Suggests "experiment with different rebalancing frequencies, different sets of assets." No formal smoke-test structure. |
| 19 | https://medium.com/@timemoneycode/mastering-python-backtesting-for-trading-strategies-1f7df773fdf5 | 2026-05-19 | Practitioner blog | WebFetch full | Tiered workflow recommendation: "Prototype ideas quickly with backtesting.py, validate execution realism with Backtrader, scale research... with vectorbt." Implicit small-to-large basket sizing for dev-stage iteration. |
| 20 | https://en.wikipedia.org/wiki/List_of_S%26P_500_companies | 2026-05-19 | Canonical source | WebFetch full | SP500 table columns: Symbol, Security, GICS Sector, GICS Sub-Industry, Headquarters, Date added, CIK, Founded. Wikipedia is the de-facto canonical scrape source for SP500 + GICS sector. Confirms `screener.py` SP500_URL points to the right table (Sector + Symbol + Sub-Industry rows). |
| 21 | https://arxiv.org/html/2511.18578v1 | 2026-05-19 | Peer-reviewed (arXiv) | WebFetch full | TSFM in finance: 153 JKP factors in 13 clusters incl. Investment / Value / Low Risk / Quality. Decile-based long-short approach (top - bottom decile). CatBoost benchmark Sharpe 6.79, ann ret 46.5%; Chronos-small Sharpe 5.42. "Domain-specific pre-training on financial data proves essential." |
| 22 | https://arxiv.org/html/2505.07078v5 [ADVERSARIAL] | 2026-05-19 | Peer-reviewed (arXiv) | WebFetch full | **ADVERSARIAL to LLM-agent hype**: under fair eval 2004-2024 with delisted stocks + PIT constituents, Buy&Hold Sharpe 0.703 BEATS FinMem (-0.228), FinAgent (0.241), ARIMA (0.542). "Reported advantages vanish under broader and longer evaluations." Critique of TradingAgents-style claims; mitigates survivorship bias by design. Directly contradicts source 24. |
| 23 | https://arxiv.org/html/2508.11152v1 | 2026-05-19 | Peer-reviewed (arXiv) | WebFetch full | AlphaAgents: 3 LLM agents (fundamental / sentiment / valuation) + round-robin debate via Microsoft AutoGen. 15-name tech basket / 4-month Feb-May 2024 backtest. Multi-agent > single-agent in risk-neutral; underperforms in risk-averse bull market. |
| 24 | https://arxiv.org/html/2412.20138v5 | 2026-05-19 | Peer-reviewed (arXiv) | WebFetch full | TradingAgents: 4 analyst roles + Researcher Team (bull/bear debate) + Risk team (3 risk perspectives). 3-month backtest AAPL/GOOGL/AMZN: Sharpe 8.21/6.39/5.60 vs baselines. AUTHORS FLAG "few pullbacks" caveat -- not typical market. Directly contradicted by source 22 [ADVERSARIAL pair]. |
| 25 | https://arxiv.org/html/2603.17692 | 2026-05-19 | Peer-reviewed (arXiv) | WebFetch full | BlindTrade: anonymization-first portfolio opt. Tickers replaced with synthetic codes; sector + technical preserved. Sharpe 1.40 +- 0.22 in 2025 YTD vs SP500 Sharpe 0.64. Critically: PIT SP500 constituents via EODHD prevent survivorship bias by design (matches the rule above). |
| 26 | https://arxiv.org/html/2409.06289v4 | 2026-05-19 | Peer-reviewed (arXiv) | WebFetch full | LLM-driven strategy finding: 100 seed alphas in 9 categories (Momentum, Mean Reversion, Volatility, Fundamental, Liquidity, Quality, Growth, Technical, Macroeconomic) -> filtered + weighted via MLP. top-k=13 / drop-n=5 portfolio. SSE50 Sharpe 0.287, cum-ret 53.17%. |
| 27 | https://arxiv.org/html/2505.15155v2 | 2026-05-19 | Peer-reviewed (arXiv) | WebFetch full | R&D-Agent-Quant: 5-module loop (Spec / Synthesis / Impl / Validation / Analysis). Z-score normalization + IC threshold 0.99 dedupe. CSI 300 IC 0.0532, IR 1.74, ARR 14.21%, MDD -7.42%. 2x ARR with 70% fewer factors. |
| 28 | https://medium.com/@jpolec_72972/survivorship-bias-revealing-the-hidden-truths-of-the-s-p-500-b70da639af9f | 2026-05-19 | Practitioner blog | WebFetch full | SP500 survivorship -- excluding Lehman / Enron / etc. yields "artificially low risk levels" + return overestimation. Mitigation: PIT constituents + bankrupt/merged/delisted included. Methodology in QuantJourney. |

## Pass 2: Adversarial / cross-domain

| # | URL | Accessed | Kind | Fetched | Disagreement noted |
|---|-----|----------|------|---------|---------------------|

## Snippet-only set (context, does NOT count toward gate)

| URL | Kind | Why not fetched |
|-----|------|-----------------|

## Recency scan (last 2 years: 2024-2026)

(Mandatory section -- filled after Pass 2)

## Key findings

(Filled after Pass 2)

## Internal code inventory

| File | Lines | Role | Status |
|------|-------|------|--------|

## Test design for Stage 1

(Filled after code audit)

## Research Gate Checklist

- [ ] >=20 authoritative external sources READ IN FULL via WebFetch
- [ ] 40+ unique URLs total (incl. snippet-only)
- [ ] Recency scan (last 2 years) performed + reported
- [ ] Pass 1 / Pass 2 / Pass 3 structure documented
- [ ] >=1 [ADVERSARIAL] source present
- [ ] Three-variant search visible in brief
- [ ] file:line anchors for every internal claim

```json
{
  "tier": "deep",
  "external_sources_read_in_full": 0,
  "snippet_only_sources": 0,
  "urls_collected": 0,
  "recency_scan_performed": false,
  "internal_files_inspected": 0,
  "adversarial_tags_present": false,
  "gate_passed": false
}
```
