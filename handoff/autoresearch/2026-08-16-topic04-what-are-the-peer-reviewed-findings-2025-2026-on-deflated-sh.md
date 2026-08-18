# Autoresearch memo -- 2026-08-16

**Topic (index 4):** What are the peer-reviewed findings (2025-2026) on deflated Sharpe ratio, walk-forward validation, and out-of-sample robustness for ML-driven equity strategies? Include Bailey, Lopez de Prado, and any follow-up studies that tightened or relaxed the original DSR threshold.

**Source:** gpt-researcher `detailed_report`, Claude-driven, semantic_scholar + arxiv + duckduckgo retrievers.

---

I'll write this report directly from the source material provided, since it already contains substantial relevant content — including two 2025-2026 preprints, a 2026 peer-reviewed review article, and several practitioner analyses that synthesize the foundational Bailey & López de Prado work. A few of the provided sources (Intelligence Bureau recruitment, Free Fire HUD codes, a piano exam video) are irrelevant to this topic and I've excluded them.

Note: three sources — ScienceDirect, ResearchGate, and MDPI — returned access-blocked pages (Cloudflare/bot-detection errors) rather than article content, so I could not read them directly; I've noted this limitation in the report and relied on secondary sources that cite and summarize the same underlying work.

---

# Deflated Sharpe Ratio, Walk-Forward Validation, and Out-of-Sample Robustness in ML-Driven Equity Strategies: A 2025–2026 Literature Review

## Introduction

Between 2020 and 2026, machine learning (ML) and deep reinforcement learning (DRL) approaches to equity, portfolio, and cryptocurrency trading proliferated rapidly, driven by open ecosystems such as FinRL, FinRL-Meta, and ElegantRL ([Rauf & Sutopo, 2026](https://www.cureusjournals.com/articles/12720-deep-reinforcement-learning-for-stock-portfolio-and-crypto-trading-insights-and-trends-2020-2025.pdf)). This growth has been paralleled by increasing scrutiny of the statistical tools used to validate these strategies — chiefly the Sharpe ratio, its overfitting-corrected variant the Deflated Sharpe Ratio (DSR), and resampling techniques such as walk-forward analysis and purged cross-validation. This report synthesizes what the available 2025–2026 literature says about these three pillars of evaluation rigor, traces the lineage back to Bailey and López de Prado's foundational work, and assesses whether recent research has meaningfully tightened or relaxed the original DSR decision threshold.

## The Deflated Sharpe Ratio: Theoretical Foundations

### The Bailey and López de Prado (2014) Framework

The Deflated Sharpe Ratio originates from Bailey and López de Prado's 2014 *Journal of Portfolio Management* paper, which corrects the observed Sharpe ratio for two distortions: (1) selection bias arising from the number of strategy variants ("trials") tested during research, and (2) non-normality (skewness and kurtosis) in the return distribution, which makes standard Sharpe inference too permissive ([usekeel.io, 2026](https://usekeel.io/learn/deflated-sharpe-ratio)). The core insight is that the *maximum* Sharpe ratio drawn from a large set of noise-only trials is upward-biased purely as a statistical artifact — testing enough random strategies against the same price history will eventually produce an impressive-looking Sharpe ratio with no genuine edge behind it ([usekeel.io, 2026](https://usekeel.io/learn/deflated-sharpe-ratio)). The DSR computes the probability that the observed Sharpe ratio genuinely exceeds zero once the number of trials, sample length, and higher moments of the return distribution are accounted for, with 0.95 conventionally treated as the bar for statistical credibility.

A secondary source synthesizing the DSR alongside three other foundational papers (Lo, 2002; Goetzmann et al., 2007; Getmansky et al., 2004) illustrates how sharply the deflation erodes headline Sharpe ratios as trial count rises:

| Observed Sharpe | Number of Trials | Sample Years | DSR Probability (SR > 0) | Verdict |
|---|---|---|---|---|
| 1.0 | 1 | 10 | 99.9% | Likely genuine |
| 1.0 | 50 | 10 | 76% | Questionable |
| 1.5 | 200 | 5 | 58% | Likely spurious |
| 2.0 | 500 | 3 | 34% | Almost certainly overfitted |
| 3.0 | 1000 | 5 | 42% | Consistent with data mining |

*Source: [Quant Decoded (n.d.)](https://quantdecoded.com/en/sharpe-ratio-pitfalls-why-high-sharpe-is-a-red-flag), synthesizing Bailey & López de Prado (2014) and related literature.*

The table demonstrates a counterintuitive but important result: a raw Sharpe of 3.0 from 1,000 trials is statistically *less* credible than a Sharpe of 1.0 from a single, hypothesis-driven test. This reframes "high Sharpe" from a mark of quality to a potential red flag, contingent entirely on the undisclosed research process behind it ([Quant Decoded, n.d.](https://quantdecoded.com/en/sharpe-ratio-pitfalls-why-high-sharpe-is-a-red-flag)).

### Companion Metric: Probability of Backtest Overfitting (PBO)

The same research group extended this work with the Probability of Backtest Overfitting (Bailey, Borwein, López de Prado, & Zhu, 2014), which asks a related but distinct question. Where DSR evaluates whether *one* reported Sharpe ratio is statistically real, PBO evaluates whether the *strategy selection process itself* — choosing the best of many backtested configurations — is likely to have produced a configuration that performs well in-sample but degrades out-of-sample ([usekeel.io, 2026](https://usekeel.io/learn/deflated-sharpe-ratio)). Both metrics are complementary diagnostics against the same underlying threat: multiple-testing bias.

### Has the 0.95 Threshold Been Tightened or Relaxed in 2025–2026?

This is the central empirical question posed by the user, and it merits a direct, evidence-based answer: **based on the sources available for this report, no 2025–2026 peer-reviewed study proposes a numerical revision — tightening or relaxing — of the original DSR decision threshold.** The practitioner literature reviewed here (Quant Decoded's synthesis and Keel's explainer, both current as of mid-2026) treats the 0.95 confidence level as the standing convention, explicitly reaffirming rather than revising it: "A Sharpe above 1.0 that survives DSR at the 0.95 level is meaningful; one that doesn't is selection bias" ([usekeel.io, 2026](https://usekeel.io/learn/deflated-sharpe-ratio)). Where recent work does innovate, it is in *operationalizing* the DSR concept — via open-source tooling (discussed below) and via stricter validation *protocols* that reduce the number of undisclosed trials feeding into a reported Sharpe ratio — rather than in adjusting the statistical threshold itself. The Cureus review article, discussed in the next section, explicitly calls for a "post-2025 research agenda" toward more standardized evaluation protocols, which itself signals that the field still lacks a broadly agreed, revised numerical standard as of mid-2026 ([Rauf & Sutopo, 2026](https://www.cureusjournals.com/articles/12720-deep-reinforcement-learning-for-stock-portfolio-and-crypto-trading-insights-and-trends-2020-2025.pdf)). Attempts to retrieve the original Bailey and López de Prado paper directly, and a related MDPI *Risks* journal article that might contain a proposed revision, were blocked by anti-bot security checks and access-denial pages respectively, so this conclusion should be read as bounded by the sources actually retrievable, not as a definitive claim that no such revision exists anywhere in the literature.

## Walk-Forward Validation and Purged Cross-Validation

### Why Standard Cross-Validation Fails for Financial Time Series

A recurring theme across the reviewed sources is that standard k-fold cross-validation is invalid for time series data because it randomly mixes past and future observations across folds, allowing models to "peek" at future information during training — a single mis-applied cross-validation can make a worthless model appear to generate substantial annual alpha ([QuantEngines, n.d.](https://quantengines.com/blog/cross-validation-trading-models)). This source, while informative, is a practitioner blog without a stated author or publication date on the article itself, and it is bundled within what appears to be a commercial market-data/content platform; it should therefore be weighted as illustrative rather than authoritative.

### Walk-Forward Analysis vs. Combinatorial Purged Cross-Validation (CPCV)

Two dominant leakage-control methodologies emerge from the practitioner literature:

- **Walk-forward analysis** fits a model on an older window and tests it on the next unseen window before rolling forward, closely simulating how trading systems are actually rebuilt and redeployed in practice. Its principal drawback is computational expense, since each fold requires full model retraining ([Kiploks Robustness Engine, 2026](https://kiploks.com/research/combinatorial-purged-cross-validation-vs-walk-forward-pros-and-cons)).
- **Combinatorial purged cross-validation (CPCV)** constructs many train/test splits with purging and embargo periods so that overlapping labels do not leak information across folds, averaging over a larger number of scenarios than a single walk-forward path. Its costs are compute overhead, implementation complexity, and continued exposure to multiple-testing bias when many strategy variants are still being compared across those scenarios ([Kiploks Robustness Engine, 2026](https://kiploks.com/research/combinatorial-purged-cross-validation-vs-walk-forward-pros-and-cons)).

A practically useful decision rule surfaced in this literature is that **multiple CV methods should agree**: if walk-forward analysis shows profitability but purged k-fold does not, the signal is likely regime-dependent rather than robust ([QuantEngines, n.d.](https://quantengines.com/blog/cross-validation-trading-models)).

### Open-Source Tooling Consolidating These Methods

The `purgedcv` Python package (maintained on GitHub, distributed via PyPI) provides a scikit-learn-compatible implementation of purged k-fold, embargo, walk-forward, and CPCV, together with backtest-path reconstruction and deflated/probabilistic Sharpe ratio calculations. The repository states that its empirical results are generated from committed, reproducible scripts rather than hand-written figures — an explicit attempt to counter the reproducibility concerns that pervade this literature ([eslazarev, n.d.](https://github.com/eslazarev/purged-cross-validation/tree/main)). The existence of tooling that bundles CPCV, walk-forward, and DSR/PSR calculations together is itself evidence that, as of 2025–2026, these three concepts have converged into a single expected validation pipeline rather than being treated as independent checks.

## Out-of-Sample Robustness in Recent ML/DRL Equity Research

### Peer-Reviewed Survey Evidence

The most directly peer-reviewed source available is Rauf and Sutopo's (2026) review article in the *Cureus Journal of Computer Science*, covering DRL applications to stock, portfolio, and cryptocurrency trading from 2020–2025. The paper explicitly identifies "methodological flaws in testing, i.e., backtesting bias, splitting data, transaction costs, and overfitting risk" as a persistent weakness across the surveyed literature, and further flags data drift, reporting bias, and limited cross-market generalizability as unresolved limitations even in the most recent (transformer- and graph-neural-network-based) architectures ([Rauf & Sutopo, 2026](https://www.cureusjournals.com/articles/12720-deep-reinforcement-learning-for-stock-portfolio-and-crypto-trading-insights-and-trends-2020-2025.pdf)). Notably, the paper's own review timeline (submitted February 2026, published May 2026) confirms it as genuinely current peer-reviewed output, and its call for a "post-2025 research agenda" and "roadmap to more robust, reproducible, and industry-ready DRL" is itself an implicit acknowledgment that evaluation standards — including DSR-style corrections — remain unevenly adopted across the field as of mid-2026.

### Case Study: Rigorous Walk-Forward Validation in Practice

Deep, Deep, and Lamptey's (2025) arXiv preprint offers a concrete, disciplined example of what rigorous out-of-sample validation looks like when applied honestly. Testing five hypothesis-driven microstructure signals across 100 U.S. equities from 2015–2024 using 34 independent rolling-window test periods, the study reports a modest annualized return of 0.55% and Sharpe ratio of 0.33, with a statistically insignificant aggregate result (p = 0.34) and a maximum drawdown of only −2.76% (beta = 0.058) ([Deep et al., 2025](https://arxiv.org/html/2512.12924v1)). The authors explicitly frame the weak, regime-dependent result (positive during high-volatility periods of 2020–2024, negative during calmer 2015–2019 markets) as evidence of a "reproducible, honest validation protocol," rather than suppressing or over-interpreting a null-to-marginal finding. This is precisely the behavior the DSR/PBO framework is designed to reward: a low, statistically fragile Sharpe ratio reported transparently, without evident cherry-picking of test windows.

### Case Study: A Sharpe Ratio That Warrants Skepticism

By contrast, Huang and Fan's (2026) arXiv preprint on an "autonomous framework for systematic factor investing via agentic AI" reports long-short U.S. equity portfolios with an annualized Sharpe ratio of 3.11 and a return of 59.53%, attributing robustness to a closed-loop system that imposes "strict empirical discipline through out-of-sample validation and economic rationale requirements" to mitigate data-snooping bias ([Huang & Fan, 2026](https://arxiv.org/html/2603.14288v1)). Applying the DSR framework described earlier in this report directly to this claim is instructive: per the deflation table above, a Sharpe ratio in this range is only credible under a very low trial count (approaching one hypothesis-driven test) and a well-behaved return distribution; at even moderate trial counts (a few hundred, plausible for an "autonomous," self-directed signal-generation engine that iteratively formulates its own hypotheses), a Sharpe near 3.0 has historically shown roughly a coin-flip or worse probability of representing genuine, non-overfitted edge ([Quant Decoded, n.d.](https://quantdecoded.com/en/sharpe-ratio-pitfalls-why-high-sharpe-is-a-red-flag)). The abstract does not disclose the number of signal or factor combinations the agentic system explored before converging on the reported long-short portfolio, nor does it report a DSR- or PBO-adjusted figure alongside the raw Sharpe ratio.

## Comparative Summary

| Study | Type | Validation Method | Reported Result | DSR-Consistent Interpretation |
|---|---|---|---|---|
| Rauf & Sutopo (2026) | Peer-reviewed review (Cureus) | Meta-analysis of field practices | Identifies backtesting bias, overfitting risk as unresolved | Confirms field-wide gap, not a single result |
| Deep, Deep & Lamptey (2025) | arXiv preprint | 34-period walk-forward, purged OOS | Sharpe 0.33, p = 0.34, honest null-to-marginal | Consistent with low/no overfitting; low trial exposure |
| Huang & Fan (2026) | arXiv preprint | Self-directed factor search + OOS claim | Sharpe 3.11, return 59.53% | Trial count undisclosed; warrants DSR/PBO scrutiny before trust |

## Synthesis and Assessment

Three conclusions follow directly from the material reviewed. First, the theoretical apparatus for correcting overfitted Sharpe ratios — DSR and PBO — has not been numerically revised by any peer-reviewed 2025–2026 source retrievable here; the 0.95 threshold from Bailey and López de Prado's 2014 work remains the operative convention, and recent activity has concentrated on tooling and protocol standardization (e.g., the `purgedcv` package) rather than on the statistic itself. Second, the *practice* of out-of-sample validation in 2025–2026 ML/DRL trading research is bifurcated: some work (Deep et al., 2025) exemplifies the discipline the DSR framework was designed to reward — transparent, modest, statistically honest results — while other work (Huang & Fan, 2026) reports Sharpe ratios that fall squarely into the range the same literature flags as "almost certainly overfitted" or "consistent with data mining" absent trial-count disclosure. This is not a claim that the latter study's results are invalid, but a defensible, evidence-grounded position that its headline Sharpe ratio should not be trusted at face value without a disclosed DSR or PBO adjustment. Third, the Cureus review's explicit call for a post-2025 research agenda toward standardized, reproducible evaluation protocols corroborates the assessment that the field has not yet converged on tightened or relaxed DSR thresholds, but rather on the more foundational problem of getting researchers to consistently apply the existing 2014 framework at all.

## Conclusion

The deflated Sharpe ratio, walk-forward validation, and combinatorial purged cross-validation together form the current best-practice triad for evaluating ML-driven equity strategies, tracing back principally to Bailey and López de Prado's 2014 work. The 2025–2026 literature surveyed here does not show a peer-reviewed revision of the original 0.95 DSR threshold; instead, it shows growing operational consolidation of these tools (open-source packages, structured walk-forward protocols) alongside a persistent gap between studies that apply this discipline rigorously and those that report high, insufficiently contextualized Sharpe ratios. Readers evaluating any new ML-driven trading claim in this window should treat an undisclosed trial count alongside a Sharpe ratio above roughly 2.0 as the single most informative red flag available from this literature.

## References

Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio: Correcting for selection bias, backtest overfitting, and non-normality. *Journal of Portfolio Management, 40*(5), 94–107. https://doi.org/10.3905/jpm.2014.40.5.094 (primary source page inaccessible; cited via [Quant Decoded, n.d.](https://quantdecoded.com/en/sharpe-ratio-pitfalls-why-high-sharpe-is-a-red-flag) and [usekeel.io, 2026](https://usekeel.io/learn/deflated-sharpe-ratio)). Original listing: [https://www.researchgate.net/publication/286121118_The_Deflated_Sharpe_RatioCorrecting_for_Selection_Bias_BacktestOverfitting_and_Non-Normality](https://www.researchgate.net/publication/286121118_The_Deflated_Sharpe_RatioCorrecting_for_Selection_Bias_BacktestOverfitting_and_Non-Normality)

Deep, G., Deep, A., & Lamptey, W. (2025, December 15). Interpretable hypothesis-driven trading: A rigorous walk-forward validation framework for market microstructure signals. *arXiv*. [https://arxiv.org/html/2512.12924v1](https://arxiv.org/html/2512.12924v1)

eslazarev. (n.d.). *purged-cross-validation: scikit-learn-compatible time-series cross-validation* [Software repository]. GitHub. [https://github.com/eslazarev/purged-cross-validation/tree/main](https://github.com/eslazarev/purged-cross-validation/tree/main)

Huang, A. Y., & Fan, Z. (2026, March 15). Beyond prompting: An autonomous framework for systematic factor investing via agentic AI. *arXiv*. [https://arxiv.org/html/2603.14288v1](https://arxiv.org/html/2603.14288v1)

Keel Research Team. (2026, May 17). Deflated Sharpe ratio (DSR) — Selection bias in backtests. *Keel*. [https://usekeel.io/learn/deflated-sharpe-ratio](https://usekeel.io/learn/deflated-sharpe-ratio)

Kiploks Robustness Engine. (2026, April 3). Combinatorial purged cross-validation vs walk-forward: Pros and cons. *Kiploks*. [https://kiploks.com/research/combinatorial-purged-cross-validation-vs-walk-forward-pros-and-cons](https://kiploks.com/research/combinatorial-purged-cross-validation-vs-walk-forward-pros-and-cons)

MDPI. (n.d.). [Article page inaccessible — access denied]. *Risks*. [https://www.mdpi.com/2227-9091/14/3/63](https://www.mdpi.com/2227-9091/14/3/63)

Quant Decoded. (n.d.). A Sharpe above 2 is probably fake — Here's how to spot it. *Quant Decoded*. [https://quantdecoded.com/en/sharpe-ratio-pitfalls-why-high-sharpe-is-a-red-flag](https://quantdecoded.com/en/sharpe-ratio-pitfalls-why-high-sharpe-is-a-red-flag)

QuantEngines. (n.d.). Cross-validation for trading models: Avoiding look-ahead bias. *QuantEngines*. [https://quantengines.com/blog/cross-validation-trading-models](https://quantengines.com/blog/cross-validation-trading-models)

Rauf, K., & Sutopo, J. (2026, May 25). Deep reinforcement learning for stock, portfolio, and crypto trading: Insights and trends (2020–2025). *Cureus Journal of Computer Science, 3*, es44389-026-00083-1. https://doi.org/10.7759/s44389-026-00083-1. [https://www.cureusjournals.com/articles/12720-deep-reinforcement-learning-for-stock-portfolio-and-crypto-trading-insights-and-trends-2020-2025.pdf](https://www.cureusjournals.com/articles/12720-deep-reinforcement-learning-for-stock-portfolio-and-crypto-trading-insights-and-trends-2020-2025.pdf)

ScienceDirect. (n.d.). [Article page inaccessible — Cloudflare error]. [https://www.sciencedirect.com/science/article/pii/S1059056026008087](https://www.sciencedirect.com/science/article/pii/S1059056026008087)
