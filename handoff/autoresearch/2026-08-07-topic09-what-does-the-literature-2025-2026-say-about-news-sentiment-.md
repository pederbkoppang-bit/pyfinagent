# Autoresearch memo -- 2026-08-07

**Topic (index 9):** What does the literature (2025-2026) say about news sentiment alpha decay, half-life, and overcrowding when many desks run the same LLM-based sentiment pipeline?

**Source:** gpt-researcher `detailed_report`, Claude-driven, semantic_scholar + arxiv + duckduckgo retrievers.

---

# News Sentiment Alpha Decay, Half-Life, and Crowding Risk in LLM-Based Trading Pipelines: A Review of the 2025–2026 Literature

## Introduction

The rapid adoption of large language models (LLMs) for financial news sentiment extraction has raised an obvious question for quantitative desks: once every fund runs a similar LLM pipeline over the same newswire feed, how quickly does the resulting alpha erode, and does the signal survive at all under realistic trading frictions? This report synthesizes what the available 2025–2026 literature says about that question, with a primary focus on a directly relevant peer-reviewed study — Kirtac's (2026) evaluation of LLM-based news sentiment under liquidity and market-friction constraints, published at the ACL EvalEval workshop — supplemented by context from the Reuters Institute's 2026 Digital News Report on how AI is reshaping the broader news-information ecosystem. It should be stated plainly at the outset: several sources retrieved during research (ResearchGate, ScienceDirect ×2, and MDPI) returned access-blocked or bot-verification pages rather than readable content, so their substantive claims cannot be reported here. This limits the breadth of literature that can be directly cited, and that limitation is treated as a finding in itself rather than glossed over.

## What the Evidence Actually Shows: The Kirtac (2026) Study

The most substantive and directly relevant piece of evidence available is Kirtac's (2026) paper, "Evaluating Large Language Model News Sentiment in Finance under Liquidity and Market Frictions," presented at the Workshop on Evaluating Evaluations (EvalEval) at ACL 2026 in San Diego ([Kirtac, 2026](https://aclanthology.org/2026.evaleval-1.4/)). The study's central contribution is methodological: rather than reporting only offline classification accuracy — the standard in most financial NLP papers — it evaluates sentiment signals through three lenses: classification performance, return predictability, and *implementable* portfolio performance net of trading costs ([Kirtac, 2026](https://aclanthology.org/2026.evaleval-1.4/)).

### Sample and Methodology

The dataset is large and recent, which matters for a topic centered on decay dynamics. It links Refinitiv News Analytics to CRSP return data, beginning with 3,129,924 U.S. news items published between January 1, 2010, and January 30, 2026 ([Kirtac, 2026](https://aclanthology.org/2026.evaleval-1.4/)). Filtering — restricting to single-firm stories, removing redundant coverage via a five-day cosine-similarity novelty screen, and keeping only tradable stocks with positive bid/ask quotes, minimum volume thresholds, spreads under 20%, and available Amihud illiquidity and Kyle's lambda estimates — reduces this to 973,481 tradable news items linked to 3,452 firms ([Kirtac, 2026](https://aclanthology.org/2026.evaleval-1.4/)). The novelty screen is itself notable for the crowding question: by explicitly discarding repeated coverage of the same event, the design implicitly acknowledges that redundant, widely re-reported news is a primary channel through which many independent desks would end up trading the identical signal at the identical moment — a structural precondition for crowding.

### Performance Comparison Across Sentiment Methods

The paper benchmarks six approaches: LLaMA-3, OPT, RoBERTa, BERT, FinBERT, and the classic Loughran–McDonald financial dictionary, evaluated via daily-rebalanced long-short portfolios with a 5-basis-point trading cost assumption ([Kirtac, 2026](https://aclanthology.org/2026.evaleval-1.4/)).

| Method | Classification accuracy | Cumulative return (June 2024–Jan 2026) | Signal type |
|---|---|---|---|
| LLaMA-3 | 78.2% (highest) | ~180% | High-capacity LLM |
| OPT | Lower than LLaMA-3 | ~155% | High-capacity LLM |
| RoBERTa | Lower than LLaMA-3 | ~120% | Transformer (non-generative) |
| BERT | Not separately reported in abstract | Not separately reported | Transformer (non-generative) |
| FinBERT | Not separately reported in abstract | Not separately reported | Domain-tuned transformer |
| Loughran–McDonald dictionary | Weakest | **−9%** (loss) | Lexicon-based |

*Source: [Kirtac (2026)](https://aclanthology.org/2026.evaleval-1.4/).*

Two findings bear directly on the alpha-decay and crowding question. First, LLaMA-3 "produces the largest predictive coefficients in panel regressions" and its long-short strategy still earned roughly 180% cumulative return through January 2026 — the most recent month in the sample — even after a 5-bps cost haircut ([Kirtac, 2026](https://aclanthology.org/2026.evaleval-1.4/)). That the signal remains strongly profitable at the very edge of the sample window is evidence against the hypothesis that LLM-sentiment alpha has already been arbitraged to zero as of early 2026, at least for the highest-capability model tested. Second, and in the opposite direction, the dictionary-based method — the oldest, cheapest, and most widely replicated approach in the industry — now *loses* money (−9%) once realistic frictions are applied ([Kirtac, 2026](https://aclanthology.org/2026.evaleval-1.4/)). This is the clearest empirical signature of decay/crowding available in the reviewed literature: the commoditized signal has gone negative, while the frontier signal has not.

## Interpreting This as Alpha Decay and Half-Life

None of the accessible literature reports an explicit statistical half-life (e.g., an exponential decay-rate parameter) for LLM-derived sentiment alpha. This is a genuine gap: Kirtac (2026) reports cumulative and panel-regression results but not a decay curve or autocorrelation-based half-life estimate. Any half-life figure attributed to LLM sentiment alpha in this report would therefore be fabricated, and none is offered. What the data does support is a *relative* decay ordering inferable from the cross-method comparison: the Loughran–McDonald dictionary — in use across the industry for over a decade and requiring no proprietary infrastructure to replicate — has decayed past the point of profitability, consistent with the standard finance-literature expectation that publicly known, cheaply replicable signals get arbitraged away as more capital copies them. The fact that RoBERTa, OPT, and LLaMA-3 rank in ascending order of both accuracy and return, with LLaMA-3 on top, suggests decay tracks *replication cost and model sophistication* rather than time elapsed since publication alone: harder-to-replicate signals decay more slowly because fewer desks can economically stand them up at scale (compute cost, licensing, engineering effort, prompt/fine-tuning expertise).

## Overcrowding When Many Desks Run the Same Pipeline

This is the part of the question the accessible literature does not test directly — Kirtac (2026) evaluates six *different* model architectures against each other, not the effect of many independent funds simultaneously deploying the *same* model on the *same* news feed. No capacity, crowding-elasticity, or multi-participant simulation is reported in the available abstract and sections. That said, three pieces of evidence in the study bear on the question by implication, and it is reasonable to draw an inference from them while being explicit that it is inference rather than a direct finding:

1. **The novelty filter as an implicit crowding proxy.** By removing near-duplicate coverage within a five-day window before computing returns, the study effectively excludes the very scenario — many outlets and, by extension, many algorithmic readers, reacting to the same restated news — where crowding would be most acute. Real-world desks running LLM sentiment on raw, unfiltered wires would face more of this redundant-signal exposure than the paper's cleaned sample implies, meaning the reported 180% return for LLaMA-3 likely represents a *ceiling*, not a realistic estimate of what a crowded, unfiltered deployment would achieve.

2. **Model homogeneity compresses the very differentiation that generates alpha.** Because LLaMA-3 is an open-weight model, its outputs on identical news text are largely deterministic and near-identical across users. If, as this literature implies, the current alpha premium of LLM sentiment over lexicon sentiment stems from replication difficulty and inference cost, that moat narrows mechanically as the open-weight model itself becomes the industry-standard tool — the same commoditization dynamic that already sank the Loughran–McDonald dictionary. In other words, the paper's own results supply the mechanism for future overcrowding even though it does not simulate it directly.

3. **The trading-cost assumption is a lower bound on the true friction from crowding.** The 5-bps cost figure captures typical transaction costs but not the price-impact amplification that occurs when many participants trade the same direction on the same signal within the same short window — a well-established mechanism in the crowded-trade literature more broadly, though not modeled in this specific paper.

## The Broader Information Environment as a Contextual Factor

The Reuters Institute's 2026 Digital News Report adds relevant macro-context, though it does not address trading strategies. It finds that trust in AI chatbots as a news source stands at just 20% globally, and that Google organic search referral traffic to publishers fell 33% globally (38% in the U.S.) between November 2024 and November 2025, with publishers expecting a further ~43% decline over three years ([Reuters Institute, 2026](https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2026/dnr-executive-summary)). For the sentiment-alpha question, this matters indirectly: as AI-mediated summarization and chatbot consumption increasingly stand between raw news and human readers, the "raw wire text" that sentiment pipelines are trained and run on may itself increasingly be competing with, or converging toward, AI-generated summaries and framing — a second-order homogenization pressure on top of model homogenization. This is speculative extrapolation on my part, not a claim made in the Reuters Institute report itself, which is a media-consumption study, not a market-microstructure one.

## My Assessment

Based strictly on the evidence available, I conclude the following: (1) the reviewed literature does not yet document a quantified half-life for LLM-based news sentiment alpha, and any such figure circulating informally on trading desks should be treated as proprietary or unverified rather than peer-reviewed; (2) the clearest hard evidence of decay/crowding in the accessible literature is the negative net return of the decade-old, publicly replicable Loughran–McDonald dictionary approach, which functions as a natural experiment in what happens once a sentiment signal becomes common knowledge; (3) frontier LLM-based sentiment (LLaMA-3 in this study) still shows strong, statistically robust predictive value through January 2026, but the study's own methodological choices (novelty filtering, single-model-per-backtest design) mean its reported returns likely overstate what is achievable once many desks run identical open-weight pipelines against the same live, duplicated wire; and (4) the mechanism most likely to compress LLM-sentiment alpha going forward is not primarily statistical staleness of the underlying relationship between sentiment and returns, but the falling cost of replication as open-weight LLMs become standard infrastructure — the same commoditization pathway that already eliminated profitability for dictionary-based methods. Firms seeking to preserve edge should treat model choice, proprietary fine-tuning, and de-duplication/filtering logic (rather than the base sentiment-extraction task itself) as the more durable sources of differentiation.

## Limitations of This Review

Several sources identified during research — a ResearchGate review titled "Large Language Models and Sentiment Analysis in Financial Markets: A Review, Datasets and Case Study," and two ScienceDirect articles — returned bot-detection or access-restriction pages, and an MDPI article returned an access-denied error, preventing verification of their content. These sources are listed below for transparency but are not cited for any factual claim in this report, since their actual content could not be confirmed.

## References

Kirtac, K. (2026, July). Evaluating large language model news sentiment in finance under liquidity and market frictions. In M. Akhtar, J. Batzner, L. Choshen, A. Ghosh, U. Gohar, J. Mickel, I. Pant, Z. Talat, & M. Lin (Eds.), *Proceedings of the Workshop on Evaluating Evaluations (EvalEval)* (pp. 26–35). Association for Computational Linguistics. [https://aclanthology.org/2026.evaleval-1.4/](https://aclanthology.org/2026.evaleval-1.4/)

Egan, J. (2026, June 16). Overview and key findings of the 2026 Digital News Report. *Reuters Institute for the Study of Journalism*. [https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2026/dnr-executive-summary](https://reutersinstitute.politics.ox.ac.uk/digital-news-report/2026/dnr-executive-summary)

### Sources identified but inaccessible (content not verified, not cited for factual claims)

Large language models and sentiment analysis in financial markets: A review, datasets and case study. *ResearchGate*. [https://www.researchgate.net/publication/383248674_Large_Language_Models_and_Sentiment_Analysis_in_Financial_Markets_A_Review_Datasets_and_Case_Study](https://www.researchgate.net/publication/383248674_Large_Language_Models_and_Sentiment_Analysis_in_Financial_Markets_A_Review_Datasets_and_Case_Study)

ScienceDirect. *ScienceDirect*. [https://www.sciencedirect.com/science/article/pii/S1544612324002575](https://www.sciencedirect.com/science/article/pii/S1544612324002575)

ScienceDirect. *ScienceDirect*. [https://www.sciencedirect.com/science/article/pii/S0020025526006420](https://www.sciencedirect.com/science/article/pii/S0020025526006420)

Access denied. *MDPI*. [https://www.mdpi.com/2673-2688/7/4/138](https://www.mdpi.com/2673-2688/7/4/138)
