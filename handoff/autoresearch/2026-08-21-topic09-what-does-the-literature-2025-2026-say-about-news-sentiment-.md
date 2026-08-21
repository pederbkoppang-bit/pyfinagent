# Autoresearch memo -- 2026-08-21

**Topic (index 9):** What does the literature (2025-2026) say about news sentiment alpha decay, half-life, and overcrowding when many desks run the same LLM-based sentiment pipeline?

**Source:** gpt-researcher `detailed_report`, Claude-driven, semantic_scholar + arxiv + duckduckgo retrievers.

---

# News Sentiment Alpha Decay, Half-Life, and Overcrowding in LLM-Based Trading Pipelines: A Literature Synthesis (2025–2026)

## Introduction

The rapid adoption of large language models (LLMs) for extracting sentiment signals from financial news has created a paradox that the 2025–2026 literature is only now beginning to formalize: the very success of LLM-based sentiment pipelines in generating short-term predictive power appears to be accelerating the erosion of that power. As desks across the industry converge on similar model architectures (DeepSeek, GPT-4, FinBERT, RoBERTa, DeBERTa) and similar data sources (newswires, earnings calls, social media), the signals these models extract increasingly resemble each other, and the resulting trades increasingly crowd the same side of the market. This report synthesizes the available 2025–2026 literature on three interlocking phenomena — alpha decay, signal half-life, and overcrowding — as they apply specifically to LLM-driven sentiment strategies, and offers a concrete assessment of where the evidence currently points.

## The Theoretical Backbone: Modeling Alpha Half-Life Under AI Adoption

The most directly relevant theoretical contribution is Meng and Chen's (2026) model of "AI-driven alpha decay," which formalizes why AI-driven investment strategies are, in the authors' words, "inherently self-defeating at scale" ([Meng & Chen, 2026](https://arxiv.org/html/2605.23905)). The paper identifies three mutually reinforcing channels that compress excess returns as AI adoption rises: **signal crowding** (multiple desks acting on the same extracted signal simultaneously), **performative signal erosion** (the act of trading on a signal changes the market conditions the signal was measuring), and **Red Queen competition** (firms must continuously upgrade models just to preserve a shrinking edge, analogous to evolutionary arms races) ([Meng & Chen, 2026](https://arxiv.org/html/2605.23905)).

### The Half-Life Formula and Current Estimates

The authors derive an explicit alpha half-life function:

$$h(\varphi) = \frac{\ln 2}{\theta + \delta(\varphi)}$$

where θ is the natural mean-reversion rate of a signal absent AI competition, and δ(φ) = Nφρa/λ(φ) is the AI-accelerated decay component — a function of the number of competing AI adopters (N), the adoption rate (φ), and signal correlation across adopters (ρ). Critically, δ(φ) is *convex-decreasing* in adoption, meaning decay accelerates disproportionately as more desks pile into overlapping strategies ([Meng & Chen, 2026](https://arxiv.org/html/2605.23905)).

Plugging in current market conditions — adoption φ ≈ 0.7 and cross-desk signal correlation ρ ≈ 0.6 — the model implies signal half-lives of approximately **18 months**, compared with **5–7 years** in the pre-AI era ([Meng & Chen, 2026](https://arxiv.org/html/2605.23905)). This is a three-to-four-fold compression in the useful lifespan of a given sentiment signal, and it is the single most concrete quantitative estimate available in the current literature for how fast LLM-era sentiment alpha decays.

### Four Theoretical Results: Cascades, Red Queen Dynamics, and Fragility

Beyond the half-life formula, Meng and Chen (2026) establish four results with direct relevance to overcrowding:

| Result | Description | Implication for sentiment desks |
|---|---|---|
| Alpha half-life theorem | Signal lifespans are convex-decreasing in AI adoption | Refresh cycles must shorten as more competitors adopt similar LLM pipelines |
| Signal extinction cascade | Beyond a critical adoption threshold φ*, decay of one signal class triggers accelerated competition for remaining signals | Crowding in one sentiment niche (e.g., earnings-call tone) pushes capital into adjacent niches (e.g., social sentiment), compressing those too |
| Red Queen impossibility | In the "monoculture equilibrium" (most/all desks using similar LLMs), net alpha is identically zero despite heavy AI investment | Homogeneous LLM adoption across the industry is self-neutralizing, not self-reinforcing |
| Fragility–efficiency tradeoff | The adoption level that maximizes price discovery strictly exceeds the level that minimizes systemic fragility | There is no adoption level that is simultaneously optimal for market efficiency and for stability |

Source: [Meng & Chen (2026)](https://arxiv.org/html/2605.23905)

The empirical component of the paper is notable for grounding these theoretical claims in real institutional data: the authors calibrate portfolio convergence against SEC Form 13F filing patterns spanning **99.5 million holdings from 2013 to 2024**, finding that simulated institutional portfolio convergence increased by **42% over the sample period** ([Meng & Chen, 2026](https://arxiv.org/html/2605.23905)). They further simulate declining cross-sectional return dispersion among AI-adopting hedge funds and use a simulation of the 2010 Flash Crash to illustrate how homogenized, AI-driven positioning can amplify systemic fragility ([Meng & Chen, 2026](https://arxiv.org/html/2605.23905)).

## Empirical Signals of Crowding: Evidence from Live Markets

While Meng and Chen (2026) provide the theoretical apparatus, the earlier and highly-cited empirical study by Lopez-Lira and Tang (2025) offers direct market evidence consistent with the crowding thesis, even though it predates the formal half-life model. Using GPT-4 to score news headlines for Dow-adjacent stocks, the authors found the model captured approximately **90% portfolio-day hit rates** for the non-tradable initial market reaction, with GPT-4 scores also predicting subsequent price drift, particularly for small stocks and negative news ([Lopez-Lira & Tang, 2025](https://arxiv.org/html/2304.07619)). Crucially, the authors report that **"strategy returns decline as LLM adoption rises, consistent with improved price efficiency"** ([Lopez-Lira & Tang, 2025](https://arxiv.org/html/2304.07619)). They formalize this in a theoretical model incorporating LLM technology, information-processing capacity constraints, underreaction, and limits to arbitrage — a smaller-scale precursor to the crowding/decay dynamics that Meng and Chen (2026) later generalize into a full half-life framework.

## Is There Still Alpha Left? Evidence from Recent Backtests

Despite the decay narrative, several 2025–2026 studies report that LLM/NLP-derived sentiment signals still generate *positive* alpha when backtested, which is important context: decay does not mean the signal is exhausted, only that its useful window is shrinking and must be continuously refreshed.

Linhares Pontes et al. (2025), working from Trading Central Labs, backtested three sentiment models (two classification, one regression) against Dow Jones 30 news over a 28-month period. All three models produced positive returns versus a Buy&Hold benchmark, with the regression model achieving the highest return of **50.63%**, outperforming Buy&Hold ([Linhares Pontes et al., 2025](https://aclanthology.org/2025.jeptalnrecital-industrielle.2.pdf)). This is one of the more concrete recent demonstrations that sentiment-based alpha remained *extractable* as of the study period, though the authors do not address decay or crowding directly.

Similarly, Siala, Khanfir, and Papadakis (2026) compared DeBERTa, RoBERTa, and FinBERT for LLM-based news sentiment stock movement prediction, finding DeBERTa alone reached **75% accuracy**, while an ensemble of all three models pushed accuracy to approximately **80%** ([Siala et al., 2026](https://arxiv.org/abs/2602.00086)). This suggests that model diversification/ensembling — rather than reliance on a single dominant architecture — may partially offset the homogenization risk that drives crowding in Meng and Chen's framework, since ensembles reduce correlation with any single "monoculture" signal.

Chiang, Lee, and Tsai (2026) apply DeepSeek-derived sentiment indices to the Shanghai Composite Index using a prospect-theory framework, analyzing 29,077 news items from January 2024 to January 2026. They find the total subjective value variable has a positive, significant effect on index returns, while a weighted sentiment function has a negative significant effect, with the authors' Lasso-regularized model still subject to heteroscedasticity, multicollinearity, and non-normal residuals ([Chiang et al., 2026](https://www.tandfonline.com/doi/full/10.1080/13504851.2026.2662546)). This paper is notable for demonstrating that LLM-sentiment effects are not confined to US equities but extend to the A-share market, though the reported statistical irregularities are a caution against over-reading the precision of these effect sizes.

## Practitioner Perspective: Productivity Infrastructure, Not Alpha Engines

Industry commentary broadly corroborates the academic decay narrative, though from a more skeptical starting position. Zerve's 2026 review argues plainly that "LLMs do not generate alpha... The funds that treat LLMs as research productivity infrastructure have benefited. The funds that treated LLMs as alpha-generation infrastructure have not" ([Hayes, 2026](https://www.zerve.ai/blog/llms-in-quantitative-research)). This framing is consistent with the Red Queen impossibility result: if most competing funds are using similar LLM tooling for signal generation, the marginal fund gains little durable edge from the technology itself, only from proprietary data, execution, or non-LLM-crowded angles.

Quantt's (2026) industry survey similarly notes that "news sentiment analysis at scale" is one of the areas AI most directly touches in alpha generation, but frames the current state as "uneven," with some applications transformative and others "oversold and underwhelmed" ([Quantt, 2026](https://www.quantt.co.uk/resources/ai-revolution-in-quant-trading-2026)). Notably, firms with distinctive technology cultures (Jane Street's OCaml-based, functional-programming-first stack; Renaissance Technologies' historically secretive, pre-LLM statistical ML approach) are described as adopting LLM tooling more selectively — arguably a structural hedge against the very homogenization that Meng and Chen (2026) identify as the driver of decay ([Quantt, 2026](https://www.quantt.co.uk/resources/ai-revolution-in-quant-trading-2026)).

## Methodological Caveats Limiting the Literature's Conclusions

Zhang and Zhang's (2026) hedge-fund-oriented review of LLMs for stock forecasting is useful precisely because it catalogs the practical pitfalls "often understated in the literature," including fragility in sentiment analysis itself, dataset and horizon design choices, inconsistent performance evaluation metrics, data leakage, illiquidity premia, and fundamental limits to stock price predictability ([Zhang & Zhang, 2026](https://arxiv.org/html/2605.05211v1)). This matters for interpreting the positive-alpha backtests above: studies like Linhares Pontes et al. (2025) and Siala et al. (2026) report strong in-sample or backtested performance, but Zhang and Zhang's (2026) catalogue of pitfalls implies that some portion of these reported returns may not survive live deployment once crowding, data leakage, or illiquidity effects are accounted for. Cao et al.'s (2025) broader survey of AI in quantitative investment situates LLM-based sentiment work within the longer arc from hand-crafted alpha factors to deep learning to LLM-driven pipelines, describing this as a "potential paradigm shift" while stopping short of quantifying decay ([Cao et al., 2025](https://arxiv.org/html/2503.21422v1)).

## Synthesis and Assessment

Weighing the theoretical model against the empirical backtests, the evidence supports a specific, non-hedged conclusion: **news-sentiment alpha extracted via LLM pipelines is real but has a rapidly shrinking half-life, and the primary driver of that shrinkage is architectural homogeneity across competing desks rather than any inherent limit of the sentiment signal itself.** The ~18-month half-life estimate from Meng and Chen (2026), triangulated against Lopez-Lira and Tang's (2025) direct observation that strategy returns fall as LLM adoption rises, and against the 42% rise in simulated 13F portfolio convergence, paints a consistent picture: this is not a hypothetical risk but a measurable, ongoing trend as of 2026.

At the same time, the positive-alpha backtests (Linhares Pontes et al., 2025; Siala et al., 2026; Chiang et al., 2026) show that the decay process has not yet reached the "Red Queen impossibility" endpoint where net alpha is driven to zero — there is still exploitable signal, particularly in less-crowded markets (e.g., the A-share market) or with non-standard architectures (ensembles rather than a single dominant model). The practical implication for a desk evaluating an LLM sentiment pipeline is therefore not "avoid it," but "treat it as a decaying asset requiring continuous model differentiation." Ensembling distinct architectures (as in Siala et al., 2026) and targeting less crowded markets or data sources appear to be the two concrete, evidence-backed levers for extending a signal's effective half-life beyond the roughly 18-month baseline implied by current adoption levels. Firms that instead standardize on a single popular LLM sentiment stack are, per the Red Queen and monoculture-equilibrium results, effectively racing to a zero-net-alpha equilibrium even as they increase AI spending — a dynamic the practitioner literature (Hayes, 2026; Quantt, 2026) independently corroborates through its "productivity infrastructure, not alpha engine" framing.

## References

Cao, B., Wang, S., Lin, X., Wu, X., Zhang, H., Ni, L. M., & Guo, J. (2025). *From deep learning to LLMs: A survey of AI in quantitative investment*. arXiv. [https://arxiv.org/html/2503.21422v1](https://arxiv.org/html/2503.21422v1)

Chiang, T.-M., Lee, M.-C., & Tsai, Z.-R. (2026). The influence mechanisms of news sentiment indicators from AI's large language models on stock market index returns: A prospect theory lens. *Applied Economics Letters*. [https://www.tandfonline.com/doi/full/10.1080/13504851.2026.2662546](https://www.tandfonline.com/doi/full/10.1080/13504851.2026.2662546)

Hayes, P. (2026). *LLMs in quant research: What actually works in 2026*. Zerve. [https://www.zerve.ai/blog/llms-in-quantitative-research](https://www.zerve.ai/blog/llms-in-quantitative-research)

Linhares Pontes, E., González-Gallardo, C.-E., Bordea, G., Moreno, J. G., Ben Jannet, M., Zhao, Y., & Doucet, A. (2025). Backtesting sentiment signals for trading: Evaluating the viability of alpha generation from sentiment analysis. *Actes de CORIA-TALN-RJCRI-RECITAL 2025, Session Industrielle*, 17–32. [https://aclanthology.org/2025.jeptalnrecital-industrielle.2.pdf](https://aclanthology.org/2025.jeptalnrecital-industrielle.2.pdf)

Lopez-Lira, A., & Tang, Y. (2025). *Can ChatGPT forecast stock price movements? Return predictability and large language models*. arXiv. [https://arxiv.org/html/2304.07619](https://arxiv.org/html/2304.07619)

Meng, S., & Chen, X. (2026). *AI-driven alpha decay: Algorithmic homogenization, reflexive signal erosion, and the paradox of intelligent markets*. arXiv. [https://arxiv.org/html/2605.23905](https://arxiv.org/html/2605.23905)

Quantt. (2026). *The AI revolution in quant trading 2026: How LLMs and foundation models are reshaping finance*. [https://www.quantt.co.uk/resources/ai-revolution-in-quant-trading-2026](https://www.quantt.co.uk/resources/ai-revolution-in-quant-trading-2026)

Siala, W., Khanfir, A., & Papadakis, M. (2026). *Impact of LLMs news sentiment analysis on stock price movement prediction*. arXiv. [https://arxiv.org/abs/2602.00086](https://arxiv.org/abs/2602.00086)

Zhang, O., & Zhang, Z. (2026). *A review of large language models for stock price forecasting from a hedge-fund perspective*. arXiv. [https://arxiv.org/html/2605.05211v1](https://arxiv.org/html/2605.05211v1)
