# Phase 5.5 Prioritized Shopping List

Top 3 must-have data sources, sorted by severity desc and cost asc. Every citation is verified present in handoff/phase-5.5-research.md.

## must-have: news_and_sentiment

- alpha_tier: S
- severity: high
- vendor: finbert_via_huggingface
- fallback: claude_haiku_chat_summarise
- cost_usd_month: 0
- effort_days: 5
- license_note: FinBERT Apache-2.0; Haiku metered (Peder approval for scale)
- rationale: Only AV Social Sentiment feeds NLP today; no RavenPack / Bloomberg / FinGPT-style transformer pipeline. NLP sentiment via Vertex is embeddings-only, not a dedicated news sentiment model.
- citations (from phase-5.5-research.md):
- https://arxiv.org/abs/1908.10063
- https://arxiv.org/abs/2306.06031
- https://www.ravenpack.com/research/sentiment-driven-stock-selection

## must-have: ai_frontier_timeseries_foundation_models

- alpha_tier: S
- severity: high
- vendor: chronos_2_via_amazon_sagemaker
- fallback: moirai_moe_via_huggingface_transformers
- cost_usd_month: 50
- effort_days: 10
- license_note: Chronos-2 Apache-2.0; Moirai-MoE Apache-2.0; SageMaker metered (Peder approval)
- rationale: No time-series foundation models in use. FinGPT / TimesFM / Moirai / Chronos-2 all landed 2024-2026 and show zero-shot forecasting superior to classical ARIMA/GBM on many horizons. Phase-8 proposal on masterplan is the right home.
- citations (from phase-5.5-research.md):
- https://arxiv.org/html/2403.07815v1
- https://arxiv.org/html/2511.11698v1
- https://arxiv.org/html/2510.15821v1
- https://github.com/amazon-science/chronos-forecasting

## must-have: institutional_filings_13F_13D_form4

- alpha_tier: A
- severity: medium
- vendor: sec_edgar_expanded
- fallback: quiverquant_free_tier
- cost_usd_month: 0
- effort_days: 3
- license_note: SEC EDGAR public domain; QuiverQuant free tier for derived congressional trades
- rationale: sec_edgar covers Form 4 insider trades only. No 13F/13D (hedge-fund holdings), no QuiverQuant congressional trades, no WhaleWisdom-style aggregation. Missing ~80% of the institutional-filing alpha surface.
- citations (from phase-5.5-research.md):
- https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets
- https://edgartools.readthedocs.io/
- https://whalewisdom.com/
