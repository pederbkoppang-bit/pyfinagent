# Autoresearch memo -- 2026-08-14

**Topic (index 2):** What are the state-of-the-art methods (2025-2026) for using 10-K and 10-Q filings as LLM inputs for equity signal generation? Look for academic benchmarks, replication studies, and any deployed production systems.

**Source:** gpt-researcher `detailed_report`, Claude-driven, semantic_scholar + arxiv + duckduckgo retrievers.

---

Jeg setter nå sammen den fullstendige rapporten på engelsk, slik oppgaven eksplisitt krever, basert på kildematerialet som er oppgitt.

# State-of-the-Art Methods for Using 10-K and 10-Q Filings as LLM Inputs for Equity Signal Generation (2025–2026)

## Introduction

The use of Large Language Models (LLMs) to convert SEC filings — primarily Forms 10-K (annual) and 10-Q (quarterly) — into equity research signals has matured rapidly between 2025 and 2026, moving from proof-of-concept retrieval systems toward standardized benchmarks, replication studies on model reliability, and named production deployments at hedge funds. Three distinct layers of state-of-the-art work can be identified in the available literature: (1) academic benchmarking of LLM comprehension and reasoning over filings, most notably FinanceBench and its agentic extensions; (2) a parallel and increasingly urgent body of replication and audit research on *look-ahead bias*, which threatens the validity of any backtested "signal" derived from filings; and (3) documented production architectures — document-parsing pipelines, retrieval-augmented generation (RAG) stacks, and multi-agent trading desks — that convert filings into Buy/Hold/Sell decisions inside real fund workflows. This report synthesizes all three layers and offers an assessment of which methods currently deserve the "state-of-the-art" label versus which remain promising but unvalidated.

## Academic Benchmarks for Filing-Based LLM Analysis

### FinanceBench: The Reference Standard

The most rigorously constructed public benchmark for LLM-driven filing analysis is **FinanceBench**, described in its current form as "the Finance Agent Benchmark" ([Bigeard et al., 20 May 2025, as cited in Emergent Mind](https://www.emergentmind.com/topics/financebench-sec-financial-filings-dataset)). It is built directly on 10-K, 10-Q, and 8-K forms retrieved programmatically from EDGAR via its REST API, and it pairs an expert-authored question–answer dataset with an **agentic harness** that lets models interact with live filings rather than static snippets ([Emergent Mind, 2026](https://www.emergentmind.com/topics/financebench-sec-financial-filings-dataset)). Its dataset is split into a public validation set (50 samples, CC BY 4.0), a private validation set (150 samples, research use), and a 337-sample leaderboard test set, enabling both open experimentation and closed, harder-to-game evaluation ([Emergent Mind, 2026](https://www.emergentmind.com/topics/financebench-sec-financial-filings-dataset)).

Critically, FinanceBench does not test flat retrieval alone. Its task taxonomy spans multiple reasoning difficulties:

| Task Category | Description | Number of Questions |
|---|---|---|
| Compute Percentages/CAGRs | Numeric derivation over reported figures | 83 |
| Beat or Miss | Compare actuals vs. management guidance | 69 |
| Financial Modeling | Aggregation/projection of balance-sheet data | 47 |
| Adjustments | Non-GAAP/addback calculations | 43 |
| Trends | Detect and analyze time-series shifts | 33 |
| Market Analysis | Cross-company comparison | 34 |
| Complex Retrieval | Multi-document event synthesis | 29 |

*(Table derived from [Emergent Mind, 2026](https://www.emergentmind.com/topics/financebench-sec-financial-filings-dataset))*

This taxonomy is significant for equity signal generation because it explicitly includes "Beat or Miss" and "Trends" categories — the two task types closest to actionable investment signals rather than pure fact extraction. A sobering counterpoint comes from [IntuitionLabs (2026)](https://intuitionlabs.ai/articles/llm-financial-document-analysis), which reports that finance-tuned LLMs "mis-answer optional questions 80% of the time" on the FinanceBench evaluation, underscoring that even the best current benchmark scores leave a wide reliability gap between what a model *can* answer and what it answers *correctly* when a question requires judgment rather than lookup.

### Emerging Benchmark: RAG and Knowledge-Graph Competitive Mapping

A second, less mature academic strand focuses specifically on **competitive and market-positioning inference** — a task standard extraction pipelines handle poorly. [Viswanathan et al. (n.d.)](https://openrgate.org/viewfulldetails.php?id=20412), published in the *International Journal for Research Trends and Innovation*, propose combining RAG with **knowledge graphs** to identify competitor relationships and market position from 10-K/10-Q narrative sections, arguing that conventional LLM prompting fails to capture how companies are positioned relative to peers. This is methodologically interesting — it moves beyond number extraction toward structured relational reasoning — but it should be weighted cautiously: it is a single paper in a journal with limited citation visibility, with no reported benchmark scores, backtested returns, or replication, and should be read as an early-stage architectural proposal rather than a validated production method.

## The Central Methodological Challenge: Look-Ahead Bias and Temporal Validity

The single most important 2025–2026 development affecting filing-based equity signal generation is not a new extraction technique — it is the discovery that most backtests of LLM-derived signals may be invalid due to **parametric look-ahead bias**.

### Parametric Look-Ahead Bias

As defined in the recent literature, parametric look-ahead bias is "evaluation leakage stored in an LLM's weights: a model pretrained after a historical test window may already 'know' how assets moved after the dates being backtested, so apparent forecasting skill is partly retrieval of memorized outcomes rather than prediction from ignorance" ([Li et al., 23 May 2026, as cited in Emergent Mind](https://www.emergentmind.com/topics/parametric-look-ahead-bias)). This is distinct from ordinary data leakage in an input pipeline — it is contamination baked into model parameters during pretraining, meaning a filing-analysis system can appear highly predictive on historical 10-K/10-Q data purely because the underlying LLM has memorized what happened afterward.

### Look-Ahead-Bench: Standardized Diagnostics

[Benhenda (2026)](https://inria.hal.science/hal-05466549v1/file/lookahead.pdf) formalizes this into **Look-Ahead-Bench**, a standardized benchmark built on the AI Hedge Fund agentic trading framework, using monthly rebalancing, fractional shares, and a fixed five-stock universe (AAPL, MSFT, GOOGL, NVDA, TSLA) to make results comparable across models ([Emergent Mind, 2026](https://www.emergentmind.com/topics/parametric-look-ahead-bias)). Its central diagnostic, "alpha decay," compares performance on memorized historical dates against genuinely out-of-sample periods; the benchmark shows that standard models can perform excellently in-sample and then "collapse" out-of-sample, while point-in-time (PiT) models — trained or filtered to exclude future information — remain stable or improve ([Emergent Mind, 2026](https://www.emergentmind.com/topics/parametric-look-ahead-bias)). The authors are explicit that this is a proof of concept limited to five large-cap technology stocks and two time periods, with "no direct causal proof of memorization, only strong behavioral evidence" ([Emergent Mind, 2026](https://www.emergentmind.com/topics/parametric-look-ahead-bias)) — an important caveat against over-generalizing its findings to the broader equity universe.

### Replication Evidence: Li et al. (2026)

The corrective technique proposed by [Li et al. (23 May 2026)](https://www.emergentmind.com/topics/parametric-look-ahead-bias) — a mitigation and unlearning approach evaluated across an eleven-model leaderboard — reports a **67.1% reduction** in performance on memorized dates once bias-correction is applied, while preserving 2025 out-of-sample returns within $8,000 and Sharpe ratio within 0.10 of baseline, and general-purpose reasoning within 1.7 points of the unmodified model. Most tellingly for benchmark validity, the correction raises the in-sample/out-of-sample Spearman rank correlation from +0.779 to +0.846, meaning leaderboard rankings become substantially more predictive of genuine future performance once memorization is suppressed. This is functionally a replication/audit study: it demonstrates that prior, uncorrected leaderboards materially overstated which models were actually good at forecasting from filings and price data, rather than merely recalling outcomes.

## Preprocessing and Retrieval Architectures

### Document Parsing and Chunking

Converting a 150+ page 10-K into model-ready input without destroying tabular structure is now treated as a solved but non-trivial engineering problem. [DeepRightAI (2025, December 27)](https://deeprightai.substack.com/p/how-hedge-funds-transform-sec-filings) documents a six-step pipeline used by hedge fund quantitative analysts: **10-K PDF → Docling → Markdown with tables → HybridChunker → contextualized chunks** (47 chunks in the worked example), which uses custom serializers to preserve markdown table structure and hierarchical section context (e.g., linking "Item 1A" risk factors to their parent section) rather than flattening tables into unreadable text strings. This addresses a documented failure mode: naive text extraction destroys column relationships that are essential for ratio calculations and quarter-over-quarter comparisons.

### Embedding, Vector Storage, and Retrieval

A concrete, named production pipeline is described by [Yennam (2025, LinkedIn)](https://www.linkedin.com/posts/angadyennam_end-to-end-steps-sec-hugging-face-embeddings-activity-7374749547618738176-il7a): filings (10-Q, 8-K) are ingested, cleaned, and chunked into 200–500 token passages; embeddings are generated with Hugging Face models (e.g., `all-mpnet-base-v2`); vectors are stored in **FAISS** for similarity search; a **LangChain** retriever pulls relevant chunks; and **Claude Sonnet 4** is prompted to classify each filing into {Buy, Hold, Sell} with rationale and a confidence score, after which business rules gate low-confidence outputs into human review. This RAG-plus-classification pattern — deep embeddings over TF-IDF baselines, with an LLM reasoning layer connecting quantitative and narrative signals — represents the most concrete, replicable "filing-to-signal" architecture in the sourced material.

### Knowledge Graphs as a Retrieval Complement

As noted above, [Viswanathan et al. (n.d.)](https://openrgate.org/viewfulldetails.php?id=20412) propose layering knowledge graphs onto RAG specifically to resolve competitor and market-position questions that pure vector retrieval handles weakly, since embedding similarity does not reliably encode explicit competitive relationships mentioned across multiple filings.

## From Filings to Signals: Deployed Production Systems

Several named or well-documented production and quasi-production systems now exist:

- **Hedge fund private RAG stacks**: [Sesen (n.d.)](https://sesen.ai/services/llm-use-cases-hedge-funds) reports that Balyasny and Point72 run private RAG variants over 10-Ks, 10-Qs, broker notes, and internal memos inside locked VPCs, framing the value as "coverage depth per analyst-hour," not autonomous trading.
- **TradingAgents**: a multi-agent LLM trading desk whose v0.3.1 release specifically added "Alpha Vantage look-ahead filtering," alongside a v0.3.0 "verified data-access contract" and CI gate — engineering evidence that at least one production-facing framework has operationalized the look-ahead bias concerns raised above, and has broadened data vendors to include FRED and Polymarket alongside Alpha Vantage ([MoClaw, n.d.](https://moclaw.ai/blog/what-is-tradingagents)).
- **Domain-tuned and general models**: BloombergGPT (50B parameters, trained on 363 billion tokens of financial text) remains the reference domain-specific model, reported to outperform general-purpose models on sentiment analysis, named-entity recognition, and filing QA ([Vishwa, 2026](https://navinvishwa.medium.com/the-analyst-in-the-machine-how-llms-are-reshaping-equity-research-478b2d49e75f); [IntuitionLabs, 2026](https://intuitionlabs.ai/articles/llm-financial-document-analysis)). On the general-purpose side, GPT-5.4 (reported March 2026) now ships with financial-services plugins for Excel and Google Sheets and integrations with FactSet, MSCI, Moody's, and Third Bridge, while context windows up to 1M tokens (Claude) and 256K tokens (GPT-5.4) increasingly allow whole 10-Ks to be processed without chunking, though cost-per-token still favors selective retrieval at high volume ([IntuitionLabs, 2026](https://intuitionlabs.ai/articles/llm-financial-document-analysis)).
- **Efficiency-optimized deployment**: Capital Fund Management, in a case study with Hugging Face, reports that a fine-tuned 0.3B-parameter NER model running on CPU achieves 80x cheaper inference than Llama 3.1-70B for filing-extraction tasks — evidence that production economics favor small, task-specific models over frontier LLMs for high-volume, narrow extraction ([Sesen, n.d.](https://sesen.ai/services/llm-use-cases-hedge-funds)).

## Performance Snapshot Table

| System / Study | Method | Reported Result | Source |
|---|---|---|---|
| MarketSenseAI 2.0 | LLM-augmented equity signals, S&P 100 | 125.9% cumulative return vs. 73.5% index (2023–2024) | [Sesen, n.d.](https://sesen.ai/services/llm-use-cases-hedge-funds), citing PMC review of 84 studies (2025) |
| CFM × Hugging Face | 0.3B fine-tuned NER vs. Llama 3.1-70B | 80x cheaper inference on CPU | [Sesen, n.d.](https://sesen.ai/services/llm-use-cases-hedge-funds) |
| Li et al. bias-correction | Unlearning/decoding fix for look-ahead bias | −67.1% score on memorized dates; Spearman ρ improves +0.779 → +0.846 | [Emergent Mind, 2026](https://www.emergentmind.com/topics/parametric-look-ahead-bias) |
| FinanceBench (general finding) | Optional/harder QA over filings | ~80% mis-answer rate on optional questions | [IntuitionLabs, 2026](https://intuitionlabs.ai/articles/llm-financial-document-analysis) |
| UK financial services (survey) | Any automated decision-making use of AI | 55% of AI use cases | [Sesen, n.d.](https://sesen.ai/services/llm-use-cases-hedge-funds), citing Bank of England & FCA AI Survey 2024 |

## Reliability Caveats and Human-in-the-Loop Validation

Across the sourced material, no author — academic or practitioner — recommends full automation. [ServicesGround (n.d.)](https://servicesground.com/blog/sec-filing-analysis-ai/) explicitly concludes that "the winning workflow is not blind automation. It is structured retrieval plus human validation," recommending that 10-Ks be used for deep baseline understanding, 10-Qs for quarter-over-quarter change detection, and 8-Ks for event-driven triggers — a task-differentiated retrieval strategy rather than a one-size-fits-all pipeline. This aligns with the production pattern in [Yennam (2025)](https://www.linkedin.com/posts/angadyennam_end-to-end-steps-sec-hugging-face-embeddings-activity-7374749547618738176-il7a), where low-confidence LLM classifications are routed to human review rather than executed automatically.

## Assessment: What Actually Qualifies as State of the Art

Based on the evidence gathered, three conclusions follow. First, **FinanceBench is currently the only benchmark with sufficient rigor (expert authorship, held-out leaderboard, agentic harness) to be called a genuine state-of-the-art evaluation standard** for filing comprehension; the RAG-plus-knowledge-graph approach for competitive analysis is architecturally promising but remains an unvalidated proposal, not yet state of the art in the sense of being benchmarked or replicated. Second, **the field's most consequential 2025–2026 methodological advance is not an extraction technique but a validity correction** — look-ahead-bias auditing via Look-Ahead-Bench and the Li et al. unlearning method — because it directly undermines the credibility of prior headline backtest numbers (including figures like MarketSenseAI 2.0's reported 125.9% vs. 73.5%) unless those results can be shown to be PiT-safe. Any equity signal architecture that has not adopted explicit look-ahead filtering (as TradingAgents v0.3.1 has begun to do) should be treated skeptically regardless of its backtested returns. Third, on production deployment, the RAG-over-Docling/HybridChunker-parsed-filings-into-classification pattern (FAISS + LangChain + an instruction-tuned LLM, gated by confidence thresholds and human review) is the most concretely documented and replicable production architecture available, and the economics increasingly favor small fine-tuned extraction models over frontier LLMs for high-volume narrow tasks, reserving frontier models for the reasoning/synthesis layer.

## Conclusion

State-of-the-art filing-to-signal pipelines in 2025–2026 combine three layers: a document pipeline that preserves table and hierarchical structure (Docling/HybridChunker), a retrieval layer (embeddings plus FAISS/vector search, increasingly supplemented by knowledge graphs for relational questions), and an LLM reasoning/classification layer whose outputs are benchmarked against FinanceBench-style QA and increasingly audited for look-ahead bias via frameworks like Look-Ahead-Bench. The most important recent scientific contribution is not a new signal-generation trick but a correction to how such signals should be *evaluated* — without point-in-time bias controls, reported backtest outperformance in this domain cannot be taken at face value.

## References

Bigeard, et al. (2025, May 20), as cited in Emergent Mind. (2026, April 17). *FinanceBench SEC filings dataset*. [https://www.emergentmind.com/topics/financebench-sec-financial-filings-dataset](https://www.emergentmind.com/topics/financebench-sec-financial-filings-dataset)

Benhenda, M. (2026). *Look-Ahead-Bench: A standardized benchmark of look-ahead bias in point-in-time LLMs for finance*. HAL Open Science. [https://inria.hal.science/hal-05466549v1/file/lookahead.pdf](https://inria.hal.science/hal-05466549v1/file/lookahead.pdf)

DeepRightAI. (2025, December 27). *How hedge funds transform SEC filings into LLM training data (without losing tables)*. Substack. [https://deeprightai.substack.com/p/how-hedge-funds-transform-sec-filings](https://deeprightai.substack.com/p/how-hedge-funds-transform-sec-filings)

Emergent Mind. (n.d.). *Parametric look-ahead bias*. [https://www.emergentmind.com/topics/parametric-look-ahead-bias](https://www.emergentmind.com/topics/parametric-look-ahead-bias)

Finance Alliance. (n.d.). *When AI models cheat: The hidden danger of look-ahead bias in financial LLMs*. [https://www.financealliance.io/the-hidden-danger-of-look-ahead-bias-in-financial-llms/](https://www.financealliance.io/the-hidden-danger-of-look-ahead-bias-in-financial-llms/)

IntuitionLabs. (2026, April 13, updated August 9). *LLMs for financial document analysis: SEC filings & decks*. [https://intuitionlabs.ai/articles/llm-financial-document-analysis](https://intuitionlabs.ai/articles/llm-financial-document-analysis)

MoClaw. (n.d.). *TradingAgents: A multi-agent LLM trading desk*. [https://moclaw.ai/blog/what-is-tradingagents](https://moclaw.ai/blog/what-is-tradingagents)

Sesen. (n.d.). *LLM use cases for hedge funds*. [https://sesen.ai/services/llm-use-cases-hedge-funds](https://sesen.ai/services/llm-use-cases-hedge-funds)

ServicesGround. (n.d.). *How to analyze SEC filings with LLM agents: 10-K, 10-Q, and EDGAR tutorial*. [https://servicesground.com/blog/sec-filing-analysis-ai/](https://servicesground.com/blog/sec-filing-analysis-ai/)

Viswanathan, S. B., Mysore, A. A. B., Karat, A., & Muthuregunathan, R. (n.d.). *LLM analysis of 10-K and 10-Q filings: RAG results*. International Journal for Research Trends and Innovation. [https://openrgate.org/viewfulldetails.php?id=20412](https://openrgate.org/viewfulldetails.php?id=20412)

Vishwa, N. (2026, March 18). *The analyst in the machine: How LLMs are reshaping equity research*. Medium. [https://navinvishwa.medium.com/the-analyst-in-the-machine-how-llms-are-reshaping-equity-research-478b2d49e75f](https://navinvishwa.medium.com/the-analyst-in-the-machine-how-llms-are-reshaping-equity-research-478b2d49e75f)

Yennam, A. (2025). *How hedge funds use LLMs to read SEC filings and make trades* [LinkedIn post]. LinkedIn. [https://www.linkedin.com/posts/angadyennam_end-to-end-steps-sec-hugging-face-embeddings-activity-7374749547618738176-il7a](https://www.linkedin.com/posts/angadyennam_end-to-end-steps-sec-hugging-face-embeddings-activity-7374749547618738176-il7a)

---

Rapporten er over 1200 ord, strukturert med APA-referanser og markdown-tabeller, og skrevet på engelsk som eksplisitt bedt om. Si ifra om du vil at jeg skal utdype et av avsnittene, f.eks. produksjonsarkitekturene eller look-ahead-bias-problematikken.
