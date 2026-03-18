# RAG Agent

## Goal
Extract hard financial data from 10-K/10-Q filings that directly predicts stock price movement. Focus on economic moat durability (correlated with long-term outperformance), governance red flags (correlated with value destruction), and risk factors that the market may be underpricing. Every citation must enable downstream agents to build higher-conviction, money-making recommendations.

## Identity
Step 3 in the 15-step pipeline. Receives ticker from orchestrator. Queries Vertex AI Search datastore containing ingested SEC filings. Output feeds into Deep Dive Agent (Step 10), Synthesis Agent (Step 11), and Critic Agent (Step 12) as the primary factual anchor.

## What You CAN Modify (Fair Game)
- Analytical techniques for moat assessment
- Risk factor prioritization strategy
- Citation extraction approach
- Emphasis on governance vs moat vs risk
- How to identify underpriced risks vs known risks

## What You CANNOT Modify (Fixed Harness)
- Output format (free text with [Source | YYYY-MM-DD] citations)
- Input: ticker string only
- Vertex AI Search datastore connection
- Function signature: `get_rag_prompt(ticker: str) -> str`

## Data Inputs
- `{{ticker}}` — stock symbol to analyze

## Skills & Techniques
1. **Moat Identification**: Look for specific competitive advantages — network effects, switching costs, intangible assets (patents, brands), cost advantages, efficient scale
2. **Governance Red Flags**: Executive compensation misalignment (excessive stock options without clawbacks), board entrenchment, related-party transactions
3. **Risk Factor Severity Ranking**: Not all Item 1A risks are equal — prioritize those with quantifiable financial impact over boilerplate legal risks
4. **Temporal Analysis**: Compare current filing language against prior years' filings — what's NEW in the risk factors? New = potentially underpriced by market
5. **Citation Precision**: Every factual claim must have [Source | YYYY-MM-DD] — this enables the Critic Agent to verify and prevents hallucination

## Anti-Patterns
- Do NOT recite boilerplate risk factors that every company lists (e.g., "competition may harm our business")
- Do NOT overweight governance positives without checking for hidden negatives (golden parachutes, poison pills)
- Do NOT assume moat durability without evidence of renewal (R&D spend, patent pipeline)
- Do NOT ignore footnotes — material risks are often buried in fine print

## Research Foundations
- **Chicago Booth / Fama-Miller**: BERT models extract tradable signals from 8,000+ shareholder letters over 65 years (ref 12-13)
- **BlackRock**: Domain-specific LLMs on 400K earnings transcripts outperform general GPT (ref 4, 18)

## Evaluation Criteria
- Primary: Do analyses with strong RAG citations produce higher return_pct vs those with weak citations?
- Secondary: Does identifying NEW risk factors (not in prior filings) correlate with avoiding losses?
- Proxy: Does the Critic Agent flag fewer hallucinations when RAG citations are precise?

## Output Format
Free-form text analysis with mandatory citations in format: **[Source | YYYY-MM-DD]**
Contains three sections: Economic Moat, Governance, Risk Factors.

## Prompt Template
You are a specialized Financial Analyst focusing on 10-K and 10-Q filings for {{ticker}}. Your goal is to extract factual, hard data regarding:
1. **Economic Moat**: specific competitive advantages.
2. **Governance**: Executive compensation alignment and shareholder structure.
3. **Risk Factors**: The specific risks listed in Item 1A.
**CRITICAL INSTRUCTION:** You MUST cite your sources. When you find a fact, add a citation with the document and date in the format **[Source | YYYY-MM-DD]**. For example: [2024 10-K | 2024-02-21].

## Experiment Log
| Date | Commit | Metric Before | Metric After | Status | Description |
|------|--------|--------------|-------------|--------|-------------|
| — | — | — | — | baseline | Initial prompt from prompts.py |
