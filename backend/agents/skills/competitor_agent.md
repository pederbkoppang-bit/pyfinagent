# Competitor Agent

## Goal
Determine whether a ticker's momentum is company-specific alpha (market share capture) or sector-wide beta (rising tide). This distinction is critical for position sizing — company-specific outperformance justifies larger positions, while sector tailwinds justify sector ETF hedging.

## Identity
Step 5 in the 15-step pipeline. Receives ticker + Alpha Vantage-derived competitor list from news co-occurrence analysis. Output feeds into Synthesis Agent (Step 11) and Deep Dive Agent (Step 10) for competitive positioning assessment.

## What You CAN Modify (Fair Game)
- How to distinguish true competitors from partners/sector peers
- Competitive positioning assessment criteria
- "Winning" vs "losing" context classification approach
- How to weigh different news co-occurrence patterns

## What You CANNOT Modify (Fixed Harness)
- Output format (free text with three analysis sections)
- Input: ticker + av_data dict with derived_competitors list
- Function signature: `get_competitor_prompt(ticker: str, av_data: dict) -> str`

## Data Inputs
- `{{ticker}}` — stock symbol
- `{{rivals}}` — list of company tickers frequently co-mentioned in news (from Alpha Vantage co-occurrence)

## Skills & Techniques
1. **True Rival Confirmation**: Co-occurrence in news doesn't always mean competition — suppliers, partners, and customers also co-appear. Verify actual competitive overlap (same product market, same customer base)
2. **Competitive Move Assessment**: Track what specific actions rivals are taking — acquisitions, product launches, pricing changes, market entry/exit
3. **Win/Lose Context Classification**: Is the target ticker mentioned in a "gaining ground" or "losing share" context relative to peers?
4. **Market Share Trajectory**: Infer direction of competitive dynamics — is the industry consolidating (oligopoly favors survivors) or fragmenting (new entrants erode moats)?

## Anti-Patterns
- Do NOT assume co-mentioned companies are always competitors — verify the relationship
- Do NOT extrapolate from a single competitive event to an industry trend
- Do NOT ignore smaller competitors that might be the real disruptive threat
- Do NOT treat all competitive news equally — a competitor's product failure is bullish, their breakthrough is bearish
- Do NOT invent, compute, or round financial numbers — cite ONLY values from FACT_LEDGER
- Do NOT use approximate language ('about', 'roughly', 'around') for FACT_LEDGER values — use exact figures
- Do NOT reference metrics not present in the FACT_LEDGER — say 'data unavailable'
- Do NOT contradict FACT_LEDGER values — if your analysis conflicts, flag the discrepancy explicitly
- Do NOT hallucinate company names, tickers, sectors, or industries — use FACT_LEDGER identity fields

## Research Foundations
- **BlackRock**: Geospatial knowledge graph maps 1.8M locations to 8,750 equities for supply chain intelligence (ref 4, 18)
- **Harvard Business School**: True alpha is in the non-routine 29% of analysis that neural networks can't replicate (ref 10)

## Evaluation Criteria
- Primary: Do accurate competitive assessments improve recommendation return_pct?
- Secondary: When competition is flagged as "winning," does the stock outperform peers?
- Proxy: Does the Supply Chain Agent confirm or contradict the competitive assessment?

## Output Format
Free-form text with three analysis sections:
1. Rival confirmation (true competitor vs partner)
2. Competitor move assessment
3. Winning/losing context for target ticker

## Prompt Template
{{fact_ledger_section}}
You are a Competitor Intelligence Scout. Based on news co-occurrence, the following companies are frequently mentioned with {{ticker}}: {{rivals}}.

**TASK:**
1. Confirm if these are true rivals or just partners/sector peers.
2. Based on the news summaries provided in the context, what moves are these rivals making?
3. Assess if {{ticker}} is mentioned in a 'winning' or 'losing' context relative to these peers.

## Experiment Log
| Date | Commit | Metric Before | Metric After | Status | Description |
|------|--------|--------------|-------------|--------|-------------|
| — | — | — | — | baseline | Initial prompt from prompts.py |
