"""
agent_prompts.py
Updated for Alpha Vantage Data Structures.
This module separates the 'Brain' (prompts) from the 'Body' (Home.py).
"""

import json

def get_rag_prompt(ticker: str, status_handler=None) -> str:
    if status_handler: status_handler.log("   Generating prompt for RAG Agent...")
    return (
        f"You are a specialized Financial Analyst focusing on 10-K and 10-Q filings for {ticker}. "
        "Your goal is to extract factual, hard data regarding:\n"
        "1. **Economic Moat**: specific competitive advantages.\n"
        "2. **Governance**: Executive compensation alignment and shareholder structure.\n"
        "3. **Risk Factors**: The specific risks listed in Item 1A.\n\n"
        "**CRITICAL INSTRUCTION:** You must cite your sources. When you find a fact, "
        "mention which document (e.g., '2024 10-K') it came from."
    )

def get_market_prompt(ticker: str, av_data: dict, status_handler=None) -> str:
    """
    Uses Alpha Vantage News Sentiment Data.
    """
    if status_handler: status_handler.log("   Generating prompt for Market Agent...")
    return (
        f"You are a Market Sentiment Specialist. Analyze the provided News Feed for {ticker}.\n"
        "--- DATA ---\n"
        f"{json.dumps(av_data.get('sentiment_summary', [])[:10])}\n"
        "------------\n"
        "**TASK:**\n"
        "1. **Sentiment Trend**: The data includes 'sentiment_score'. Is the trend positive or negative?\n"
        "2. **Key Narratives**: What are the dominant stories driving the stock?\n"
        "3. **Macro Context**: Do these stories relate to macro factors (rates, war) or company-specific execution?"
    )

def get_competitor_prompt(ticker: str, av_data: dict, status_handler=None) -> str:
    """
    Uses Dynamic Competitor Discovery from Alpha Vantage.
    """
    rivals = av_data.get('derived_competitors', [])
    if status_handler: status_handler.log("   Generating prompt for Competitor Agent...")
    return (
        f"You are a Competitor Intelligence Scout. Based on news co-occurrence, the following companies are frequently mentioned with {ticker}: {rivals}.\n\n"
        "**TASK:**\n"
        "1. Confirm if these are true rivals or just partners/sector peers.\n"
        "2. Based on the news summaries provided in the context, what moves are these rivals making?\n"
        "3. Assess if {ticker} is mentioned in a 'winning' or 'losing' context relative to these peers."
    )

def get_macro_prompt(ticker: str, av_data: dict, status_handler=None) -> str:
    """
    Uses Alpha Vantage Macroeconomic Data (CPI, Rates, GDP).
    """
    if status_handler: status_handler.log("   Generating prompt for Macro Agent...")
    return (
        f"You are a Macroeconomic Strategist. Analyze the provided economic indicators in the context of {ticker}.\n"
        "--- DATA ---\n"
        f"{json.dumps(av_data.get('macro_summary', {}))}\n"
        "------------\n"
        "**TASK:**\n"
        "1. **Economic Climate**: Based on CPI, Interest Rates, and GDP, what is the overall economic environment (e.g., inflationary, recessionary, growing)?\n"
        "2. **Impact on {ticker}**: How might this climate specifically affect {ticker}'s business? (e.g., consumer spending, borrowing costs).\n"
        "3. **Forward Outlook**: Are the trends in these indicators getting better or worse for the company?"
    )

def get_deep_dive_prompt(ticker: str, quant_data: dict, rag_text: str, market_text: str, competitor_text: str, status_handler=None) -> str:
    if status_handler: status_handler.log("   Generating questions for Deep-Dive Agent...")
    return (
        f"You are a Senior Investment Investigator. Your job is NOT to summarize, but to PROBE.\n"
        f"I have four reports for {ticker} that may contain contradictions or gaps.\n\n"
        "--- DATA SOURCES ---\n"
        f"1. QUANT (Financials): {json.dumps(quant_data)}\n"
        f"2. RAG (Filings): {rag_text[:3000]}...\n"
        f"3. MARKET (Sentiment): {market_text[:3000]}...\n"
        f"4. COMPETITOR (Rivals): {competitor_text[:3000]}...\n"
        "--------------------\n\n"
        "**TASK:**\n"
        "Identify 3 critical 'tensions' or 'contradictions' between these sources. "
        "Formulate 3 specific questions to resolve these tensions using the 10-K/10-Q."
        "Output ONLY the numbered list of questions."
    )

def get_synthesis_prompt(ticker: str, quant_data: dict, rag_text: str, market_text: str, competitor_text: str, deep_dive_text: str, status_handler=None, step_num: float = 8.1) -> str:
    if status_handler: status_handler.log(f"   -> Step {step_num}: Generating prompt for Synthesis Agent...")
    return (
        f"You are the Lead Analyst for {ticker}. Synthesize the following agent reports into a cohesive investment thesis.\n\n"
        "--- INPUTS ---\n"
        f"QUANT: {json.dumps(quant_data)}\n"
        f"RAG: {rag_text}\n"
        f"MARKET: {market_text}\n"
        f"COMPETITOR: {competitor_text}\n"
        f"DEEP DIVE: {deep_dive_text}\n"
        "--------------\n\n"
        "**REQUIREMENTS:**\n"
        "1. **Scoring Matrix**: Assign 0-10 scores for Corporate, Industry, Valuation, Sentiment, Governance.\n"
        "2. **Recommendation**: Strong Buy / Buy / Hold / Sell / Strong Sell.\n"
        "3. **Summary**: A professional narrative connecting the dots.\n\n"
        "**OUTPUT FORMAT:**\n"
        "Return a JSON object with keys: `scoring_matrix`, `recommendation`, `final_summary`, `key_risks`."
    )

def get_critic_prompt(ticker: str, draft_report: str, quant_data: dict, status_handler=None, step_num: float = 8.2) -> str:
    if status_handler: status_handler.log(f"   -> Step {step_num}: Generating prompt for Critic Agent...")
    return (
        f"You are the Compliance & Quality Control Officer. Review the draft report for {ticker}.\n\n"
        "--- HARD DATA (TRUTH) ---\n"
        f"{json.dumps(quant_data)}\n\n"
        "--- DRAFT REPORT ---\n"
        f"{draft_report}\n\n"
        "**YOUR TASK:**\n"
        "1. Check for **Hallucinations**: Does the draft mention numbers that contradict the Hard Data?\n"
        "2. Check for **Logic Errors**: Does a 'Strong Buy' recommendation accompany a low score (e.g., 3/10)?\n"
        "3. Check for **JSON Validity**: Is the structure correct?\n\n"
        "If the report is good, output the JSON exactly as is.\n"
        "If there are errors, correct the JSON values and summary to match the Hard Data, then output the CORRECTED JSON.\n"
        "Do not output markdown code blocks, just the raw JSON string."
    )