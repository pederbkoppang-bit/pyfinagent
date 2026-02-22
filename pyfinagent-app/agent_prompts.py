"""
agent_prompts.py
Updated for Alpha Vantage Data Structures.
This module separates the 'Brain' (prompts) from the 'Body' (Home.py).
"""

import json
from pathlib import Path

# Load the detailed synthesis prompt from the adjacent text file.
SYNTHESIS_PROMPT_PATH = Path(__file__).parent / "synthesis_prompt.txt"
SYNTHESIS_PROMPT_TEMPLATE = SYNTHESIS_PROMPT_PATH.read_text()

CRITIC_PROMPT_TEMPLATE = """You are the Compliance & Quality Control Officer. Review the draft report for {ticker}.

--- HARD DATA (TRUTH) ---
{quant_data}

--- DRAFT REPORT ---
{draft_report}

**YOUR TASK:**
1. Check for **Hallucinations**: Does the draft mention numbers that contradict the Hard Data?
2. Check for **Logic Errors**: Does a 'Strong Buy' recommendation accompany a low score (e.g., 3/10)?
3. Check for **JSON Validity**: Is the structure correct?

If the report is good, output the JSON exactly as is.
If there are errors, correct the JSON values and summary to match the Hard Data, then output the CORRECTED JSON.
Do not output markdown code blocks, just the raw JSON string."""

def get_rag_prompt(ticker: str, status_handler=None) -> str:
    if status_handler: status_handler.log("   Generating prompt for RAG Agent...")
    return (
        f"You are a specialized Financial Analyst focusing on 10-K and 10-Q filings for {ticker}. "
        "Your goal is to extract factual, hard data regarding:\n"
        "1. **Economic Moat**: specific competitive advantages.\n"
        "2. **Governance**: Executive compensation alignment and shareholder structure.\n"
        "3. **Risk Factors**: The specific risks listed in Item 1A.\n"
        "**CRITICAL INSTRUCTION:** You MUST cite your sources. When you find a fact, "
        "add a citation with the document and date in the format **[Source | YYYY-MM-DD]**. For example: [2024 10-K | 2024-02-21]."
    )

def get_market_prompt(ticker: str, av_data: dict, status_handler=None) -> str:
    """
    Uses Alpha Vantage News Sentiment Data.
    """
    if status_handler: status_handler.log("   Generating prompt for Market Agent (Sentiment Divergence)...")

    # The prompt is designed based on academic research (e.g., Tetlock, 2007) that media sentiment
    # can be a leading indicator for stock price movements, especially when sentiment diverges from price.
    return (
        f"You are an advanced quantitative sentiment analyst for {ticker}. Your task is to detect early breakout signals by identifying sentiment-price divergence.\n"
        "You will analyze up to 50 recent news articles to find evidence of an 'Accumulation Phase' where positive news sentiment is rising but the market has not yet priced it in.\n\n"
        "--- RECENT NEWS SENTIMENT DATA ---\n"
        f"{json.dumps(av_data.get('sentiment_summary', [])[:50])}\n"
        "----------------------------------\n\n"
        "**YOUR TASK:**\n"
        "Execute the following analysis and structure your output into the three sections specified below.\n\n"
        "1.  **Calculate Sentiment Velocity**: Assess the momentum of sentiment. Is the average `sentiment_score` across the articles strongly positive (e.g., > 0.35)? Is the narrative strengthening over time?\n\n"
        "2.  **Check for Divergence**: This is the critical signal. Search the news summaries for narratives suggesting the stock price is 'undervalued,' 'ignored,' 'flat,' 'range-bound,' or has 'not yet reacted.' If you find strongly positive sentiment combined with these price-suppression narratives, issue a 'Divergence Warning'.\n\n"
        "3.  **Identify Catalyst Phrasing**: Scan the news summaries for specific, forward-looking institutional catalyst keywords. These are often associated with Q3 tech/cyclical breakouts. Keywords to look for include: 'inventory bottoming,' 'unmet demand,' 'supply chain recovery,' 'upgraded guidance,' 'new cycle,' 'pent-up demand.'\n\n"
        "**OUTPUT STRUCTURE:**\n"
        "Provide your analysis in the following format ONLY:\n\n"
        "[1] Average Sentiment Momentum: <Your analysis of sentiment velocity and strength.>\n"
        "[2] Divergence Analysis (Is this a hidden breakout?): <State whether you've found a divergence. If so, issue the 'Divergence Warning' and explain why. Otherwise, explain why not.>\n"
        "[3] Key Institutional Catalysts: <List any catalyst keywords you found and the context in which they appeared. If none, state 'No specific catalysts identified.'>"
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

def get_synthesis_prompt(ticker: str, quant_report: dict, rag_report: str, market_report: str, deep_dive_analysis: str, status_handler=None, step_num: float = 8.1) -> str:
    if status_handler: status_handler.log(f"   -> Step {step_num}: Generating prompt for Synthesis Agent...")
    # Use the loaded template and format it with the provided context.
    # Note: The template expects market_report, but the old function signature had competitor and macro.
    # I am aligning with the template's variable names.
    return SYNTHESIS_PROMPT_TEMPLATE.format(
        ticker=ticker,
        quant_report=json.dumps(quant_report, indent=2),
        rag_report=rag_report,
        market_report=market_report, # Assuming this combines market, competitor, and macro reports.
        deep_dive_analysis=deep_dive_analysis
    )

def get_critic_prompt(ticker: str, draft_report: str, quant_data: dict, status_handler=None, step_num: float = 8.2) -> str:
    if status_handler: status_handler.log(f"   -> Step {step_num}: Generating prompt for Critic Agent...")
    # The draft_report might be a JSON string. For consistency in the prompt,
    # it's better to load it into a Python object and then dump it back to a
    # consistently formatted string.
    try:
        # Ensure the draft report is a consistently formatted JSON string.
        draft_obj_str = json.dumps(json.loads(draft_report), indent=2)
    except json.JSONDecodeError:
        draft_obj_str = json.dumps({"error": "Could not parse draft report", "raw": draft_report})

    return CRITIC_PROMPT_TEMPLATE.format(
        ticker=ticker,
        quant_data=json.dumps(quant_data, indent=2),
        draft_report=draft_obj_str
    )