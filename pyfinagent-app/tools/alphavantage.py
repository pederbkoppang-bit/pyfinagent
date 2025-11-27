import requests
import os
from collections import Counter

# Using the key provided. In production, keep this in secrets.toml
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "TV5O5XN8IS2NLR6X")

def get_market_intel(ticker: str, status_handler=None):
    """
    Fetches News & Sentiment from Alpha Vantage.
    
    POWER FEATURE: Dynamic Competitor Discovery.
    We analyze which OTHER tickers are mentioned in the same news stories 
    to automatically identify rivals without a hardcoded list.

    Args:
        ticker (str): The stock ticker to analyze.
        status_handler (StatusHandler, optional): Handler to log progress. Defaults to None.
    """
    if status_handler: status_handler.log(f"Fetching news & sentiment for {ticker} from Alpha Vantage...")
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=50&apikey={ALPHAVANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        feed = data.get('feed', [])
        if not feed:
            return {"error": "No news found or API limit reached.", "feed": []}

        # --- Dynamic Competitor Discovery ---
        if status_handler: status_handler.log(f"Found {len(feed)} news articles. Discovering competitors...")
        # "Who else is mentioned in stories about this ticker?"
        mentioned_tickers = []
        for article in feed:
            for t_sent in article.get('ticker_sentiment', []):
                sym = t_sent.get('ticker')
                # Exclude the target ticker and common ETFs/Indices if needed
                if sym and sym != ticker and sym not in ['FOREX', 'CRYPTO']:
                    mentioned_tickers.append(sym)
        
        # Get top 5 most co-mentioned tickers (Likely Rivals)
        likely_rivals = [t[0] for t in Counter(mentioned_tickers).most_common(5)]
        if status_handler and likely_rivals: status_handler.log(f"   -> Derived competitors: {', '.join(likely_rivals)}")

        # --- Structured News Data ---
        news_summary = []
        for article in feed[:15]: # Limit to top 15 stories for token efficiency
            title = article.get('title')
            source = article.get('source')
            if status_handler and title: status_handler.log(f"    Processing news: {title} (Source: {source})")
            news_summary.append({
                "title": article.get('title'),
                "published": article.get('time_published'),
                "source": article.get('source'),
                "sentiment_score": article.get('overall_sentiment_score'),
                "sentiment_label": article.get('overall_sentiment_label'),
                "summary": article.get('summary')
            })

        return {
            "target": ticker,
            "derived_competitors": likely_rivals,
            "sentiment_summary": news_summary
        }

    except Exception as e:
        if status_handler: status_handler.log(f"ERROR: Failed to fetch market intel from Alpha Vantage: {e}")
        return {"error": str(e)}

def get_fundamental_overview(ticker: str, status_handler=None):
    """
    Fetches the Company Overview (Market Cap, PE, Description).
    Useful for the Quant Agent to get a quick snapshot.

    Args:
        ticker (str): The stock ticker to analyze.
        status_handler (StatusHandler, optional): Handler to log progress. Defaults to None.
    """
    if status_handler: status_handler.log(f"Fetching fundamental overview for {ticker} from Alpha Vantage...")
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        if status_handler: status_handler.log("   -> Fundamental overview received.")
        return response.json()
    except Exception as e:
        if status_handler: status_handler.log(f"ERROR: Failed to fetch fundamental overview: {e}")
        return {"error": str(e)}