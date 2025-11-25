import requests
import os
from collections import Counter

# Using the key provided. In production, keep this in secrets.toml
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "TV5O5XN8IS2NLR6X")

def get_market_intel(ticker: str):
    """
    Fetches News & Sentiment from Alpha Vantage.
    
    POWER FEATURE: Dynamic Competitor Discovery.
    We analyze which OTHER tickers are mentioned in the same news stories 
    to automatically identify rivals without a hardcoded list.
    """
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=50&apikey={ALPHAVANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        
        feed = data.get('feed', [])
        if not feed:
            return {"error": "No news found or API limit reached.", "feed": []}

        # --- Dynamic Competitor Discovery ---
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

        # --- Structured News Data ---
        news_summary = []
        for article in feed[:15]: # Limit to top 15 stories for token efficiency
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
        return {"error": str(e)}

def get_fundamental_overview(ticker: str):
    """
    Fetches the Company Overview (Market Cap, PE, Description).
    Useful for the Quant Agent to get a quick snapshot.
    """
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    try:
        return requests.get(url).json()
    except Exception as e:
        return {"error": str(e)}