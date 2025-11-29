import yfinance as yf
import pandas as pd

def get_comprehensive_financials(ticker: str):
    """
    Fetches deep fundamental data that is critical for a "Warren Buffett" style analysis.
    Returns a dictionary of structured data.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # 1. Valuation Metrics
        valuation = {
            "Current Price": info.get('currentPrice'),
            "Market Cap": info.get('marketCap'),
            "P/E Ratio": info.get('trailingPE'),
            "Forward P/E": info.get('forwardPE'),
            "PEG Ratio": info.get('pegRatio'), # Critical for Growth vs Value
            "Price/Book": info.get('priceToBook'),
            "Dividend Yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        }

        # 2. Operational Efficiency (The "Quality" Score)
        efficiency = {
            "Profit Margin": info.get('profitMargins', 0) * 100,
            "Operating Margin": info.get('operatingMargins', 0) * 100,
            "Return on Equity (ROE)": info.get('returnOnEquity', 0) * 100,
            "Revenue Growth": info.get('revenueGrowth', 0) * 100
        }

        # 3. Financial Health (The "Safety" Score)
        # Avoid companies that will go bankrupt
        health = {
            "Total Cash": info.get('totalCash'),
            "Total Debt": info.get('totalDebt'),
            "Debt/Equity Ratio": info.get('debtToEquity'),
            "Current Ratio": info.get('currentRatio'), # Liquidity check
            "Free Cash Flow": info.get('freeCashflow')
        }

        # 4. Institutional Trust (Smart Money)
        # If institutions are buying, it's a good sign
        institutional = {
            "Inst. Ownership %": info.get('heldPercentInstitutions', 0) * 100,
            "Insider Ownership %": info.get('heldPercentInsiders', 0) * 100,
            "Short Ratio": info.get('shortRatio') # High short ratio = Squeeze potential or Garbage
        }

        return {
            "valuation": valuation,
            "efficiency": efficiency,
            "health": health,
            "institutional": institutional,
            "company_name": info.get('longName', ticker)
        }

    except Exception as e:
        return {"error": f"Failed to fetch yfinance data: {str(e)}"}

def get_price_history(ticker: str, period="1y"):
    """
    Fetches historical OHLCV data for charts/ML training.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df.reset_index().to_dict(orient='records')