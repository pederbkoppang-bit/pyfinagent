"""
Extend historical data in BigQuery for deeper walk-forward backtesting.

Current state: 49 tickers, 2023-01 to 2025-12
Target state: 150+ tickers, 2018-01 to 2025-12

This gives us 7+ years of data for walk-forward validation, which means:
- With 12-month training + 3-month test + 5-day embargo:
  ~24 walk-forward windows instead of ~8
- More windows = more reliable DSR
- Includes COVID crash (2020), 2022 bear market, 2023-2025 recovery
- Covers multiple market regimes for robust strategy testing

Run once: python extend_historical_data.py
"""

import json
import os
import sys
import logging

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv("backend/.env")

from google.cloud import bigquery
from google.oauth2 import service_account
from backend.config.settings import get_settings
from backend.backtest.data_ingestion import DataIngestionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Extended universe: top 150 S&P 500 by market cap + sector diversification
# Includes tickers that have been in S&P 500 continuously since 2018
# (minimizes survivorship bias for the extended period)
EXTENDED_UNIVERSE = [
    # Already ingested (49)
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AMD", "AMZN", "AVGO", "BA", "BMY",
    "BRK-B", "COST", "CRM", "CSCO", "CVX", "DHR", "GOOGL", "HD", "HON", "INTC",
    "JNJ", "JPM", "KO", "LLY", "LOW", "MA", "MCD", "META", "MRK", "MSFT",
    "NEE", "NKE", "NVDA", "ORCL", "PEP", "PG", "PM", "QCOM", "RTX", "SBUX",
    "TMO", "TSLA", "TXN", "UNH", "UNP", "UPS", "V", "WMT", "XOM",
    # NEW — Large caps, stable S&P 500 members since 2018
    # Technology
    "AMAT", "ADI", "KLAC", "LRCX", "MCHP", "SNPS", "CDNS", "FTNT", "NOW", "PANW",
    # Healthcare
    "AMGN", "GILD", "ISRG", "MDT", "SYK", "ZTS", "REGN", "VRTX", "EW", "BDX",
    # Financials
    "GS", "MS", "AXP", "BLK", "C", "WFC", "SCHW", "CB", "MMC", "PNC",
    "USB", "TFC", "AIG", "MET", "PRU",
    # Consumer Discretionary
    "TJX", "ROST", "YUM", "CMG", "MAR", "HLT", "GM", "F", "EBAY", "BKNG",
    # Consumer Staples
    "CL", "EL", "GIS", "K", "KMB", "SJM", "MKC", "HSY", "MNST", "STZ",
    # Industrials
    "CAT", "GE", "MMM", "DE", "LMT", "NOC", "GD", "ITW", "EMR", "ROK",
    "FDX", "NSC", "CSX",
    # Energy
    "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HES",
    # Utilities
    "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC",
    # Real Estate
    "PLD", "AMT", "CCI", "EQIX", "SPG", "PSA",
    # Materials
    "LIN", "APD", "SHW", "ECL", "NEM", "FCX",
    # Communication
    "DIS", "CMCSA", "NFLX", "TMUS", "VZ", "T", "CHTR",
]

# Deduplicate
EXTENDED_UNIVERSE = list(dict.fromkeys(EXTENDED_UNIVERSE))


def main():
    settings = get_settings()
    creds_json = os.environ.get("GCP_CREDENTIALS_JSON", "").strip("'")
    creds = service_account.Credentials.from_service_account_info(json.loads(creds_json))
    client = bigquery.Client(credentials=creds, project=settings.gcp_project_id)

    ingestion = DataIngestionService(client, settings)

    logger.info(f"Extending historical data: {len(EXTENDED_UNIVERSE)} tickers, 2018-01-01 to 2025-12-31")

    # Check current state
    status = ingestion.get_ingestion_status()
    logger.info(f"Current BQ state: {status}")

    # Run full ingestion with extended date range
    # The ingestion service skips existing (ticker, date) pairs, so this is safe to re-run
    result = ingestion.run_full_ingestion(
        tickers=EXTENDED_UNIVERSE,
        start_date="2018-01-01",
        end_date="2025-12-31",
        fred_api_key=os.environ.get("FRED_API_KEY", ""),
    )

    logger.info(f"Ingestion complete: {result}")

    # Verify
    new_status = ingestion.get_ingestion_status()
    logger.info(f"New BQ state: {new_status}")
    logger.info(f"Price rows added: {new_status.get('historical_prices', 0) - status.get('historical_prices', 0)}")
    logger.info(f"Fundamental rows added: {new_status.get('historical_fundamentals', 0) - status.get('historical_fundamentals', 0)}")


if __name__ == "__main__":
    main()
