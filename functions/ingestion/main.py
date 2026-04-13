# ingestion-agent/main.py (Refactored)
import functions_framework
import logging
from datetime import datetime, timedelta
import pandas as pd
from utils.data_fetchers import fetch_raw_market_data
from utils.bigquery_utils import load_data_to_bigquery
from config import (
    PROJECT_ID, DATASET_ID, MARKET_TABLE_ID, START_DATE, API_SOURCE_MARKET
)

logging.basicConfig(level=logging.INFO)

def get_historical_universe(target_date):
    """
    CRITICAL TODO: Implement logic to fetch historical index constituents.
    To prevent survivorship bias, this function must return the list of stocks 
    that were active on the target_date, including those now delisted.
    """
    logging.warning("CRITICAL: Using placeholder universe. Implement historical constituent fetching ASAP.")
    # Placeholder static list
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPY', 'QQQ']

@functions_framework.http
def ingest_market_data_el(request):
    """
    HTTP Cloud Function to execute the E-L process for raw market data.
    """
    request_json = request.get_json(silent=True)
    
    # 1. Define the Time Range and Mode
    mode = request_json.get('mode', 'incremental') if request_json else 'incremental'
    
    # Determine Start and End Dates
    today = pd.Timestamp.utcnow().normalize() # Use normalized UTC date
    if mode == 'backfill':
        start_date = START_DATE
        end_date = today.strftime('%Y-%m-%d')
    else:
        # Incremental: Fetch the last few days to capture recent data/corrections
        days_back = 5
        end_date = today.strftime('%Y-%m-%d')
        start_date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')

    # 2. Define the Universe (Tickers)
    # The universe must be historically accurate
    tickers = get_historical_universe(start_date)

    # Allow overriding tickers via the request for specific troubleshooting
    if request_json and 'tickers' in request_json:
        tickers = request_json['tickers']
        logging.info(f"Overriding universe with specified tickers: {tickers}")

    logging.info(f"Starting ingestion. Mode: {mode}. Range: {start_date} to {end_date}. Tickers: {len(tickers)}")

    # 3. Execute E-L Process
    
    # Extract and Standardize
    raw_data_df = fetch_raw_market_data(tickers, start_date, end_date, API_SOURCE_MARKET)
    
    if not raw_data_df.empty:
        # Load to Staging
        try:
            load_data_to_bigquery(
                raw_data_df, 
                PROJECT_ID, 
                DATASET_ID, 
                MARKET_TABLE_ID, 
                schema_type="MARKET"
            )
            status = "Success"
        except Exception as e:
            logging.error(f"Failed to load data to BigQuery: {e}")
            status = "Failure"
    else:
        logging.warning("No data fetched. Skipping BigQuery load.")
        status = "Success (No Data)"

    return f"Ingestion completed. Status: {status}. Rows: {len(raw_data_df)}."