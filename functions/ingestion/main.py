# ingestion-agent/main.py (Refactored)
import functions_framework
import logging
from datetime import timedelta
import pandas as pd
from utils.data_fetchers import fetch_raw_market_data
from utils.bigquery_utils import load_data_to_bigquery
from config import (
    PROJECT_ID, DATASET_ID, MARKET_TABLE_ID, START_DATE, API_SOURCE_MARKET
)
from response import decide_response

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

    # Extract and Standardize. data_fetchers.fetch_raw_market_data now
    # re-raises genuine fetch errors (phase-75.16 leg c) instead of
    # swallowing them into an empty DataFrame, so a real exception here
    # is distinguishable from "ran fine, no rows for this range."
    try:
        raw_data_df = fetch_raw_market_data(tickers, start_date, end_date, API_SOURCE_MARKET)
        fetch_ok = True
    except Exception as e:
        logging.error(f"Fetch failed for ingestion run (mode={mode}): {e}")
        raw_data_df = pd.DataFrame()
        fetch_ok = False

    rows_fetched = len(raw_data_df) if fetch_ok else 0
    load_ok = None  # meaningful only when fetch_ok and rows_fetched > 0

    if fetch_ok and rows_fetched > 0:
        # Load to Staging
        try:
            load_data_to_bigquery(
                raw_data_df,
                PROJECT_ID,
                DATASET_ID,
                MARKET_TABLE_ID,
                schema_type="MARKET"
            )
            load_ok = True
        except Exception as e:
            logging.error(f"Failed to load data to BigQuery: {e}")
            load_ok = False
    elif fetch_ok:
        logging.warning("No data fetched. Skipping BigQuery load.")

    body, status_code = decide_response(fetch_ok, rows_fetched, load_ok)
    logging.info(body)
    return (body, status_code)