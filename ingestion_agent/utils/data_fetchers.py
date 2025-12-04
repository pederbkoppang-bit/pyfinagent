# ingestion-agent/utils/data_fetchers.py (Refactored)
import yfinance as yf
import pandas as pd
import logging
import pytz

logging.basicConfig(level=logging.INFO)

# Assuming NYSE/NASDAQ data for yfinance
SOURCE_TIMEZONE = pytz.timezone("America/New_York")

def fetch_raw_market_data(tickers, start_date, end_date, api_source):
    """
    Fetches raw, unadjusted market data (OHLCV, Dividends, Splits).
    """
    try:
        # 1. EXTRACT
        # CRITICAL: auto_adjust=False ensures we get the raw prices.
        # actions=True ensures we get Dividends and Stock Splits.
        logging.info(f"Fetching data for {len(tickers)} tickers...")
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date, 
            auto_adjust=False, 
            actions=True,
            group_by='ticker'
        )
        
        if data.empty:
            return pd.DataFrame()

        # 2. RESHAPE
        # Flatten the MultiIndex DataFrame (if fetching multiple tickers)
        if len(tickers) > 1 and hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
            stack_data = data.stack(level=0).rename_axis(['Timestamp', 'Ticker']).reset_index()
        else:
            stack_data = data.rename_axis('Timestamp').reset_index()
            if 'Ticker' not in stack_data.columns:
                stack_data['Ticker'] = tickers[0] # Handle single ticker

        # 3. STANDARDIZE TIMESTAMPS (Crucial for PiT)
        if stack_data['Timestamp'].dt.tz is None:
            # If naive, localize to source timezone and handle potential DST ambiguities
            logging.warning("Timestamps are timezone-naive. Localizing to America/New_York.")
            try:
                stack_data['Timestamp_UTC'] = stack_data['Timestamp'].dt.tz_localize(SOURCE_TIMEZONE, ambiguous='infer').dt.tz_convert('UTC')
            except pytz.exceptions.AmbiguousTimeError as e:
                logging.error(f"Ambiguous time localization failed: {e}")
                # Fallback or error handling required here
                raise
        else:
            # If timezone-aware, convert to UTC
            stack_data['Timestamp_UTC'] = stack_data['Timestamp'].dt.tz_convert('UTC')

        # Make naive UTC for BigQuery TIMESTAMP type compatibility
        stack_data['Timestamp_UTC'] = stack_data['Timestamp_UTC'].dt.tz_localize(None)
        
        # 4. Align Schema
        stack_data.rename(columns={'Stock Splits': 'Stock_Splits'}, inplace=True)
        
        # 5. Add Metadata
        stack_data['API_Source'] = api_source
        stack_data['Ingestion_Timestamp'] = pd.Timestamp.utcnow().tz_localize(None)
        
        # 6. Data Types (No Imputation)
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock_Splits']
        for col in numeric_cols:
             if col in stack_data.columns:
                stack_data[col] = pd.to_numeric(stack_data[col], errors='coerce')

        # Select Final Columns
        required_columns = [
            'API_Source', 'Ticker', 'Timestamp_UTC', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Dividends', 'Stock_Splits', 'Ingestion_Timestamp'
        ]
        
        return stack_data[required_columns]

    except Exception as e:
        logging.error(f"Error fetching market data: {e}")
        return pd.DataFrame()