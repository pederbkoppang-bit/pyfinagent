# tools/portfolio.py
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
import logging
import os
from datetime import datetime, timezone
import uuid

# Configure logging and initialization
logging.basicConfig(level=logging.INFO)
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
BQ_DATASET = "pyfinagent_pms"

# Constants (Mirrored from risk-management-agent/main.py)
MAX_GROSS_LEVERAGE = 1.5
MAX_DRAWDOWN = 0.15

try:
    BQ_CLIENT = bigquery.Client(project=GCP_PROJECT_ID)
except Exception as e:
    logging.warning(f"Failed to initialize BigQuery Client: {e}. Using mock data.")
    BQ_CLIENT = None

# --- Mock Data Functions (Used if BQ_CLIENT is None) ---
def get_mock_holdings():
    data = {
        'ticker': ['AAPL', 'NVDA', 'MSFT', 'TSLA'],
        'quantity': [100.0, 50.0, 75.0, 20.0],
        'avg_entry_price': [150.0, 800.0, 300.0, 170.0],
        'cost_basis_base_ccy': [15000.0, 40000.0, 22500.0, 3400.0],
        'sector': ['Technology', 'Technology', 'Technology', 'Consumer Cyclical'],
        'strategy_tag': ['Core', 'AI_Momentum', 'Core', 'Sentiment']
    }
    return pd.DataFrame(data)

def get_mock_status():
    # Mock status reflecting the holdings above + cash
    return {
        'total_equity': 200000.0, 'cash_balance': 119100.0, 'daily_pnl': 1500.0,
        'peak_equity': 210000.0, 'current_drawdown': 0.0476
    }

def get_mock_historical_equity():
    """Generates mock historical equity data for the last 5 years."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - pd.Timedelta(days=365 * 5)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simulate a random walk for equity
    np.random.seed(42)
    returns = np.random.normal(loc=0.0005, scale=0.015, size=len(dates))
    equity = 150000 * (1 + returns).cumprod()
    
    return pd.DataFrame({'timestamp': dates, 'equity': equity})

def get_mock_trade_history():
    """Generates a mock trade history DataFrame."""
    data = {
        'timestamp': pd.to_datetime(['2024-07-15 14:30:00', '2024-07-12 10:05:00', '2024-06-28 11:00:00', '2024-06-20 09:35:00']),
        'ticker': ['NVDA', 'TSLA', 'AAPL', 'MSFT'],
        'side': ['BUY', 'SELL', 'BUY', 'BUY'],
        'quantity': [10.0, 20.0, 50.0, 25.0],
        'price': [950.0, 175.0, 148.0, 298.0],
        'cost_basis': [9500.0, -3500.0, 7400.0, 7450.0],
        'strategy_tag': ['AI_Momentum', 'Sentiment', 'Core', 'Core']
    }
    return pd.DataFrame(data)
# --- Data Fetching Functions ---

@st.cache_data(ttl=30)
def fetch_active_holdings(user_id: str, account_id: str) -> pd.DataFrame:
    """Fetches current holdings from the BigQuery VIEW."""
    if BQ_CLIENT is None:
        return get_mock_holdings()

    query = f"""
        SELECT ticker, quantity, avg_entry_price, cost_basis_base_ccy, sector, strategy_tag 
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.active_holdings_view`
        WHERE user_id = @user_id AND account_id = @account_id
    """
    # In a real implementation, this executes the query.
    # job_config = bigquery.QueryJobConfig(...)
    # return BQ_CLIENT.query(query, job_config=job_config).to_dataframe()
    logging.info("Fetching holdings (Mock data used for demonstration)")
    return get_mock_holdings() 

@st.cache_data(ttl=30)
def fetch_portfolio_snapshot(user_id: str, account_id: str) -> dict:
    """Fetches the high-level account status snapshot."""
    if BQ_CLIENT is None:
        return get_mock_status()

    query = f"""
        SELECT * FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.portfolio_status_snapshot`
        WHERE user_id = @user_id AND account_id = @account_id 
        ORDER BY snapshot_timestamp DESC LIMIT 1
    """
    # (Query execution implementation omitted for brevity)
    logging.info("Fetching snapshot (Mock data used for demonstration)")
    return get_mock_status() 

@st.cache_data(ttl=300)
def get_historical_equity(user_id: str, account_id: str) -> pd.DataFrame:
    """Fetches historical equity curve data from BigQuery."""
    if BQ_CLIENT is None:
        return get_mock_historical_equity()

    query = f"""
        SELECT timestamp, equity
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.historical_equity`
        WHERE user_id = @user_id AND account_id = @account_id
        ORDER BY timestamp ASC
    """
    logging.info("Fetching historical equity (Mock data used for demonstration)")
    return get_mock_historical_equity()

@st.cache_data(ttl=300)
def get_trade_history(user_id: str, account_id: str) -> pd.DataFrame:
    """Fetches trade execution history from BigQuery."""
    if BQ_CLIENT is None:
        return get_mock_trade_history()

    query = f"""
        SELECT timestamp, ticker, side, quantity, price, cost_basis, strategy_tag
        FROM `{GCP_PROJECT_ID}.{BQ_DATASET}.trade_history`
        WHERE user_id = @user_id AND account_id = @account_id
        ORDER BY timestamp DESC
    """
    logging.info("Fetching trade history (Mock data used for demonstration)")
    return get_mock_trade_history()

@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_benchmark_data(tickers: list) -> pd.DataFrame:
    """Fetches historical close prices for a list of benchmark tickers."""
    if not tickers:
        return pd.DataFrame()
    
    try:
        # Fetch data for the last 5 years to ensure we have enough history
        data = yf.download(tickers, period="5y", interval="1d", auto_adjust=True)['Close']
        if data.empty:
            return pd.DataFrame()
        # If only one ticker is fetched, it returns a Series. Convert to DataFrame.
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data.tz_localize('UTC') # Localize to UTC to match portfolio data
    except Exception as e:
        logging.error(f"Failed to fetch benchmark data for {tickers}: {e}")
        return pd.DataFrame()

def fetch_live_market_data(tickers: list) -> dict:
    """Fetches live prices and sparkline data using yfinance (Bulk fetch)."""
    if not tickers: return {}
    data = {}
    try:
        # Bulk fetch using yf.Tickers
        yf_data = yf.Tickers(" ".join(tickers))
        
        # Fetch 7-day history (1h interval) for sparklines
        history = yf_data.history(period="7d", interval="1h", actions=False, auto_adjust=True)['Close']

        for ticker in tickers:
            try:
                info = yf_data.tickers[ticker].info
                # Get current price
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                
                # Extract sparkline data
                if len(tickers) == 1 and isinstance(history, pd.Series):
                    sparkline = history.dropna().tolist()
                elif isinstance(history, pd.DataFrame) and ticker in history.columns:
                    sparkline = history[ticker].dropna().tolist()
                else:
                    sparkline = []

                if current_price:
                    data[ticker] = {'current_price': current_price, 'sparkline_7d': sparkline}
            except Exception as e:
                logging.warning(f"Could not fetch data for {ticker}: {e}")
                data[ticker] = {'current_price': None, 'sparkline_7d': []}
    except Exception as e:
        logging.error(f"yfinance batch error: {e}")
    return data

# --- Main Hybridization Logic ---

def get_live_portfolio_state(user_id: str, account_id: str) -> dict:
    """The Nervous System: Merges BQ data with live prices and calculates risk."""
    
    # 1. Fetch Static Data
    df_holdings = fetch_active_holdings(user_id, account_id)
    snapshot = fetch_portfolio_snapshot(user_id, account_id)
    
    if not snapshot:
        return {"holdings_df": pd.DataFrame(), "metrics": {}, "error": "Account snapshot not found."}

    # 2. Fetch Live Market Data
    tickers = df_holdings['ticker'].unique().tolist()
    market_data = fetch_live_market_data(tickers)

    # 3. Hybridization
    def enrich_row(row):
        data = market_data.get(row['ticker'])
        if data and data['current_price']:
            price = data['current_price']
            row['current_price'] = price
            row['market_value'] = row['quantity'] * price
            
            # PnL = (Current Price - Entry Price) * Quantity (Handles Long/Short)
            row['pnl_unrealized'] = (price - row['avg_entry_price']) * row['quantity']
            
            # PnL % relative to cost basis (use absolute cost basis for shorts)
            if abs(row['cost_basis_base_ccy']) > 0:
                 row['pnl_pct'] = row['pnl_unrealized'] / abs(row['cost_basis_base_ccy'])
            else:
                row['pnl_pct'] = 0
            row['sparkline_7d'] = data.get('sparkline_7d', [])
        else:
            # Fallback if live data fails
            row['current_price'] = row['avg_entry_price']
            row['market_value'] = row['quantity'] * row['avg_entry_price']
            row['pnl_unrealized'] = 0
            row['pnl_pct'] = 0
            row['sparkline_7d'] = []
        return row

    if not df_holdings.empty:
        df_holdings = df_holdings.apply(enrich_row, axis=1)

    # 4. Calculate Live Portfolio Metrics (Required by Risk Agent)
    live_metrics = snapshot.copy() # Start with snapshot values (Cash, Daily PnL, HWM)
    
    if not df_holdings.empty:
        total_market_value = df_holdings['market_value'].sum()
        # Gross Exposure: Sum of absolute market values
        live_metrics['gross_exposure'] = df_holdings['market_value'].abs().sum()
        live_metrics['net_exposure'] = total_market_value
    else:
        total_market_value = 0
        live_metrics['gross_exposure'] = 0
        live_metrics['net_exposure'] = 0

    # Recalculate Equity (using snapshot cash + live market value)
    live_metrics['total_equity'] = live_metrics['cash_balance'] + total_market_value
    
    # Calculate Leverage
    if live_metrics['total_equity'] > 0:
        live_metrics['leverage'] = live_metrics['gross_exposure'] / live_metrics['total_equity']
    else:
        live_metrics['leverage'] = 0
        
    # Calculate Live Drawdown (using snapshot HWM and live equity)
    hwm = live_metrics['peak_equity']
    if live_metrics['total_equity'] > hwm:
        # New peak reached (UI visualization only, backend should update BQ HWM)
        live_metrics['current_drawdown'] = 0
    elif hwm > 0:
        live_metrics['current_drawdown'] = (hwm - live_metrics['total_equity']) / hwm

    # 5. Calculate Weights
    if not df_holdings.empty:
        if live_metrics['total_equity'] > 0:
            df_holdings['weight'] = df_holdings['market_value'] / live_metrics['total_equity']
        else:
            df_holdings['weight'] = 0
        df_holdings = df_holdings.sort_values(by='market_value', ascending=False, key=abs)

    return {
        "holdings_df": df_holdings,
        "metrics": live_metrics,
        "timestamp": pd.Timestamp.now(tz='UTC')
    }

# --- Transactional Functions ---

def execute_cash_operation(user_id, account_id, amount, operation_type, currency="USD"):
    """Logs a deposit or withdrawal to the BigQuery Ledger."""
    if BQ_CLIENT is None:
        st.warning("System in Mock Mode. Transaction not persisted to BigQuery.")
        # Clear cache to force UI refresh simulation
        st.cache_data.clear()
        return True

    if operation_type not in ["DEPOSIT", "WITHDRAWAL"]:
        raise ValueError("Invalid operation type.")

    # Determine quantity sign (Positive for DEPOSIT, Negative for WITHDRAWAL)
    quantity = abs(amount) if operation_type == 'DEPOSIT' else -abs(amount)
    
    # Determine cash flow impact (Positive for DEPOSIT, Negative for WITHDRAWAL)
    cash_flow = quantity

    # Simplified FX rate (assuming USD base for this example)
    fx_rate = 1.0 

    transaction = {
        "transaction_id": str(uuid.uuid4()),
        "transaction_timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "account_id": account_id,
        "transaction_type": operation_type,
        "quantity": quantity,
        "transaction_currency": currency,
        "base_currency": "USD",
        "fx_rate": fx_rate,
        "gross_value_base_ccy": quantity * fx_rate,
        "fees_base_ccy": 0.0,
        "net_cash_flow_base_ccy": cash_flow * fx_rate,
        "agent_id": "USER_INTERFACE"
    }

    try:
        table_ref = BQ_CLIENT.dataset(BQ_DATASET).table("portfolio_transactions")
        
        # In a production system, this insert MUST also trigger an update to the portfolio_status_snapshot table.
        # errors = BQ_CLIENT.insert_rows_json(table_ref, [transaction])
        logging.info(f"Simulating BQ insert: {transaction}")
        
        # Clear cache so the UI updates
        st.cache_data.clear()
        return True
    except Exception as e:
        logging.error(f"Exception during BigQuery transaction insert: {e}")
        return False