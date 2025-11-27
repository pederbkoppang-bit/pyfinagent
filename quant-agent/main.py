import functions_framework
import requests
import json
import sys
import traceback
import yfinance as yf
import os
import logging
from dotenv import load_dotenv
from google.cloud import storage
from flask import Response, stream_with_context

# --- CONFIG ---
# A handler that captures logs to be streamed over HTTP
class StreamHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.queue = []
    def emit(self, record):
        self.queue.append(self.format(record) + '\n')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
# Standard handler for Cloud Logging
# This one will log to the console of the Cloud Function for debugging.
stdout_handler = logging.StreamHandler(sys.stdout)
# Use a simple formatter as the severity is not needed for the UI log.
stdout_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(stdout_handler)

load_dotenv()

# Use the email you provided for the SEC User-Agent
YOUR_EMAIL = os.getenv("USER_AGENT_EMAIL", "peder.bkoppang@hotmail.no")
# [FIX] Corrected the URL to be a simple string
CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json" 
SEC_API_HEADERS = {
    'User-Agent': f"PyFinAgent {YOUR_EMAIL}",
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'data.sec.gov'
}

# --- GCP Clients (Initialized globally) ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME") # e.g., 'pyfinagent-filings'
storage_client = None # Initialize as None

_cik_map_cache = None

# --- CIK LOOKUP (Helper) ---
def get_cik_map():
    """Fetches and caches the SEC Ticker->CIK mapping."""
    global _cik_map_cache
    # If the cache is populated and is a non-empty dictionary, return it.
    if _cik_map_cache and isinstance(_cik_map_cache, dict) and len(_cik_map_cache) > 0:
        return _cik_map_cache

    logging.info("Fetching and caching SEC CIK map...")
    try:
        headers = {'User-Agent': f"PyFinAgent {YOUR_EMAIL}"}
        response = requests.get(CIK_MAP_URL, headers=headers)
        response.raise_for_status()
        company_data = response.json()
        _cik_map_cache = {
            item['ticker']: str(item['cik_str']).zfill(10)
            for item in company_data.values()
        }
        return _cik_map_cache
    except Exception as e:
        logging.error(f"Failed to fetch CIK map: {e}")
        # Do not raise an exception here during startup.
        # Instead, return None and let the calling function handle the error.
        return None

def get_cik(ticker: str) -> str:
    """Gets a 10-digit CIK string for a given ticker."""
    cik_map = get_cik_map()
    cik = cik_map.get(ticker.upper())
    if not cik:
        raise ValueError(f"Ticker {ticker} not found in SEC CIK mapping.")
    return cik

# --- SEC API (Helper) ---
def get_latest_financial_fact(facts, fact_name, unit):
    """
    Retrieves the most recent fact from the SEC /companyfacts/ JSON.
    [cite: Comprehensive Financial Analysis Template.pdf.pdf, Part 1.1]
    """
    try:
        # Navigate the JSON: facts -> us-gaap -> FactName -> units -> USD -> [list of facts]
        fact_data = facts['us-gaap'][fact_name]['units'][unit]
        latest_fact = fact_data[-1] # Get the most recent fact
        return {
            "value": latest_fact['val'],
            "period": latest_fact.get('fp', 'N/A'),
            "filed": latest_fact['filed']
        }
    except (KeyError, IndexError, TypeError):
        logging.warning(f"Could not find fact '{fact_name}' with unit '{unit}'")
        return None

def get_company_facts(cik_10_digit: str) -> dict:
    """
    Retrieves company facts. First, it tries to load from GCS bucket where
    ingestion-agent should have placed it. If it fails, it falls back to
    fetching directly from the SEC API.
    """
    # Try to fetch from GCS first
    global storage_client
    if storage_client is None:
        logging.info("Initializing Google Cloud Storage client...")
        storage_client = storage.Client()

    if GCS_BUCKET_NAME:
        try:
            gcs_path = f"sec-filings/CIK{cik_10_digit}/companyfacts.json"
            gcs_path = f"sec-filings/CIK{cik_10_digit}/companyfacts.json" # Corrected path
            logging.info(f"Attempting to load company facts from GCS: gs://{GCS_BUCKET_NAME}/{gcs_path}")
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(gcs_path)
            if blob.exists():
                facts_json = blob.download_as_string()
                logging.info("Successfully loaded company facts from GCS.")
                return json.loads(facts_json)
            else:
                logging.warning("companyfacts.json not found in GCS. Falling back to SEC API.")
        except Exception as e:
            logging.error(f"Failed to load from GCS, falling back to SEC API. Error: {e}", exc_info=True)

    # Fallback to SEC API
    logging.info(f"Fetching company facts directly from SEC API for CIK{cik_10_digit}.")
    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_10_digit}.json"
    response = requests.get(facts_url, headers=SEC_API_HEADERS)
    response.raise_for_status()
    facts = response.json()
    logging.info("Successfully fetched company facts from SEC API.")
    return facts

def get_historical_prices(ticker_str: str, cik_10_digit: str) -> str:
    """
    Retrieves 5 years of historical stock prices. First, it tries to load from a
    pre-processed JSON file in the GCS bucket. If it fails, it falls back to
    fetching live data from yfinance.
    """
    # Try to fetch from GCS first
    global storage_client
    if storage_client is None:
        logging.info("Initializing Google Cloud Storage client...")
        storage_client = storage.Client()

    if GCS_BUCKET_NAME:
        try:
            # Assuming ingestion-agent saves prices at this conventional path
            gcs_path = f"sec-filings/CIK{cik_10_digit}/historical_prices.json"
            logging.info(f"Attempting to load historical prices from GCS: gs://{GCS_BUCKET_NAME}/{gcs_path}")
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(gcs_path)
            if blob.exists():
                prices_json_string = blob.download_as_string().decode('utf-8')
                logging.info("Successfully loaded historical prices from GCS.")
                return prices_json_string
            else:
                logging.warning("historical_prices.json not found in GCS. Falling back to yfinance.")
        except Exception as e:
            logging.error(f"Failed to load prices from GCS, falling back to yfinance. Error: {e}", exc_info=True)

    # Fallback to yfinance API
    logging.info(f"Fetching 5-year historical prices from yfinance for {ticker_str}.")
    stock = yf.Ticker(ticker_str)
    hist_data = stock.history(period="5y").reset_index()
    hist_data['Date'] = hist_data['Date'].dt.strftime('%Y-%m-%d')
    return hist_data.to_json(orient="split", date_format="iso")

# --- AGENT MAIN FUNCTION ---
@functions_framework.http
def quant_agent(request):
    """
    This Cloud Function is the QuantAgent. It's triggered by an HTTP request
    and returns structured financial data from both SEC API and yfinance.
    [cite: Product Canvas:PyFinAgent.md]
    """
    def generate_logs_and_data():
        stream_handler = StreamHandler()
        # This formatter will be used for the logs streamed back to the main app.
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

        ticker_str = request.args.get('ticker')
        if not ticker_str:
            yield "ERROR: No ticker provided"
            return

        try:
            # --- 1. yfinance: Get Market Data (Price, Ratios) ---
            logging.info(f"Fetching yfinance market data for {ticker_str}...")
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()
            stock = yf.Ticker(ticker_str)
            info = stock.info
            market_price = info.get('currentPrice') or info.get('regularMarketPrice')
            pe_ratio = info.get('trailingPE')
            ps_ratio = info.get('priceToSalesTrailing12Months')
            market_cap = info.get('marketCap')

            # --- 2. SEC API: Get Fundamental Facts (Revenue, EPS) ---
            logging.info(f"Fetching SEC fundamental facts for {ticker_str}...")
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()
            cik_10_digit = get_cik(ticker_str.upper())
            facts = get_company_facts(cik_10_digit)
            historical_prices_json = get_historical_prices(ticker_str, cik_10_digit)

            latest_revenue = get_latest_financial_fact(facts, 'Revenues', 'USD')
            latest_net_income = get_latest_financial_fact(facts, 'NetIncomeLoss', 'USD')
            latest_eps_diluted = get_latest_financial_fact(facts, 'EarningsPerShareDiluted', 'USD/shares')

            # --- 3. Compile Final Agent Report ---
            logging.info("Compiling final quantitative report...")
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()
            report = {
                "ticker": ticker_str.upper(), # Ensure ticker is always uppercase
                "cik": cik_10_digit,
                "company_name": facts.get('entityName', 'N/A'),
                "part_1_financials": {
                    "source": "SEC EDGAR API (/api/xbrl/companyfacts/)",
                    "latest_revenue": latest_revenue,
                    "latest_net_income": latest_net_income
                },
                "part_5_valuation": {
                    "source": "yfinance & SEC API",
                    "market_price": market_price,
                    "market_cap": market_cap,
                    "pe_ratio": pe_ratio,
                    "ps_ratio": ps_ratio,
                    "latest_eps_diluted": latest_eps_diluted,
                    "historical_prices": historical_prices_json
                }
            }
            
            # Yield the final JSON report with a special prefix
            yield f"FINAL_JSON:{json.dumps(report)}"

        except Exception as e:
            error_message = f"QuantAgent failed for {ticker_str}: {str(e)}\n{traceback.format_exc()}"
            logging.critical(error_message, exc_info=True)
            for log in stream_handler.queue: yield log
            yield f"ERROR: {error_message}"
        finally:
            # IMPORTANT: Remove the handler to avoid adding it again on the next invocation
            logger.removeHandler(stream_handler)

    return Response(stream_with_context(generate_logs_and_data()), mimetype='text/plain')