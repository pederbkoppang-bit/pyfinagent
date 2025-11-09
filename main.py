import os
import json
import argparse
import logging
import requests
from dotenv import load_dotenv
from google.cloud import storage
from google.api_core import exceptions as gcp_exceptions

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# SEC provides a JSON mapping of all tickers to CIKs
CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
# Your GCS bucket name
BUCKET_NAME = "10k-filling-data"
# SEC requires a descriptive User-Agent header with a contact email.
USER_AGENT_EMAIL = os.getenv("USER_AGENT_EMAIL")

if not USER_AGENT_EMAIL:
    logging.warning("USER_AGENT_EMAIL not set in .env file. Using placeholder.")
    # Fallback to the email you provided, but .env is the best practice
    USER_AGENT_EMAIL = "peder.bkoppang@hotmail.no"

# Set up the required headers for all SEC API calls
SEC_API_HEADERS = {
    'User-Agent': f"PyFinAgent {USER_AGENT_EMAIL}",
    'Accept-Encoding': 'gzip, deflate',
    'Host': 'data.sec.gov'
}

# --- CIK LOOKUP (Optimized) ---
_cik_map_cache = None

def get_cik_map():
    """Fetches and caches the SEC's ticker-to-CIK mapping."""
    global _cik_map_cache
    if _cik_map_cache:
        return _cik_map_cache

    logging.info("Fetching and processing SEC CIK map (this happens once)...")
    try:
        # Use a standard requests User-Agent for this public file
        headers = {'User-Agent': f"PyFinAgent {USER_AGENT_EMAIL}"}
        response = requests.get(CIK_MAP_URL, headers=headers)
        response.raise_for_status()
        company_data = response.json()
        
        # Create a more efficient lookup dictionary: {TICKER: CIK}
        _cik_map_cache = {
            item['ticker']: str(item['cik_str']).zfill(10)
            for item in company_data.values()
        }
        logging.info("Successfully processed and cached CIK map.")
        return _cik_map_cache
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch CIK map from SEC: {e}")
        raise

def get_cik(ticker: str) -> str:
    """Retrieves the CIK for a given ticker from the cached mapping."""
    cik_map = get_cik_map()
    cik = cik_map.get(ticker.upper())
    if not cik:
        raise ValueError(f"Ticker {ticker} not found in SEC CIK mapping.")
    logging.info(f"Found CIK for {ticker}: {cik}")
    return cik

# --- CORE API FUNCTIONS ---

def get_latest_10k_url(cik: str) -> (str, str):
    """Gets the direct URL to the latest 10-K HTML file from the SEC API."""
    logging.info(f"Finding latest 10-K filing for CIK:{cik}...")
    # [cite: https://www.sec.gov/files/edgar/filer-information/api-overview.pdf, Page 5]
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    
    response = requests.get(url, headers=SEC_API_HEADERS)
    response.raise_for_status()
    submissions = response.json()
    
    # Find the most recent "10-K"
    filings = submissions['filings']['recent']
    for i in range(len(filings['form'])):
        if filings['form'][i] == '10-K':
            accession_num = filings['accessionNumber'][i]
            
            # --- THIS IS THE FIX ---
            # We must use the 'primaryDocument' field from the API.
            # The bug was assuming the filename was based on the accession number.
            doc_name = filings['primaryDocument'][i]
            # --- END OF FIX ---
            
            # Format the accession number (remove dashes) for the URL
            acc_no_clean = accession_num.replace('-', '')
            
            # This is the direct link to the HTML/TXT filing
            file_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{doc_name}"
            logging.info(f"Found 10-K: {doc_name} at {file_url}")
            return file_url, doc_name

    raise FileNotFoundError(f"No 10-K found for CIK {cik}.")

def upload_from_url_to_gcs(storage_client: storage.Client, file_url: str, ticker: str, doc_name: str):
    """Streams a file from the SEC website directly to GCS without saving to disk."""
    logging.info(f"Streaming {doc_name} to Cloud Storage...")
    
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_name = f"{ticker}/{doc_name}" # e.g., "AAPL/aapl-10k-2023.htm"
        blob = bucket.blob(blob_name)
        
        # Use stream=True to avoid loading the whole file into memory
        with requests.get(file_url, headers=SEC_API_HEADERS, stream=True) as r:
            r.raise_for_status()
            # Stream the file directly to the GCS blob
            blob.upload_from_file(r.raw, content_type='text/html')
        
        logging.info(f"Successfully streamed to gs://{BUCKET_NAME}/{blob_name}")
        
    except gcp_exceptions.NotFound:
        logging.error(f"Upload failed: Bucket '{BUCKET_NAME}' not found.")
    except Exception as e:
        logging.error(f"Error streaming file: {e}", exc_info=True) # Added exc_info=True for better debugging
        raise

def process_ticker(ticker: str, storage_client: storage.Client):
    """Helper function to process a single ticker."""
    try:
        # 1. Find CIK
        cik = get_cik(ticker)

        # 2. Get latest 10-K URL
        file_url, doc_name = get_latest_10k_url(cik)

        # 3. Upload to GCS
        upload_from_url_to_gcs(
            storage_client=storage_client,
            file_url=file_url,
            ticker=ticker,
            doc_name=doc_name
        )
        
        logging.info(f"Successfully ingested {ticker} 10-K.")

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {ticker}: {e}", exc_info=True) # Added exc_info=True

def main(args):
    """Main function to orchestrate the ingestion process."""
    if not BUCKET_NAME:
        logging.error("GCS_BUCKET_NAME must be set.")
        return
        
    logging.info(f"SEC API User-Agent set to: 'PyFinAgent {USER_AGENT_EMAIL}'")
    logging.info(f"Target GCS Bucket: 'gs://{BUCKET_NAME}'")

    storage_client = storage.Client()
    
    for ticker in args.tickers:
        process_ticker(ticker.strip().upper(), storage_client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch latest 10-K filings via SEC API and upload to GCS.")
    parser.add_argument('tickers', nargs='+', help="One or more stock tickers to process (e.g., AAPL NVDA GOOG).")
    args = parser.parse_args()
    main(args)