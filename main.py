import os
import json
import argparse
import logging
import requests
import pathlib
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

def get_latest_10k_info(cik: str) -> (str, str, str):
    """Gets the direct URL to the latest 10-K HTML file from the SEC API."""
    logging.info(f"Finding latest 10-K filing for CIK:{cik}...")
    # SEC API documentation: https://www.sec.gov/os/accessing-edgar-data
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    
    response = requests.get(url, headers=SEC_API_HEADERS)
    response.raise_for_status()
    submissions = response.json()
    
    # Find the most recent "10-K"
    filings = submissions['filings']['recent']
    for i in range(len(filings['form'])):
        if filings['form'][i] == '10-K':
            accession_num = filings['accessionNumber'][i]
            doc_name = filings['primaryDocument'][i]
            filing_date = filings['filingDate'][i]
            
            # Format the accession number (remove dashes) for the URL
            acc_no_clean = accession_num.replace('-', '')
            
            # The complete submission text file is consistently named after the
            # accession number with a .txt extension. This is more reliable
            # than using the 'primaryDocument' field. Both the directory and
            # the filename use the accession number without dashes.
            # Ref: https://www.sec.gov/os/accessing-edgar-data
            full_submission_doc_name = f"{acc_no_clean}.txt"
            file_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{full_submission_doc_name}"

            logging.info(f"Found 10-K filed on {filing_date}: {doc_name} (using full submission file: {full_submission_doc_name})")
            return file_url, full_submission_doc_name, filing_date

    raise FileNotFoundError(f"No 10-K found for CIK {cik}.")

def upload_from_url_to_gcs(storage_client: storage.Client, file_url: str, ticker: str, filing_date: str, doc_name: str):
    """Streams a file from the SEC website directly to GCS without saving to disk."""
    
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        
        # Construct a more descriptive blob name
        file_extension = "".join(pathlib.Path(doc_name).suffixes)
        blob_name = f"{ticker}/10-K_{filing_date}{file_extension}"

        blob = bucket.blob(blob_name)

        # Check if the blob already exists to avoid re-uploading
        if blob.exists():
            logging.info(f"File {blob_name} already exists in gs://{BUCKET_NAME}. Skipping.")
            return

        logging.info(f"Streaming {doc_name} to gs://{BUCKET_NAME}/{blob_name}...")
        
        # Use stream=True to avoid loading the whole file into memory
        with requests.get(file_url, headers=SEC_API_HEADERS, stream=True) as r:
            r.raise_for_status()
            content_type = r.headers.get('Content-Type', 'text/html')
            # Stream the file directly to the GCS blob
            blob.upload_from_file(r.raw, content_type=content_type)
        
        logging.info(f"Successfully uploaded to gs://{BUCKET_NAME}/{blob_name}")
        
    except gcp_exceptions.NotFound:
        logging.error(f"Upload failed: Bucket '{BUCKET_NAME}' not found.")
    except Exception as e:
        logging.error(f"Error streaming file: {e}")
        raise

def process_ticker(ticker: str, storage_client: storage.Client):
    """Helper function to process a single ticker."""
    try:
        # 1. Find CIK
        cik = get_cik(ticker)

        # 2. Get latest 10-K URL
        file_url, doc_name, filing_date = get_latest_10k_info(cik)

        # 3. Upload to GCS
        upload_from_url_to_gcs(
            storage_client=storage_client,
            file_url=file_url,
            ticker=ticker,
            filing_date=filing_date,
            doc_name=doc_name
        )
        
        logging.info(f"Successfully ingested {ticker} 10-K.")

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {ticker}: {e}")

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