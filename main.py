import os
import json
import argparse
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

import requests
from dotenv import load_dotenv
from google.cloud import storage
from google.api_core import exceptions as gcp_exceptions
from sec_edgar_downloader import Downloader

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file at the start
load_dotenv()

# SEC provides a JSON mapping of all tickers to CIKs
CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
BUCKET_NAME = "10k-filling-data"

# --- CIK LOOKUP ---
def _fetch_and_cache_cik_map() -> Dict[str, str]:
    """
    Fetches the SEC's ticker-to-CIK mapping and caches it as a simplified dictionary.
    This is an internal function.
    """
    logging.info("Fetching and processing SEC CIK map...")

    # SEC requires a descriptive User-Agent header with a contact email.
    user_agent = os.getenv("USER_AGENT_EMAIL")
    if not user_agent:
        raise ValueError("USER_AGENT_EMAIL environment variable is not set. This is required by the SEC.")
    headers = {'User-Agent': f"PyFinAgent {user_agent}"}
    try:
        response = requests.get(CIK_MAP_URL, headers=headers)
        response.raise_for_status()
        company_data = response.json()
        
        # Create a more efficient lookup dictionary: {TICKER: CIK}
        # CIK must be 10 digits, padded with leading zeros.
        cik_map = {
            item['ticker']: str(item['cik_str']).zfill(10)
            for item in company_data.values()
        }
        logging.info("Successfully processed CIK map.")
        return cik_map
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch CIK map from SEC: {e}")
        raise

# Create a global cache for the CIK map to avoid repeated downloads
CIK_LOOKUP_CACHE = _fetch_and_cache_cik_map()

def get_cik(ticker: str) -> str:
    """Retrieves the CIK for a given ticker from the cached mapping."""
    cik = CIK_LOOKUP_CACHE.get(ticker.upper())
    if not cik:
        raise ValueError(f"Ticker {ticker} not found in SEC CIK mapping.")
    logging.info(f"Found CIK for {ticker}: {cik}")
    return cik

# --- CORE FUNCTIONS ---
def download_latest_10k(ticker: str, cik: str, user_agent: str) -> Optional[Dict[str, any]]:
    """Downloads the latest 10-K filing to a local temporary directory."""
    logging.info(f"Downloading latest 10-K for {ticker} (CIK: {cik})...")
    
    with tempfile.TemporaryDirectory() as temp_dir: # This directory is automatically cleaned up
        dl = Downloader("PyFinAgent", user_agent, Path(temp_dir) / ticker)
        
        # Get the single most recent 10-K filing
        dl.get("10-K", cik, limit=1)
        
        # Find the path to the downloaded file more robustly
        try:
            filing_dir = Path(temp_dir) / ticker / "sec-edgar-filings" / cik / "10-K"
            # There should be only one folder since limit=1
            filing_subfolder = next(filing_dir.iterdir())
            
            # The subfolder name is like '0001234567-23-000001', where the date isn't present.
            # The actual filing date is in the 'full-submission.txt' header.
            filing_date = "unknown-date" # Default value
            
            # Find the primary TXT document.
            txt_file = filing_subfolder / "full-submission.txt"
            if txt_file.exists():
                logging.info(f"Found filing document: {txt_file.name}")
                file_content_bytes = txt_file.read_bytes()
                
                # Extract the 'FILING-DATE' from the text file content
                file_content_str = file_content_bytes.decode('utf-8', errors='ignore')
                for line in file_content_str.splitlines():
                    if line.startswith("FILING DATE:"):
                        filing_date = line.split(":")[1].strip()
                        break

                return {"content": file_content_bytes, "path": txt_file, "filing_date": filing_date}
            
            logging.warning(f"No 'full-submission.txt' file found for {ticker}.")
            return None
            
        except (StopIteration, FileNotFoundError) as e:
            logging.error(f"Error finding downloaded file for {ticker}: {e}")
            return None

def upload_to_gcs(file_content: bytes, file_name: str, bucket_name: str, destination_blob_name: str):
    """Uploads a local file to Google Cloud Storage."""
    logging.info(f"Uploading {file_name} to gs://{bucket_name}/{destination_blob_name}...")
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        
        blob.upload_from_string(file_content)
        logging.info("Upload complete.")
    except gcp_exceptions.NotFound:
        logging.error(f"Upload failed: Bucket '{bucket_name}' not found.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during upload: {e}")

def process_ticker(ticker: str, user_agent: str, bucket_name: str):
    """Helper function to process a single ticker."""
    try:
        # 1. Find CIK
        cik = get_cik(ticker)

        # 2. Download latest 10-K
        filing_data = download_latest_10k(ticker, cik, user_agent)
        if not filing_data:
            logging.error(f"Failed to download filing for {ticker}.")
            return

        # 3. Upload to GCS
        file_path = filing_data["path"] # This is the local path, not used for GCS name directly
        filing_date = filing_data["filing_date"]
        destination_name = f"filings/{ticker}_10K_{filing_date}.txt"
        upload_to_gcs(
            file_content=filing_data["content"],
            file_name=file_path.name,
            bucket_name=bucket_name,
            destination_blob_name=destination_name)

        logging.info(f"Successfully ingested {ticker} 10-K to gs://{bucket_name}/{destination_name}")

    except Exception as e:
        logging.error(f"An unexpected error occurred while processing {ticker}: {e}")

def main(args):
    """Main function to orchestrate the ingestion process."""
    user_agent = os.getenv("USER_AGENT_EMAIL")
    bucket_name = BUCKET_NAME # Always use the constant defined in the script

    if not user_agent:
        logging.error("USER_AGENT_EMAIL must be set in the .env file.")
        return

    for ticker in args.tickers:
        process_ticker(ticker.strip().upper(), user_agent, bucket_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch latest 10-K filings and upload them to GCS.")
    parser.add_argument('tickers', nargs='+', help="One or more stock tickers to process (e.g., AAPL NVDA GOOG).")
    args = parser.parse_args()
    main(args)
