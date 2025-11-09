import os
import json
import re
import argparse
import logging
import requests
from datetime import datetime
from dotenv import load_dotenv
from google.cloud import storage
from google.api_core import exceptions as gcp_exceptions
from bs4 import BeautifulSoup

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

# A separate header for downloading from www.sec.gov, which doesn't need the Host header.
SEC_DOWNLOAD_HEADERS = {
    'User-Agent': f"PyFinAgent {USER_AGENT_EMAIL}",
    'Accept-Encoding': 'gzip, deflate',
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

def get_filings_for_last_10_years(cik: str, forms: list[str]) -> list[tuple[str, str, str, str]]:
    """Gets direct URLs and info for all filings of specified forms from the last 10 years."""
    logging.info(f"Finding all {', '.join(forms)} filings from the last 10 years for CIK:{cik}...")
    base_url = "https://data.sec.gov/submissions/"
    current_year = datetime.now().year
    ten_years_ago = current_year - 10

    initial_url = f"{base_url}CIK{cik}.json"

    response = requests.get(initial_url, headers=SEC_API_HEADERS)
    response.raise_for_status()
    submissions = response.json()

    all_filings_data = [submissions['filings']['recent']]

    # The 'files' array contains metadata for older, paginated filing data.
    if 'files' in submissions['filings']:
        for file_info in submissions['filings']['files']:
            # The 'filingTo' date tells us the latest date in a historical data file.
            # If this date is before our 10-year cutoff, we can stop fetching.
            filing_to_year = int(file_info['filingTo'].split('-')[0])
            if filing_to_year < ten_years_ago:
                logging.info(f"Stopping historical data fetch; {file_info['name']} is older than 10 years.")
                break
            file_url = f"{base_url}{file_info['name']}"
            logging.info(f"Fetching historical data from {file_url}...")
            paginated_response = requests.get(file_url, headers=SEC_API_HEADERS)
            paginated_response.raise_for_status()
            all_filings_data.append(paginated_response.json())

    filing_details = []
    # Process each batch of filings (recent and all historical pages)
    for filing_batch in all_filings_data:
        for i, form_type in enumerate(filing_batch['form']):
            filing_date = filing_batch['filingDate'][i]
            filing_year = int(filing_date.split('-')[0])

            if form_type in forms and filing_year >= ten_years_ago:
                accession_num = filing_batch['accessionNumber'][i]
                doc_name = filing_batch['primaryDocument'][i]
                acc_no_clean = accession_num.replace('-', '')

                file_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{doc_name}"
                logging.info(f"Found {form_type} document from {filing_year}: {doc_name}")
                filing_details.append((file_url, doc_name, filing_date, form_type))

    if not filing_details:
        raise FileNotFoundError(f"No filings of type {', '.join(forms)} found for CIK {cik} in the last 10 years.")

    return filing_details


def get_company_facts(cik: str) -> dict:
    """Fetches company facts (financial data from XBRL) from the SEC API."""
    logging.info(f"Fetching company facts for CIK:{cik}...")
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=SEC_API_HEADERS)
    response.raise_for_status()
    facts = response.json()
    logging.info(f"Successfully fetched company facts for {facts.get('entityName', 'CIK ' + cik)}.")
    return facts


def clean_text(text: str) -> str:
    """Cleans up extracted text by removing extra whitespace and newlines."""
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)
    return cleaned_text

def parse_filing_into_sections(html_content: str) -> dict:
    """
    Parses an SEC filing's HTML to extract and clean text from major sections (Items).
    """
    logging.info("Parsing filing into semantic sections...")
    parser = 'lxml-xml' if '<?xml' in html_content.lower()[:100] else 'lxml'
    soup = BeautifulSoup(html_content, parser)

    # Extract text, using a separator to preserve some structure.
    full_text = soup.get_text(separator='\n')

    # Regex to find lines that start with "Item" followed by a number/letter.
    # This is a common pattern for section headers.
    # re.MULTILINE allows `^` to match the start of each line.
    item_pattern = re.compile(r"^\s*item\s+[\d\w]{1,2}\.?", re.IGNORECASE | re.MULTILINE)

    # Find all starting positions of headers
    matches = list(item_pattern.finditer(full_text))

    if not matches:
        logging.warning("No section headers found with regex. Saving debug file.")
        # Write the text content to a file for local debugging
        with open("debug_filing_text.txt", "w", encoding="utf-8") as f:
            f.write(full_text)
        return {}

    sections = {}
    # Extract content between each match
    for i, match in enumerate(matches):
        start_pos = match.end()
        # The end position is the start of the next match, or the end of the file
        end_pos = matches[i+1].start() if i + 1 < len(matches) else len(full_text)

        # Get the header text from the match itself
        header_text = match.group(0).strip()
        # Get the content between this header and the next
        content_text = full_text[start_pos:end_pos]

        section_key = header_text.lower().replace('.', '').replace(':', '').replace(' ', '_')
        cleaned_content = clean_text(content_text)

        if cleaned_content:
            sections[section_key] = cleaned_content

    return sections

def upload_from_url_to_gcs(storage_client: storage.Client, file_url: str, ticker: str, doc_name: str, filing_date: str, form_type: str):
    """Downloads, parses, and uploads a filing as a structured JSON to GCS."""
    logging.info(f"Processing {doc_name} for GCS upload...")
    bucket = storage_client.bucket(BUCKET_NAME)
    # Change the extension to .json as we are saving structured data
    blob_name = f"{ticker}/{form_type}_{filing_date}_{os.path.splitext(doc_name)[0]}.json"
    blob = bucket.blob(blob_name)

    if blob.exists():
        logging.info(f"File {blob_name} already exists in GCS. Skipping.")
        return

    response = requests.get(file_url, headers=SEC_DOWNLOAD_HEADERS)
    response.raise_for_status()
    
    # Parse the HTML into structured sections
    sections = parse_filing_into_sections(response.text)

    if not sections:
        logging.warning(f"Could not parse any sections for {doc_name}. Skipping upload.")
        return

    # Create the final JSON object with metadata
    output_data = {
        "ticker": ticker,
        "form_type": form_type,
        "filing_date": filing_date,
        "document_name": doc_name,
        "source_url": file_url,
        "sections": sections
    }
    
    # Upload the structured data as a JSON file
    blob.upload_from_string(json.dumps(output_data, indent=2), content_type='application/json')

    logging.info(f"Successfully uploaded structured JSON to gs://{BUCKET_NAME}/{blob_name}")

def upload_json_to_gcs(storage_client: storage.Client, data: dict, ticker: str):
    """Uploads a JSON object to GCS."""
    cik = data.get('cik')
    entity_name = data.get('entityName', '').replace(' ', '_')
    blob_name = f"{ticker}/company_facts_{cik}.json"
    logging.info(f"Uploading company facts for {entity_name} to GCS...")

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    if blob.exists():
        logging.info(f"File {blob_name} already exists in GCS. Skipping.")
        return

    blob.upload_from_string(json.dumps(data, indent=2), content_type='application/json')
    logging.info(f"Successfully uploaded to gs://{BUCKET_NAME}/{blob_name}")

def process_ticker(ticker: str, storage_client: storage.Client, forms: list[str]):
    """Helper function to process a single ticker."""
    try:
        # 1. Find CIK
        cik = get_cik(ticker)

        # 2. Get the latest 10-K/10-Q filing and upload the document
        all_filings = get_filings_for_last_10_years(cik, forms)
        for file_url, doc_name, filing_date, form_type in all_filings:
            upload_from_url_to_gcs(storage_client, file_url, ticker, doc_name, filing_date, form_type)

        # 3. Get company facts as structured JSON data (only needs to be done once per ticker)
        facts = get_company_facts(cik)

        # 4. Upload the JSON data to GCS
        upload_json_to_gcs(storage_client, facts, ticker)
        logging.info(f"Successfully processed {len(all_filings)} filings and company facts for {ticker}.")
    except gcp_exceptions.NotFound:
        logging.error(f"Upload failed for {ticker}: Bucket '{BUCKET_NAME}' not found.")
    except (requests.exceptions.RequestException, FileNotFoundError, ValueError) as e:
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
        process_ticker(ticker.strip().upper(), storage_client, args.forms)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch latest 10-K filings via SEC API and upload to GCS.")
    parser.add_argument('tickers', nargs='+', help="One or more stock tickers to process (e.g., AAPL NVDA GOOG).")
    parser.add_argument('--forms', nargs='+', default=['10-K', '10-Q'],
                        help="A list of SEC form types to download (e.g., 10-K 10-Q 8-K).")
    args = parser.parse_args()
    main(args)