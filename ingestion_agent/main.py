import os
import json
import re
import logging
import sys
import requests
import traceback
from datetime import datetime
from dotenv import load_dotenv
from flask import Response, stream_with_context
from google.cloud import storage
from google.api_core import exceptions as gcp_exceptions
from bs4 import BeautifulSoup
import functions_framework

# --- STRUCTURED LOGGING (for Google Cloud Logging) ---
class JsonFormatter(logging.Formatter):
    """Formats log records into a JSON string compatible with Cloud Logging."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "severity": record.levelname,
            "message": record.getMessage(),
            "json_payload": {}
        }
        if hasattr(record, 'context'):
            log_record['json_payload'] = record.context
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

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
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(JsonFormatter())
logger.addHandler(stdout_handler)


# --- CONFIGURATION ---
load_dotenv()

# SEC provides a JSON mapping of all tickers to CIKs
CIK_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
# Your GCS bucket name
# SEC requires a descriptive User-Agent header with a contact email.
BUCKET_NAME = os.getenv("BUCKET_NAME")
USER_AGENT_EMAIL = os.getenv("USER_AGENT_EMAIL")

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

    logger.info("Fetching and processing SEC CIK map (this happens once)...")
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
        logger.info("Successfully processed and cached CIK map.")
        return _cik_map_cache
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch CIK map from SEC: {e}")
        raise

def get_cik(ticker: str) -> str:
    """Retrieves the CIK for a given ticker from the cached mapping."""
    cik_map = get_cik_map()
    cik = cik_map.get(ticker.upper())
    if not cik:
        raise ValueError(f"Ticker {ticker} not found in SEC CIK mapping.")
    logger.info(f"Found CIK for {ticker}: {cik}")
    return cik

# --- CORE API FUNCTIONS ---

def get_filings_for_last_10_years(cik: str, forms: list[str]) -> list[tuple[str, str, str, str]]:
    """Gets direct URLs and info for all filings of specified forms from the last 10 years."""    
    logger.info(f"Finding all {', '.join(forms)} filings from the last 10 years for CIK:{cik}...")
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
                logger.info(f"Stopping historical data fetch; {file_info['name']} is older than 10 years.")
                break
            file_url = f"{base_url}{file_info['name']}"
            logger.info(f"Fetching historical data from {file_url}...")
            paginated_response = requests.get(file_url, headers=SEC_API_HEADERS)
            paginated_response.raise_for_status()
            all_filings_data.append(paginated_response.json())

    filing_details = []
    # Process each batch of filings (recent and all historical pages)
    for batch in all_filings_data:
        # Zip the lists together to process each filing as a complete record.
        # This is more robust than iterating with an index. Using .get with an empty list
        # as a default prevents KeyErrors if a key is missing in the API response.
        zipped_filings = zip(
            batch.get('accessionNumber', []),
            batch.get('filingDate', []),
            batch.get('form', []),
            batch.get('primaryDocument', [])
        )

        # Check if essential data is missing, which would indicate a problem with the source data.
        if not batch.get('accessionNumber'):
            logger.warning("A filing batch from SEC was missing 'accessionNumber' data. Skipping batch.")
            continue

        for acc_num, filing_date, form_type, doc_name in zipped_filings:
            filing_year = int(filing_date.split('-')[0])
            if form_type in forms and filing_year >= ten_years_ago:
                acc_no_clean = acc_num.replace('-', '')
                file_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{doc_name}"
                logger.info(f"Found {form_type} document from {filing_year}: {doc_name}")
                filing_details.append((file_url, doc_name, filing_date, form_type))

    if not filing_details:
        raise FileNotFoundError(f"No filings of type {', '.join(forms)} found for CIK {cik} in the last 10 years.")

    return filing_details


def get_company_facts(cik: str) -> dict:
    """Fetches company facts (financial data from XBRL) from the SEC API."""
    logger.info(f"Fetching company facts for CIK:{cik}...")
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=SEC_API_HEADERS)
    response.raise_for_status()
    facts = response.json()
    logger.info(f"Successfully fetched company facts for {facts.get('entityName', 'CIK ' + cik)}.")
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
    logger.info("Parsing filing into semantic sections...")
    parser = 'lxml-xml' if '<?xml' in html_content.lower()[:100] else 'lxml'
    soup = BeautifulSoup(html_content, parser)
    # Regex to find lines that start with "Item" followed by a number/letter.
    # This is a common pattern for section headers.
    # re.MULTILINE allows `^` to match the start of each line.
    # This improved regex is more flexible, handling variations in spacing and optional periods.
    # It looks for "Item" at the beginning of a line, followed by spaces, then the item number (e.g., 1A, 7).
    item_pattern = re.compile(r"^\s*item\s+[\d\w]{1,2}(?:\.|\s)", re.IGNORECASE | re.MULTILINE)

    # Extract text *after* HTML parsing, which normalizes whitespace and removes tags that could break the regex.
    # We also replace non-breaking spaces with regular spaces.
    full_text = soup.get_text(separator='\n').replace('\xa0', ' ')

    # Find all starting positions of headers
    matches = list(item_pattern.finditer(full_text))

    if not matches:
        # If no matches are found, it's possible the entire document is the content.
        # As a fallback, we'll save the whole cleaned text under a generic key.
        logger.warning("No section headers found with regex. Saving entire document content as a fallback.")
        cleaned_full_text = clean_text(full_text)
        if cleaned_full_text:
            return {"document_content": cleaned_full_text}
        else:
            logger.warning("No text content found after cleaning. Skipping.")
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
    logger.info(f"Processing {doc_name} for GCS upload...")
    bucket = storage_client.bucket(BUCKET_NAME)
    # Change the extension to .json as we are saving structured data
    blob_name = f"{ticker}/{form_type}_{filing_date}_{os.path.splitext(doc_name)[0]}.json"
    blob = bucket.blob(blob_name)

    if blob.exists():
        logger.info(f"File {blob_name} already exists in GCS. Skipping.", extra={'context': {'blob_name': blob_name}})
        return

    response = requests.get(file_url, headers=SEC_DOWNLOAD_HEADERS)
    response.raise_for_status()
    
    # Parse the HTML into structured sections
    sections = parse_filing_into_sections(response.text)

    if not sections:
        logger.warning(f"Could not parse any sections for {doc_name}. Skipping upload.")
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

    logger.info(f"Successfully uploaded structured JSON to gs://{BUCKET_NAME}/{blob_name}")

def upload_json_to_gcs(storage_client: storage.Client, data: dict, ticker: str):
    """Uploads a JSON object to GCS."""
    cik = data.get('cik')
    entity_name = data.get('entityName', '').replace(' ', '_')
    blob_name = f"{ticker}/company_facts_{cik}.json"
    logger.info(f"Uploading company facts for {entity_name} to GCS...")

    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    if blob.exists():
        logger.info(f"File {blob_name} already exists in GCS. Skipping.")
        return

    blob.upload_from_string(json.dumps(data, indent=2), content_type='application/json')
    logger.info(f"Successfully uploaded to gs://{BUCKET_NAME}/{blob_name}")


def process_ticker(ticker: str, storage_client: storage.Client, forms: list[str]):
    """Helper function to process a single ticker."""
    try:
        # 1. Find CIK
        cik = get_cik(ticker)

        # 2. Get the latest 10-K/10-Q filing and upload the document
        all_filings = get_filings_for_last_10_years(cik, forms)
        if not all_filings:
            logger.warning(f"No filings found for {ticker} within the specified parameters.")
            return
        for file_url, doc_name, filing_date, form_type in all_filings:
            upload_from_url_to_gcs(storage_client, file_url, ticker, doc_name, filing_date, form_type) # type: ignore

        # 3. Get company facts as structured JSON data (only needs to be done once per ticker)
        facts = get_company_facts(cik) # type: ignore

        # 4. Upload the JSON data to GCS
        upload_json_to_gcs(storage_client, facts, ticker)

        logger.info(f"Successfully processed SEC filings and facts for {ticker}.")
    except gcp_exceptions.NotFound:
        logger.error(f"Upload failed for {ticker}: Bucket '{BUCKET_NAME}' not found.")
        raise # Re-raise to signal a server error
    except (requests.exceptions.RequestException, FileNotFoundError, ValueError) as e:
        logger.error(f"An unexpected error occurred while processing {ticker}: {e}", exc_info=True)
        raise # Re-raise to signal a server error

@functions_framework.http
def ingestion_agent_http(request):
    """HTTP Cloud Function entry point."""
    def generate_logs():
        # This handler will capture logs for this specific request
        stream_handler = StreamHandler()
        # Use a formatter that includes the severity level for client-side parsing.
        stream_handler.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
        logger.addHandler(stream_handler)

        try:
            # Critical configuration check
            if not BUCKET_NAME or not USER_AGENT_EMAIL:
                raise ValueError("Server configuration error: BUCKET_NAME and/or USER_AGENT_EMAIL environment variables are not set.")

            if request.method != 'POST':
                raise ValueError("Method Not Allowed")

            request_json = request.get_json(silent=True)
            if not request_json or 'ticker' not in request_json:
                raise ValueError("Invalid request: 'ticker' is required in JSON body.")

            ticker = request_json['ticker'].strip().upper()
            run_id = request_json.get('run_id')
            context = {"ticker": ticker, "agent": "ingestion-agent", "run_id": run_id}

            logger.info("Ingestion agent triggered.", extra={'context': context})
            # Yield any logs that have been captured so far
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()

            forms_to_fetch = ['10-K', '10-Q']
            storage_client = storage.Client()

            # We need to manually iterate and yield logs during the process
            # This requires adapting the process_ticker logic slightly or just letting it run
            # and flushing logs periodically if it were a long process.
            # For now, we'll just run it and then flush.
            process_ticker(ticker, storage_client, forms_to_fetch)

            logger.info("Ingestion process completed successfully.", extra={'context': context})
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()

        except Exception as e:
            error_message = f"Ingestion agent failed: {str(e)}\n{traceback.format_exc()}"
            logger.critical(error_message, exc_info=True)
            for log in stream_handler.queue: yield log
            yield error_message # Ensure the final error is sent
        finally:
            # IMPORTANT: Remove the handler to avoid adding it again on the next invocation
            logger.removeHandler(stream_handler)

    return Response(stream_with_context(generate_logs()), mimetype='text/plain')