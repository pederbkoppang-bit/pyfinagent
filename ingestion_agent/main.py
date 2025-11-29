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
    logger.info("Parsing filing into semantic sections (Items)...")
    parser = 'lxml-xml' if '<?xml' in html_content.lower()[:100] else 'lxml'
    soup = BeautifulSoup(html_content, parser)

    # 0. Pre-emptively remove style-only tags that add no semantic meaning
    # and can interfere with text extraction.
    for tag_name in ['font', 'b', 'i', 'u', 'strong', 'em']:
        for tag in soup.find_all(tag_name):
            # Replaces the tag with its contents
            tag.unwrap()

    # 1. Find all tables, convert them to Markdown, and replace them in the soup.
    # This preserves the table structure before converting the whole document to text.
    for table in soup.find_all('table'):
        # Parse the table into a list of lists
        table_data = []
        for row in table.find_all('tr'):
            row_data = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
            if any(row_data): # Only add rows that have some content
                table_data.append(row_data)

        if not table_data:
            continue

        # Normalize table: ensure all rows have the same number of columns
        # This handles malformed tables that use colspan or have inconsistent cell counts.
        max_cols = max(len(row) for row in table_data) if table_data else 0
        normalized_table_data = [row + [''] * (max_cols - len(row)) for row in table_data]

        if not normalized_table_data or not normalized_table_data[0]:
            continue

        # Convert the list of lists to a Markdown table string
        markdown_table = "\n\n" # Add spacing around the table
        # Header
        markdown_table += "| " + " | ".join(map(str, normalized_table_data[0])) + " |\n"
        # Separator
        markdown_table += "| " + " | ".join(['---'] * len(normalized_table_data[0])) + " |\n"
        # Body
        for row in normalized_table_data[1:]:
            markdown_table += "| " + " | ".join(map(str, row)) + " |\n"
        markdown_table += "\n"

        # Replace the table tag with a new tag containing the Markdown text
        table.replace_with(soup.new_string(markdown_table))

    # 2. Pre-process text: remove non-breaking spaces and normalize
    full_text = soup.get_text(separator='\n').replace('\xa0', ' ')

    # 3. Find all section headers. This is a more robust pattern that handles cases
    # where the title is on the same line as "Item X" or on a subsequent line,
    # which is common in SEC filings.
    # It also now looks for an optional "Part X" prefix.
    #
    # Breakdown of the regex:
    #   ^\s*                      - Start of a line with optional whitespace.
    #   (?:part\s+[ivx]+[.,]?\s*)? - Optional non-capturing group for "Part I", "Part II.", etc.
    #   (item\s+[\d\w]{1,3}(?:\.|\s)?) - Group 1: Captures "Item" followed by its number/letter combo (e.g., "Item 1A", "Item 7.").
    #   \s*                       - Optional whitespace after the item number.
    #   (?:-|\u2013|\u2014)?\s*    - Optional dash or em-dash, followed by optional space.
    #   ([a-z\s,()&’'\"\d]{5,150})? - Group 2 (optional): Captures the section title if it's on the same line.
    #   \s*$                      - Must end with optional whitespace and the end of the line OR
    #   |                         - OR
    #   (?=\n\s*[a-z].+)          - A positive lookahead to assert that the next line starts with the title.
    #                             This allows us to match "Item X" on one line and the title on the next.
    item_pattern = re.compile(
        r"^\s*(?:part\s+[ivx]+[.,]?\s*)?(item\s+[\d\w]{1,3}(?:\.|\s)?)\s*(?:-|\u2013|\u2014)?\s*([a-z\s,()&’'\"\d]{5,150})?\s*$",
        re.IGNORECASE | re.MULTILINE
    )
    matches = list(item_pattern.finditer(full_text))

    # Define a pattern to find the end of the main content (e.g., "SIGNATURES", "EXHIBIT INDEX")
    end_of_document_pattern = re.compile(r"^\s*(SIGNATURES|EXHIBIT INDEX|Consolidated Financial Statements)\s*$", re.IGNORECASE | re.MULTILINE)

    if not matches:
        logger.warning("No 'Item X' section headers found with regex. Attempting fallback parsing. If this also fails, the entire document content will be saved.")
        cleaned_full_text = clean_text(full_text)
        if cleaned_full_text:
            return {"document_content": cleaned_full_text}
        else:
            logger.warning("No text content found after cleaning. Skipping file.")
            return {}

    sections = {}
    # 4. Extract content between each header
    for i, current_match in enumerate(matches):
        # Standardize the key (e.g., "item 1a", "Item 1A." -> "item1a")
        item_part = current_match.group(1).strip()
        current_key = re.sub(r'[^a-z0-9]', '', item_part.lower().replace(' ', ''))
        if not current_key.startswith('item'):
            continue # Should not happen with the regex, but as a safeguard.

        start_pos = current_match.end()

        # Default end position is the end of the document.
        # We look for a "sentinel" string like "SIGNATURES" to get a cleaner end point.
        end_of_doc_match = end_of_document_pattern.search(full_text)
        end_pos = end_of_doc_match.start() if end_of_doc_match else len(full_text)


        # Find the correct end position. The end is the start of the *next* item,
        # but we must also handle sub-items (e.g., Item 1A should end before Item 1B).
        for next_match in matches[i+1:]:
            next_key = re.sub(r'[^a-z0-9]', '', next_match.group(1).lower().replace(' ', ''))
            # If the next item is a sub-item of the current one (e.g. current='item1', next='item1a'), skip it.
            if next_key.startswith(current_key) and len(next_key) > len(current_key):
                continue
            end_pos = next_match.start()
            break # Found the correct next major item, so stop searching.

        # Get the section title and content
        header_title = (current_match.group(2) or "").strip()

        content_text = full_text[start_pos:end_pos]
        cleaned_content = clean_text(content_text)

        # If the title was not on the same line, it's likely the first line of the content.
        if not header_title and cleaned_content:
            first_line = cleaned_content.split('\n')[0]
            # A plausible title is short, capitalized, and doesn't end with a period.
            if 5 < len(first_line) < 150 and first_line.istitle() and not first_line.endswith('.'):
                header_title = first_line
                # Remove the title from the content itself
                cleaned_content = cleaned_content[len(header_title):].lstrip()


        # 5. Post-process content to remove noise
        # Remove the section title if it's repeated at the start of the content
        if cleaned_content.lower().startswith(header_title.lower()):
            cleaned_content = cleaned_content[len(header_title):].lstrip()

        # Remove common noise like "Table of Contents" and page numbers
        lines = []
        for line in cleaned_content.split('\n'):
            # Ignore lines that are just "Table of Contents" or page number indicators
            if "table of contents" in line.lower() or re.match(r'^\s*F-\d+\s*$', line, re.IGNORECASE):
                continue
            # Remove trailing page numbers like "| 71"
            line = re.sub(r'\s*\|\s*\d+\s*$', '', line)
            lines.append(line)
        cleaned_content = '\n'.join(lines).strip()

        if cleaned_content:
            # 6. Merge content if the same item key is found again
            if current_key in sections:
                sections[current_key] += "\n\n" + cleaned_content
            else:
                sections[current_key] = cleaned_content

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


def process_ticker(ticker: str, storage_client: storage.Client, forms: list[str], stream_handler: StreamHandler = None):
    """Helper function to process a single ticker."""
    try:
        # 1. Find CIK
        cik = get_cik(ticker)

        if stream_handler:
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()

        # 2. Get the latest 10-K/10-Q filing and upload the document
        all_filings = get_filings_for_last_10_years(cik, forms)
        if not all_filings:
            logger.warning(f"No filings found for {ticker} within the specified parameters.")
            return
        for file_url, doc_name, filing_date, form_type in all_filings:
            upload_from_url_to_gcs(storage_client, file_url, ticker, doc_name, filing_date, form_type) # type: ignore
            if stream_handler:
                for log in stream_handler.queue: yield log
                stream_handler.queue.clear()

        # 3. Get company facts as structured JSON data (only needs to be done once per ticker)
        facts = get_company_facts(cik) # type: ignore
        company_name = facts.get('entityName')
        if company_name and stream_handler:
            # Send company name to the client in a structured format
            yield json.dumps({"type": "metadata", "data": {"company_name": company_name}}) + "\n"
        if stream_handler:
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()

        # 4. Upload the JSON data to GCS
        upload_json_to_gcs(storage_client, facts, ticker)
        if stream_handler:
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()

        logger.info(f"Successfully processed SEC filings and facts for {ticker}.")
        if stream_handler:
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()
    except gcp_exceptions.NotFound:
        logger.error(f"Upload failed for {ticker}: Bucket '{BUCKET_NAME}' not found.")
        raise # Re-raise to signal a server error
    except (requests.exceptions.RequestException, FileNotFoundError, ValueError) as e:
        logger.error(f"An unexpected error occurred while processing {ticker}: {e}", exc_info=True)
        raise # Re-raise to signal a server error

@functions_framework.http
def ingestion_agent_http(request):
    """HTTP Cloud Function entry point that streams logs."""
    def generate_logs():
        # This handler will capture logs for this specific request
        stream_handler = StreamHandler()
        # The client will parse each line. If not JSON, it's a plain log message.
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
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
            context = {"ticker": ticker, "agent": "ingestion-agent"}

            logger.info(f"Ingestion agent triggered for {ticker}.", extra={'context': context})
            
            # Yield any startup logs
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()

            forms_to_fetch = ['10-K', '10-Q']
            storage_client = storage.Client()

            # Yield logs from the process_ticker generator
            yield from process_ticker(ticker, storage_client, forms_to_fetch, stream_handler)

            logger.info("Ingestion process completed successfully.", extra={'context': context})
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()
            
            # Signal completion
            yield "STREAM_COMPLETE"

        except Exception as e:
            error_message = f"Ingestion agent failed: {str(e)}\n{traceback.format_exc()}"
            logger.critical(error_message, exc_info=False)
            for log in stream_handler.queue:
                yield log
            # Send a structured error message
            yield f"ERROR:{str(e)}"
        finally:
            # IMPORTANT: Remove the handler to avoid adding it again on the next invocation
            logger.removeHandler(stream_handler)

    return Response(stream_with_context(generate_logs()), mimetype='text/plain')


# This is the old function, which will be replaced by the streaming version above.
# I am commenting it out to preserve it for reference, but it is no longer used.
'''
@functions_framework.http
def ingestion_agent_http(request):
    """HTTP Cloud Function entry point."""
    def generate_logs():
        # This handler will capture logs for this specific request
        stream_handler = StreamHandler()
        # Use a simple formatter. We will prepend severity in a structured way for logs,
        # and send other data (like metadata) as distinct JSON objects.
        # The client will parse each line. If it's JSON, it's structured data.
        # If not, it's a plain log message.
        # A more robust format for logs to be parsed as JSON by the client.
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
            # Send initial metadata to the client
            yield json.dumps({"type": "metadata", "data": {"ticker": ticker}}) + "\n"

            # Yield any startup logs
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()

            forms_to_fetch = ['10-K', '10-Q']
            storage_client = storage.Client()

            # Yield logs from the process_ticker generator
            yield from process_ticker(ticker, storage_client, forms_to_fetch, stream_handler)

            logger.info("Ingestion process completed successfully.", extra={'context': context})
            for log in stream_handler.queue: yield log
            stream_handler.queue.clear()

        except Exception as e:
            error_message = f"Ingestion agent failed: {str(e)}\n{traceback.format_exc()}"
            # Use logger to capture the final error
            logger.critical(error_message, exc_info=False) # exc_info=False to avoid duplication
            # Yield any remaining logs
            for log in stream_handler.queue:
                yield log
            # Also send a structured error message
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"
        finally:
            # IMPORTANT: Remove the handler to avoid adding it again on the next invocation
            logger.removeHandler(stream_handler)

    return Response(stream_with_context(generate_logs()), mimetype='text/plain')
'''