import os
import requests
import json
from flask import jsonify
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part


def get_npl_tone_analysis_prompt(transcript_text: str) -> str:
    """
    Generates a robust prompt for Gemini to analyze the tone and content of an earnings call transcript.
    This prompt is designed based on financial NLP research to extract specific, predictive signals.
    """
    return f"""
    You are a sophisticated financial NLP analyst specializing in earnings call transcripts for a quantitative hedge fund.
    Your primary goal is to detect early, non-obvious signals of a cyclical or product-driven breakout, particularly for tech and hardware companies.

    Analyze the following earnings call transcript. Pay close attention to the "Management Guidance" and "Analyst Q&A" sections.

    TRANSCRIPT:
    ---
    {transcript_text}
    ---

    **TASK:**
    Execute the following analysis and provide your output in a structured JSON format ONLY.

    1.  **Forward-Looking Sentiment Score (1-10):** Analyze the overall tone, focusing strictly on forward-looking statements (guidance, outlook, future demand). Ignore discussions of past performance. A score of 1 indicates extremely pessimistic future guidance, while a 10 indicates extremely optimistic future guidance.

    2.  **Analyst Q&A Confidence Summary:** Isolate the "Analyst Q&A" section. Analyze management's responses to tough questions. Determine if they sounded defensive and evasive, or confident and in control. Provide a brief summary of your findings.

    3.  **Cyclical Catalysts Scan:** Perform a strict keyword and theme scan for classic hardware/semiconductor breakout catalysts. The presence of ANY of the following themes should result in a 'true' value for `cyclical_catalysts_detected`.
        *   "inventory depletion" or "inventory bottoming"
        *   "unmet demand" or "pent-up demand"
        *   "supply chain recovery" or "easing supply constraints"
        *   "capacity constraints" (in the context of being unable to meet demand)
        *   "order backlog" or "strong order book"

    4.  **Key Quotes:** Extract up to three direct quotes from the transcript that are most indicative of the forward-looking sentiment and any detected catalysts.

    **OUTPUT FORMAT (JSON ONLY):**
    {{
      "forward_sentiment_score": <integer>,
      "qa_confidence_summary": "<Your summary of the Q&A section>",
      "cyclical_catalysts_detected": <boolean>,
      "key_quotes": ["<quote1>", "<quote2>", ...]
    }}
    """

def earnings_ingestion_agent(request):
    """
    HTTP-triggered Google Cloud Function to fetch earnings call transcripts.

    This function takes a stock ticker, calls the API Ninjas API for
    earnings call transcripts, and returns the result.

    Args:
        request (flask.Request): The request object.
            <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
            The request is expected to have a 'ticker' in the JSON body.

    Returns:
        A JSON response containing the earnings call transcript or an error message.
    """
    # Set CORS headers for preflight requests to allow cross-origin calls
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    # Extract ticker from request JSON body
    request_json = request.get_json(silent=True)
    if request_json and 'ticker' in request_json:
        ticker = request_json['ticker']
    else:
        return (jsonify({"error": "JSON body must include 'ticker'"}), 400, headers)

    # Securely get API key from the secret file mounted by Google Cloud Functions
    # The path is defined in the deploy script: --update-secrets="/secrets/earnings_api_key=..."
    secret_path = "/secrets/earnings_api_key"
    api_key = None
    if os.path.exists(secret_path):
        with open(secret_path, 'r') as f:
            api_key = f.read().strip()

    if not api_key:
        # This log is for Cloud Logging to help with debugging.
        print("ERROR: API key not found at expected path.")
        # This is the user-facing error.
        return (jsonify({"error": "Server configuration error: Could not retrieve API credentials."}), 500, headers)

    # Get GCS bucket name from environment variables
    bucket_name = os.environ.get("BUCKET_NAME")
    if not bucket_name:
        print("ERROR: BUCKET_NAME environment variable not set.")
        return (jsonify({"error": "Server configuration error: Storage bucket not configured."}), 500, headers)

    # Initialize Vertex AI
    try:
        project_id = os.environ.get("GCP_PROJECT")
        location = "us-central1"  # Or your preferred location
        vertexai.init(project=project_id, location=location)
    except Exception as e:
        print(f"ERROR: Failed to initialize Vertex AI: {e}")
        return (jsonify({"error": "Server configuration error: Could not initialize AI services."}), 500, headers)

    api_url = f'https://api.api-ninjas.com/v1/earningscalltranscript?ticker={ticker}'

    try:
        response = requests.get(api_url, headers={'X-Api-Key': api_key})
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        data = response.json()

        if not data:
            return (jsonify({"message": f"No earnings call transcript found for ticker: {ticker}"}), 404, headers)

        # If data is found, save it to GCS before returning
        # We assume the first item in the list is the most recent one we want.
        if isinstance(data, list) and data:
            transcript_item = data[0]
            year = transcript_item.get("year", "unknown_year")
            quarter = transcript_item.get("quarter", "unknown_quarter")
            transcript_content = transcript_item.get("content", "")

            # --- New NLP Tone Analysis Step ---
            if transcript_content:
                try:
                    model = GenerativeModel("gemini-1.5-flash-001")
                    prompt = get_npl_tone_analysis_prompt(transcript_content)
                    response = model.generate_content(prompt)
                    
                    # Clean and parse the JSON output from the model
                    cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
                    nlp_analysis = json.loads(cleaned_response)

                    # Append the NLP metadata to the transcript item
                    transcript_item['nlp_analysis'] = nlp_analysis
                    print(f"Successfully performed NLP analysis for {ticker} {year} Q{quarter}.")

                except Exception as e:
                    print(f"ERROR: Vertex AI NLP analysis failed for {ticker}: {e}")
                    # Proceed without NLP data, but log the error. The core data is still valuable.
                    transcript_item['nlp_analysis'] = {"error": f"NLP analysis failed: {str(e)}"}

            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            # Define a structured path for the transcript files
            blob = bucket.blob(f"{ticker.upper()}/transcripts/{year}_Q{quarter}.json")

            # Upload the data as a JSON string
            blob.upload_from_string(
                json.dumps(transcript_item, indent=2),
                content_type='application/json'
            )
            print(f"Successfully uploaded transcript for {ticker} {year} Q{quarter} to gs://{bucket_name}/{blob.name}")

        return (jsonify(data), 200, headers)

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request to API Ninjas failed: {e}")
        return (jsonify({"error": "Failed to fetch data from external API."}), 502, headers)