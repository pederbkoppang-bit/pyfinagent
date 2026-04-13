import functions_framework
from flask import make_response, jsonify

@functions_framework.http
def handler(request):
    """
    A lightweight HTTP endpoint to confirm the ingestion function is available.
    When the main ingestion function is running, it may not respond.
    When it's idle (i.e., finished), it will respond to this.
    """
    # Set CORS headers to allow requests from the Streamlit app's domain
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    return make_response(jsonify({"status": "idle"}), 200, headers)