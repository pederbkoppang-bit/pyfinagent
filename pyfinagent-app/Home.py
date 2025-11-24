import logging
import streamlit as st
import requests
import json
from google.cloud import bigquery
from google.cloud.logging import Client as GCPLoggingClient # Correct, specific import
from vertexai.generative_models import GenerativeModel, Tool, grounding
import traceback
from datetime import datetime
import vertexai
import pandas as pd
import uuid
import time
from queue import Queue
from components.sidebar import display_sidebar
from components.stock_chart import display_price_chart
from components.progress_bar import initialize_status_elements, update_progress, clear_progress
from components.log_display import initialize_log_display, log_to_ui, clear_log_display, display_logs
from components.evaluation_table import display_evaluation_table
from components.radar_chart import display_radar_chart
from components.reports_comparison import display_reports_comparison

# --- Structured Logging Configuration ---

class JsonFormatter(logging.Formatter):
    """Formats log records into a JSON string."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Add extra context if it exists, which we will use for agent context
        if hasattr(record, 'context'):
            log_record.update(record.context)

        # Add exception info if it exists
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_record)

def setup_logging():
    """Configures the root logger to use the JSON formatter."""
    logger = logging.getLogger()
    # Set to DEBUG for more verbosity, or INFO for production
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")


def initialize_gcp_services():
    """
    Initializes all GCP services and AI models.
    This function is called once per session and its results are stored in st.session_state.
    """
    # Access secrets inside the function to ensure they are loaded after auth.
    try:
        PROJECT_ID = st.secrets.gcp.project_id
        LOCATION = st.secrets.gcp.vertex_ai_location
    except AttributeError as e:
        st.error(f"**Error:** Failed to access a required GCP secret: `{e}`")
        st.warning(
            "This usually means the secret is missing or misspelled in your Streamlit Cloud app settings "
            "or your local `secrets.toml` file."
        )
        # Help the user debug by showing what secrets ARE available.
        available_secrets = "Available top-level secrets: " + str(list(st.secrets.keys()))
        gcp_secrets = "Available secrets under `[gcp]`: " + str(list(st.secrets.get('gcp', {}).keys()))
        st.info(f"**Debugging Info:**\n\n{available_secrets}\n\n{gcp_secrets}")
        st.stop() # Stop the app gracefully.

    RAG_DATA_STORE_ID = st.secrets.agent.rag_data_store_id
    GEMINI_MODEL = st.secrets.agent.gemini_model

    logging.info(f"Initializing models with GEMINI_MODEL: '{GEMINI_MODEL}'")
    logging.info("Initializing GCP services...")
    # Log the SDK version for easier troubleshooting of version-specific issues.
    if hasattr(vertexai, '__version__'):
        logging.info(f"Using google-cloud-aiplatform SDK version: {vertexai.__version__}")

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    bq_client = bigquery.Client(project=PROJECT_ID)
    st.session_state.bigquery = bigquery # Store module for use in sidebar component
    table_id = f"{PROJECT_ID}.financial_reports.analysis_results"

    # --- AGENT DEFINITIONS (from PyFinAgent.md canvas) ---
    # For global datastores used with models in us-central1, the datastore path must explicitly use "global".
    # The LOCATION variable is for the model endpoint, not necessarily the datastore's location.
    datastore_path = (f"projects/{PROJECT_ID}/locations/global/collections/default_collection/"
                      f"dataStores/{RAG_DATA_STORE_ID}")
    rag_tool = Tool.from_retrieval(
        grounding.Retrieval(grounding.VertexAISearch(datastore=datastore_path))
    )

    # --- Robust Google Search Tool Initialization for Different SDK Versions ---
    # The `google-cloud-aiplatform` SDK has inconsistencies across versions.
    # - Newer SDKs use `Tool.from_google_search()`.
    # - Older SDKs use `Tool.from_google_search_retrieval()`.
    # - The API backend *requires* the tool object to have a `google_search` attribute.
    # This logic handles all cases: it tries the modern method, falls back to the
    # legacy one, and then *always* patches the object to ensure it has the correct

    # `google_search` attribute for API compatibility. After multiple attempts, it's clear
    # that patching the object after creation is unreliable.
    # The definitive solution is to construct the tool from a dictionary, which
    # bypasses the inconsistent factory methods (`from_google_search...`) and ensures
    # the object is created with the exact structure the API backend requires.
    logging.info("Initializing Google Search tool using Tool.from_dict for maximum compatibility.")
    tool_dict = {
        "google_search": {} # The presence of this key enables Google Search.
    }
    market_tool = Tool.from_dict(tool_dict)
    logging.info("Successfully created Google Search tool from dictionary.")

    synthesis_model = GenerativeModel(GEMINI_MODEL)
    rag_model = GenerativeModel(GEMINI_MODEL, tools=[rag_tool])
    market_model = GenerativeModel(GEMINI_MODEL, tools=[market_tool])

    logging.info("GCP services initialized successfully.")
    return {
        "bq_client": bq_client,
        "table_id": table_id,
        "rag_model": rag_model,
        "market_model": market_model,
        "synthesis_model": synthesis_model
    }

def run_ingestion_agent(ticker: str):
    """
    Triggers the ingestion agent Cloud Function to download and process
    10-K filings for a given ticker. This function is idempotent.
    """
    context = {"agent_name": "ingestion_agent", "ticker": ticker}
    logging.info("Starting ingestion agent check.", extra={'context': context})
    log_to_ui(f"Checking for and ingesting 10-K filings for {ticker}...")
    update_progress(st.session_state.get('progress_value', 0), f"Step 1/5: Checking for and ingesting 10-K filings for {ticker}...")

    try:
        INGESTION_AGENT_URL = st.secrets.agent.ingestion_agent_url
        context["agent_url"] = INGESTION_AGENT_URL
        response = requests.post(INGESTION_AGENT_URL, json={'ticker': ticker}, timeout=900) # 15 min timeout
        response.raise_for_status()
        result = response.json()

        # Add response status to context for richer logs
        context["response_status"] = result.get('status')
        context["response_body"] = result

        if result.get('status') != 'success':
            # Log a warning but don't block the analysis, as RAG might still work with older data
            logging.warning("Ingestion agent may not have completed successfully.", extra={'context': context})
            log_to_ui("Ingestion agent finished with a non-success status. I'll continue with existing data.")
        else:
            logging.info("Ingestion agent finished successfully.", extra={'context': context})
            log_to_ui("Ingestion agent confirmed data is up-to-date.")

    except requests.exceptions.RequestException as e:
        # Try to get more details from the response if available
        error_details = "No response body."
        if e.response is not None:
            try:
                error_details = e.response.json()
            except json.JSONDecodeError:
                error_details = e.response.text
        context["error_details"] = error_details
        logging.error("Failed to trigger or complete ingestion agent.", extra={'context': context}, exc_info=True)

        # --- ENHANCED UI ERROR REPORTING ---
        # Provide a clear, user-facing error message with actionable advice.
        st.warning(
            f"**Could not ingest 10-K documents for `{ticker}`.**\n\n"
            "This usually means the backend ingestion agent failed. The analysis will continue with any "
            "previously stored data, but the report may be incomplete. See details below."
        )
        # Use an expander to show the technical details without cluttering the UI.
        with st.expander("Click to see technical error details"):
            st.error(f"**Exception:** `{e}`")
            st.write("**Response from agent (if any):**")
            st.code(json.dumps(error_details, indent=2) if isinstance(error_details, dict) else error_details, language="json")

@st.cache_data(ttl="1h") # Cache for 1 hour
def run_quant_agent(ticker: str) -> dict:
    """Executes the QuantAgent and returns the report."""
    context = {"agent_name": "quant_agent", "ticker": ticker}
    logging.info("Starting QuantAgent.", extra={'context': context})
    log_to_ui(f"Calling Quant Agent for financial data on {ticker}...")
    QUANT_AGENT_URL = st.secrets.agent.quant_agent_url  # Access secret just-in-time
    # Status text is now managed in the main execution block for parallel runs.
    
    try:
        quant_response = requests.get(f"{QUANT_AGENT_URL}?ticker={ticker}", timeout=300)
        quant_response.raise_for_status()
        quant_report = quant_response.json()
        
        if quant_report.get("error"):
            context["error_details"] = quant_report['error']
            logging.error("QuantAgent returned a functional error.", extra={'context': context})
            raise ValueError(f"QuantAgent Error: {quant_report['error']}")
            
        logging.info("QuantAgent finished successfully.", extra={'context': context})
        log_to_ui("Quant Agent returned financial data.")
        return quant_report

    except requests.exceptions.HTTPError as e:
        logging.error("HTTP Error during QuantAgent execution.", extra={'context': context}, exc_info=True)
        error_message = f"The QuantAgent returned an HTTP error: `{e}`."
        details = "No specific error details were returned in the response body."
        try:
            # Try to parse more specific error from function response
            error_details = e.response.json()
            details = error_details.get('error', json.dumps(error_details))
        except json.JSONDecodeError:
            details = e.response.text if e.response.text else "No response body."
        
        st.error(f"{error_message}")
        with st.expander("Click to see technical error details"):
            st.code(details, language="text")

        raise
    except requests.exceptions.RequestException as e:
        logging.error("Request failed for QuantAgent.", extra={'context': context}, exc_info=True)
        st.error(f"Could not connect to the QuantAgent. Please check the URL and function logs. Error: {e}")
        raise

@st.cache_data(ttl="1h", hash_funcs={GenerativeModel: lambda m: m._model_name}) # Cache for 1 hour
def run_rag_agent(rag_model: GenerativeModel, ticker: str) -> dict:
    """Executes the RAG_Agent for 10-K analysis, with error handling."""
    context = {"agent_name": "rag_agent", "ticker": ticker}
    logging.info("Starting RAG_Agent for 10-K analysis.", extra={'context': context})
    log_to_ui("Analyzing 10-K/10-Q filings for moat, governance, and risks...")
    try:
        rag_prompt = f"Using the provided 10-K (annual) and 10-Q (quarterly) documents, analyze the Economic Moat, Governance (executive compensation), and key 'Risk Factors' for {ticker}. Prioritize the most recent filings for the most current information. [cite: Comprehensive Financial Analysis Template.pdf.pdf]"
        rag_response = rag_model.generate_content(rag_prompt)
        logging.info("RAG_Agent finished successfully.", extra={'context': context})
        return {"text": rag_response.text}
    except Exception as e:
        logging.error("RAG Agent (Vertex AI) failed.", extra={'context': context}, exc_info=True)
        st.error(f"The RAG Agent encountered an error while analyzing documents. The analysis cannot continue.")
        with st.expander("Click to see technical error details"):
            st.exception(e)
        raise

@st.cache_data(ttl="1h", hash_funcs={GenerativeModel: lambda m: m._model_name}) # Cache for 1 hour
def run_market_agent(market_model: GenerativeModel, ticker: str) -> dict:
    """Executes the MarketAgent for news and sentiment analysis, with error handling."""
    context = {"agent_name": "market_agent", "ticker": ticker}
    logging.info("Starting MarketAgent for news and sentiment analysis.", extra={'context': context})
    log_to_ui("Analyzing market sentiment and news...")
    try:
        market_prompt = f"Analyze the Macro (PESTEL) and current Market Sentiment (news, social media 'scuttlebutt') for {ticker}. [cite: Comprehensive Financial Analysis Template.pdf.pdf]"
        market_response = market_model.generate_content(market_prompt)
        logging.info("MarketAgent finished successfully.", extra={'context': context})
        return {"text": market_response.text}
    except Exception as e:
        logging.error("Market Agent (Vertex AI) failed.", extra={'context': context}, exc_info=True)
        st.error(f"The Market Agent encountered an error while analyzing market data. The analysis cannot continue.")
        with st.expander("Click to see technical error details"):
            st.exception(e)
        raise

@st.cache_data(ttl="1h", hash_funcs={GenerativeModel: lambda m: m._model_name}) # Cache for 1 hour
def run_synthesis_agent(synthesis_model: GenerativeModel, ticker: str, deep_dive_analysis: str) -> dict:
    """Executes the AnalystAgent to synthesize all findings."""
    context = {"agent_name": "synthesis_agent", "ticker": ticker}
    logging.info("Starting AnalystAgent for final synthesis.", extra={'context': context})
    log_to_ui("Synthesizing all findings with the Lead Analyst Agent...")
    update_progress(st.session_state.get('progress_value', 0), "Step 4/5: LeadAnalyst synthesizing final report...")
    
    # Dynamically load the synthesis prompt from a file
    with open("synthesis_prompt.txt", "r") as f:
        synthesis_prompt_template = f.read()
    
    # The deep_dive_analysis is now a required input for the final synthesis
    synthesis_prompt = synthesis_prompt_template.format(
        ticker=ticker,
        quant_report=json.dumps(st.session_state.report['part_1_5_quant']),
        rag_report=st.session_state.report['part_1_4_6_rag']['text'],
        market_report=st.session_state.report['part_2_3_market']['text'],
        deep_dive_analysis=deep_dive_analysis
    )
    
    try:
        synthesis_response = synthesis_model.generate_content(synthesis_prompt)
        return synthesis_response
    except Exception as e:
        logging.error("Synthesis Agent (Vertex AI) failed.", extra={'context': context}, exc_info=True)
        st.error(f"The final Synthesis Agent encountered a critical error. The analysis cannot continue.")
        with st.expander("Click to see technical error details"):
            st.exception(e)
        raise

def display_report():
    """Renders the final analysis report."""
    if 'report' not in st.session_state or not st.session_state.report.get('final_synthesis'):
        return

    report_data = st.session_state.report['final_synthesis']
    
    # Get price from Quant report
    price_data = st.session_state.report.get('part_1_5_quant', {}).get('part_5_valuation', {}).get('market_price', 'N/A')
    price_str = f"${price_data:.2f}" if isinstance(price_data, (int, float)) else str(price_data)

    # --- Main Score and Recommendation ---
    st.success("Analysis Complete!")

    # Display the detailed scoring table first for prominence
    display_evaluation_table()
    st.divider()

    # Create a two-column layout for the charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        display_radar_chart()
    with chart_col2:
        display_price_chart()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(
            label="Final Score",
            value=f"{report_data['final_weighted_score']:.2f} / 10"
        )
        st.metric(
            label="Recommendation",
            value=report_data['recommendation']['action']
        )
        st.caption(f"at {price_str}")

    with col2:
        st.subheader("Justification")
        st.write(report_data['recommendation']['justification'])

    st.subheader("Final Summary")
    st.write(report_data['final_summary'])
    
    with st.expander("View Full Raw Data (JSON)"):
        st.json(st.session_state.report, expanded=True)

# The email of the authorized user - this can also be moved to secrets if needed
AUTHORIZED_EMAIL = "peder.bkoppang@hotmail.no" 

def poll_gcp_logs(run_id: str, project_id: str, stop_event, log_queue: Queue):
    """Polls Google Cloud Logging for new log entries with a specific run_id."""

    client = GCPLoggingClient(project=project_id)
    last_timestamp = datetime.utcnow().isoformat() + "Z"
    seen_log_ids = set()

    while not stop_event.is_set():
        try:
            # Filter for logs with the specific run_id and a severity of INFO or higher
            log_filter = (
                f'jsonPayload.run_id="{run_id}" AND '
                f'timestamp > "{last_timestamp}" AND '
                f'severity >= INFO'
            )
            
            # list_entries is a paginated API call from the high-level client
            for entry in client.list_entries(filter_=log_filter, order_by="timestamp desc"):
                if entry.insert_id not in seen_log_ids:
                    seen_log_ids.add(entry.insert_id)
                    message = entry.payload.get('message', str(entry.payload))
                    log_queue.put(message) # Put message into the thread-safe queue

            # Update last_timestamp after processing a batch
            time.sleep(2) # Poll every 2 seconds

        except Exception as e:
            # Log errors but don't crash the polling thread
            logging.error(f"Log polling thread failed: {e}", exc_info=True)
            time.sleep(10) # Wait longer if there's an error

def check_for_cancellation():
    """Checks if the user has clicked the cancel button and cleans up if so."""
    if st.session_state.get('analysis_cancelled', False):
        st.warning("Analysis has been cancelled by the user.")
        logging.info("Analysis cancelled by user.", extra={'context': {'ticker': st.session_state.get('ticker')}})

        # --- CANCELLATION CLEANUP ---
        if 'stop_log_polling' in st.session_state:
            st.session_state.stop_log_polling.set() # Stop polling thread

        clear_progress()
        clear_log_display()

        # Reset all relevant session state keys
        st.session_state.analysis_in_progress = False
        st.session_state.analysis_cancelled = False # Reset the flag
        if 'report' in st.session_state:
            del st.session_state.report # Clear partial report

        # Clear the cancel button itself
        st.session_state.cancel_button_placeholder.empty()

        st.stop() # Stop the script execution for this run

def main():
    # --- Initialize Default Score Weights ---
    # This ensures that weights are available even if the user hasn't visited the Settings page.
    if 'score_weights' not in st.session_state:
        st.session_state.score_weights = {
            'pillar_1_corporate': 0.35,
            'pillar_2_industry': 0.20,
            'pillar_3_valuation': 0.20,
            'pillar_4_sentiment': 0.15,
            'pillar_5_governance': 0.10
        }

    """Main function to run the Streamlit application."""
    # Setup structured logging at the start of the app.
    setup_logging()

    st.title("PyFinAgent Dashboard: AI Financial Analyst")
    st.caption(f"A Multi-Agent AI built on the Comprehensive Financial Analysis Template")

    # Initialize services only once per session using session_state as a flag.
    try:
        if 'gcp_services' not in st.session_state:
            st.session_state.gcp_services = initialize_gcp_services()

        services = st.session_state.gcp_services
        bq_client = services.get("bq_client")
        table_id = services.get("table_id")
        rag_model = services.get("rag_model")
        market_model = services.get("market_model")
        synthesis_model = services.get("synthesis_model")
    except Exception as e:
        logging.error("Failed to initialize GCP services.", exc_info=True)
        st.error("Failed to initialize critical GCP services. The application cannot continue.")
        st.exception(e)
        return # Stop execution if services fail

    # Display the custom sidebar components. This should be called on every page for consistency.
    display_sidebar(bq_client, table_id, st.session_state.get("ticker"))

    # --- Logic to Load a Selected Past Report ---
    # This block is triggered by navigating from the 'Past Reports' page
    if 'reports_to_load' in st.session_state:
        reports_info = st.session_state.reports_to_load
        del st.session_state.reports_to_load # Clear trigger state

        ticker = reports_info[0]['Ticker']
        st.session_state.ticker = ticker
        st.session_state.loaded_reports = []

        with st.spinner(f"Loading {len(reports_info)} report(s) for {ticker}..."):
            try:
                for report_info in reports_info:
                    analysis_date_obj = pd.to_datetime(report_info['Analysis Date'])
                    
                    query = f"""
                        SELECT full_report_json FROM `{table_id}`
                        WHERE ticker = @ticker AND TIMESTAMP_TRUNC(analysis_date, SECOND) = @analysis_date
                        LIMIT 1
                    """
                    job_config = bigquery.QueryJobConfig(
                        query_parameters=[
                            bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                            bigquery.ScalarQueryParameter("analysis_date", "DATETIME", analysis_date_obj),
                        ]
                    )
                    query_job = bq_client.query(query, job_config=job_config)
                    result = list(query_job.result())
                    
                    if result:
                        retrieved_data = result[0].full_report_json
                        report_data = json.loads(retrieved_data) if isinstance(retrieved_data, str) else retrieved_data
                        st.session_state.loaded_reports.append(report_data)

                # If only one report was loaded, set it as the main report for the standard view
                if len(st.session_state.loaded_reports) == 1:
                    st.session_state.report = st.session_state.loaded_reports[0]
                # If multiple, we will use the comparison view

            except Exception as e:
                st.error(f"Failed to load the selected report: {e}")
                logging.error("Failed to load past reports from Home page", exc_info=True)

    # --- Ticker Input and Clear Button ---
    col1, col2 = st.columns([4, 1])
    with col1:
        # Use a key to automatically save the input's state across reruns and page navigations.
        st.text_input("Enter Company Ticker (e.g., NVDA, AAPL):", key="ticker", label_visibility="collapsed", placeholder="Enter Company Ticker (e.g., NVDA, AAPL)")
    with col2:
        if st.button("Clear", use_container_width=True):
            with st.spinner("Clearing..."):
                # Clear all report, analysis, and UI states
                for key in ['ticker', 'report', 'loaded_reports', 'analysis_in_progress']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.ticker = ""
                clear_log_display()
            st.rerun() # Rerun to reflect the cleared state immediately

    # Use a form to allow submission on pressing Enter
    with st.form(key='analysis_form'):
        submitted = st.form_submit_button("Run Comprehensive Analysis")

    # --- Progress Bar and Status Area ---
    # These placeholders will be controlled during the analysis run.
    initialize_status_elements()
    initialize_log_display()
    
    # Placeholder for the cancel button, which is only shown during a run
    if 'cancel_button_placeholder' not in st.session_state:
        st.session_state.cancel_button_placeholder = st.empty()


    # --- ANALYSIS PIPELINE ---
    # The analysis now runs if the form was submitted OR if an analysis is already in progress.
    # We use 'analysis_in_progress' in session_state to manage this across st.rerun calls.
    analysis_in_progress = st.session_state.get('analysis_in_progress', False)

    if (submitted or analysis_in_progress) and st.session_state.ticker:
        with st.spinner("Running comprehensive analysis... Please wait."):
            ticker = st.session_state.ticker
            
            # --- Live Log Display Update ---
            if 'log_queue' in st.session_state:
                while not st.session_state.log_queue.empty():
                    message = st.session_state.log_queue.get_nowait()
                    st.session_state.log_messages.append(message)

            display_logs()
            # --- INITIALIZATION (only on first run) ---
            if submitted:
                st.session_state.analysis_in_progress = True
                st.session_state.analysis_cancelled = False # Ensure cancel flag is reset
                st.session_state.run_id = str(uuid.uuid4()) # Generate unique ID for this run
                st.session_state.report = {}
                st.session_state.progress_value = 0
                st.session_state.log_messages = [] # Clear logs on new run

                # --- Start Log Polling Thread ---
                import threading
                st.session_state.log_queue = Queue() # Initialize the thread-safe queue
                st.session_state.stop_log_polling = threading.Event()
                polling_thread = threading.Thread(target=poll_gcp_logs, args=(st.session_state.run_id, st.secrets.gcp.project_id, st.session_state.stop_log_polling, st.session_state.log_queue))
                polling_thread.daemon = True
                polling_thread.start()

            try:
                # --- PIPELINE STAGE 1: INGESTION ---
                if 'ingestion_agent' not in st.session_state.report:
                    update_progress(0, f"First, I need to gather the latest 10-K and 10-Q filings for **{ticker}**.")
                    run_ingestion_agent(ticker)
                    st.session_state.report['ingestion_agent'] = True
                    update_progress(15, "Filings are ingested. Now, let's start the multi-agent analysis.")

                # --- PIPELINE STAGE 2: AGENT EXECUTION (SEQUENTIAL) ---
                if 'part_1_5_quant' not in st.session_state.report:
                    update_progress(15, "Running the **QuantAgent** to pull key financial metrics and ratios.")
                    st.session_state.report['part_1_5_quant'] = run_quant_agent(ticker)
                    update_progress(30, "Quantitative data acquired.")

                if 'part_1_4_6_rag' not in st.session_state.report:
                    update_progress(30, "Deploying the **RAGAgent** to read through the 10-K filings.")
                    st.session_state.report['part_1_4_6_rag'] = run_rag_agent(rag_model, ticker)
                    update_progress(45, "Document analysis complete.")

                if 'part_2_3_market' not in st.session_state.report:
                    update_progress(45, "Engaging the **MarketAgent** to scan for recent news and sentiment.")
                    st.session_state.report['part_2_3_market'] = run_market_agent(market_model, ticker)
                    update_progress(60, "Market and sentiment analysis is done.")

                # --- PIPELINE STAGE 3: DEEP DIVE & SYNTHESIS ---
                if 'final_synthesis' not in st.session_state.report:
                    update_progress(60, "Now for the deep dive. I'll cross-reference the findings to generate critical questions.")
                    log_to_ui("Generating 'deep dive' questions from agent reports...")
                    deep_dive_prompt = (
                        "You are a world-class financial investigator. Your mission is to synthesize the three reports below to formulate 2-3 critical questions. "
                        "These questions will be used to probe the company's latest 10-K (annual) and 10-Q (quarterly) filings for definitive answers. "
                        "Your goal is to uncover hidden risks, validate opportunities, or challenge assumptions by finding contradictions or connections between the reports. "
                        "Focus on questions where the answer is likely to materially impact an investment thesis.\n\n"
                        "For example, if the quant report shows declining margins, and the market report mentions new competitors, a good question would be: 'What does the latest 10-Q say about the impact of new market entrants on pricing power and profit margins?'\n\n"
                        "Output ONLY the questions as a numbered list, with no introductory text.\n\n"
                        "---REPORTS---\n"
                        f"1. Quantitative Report (Financials & Valuation):\n{json.dumps(st.session_state.report['part_1_5_quant'])}\n\n"
                        f"2. 10-K Analysis Report (Moat, Governance, Risks):\n{st.session_state.report['part_1_4_6_rag']['text']}\n\n"
                        f"3. Market & Sentiment Report (Macro, News):\n{st.session_state.report['part_2_3_market']['text']}"
                        "\n---END REPORTS---"
                    )
                    question_response = synthesis_model.generate_content(deep_dive_prompt)
                    deep_dive_questions = question_response.text.strip().split('\n')
                    log_to_ui("I've formulated these questions to investigate further:")
                    for i, q in enumerate(deep_dive_questions):
                        if q.strip(): log_to_ui(f"   *{q.strip()}*")
                    
                    update_progress(65, "Using the **RAGAgent** again to find precise answers in the source documents.")
                    log_to_ui("Now, I'm finding answers in the source filings...")
                    deep_dive_answers = []
                    for question in deep_dive_questions:
                        if question.strip():
                            answer_response = rag_model.generate_content(f"Using the provided 10-K and 10-Q documents, find the most relevant information to answer the following critical question, prioritizing the most recent filings: {question}")
                            deep_dive_answers.append(f"Q: {question}\nA: {answer_response.text.strip()}")
                    st.session_state.report['deep_dive_analysis'] = "\n\n".join(deep_dive_answers)
                    check_for_cancellation()

                    update_progress(70, "All data is in. The **LeadAnalyst** is synthesizing everything into a final report.")
                    synthesis_response = run_synthesis_agent(synthesis_model, ticker, st.session_state.report['deep_dive_analysis'])
                    response_text = synthesis_response.text.strip()
                    try:
                        if response_text.startswith("```json"): response_text = response_text[7:-3].strip()
                        elif response_text.startswith("```"): response_text = response_text[3:-3].strip()
                        final_report = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logging.error("Failed to parse JSON from synthesis agent.", extra={'context': {"ticker": ticker, "llm_response": response_text}})
                        st.error("The final analysis report returned an invalid format."); st.code(response_text, language="text"); raise e

                    update_progress(80, "The final report is drafted. Calculating final score.")
                    
                    scores = final_report['scoring_matrix']
                    weights = st.session_state.score_weights
                    final_score = (scores.get('pillar_1_corporate', 5.0) * weights['pillar_1_corporate'] + scores.get('pillar_2_industry', 5.0) * weights['pillar_2_industry'] + scores.get('pillar_3_valuation', 5.0) * weights['pillar_3_valuation'] + scores.get('pillar_4_sentiment', 5.0) * weights['pillar_4_sentiment'] + scores.get('pillar_5_governance', 5.0) * weights['pillar_5_governance'])
                    final_report['final_weighted_score'] = round(final_score, 2)
                    st.session_state.report['final_synthesis'] = final_report
                    logging.info("Final weighted score calculated.", extra={'context': {"ticker": ticker, "final_score": final_report['final_weighted_score']}})

                # --- PIPELINE STAGE 5: SAVING REPORT ---
                if 'report_saved' not in st.session_state.report:
                    update_progress(80, "Saving the complete analysis to BigQuery for future reference.")
                    log_to_ui("Saving the final report to BigQuery for future reference...")
                    company_name = st.session_state.report.get('part_1_5_quant', {}).get('company_name', 'N/A')
                    row_to_insert = {"ticker": ticker, "company_name": company_name, "analysis_date": datetime.now().isoformat(), "final_score": st.session_state.report['final_synthesis']['final_weighted_score'], "recommendation": st.session_state.report['final_synthesis']['recommendation']['action'], "summary": st.session_state.report['final_synthesis']['final_summary'], "full_report_json": json.dumps(st.session_state.report)}
                    errors = bq_client.insert_rows_json(table_id, [row_to_insert])
                    if errors:
                        logging.error("Failed to write report to BigQuery.", extra={'context': {"ticker": ticker, "bq_errors": errors}})
                        st.error(f"Failed to write to BigQuery: {errors}")
                    else:
                        logging.info("Report successfully saved to BigQuery.", extra={'context': {"ticker": ticker}})
                        log_to_ui("Analysis complete and saved!")
                    st.session_state.report['report_saved'] = True
                    update_progress(100, "All done! The comprehensive analysis is complete.")

                # --- FINAL CLEANUP ---
                if 'stop_log_polling' in st.session_state:
                    st.session_state.stop_log_polling.set()
                st.session_state.cancel_button_placeholder.empty()
                st.session_state.analysis_in_progress = False # Unlock state

            except (ValueError, Exception) as e:
                # --- ERROR CLEANUP ---
                if 'stop_log_polling' in st.session_state:
                    st.session_state.stop_log_polling.set()
                st.session_state.cancel_button_placeholder.empty()
                logging.error("An error occurred during the analysis pipeline.", extra={'context': {"ticker": ticker}}, exc_info=True)
                st.error(f"An error occurred during analysis: {e}")
                st.write(traceback.format_exc())
                st.session_state.analysis_in_progress = False

    # --- Display Logic ---
    # This will now display either a newly generated report or a loaded past report
    if 'loaded_reports' in st.session_state and len(st.session_state.loaded_reports) > 1:
        display_reports_comparison()
    else:
        display_report()

main()