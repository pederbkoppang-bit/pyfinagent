import logging
import streamlit as st
import requests
import json
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel, Tool, grounding
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import vertexai
import pandas as pd
from components.sidebar import display_sidebar

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="PyFinAgent Dashboard", page_icon="🏠", layout="wide")


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
    logging.info(f"Starting ingestion check for ticker: {ticker}")
    st.session_state.status_text.text(f"Step 1/5: Checking for and ingesting 10-K filings for {ticker}...")
    
    try:
        INGESTION_AGENT_URL = st.secrets.agent.ingestion_agent_url
        # The body now includes the ticker
        response = requests.post(INGESTION_AGENT_URL, json={'ticker': ticker}, timeout=900) # 15 min timeout
        response.raise_for_status()
        result = response.json()
        logging.info(f"Ingestion agent for {ticker} finished: {result.get('status', 'No status returned')}")
        if result.get('status') != 'success':
             # Log a warning but don't block the analysis, as RAG might still work with older data
             logging.warning(f"Ingestion agent may not have completed successfully: {result}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to trigger or complete ingestion for {ticker}: {e}", exc_info=True)
        # Display a warning in the UI but allow the process to continue
        st.warning(f"Could not ingest 10-K documents for {ticker}. Analysis will proceed, but may be incomplete.")

def run_quant_agent(ticker: str) -> dict:
    """Executes the QuantAgent and returns the report."""
    logging.info(f"Starting QuantAgent for ticker: {ticker}")
    QUANT_AGENT_URL = st.secrets.agent.quant_agent_url  # Access secret just-in-time
    # Status text is now managed in the main execution block for parallel runs.
    
    try:
        quant_response = requests.get(f"{QUANT_AGENT_URL}?ticker={ticker}", timeout=300)
        quant_response.raise_for_status()
        quant_report = quant_response.json()
        
        if quant_report.get("error"):
            logging.error(f"QuantAgent returned an error: {quant_report['error']}")
            raise ValueError(f"QuantAgent Error: {quant_report['error']}")
            
        logging.info("QuantAgent finished successfully.")
        return quant_report

    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error during QuantAgent execution: {e}", exc_info=True)
        error_message = f"HTTP Error: {e}. The QuantAgent at {QUANT_AGENT_URL} failed."
        try:
            # Try to parse more specific error from function response
            error_details = e.response.json()
            st.error(f"{error_message} Details: {error_details.get('error', 'No details provided.')}")
        except json.JSONDecodeError:
            st.error(f"{error_message} Could not parse error response.")
        raise
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for QuantAgent: {e}", exc_info=True)
        st.error(f"Could not connect to the QuantAgent. Please check the URL and function logs. Error: {e}")
        raise

def run_rag_agent(rag_model, ticker: str) -> dict:
    """Executes the RAG_Agent for 10-K analysis."""
    logging.info("Starting RAG_Agent for 10-K analysis.")
    rag_prompt = f"Using ONLY the provided 10-K documents, analyze the Economic Moat and Governance (exec compensation), and key 'Risk Factors' for {ticker}. [cite: Comprehensive Financial Analysis Template.pdf.pdf]"
    rag_response = rag_model.generate_content(rag_prompt)
    logging.info("RAG_Agent finished successfully.")
    return {"text": rag_response.text}

def run_market_agent(market_model, ticker: str) -> dict:
    """Executes the MarketAgent for news and sentiment analysis."""
    logging.info("Starting MarketAgent for news and sentiment analysis.")
    market_prompt = f"Analyze the Macro (PESTEL) and current Market Sentiment (news, social media 'scuttlebutt') for {ticker}. [cite: Comprehensive Financial Analysis Template.pdf.pdf]"
    market_response = market_model.generate_content(market_prompt)
    logging.info("MarketAgent finished successfully.")
    return {"text": market_response.text}

def run_synthesis_agent(synthesis_model, ticker: str) -> dict:
    """Executes the AnalystAgent to synthesize all findings."""
    logging.info("Starting AnalystAgent for final synthesis.")
    st.session_state.status_text.text("Step 4/5: LeadAnalyst synthesizing final report...")
    
    # Dynamically load the synthesis prompt from a file
    with open("synthesis_prompt.txt", "r") as f:
        synthesis_prompt_template = f.read()

    synthesis_prompt = synthesis_prompt_template.format(
        ticker=ticker,
        quant_report=json.dumps(st.session_state.report['part_1_5_quant']),
        rag_report=st.session_state.report['part_1_4_6_rag']['text'],
        market_report=st.session_state.report['part_2_3_market']['text']
    )
    
    synthesis_response = synthesis_model.generate_content(synthesis_prompt)
    return synthesis_response

def display_price_chart():
    """Renders a line chart of historical stock prices if available."""
    if 'report' not in st.session_state or not st.session_state.report.get('part_1_5_quant'):
        return

    chart_container = st.session_state.get('chart_container')
    historical_prices = st.session_state.report['part_1_5_quant'].get('historical_prices')
    if historical_prices and chart_container:
        price_df = pd.read_json(historical_prices, orient='split')
        with chart_container.container():
            st.subheader("Historical Price Chart")
            st.line_chart(price_df.set_index('Date')['Close'])

def display_report():
    """Renders the final analysis report in a structured and appealing layout."""
    if 'report' not in st.session_state or not st.session_state.report.get('final_synthesis'):
        return

    report_data = st.session_state.report['final_synthesis']
    
    # Get the price from the Quant report for context
    price_data = st.session_state.report.get('part_1_5_quant', {}).get('part_5_valuation', {}).get('market_price', 'N/A')
    price_str = f"${price_data:.2f}" if isinstance(price_data, (int, float)) else str(price_data)

    # --- Main Score and Recommendation ---
    st.success("Analysis Complete!")
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

    st.divider()

    st.subheader("Final Summary")
    st.write(report_data['final_summary'])
    
    with st.expander("View Full Raw Data (JSON)"):
        st.json(st.session_state.report)

# The email of the authorized user - this can also be moved to secrets if needed
AUTHORIZED_EMAIL = "peder.bkoppang@hotmail.no" 

def main():
    """Main function to run the Streamlit application."""
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
    if 'report_to_load' in st.session_state:
        report_info = st.session_state.report_to_load
        # Clear the trigger state
        del st.session_state.report_to_load

        with st.spinner(f"Loading report for {report_info['Ticker']} from {report_info['Analysis Date']}..."):
            try:
                # Extract ticker and date from the report info dictionary
                ticker = report_info['Ticker']
                # The date is already a string, convert it to a datetime object for the query
                analysis_date_obj = pd.to_datetime(report_info['Analysis Date'])
                
                # Query for the full JSON report
                query = f"""
                    SELECT full_report_json FROM `{table_id}`
                    WHERE ticker = @ticker AND TIMESTAMP_TRUNC(analysis_date, MINUTE) = @analysis_date
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
                
                # The BigQuery client might auto-parse the JSON column into a dict.
                # If not, it will be a string that needs parsing. This handles both cases.
                retrieved_data = result[0].full_report_json
                if isinstance(retrieved_data, str):
                    st.session_state.report = json.loads(retrieved_data)
                else:
                    st.session_state.report = retrieved_data # It's already a dict
            except Exception as e:
                st.error(f"Failed to load the selected report: {e}")

    # --- Ticker Input and Clear Button ---
    col1, col2 = st.columns([4, 1])
    with col1:
        # Use a key to automatically save the input's state across reruns and page navigations.
        st.text_input("Enter Company Ticker (e.g., NVDA, AAPL):", key="ticker", label_visibility="collapsed", placeholder="Enter Company Ticker (e.g., NVDA, AAPL)")
    with col2:
        if st.button("Clear", use_container_width=True):
            with st.spinner("Clearing..."):
                st.session_state.ticker = ""
            st.rerun() # Rerun to reflect the cleared state immediately

    # Use a form to allow submission on pressing Enter
    with st.form(key='analysis_form'):
        submitted = st.form_submit_button("Run Comprehensive Analysis")

    # Create a placeholder for the chart that will be filled later.
    st.session_state.chart_container = st.empty()

    st.divider()

    # Run analysis only if the form was submitted and a ticker was provided
    if submitted and st.session_state.ticker:
        ticker = st.session_state.ticker # Use a local variable for clarity within this block
        st.session_state.chart_container.empty() # Clear previous chart
        st.session_state.report = {}
        st.session_state.status_text = st.empty()
        
        with st.spinner("Running Analysis Pipeline..."):
            try:
                # --- 1. Run Ingestion Agent (Synchronously) ---
                # This ensures documents are ready before the RAG agent runs.
                run_ingestion_agent(ticker)

                # --- 2. Run Independent Agents in Parallel ---
                st.session_state.status_text.text("Step 2/5: Running Quant, RAG, and Market agents in parallel...")
                with ThreadPoolExecutor(max_workers=3) as executor:
                    quant_future = executor.submit(run_quant_agent, ticker)
                    rag_future = executor.submit(run_rag_agent, rag_model, ticker)
                    market_future = executor.submit(run_market_agent, market_model, ticker)

                    # Get results, this will block until each future is complete
                    st.session_state.report['part_1_5_quant'] = quant_future.result()
                    st.session_state.report['part_1_4_6_rag'] = rag_future.result()
                    st.session_state.report['part_2_3_market'] = market_future.result()

                logging.info("All parallel agents finished successfully.")

                # --- 3. Display Chart Immediately After Quant Data is Available ---
                display_price_chart()

                # --- 4. Run Synthesis Agent (Depends on previous results) ---
                # The status text for this agent is handled inside its function.
                synthesis_response = run_synthesis_agent(synthesis_model, ticker)
                
                # This is a duplicate call, let's remove it. The one above is sufficient.
                synthesis_response = run_synthesis_agent(synthesis_model, ticker)
                
                # Clean up potential markdown formatting from the LLM response before parsing
                response_text = synthesis_response.text.strip()
                try:
                    if response_text.startswith("```json"):
                        response_text = response_text[7:-3].strip()
                    elif response_text.startswith("```"):
                        response_text = response_text[3:-3].strip()

                    final_report = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse JSON from synthesis agent for ticker {ticker}.")
                    logging.error(f"LLM Response Text was: {response_text}")
                    st.error("The final analysis report returned an invalid format and could not be parsed.")
                    st.code(response_text, language="text")
                    raise e
                
                logging.info(f"AnalystAgent Response: {final_report}")
                # --- 5. Final Calculation & Saving ---
                # [cite: PyFinAgent.md - Python-First, Part 8]
                scores = final_report['scoring_matrix']
                final_score = (
                    scores.get('pillar_1_corporate', 5.0) * 0.35 +
                    scores.get('pillar_2_industry', 5.0) * 0.20 +
                    scores.get('pillar_3_valuation', 5.0) * 0.20 +
                    scores.get('pillar_4_sentiment', 5.0) * 0.15 +
                    scores.get('pillar_5_governance', 5.0) * 0.10
                )
                final_report['final_weighted_score'] = round(final_score, 2)
                st.session_state.report['final_synthesis'] = final_report
                logging.info(f"Final weighted score calculated: {final_report['final_weighted_score']}")
                
                st.session_state.status_text.text("Step 5/5: Saving report to BigQuery...")
                row_to_insert = {
                    "ticker": ticker,
                    "analysis_date": datetime.now().isoformat(),
                    "final_score": final_report['final_weighted_score'],
                    "recommendation": final_report['recommendation']['action'],
                    "summary": final_report['final_summary'],
                    "full_report_json": json.dumps(st.session_state.report)
                }
                errors = bq_client.insert_rows_json(table_id, [row_to_insert])
                if errors:
                    logging.error(f"Failed to write to BigQuery: {errors}")
                    st.error(f"Failed to write to BigQuery: {errors}")
                else:
                    logging.info("Report successfully saved to BigQuery.")
                
                st.session_state.status_text.empty()
                del st.session_state.status_text
                
            except ValueError as e:
                # Catches specific errors raised by agents (like QuantAgent error)
                st.error(str(e))
            except Exception as e:
                logging.error(f"An error occurred during analysis for ticker '{ticker}': {e}", exc_info=True)
                st.session_state.status_text.empty()
                st.error(f"An error occurred during analysis: {e}")
                st.write(traceback.format_exc())

    # --- Display Logic ---
    # This will now display either a newly generated report or a loaded past report
    display_price_chart()
    display_report()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # This is a global failsafe.
        logging.error("A critical error occurred in the main application.", exc_info=True)
        # Attempt to display the error in the Streamlit UI if possible.
        st.error("A critical error occurred. Please check the logs for more details.")
        st.exception(e)