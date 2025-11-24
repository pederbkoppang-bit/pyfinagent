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
from components.stock_chart import display_price_chart
from components.progress_bar import initialize_status_elements, update_progress, clear_progress
from components.log_display import initialize_log_display, log_to_ui, clear_log_display
from components.evaluation_table import display_evaluation_table
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
    log_to_ui(f"Contacting Ingestion Agent for ticker: **{ticker}**.")
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
            log_to_ui("Ingestion agent finished with a non-success status. Continuing with existing data.")
        else:
            logging.info("Ingestion agent finished successfully.", extra={'context': context})
            log_to_ui("Ingestion agent confirmed data is processed and up-to-date.")

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

def run_quant_agent(ticker: str) -> dict:
    """Executes the QuantAgent and returns the report."""
    context = {"agent_name": "quant_agent", "ticker": ticker}
    logging.info("Starting QuantAgent.", extra={'context': context})
    log_to_ui(f"Calling Quant Agent to get financial data for **{ticker}**.")
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
        log_to_ui("Quant Agent returned financial data successfully.")
        return quant_report

    except requests.exceptions.HTTPError as e:
        logging.error("HTTP Error during QuantAgent execution.", extra={'context': context}, exc_info=True)
        error_message = f"HTTP Error: {e}. The QuantAgent at {QUANT_AGENT_URL} failed."
        try:
            # Try to parse more specific error from function response
            error_details = e.response.json()
            st.error(f"{error_message} Details: {error_details.get('error', 'No details provided.')}")
        except json.JSONDecodeError:
            st.error(f"{error_message} Could not parse error response.")
        raise
    except requests.exceptions.RequestException as e:
        logging.error("Request failed for QuantAgent.", extra={'context': context}, exc_info=True)
        st.error(f"Could not connect to the QuantAgent. Please check the URL and function logs. Error: {e}")
        raise

def run_rag_agent(rag_model, ticker: str) -> dict:
    """Executes the RAG_Agent for 10-K analysis."""
    context = {"agent_name": "rag_agent", "ticker": ticker}
    logging.info("Starting RAG_Agent for 10-K analysis.", extra={'context': context})
    log_to_ui("Running RAG Agent to analyze 10-K/10-Q filings...")
    rag_prompt = f"Using the provided 10-K (annual) and 10-Q (quarterly) documents, analyze the Economic Moat, Governance (executive compensation), and key 'Risk Factors' for {ticker}. Prioritize the most recent filings for the most current information. [cite: Comprehensive Financial Analysis Template.pdf.pdf]"
    rag_response = rag_model.generate_content(rag_prompt)
    logging.info("RAG_Agent finished successfully.", extra={'context': context})
    return {"text": rag_response.text}

def run_market_agent(market_model, ticker: str) -> dict:
    """Executes the MarketAgent for news and sentiment analysis."""
    context = {"agent_name": "market_agent", "ticker": ticker}
    logging.info("Starting MarketAgent for news and sentiment analysis.", extra={'context': context})
    log_to_ui("Running Market Agent to analyze market sentiment and news...")
    market_prompt = f"Analyze the Macro (PESTEL) and current Market Sentiment (news, social media 'scuttlebutt') for {ticker}. [cite: Comprehensive Financial Analysis Template.pdf.pdf]"
    market_response = market_model.generate_content(market_prompt)
    logging.info("MarketAgent finished successfully.", extra={'context': context})
    return {"text": market_response.text}

def run_synthesis_agent(synthesis_model, ticker: str) -> dict:
    """Executes the AnalystAgent to synthesize all findings."""
    context = {"agent_name": "synthesis_agent", "ticker": ticker}
    logging.info("Starting AnalystAgent for final synthesis.", extra={'context': context})
    log_to_ui("Starting final synthesis with the Lead Analyst Agent...")
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
        deep_dive_analysis=st.session_state.report['deep_dive_analysis']
    )
    
    synthesis_response = synthesis_model.generate_content(synthesis_prompt)
    return synthesis_response

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

    # Display the detailed scoring table first for prominence
    display_evaluation_table()
    st.divider()
    
    # Display the price chart here, after the evaluation table
    if st.session_state.ticker:
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

def main():
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


    # --- ANALYSIS PIPELINE ---
    # The analysis now runs if the form was submitted OR if an analysis is already in progress.
    # We use 'analysis_in_progress' in session_state to manage this across st.rerun calls.
    analysis_in_progress = st.session_state.get('analysis_in_progress', False)

    if (submitted or analysis_in_progress) and st.session_state.ticker:
        ticker = st.session_state.ticker
        
        # --- INITIALIZATION (only on first run) ---
        if submitted:
            st.session_state.analysis_in_progress = True
            st.session_state.report = {}
            st.session_state.progress_value = 0
            st.session_state.log_messages = [] # Clear logs on new run

        try:
            # --- PIPELINE STAGE 1: INGESTION ---
            if 'ingestion_agent' not in st.session_state.report:
                update_progress(0, f"First, I need to gather the latest 10-K and 10-Q filings for **{ticker}**. This ensures my analysis is based on the most recent official data.")
                run_ingestion_agent(ticker)
                st.session_state.report['ingestion_agent'] = True
                update_progress(15, "Filings are ingested. Now, let's start the multi-agent analysis.")

            # --- PIPELINE STAGE 2: AGENT EXECUTION (SEQUENTIAL) ---
            if 'part_1_5_quant' not in st.session_state.report:
                update_progress(15, "Running the **QuantAgent** to pull key financial metrics, ratios, and valuation data. This forms the quantitative backbone of my analysis.")
                st.session_state.report['part_1_5_quant'] = run_quant_agent(ticker)
                update_progress(30, "Quantitative data acquired.")

            if 'part_1_4_6_rag' not in st.session_state.report:
                update_progress(30, "Deploying the **RAGAgent** to read through the 10-K filings. I'm looking for details on economic moat, governance, and stated risk factors.")
                st.session_state.report['part_1_4_6_rag'] = run_rag_agent(rag_model, ticker)
                update_progress(45, "Document analysis complete.")

            if 'part_2_3_market' not in st.session_state.report:
                update_progress(45, "Engaging the **MarketAgent** to scan for recent news, market sentiment, and macroeconomic trends related to the company.")
                st.session_state.report['part_2_3_market'] = run_market_agent(market_model, ticker)
                update_progress(60, "Market and sentiment analysis is done.")

            # --- PIPELINE STAGE 3: DEEP DIVE & SYNTHESIS ---
            if 'final_synthesis' not in st.session_state.report:
                # --- Deep Dive Step ---
                update_progress(60, "Now for the deep dive. I'll cross-reference the findings from all agents to generate critical questions that connect the dots.")
                log_to_ui("Generating critical 'deep dive' questions from agent reports...")
                deep_dive_prompt = (
                    "You are a senior financial analyst. Your task is to generate 2-3 critical and insightful questions by cross-referencing the following three reports. "
                    "Identify discrepancies, risks, or opportunities that arise from combining this information. The questions should guide a deeper investigation into the company's 10-K filings. "
                    "Format the output as a simple list of questions, with no preamble.\n\n"
                    f"1. Quantitative Report (Financials & Valuation):\n{json.dumps(st.session_state.report['part_1_5_quant'])}\n\n"
                    f"2. 10-K Analysis Report (Moat, Governance, Risks):\n{st.session_state.report['part_1_4_6_rag']['text']}\n\n"
                    f"3. Market & Sentiment Report (Macro, News):\n{st.session_state.report['part_2_3_market']['text']}"
                )

                question_response = synthesis_model.generate_content(deep_dive_prompt)
                deep_dive_questions = question_response.text.strip().split('\n')
                
                update_progress(65, "With the questions formulated, I'm using the **RAGAgent** again to find precise answers within the source documents.")
                log_to_ui("Answering deep dive questions using RAG Agent on source filings...")
                deep_dive_answers = []
                for question in deep_dive_questions:
                    if question.strip(): # Ensure not an empty line
                        answer_response = rag_model.generate_content(f"Using the provided 10-K and 10-Q documents, find the most relevant information to answer the following critical question, prioritizing the most recent filings: {question}")
                        deep_dive_answers.append(f"Q: {question}\nA: {answer_response.text.strip()}")
                
                st.session_state.report['deep_dive_analysis'] = "\n\n".join(deep_dive_answers)

                # --- Final Synthesis Step ---
                update_progress(70, "All data is in. The **LeadAnalyst** is now synthesizing everything into a final, structured report with a score and recommendation.")
                synthesis_response = run_synthesis_agent(synthesis_model, ticker)
                response_text = synthesis_response.text.strip()
                try:
                    if response_text.startswith("```json"):
                        response_text = response_text[7:-3].strip()
                    elif response_text.startswith("```"):
                        response_text = response_text[3:-3].strip()
                    final_report = json.loads(response_text)
                except json.JSONDecodeError as e:
                    # Handle parsing error
                    error_context = {"ticker": ticker, "llm_response": response_text}
                    logging.error("Failed to parse JSON from synthesis agent.", extra={'context': error_context})
                    st.error("The final analysis report returned an invalid format.")
                    st.code(response_text, language="text")
                    raise e

                update_progress(80, "The final report is drafted. Just a few more steps.")
                
                # --- PIPELINE STAGE 4: FINAL SCORE CALCULATION ---
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
                logging.info("Final weighted score calculated.", extra={'context': {"ticker": ticker, "final_score": final_report['final_weighted_score']}})

            # --- PIPELINE STAGE 5: SAVING REPORT ---
            if 'report_saved' not in st.session_state.report:
                update_progress(80, "Saving the complete analysis to BigQuery for future reference. This is the final step.")
                log_to_ui("Saving the final report to BigQuery...")
                company_name = st.session_state.report.get('part_1_5_quant', {}).get('company_name', 'N/A')
                row_to_insert = {
                    "ticker": ticker, "company_name": company_name,
                    "analysis_date": datetime.now().isoformat(),
                    "final_score": st.session_state.report['final_synthesis']['final_weighted_score'],
                    "recommendation": st.session_state.report['final_synthesis']['recommendation']['action'],
                    "summary": st.session_state.report['final_synthesis']['final_summary'],
                    "full_report_json": json.dumps(st.session_state.report)
                }
                errors = bq_client.insert_rows_json(table_id, [row_to_insert])
                if errors:
                    logging.error("Failed to write report to BigQuery.", extra={'context': {"ticker": ticker, "bq_errors": errors}})
                    st.error(f"Failed to write to BigQuery: {errors}")
                else:
                    logging.info("Report successfully saved to BigQuery.", extra={'context': {"ticker": ticker}})
                    log_to_ui("Analysis complete and saved successfully!")
                st.session_state.report['report_saved'] = True
                update_progress(100, "All done! The comprehensive analysis is complete.")

            # --- FINAL CLEANUP ---
            clear_log_display()
            clear_progress()
            st.session_state.analysis_in_progress = False # Unlock state
            st.rerun() # Rerun one last time to display the final report cleanly

        except (ValueError, Exception) as e:
            # General error handling
            error_context = {"ticker": ticker}
            logging.error("An error occurred during the analysis pipeline.", extra={'context': error_context}, exc_info=True)
            clear_progress()
            clear_log_display()
            st.error(f"An error occurred during analysis: {e}")
            st.write(traceback.format_exc())
            st.session_state.analysis_in_progress = False # Unlock state on error

    # --- Display Logic ---
    # This will now display either a newly generated report or a loaded past report
    if 'loaded_reports' in st.session_state and len(st.session_state.loaded_reports) > 1:
        display_reports_comparison()
    else:
        display_report()

main()