import logging
import streamlit as st
import requests
import json
import re
from google.cloud import bigquery
from google.cloud.logging import Client as GCPLoggingClient
from vertexai.generative_models import GenerativeModel, Tool, grounding
import traceback
from datetime import datetime
import vertexai
import pandas as pd
import uuid
import time
from queue import Queue

# --- Custom Modules ---
from components.sidebar import display_sidebar
from components.stock_chart import display_price_chart
from components.status_handler import StatusHandler
from components.evaluation_table import display_evaluation_table
from components.radar_chart import display_radar_chart
from components.reports_comparison import display_reports_comparison
import agent_prompts 
import tools_alphavantage 

# --- Structured Logging ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {"timestamp": self.formatTime(record, self.datefmt), "level": record.levelname, "message": record.getMessage()}
        if hasattr(record, 'context'): log_record.update(record.context)
        if record.exc_info: log_record['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

# --- Helper Functions ---

def clean_json_output(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text)
        text = re.sub(r"```$", "", text)
    return text.strip()

def initialize_gcp_services():
    try:
        PROJECT_ID = st.secrets.gcp.project_id
        LOCATION = st.secrets.gcp.vertex_ai_location
    except AttributeError:
        st.error("Missing GCP secrets.")
        st.stop()

    RAG_DATA_STORE_ID = st.secrets.agent.rag_data_store_id
    GEMINI_MODEL = st.secrets.agent.gemini_model

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    bq_client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.financial_reports.analysis_results"

    datastore_path = (f"projects/{PROJECT_ID}/locations/global/collections/default_collection/"
                      f"dataStores/{RAG_DATA_STORE_ID}")
    
    rag_tool = Tool.from_retrieval(grounding.Retrieval(grounding.VertexAISearch(datastore=datastore_path)))
    
    # Standard model setup
    synthesis_model = GenerativeModel(GEMINI_MODEL)
    rag_model = GenerativeModel(GEMINI_MODEL, tools=[rag_tool])
    market_model = GenerativeModel(GEMINI_MODEL) 

    return {
        "bq_client": bq_client,
        "table_id": table_id,
        "rag_model": rag_model,
        "market_model": market_model,
        "synthesis_model": synthesis_model
    }

# --- Display Function (Moved Up to Prevent Crash) ---

def display_report():
    """Renders the final analysis report."""
    if 'report' not in st.session_state or not st.session_state.report.get('final_synthesis'):
        return

    report_data = st.session_state.report['final_synthesis']
    
    # Get price safely
    try:
        price_data = st.session_state.report.get('part_1_5_quant', {}).get('part_5_valuation', {}).get('market_price', 'N/A')
        price_str = f"${price_data:.2f}" if isinstance(price_data, (int, float)) else str(price_data)
    except:
        price_str = "N/A"

    st.success("Analysis Complete!")
    display_evaluation_table()
    st.divider()

    # New Layout: Main content on the left (2/3 width), dashboard on the right (1/3 width)
    main_col, dashboard_col = st.columns([2, 1])

    with main_col:
        st.subheader("Stock Price Chart (5-Year History)")
        # Pass the ticker from the report data to avoid relying on session state
        display_price_chart(ticker=st.session_state.report.get('part_1_5_quant', {}).get('ticker'))

        st.subheader("Justification")
        st.write(report_data.get('recommendation', {}).get('justification', 'No justification provided.'))

        st.subheader("Final Summary")
        st.write(report_data.get('final_summary', 'No summary provided.'))

    with dashboard_col:
        st.metric("Final Score", f"{report_data.get('final_weighted_score', 0):.2f} / 10")
        st.metric("Recommendation", report_data.get('recommendation', {}).get('action', 'N/A'))
        st.caption(f"at {price_str}")
        st.subheader("Pillar Score Analysis")
        display_radar_chart()
    
    with st.expander("View Full Raw Data (JSON)"):
        st.json(st.session_state.report, expanded=True)

def _handle_report_load(bq_client, table_id, _status_handler=None):
    """
    Ensures the main 'report' object in session state is a valid dictionary.
    It handles two cases:
    1. Loading a specific report from BigQuery via the 'report_to_load' flag.
    2. Parsing the 'report' object if it has been converted to a JSON string.
    """
    try:
        # Case 1: A specific report was selected from the sidebar.
        # This takes priority.
        if 'report_to_load' in st.session_state and st.session_state.report_to_load:
            report_info = st.session_state.report_to_load
            query = f"""
                SELECT full_report_json FROM `{table_id}`
                WHERE ticker = @ticker AND analysis_date = @analysis_date LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("ticker", "STRING", report_info['Ticker']),
                    bigquery.ScalarQueryParameter("analysis_date", "TIMESTAMP", report_info['Analysis Date']),
                ]
            )
            results = list(bq_client.query(query, job_config=job_config).result())
            if results:
                st.session_state.report = json.loads(results[0].full_report_json)
            del st.session_state.report_to_load # Clear flag

        # Case 2: The current report object is a string.
        # This check runs *independently* to fix the state after an analysis completes.
        # This is the definitive fix for the end-of-pipeline error.
        if 'report' in st.session_state and isinstance(st.session_state.report, str):
            if st.session_state.report: # Avoid parsing empty strings
                st.session_state.report = json.loads(st.session_state.report)

    except (json.JSONDecodeError, TypeError, AttributeError) as e:
        st.error(f"Failed to load or parse report data: {e}")
        st.session_state.report = {} # Reset to a safe state

def run_ingestion_agent(ticker: str, status_handler: StatusHandler):
    """Triggers the data ingestion agent and waits for it to complete."""
    INGESTION_AGENT_URL = st.secrets.agent.ingestion_agent_url
    success = False # Default to failure unless success message is seen

    try:
        # Make a POST request with stream=True to handle the streaming response.
        # The agent will now send back logs line-by-line.
        response = requests.post(
            INGESTION_AGENT_URL, json={'ticker': ticker}, timeout=300, stream=True
        )
        response.raise_for_status() # Check for immediate errors like 4xx/5xx

        # Iterate over the response content line by line.
        for line in response.iter_lines():
            if not line:
                continue

            message = line.decode('utf-8')
            try:
                # Attempt to parse the line as a JSON object from the agent
                data = json.loads(message)
                if data.get("type") == "metadata" and "company_name" in data.get("data", {}):
                    # If we receive the company name, store it and rerun to display it
                    st.session_state['company_name'] = data['data']['company_name']
                    # Do NOT rerun here. Let the step complete first.
                elif data.get("type") == "error":
                    status_handler.log(f"ERROR: {data.get('message')}")
                    success = False
                    break
            except json.JSONDecodeError:
                # If it's not JSON, treat it as a plain log message
                log_message = message
                cleaned_log = re.sub(r'^(INFO|WARNING|ERROR|CRITICAL):', '', log_message).strip()
                if cleaned_log:
                    status_handler.log(cleaned_log)

                if "Ingestion process completed successfully." in log_message:
                    success = True
                elif "Ingestion agent failed" in log_message:
                    success = False
                    break
            
        return success
    except Exception as e:
        st.error(f"Failed to connect to or stream from ingestion agent: {e}")
        return False

@st.cache_data(ttl="1h")
def run_quant_agent(ticker: str, _status_handler: StatusHandler) -> dict:
    """Triggers the Quant Agent and streams its logs to the UI."""
    QUANT_AGENT_URL = st.secrets.agent.quant_agent_url
    final_data = None

    try:
        # Use stream=True to handle the streaming response
        response = requests.get(f"{QUANT_AGENT_URL}?ticker={ticker}", timeout=600, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                log_message = line.decode('utf-8')
                if log_message.startswith("FINAL_JSON:"):
                    # This is the final data payload
                    json_str = log_message.replace("FINAL_JSON:", "", 1)
                    final_data = json.loads(json_str)
                    break # Stop processing after getting the final data
                elif log_message.startswith("ERROR:"):
                    raise ValueError(log_message)
                else:
                    # This is a regular log message, update the UI
                    cleaned_log = re.sub(r'^(INFO|WARNING|ERROR|CRITICAL):', '', log_message).strip()
                    _status_handler.log(cleaned_log)

        if final_data:
            # Enhance Quant Agent with Alpha Vantage Overview after getting the data
            # Pass the handler to the overview function for detailed logging
            av_overview = tools_alphavantage.get_fundamental_overview(ticker, _status_handler)
            if "part_5_valuation" in final_data and isinstance(av_overview, dict):
                final_data["part_5_valuation"]["alpha_vantage_overview"] = av_overview
            return final_data
        else:
            raise ValueError("Quant Agent did not return a final JSON payload.")
    except Exception as e:
        st.error(f"Quant Agent failed: {e}")
        raise

def run_rag_agent(rag_model, ticker, status_handler: StatusHandler):
    prompt = agent_prompts.get_rag_prompt(ticker, status_handler)
    status_handler.log("   -> RAG Agent is processing...")
    response = rag_model.generate_content(prompt)
    status_handler.log("RAG analysis complete.")
    return {"text": response.candidates[0].content.parts[0].text}

def run_market_agent(market_model, ticker, av_data, status_handler: StatusHandler):
    prompt = agent_prompts.get_market_prompt(ticker, av_data, status_handler)
    status_handler.log("   -> Market Agent is processing...")
    response = market_model.generate_content(prompt)
    status_handler.log("Market analysis complete.")
    return {"text": response.candidates[0].content.parts[0].text}

def run_competitor_agent(market_model, ticker, av_data, status_handler: StatusHandler):
    prompt = agent_prompts.get_competitor_prompt(ticker, av_data, status_handler)
    status_handler.log("   -> Competitor Agent is processing...")
    response = market_model.generate_content(prompt)
    status_handler.log("Competitor analysis complete.")
    return {"text": response.candidates[0].content.parts[0].text}

def run_deep_dive_agent(synthesis_model, rag_model, ticker, report, status_handler: StatusHandler):
    prompt = agent_prompts.get_deep_dive_prompt(
        ticker,
        report['part_1_5_quant'],
        report['part_1_4_6_rag']['text'],
        report['part_2_3_market']['text'],
        report.get('part_6_competitor', {'text': 'No competitor data.'})['text'],
        status_handler
    )
    status_handler.log("Generated deep dive questions. Now investigating...")
    response = synthesis_model.generate_content(prompt)
    questions = response.candidates[0].content.parts[0].text.strip().split('\n')
    answers = []
    for q in questions:
        if q.strip():
            status_handler.log(f"   ❓ Investigating: {q.strip()}")
            ans_resp = rag_model.generate_content(f"Answer this using 10-K: {q}")
            answers.append(f"Q: {q}\nA: {ans_resp.candidates[0].content.parts[0].text}")
            time.sleep(2) # Add a 2-second delay to avoid hitting API rate limits
    status_handler.log("Deep dive investigation complete.")
    return "\n\n".join(answers)

def run_synthesis_pipeline(synthesis_model, ticker, report, status_handler: StatusHandler):
    draft_prompt = agent_prompts.get_synthesis_prompt(
        ticker, report['part_1_5_quant'], report['part_1_4_6_rag']['text'],
        report['part_2_3_market']['text'], report.get('part_6_competitor', {'text': 'No competitor data.'})['text'],
        report['deep_dive_analysis'], status_handler, step_num=8.1
    )
    status_handler.update_step("Step 8.1: Drafting initial synthesis...")
    draft_response = synthesis_model.generate_content(draft_prompt)
    draft_text = clean_json_output(draft_response.candidates[0].content.parts[0].text)
    
    status_handler.update_step("Step 8.2: Critic Agent validating data points...")
    critic_prompt = agent_prompts.get_critic_prompt(ticker, draft_text, report['part_1_5_quant'], status_handler, step_num=8.2)
    final_response = synthesis_model.generate_content(critic_prompt)
    final_text = clean_json_output(final_response.candidates[0].content.parts[0].text)
    
    try:
        # The final output from the critic should be a clean JSON string.
        # We parse it here to return a dictionary.
        return json.loads(final_text)
    except (json.JSONDecodeError, TypeError):
        # If the critic's output is malformed, fall back to the original draft.
        return json.loads(draft_text)

# --- Main App ---

def main():
    setup_logging()
    # Initialize GCP services at the start of every run to ensure they are available.
    if 'gcp_services' not in st.session_state:
        st.session_state.gcp_services = initialize_gcp_services()

    if 'score_weights' not in st.session_state:
        # Initialize all required state keys to prevent KeyErrors
        st.session_state.score_weights = {'pillar_1_corporate': 0.35, 'pillar_2_industry': 0.20, 'pillar_3_valuation': 0.20, 'pillar_4_sentiment': 0.15, 'pillar_5_governance': 0.10}
        st.session_state.report = {}
        st.session_state.ticker = ""
        st.session_state.analysis_in_progress = False
        st.session_state.company_name = ""
        st.session_state.av_data = {}

    if 'score_weights' not in st.session_state:
        st.session_state.score_weights = {'pillar_1_corporate': 0.35, 'pillar_2_industry': 0.20, 'pillar_3_valuation': 0.20, 'pillar_4_sentiment': 0.15, 'pillar_5_governance': 0.10}

    st.title("PyFinAgent: AI Financial Analyst (Agentic)")
    st.caption("Powered by Alpha Vantage Data & Reflexion Architecture")

    services = st.session_state.gcp_services
    # Only display the sidebar reports if an analysis is NOT in progress to prevent crashes.
    # This function now handles all report loading and parsing logic.
    if not st.session_state.get('analysis_in_progress'):
        # Handle loading a past report if triggered from another page
        _handle_report_load(services['bq_client'], services['table_id']) # This now fixes the object type
        display_sidebar(services['bq_client'], services['table_id'], st.session_state.get("ticker"))
    
    def start_analysis_callback():
        """Sets the state to begin the analysis."""
        # Persist the ticker value for the analysis run
        st.session_state.analysis_ticker = st.session_state.ticker
        st.session_state.report = {}
        st.session_state.av_data = {}
        st.session_state.company_name = ""
        st.session_state.status_handler = None # Clear previous handler
        st.session_state.analysis_in_progress = True

    # Only draw the main UI if an analysis is NOT in progress
    if not st.session_state.get('analysis_in_progress'):
        def clear_state_callback():
            """Resets the session state for a new analysis, preserving services."""
            for key in ['ticker', 'report', 'analysis_in_progress', 'av_data']:
                if key in st.session_state: # Reset company_name as well
                    st.session_state[key] = "" if key == 'ticker' else {}
    
        col1, col2 = st.columns([4, 1])
        with col1: st.text_input("Enter Ticker:", key="ticker", label_visibility="collapsed", placeholder="e.g. NVDA")
        with col2: 
            st.button("Clear", on_click=clear_state_callback, use_container_width=True)
    
        st.button("Run Comprehensive Analysis", type="primary", on_click=start_analysis_callback, disabled=not st.session_state.get("ticker"), use_container_width=True)

    # --- EXECUTION LOOP ---
    if st.session_state.get('analysis_in_progress') and st.session_state.get('analysis_ticker'):
        ticker = st.session_state.analysis_ticker
        report = st.session_state.report

        # --- Display Company Name and Ticker ---
        # Show company name in subheader if available, otherwise default to ticker.
        company_name = st.session_state.get('company_name')
        header_text = company_name if company_name else ticker
        st.subheader(f"Analyzing: {header_text}")
        if company_name: # Only show the ticker underneath if we have the company name
            st.caption(f"Ticker: {ticker}")
        # st.divider() # Removed as requested to clean up the analysis view.

        # Define the total number of steps for the progress bar
        TOTAL_STEPS = 8

        # Create placeholders for the new layout
        progress_bar_placeholder = st.empty()

        # Initialize or retrieve the status handler from session state
        if 'status_handler' not in st.session_state or st.session_state.status_handler is None:
            st.session_state.status_handler = StatusHandler(total_steps=TOTAL_STEPS, progress_bar=progress_bar_placeholder)
        status_handler = st.session_state.status_handler

        try:
            # Step 1: Data Ingestion
            if 'ingestion_complete' not in report:
                status_handler.update_step("Step 1: Ingesting SEC Filings...")
                if not run_ingestion_agent(ticker, status_handler):
                    raise Exception("Data Ingestion Failed")
                report['ingestion_complete'] = True
                st.rerun()

            # Step 2: Fetch Alpha Vantage Market Intel
            if 'av_data' not in st.session_state or not st.session_state.av_data:
                status_handler.update_step("Step 2: Fetching Market Intelligence...")
                st.session_state.av_data = tools_alphavantage.get_market_intel(ticker, status_handler)
                st.rerun()

            # Step 3: Quantitative Analysis
            if 'part_1_5_quant' not in report:
                status_handler.update_step("Step 3: Running Quantitative Analysis...")
                report['part_1_5_quant'] = run_quant_agent(ticker, status_handler)
                st.rerun()
            
            # Step 4: RAG Analysis on 10-K Filings
            if 'part_1_4_6_rag' not in report:
                status_handler.update_step("Step 4: Analyzing SEC Filings (RAG)...")
                report['part_1_4_6_rag'] = run_rag_agent(services['rag_model'], ticker, status_handler)
                st.rerun()

            # Step 5: Market News Analysis
            if 'part_2_3_market' not in report:
                status_handler.update_step("Step 5: Analyzing Market News & Sentiment...")
                report['part_2_3_market'] = run_market_agent(services['market_model'], ticker, st.session_state.av_data, status_handler)
                st.rerun()

            # Step 6: Competitor Analysis
            if 'part_6_competitor' not in report:
                status_handler.update_step("Step 6: Identifying and Analyzing Competitors...")
                report['part_6_competitor'] = run_competitor_agent(services['market_model'], ticker, st.session_state.av_data, status_handler)
                st.rerun()

            # Step 7: Deep Dive Contradiction Analysis
            if 'deep_dive_analysis' not in report:
                status_handler.update_step("Step 7: Performing Deep-Dive Contradiction Analysis...")
                report['deep_dive_analysis'] = run_deep_dive_agent(
                    services['synthesis_model'], services['rag_model'], ticker, report, status_handler
                )
                st.rerun()

            # Step 8: Final Synthesis and Scoring
            if 'final_synthesis' not in report:
                # This step is now broken into sub-steps within the pipeline and below
                final_json = run_synthesis_pipeline(
                    services['synthesis_model'], ticker, st.session_state.report, status_handler
                )
                scores = final_json.get('scoring_matrix', {})
                status_handler.update_step("Step 8.3: Calculating final weighted score...")
                w = st.session_state.score_weights
                final_score = sum(scores.get(k, 0) * v for k, v in w.items())
                final_json['final_weighted_score'] = round(final_score, 2)
                report['final_synthesis'] = final_json
                st.rerun()

            # Final Step: Save the report
            if 'report_saved' not in report:
                status_handler.update_step("Step 8.4: Saving report to BigQuery...")
                
                # Define the row dictionary before the try block to ensure it always exists.
                company_name = report.get('part_1_5_quant', {}).get('company_name', 'N/A')
                row = {"ticker": ticker, "company_name": company_name, "analysis_date": datetime.now().isoformat(), "final_score": report['final_synthesis']['final_weighted_score'], "recommendation": report['final_synthesis'].get('recommendation', {}).get('action', 'N/A'), "summary": report['final_synthesis'].get('final_summary', ''), "full_report_json": json.dumps(report)}
                
                try:
                    errors = services['bq_client'].insert_rows_json(services['table_id'], [row])
                    if errors: logging.error(f"BQ Error: {errors}")
                except Exception as e:
                    logging.error(f"Save failed: {e}")

                # The st.session_state.report is already a dictionary.
                # The problematic line that caused the crash has been removed.
                st.session_state.report['report_saved'] = True
                status_handler.complete()
                st.session_state.analysis_in_progress = False
                st.rerun()

        except Exception as e:
            status_handler.error(f"Analysis halted: {e}")
            st.write(traceback.format_exc())
            st.session_state.analysis_in_progress = False
            logging.error("Pipeline failed", exc_info=True)

    # --- Render Final Report ---
    if st.session_state.get('report', {}).get('final_synthesis'):
        display_report()

if __name__ == "__main__":
    main()