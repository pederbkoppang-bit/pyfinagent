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
from components.progress_bar import initialize_status_elements, update_progress, clear_progress
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

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        display_radar_chart()
    with chart_col2:
        display_price_chart()
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Final Score", f"{report_data.get('final_weighted_score', 0):.2f} / 10")
        st.metric("Recommendation", report_data.get('recommendation', {}).get('action', 'N/A'))
        st.caption(f"at {price_str}")

    with col2:
        st.subheader("Justification")
        st.write(report_data.get('recommendation', {}).get('justification', 'No justification provided.'))

    st.subheader("Final Summary")
    st.write(report_data.get('final_summary', 'No summary provided.'))
    
    with st.expander("View Full Raw Data (JSON)"):
        st.json(st.session_state.report, expanded=True)

def _handle_report_load(bq_client, table_id):
    """
    Checks session state for a report to load (triggered from sidebar or past reports page),
    queries BigQuery for the full report data, and populates the main
    session state 'report' object.
    """
    # This function handles loading a single report.
    if 'report_to_load' in st.session_state and st.session_state.report_to_load:
        report_info = st.session_state.report_to_load
        ticker = report_info['Ticker']
        analysis_date = report_info['Analysis Date']
        
        query = f"""
            SELECT full_report_json
            FROM `{table_id}`
            WHERE ticker = @ticker AND analysis_date = @analysis_date
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("ticker", "STRING", ticker),
                bigquery.ScalarQueryParameter("analysis_date", "TIMESTAMP", analysis_date),
            ]
        )
        query_job = bq_client.query(query, job_config=job_config)
        results = list(query_job.result())
        
        if results:
            # The full report is stored as a JSON string in BigQuery
            st.session_state.report = json.loads(results[0].full_report_json)
        
        # Clear the flag to prevent re-loading on the next rerun
        del st.session_state.report_to_load

def run_ingestion_agent(ticker: str, status):
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
            if line:
                log_message = line.decode('utf-8')
                # Update the status container's label with the latest log message.
                status.update(label=log_message)
                # Check for the specific success message to confirm completion.
                if "Ingestion process completed successfully." in log_message:
                    success = True
                    break # Exit the loop as we have confirmation of success.

                # Check for failure messages to stop early.
                if "failed" in log_message.lower() or "error" in log_message.lower():
                    success = False # Explicitly mark as failed
                    break # Exit the loop on failure
            
        return success
    except Exception as e:
        st.error(f"Failed to connect to or stream from ingestion agent: {e}")
        return False

@st.cache_data(ttl="1h")
def run_quant_agent(ticker: str, _status) -> dict:
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
                    _status.update(label=log_message)

        if final_data:
            # Enhance Quant Agent with Alpha Vantage Overview after getting the data
            _status.update(label="Fetching additional overview from Alpha Vantage...")
            av_overview = tools_alphavantage.get_fundamental_overview(ticker)
            if "part_5_valuation" in final_data and isinstance(av_overview, dict):
                final_data["part_5_valuation"]["alpha_vantage_overview"] = av_overview
            _status.update(label="✅ Quant data acquired.")
            return final_data
        else:
            raise ValueError("Quant Agent did not return a final JSON payload.")
    except Exception as e:
        st.error(f"Quant Agent failed: {e}")
        raise

def run_rag_agent(rag_model, ticker, status):
    status.update(label="📑 RAG Agent analyzing documents...")
    prompt = agent_prompts.get_rag_prompt(ticker)
    response = rag_model.generate_content(prompt)
    status.update(label="✅ RAG analysis complete.")
    return {"text": response.text}

def run_market_agent(market_model, ticker, av_data, status):
    status.update(label="🌍 Market Agent analyzing Alpha Vantage sentiment feed...")
    prompt = agent_prompts.get_market_prompt(ticker, av_data)
    response = market_model.generate_content(prompt)
    status.update(label="✅ Market analysis complete.")
    return {"text": response.text}

def run_competitor_agent(market_model, ticker, av_data, status):
    status.update(label="⚔️ Competitor Scout analyzing co-occurring rivals...")
    prompt = agent_prompts.get_competitor_prompt(ticker, av_data)
    response = market_model.generate_content(prompt)
    status.update(label="✅ Competitor analysis complete.")
    return {"text": response.text}

def run_deep_dive_agent(synthesis_model, rag_model, ticker, report, status):
    status.update(label="🕵️ Deep Dive Agent: Looking for contradictions...")
    prompt = agent_prompts.get_deep_dive_prompt(
        ticker, 
        report['part_1_5_quant'],
        report['part_1_4_6_rag']['text'],
        report['part_2_3_market']['text'],
        report.get('part_6_competitor', {'text': 'No competitor data.'})['text']
    )
    response = synthesis_model.generate_content(prompt)
    questions = response.text.strip().split('\n')
    answers = []
    for q in questions:
        if q.strip():
            status.update(label=f"   ❓ Investigating: {q.strip()}")
            ans_resp = rag_model.generate_content(f"Answer this using 10-K: {q}")
            answers.append(f"Q: {q}\nA: {ans_resp.text}")
            time.sleep(2) # Add a 2-second delay to avoid hitting API rate limits
    return "\n\n".join(answers)

def run_synthesis_pipeline(synthesis_model, ticker, report, status):
    status.update(label="✍️ Lead Analyst: Drafting thesis...")
    draft_prompt = agent_prompts.get_synthesis_prompt(
        ticker,
        report['part_1_5_quant'],
        report['part_1_4_6_rag']['text'],
        report['part_2_3_market']['text'],
        report.get('part_6_competitor', {'text': 'No competitor data.'})['text'],
        report['deep_dive_analysis']
    )
    draft_response = synthesis_model.generate_content(draft_prompt)
    draft_text = clean_json_output(draft_response.text)
    
    status.update(label="🧐 Critic Agent: Validating data points...")
    critic_prompt = agent_prompts.get_critic_prompt(ticker, draft_text, report['part_1_5_quant'])
    final_response = synthesis_model.generate_content(critic_prompt)
    final_text = clean_json_output(final_response.text)
    
    try:
        return json.loads(final_text)
    except:
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
        st.session_state.av_data = {}

    if 'score_weights' not in st.session_state:
        st.session_state.score_weights = {'pillar_1_corporate': 0.35, 'pillar_2_industry': 0.20, 'pillar_3_valuation': 0.20, 'pillar_4_sentiment': 0.15, 'pillar_5_governance': 0.10}

    st.title("PyFinAgent: AI Financial Analyst (Agentic)")
    st.caption("Powered by Alpha Vantage Data & Reflexion Architecture")

    services = st.session_state.gcp_services
    # Only display the sidebar reports if an analysis is NOT in progress to prevent crashes.
    if not st.session_state.get('analysis_in_progress'):
        # Handle loading a past report if triggered from another page
        _handle_report_load(services['bq_client'], services['table_id'])
        display_sidebar(services['bq_client'], services['table_id'], st.session_state.get("ticker"))

    # This check MUST run on every script run to handle navigation from past reports.
    if 'report' in st.session_state and isinstance(st.session_state.report.get('final_synthesis'), str):
        try:
            st.session_state.report['final_synthesis'] = json.loads(st.session_state.report['final_synthesis'])
        except (json.JSONDecodeError, TypeError):
            st.error("Failed to parse report data from previous page. Please start a new analysis.")
            st.session_state.report = {} # Clear corrupted report
    
    def start_analysis_callback():
        """Sets the state to begin the analysis."""
        # Persist the ticker value for the analysis run
        st.session_state.analysis_ticker = st.session_state.ticker
        st.session_state.report = {}
        st.session_state.av_data = {}
        st.session_state.analysis_in_progress = True

    # Only draw the main UI if an analysis is NOT in progress
    if not st.session_state.get('analysis_in_progress'):
        def clear_state_callback():
            """Resets the session state for a new analysis, preserving services."""
            for key in ['ticker', 'report', 'analysis_in_progress', 'av_data']:
                if key in st.session_state:
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
        
        # Define the total number of steps for the progress bar
        TOTAL_STEPS = 8

        initialize_status_elements()
        with st.status("Agent Thought Process", expanded=True) as status:
            try:
                # Step 0: Data Ingestion
                if 'ingestion_complete' not in report:
                    update_progress(int(1/TOTAL_STEPS * 100), "Step 1: Ingesting SEC Filings...")
                    status.update(label="Starting data ingestion...")
                    if not run_ingestion_agent(ticker, status):
                        raise Exception("Data Ingestion Failed")
                    report['ingestion_complete'] = True
                    st.rerun()

                # Step 1: Fetch Alpha Vantage Market Intel
                if 'av_data' not in st.session_state or not st.session_state.av_data:
                    update_progress(int(2/TOTAL_STEPS * 100), "Step 2: Fetching Market Intelligence...")
                    status.update(label="📡 Fetching Alpha Vantage Market Intel...")
                    st.session_state.av_data = tools_alphavantage.get_market_intel(ticker)
                    st.rerun()

                # Step 2: Quantitative Analysis
                if 'part_1_5_quant' not in report:
                    update_progress(int(3/TOTAL_STEPS * 100), "Step 3: Running Quantitative Analysis...")
                    report['part_1_5_quant'] = run_quant_agent(ticker, status)
                    st.rerun()
                
                # Step 3: RAG Analysis on 10-K Filings
                if 'part_1_4_6_rag' not in report:
                    update_progress(int(4/TOTAL_STEPS * 100), "Step 4: Analyzing SEC Filings (RAG)...")
                    report['part_1_4_6_rag'] = run_rag_agent(services['rag_model'], ticker, status)
                    st.rerun()

                # Step 4: Market News Analysis
                if 'part_2_3_market' not in report:
                    update_progress(int(5/TOTAL_STEPS * 100), "Step 5: Analyzing Market News & Sentiment...")
                    report['part_2_3_market'] = run_market_agent(services['market_model'], ticker, st.session_state.av_data, status)
                    st.rerun()

                # Step 5: Competitor Analysis
                if 'part_6_competitor' not in report:
                    update_progress(int(6/TOTAL_STEPS * 100), "Step 6: Identifying and Analyzing Competitors...")
                    report['part_6_competitor'] = run_competitor_agent(services['market_model'], ticker, st.session_state.av_data, status)
                    st.rerun()

                # Step 6: Deep Dive Contradiction Analysis
                if 'deep_dive_analysis' not in report:
                    update_progress(int(7/TOTAL_STEPS * 100), "Step 7: Performing Deep-Dive Contradiction Analysis...")
                    report['deep_dive_analysis'] = run_deep_dive_agent(
                        services['synthesis_model'], services['rag_model'], ticker, report, status
                    )
                    st.rerun()

                # Step 7: Final Synthesis and Scoring
                if 'final_synthesis' not in report:
                    update_progress(int(8/TOTAL_STEPS * 100), "Step 8: Synthesizing Final Report and Scoring...")
                    final_json = run_synthesis_pipeline(
                        services['synthesis_model'], ticker, st.session_state.report, status
                    )
                    scores = final_json.get('scoring_matrix', {})
                    w = st.session_state.score_weights
                    final_score = sum(scores.get(k, 0) * v for k, v in w.items())
                    final_json['final_weighted_score'] = round(final_score, 2)
                    report['final_synthesis'] = final_json
                    update_progress(100, "Finalizing Report...")
                    st.rerun()

                # Step 8: Save the final report to BigQuery
                if 'report_saved' not in report:
                    status.update(label="Saving report...")
                    try:
                        company_name = report.get('part_1_5_quant', {}).get('company_name', 'N/A')
                        row = {"ticker": ticker, "company_name": company_name, "analysis_date": datetime.now().isoformat(), "final_score": report['final_synthesis']['final_weighted_score'], "recommendation": report['final_synthesis'].get('recommendation', {}).get('action', 'N/A'), "summary": report['final_synthesis'].get('final_summary', ''), "full_report_json": json.dumps(report)}
                        errors = services['bq_client'].insert_rows_json(services['table_id'], [row])
                        if errors: logging.error(f"BQ Error: {errors}")
                    except Exception as e:
                        logging.error(f"Save failed: {e}")

                    st.session_state.report['report_saved'] = True
                    status.update(label="Analysis Complete!", state="complete", expanded=False)
                    clear_progress() # Clear the progress bar on completion
                    st.session_state.analysis_in_progress = False
                    st.rerun()

            except Exception as e:
                status.update(label="Analysis Failed", state="error")
                st.error(f"Analysis halted: {e}")
                st.write(traceback.format_exc())
                st.session_state.analysis_in_progress = False
                clear_progress() # Clear the progress bar on failure
                logging.error("Pipeline failed", exc_info=True)

    # --- Render Final Report ---
    if st.session_state.get('report', {}).get('final_synthesis'):
        display_report()

if __name__ == "__main__":
    main()