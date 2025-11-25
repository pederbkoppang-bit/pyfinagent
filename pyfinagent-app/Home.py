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
from components.log_display import initialize_log_display, log_to_ui, clear_log_display, display_logs
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

# --- Agent Functions ---

def run_ingestion_agent(ticker: str):
    log_to_ui(f"🔍 Checking filings for {ticker}...")
    try:
        INGESTION_AGENT_URL = st.secrets.agent.ingestion_agent_url
        requests.post(INGESTION_AGENT_URL, json={'ticker': ticker}, timeout=900)
        log_to_ui("✅ Ingestion check complete.")
    except Exception as e:
        log_to_ui("⚠️ Ingestion agent warning.")

@st.cache_data(ttl="1h")
def run_quant_agent(ticker: str) -> dict:
    log_to_ui(f"📊 Running Quant Agent for {ticker}...")
    
    # Enhance Quant Agent with Alpha Vantage Overview
    av_overview = tools_alphavantage.get_fundamental_overview(ticker)
    
    QUANT_AGENT_URL = st.secrets.agent.quant_agent_url
    try:
        response = requests.get(f"{QUANT_AGENT_URL}?ticker={ticker}", timeout=300)
        data = response.json()
        if "error" in data: raise ValueError(data["error"])
        
        # Merge Alpha Vantage Data into Report
        if "part_5_valuation" in data and isinstance(av_overview, dict):
            data["part_5_valuation"]["alpha_vantage_overview"] = av_overview
            
        log_to_ui("✅ Quant data acquired.")
        return data
    except Exception as e:
        st.error(f"Quant Agent failed: {e}")
        raise

def run_rag_agent(rag_model, ticker):
    log_to_ui("📑 RAG Agent analyzing documents...")
    prompt = agent_prompts.get_rag_prompt(ticker)
    response = rag_model.generate_content(prompt)
    log_to_ui("✅ RAG analysis complete.")
    return {"text": response.text}

def run_market_agent(market_model, ticker, av_data):
    log_to_ui("🌍 Market Agent analyzing Alpha Vantage sentiment feed...")
    prompt = agent_prompts.get_market_prompt(ticker, av_data)
    response = market_model.generate_content(prompt)
    log_to_ui("✅ Market analysis complete.")
    return {"text": response.text}

def run_competitor_agent(market_model, ticker, av_data):
    log_to_ui("⚔️ Competitor Scout analyzing co-occurring rivals...")
    prompt = agent_prompts.get_competitor_prompt(ticker, av_data)
    response = market_model.generate_content(prompt)
    log_to_ui("✅ Competitor analysis complete.")
    return {"text": response.text}

def run_deep_dive_agent(synthesis_model, rag_model, ticker, report):
    log_to_ui("🕵️ Deep Dive Agent: Looking for contradictions...")
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
            log_to_ui(f"   ❓ Investigating: {q.strip()}")
            ans_resp = rag_model.generate_content(f"Answer this using 10-K: {q}")
            answers.append(f"Q: {q}\nA: {ans_resp.text}")
    return "\n\n".join(answers)

def run_synthesis_pipeline(synthesis_model, ticker, report):
    update_progress(70, "Lead Analyst drafting report...")
    log_to_ui("✍️ Lead Analyst: Drafting thesis...")
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
    
    update_progress(85, "Critic reviewing...")
    log_to_ui("🧐 Critic Agent: Validating data points...")
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
    if 'score_weights' not in st.session_state:
        st.session_state.score_weights = {'pillar_1_corporate': 0.35, 'pillar_2_industry': 0.20, 'pillar_3_valuation': 0.20, 'pillar_4_sentiment': 0.15, 'pillar_5_governance': 0.10}

    st.title("PyFinAgent: AI Financial Analyst (Agentic)")
    st.caption("Powered by Alpha Vantage Data & Reflexion Architecture")

    if 'gcp_services' not in st.session_state:
        st.session_state.gcp_services = initialize_gcp_services()
    services = st.session_state.gcp_services
    display_sidebar(services['bq_client'], services['table_id'], st.session_state.get("ticker"))

    col1, col2 = st.columns([4, 1])
    with col1: st.text_input("Enter Ticker:", key="ticker", label_visibility="collapsed", placeholder="e.g. NVDA")
    with col2: 
        if st.button("Clear", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    if st.button("Run Comprehensive Analysis", type="primary"):
        st.session_state.analysis_in_progress = True

    initialize_status_elements()
    initialize_log_display()
    display_logs()

    # --- EXECUTION LOOP ---
    if st.session_state.get('analysis_in_progress') and st.session_state.ticker:
        ticker = st.session_state.ticker
        if 'report' not in st.session_state: st.session_state.report = {}
        
        try:
            # 0. Fetch Alpha Vantage Data
            if 'av_data' not in st.session_state:
                log_to_ui("📡 Fetching Alpha Vantage Market Intel...")
                st.session_state.av_data = tools_alphavantage.get_market_intel(ticker)

            # 1. Pipeline
            if 'ingestion_agent' not in st.session_state.report:
                run_ingestion_agent(ticker)
                st.session_state.report['ingestion_agent'] = True

            if 'part_1_5_quant' not in st.session_state.report:
                update_progress(20, "Gathering Financials...")
                st.session_state.report['part_1_5_quant'] = run_quant_agent(ticker)
            
            if 'part_1_4_6_rag' not in st.session_state.report:
                update_progress(35, "Analyzing 10-Ks...")
                st.session_state.report['part_1_4_6_rag'] = run_rag_agent(services['rag_model'], ticker)

            if 'part_2_3_market' not in st.session_state.report:
                update_progress(50, "Scanning Market News...")
                st.session_state.report['part_2_3_market'] = run_market_agent(services['market_model'], ticker, st.session_state.av_data)

            if 'part_6_competitor' not in st.session_state.report:
                update_progress(55, "Analyzing Rivals...")
                st.session_state.report['part_6_competitor'] = run_competitor_agent(services['market_model'], ticker, st.session_state.av_data)

            if 'deep_dive_analysis' not in st.session_state.report:
                update_progress(60, "Deep Dive...")
                st.session_state.report['deep_dive_analysis'] = run_deep_dive_agent(services['synthesis_model'], services['rag_model'], ticker, st.session_state.report)

            if 'final_synthesis' not in st.session_state.report:
                final_json = run_synthesis_pipeline(services['synthesis_model'], ticker, st.session_state.report)
                # Score Calculation
                scores = final_json.get('scoring_matrix', {})
                w = st.session_state.score_weights
                final_score = sum(scores.get(k, 0) * v for k, v in w.items())
                final_json['final_weighted_score'] = round(final_score, 2)
                st.session_state.report['final_synthesis'] = final_json

            if 'report_saved' not in st.session_state.report:
                update_progress(95, "Saving report...")
                
                # --- BIGQUERY SAVE LOGIC ---
                try:
                    company_name = st.session_state.report.get('part_1_5_quant', {}).get('company_name', 'N/A')
                    row = {
                        "ticker": ticker, 
                        "company_name": company_name, 
                        "analysis_date": datetime.now().isoformat(), 
                        "final_score": st.session_state.report['final_synthesis']['final_weighted_score'], 
                        "recommendation": st.session_state.report['final_synthesis'].get('recommendation', {}).get('action', 'N/A'), 
                        "summary": st.session_state.report['final_synthesis'].get('final_summary', ''), 
                        "full_report_json": json.dumps(st.session_state.report)
                    }
                    errors = services['bq_client'].insert_rows_json(services['table_id'], [row])
                    if errors: logging.error(f"BQ Error: {errors}")
                except Exception as e:
                    logging.error(f"Save failed: {e}")

                st.session_state.report['report_saved'] = True
                update_progress(100, "Analysis Complete.")
                st.session_state.analysis_in_progress = False
                st.rerun()

        except Exception as e:
            st.error(f"Analysis halted: {e}")
            st.write(traceback.format_exc())
            st.session_state.analysis_in_progress = False
            logging.error("Pipeline failed", exc_info=True)

    # --- Render Report ---
    if st.session_state.get('report', {}).get('final_synthesis'):
        display_report()

if __name__ == "__main__":
    main()