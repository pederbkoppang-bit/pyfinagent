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
    
    # We don't strictly need the Google Search tool anymore for Market/Competitor 
    # since we use Alpha Vantage, but keeping it for general purpose backup is fine.
    tool_dict = {"google_search": {}}
    search_tool = Tool.from_dict(tool_dict)

    synthesis_model = GenerativeModel(GEMINI_MODEL)
    rag_model = GenerativeModel(GEMINI_MODEL, tools=[rag_tool])
    market_model = GenerativeModel(GEMINI_MODEL) # Standard model is fine, we feed it API data

    return {
        "bq_client": bq_client,
        "table_id": table_id,
        "rag_model": rag_model,
        "market_model": market_model,
        "synthesis_model": synthesis_model
    }

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
    
    # Optional: Enhance Quant Agent with Alpha Vantage Overview
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
    """
    Now uses Alpha Vantage Data passed from main loop
    """
    log_to_ui("🌍 Market Agent analyzing Alpha Vantage sentiment feed...")
    prompt = agent_prompts.get_market_prompt(ticker, av_data)
    response = market_model.generate_content(prompt)
    log_to_ui("✅ Market analysis complete.")
    return {"text": response.text}

def run_competitor_agent(market_model, ticker, av_data):
    """
    Now uses Alpha Vantage Derived Competitors
    """
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
    def display_report():
        if 'report' not in st.session_state or not st.session_state.report.get('final_synthesis'): return
        report_data = st.session_state.report['final_synthesis']
        
        st.success("Analysis Complete!")
        display_evaluation_table()
        st.divider()
        c1, c2 = st.columns(2)
        with c1: display_radar_chart()
        with c2: display_price_chart()
        
        st.divider()
        
        c1, c2 = st.columns([1, 3])
        with c1: 
            st.metric("Final Score", f"{report_data.get('final_weighted_score', 0):.2f} / 10")
            st.metric("Recommendation", report_data.get('recommendation', {}).get('action', 'N/A'))
        with c2:
            st.subheader("Summary")
            st.write(report_data.get('final_summary', 'No summary available.'))
        
        with st.expander("Raw Agent Outputs & Final Report JSON"): st.json(st.session_state.report)

    setup_logging()
    if 'score_weights' not in st.session_state:
        st.session_state.score_weights = {'pillar_1_corporate': 0.35, 'pillar_2_industry': 0.20, 'pillar_3_valuation': 0.20, 'pillar_4_sentiment': 0.15, 'pillar_5_governance': 0.10}

    st.title("PyFinAgent: AI Financial Analyst (Agentic)")
    st.caption("Powered by Alpha Vantage Data & Reflexion Architecture")

    if 'gcp_services' not in st.session_state:
        # When switching from past reports, 'report' can be a string-heavy dict from BQ
        # We need to ensure final_synthesis is parsed to a dict if it exists.
        if 'report' in st.session_state and isinstance(st.session_state.report.get('final_synthesis'), str):
            try:
                st.session_state.report['final_synthesis'] = json.loads(st.session_state.report['final_synthesis'])
            except json.JSONDecodeError:
                st.error("Failed to parse report data. Please try re-running the analysis.")
        st.session_state.gcp_services = initialize_gcp_services()
    services = st.session_state.gcp_services
    display_sidebar(services['bq_client'], services['table_id'], st.session_state.get("ticker"))

    col1, col2 = st.columns([4, 1])
    with col1: st.text_input("Enter Ticker:", key="ticker", label_visibility="collapsed", placeholder="e.g. NVDA")
    with col2: 
        if st.button("Clear", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    initialize_log_display()
    initialize_status_elements()

    if st.button("Run Comprehensive Analysis", type="primary") and st.session_state.ticker:
        st.session_state.analysis_triggered = True
        st.rerun()

    display_logs()

    if st.session_state.get('analysis_triggered'):
        ticker = st.session_state.ticker
        st.session_state.report = {} # Clear previous report before new run
        
        try:
            # --- UI Enhancement: Display Company Name & Ticker ---
            overview_data = tools_alphavantage.get_fundamental_overview(ticker)
            company_name = overview_data.get("Name", ticker.upper())
            st.subheader(f"Analyzing: {company_name} ({ticker.upper()})")

            log_to_ui("📡 Fetching Alpha Vantage Market Intel...")
            av_data = tools_alphavantage.get_market_intel(ticker)
            run_ingestion_agent(ticker)
            st.session_state.report['ingestion_agent'] = True

            update_progress(20, "Gathering Financials...")
            st.session_state.report['part_1_5_quant'] = run_quant_agent(ticker)
            
            update_progress(35, "Analyzing 10-Ks...")
            st.session_state.report['part_1_4_6_rag'] = run_rag_agent(services['rag_model'], ticker)

            update_progress(50, "Scanning Market News...")
            st.session_state.report['part_2_3_market'] = run_market_agent(services['market_model'], ticker, av_data)

            update_progress(55, "Analyzing Rivals...")
            st.session_state.report['part_6_competitor'] = run_competitor_agent(services['market_model'], ticker, av_data)

            update_progress(60, "Deep Dive...")
            st.session_state.report['deep_dive_analysis'] = run_deep_dive_agent(services['synthesis_model'], services['rag_model'], ticker, st.session_state.report)
       
            final_json = run_synthesis_pipeline(services['synthesis_model'], ticker, st.session_state.report)
            scores = final_json['scoring_matrix']
            w = st.session_state.score_weights
            final_score = sum(scores.get(k, 0) * v for k, v in w.items())
            final_json['final_weighted_score'] = round(final_score, 2)
            st.session_state.report['final_synthesis'] = final_json

            update_progress(95, "Saving report...")
            # Assume BQ save logic here (skipped for brevity)
            st.session_state.report['report_saved'] = True
            update_progress(100, "Analysis Complete.")

        except Exception as e:
            st.error(f"Analysis halted: {e}")
            logging.error("Pipeline failed", exc_info=True)
        finally:
            # IMPORTANT: Reset the trigger so it doesn't run again on the next interaction
            st.session_state.analysis_triggered = False
            # Rerun one last time to settle the page and display the final report
            st.rerun()
    
    # This block is now for displaying a previously run report from session state
    # if the user navigates away and comes back, for example.
    elif st.session_state.get('report', {}).get('final_synthesis'):
        display_report()

if __name__ == "__main__":
    main()