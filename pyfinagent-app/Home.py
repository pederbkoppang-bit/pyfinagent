import sys
import os

# Ensure the app's root directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import streamlit as st
import requests
import json
import re
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel, Tool, grounding
from google.oauth2 import service_account
from google.api_core import exceptions
import traceback
from datetime import datetime
import vertexai
import pandas as pd

import time
from components.sidebar import display_sidebar
from components.stock_chart import display_price_chart
from components.status_handler import StatusHandler
from components.reports_comparison import display_reports_comparison
from components.glass_box_dashboard import display_glass_box_dashboard
import agent_prompts
from tools import alphavantage as tools_alphavantage
from tools import yfinance as tools_yfinance
from tools import slack as tools_slack

# --- Overhauled Structured Logging ---
class JsonFormatter(logging.Formatter):
    """Formats log records into a JSON string for structured logging."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "file": record.pathname,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if hasattr(record, 'context'):
            log_record.update(record.context)
        if record.exc_info:
            # Add stack trace for errors
            log_record['stack_trace'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logging():
    """Sets up a global JSON-formatted logger."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
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
        LOCATION = st.secrets.gcp.vertex_ai_location # This should exist
    except AttributeError as e:
        st.error(f"Missing critical GCP secrets (project_id or vertex_ai_location): {e}")
        st.stop()

    credentials = None
    try:
        # Recommended: Use a single secret for the entire JSON keyfile
        credentials_str = st.secrets.gcp.credentials
        if not credentials_str:
            raise AttributeError("gcp.credentials secret is empty.")
            
        credentials_info = json.loads(credentials_str)
        
        # --- Robustness Check ---
        # The error "missing 'email' field" often means the JSON is malformed or
        # the wrong fields are present. We explicitly check for standard fields.
        if "client_email" not in credentials_info or "private_key" not in credentials_info:
            raise ValueError("Service account JSON is missing 'client_email' or 'private_key'.")
        if credentials_info.get("project_id") != PROJECT_ID:
            logging.warning(f"Project ID mismatch: Secret expects '{PROJECT_ID}', but keyfile has '{credentials_info.get('project_id')}'. Using secret's ID.")

        credentials = service_account.Credentials.from_service_account_info(credentials_info)
        logging.info("Successfully authenticated using service account from st.secrets.")
    except (AttributeError, json.JSONDecodeError, ValueError) as e:
        logging.warning("st.secrets.gcp.credentials not found or invalid. Falling back to Application Default Credentials.")
    # --- Custom CSS for Bento Grid, Citations, and Mobile ---
    st.markdown("""
        <style>
            /* Main container styling */
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            /* Bento Grid Card Styling */
            .bento-card {
                background-color: #1a1a1a;
                border: 1px solid #2e2e2e;
                border-radius: 12px;
                padding: 20px;
                height: 100%;
                display: flex;
                flex-direction: column;
            }
            .bento-card h3 {
                font-size: 1.1rem;
                color: #a0a0a0;
                margin-bottom: 15px;
            }
            /* Citation Styling */
            .citation {
                display: inline-block;
                font-size: 0.7rem;
                font-weight: bold;
                color: #00aaff;
                background-color: #003366;
                border-radius: 4px;
                padding: 2px 5px;
                vertical-align: super;
                cursor: pointer;
            }
            /* Custom LLM-generated Citation Styling */
            .llm-citation {
                display: inline-block;
                font-size: 0.8rem;
                font-weight: 500;
                color: #c7d2fe; /* A lighter blue/purple for distinction */
                background-color: rgba(34, 211, 238, 0.1); /* Cyan with transparency */
                border-radius: 5px;
                padding: 2px 6px;
            }
        </style>
    """, unsafe_allow_html=True)

    RAG_DATA_STORE_ID = st.secrets.agent.rag_data_store_id
    GEMINI_MODEL = st.secrets.agent.gemini_model

    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials) # credentials can be None
    bq_client = bigquery.Client(project=PROJECT_ID, credentials=credentials) # credentials can be None
    table_id = f"{PROJECT_ID}.financial_reports.analysis_results"

    datastore_path = (f"projects/{PROJECT_ID}/locations/global/collections/default_collection/"
                      f"dataStores/{RAG_DATA_STORE_ID}")
    
    rag_tool = Tool.from_retrieval(grounding.Retrieval(grounding.VertexAISearch(datastore=datastore_path)))
    
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

def _render_citations(text: str, citations: list) -> str:
    """
    Parses text for citation markers like [1] and replaces them with
    Streamlit popovers containing the source information.
    """
    if not citations:
        return text

    # Create a mapping from citation number to source details
    citation_map = {f"[{i+1}]": c for i, c in enumerate(citations)}

    # Use a function for replacement to handle popovers, which can't be in a simple f-string
    def replace_match(match):
        marker = match.group(0)
        if marker in citation_map:
            source = citation_map[marker]
            uri = source.get('uri', 'N/A')
            title = source.get('title', 'No Title')
            
            # Using a unique key for each popover
            popover_key = f"popover_{marker.strip('[]')}_{hash(text[:20])}"
            
            # This is a conceptual representation. Streamlit doesn't allow embedding widgets
            # directly into a string like this. We will handle this in the display logic.
            # For now, we return a styled placeholder.
            return f'<span class="citation" title="Source: {title} - {uri}">{marker}</span>'
        return marker

    # This approach is limited. A better way is to split the text and render parts.
    parts = re.split(r'(\[\d+\])', text)
    for part in parts:
        if re.match(r'(\[\d+\])', part):
            marker_num = int(part.strip('[]'))
            if 1 <= marker_num <= len(citations):
                source = citations[marker_num - 1]
                uri = source.get('uri', 'N/A')
                title = source.get('title', 'No Title')
                with st.popover(part):
                    st.markdown(f"**Source:** {title}")
                    st.caption(f"URI: {uri}")
        else:
            st.write(part, end="")
    st.write("") # for newline

def render_with_citations(text: str):
    """
    Uses regex to find custom citation patterns like [Source | Date] and
    replaces them with a styled HTML span for display with st.markdown.
    """
    if not isinstance(text, str):
        return

    def replace_citation(match):
        source, date = match.groups()
        return f'<span class="llm-citation" title="Source: {source.strip()} | Date: {date.strip()}">[{source.strip()}]</span>'

    pattern = r'\[([^|]+?)\s*\|\s*([^\]]+?)\]'
    formatted_text = re.sub(pattern, replace_citation, text)
    st.markdown(formatted_text, unsafe_allow_html=True)

def display_football_field_chart(price: float, valuation_data: dict):
    """Renders a Plotly Football Field chart for valuation ranges."""
    # This is a placeholder for the actual Plotly chart generation.
    # In a real implementation, you would use Plotly Express or Go to create the chart.
    st.info("🏈 Valuation Football Field Chart Placeholder")
    st.write(f"Current Price: ${price:.2f}")
    st.json(valuation_data)
    st.caption("This chart would show valuation ranges (e.g., P/E, DCF) against the current price.")

def display_report():
    """
    Performs data hybridization and renders the high-fidelity "Glass Box" dashboard.
    This function replaces the previous widget-based report display.
    """
    if 'report' not in st.session_state or not st.session_state.report.get('final_synthesis'):
        return

    report = st.session_state.report
    synthesis = report.get('final_synthesis', {})
    quant = report.get('part_1_5_quant', {})

    # 1. Data Hybridization
    try:
        # Process key_risks into a structured format for the 5x5 matrix
        processed_risks = []
        for i, risk_text in enumerate(synthesis.get('key_risks', [])):
            # Simple heuristic to assign coordinates for visualization
            # A more advanced implementation could use another LLM call for this
            impact = (hash(risk_text) % 5) + 1
            likelihood = ((hash(risk_text) >> 8) % 5) + 1
            processed_risks.append({
                "id": i,
                "text": risk_text.replace("🚨", "").strip(),
                "impact": impact,
                "likelihood": likelihood
            })

        # Safely access nested financial data
        part_1_financials = quant.get('part_1_financials') if quant.get('part_1_financials') is not None else {}
        yf_data = quant.get('yf_data') if quant.get('yf_data') is not None else {}

        # Create the final JSON payload for the frontend
        payload = {
            "ticker": st.session_state.ticker,
            "ai_score": synthesis.get('final_weighted_score', 0),
            "recommendation": synthesis.get('recommendation', {}),
            "summary": synthesis.get('final_summary', 'No summary available.'),
            "current_price": quant.get('yf_data', {}).get('valuation', {}).get('Current Price', 0),
            "financials": {
                "revenue": part_1_financials.get('latest_revenue', {}).get('value'),
                "net_income": part_1_financials.get('latest_net_income', {}).get('value'),
                "market_cap": yf_data.get('valuation', {}).get('Market Cap')
            },
            "valuation_ranges": quant.get('part_5_valuation', {}),
            "key_risks": processed_risks
        }

        # 2. Render the HTML Component
        display_glass_box_dashboard(payload)

    except Exception as e:
        logging.error(f"Failed to build Glass Box dashboard payload: {e}", exc_info=True)
        st.error("An error occurred while preparing the report display. Please see logs for details.")
        st.code(traceback.format_exc())

# --- Agent Functions ---

def run_ingestion_agent(ticker: str, status_handler: StatusHandler):
    """
    Calls the Ingestion Agent Cloud Function and streams logs to the UI.
    """
    status_handler.log(f"🔍 Ingestion Agent: Checking for new filings for {ticker}...")
    INGESTION_AGENT_URL = st.secrets.agent.ingestion_agent_url

    with requests.post(INGESTION_AGENT_URL, json={'ticker': ticker}, stream=True, timeout=900) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str == "STREAM_COMPLETE":
                    status_handler.log("   -> Ingestion complete.")
                    break
                elif line_str.startswith("ERROR:"):
                    raise Exception(f"Ingestion Agent Error: {line_str}")
                else:
                    # It's a log message, display it
                    status_handler.log(f"   -> {line_str}")

def run_quant_agent(ticker: str, status_handler: StatusHandler) -> dict:
    """
    Calls the Quant Agent Cloud Function, which now streams logs.
    This function processes the stream, passing logs to the status_handler
    and returning the final JSON data.
    """
    QUANT_AGENT_URL = st.secrets.agent.quant_agent_url
    final_json = None

    # Use stream=True to handle the streaming response
    with requests.get(f"{QUANT_AGENT_URL}?ticker={ticker}", stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("FINAL_JSON:"):
                    # Extract the JSON part and store it
                    json_str = line.split("FINAL_JSON:", 1)[1]
                    final_json = json.loads(json_str)
                elif line.startswith("ERROR:"):
                    # If the stream sends an error, raise it
                    raise Exception(line)
                else:
                    # Otherwise, it's a log message
                    status_handler.log(f"   -> {line}")

    if final_json is None:
        raise Exception("Quant Agent did not return the final JSON data.")

    # Merge yfinance data for a comprehensive report
    yf_data = tools_yfinance.get_comprehensive_financials(ticker)
    final_json['yf_data'] = yf_data

    # Standardize column names to prevent KeyErrors in charting
    if 'chart_data' in yf_data and isinstance(yf_data['chart_data'], pd.DataFrame):
        yf_data['chart_data'].columns = [col.lower() for col in yf_data['chart_data'].columns]

    return final_json

def _generate_content_with_retry(model, prompt, status_handler: StatusHandler, agent_name: str):
    """Wrapper for model.generate_content with retry logic for transient errors."""
    max_retries = 3
    delay = 5  # Initial delay in seconds
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt)
        except (exceptions.ServiceUnavailable, exceptions.ResourceExhausted) as e:
            error_type = "unavailable" if isinstance(e, exceptions.ServiceUnavailable) else "rate limited"
            if attempt < max_retries - 1:
                status_handler.log(f"⚠️ {agent_name} Agent is {error_type}. Retrying in {delay}s... ({attempt + 1}/{max_retries-1})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                status_handler.error(f"🚨 {agent_name} Agent failed after {max_retries} attempts due to being {error_type}.")
                raise e # Re-raise the exception after the final attempt

def run_rag_agent(rag_model, ticker, status_handler: StatusHandler):
    status_handler.log("📑 RAG Agent analyzing documents...")
    prompt = agent_prompts.get_rag_prompt(ticker)
    response = _generate_content_with_retry(rag_model, prompt, status_handler, "RAG")
    
    # Extract citation metadata for Radical Transparency
    citations = []
    if response and hasattr(response, 'grounding_metadata') and response.grounding_metadata:
        for citation in response.grounding_metadata.grounding_attributions:
             citations.append({'uri': citation.web.uri, 'title': citation.web.title})

    return {"text": response.text, "citations": citations}

def run_market_agent(market_model, ticker, av_data, status_handler: StatusHandler):
    status_handler.log("🌍 Market Agent analyzing Sentiment (Alpha Vantage)...")
    prompt = agent_prompts.get_market_prompt(ticker, av_data)
    response = _generate_content_with_retry(market_model, prompt, status_handler, "Market")
    return {"text": response.text}

def run_competitor_agent(market_model, ticker, av_data, status_handler: StatusHandler):
    status_handler.log("⚔️ Competitor Scout analyzing rivals...")
    prompt = agent_prompts.get_competitor_prompt(ticker, av_data)
    response = _generate_content_with_retry(market_model, prompt, status_handler, "Competitor")
    return {"text": response.text}

def run_macro_agent(market_model, ticker, av_data, status_handler: StatusHandler):
    status_handler.log("🏛️ Macro Strategist: Fetching Rates, CPI, and GDP...")
    # Macro data is now part of the comprehensive av_data fetch
    prompt = agent_prompts.get_macro_prompt(ticker, av_data)
    response = _generate_content_with_retry(market_model, prompt, status_handler, "Macro")
    return {"text": response.text}

def _parse_json_with_fallback(json_string: str, status_handler: StatusHandler, agent_name: str):
    """Attempts to parse a JSON string, with a fallback for double-encoded JSON."""
    try:
        data = json.loads(json_string)
        # Handle cases where the model returns a JSON string inside a string
        if isinstance(data, str):
            status_handler.log(f"⚠️ {agent_name} agent returned a string-in-a-string. Attempting re-parse.")
            return json.loads(data)
        return data
    except json.JSONDecodeError:
        return None

def run_deep_dive_agent(synthesis_model, rag_model, ticker, report, status_handler: StatusHandler):
    status_handler.log("🕵️ Deep Dive Agent: Finding contradictions...")

    # Extract text safely
    quant_context = report.get('part_1_5_quant', {})
    
    prompt = agent_prompts.get_deep_dive_prompt(
        ticker,
        quant_context,
        report['part_1_4_6_rag']['text'],
        report['part_2_3_market']['text'],
        report.get('part_6_competitor', {'text': 'No competitor data.'})['text'],
        status_handler=status_handler # Corrected: Pass the status_handler object
    )
    response = _generate_content_with_retry(synthesis_model, prompt, status_handler, "Deep Dive (Question Generation)")
    questions = response.text.strip().split('\n')
    answers = []
    for q in questions:
        if q.strip():
            status_handler.log(f"   ❓ Investigating: {q.strip()}")
            ans_resp = _generate_content_with_retry(rag_model, f"Answer this using 10-K: {q}", status_handler, "Deep Dive (Answer Generation)")
            time.sleep(2) # Add a delay to avoid hitting API rate limits
            answers.append(f"Q: {q}\nA: {ans_resp.text}")
    return "\n\n".join(answers)

def run_synthesis_pipeline(synthesis_model, ticker, report, status_handler: StatusHandler):
    status_handler.update_step("Step 8: Drafting final report...")
    status_handler.log("   -> Step 8.1: Generating prompt for Synthesis Agent...")
    with st.expander("Show Synthesis Prompt"):
        st.text(agent_prompts.SYNTHESIS_PROMPT_TEMPLATE)

    # Combine market, competitor, and macro reports into a single context string for the prompt.
    market_context = (
        f"Market News & Sentiment:\n{report['part_2_3_market']['text']}\n\n"
        f"Competitor Analysis:\n{report.get('part_6_competitor', {'text': 'No competitor data.'})['text']}\n\n"
        f"Macroeconomic Outlook:\n{report.get('part_7_macro', {'text': 'No macro data.'})['text']}"
    )

    draft_prompt = agent_prompts.get_synthesis_prompt(
        ticker,
        report['part_1_5_quant'],
        report['part_1_4_6_rag']['text'],
        market_context,
        report['deep_dive_analysis'],
        status_handler=status_handler
    )
    draft_response = _generate_content_with_retry(synthesis_model, draft_prompt, status_handler, "Synthesis")
    draft_text = clean_json_output(draft_response.text)

    status_handler.update_step("Step 9: Critic reviewing draft...")
    status_handler.log("   -> Step 9.1: Generating prompt for Critic Agent...")
    with st.expander("Show Critic Prompt"):
        st.text(agent_prompts.CRITIC_PROMPT_TEMPLATE)

    critic_prompt = agent_prompts.get_critic_prompt(ticker, draft_text, report['part_1_5_quant'])
    final_response = _generate_content_with_retry(synthesis_model, critic_prompt, status_handler, "Critic")
    final_text = clean_json_output(final_response.text)
    
    # Attempt to parse the critic's response first
    final_data = _parse_json_with_fallback(final_text, status_handler, "Critic")
    if final_data:
        return final_data
    else:
        status_handler.log("⚠️ Critic agent returned invalid JSON. Falling back to draft.")
        draft_data = _parse_json_with_fallback(draft_text, status_handler, "Synthesis")
        if draft_data: return draft_data
    
    status_handler.error("Fatal: Both Synthesis and Critic agents failed to produce valid JSON.")
    return {"error": "Failed to parse final report from both agents."}

def get_nvda_dummy_data():
    """Returns a complete dummy report structure for NVDA for testing."""
    return {
        'ingestion_agent': True,
        'part_1_5_quant': {
            "ticker": "NVDA", "cik": "0001045810", "company_name": "NVIDIA CORP (DUMMY)",
            "part_1_financials": {
                "latest_revenue": {"value": 92276000000, "period": "FY", "filed": "2024-02-21"},
                "latest_net_income": {"value": 49994000000, "period": "FY", "filed": "2024-02-21"}
            },
            "part_5_valuation": {
                "market_price": 950.02, "market_cap": 2375000000000, "pe_ratio": 75.9, "ps_ratio": 35.1,
                "latest_eps_diluted": {"value": 11.93, "period": "FY", "filed": "2024-02-21"}
            },
            'yf_data': {
                'valuation': {'Current Price': 950.02, 'Market Cap': 2375000000000},
                'financials': {'P/E Ratio': 75.9, 'P/S Ratio': 35.1},
                'chart_data': pd.DataFrame({
                    'date': pd.to_datetime(['2023-01-01', '2024-01-01', '2024-05-01']),
                    'open': [390, 890, 940],
                    'high': [410, 910, 960],
                    'low': [385, 885, 935],
                    'close': [400, 900, 950.02],
                    'volume': [1000000, 2000000, 1500000]
                }).set_index('date')
            }
        },
        'part_1_4_6_rag': {"text": "Dummy RAG analysis: NVIDIA has a strong moat in AI chips due to its CUDA platform (Source: 2024 10-K)."},
        'part_2_3_market': {"text": "Dummy Market analysis: Sentiment is overwhelmingly positive, driven by AI demand."},
        'part_6_competitor': {"text": "Dummy Competitor analysis: AMD and Intel are key rivals, but NVIDIA leads in the data center space."},
        'part_7_macro': {"text": "Dummy Macro analysis: The current economic climate is favorable for tech spending. [1]"},
        'deep_dive_analysis': "Dummy Deep Dive: Q: Is the high P/E ratio justified? A: The 10-K suggests future growth in AI and automotive sectors supports the valuation.",
        'final_synthesis': {
            'scoring_matrix': {
                'pillar_1_corporate': 9, 'pillar_2_industry': 8, 'pillar_3_valuation': 6,
                'pillar_4_sentiment': 9, 'pillar_5_governance': 7
            },
            'recommendation': {'action': 'Strong Buy', 'justification': 'Dominant market position and explosive growth in AI justify the premium valuation.'},
            'final_summary': 'This is a dummy summary. NVIDIA shows exceptional performance driven by its leadership in the AI sector. While valuation is high, its technological moat and growth prospects remain unparalleled.',
            'key_risks': 'Dummy Key Risks: High dependency on the AI market, geopolitical risks related to chip manufacturing.',
            'final_weighted_score': 8.15
        },
        'report_saved': True,
        'notified_start': True
    }


# --- Main App ---
def main():
    setup_logging()
    if 'score_weights' not in st.session_state:
        st.session_state.score_weights = {'pillar_1_corporate': 0.35, 'pillar_2_industry': 0.20, 'pillar_3_valuation': 0.20, 'pillar_4_sentiment': 0.15, 'pillar_5_governance': 0.10}

    st.title("PyFinAgent: AI Financial Analyst (Agentic)")
    st.caption("A 'Glass Box' Financial Copilot")

    if 'gcp_services' not in st.session_state:
        st.session_state.gcp_services = initialize_gcp_services()
    services = st.session_state.gcp_services
    display_sidebar(services['bq_client'], services['table_id'], st.session_state.get("ticker"))

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1: st.text_input("Enter Ticker:", key="ticker", label_visibility="collapsed", placeholder="e.g. NVDA")
    with col3: 
        if st.button("Clear", use_container_width=True):
            st.session_state.clear()
            if 'status_handler_initialized' in st.session_state:
                del st.session_state.status_handler_initialized
            st.rerun()

    with col2:
        if st.button("Run Comprehensive Analysis", type="primary", use_container_width=True):
            # Clear previous report data to ensure a fresh start
            if 'report' in st.session_state:
                del st.session_state.report
            # Set flags to start the new analysis
            st.session_state.analysis_in_progress = True
            st.session_state.test_mode = False

        if st.button("Test with NVDA", use_container_width=True):
            st.session_state.clear()
            st.session_state.ticker = "NVDA"
            st.session_state.report = get_nvda_dummy_data()
            st.session_state.analysis_in_progress = False
            st.session_state.test_mode = True
            st.rerun()

    if st.session_state.get('analysis_in_progress') and st.session_state.ticker:
        ticker = st.session_state.ticker
        if 'report' not in st.session_state: st.session_state.report = {}
        
        # Initialize StatusHandler
        progress_bar_placeholder = st.empty()
        status_handler = StatusHandler(total_steps=10, progress_bar=progress_bar_placeholder)

        try:
            if 'notified_start' not in st.session_state:
                status_handler.update_step("Step 1: Starting Analysis...")
                tools_slack.send_notification(
                    f"🚀 Analysis Started: {ticker}", 
                    {"User": "Lead Analyst", "Strategy": "Hybrid AI"}, 
                    "info"
                )
                st.session_state.notified_start = True

            if 'av_data' not in st.session_state:
                status_handler.log("📡 Fetching Alpha Vantage Market Intel...")
                st.session_state.av_data = tools_alphavantage.get_market_intel(ticker)

            if 'ingestion_agent' not in st.session_state.report:
                run_ingestion_agent(ticker, status_handler)
                st.session_state.report['ingestion_agent'] = True

            if 'part_1_5_quant' not in st.session_state.report:
                status_handler.update_step("Step 2: Gathering Deep Financials (yfinance)...")
                st.session_state.report['part_1_5_quant'] = run_quant_agent(ticker, status_handler)
            
            if 'part_1_4_6_rag' not in st.session_state.report:
                status_handler.update_step("Step 3: Analyzing 10-Ks with RAG Agent...")
                st.session_state.report['part_1_4_6_rag'] = run_rag_agent(services['rag_model'], ticker, status_handler)

            if 'part_2_3_market' not in st.session_state.report:
                status_handler.update_step("Step 4: Scanning Market News...")
                st.session_state.report['part_2_3_market'] = run_market_agent(services['market_model'], ticker, st.session_state.av_data, status_handler)

            if 'part_6_competitor' not in st.session_state.report:
                status_handler.update_step("Step 5: Analyzing Rivals...")
                st.session_state.report['part_6_competitor'] = run_competitor_agent(services['market_model'], ticker, st.session_state.av_data, status_handler)

            if 'part_7_macro' not in st.session_state.report:
                status_handler.update_step("Step 6: Analyzing Macro Economy...")
                st.session_state.report['part_7_macro'] = run_macro_agent(services['market_model'], ticker, st.session_state.av_data, status_handler)

            if 'deep_dive_analysis' not in st.session_state.report:
                status_handler.update_step("Step 7: Deep Dive Analysis...")
                st.session_state.report['deep_dive_analysis'] = run_deep_dive_agent(services['synthesis_model'], services['rag_model'], ticker, st.session_state.report, status_handler)

            if 'final_synthesis' not in st.session_state.report:
                final_json = run_synthesis_pipeline(services['synthesis_model'], ticker, st.session_state.report, status_handler)
                scores = final_json.get('scoring_matrix', {})
                w = st.session_state.score_weights
                final_score = sum(scores.get(k, 0) * v for k, v in w.items())
                final_json['final_weighted_score'] = round(final_score, 2)
                st.session_state.report['final_synthesis'] = final_json

            if 'report_saved' not in st.session_state.report:
                status_handler.update_step("Step 10: Saving report to BigQuery...")
                try:
                    row = {
                        "ticker": ticker, "company_name": st.session_state.report.get('part_1_5_quant', {}).get('company_name', 'N/A'),
                        "analysis_date": datetime.now().isoformat(), "final_score": st.session_state.report['final_synthesis']['final_weighted_score'],
                        "recommendation": st.session_state.report['final_synthesis'].get('recommendation', {}).get('action', 'N/A'),
                        "summary": st.session_state.report['final_synthesis'].get('final_summary', ''), "full_report_json": json.dumps(st.session_state.report)
                    }
                    services['bq_client'].insert_rows_json(services['table_id'], [row])
                    tools_slack.send_notification(f"✅ Complete: {ticker}", {"Score": f"{row['final_score']}/10", "Verdict": row['recommendation']}, "success")
                except Exception as e: logging.error(f"Save failed: {e}")

                st.session_state.report['report_saved'] = True
                status_handler.complete("Analysis Complete!")
                st.session_state.analysis_in_progress = False
                st.rerun()

        except Exception as e:
            # Log the full error with stack trace to the console
            logging.error("An unhandled exception occurred during analysis.", exc_info=True)
            # Display a user-friendly error in the UI
            status_handler.error(f"Analysis halted: {str(e)}")
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())
            st.session_state.analysis_in_progress = False
            tools_slack.send_notification(f"❌ Analysis Failed: {ticker}", {"Error": str(e)[:200]}, "error")

    if st.session_state.get('report', {}).get('final_synthesis'): display_report()

if __name__ == "__main__":
    main()