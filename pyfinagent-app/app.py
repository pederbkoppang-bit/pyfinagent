import streamlit as st
import requests
import json
from datetime import datetime
from google.cloud import bigquery
import vertexai

# [--- THE FIX IS HERE ---]
# We now import SearchTool and GoogleSearchRetrieval from the correct modules
from vertexai.preview.generative_models import GenerativeModel, Tool, Part, SearchTool 
from vertexai.preview.language_models import GoogleSearchRetrieval
# [--- END OF FIX ---]


# --- 1. CONFIGURATION ---
PROJECT_ID = "sunny-might-477607-p8" 
LOCATION = "us-central1"

# [--- UPDATE 1 ---]
# This is the URL you got from deploying your QuantAgent in Phase 3
# (Fixed to be a valid Python string)
QUANT_AGENT_URL = "[https://quant-agent-afytokcdfq-uc.a.run.app/](https://quant-agent-afytokcdfq-uc.a.run.app/)" 

# [--- UPDATE 2 ---]
# This is the RAG Data Store ID you just provided
RAG_DATA_STORE_ID = "10-k-data_1762684273198_gcs_store" 
# --- (END OF CONFIGURATION) ---

# --- GCP Service Initialization ---
try:
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    bq_client = bigquery.Client(project=PROJECT_ID)
    TABLE_ID = f"{PROJECT_ID}.financial_reports.analysis_results"

    # --- AGENT DEFINITIONS (from PyFinAgent.md canvas) ---
    
    # 1. RAG_Agent (Tool)
    # [cite: PyFinAgent.md - This is the 'AI Memory' from Phase 2]
    # [--- THE FIX IS HERE ---]
    # We call SearchTool(...) directly instead of language_models.SearchTool(...)
    rag_tool = Tool.from_retrieval(
        SearchTool(data_store_id=RAG_DATA_STORE_ID)
    )
    # [--- END OF FIX ---]
    
    # 2. MarketAgent (Tool)
    # [cite: PyFinAgent.md - This is the 'Live Web Search' agent]
    market_tool = Tool.from_google_search_retrieval(
        GoogleSearchRetrieval()
    )

    # 3. AnalystAgent (Models)
    # This is the "Lead Analyst" that synthesizes everything
    synthesis_model = GenerativeModel("gemini-1.5-pro")
    
    # These models will *only* use their assigned tools
    rag_model = GenerativeModel("gemini-1.5-pro", tools=[rag_tool])
    market_model = GenerativeModel("gemini-1.5-pro", tools=[market_tool])

except Exception as e:
    st.error(f"Failed to initialize GCP services. Check configuration. Error: {e}")
    st.stop()

# --- STREAMLIT UI ---
st.set_page_config(page_title="PyFinAgent", layout="wide")
st.title("PyFinAgent: AI Financial Analyst")
st.caption(f"A Multi-Agent AI built on the Comprehensive Financial Analysis Template")

ticker = st.text_input("Enter Company Ticker (e.g., NVDA, AAPL):")

if st.button("Run Comprehensive Analysis", disabled=(not ticker)):
    st.session_state.report = {}
    
    with st.spinner("Running Analysis Pipeline..."):
        try:
            # --- 2. AGENT EXECUTION ---
            status_text = st.empty()

            # --- Agent 1: QuantAgent (Python-First) ---
            # [cite: PyFinAgent.md - Part 1.1, Part 5]
            status_text.text("Task 1/4: QuantAgent fetching hard financials (SEC + yfinance)...")
            # This is the HTTP call to the Cloud Function you just deployed
            quant_response = requests.get(f"{QUANT_AGENT_URL}?ticker={ticker}")
            quant_response.raise_for_status() # Will raise an error if the function failed
            quant_report = quant_response.json()
            
            if quant_report.get("error"):
                raise Exception(f"QuantAgent Error: {quant_report['error']}")
            st.session_state.report['part_1_5_quant'] = quant_report

            # --- Agent 2: RAG_Agent (10-K Analysis) ---
            # [cite: PyFinAgent.md - Part 1.2, Part 4, Part 6]
            status_text.text("Task 2/4: RAG_Agent analyzing 10-K filings...")
            rag_prompt = f"Using ONLY the provided 10-K documents, analyze the Economic Moat and Governance (exec compensation), and key 'Risk Factors' for {ticker}. [cite: Comprehensive Financial Analysis Template.pdf.pdf]"
            rag_response = rag_model.generate_content(rag_prompt)
            st.session_state.report['part_1_4_6_rag'] = {"text": rag_response.text}

            # --- Agent 3: MarketAgent (Live Web Search) ---
            # [cite: PyFinAgent.md - Part 2, Part 3]
            status_text.text("Task 3/4: MarketAgent scanning news/sentiment...")
            market_prompt = f"Analyze the Macro (PESTEL) and current Market Sentiment (news, social media 'scuttlebutt') for {ticker}. [cite: Comprehensive Financial Analysis Template.pdf.pdf]"
            market_response = market_model.generate_content(market_prompt)
            st.session_state.report['part_2_3_market'] = {"text": market_response.text}
            
            # --- Agent 4: AnalystAgent (Synthesis & Scoring) ---
            # [cite: PyFinAgent.md - Part 7, Part 8]
            status_text.text("Task 4/4: LeadAnalyst synthesizing final report...")
            synthesis_prompt = f"""
            You are the Lead Analyst. Your team has submitted their findings for {ticker}.
            Your job is to synthesize all reports into a final recommendation and score,
            as specified in the Comprehensive Financial Analysis Template.

            [QUANT REPORT (Facts & Ratios)]: {json.dumps(st.session_state.report['part_1_5_quant'])}
            [RAG REPORT (10-K Analysis)]: {st.session_state.report['part_1_4_6_rag']['text']}
            [MARKET REPORT (News & Sentiment)]: {st.session_state.report['part_2_3_market']['text']}

            Perform the following actions:
            1.  **Part 7 (Recommendation):** Provide a BUY/HOLD/SELL recommendation, time horizon, and justification.
            2.  **Part 8 (Scoring):** Provide a raw score (1-10, float) for each of the 5 pillars from the template.
            3.  **Part 8 (Summary):** Write the 2-3 sentence final summary.
            
            Return ONLY a JSON object:
            {{
                "recommendation": {{"action": "BUY", "justification": "..."}},
                "scoring_matrix": {{
                    "pillar_1_corporate": 8.0,
                    "pillar_2_industry": 7.0,
                    "pillar_3_valuation": 6.0,
                    "pillar_4_sentiment": 7.5,
                    "pillar_5_governance": 8.0
                }},
                "final_summary": "..."
            }}
            """
            
            synthesis_response = synthesis_model.generate_content(synthesis_prompt)
            final_report = json.loads(synthesis_response.text)
            
            # --- 5. FINAL CALCULATION (PYTHON) ---
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
            
            # --- 6. WRITE TO BIGQUERY ---
            status_text.text("Saving report to BigQuery...")
            row_to_insert = {
                "ticker": ticker,
                "analysis_date": datetime.now().isoformat(),
                "final_score": final_report['final_weighted_score'],
                "recommendation": final_report['recommendation']['action'],
                "summary": final_report['final_summary'],
                "full_report_json": json.dumps(st.session_state.report)
            }
            errors = bq_client.insert_rows_json(TABLE_ID, [row_to_insert])
            if errors:
                st.error(f"Failed to write to BigQuery: {errors}")
            
            status_text.empty()
            
        except requests.exceptions.HTTPError as e:
            st.error(f"HTTP Error: {e}. The QuantAgent failed. Check its logs.")
            st.json(e.response.json())
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")

# --- 7. DISPLAY RESULTS ---
if 'report' in st.session_state and st.session_state.report.get('final_synthesis'):
    st.success("Analysis Complete!")
    report_data = st.session_state.report['final_synthesis']
    
    # Get the price from the Quant report
    price_data = st.session_state.report.get('part_1_5_quant', {}).get('part_5_valuation', {}).get('market_price')
    price_str = f"${price_data:.2f}" if isinstance(price_data, (int, float)) else "N/A"
    
    st.header(f"Final Score: {report_data['final_weighted_score']} / 10")
    st.subheader(f"Recommendation: {report_data['recommendation']['action']} (at {price_str})")
    st.write(report_data['recommendation']['justification'])
    st.subheader("Final Summary")
    st.write(report_data['final_summary'])
    
    with st.expander("View Full Data (JSON)"):
        st.json(st.session_state.report)