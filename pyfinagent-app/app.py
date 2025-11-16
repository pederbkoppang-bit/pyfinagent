import logging
import streamlit as st
import requests
import json
from google.cloud import bigquery
from vertexai.generative_models import GenerativeModel, Tool, grounding
import traceback
from datetime import datetime
import vertexai
import pandas as pd
from fpdf import FPDF

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. CONFIGURATION (from secrets.toml) ---
PROJECT_ID = st.secrets.gcp.project_id
LOCATION = st.secrets.gcp.location
QUANT_AGENT_URL = st.secrets.agent.quant_agent_url
RAG_DATA_STORE_ID = st.secrets.agent.rag_data_store_id
GEMINI_MODEL = st.secrets.agent.gemini_model

@st.cache_resource
def initialize_gcp_services():
    """
    Initializes all GCP services and AI models.
    Using @st.cache_resource ensures this expensive operation runs only once.
    """
    logging.info(f"Initializing models with GEMINI_MODEL: '{GEMINI_MODEL}'")
    logging.info("Initializing GCP services...")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    bq_client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.financial_reports.analysis_results"

    # --- AGENT DEFINITIONS (from PyFinAgent.md canvas) ---
    datastore_path = (f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/"
                      f"dataStores/{RAG_DATA_STORE_ID}")
    rag_tool = Tool.from_retrieval(
        grounding.Retrieval(grounding.VertexAISearch(datastore=datastore_path))
    )
    market_tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
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

def run_quant_agent(ticker: str) -> dict:
    """Executes the QuantAgent and returns the report."""
    logging.info(f"Starting QuantAgent for ticker: {ticker}")
    st.session_state.status_text.text("Task 1/4: QuantAgent fetching hard financials (SEC + yfinance)...")
    
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
    st.session_state.status_text.text("Task 2/4: RAG_Agent analyzing 10-K filings...")
    rag_prompt = f"Using ONLY the provided 10-K documents, analyze the Economic Moat and Governance (exec compensation), and key 'Risk Factors' for {ticker}. [cite: Comprehensive Financial Analysis Template.pdf.pdf]"
    rag_response = rag_model.generate_content(rag_prompt)
    logging.info("RAG_Agent finished successfully.")
    return {"text": rag_response.text}

def run_market_agent(market_model, ticker: str) -> dict:
    """Executes the MarketAgent for news and sentiment analysis."""
    logging.info("Starting MarketAgent for news and sentiment analysis.")
    st.session_state.status_text.text("Task 3/4: MarketAgent scanning news/sentiment...")
    market_prompt = f"Analyze the Macro (PESTEL) and current Market Sentiment (news, social media 'scuttlebutt') for {ticker}. [cite: Comprehensive Financial Analysis Template.pdf.pdf]"
    market_response = market_model.generate_content(market_prompt)
    logging.info("MarketAgent finished successfully.")
    return {"text": market_response.text}

def run_synthesis_agent(synthesis_model, ticker: str) -> dict:
    """Executes the AnalystAgent to synthesize all findings."""
    logging.info("Starting AnalystAgent for final synthesis.")
    st.session_state.status_text.text("Task 4/4: LeadAnalyst synthesizing final report...")
    
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

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_historical_data(_bq_client, table_id: str, ticker: str):
    """
    Queries BigQuery for historical analysis of a given ticker.
    The bq_client argument is prefixed with an underscore to tell Streamlit's
    caching mechanism not to hash it.
    """
    if not ticker:
        return None

    logging.info(f"Fetching historical data for {ticker} from BigQuery.")
    query = f"""
        SELECT
            analysis_date,
            final_score,
            recommendation,
            summary
        FROM `{table_id}`
        WHERE ticker = @ticker
        ORDER BY analysis_date DESC
        LIMIT 10
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("ticker", "STRING", ticker.upper())]
    )
    query_job = _bq_client.query(query, job_config=job_config)
    return query_job.to_dataframe()

def generate_pdf_report(report: dict) -> bytes:
    """Generates a PDF report from the analysis data."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)

    ticker = report.get('part_1_5_quant', {}).get('ticker', 'N/A')
    pdf.cell(0, 10, f"PyFinAgent Analysis Report: {ticker}", 0, 1, "C")
    pdf.ln(10)

    synthesis = report.get('final_synthesis', {})
    if synthesis:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Executive Summary", 0, 1)

        pdf.set_font("Helvetica", "", 10)
        score = synthesis.get('final_weighted_score', 'N/A')
        recommendation = synthesis.get('recommendation', {}).get('action', 'N/A')
        justification = synthesis.get('recommendation', {}).get('justification', 'N/A')
        summary = synthesis.get('final_summary', 'N/A')

        pdf.multi_cell(0, 5, f"Final Score: {score} / 10")
        pdf.multi_cell(0, 5, f"Recommendation: {recommendation}")
        pdf.multi_cell(0, 5, f"Justification: {justification}")
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "Summary:", 0, 1)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, summary)
        pdf.ln(10)

    def write_section(title, content_key):
        content = report.get(content_key, {}).get('text', 'Not available.')
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, title, 0, 1)
        pdf.set_font("Helvetica", "", 10)
        # Use write() for better handling of unicode characters
        pdf.write(5, content)
        pdf.ln(10)

    write_section("10-K Analysis (Economic Moat, Governance, Risks)", 'part_1_4_6_rag')
    write_section("Market Analysis (Macro & Sentiment)", 'part_2_3_market')

    # The FPDF library expects latin-1 encoding when outputting to a string.
    return pdf.output(dest='S').encode('latin-1')

def display_price_chart():
    """Renders a line chart of historical stock prices if available."""
    if 'report' not in st.session_state or not st.session_state.report.get('part_1_5_quant'):
        return

    historical_prices = st.session_state.report['part_1_5_quant'].get('historical_prices')
    if historical_prices:
        st.subheader("Historical Price Chart")
        price_df = pd.read_json(historical_prices, orient='split')
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

    # --- Final Summary & Full Report ---
    display_price_chart()
    st.divider()

    st.subheader("Final Summary")
    st.write(report_data['final_summary'])
    
    with st.expander("View Full Raw Data (JSON)"):
        st.json(st.session_state.report)

# The email of the authorized user
AUTHORIZED_EMAIL = "peder.bkoppang@hotmail.no"

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="PyFinAgent", page_icon="📈", layout="wide")

    # --- Authentication Check ---
    if not st.user.is_logged_in:
        st.title("Welcome to PyFinAgent")
        st.write("Please log in to continue.")
        if st.button("Log in with Google"):
            st.login()
        return

    if st.user.email != AUTHORIZED_EMAIL:
        st.error("Access denied. You are not an authorized user.")
        st.write(f"You are logged in as: {st.user.email}")
        if st.button("Log out"):
            st.logout()
        return

    # --- Authorized Application Code ---
    # If we get here, the user is logged in and authorized.

    # --- STREAMLIT UI ---
    st.title("PyFinAgent Dashboard: AI Financial Analyst")
    st.caption(f"A Multi-Agent AI built on the Comprehensive Financial Analysis Template")

    ticker = st.text_input("Enter Company Ticker (e.g., NVDA, AAPL):")

    # Load services only when the button is pressed or on first run
    try:
        services = initialize_gcp_services()
        bq_client = services["bq_client"]
        table_id = services["table_id"]
        rag_model = services["rag_model"]
        market_model = services["market_model"]
        synthesis_model = services["synthesis_model"]
    except Exception as e:
        logging.error("Failed to initialize GCP services.", exc_info=True)
        st.error("Failed to initialize critical GCP services. The application cannot continue.")
        st.exception(e)
        return # Stop execution if services fail

    with st.sidebar:
        st.write(f"Welcome, {st.user.name}!")
        if st.button("Log out"):
            st.logout()
        st.divider()

        st.header("⚙️ Actions")
        if st.button("Clear All Caches"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("All caches have been cleared!")

        if st.session_state.get('report', {}).get('final_synthesis'):
            st.download_button(
                label="Download Report as PDF",
                data=generate_pdf_report(st.session_state.report),
                file_name=f"PyFinAgent_Report_{st.session_state.report.get('part_1_5_quant', {}).get('ticker', 'NA')}.pdf",
                mime="application/pdf",
            )

        st.header("📄 Recent Reports")
        if ticker:
            try:
                historical_df = get_historical_data(bq_client, table_id, ticker)
                if not historical_df.empty:
                    st.caption(f"Past Analysis for {ticker.upper()}")
                    # Format date for better readability
                    historical_df["analysis_date"] = pd.to_datetime(historical_df["analysis_date"]).dt.strftime('%Y-%m-%d %H:%M')
                    
                    # Rename columns for presentation
                    display_df = historical_df.rename(columns={
                        "analysis_date": "Date",
                        "final_score": "Score",
                        "recommendation": "Recommendation",
                        "summary": "Summary"
                    })
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No historical reports found for {ticker.upper()}.")
            except Exception as e:
                logging.error(f"Failed to display historical data: {e}", exc_info=True)
                st.warning("Could not load historical reports.")

    st.divider()

    if st.button("Run Comprehensive Analysis", disabled=(not ticker)):
        st.session_state.report = {}
        st.session_state.status_text = st.empty()
        
        with st.spinner("Running Analysis Pipeline..."):
            try:
                # --- 2. AGENT EXECUTION ---
                st.session_state.report['part_1_5_quant'] = run_quant_agent(ticker)
                st.session_state.report['part_1_4_6_rag'] = run_rag_agent(rag_model, ticker)
                st.session_state.report['part_2_3_market'] = run_market_agent(market_model, ticker)
                
                # --- Agent 4: AnalystAgent (Synthesis & Scoring) ---
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
                logging.info(f"Final weighted score calculated: {final_report['final_weighted_score']}")
                
                # --- 6. WRITE TO BIGQUERY ---
                st.session_state.status_text.text("Saving report to BigQuery...")
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
                
            except Exception as e:
                logging.error(f"An error occurred during analysis for ticker '{ticker}': {e}", exc_info=True)
                st.session_state.status_text.empty()
                st.error(f"An error occurred during analysis: {e}")
                st.write(traceback.format_exc())

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