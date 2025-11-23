import streamlit as st
import pandas as pd
import logging
from fpdf import FPDF
from components.version import display_version

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
            company_name,
            final_score,
            recommendation,
            summary
        FROM `{table_id}`
        WHERE ticker = @ticker
        ORDER BY analysis_date DESC
        LIMIT 10
    """
    job_config = st.session_state.bigquery.QueryJobConfig(
        query_parameters=[st.session_state.bigquery.ScalarQueryParameter("ticker", "STRING", ticker.upper())]
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

    return pdf.output(dest='S').encode('latin-1')

def _handle_report_load(ticker, analysis_date):
    """
    Sets session state to trigger loading a past report on the main page.
    """
    st.session_state.report_to_load = {
        "Ticker": ticker,
        "Analysis Date": analysis_date
    }
    # We need to rerun the app for the main page to process this state change
    st.rerun()

def display_sidebar(bq_client, table_id, ticker):
    """Renders the sidebar with all its components."""
    with st.sidebar:
        display_version()
        st.divider()

        st.info("Running in a self-hosted environment.")

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
                    # Create a header for our clickable list
                    cols = st.columns([2, 2, 1])
                    cols[0].markdown("**Date**")
                    cols[1].markdown("**Company**")
                    cols[2].markdown("**Score**")

                    # Iterate over rows to create a clickable entry for each report
                    for index, row in historical_df.iterrows():
                        # Format date for display
                        date_str = pd.to_datetime(row["analysis_date"]).strftime('%Y-%m-%d %H:%M')
                        cols = st.columns([2, 2, 1, 1])
                        cols[0].markdown(f"`{date_str}`")
                        cols[1].write(row.get("company_name", "N/A"))
                        cols[2].write(f"{row['final_score']:.2f}")
                        # The button that triggers the report load
                        cols[3].button("Load", key=f"load_{index}", on_click=_handle_report_load, args=(ticker, date_str), use_container_width=True)
                else:
                    st.info(f"No historical reports found for {ticker.upper()}.")
            except Exception as e:
                logging.error(f"Failed to display historical data: {e}", exc_info=True)
                st.warning("Could not load historical reports.")