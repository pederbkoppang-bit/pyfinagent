import streamlit as st
import logging
from fpdf import FPDF
from components.version import display_version
from components.recent_reports import display_recent_reports

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
        pdf.multi_cell(0, 5, f"Recommendation: {recommendation}", border=0, ln=1)
        pdf.multi_cell(0, 5, f"Justification: {justification}", border=0, ln=1)
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "Summary:", 0, 1)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, summary, 0, 1)
        pdf.ln(10)

    return pdf.output(dest='S').encode('latin-1')

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

        # Display the recent reports component
        display_recent_reports(bq_client, table_id, ticker)