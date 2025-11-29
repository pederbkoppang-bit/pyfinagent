import streamlit as st
import logging
from fpdf import FPDF
from components.version import display_version
from components.recent_reports import display_recent_reports

def sanitize_for_fpdf(text: str) -> str:
    """
    Removes or replaces characters that FPDF's core fonts (latin-1) cannot handle.
    This prevents FPDFException: 'Not enough horizontal space to render a single character'.
    """
    if not isinstance(text, str):
        text = str(text)
    return text.encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_report(report: dict) -> bytes:
    """Generates a PDF report from the analysis data."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)

    ticker = sanitize_for_fpdf(report.get('part_1_5_quant', {}).get('ticker', 'N/A'))
    pdf.cell(0, 10, f"PyFinAgent Analysis Report: {ticker}", 0, 1, "C")
    pdf.ln(10)

    synthesis = report.get('final_synthesis', {})
    # Defensive programming: If synthesis is a string, try to parse it.
    if isinstance(synthesis, str):
        try:
            synthesis = json.loads(synthesis)
        except json.JSONDecodeError:
            synthesis = {} # If parsing fails, default to an empty dict.
    if synthesis:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Executive Summary", 0, 1)

        pdf.set_font("Helvetica", "", 10)
        score = synthesis.get('final_weighted_score', 'N/A')
        recommendation_obj = synthesis.get('recommendation', {})
        # Defensive check: ensure recommendation_obj is a dict before using .get()
        recommendation = sanitize_for_fpdf(recommendation_obj.get('action', 'N/A') if isinstance(recommendation_obj, dict) else 'N/A')
        justification = sanitize_for_fpdf(recommendation_obj.get('justification', 'N/A') if isinstance(recommendation_obj, dict) else 'N/A')
        summary = sanitize_for_fpdf(synthesis.get('final_summary', 'N/A'))

        pdf.multi_cell(0, 5, f"Final Score: {score} / 10", ln=1)
        pdf.multi_cell(0, 5, f"Recommendation: {recommendation}", border=0, ln=1)
        pdf.multi_cell(0, 5, f"Justification: {justification}", border=0, ln=1)
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "Summary:", 0, 1)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, summary)
        pdf.ln(10)

    return bytes(pdf.output())

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