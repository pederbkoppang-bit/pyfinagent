import streamlit as st
import pandas as pd
from google.cloud.exceptions import NotFound
import logging

st.set_page_config(page_title="Past Reports", page_icon="📜", layout="wide")

st.title("📜 Past Analysis Reports")

def load_and_display_reports():
    """
    Queries BigQuery for all past reports and displays them in a table.
    Provides a button to navigate to the home page to view the selected report.
    """
    # Ensure services are initialized
    if 'gcp_services' not in st.session_state:
        st.warning("Please visit the 🏠 Home page first to initialize the application.", icon="👈")
        st.stop()

    bq_client = st.session_state.gcp_services.get("bq_client")
    table_id = st.session_state.gcp_services.get("table_id")

    if not bq_client or not table_id:
        st.error("BigQuery client is not available. Please check initialization on the Home page.")
        st.stop()

    try:
        query = f"""
            SELECT
                ticker,
                analysis_date,
                final_score,
                recommendation
            FROM `{table_id}`
            ORDER BY analysis_date DESC
        """
        query_job = bq_client.query(query)
        reports_df = query_job.to_dataframe()

        if reports_df.empty:
            st.info("No past reports found in the database.")
            return

        # Format the dataframe for better display
        reports_df['analysis_date'] = pd.to_datetime(reports_df['analysis_date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        reports_df = reports_df.rename(columns={
            "ticker": "Ticker",
            "analysis_date": "Analysis Date",
            "final_score": "Final Score",
            "recommendation": "Recommendation"
        })

        st.info("Click on a row to select a report, then click the button below to view it on the Home page.")
        
        # Use st.data_editor to make rows selectable
        selected_row = st.data_editor(
            reports_df,
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            on_change=None, # We will handle selection via a button
            key="report_editor"
        )

        if st.button("View Selected Report on Home Page", use_container_width=True):
            # Get the index of the selected row from the editor's state
            selected_indices = st.session_state.report_editor.get("edited_rows", {})
            if not selected_indices:
                st.warning("Please click on a row in the table to select a report first.")
            else:
                # We only care about the first selection
                selected_index = list(selected_indices.keys())[0]
                st.session_state.report_to_load = reports_df.iloc[selected_index].to_dict()
                st.switch_page("pages/1_🏠_Home.py")

    except NotFound:
        st.error(f"The table `{table_id}` was not found. Please ensure it exists and you have permissions.")
    except Exception as e:
        st.error(f"An error occurred while fetching reports: {e}")
        logging.error("Failed to load past reports", exc_info=True)

if __name__ == "__main__":
    load_and_display_reports()