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
        # --- Sorting Controls ---
        st.sidebar.subheader("Table Sorting")
        column_mapping = {
            "Analysis Date": "analysis_date",
            "Ticker": "ticker",
            "Company Name": "company_name",
            "Final Score": "final_score",
            "Recommendation": "recommendation"
        }
        
        sort_column_display = st.sidebar.selectbox(
            "Sort by:",
            options=list(column_mapping.keys()),
            index=0, # Default to 'Analysis Date'
            key='sort_column'
        )
        sort_order = st.sidebar.radio(
            "Order:",
            options=["Descending", "Ascending"],
            index=0, # Default to 'Descending'
            key='sort_order',
            horizontal=True
        )

        sort_column_db = column_mapping[sort_column_display]
        sort_order_sql = "DESC" if sort_order == "Descending" else "ASC"

        query = f"""
            SELECT
                ticker,
                company_name,
                analysis_date,
                final_score,
                recommendation
            FROM `{table_id}`
            ORDER BY {sort_column_db} {sort_order_sql}
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
            "company_name": "Company Name",
            "analysis_date": "Analysis Date",
            "final_score": "Final Score",
            "recommendation": "Recommendation"
        })

        # --- Search/Filter Bar ---
        search_query = st.text_input(
            "Search by Ticker or Company Name:",
            placeholder="e.g., AAPL or Apple Inc."
        )

        if search_query:
            search_query_lower = search_query.lower()
            filtered_df = reports_df[
                reports_df['Ticker'].str.lower().str.contains(search_query_lower) |
                reports_df['Company Name'].str.lower().str.contains(search_query_lower)
            ]
        else:
            filtered_df = reports_df

        # Add a 'select' column to the dataframe for the checkbox
        filtered_df.insert(0, "select", False)

        st.info("Click on a row to select a report, then click the button below to view it on the Home page.")
        
        # Use st.data_editor to make rows selectable
        edited_df = st.data_editor(
            filtered_df, # Display the filtered dataframe
            hide_index=True,
            use_container_width=True,
            key="report_editor",
            column_config={
                "select": st.column_config.CheckboxColumn(required=True, help="Select reports to view or delete")
            },
            disabled=[col for col in reports_df.columns if col != 'select'], # Make all data columns read-only
        )

        # --- Action Buttons ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("View Selected Report(s) on Home Page", use_container_width=True):
                # Get the selected rows from the editor's state
                selected_rows = edited_df[edited_df.select]
                if selected_rows.empty:
                    st.warning("Please click on a row in the table to select a report first.")
                else:
                    # Get all selected reports
                    selected_reports = selected_rows.to_dict(orient="records")
                    st.session_state.reports_to_load = selected_reports
                    # Navigate to the main Home page to load the report
                    st.switch_page("Home.py")
        
        with col2:
            if st.button("Delete Selected Report", use_container_width=True, type="primary"):
                st.session_state.show_delete_confirmation = True

        # --- Delete Confirmation Modal ---
        if st.session_state.get("show_delete_confirmation"):
            selected_rows = edited_df[edited_df.select]
            if selected_rows.empty:
                st.warning("Please click on a row in the table to select a report first.")
                st.session_state.show_delete_confirmation = False # Reset state
            else:
                with st.warning(f"Are you sure you want to delete these {len(selected_rows)} report(s)? This action cannot be undone."):
                    c1, c2 = st.columns(2)
                    if c1.button("Yes, Delete", use_container_width=True):
                        report_to_delete = selected_rows.iloc[0] # Example: deleting first selected
                        
                        delete_query = f"""
                            DELETE FROM `{table_id}`
                            WHERE ticker = '{report_to_delete['Ticker']}' 
                            AND analysis_date = TIMESTAMP('{report_to_delete['Analysis Date']}')
                        """
                        bq_client.query(delete_query).result() # Execute and wait for completion
                        st.success(f"Report for {report_to_delete['Ticker']} from {report_to_delete['Analysis Date']} has been deleted.")
                        st.session_state.show_delete_confirmation = False
                        st.rerun() # Refresh the page to show the updated table
                    if c2.button("No, Cancel", use_container_width=True):
                        st.session_state.show_delete_confirmation = False
                        st.rerun()

    except NotFound:
        st.error(f"The table `{table_id}` was not found. Please ensure it exists and you have permissions.")
    except Exception as e:
        st.error(f"An error occurred while fetching reports: {e}")
        logging.error("Failed to load past reports", exc_info=True)

if __name__ == "__main__":
    load_and_display_reports()