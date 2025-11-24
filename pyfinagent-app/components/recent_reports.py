import streamlit as st
import pandas as pd
import logging

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

def _handle_report_load(ticker, analysis_date):
    """
    Sets session state to trigger loading a past report on the main page.
    """
    st.session_state.report_to_load = {
        "Ticker": ticker,
        "Analysis Date": analysis_date
    }
    # Switch to the home page to load the report.
    st.switch_page("Home.py")

def display_recent_reports(bq_client, table_id, ticker):
    """Renders the recent reports list for a given ticker."""
    st.header("📄 Recent Reports")
    if ticker:
        try:
            historical_df = get_historical_data(bq_client, table_id, ticker)
            if not historical_df.empty:
                # Use an expander for each report for a cleaner, more compact UI
                for index, row in historical_df.iterrows():
                    date_obj = pd.to_datetime(row["analysis_date"])
                    datetime_str = date_obj.strftime('%Y-%m-%d %H:%M')
                    
                    with st.expander(f"**{datetime_str}** - Score: **{row['final_score']:.1f}**"):
                        st.caption(f"Company: {row.get('company_name', 'N/A')}")
                        st.caption(f"Recommendation: {row.get('recommendation', 'N/A')}")
                        
                        # The button that triggers the report load
                        if st.button("Load Report", key=f"load_{index}", use_container_width=True):
                            # Pass the full datetime string for precise querying
                            _handle_report_load(ticker, date_obj.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                st.info(f"No historical reports found for {ticker.upper()}.")
        except Exception as e:
            logging.error(f"Failed to display historical data: {e}", exc_info=True)
            st.warning("Could not load historical reports.")