import streamlit as st
import pandas as pd
from components.stock_chart import display_price_chart

def display_reports_comparison():
    """
    Renders a sophisticated, Bloomberg-style comparison view for multiple loaded reports.
    """
    reports = st.session_state.get("loaded_reports", [])
    if not reports:
        return

    st.header(f"Comparative Analysis for {st.session_state.ticker}")
    st.markdown(f"Comparing **{len(reports)}** reports.")
    st.divider()

    # --- 1. Stock Chart with Analysis Markers ---
    # Display the price chart first to provide historical context for the analyses.
    st.subheader("Price Performance Context")
    if st.session_state.ticker:
        # Pass the report dates to the chart function to create markers
        analysis_dates = [pd.to_datetime(r.get('analysis_date')) for r in reports]
        display_price_chart(analysis_dates=analysis_dates)
    
    st.divider()

    # --- 2. Comparative Score Table ---
    st.subheader("Quantitative Score Comparison")
    comparison_data = []
    for report in reports:
        synthesis = report.get('final_synthesis', {})
        scores = synthesis.get('scoring_matrix', {})
        comparison_data.append({
            "Analysis Date": pd.to_datetime(report.get('analysis_date')).strftime('%Y-%m-%d %H:%M'),
            "Final Score": synthesis.get('final_weighted_score'),
            "Recommendation": synthesis.get('recommendation', {}).get('action'),
            "Corporate Profile": scores.get('pillar_1_corporate'),
            "Industry & Macro": scores.get('pillar_2_industry'),
            "Valuation": scores.get('pillar_3_valuation'),
            "Sentiment": scores.get('pillar_4_sentiment'),
            "Governance": scores.get('pillar_5_governance'),
            # Keep the full summary for the expander below
            "summary_text": synthesis.get('final_summary'),
            "justification_text": synthesis.get('recommendation', {}).get('justification')
        })
    
    # Create and display the main comparison DataFrame
    df = pd.DataFrame(comparison_data).sort_values(by="Analysis Date", ascending=False).reset_index(drop=True)
    
    # Define columns to display in the table (exclude the text for expanders)
    display_columns = [col for col in df.columns if '_text' not in col]
    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Final Score": st.column_config.NumberColumn(format="%.2f"),
            "Corporate Profile": st.column_config.NumberColumn(format="%.2f"),
            "Industry & Macro": st.column_config.NumberColumn(format="%.2f"),
            "Valuation": st.column_config.NumberColumn(format="%.2f"),
            "Sentiment": st.column_config.NumberColumn(format="%.2f"),
            "Governance": st.column_config.NumberColumn(format="%.2f"),
        }
    )
    st.divider()

    # --- 3. Detailed Qualitative Summaries ---
    st.subheader("Qualitative Analysis Breakdown")
    for index, row in df.iterrows():
        with st.expander(f"**Analysis from {row['Analysis Date']}** - Recommendation: **{row['Recommendation']}**"):
            st.markdown("##### Justification")
            st.write(row['justification_text'])
            st.markdown("##### Summary")
            st.write(row['summary_text'])