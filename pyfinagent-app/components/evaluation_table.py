import streamlit as st

def display_evaluation_table():
    """
    Displays a dashboard-style breakdown of the scoring pillars, their weights,
    and the scores they received.
    """
    if 'report' not in st.session_state or not st.session_state.report.get('final_synthesis'):
        return

    report_data = st.session_state.report['final_synthesis']
    scores = report_data.get('scoring_matrix', {})
    
    # Define the pillars and their weights
    pillar_weights = {
        "Corporate Profile": {"key": "pillar_1_corporate", "weight": 0.35},
        "Industry & Macro": {"key": "pillar_2_industry", "weight": 0.20},
        "Valuation": {"key": "pillar_3_valuation", "weight": 0.20},
        "Market Sentiment": {"key": "pillar_4_sentiment", "weight": 0.15},
        "Governance": {"key": "pillar_5_governance", "weight": 0.10},
    }

    st.subheader("Evaluation Score Breakdown")
    
    # Create 5 columns for the 5 pillars
    cols = st.columns(5)
    
    # Use a list comprehension to iterate through pillars and columns together
    for col, (pillar_name, info) in zip(cols, pillar_weights.items()):
        with col:
            score = scores.get(info['key'], 0.0)
            st.metric(
                label=pillar_name,
                value=f"{score:.2f}",
                help=f"This pillar has a {info['weight']:.0%} weight in the final score."
            )

    st.info(f"The final weighted score of **{report_data.get('final_weighted_score', 0.0):.2f} / 10** is calculated based on the scores and weights of these pillars.")