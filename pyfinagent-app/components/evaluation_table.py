import streamlit as st

def display_evaluation_table(scores: dict):
    """
    Displays a dashboard-style breakdown of the scoring pillars, their weights,
    and the scores they received.
    """
    if not scores:
        return
    
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

    # This component's responsibility is just the table. The final score is shown elsewhere.
    # We can remove the info box to keep the component focused.