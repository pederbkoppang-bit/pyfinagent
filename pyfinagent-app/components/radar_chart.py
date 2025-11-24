import streamlit as st
import plotly.graph_objects as go

def display_radar_chart():
    """
    Displays a radar chart of the scoring matrix from the session state report.
    This provides a quick visual summary of the analysis pillars.
    """
    if 'report' not in st.session_state or not st.session_state.report.get('final_synthesis'):
        return

    score_data = st.session_state.report['final_synthesis'].get('scoring_matrix', {})
    if not score_data:
        st.warning("Scoring data not available to generate radar chart.")
        return

    # A more readable mapping from pillar keys to display names
    pillar_names = {
        'pillar_1_corporate': 'Corporate',
        'pillar_2_industry': 'Industry',
        'pillar_3_valuation': 'Valuation',
        'pillar_4_sentiment': 'Sentiment',
        'pillar_5_governance': 'Governance'
    }

    categories = [pillar_names.get(key, key) for key in score_data.keys()]
    values = list(score_data.values())

    # To close the loop of the radar chart, the first value must be appended to the end
    values_closed = values + [values[0]]
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='Score'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10]) # Scores are from 0 to 10
        ),
        showlegend=False,
        title={'text': "Analysis Pillars Radar Chart", 'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
        margin=dict(l=40, r=40, t=80, b=40) # Adjust margins
    )

    st.plotly_chart(fig, use_container_width=True)