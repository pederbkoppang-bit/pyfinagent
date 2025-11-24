import streamlit as st

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")

st.title("⚙️ Settings")
st.caption("Adjust application parameters here.")

# --- Default Weights Definition ---
# This dictionary is the single source of truth for default weights.
DEFAULT_WEIGHTS = {
    'pillar_1_corporate': 0.35,
    'pillar_2_industry': 0.20,
    'pillar_3_valuation': 0.20,
    'pillar_4_sentiment': 0.15,
    'pillar_5_governance': 0.10
}

# Initialize weights in session state if they don't exist
if 'score_weights' not in st.session_state:
    st.session_state.score_weights = DEFAULT_WEIGHTS.copy()


st.header("Final Score Calculation Weights")
st.write(
    "Adjust the sliders to change the importance of each pillar in the final weighted score. "
    "The total of all weights should equal 100%."
)

weights = st.session_state.score_weights

# Use a form to group the sliders
with st.form("weights_form"):
    weights['pillar_1_corporate'] = st.slider("Pillar 1: Corporate (Business Model, Financials, Moat)", 0.0, 1.0, weights['pillar_1_corporate'], 0.05)
    weights['pillar_2_industry'] = st.slider("Pillar 2: Industry (Trends, Macro Factors)", 0.0, 1.0, weights['pillar_2_industry'], 0.05)
    weights['pillar_3_valuation'] = st.slider("Pillar 3: Valuation (Metrics, Comps)", 0.0, 1.0, weights['pillar_3_valuation'], 0.05)
    weights['pillar_4_sentiment'] = st.slider("Pillar 4: Market Sentiment (News, Social)", 0.0, 1.0, weights['pillar_4_sentiment'], 0.05)
    weights['pillar_5_governance'] = st.slider("Pillar 5: Governance (Compensation, Shareholder Friendliness)", 0.0, 1.0, weights['pillar_5_governance'], 0.05)

    submitted = st.form_submit_button("Save Weights")
    if submitted:
        st.success("Weights saved successfully!")

# --- Validation and Reset ---
total_weight = sum(weights.values())
st.metric("Total Weight", f"{total_weight:.0%}")

if not (0.999 < total_weight < 1.001): # Use a small tolerance for float precision
    st.warning(f"The total weight is {total_weight:.0%}, which is not 100%. Please adjust the sliders.")

col1, col2 = st.columns(2)
with col1:
    if st.button("Normalize to 100%", use_container_width=True):
        if total_weight == 0:
            st.error("Cannot normalize weights when the total is zero. Please set at least one weight.")
        else:
            normalized_weights = {key: value / total_weight for key, value in weights.items()}
            st.session_state.score_weights = normalized_weights
            st.success("Weights have been normalized to 100%.")
            st.rerun()

with col2:
    if st.button("Reset to Default Weights", use_container_width=True):
        st.session_state.score_weights = DEFAULT_WEIGHTS.copy()
        st.success("Weights have been reset to their default values.")
        st.rerun()