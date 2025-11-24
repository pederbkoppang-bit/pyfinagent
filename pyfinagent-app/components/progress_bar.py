import streamlit as st

def initialize_status_elements():
    """Initializes placeholders for the progress bar and status text in the session state."""
    if 'progress_bar_placeholder' not in st.session_state:
        st.session_state.progress_bar_placeholder = st.empty()
    if 'status_text_placeholder' not in st.session_state:
        st.session_state.status_text_placeholder = st.empty()

def update_progress(value: int, text: str):
    """
    Updates the progress bar and status text based on the current analysis stage.

    Args:
        value (int): The progress value (0-100).
        text (str): The status message to display to the user.
    """
    st.session_state.progress_value = value
    if 'progress_bar_placeholder' in st.session_state:
        st.session_state.progress_bar_placeholder.progress(value)
    
    # Use a more "chat-like" or "thought" bubble for the status text
    if 'status_text_placeholder' in st.session_state:
        # Using a markdown-based container to simulate a "thought" from the agent
        st.session_state.status_text_placeholder.markdown(f"🤔 **Agent thought:** {text}")

def clear_progress():
    """Clears the progress bar and status text from the UI."""
    if 'progress_bar_placeholder' in st.session_state:
        st.session_state.progress_bar_placeholder.empty()
    if 'status_text_placeholder' in st.session_state:
        st.session_state.status_text_placeholder.empty()