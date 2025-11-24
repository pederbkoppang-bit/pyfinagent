import streamlit as st

def initialize_log_display():
    """Initializes a placeholder for the log display in the session state."""
    if 'log_display_placeholder' not in st.session_state:
        st.session_state.log_display_placeholder = st.empty()
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []

def log_to_ui(message: str):
    """
    Appends a message to the UI log display.

    Args:
        message (str): The log message to display.
    """
    st.session_state.log_messages.append(f"- {message}")
    # Display the last 10 messages to keep the UI clean
    log_content = "\n".join(st.session_state.log_messages[-10:])
    st.session_state.log_display_placeholder.info(f"**Latest Activities:**\n\n{log_content}")

def clear_log_display():
    """Clears the log messages and the display area."""
    st.session_state.log_messages = []
    if 'log_display_placeholder' in st.session_state:
        st.session_state.log_display_placeholder.empty()