import streamlit as st
import time

def initialize_log_display():
    """
    Initializes the container for the dynamic log display and the log messages list
    in the session state.
    """
    if 'log_container' not in st.session_state:
        st.session_state.log_container = st.container()
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []

def log_to_ui(message: str):
    """
    Appends a message to the session state log queue. This function doesn't
    render the message directly; display_logs() does.

    Args:
        message (str): The log message to display.
    """
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    st.session_state.log_messages.append(message)

def display_logs():
    """
    Renders the log messages from the session state queue in a dynamic,
    conversational format within the designated log container.
    """
    if 'log_container' in st.session_state and 'log_messages' in st.session_state:
        with st.session_state.log_container:
            st.write("#### Thought Process")
            # Display all but the last message as completed
            for msg in st.session_state.log_messages[:-1]:
                st.success(msg, icon="✔️")
            # Display the last message with a spinner as the current activity
            if st.session_state.log_messages:
                with st.spinner(st.session_state.log_messages[-1]):
                    time.sleep(0.5) # Brief pause to make the animation noticeable

def clear_log_display():
    """Clears the log messages and the display area."""
    st.session_state.log_messages = []
    if 'log_container' in st.session_state:
        st.session_state.log_container.empty()