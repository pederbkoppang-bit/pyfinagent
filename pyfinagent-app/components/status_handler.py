import streamlit as st
import re

class StatusHandler:
    """
    A robust, centralized handler for displaying analysis progress, status, and logs.
    This class encapsulates a progress bar and a primary status message, using
    st.status to efficiently handle and display detailed streaming logs.
    """
    # Class constant for the spinner's HTML structure.
    _SPINNER_HTML = '<div class="gemini-spinner"><div class="gemini-spinner-bar"></div><div class="gemini-spinner-bar"></div><div class="gemini-spinner-bar"></div><div class="gemini-spinner-bar"></div></div>'

    def __init__(self, total_steps: int, progress_bar):
        """
        Initializes all necessary UI placeholders.

        Args:
            total_steps (int): The total number of major steps in the analysis for the progress bar.
            progress_bar: A Streamlit placeholder (st.empty()) for the progress bar.
        """
        self.total_steps = total_steps
        self.current_step = 0

        # Store the placeholder but only draw the UI elements once.
        # This check prevents re-drawing on every st.rerun.
        if 'status_handler_initialized' not in st.session_state:
            st.session_state._progress_bar_placeholder = progress_bar
            # This placeholder will act as our custom, HTML-enabled status label
            st.session_state._status_label_placeholder = st.empty()
            # st.status will manage the expandable log area and its state.
            st.session_state._status_context = st.status("Logs", expanded=False)
            self._inject_spinner_css() # Inject CSS once
            st.session_state.status_handler_initialized = True

    def update_step(self, status_text: str):
        """
        Updates the progress bar and status message. It can handle main steps ("Step 1: ...")
        and sub-steps ("Step 1.1: ...") to show more granular progress.
        """
        # Extract step number (e.g., "1" from "Step 1:" or "8.1" from "Step 8.1:")
        match = re.search(r'Step ([\d\.]+):', status_text)
        if match:
            step_num_str = match.group(1)
            try:
                step_num = float(step_num_str)
                # Only update if we are moving forward.
                if step_num > self.current_step:
                    self.current_step = step_num
            except ValueError:
                # Fallback for non-numeric/float step numbers, though the regex should prevent this.
                pass
        else:
            self.current_step += 0.1 # Increment slightly for unnumbered steps

        progress_value = int((self.current_step / self.total_steps) * 100)
        st.session_state._progress_bar_placeholder.progress(min(progress_value, 100), text=status_text)
        # The main step update is a significant log, so we treat it as such.
        self.log(f"✅ {status_text}") # Also add the major step to the detailed log

    def _inject_spinner_css(self):
        """
        Injects the CSS for the Gemini spinner into the app.
        This should only be called once.
        """
        st.markdown("""
            <style>
                .gemini-spinner {
                    display: inline-flex; /* Use inline-flex to appear next to text */
                    justify-content: center;
                    align-items: center;
                    gap: 5px;
                    vertical-align: middle; /* Align spinner with the text */
                    margin-right: 10px; /* Space between spinner and text */
                }
                .gemini-spinner-bar {
                    width: 4px;
                    height: 16px;
                    background-color: #4285F4; /* Google Blue */
                    border-radius: 2px;
                    animation: bounce 1.2s ease-in-out infinite;
                }
                .gemini-spinner-bar:nth-child(2) { background-color: #DB4437; animation-delay: -1.0s; } /* Google Red */
                .gemini-spinner-bar:nth-child(3) { background-color: #F4B400; animation-delay: -0.8s; } /* Google Yellow */
                .gemini-spinner-bar:nth-child(4) { background-color: #0F9D58; animation-delay: -0.6s; } /* Google Green */
                @keyframes bounce {
                    0%, 80%, 100% { transform: scaleY(0.5); opacity: 0.5; }
                    40% { transform: scaleY(1.0); opacity: 1.0; }
                }
            </style>
        """, unsafe_allow_html=True)

    def log(self, message: str):
        """
        Updates the status with a custom Gemini-style spinner and the latest message.
        It also logs the message inside the expandable container.
        """
        # Update the external placeholder (our custom label) with the spinner and message
        st.session_state._status_label_placeholder.markdown(f"{self._SPINNER_HTML} {message}", unsafe_allow_html=True)

        with st.session_state._status_context:
            st.write(f"> {message}") # Log message with a quote for distinction

    def complete(self, final_message: str = "Analysis Complete!"):
        """Clears the progress bar and shows a final completion message."""
        st.session_state._progress_bar_placeholder.empty()
        st.session_state._status_label_placeholder.empty() # Clear the custom label
        # Set the final state of the status container
        st.session_state._status_context.update(label=final_message, state="complete", expanded=False)

    def error(self, error_message: str):
        """Displays an error message in the status area."""
        st.session_state._status_context.update(label=error_message, state="error", expanded=True)
        # We can also log the error inside the container for more details if needed
        self.log(f"🚨 ERROR: {error_message}")