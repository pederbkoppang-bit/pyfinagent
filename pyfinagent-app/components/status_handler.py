import streamlit as st
import re

class StatusHandler:
    """
    A robust, centralized handler for displaying analysis progress, status, and logs.
    This class encapsulates a progress bar and a primary status message, using
    st.status to efficiently handle and display detailed streaming logs.
    """
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
            # st.status will manage the expandable log area and its state.
            st.session_state._status_context = st.status("Initializing analysis...", expanded=False)
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

    def log(self, message: str):
        """
        Updates the status label to the latest message.
        """
        st.session_state._status_context.update(label=message)

    def complete(self, final_message: str = "Analysis Complete!"):
        """Clears the progress bar and shows a final completion message."""
        st.session_state._progress_bar_placeholder.empty()
        # Set the final state of the status container
        st.session_state._status_context.update(label=final_message, state="complete", expanded=False)

    def error(self, error_message: str):
        """Displays an error message in the status area."""
        st.session_state._status_context.update(label=error_message, state="error", expanded=True)
        # We can also log the error inside the container for more details if needed
        self.log(f"🚨 ERROR: {error_message}")