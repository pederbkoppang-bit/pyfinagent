import streamlit as st
from collections import deque

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
        self._logs = deque(maxlen=10) # Store the last 10 log messages

        # Store the placeholder but only draw the UI elements once.
        # This check prevents re-drawing on every st.rerun.
        if not hasattr(self, '_progress_bar_placeholder'):
            self._progress_bar_placeholder = progress_bar

            # st.status will manage the expandable log area and its state.
            # Start collapsed and create a placeholder for the log content.
            st.divider() # Add a divider before the status log
            self._status_context = st.status("Initializing analysis...", expanded=False)
            self._log_placeholder = self._status_context.empty()


    def update_step(self, status_text: str):
        """
        Advances the progress by one step and updates the primary status message.
        """
        self.current_step += 1
        progress_value = int((self.current_step / self.total_steps) * 100)
        
        self._progress_bar_placeholder.progress(progress_value, text=status_text)
        # The main step update is a significant log, so we treat it as such.
        self.log(f"✅ {status_text}") # Also add the major step to the detailed log

    def log(self, message: str):
        """
        Appends a message to the log queue, updates the status label, and
        refreshes the displayed logs.
        """
        self._logs.append(message)
        
        # Update the main status label to the latest message
        self._status_context.update(label=message)
        
        # Update the content inside the collapsed status box
        self._log_placeholder.markdown("\n".join(f"- {log}" for log in self._logs))

    def complete(self, final_message: str = "Analysis Complete!"):
        """Clears the progress bar and shows a final completion message."""
        self._progress_bar_placeholder.empty()
        # Set the final state of the status container
        self._status_context.update(label=final_message, state="complete", expanded=False)

    def error(self, error_message: str):
        """Displays an error message in the status area."""
        # Set the final state of the status container to 'error'
        self._status_context.update(label=error_message, state="error", expanded=True)
        # We can also log the error inside the container for more details if needed
        self.log(f"🚨 ERROR: {error_message}")