import os
import requests
import json
import logging
import streamlit as st

def get_slack_webhook_url():
    """
    Retrieves the Slack webhook URL from either environment variables or Streamlit secrets.
    This centralized function ensures consistent webhook retrieval across the app.
    """
    # Priority 1: Environment variable (for Docker/production)
    env_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if env_webhook:
        return env_webhook

    # Priority 2: Streamlit secrets (for local development)
    if hasattr(st, "secrets") and "slack" in st.secrets:
        try:
            return st.secrets.slack.webhook_url
        except AttributeError:
            pass

    return None

def send_notification(message: str, metadata: dict, alert_type: str = "info"):
    """
    Sends a formatted notification to a Slack channel using a webhook.

    Args:
        message (str): The main content of the alert.
        metadata (dict): A dictionary of key-value pairs to include as context.
        alert_type (str): The type of alert ('info', 'success', 'warning', 'error').
                          This determines the color of the Slack attachment.
    """
    webhook_url = get_slack_webhook_url()
    if not webhook_url:
        logging.error("Slack webhook URL not found. Cannot send notification.")
        return

    colors = {
        "info": "#17a2b8",      # Blue
        "success": "#28a745",   # Green
        "warning": "#ffc107",   # Yellow
        "error": "#dc3545"      # Red
    }

    # Format metadata into a string for the Slack message body
    metadata_str = "\n".join([f"*{key}:* {value}" for key, value in metadata.items()])

    slack_payload = {
        "attachments": [{
            "color": colors.get(alert_type, "#6c757d"), # Default to grey
            "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": f"*{message}*\n{metadata_str}"}}]
        }]
    }

    try:
        response = requests.post(webhook_url, data=json.dumps(slack_payload), headers={'Content-Type': 'application/json'})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send Slack notification: {e}")
        st.error(f"Error sending to Slack: {e}")