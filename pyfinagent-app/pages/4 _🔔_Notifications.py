import streamlit as st
import tools_slack
import os

st.set_page_config(page_title="Notifications", page_icon="🔔", layout="wide")

st.title("🔔 Slack Notifications")
st.caption("Manage and test your integration with Slack for real-time agent alerts.")

# --- 1. Configuration Check ---
st.subheader("Configuration Status")

# Check where the webhook is coming from (Environment vs Secrets)
env_webhook = os.getenv("SLACK_WEBHOOK_URL")
secrets_webhook = None
if hasattr(st, "secrets") and "slack" in st.secrets:
    try:
        secrets_webhook = st.secrets.slack.webhook_url
    except AttributeError:
        pass

if env_webhook:
    st.success("✅ **Environment Variable Detected:** `SLACK_WEBHOOK_URL` is set.")
    active_webhook = env_webhook
elif secrets_webhook:
    st.success("✅ **Secrets File Detected:** `[slack] webhook_url` found in `secrets.toml`.")
    active_webhook = secrets_webhook
else:
    st.error("❌ **No Webhook Found:** Please configure your Slack Webhook URL.")
    st.markdown("""
    **How to fix:**
    1. Create a Slack App at [api.slack.com/apps](https://api.slack.com/apps).
    2. Enable **Incoming Webhooks**.
    3. Add the URL to your `.streamlit/secrets.toml` file:
    ```toml
    [slack]
    webhook_url = "[https://hooks.slack.com/services/](https://hooks.slack.com/services/)..."
    ```
    """)
    active_webhook = None

st.divider()

# --- 2. Test Center ---
st.subheader("🚀 Test Your Integration")

with st.form("test_slack"):
    c1, c2 = st.columns([3, 1])
    test_msg = c1.text_input("Test Message", value="This is a test alert from PyFinAgent.")
    alert_type = c2.selectbox("Alert Type", ["info", "success", "warning", "error"])
    
    if st.form_submit_button("Send Test Notification"):
        if active_webhook:
            with st.spinner("Sending..."):
                # Use the tools_slack module to send the message
                tools_slack.send_notification(
                    test_msg,
                    {"User": "Admin", "Source": "Notification Page"},
                    alert_type
                )
                st.success("Notification sent! Check your Slack channel.")
        else:
            st.warning("Cannot send: No Webhook URL configured.")

# --- 3. Recent Alerts Log (Placeholder) ---
st.divider()
st.info("💡 **Pro Tip:** This page allows you to verify that your 'Critical Loss' alerts will reach you before the AI runs autonomously.")