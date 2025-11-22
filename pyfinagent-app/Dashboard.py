import streamlit as st
import os

st.set_page_config(page_title="Dashboard", page_icon="📈", layout="wide")

# Get the version from the environment variable. Default to 'local' if not set.
app_version = os.getenv("APP_VERSION", "local")

# Display the version in the sidebar
st.sidebar.info(f"Version: {app_version}")

st.title("Dashboard")
st.write("Welcome to the PyFinAgent Dashboard. Please select a page from the sidebar.")