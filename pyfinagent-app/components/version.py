import streamlit as st
import os

def display_version():
    """
    Renders a compact, badge-style version display for the sidebar.
    """
    app_version = os.getenv("APP_VERSION", "local")
    st.markdown(
        f"""
        <div style="
            background-color: #262730; 
            border-radius: 0.5rem; 
            padding: 0.2rem 0.6rem; 
            text-align: center;
            display: inline-block;">
            <span style="font-size: 0.75rem; color: #808495;">Version: {app_version}</span>
        </div>
        """, unsafe_allow_html=True)