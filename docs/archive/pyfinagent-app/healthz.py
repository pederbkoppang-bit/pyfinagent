import streamlit as st

def main():
    """
    A simple, unauthenticated health check endpoint.
    It returns a 200 OK status if the Streamlit server is running.
    """
    st.set_page_config(layout="centered")
    st.text("OK")

if __name__ == "__main__":
    main()