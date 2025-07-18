import streamlit as st
from datetime import datetime

# Fixed username and password
USERNAME = "LTPlabs"
PASSWORD = datetime.today().strftime("%m") + datetime.today().strftime("%Y")

def login():
    # Input fields
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Login button
    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state["logged_in"] = True
            st.session_state.page = 1
            st.rerun()
        else:
            st.error("Invalid username or password")