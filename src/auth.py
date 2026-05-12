import streamlit as st

def login():

    st.sidebar.title("🔐 Login")

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):

        if username == "admin" and password == "admin":
            st.session_state.auth = True
            st.sidebar.success("Login Successful")

        else:
            st.sidebar.error("Invalid Credentials")



def check_auth():

    return st.session_state.get("auth", False)