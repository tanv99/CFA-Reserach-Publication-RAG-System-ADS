import streamlit as st
import requests
from dotenv import load_dotenv
import os
import pathlib

env_path = pathlib.Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

FASTAPI_URL = os.getenv("FASTAPI_URL")  # Adjust based on where FastAPI is running

def display_registration_instructions():
    st.subheader("Registration Instructions")
    st.write("Please provide the following information to register:")
    st.write("- **Username**: Your desired username.")
    st.write("- **Email**: A valid email address.")
    st.write("- **Password**: A password that is at least 8 characters long and contains at least one uppercase letter and one lowercase letter.")
    st.write("- **Confirm Password**: Re-enter your password to confirm it.")
    st.write("After filling in all the fields, click the 'Submit Registration' button to complete the registration process.")

def register_user(username, email, password, confirm_password):
    url = f"{FASTAPI_URL}/register"
    data = {
        "username": username,
        "email": email,
        "password": password,
        "confirm_password": confirm_password
    }
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        st.success("User registered successfully!")
    else:
        error_details = response.json().get('detail')
        if isinstance(error_details, list):
            error_message = '\n'.join([error['msg'] for error in error_details])
        else:
            error_message = error_details
        st.error(f"Registration failed: {error_message}")

def login_user(email, password):
    url = f"{FASTAPI_URL}/token"
    data = {
        "username": email,
        "password": password
    }
    response = requests.post(url, data=data)
    
    if response.status_code == 200:
        token_data = response.json()
        return token_data['access_token']
    else:
        st.error(response.json().get('detail'))
        return None

def get_current_user(token):
    url = f"{FASTAPI_URL}/users/me"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to fetch current user details.")
        return None

def main():
    st.title("User Authentication with FastAPI")

    if 'show_login' not in st.session_state:
        st.session_state.show_login = False
    if 'show_register' not in st.session_state:
        st.session_state.show_register = False

    if st.button("Login"):
        st.session_state.show_login = True
        st.session_state.show_register = False
    if st.button("Register"):
        st.session_state.show_register = True
        st.session_state.show_login = False

    if st.session_state.show_login:
        st.subheader("Login")
        login_email = st.text_input("Enter email", key="login_email")
        login_password = st.text_input("Enter password", type="password", key="login_password")
        if st.button("Submit Login"):
            if login_email and login_password:
                token = login_user(login_email, login_password)
                if token:
                    st.session_state['token'] = token
                    st.success("Login successful!")
            else:
                st.error("Please enter both email and password")

    if st.session_state.show_register:
        st.subheader("Register")
        display_registration_instructions()

        register_username = st.text_input("Enter username", key="register_username")
        register_email = st.text_input("Enter email", key="register_email")
        register_password = st.text_input("Enter password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm password", type="password", key="confirm_password")
        if st.button("Submit Registration"):
            if register_username and register_email and register_password and confirm_password:
                register_user(register_username, register_email, register_password, confirm_password)
            else:
                st.error("Please fill in all fields")

    if 'token' in st.session_state:
        st.subheader("Logged in as:")
        user_info = get_current_user(st.session_state['token'])
        if user_info:
            st.write(f"Email: {user_info['email']}")

if __name__ == "__main__":
    main()