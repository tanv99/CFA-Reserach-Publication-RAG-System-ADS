import streamlit as st

# Title of the Streamlit app
st.title("Hello from Streamlit!")

# A simple input field
name = st.text_input("Enter your name:")

# Display a welcome message when the user enters their name
if name:
    st.write(f"Hello, {name}!")
