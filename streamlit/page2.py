import streamlit as st

def show():
    st.title("Page Two")
    
    # Add your page content here
    st.header("Welcome to Page Two")
    st.write("This is the content of page two. You can customize this page with your own content.")
    
    # Example content
    with st.expander("Click to expand"):
        st.write("This is expandable content!")
        st.write("You can put any content here.")
        
    # Example form
    with st.form("example_form"):
        st.write("Example Form")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, max_value=120)
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success(f"Form submitted with Name: {name} and Age: {age}")
    
    # Add more sections as needed
    st.markdown("---")
    st.subheader("Data Visualization Example")
    st.write("Add charts or other visualizations here...")