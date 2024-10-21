import streamlit as st

def show():
    st.title("Page One")
    
    # Add your page content here
    st.header("Welcome to Page One")
    st.write("This is the content of page one. You can customize this page with your own content.")
    
    # Example content
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Section 1")
            st.write("This is the left column content.")
            if st.button("Click Me!", key="page1_button1"):
                st.write("Button clicked!")
                
        with col2:
            st.subheader("Section 2")
            st.write("This is the right column content.")
            user_input = st.text_input("Enter something:", key="page1_input1")
            if user_input:
                st.write(f"You entered: {user_input}")

    # Add more sections as needed
    st.markdown("---")
    st.subheader("Additional Content")
    st.write("Add more content here...")