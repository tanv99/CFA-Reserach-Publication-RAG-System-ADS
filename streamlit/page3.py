import streamlit as st
import pandas as pd

def show():
    st.markdown("<h1 style='text-align: center;'>Page Three</h1>", unsafe_allow_html=True)
    
    # Add a container with custom styling
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);'>
            <h2 style='text-align: center; margin-bottom: 1.5rem;'>Welcome to Page Three</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Section One")
        # Add some interactive elements
        option = st.selectbox(
            'Choose an option',
            ['Option A', 'Option B', 'Option C']
        )
        
        number = st.slider('Select a number', 0, 100, 50)
        
        if st.button('Show Selection'):
            st.write(f'You selected {option} and number {number}')
    
    with col2:
        st.subheader("Section Two")
        # Add a file uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head())
                
                st.write("Basic Statistics:")
                st.write(df.describe())
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Add an expandable section
    with st.expander("Additional Information"):
        st.write("""
            This is an expandable section where you can put additional information,
            documentation, or any other content that doesn't need to be immediately visible.
        """)
        
        st.info("You can customize this page further based on your specific needs!")