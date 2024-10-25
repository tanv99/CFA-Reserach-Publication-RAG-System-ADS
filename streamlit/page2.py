import streamlit as st
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

FASTAPI_URL = os.getenv("FASTAPI_URL")

def load_css():
    st.markdown("""
        <style>
            .summary-container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .summary-content {
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 4px;
                margin: 1rem 0;
            }
            
            .summary-title {
                text-align: center;
                margin-bottom: 1.5rem;
            }
            
            .nav-buttons {
                display: flex;
                gap: 1rem;
                justify-content: center;
                margin: 1rem 0;
            }
        </style>
    """, unsafe_allow_html=True)

def show():
    load_css()
    
    # Get the PDF title from session state
    pdf_title = st.session_state.get('pdf_for_summary')
    
    if not pdf_title:
        st.info("Please select a PDF from the PDF Selection page first")
        return
    
    try:
        # Display the summary page content
        st.markdown(f"<h1 class='summary-title'>Summary: {pdf_title}</h1>", unsafe_allow_html=True)
        
        # Navigation buttons at the top
        with st.container():
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("← Back to PDF View", use_container_width=True):
                    st.session_state['current_page'] = "PDF Selection"
                    st.rerun()
        
        # Summary content
        with st.container():
            st.markdown("<div class='summary-content'>", unsafe_allow_html=True)
            
            # Add a loading spinner while generating summary
            with st.spinner("Generating summary..."):
                # Make API call to get summary
                try:
                    response = requests.get(f"{FASTAPI_URL}/pdfs/{pdf_title}/summary")
                    if response.status_code == 200:
                        summary_data = response.json()
                        
                        # Display summary sections
                        st.subheader("Key Points")
                        st.write(summary_data.get("key_points", "No key points available"))
                        
                        st.subheader("Main Topics")
                        st.write(summary_data.get("main_topics", "No main topics available"))
                        
                        st.subheader("Detailed Summary")
                        st.write(summary_data.get("detailed_summary", "No detailed summary available"))
                    else:
                        st.error("Failed to load summary. Please try again.")
                except Exception as e:
                    st.error("Error connecting to server")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Bottom navigation
        with st.container():
            if st.button("← Back to PDF View", key="bottom_back", use_container_width=True):
                st.session_state['current_page'] = "PDF Selection"
                st.rerun()
                    
    except Exception as e:
        st.error("An error occurred while loading the summary page")
        if st.button("← Back to PDF Selection"):
            st.session_state['current_page'] = "PDF Selection"
            st.rerun()

if __name__ == "__main__":
    show()