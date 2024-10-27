import streamlit as st
import requests
import base64
from io import BytesIO
import os
from datetime import datetime
from typing import Dict, Any

# Load environment variables
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

def init_session_state():
    """Initialize session state variables"""
    if 'selected_pdf' not in st.session_state:
        st.session_state.selected_pdf = None

def load_css():
    """Load custom CSS styles"""
    st.markdown("""
        <style>
            /* Main container */
            .pdf-container {
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
            
            /* PDF viewer */
            .pdf-viewer {
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-top: 1rem;
            }
            
            /* Document info */
            .doc-info {
                background-color: #ffffff;
                padding: 1.5rem;
                border-radius: 8px;
                margin: 1rem 0;
                border: 1px solid #e9ecef;
            }
            
            /* PDF iframe */
            .pdf-iframe {
                width: 100%;
                height: 750px;
                border: none;
                border-radius: 8px;
            }
            
            /* Cover image */
            .cover-image {
                max-width: 200px;
                margin: 1rem 0;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    """, unsafe_allow_html=True)

def fetch_pdfs():
    """Fetch available PDFs from the API"""
    try:
        response = requests.get(f"{FASTAPI_URL}/pdfs/all")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to load PDF list: {str(e)}")
        return []


def display_pdf_viewer(pdf_name: str):
    """Display the PDF viewer"""
    try:
        # Get PDF document from the document endpoint
        response = requests.get(f"{FASTAPI_URL}/pdfs/{pdf_name}/document")
        response.raise_for_status()
        
        st.markdown("<div class='pdf-viewer'>", unsafe_allow_html=True)
        
        # Display PDF using iframe
        pdf_display = F'<iframe src="data:application/pdf;base64,{base64.b64encode(response.content).decode()}" class="pdf-iframe"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        # Add download button
        st.download_button(
            label="Download PDF",
            data=response.content,
            file_name=f"{pdf_name}.pdf",
            mime="application/pdf"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    except requests.RequestException as e:
        st.error(f"Error loading PDF: {str(e)}")
    except Exception as e:
        st.error(f"Error displaying PDF viewer: {str(e)}")

def show():
    """Main function to display the PDF viewer page"""
    init_session_state()
    load_css()
    
    st.title("PDF Document Viewer")
    
    # Create layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Fetch and display PDF list
        pdfs = fetch_pdfs()
        if not pdfs:
            st.warning("No documents available")
            return
        
        selected_pdf = st.selectbox(
            "Select a document",
            options=[""] + [pdf['title'] for pdf in pdfs],
            index=0,
            key="pdf_selector"
        )
        
        if selected_pdf:
            # Find the selected PDF data
            pdf_data = next((pdf for pdf in pdfs if pdf['title'] == selected_pdf), None)
            if pdf_data:
                st.session_state.selected_pdf = selected_pdf
                # st.session_state.pdf_metadata = pdf_data
                
                # Display document summary button
                # if st.button("Toggle Summary", use_container_width=True):
                #     st.session_state.show_summary = not st.session_state.get('show_summary', False)
    
    with col2:
        if st.session_state.selected_pdf:
            st.markdown("<div class='pdf-container'>", unsafe_allow_html=True)
            
            # Display PDF viewer
            display_pdf_viewer(st.session_state.selected_pdf)

if __name__ == "__main__":
    show()