import streamlit as st
import requests
import base64
from io import BytesIO
import os
from datetime import datetime
from typing import Dict, Any
import urllib.parse

# Load environment variables
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

def init_session_state():
    """Initialize session state variables"""
    if 'selected_pdf' not in st.session_state:
        st.session_state.selected_pdf = None
    if 'current_answer' not in st.session_state:
        st.session_state.current_answer = None
    if 'saved_notes' not in st.session_state:
        st.session_state.saved_notes = []

def load_css():
    """Load custom CSS styles"""
    st.markdown("""
        <style>
            /* Document selection */
            .doc-selector {
                background-color: #ffffff;
                padding: 1.5rem;
                border-radius: 8px;
                margin: 1rem auto;
                max-width: 800px;
                border: 1px solid #e9ecef;
            }
            
            /* PDF viewer */
            .pdf-container {
                width: 100%;
                max-width: 800px;
                margin: 1rem auto;
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .pdf-viewer {
                width: 100%;
                height: 800px;
                border: none;
                border-radius: 4px;
            }
            
            /* QA section */
            .qa-container {
                background-color: #f8f9fa;
                padding: 1.5rem;
                border-radius: 8px;
                margin: 1rem auto;
                max-width: 800px;
            }
            
            .answer-container {
                background-color: #ffffff;
                padding: 1.5rem;
                border-radius: 8px;
                margin: 1rem 0;
                border: 1px solid #e9ecef;
            }
            
            .evidence-container {
                background-color: #f1f3f5;
                padding: 1rem;
                border-radius: 4px;
                margin: 0.5rem 0;
                font-size: 0.9rem;
            }
            
            .score-badge {
                background-color: #4CAF50;
                color: white;
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                font-size: 0.8rem;
            }
            
            iframe {
                border: none !important;
                width: 100% !important;
                height: 800px !important;
            }
        </style>
    """, unsafe_allow_html=True)

def display_pdf_viewer(pdf_name: str):
    """Display the PDF viewer"""
    try:
        response = requests.get(f"{FASTAPI_URL}/pdfs/{pdf_name}/document")
        response.raise_for_status()
        
        # Convert PDF to base64
        base64_pdf = base64.b64encode(response.content).decode('utf-8')
        
        # PDF container
        st.markdown('<div class="pdf-container">', unsafe_allow_html=True)
        
        # Embed PDF viewer
        pdf_display = (
            f'<iframe class="pdf-viewer" '
            f'src="data:application/pdf;base64,{base64_pdf}#toolbar=1&navpanes=1&scrollbar=1" '
            f'type="application/pdf" '
            f'></iframe>'
        )
        st.markdown(pdf_display, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add download button below viewer
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="Download PDF",
                data=response.content,
                file_name=f"{pdf_name}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
    except requests.RequestException as e:
        st.error(f"Error loading PDF: {str(e)}")

def fetch_pdfs():
    """Fetch available PDFs from the API"""
    try:
        response = requests.get(f"{FASTAPI_URL}/pdfs/all")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Failed to load PDF list: {str(e)}")
        return []

def ask_question(query: str, folder_name: str, top_k: int = 5):
    """Send question to API and get response"""
    try:
        # URL encode the folder name to handle special characters
        encoded_folder_name = urllib.parse.quote(folder_name)
        
        # Format the query according to the SearchQuery model
        search_query = {
            "query": query,
            "top_k": top_k,
            "pdf_id": folder_name  # Add the required pdf_id field
        }
        
        response = requests.post(
            f"{FASTAPI_URL}/pdfs/{encoded_folder_name}/search-and-process",
            json=search_query
        )
        
        # If we get a 422 error, log the request details for debugging
        if response.status_code == 422:
            st.error(f"Request validation error. Details: {response.text}")
            return None
            
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error getting answer: {str(e)}")
        if hasattr(e.response, 'text'):
            st.error(f"Response details: {e.response.text}")
        return None

def save_as_notes(answer_data: Dict):
    """Save the current answer as notes"""
    note = {
        "timestamp": datetime.now().isoformat(),
        "question": answer_data.get("query", ""),  # Updated to handle new response structure
        "answer": answer_data.get("analysis", {}).get("summary", ""),
        "document": answer_data.get("query_info", {}).get("folder_name", "Unknown"),
    }
    st.session_state.saved_notes.append(note)
    return len(st.session_state.saved_notes)

def show():
    """Main function to display the document Q&A interface"""
    init_session_state()
    load_css()
    
    st.title("Document Q&A Interface")
    
    # Document Selection
    st.markdown("<div class='doc-selector'>", unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)
    
    if selected_pdf:
        st.session_state.selected_pdf = selected_pdf
        
        # PDF Viewer (centered)
        display_pdf_viewer(selected_pdf)
        
        # Q&A Interface (below PDF)
        st.markdown("<div class='qa-container'>", unsafe_allow_html=True)
        
        # Question input and controls in columns
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_area(
                "Ask a question about this document:",
                height=100,
                placeholder="Enter your question here..."
            )
        
        with col2:
            top_k = st.number_input(
                "Results",
                min_value=1,
                max_value=10,
                value=5,
                help="Number of relevant chunks to analyze"
            )

            if st.button("Ask", use_container_width=True):
                if question:
                    with st.spinner("Analyzing document..."):
                        result = ask_question(question, selected_pdf, top_k)
                        if result:  # Only update if we got a valid response
                            st.session_state.current_answer = result
        
        # Display answer if available
        if st.session_state.current_answer:
            result = st.session_state.current_answer
            if result and result.get("status") == "success":
                # Display answer
                st.markdown("<div class='answer-container'>", unsafe_allow_html=True)
                st.markdown("### Analysis")
                st.write(result["analysis"]["summary"])
                
                # Save as Notes button
                if st.button("Save as Notes", use_container_width=True):
                    note_number = save_as_notes(result)
                    st.success(f"Saved as Note #{note_number}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display supporting evidence
                with st.expander("View Source Materials"):
                    for chunk in result["analysis"]["source_materials"]:
                        st.markdown("<div class='evidence-container'>", unsafe_allow_html=True)
                        st.markdown(f"<span class='score-badge'>Relevance: {chunk['relevance_score']:.3f}</span>", 
                                  unsafe_allow_html=True)
                        st.markdown(f"**Page {chunk['page_number']}**")
                        st.write(chunk['content'])
                        
                        # Display image if available
                        if chunk['image_info']['available']:
                            try:
                                st.image(chunk['image_info']['path'], 
                                       caption=f"Related image from page {chunk['page_number']}")
                            except Exception as e:
                                st.warning(f"Could not load image: {chunk['image_info']['path']}")
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Display metadata
                with st.expander("Analysis Details"):
                    st.json({
                        "Processing Time": result["analysis"]["metadata"]["processing_timestamp"],
                        "Model Used": result["analysis"]["metadata"]["model_used"],
                        "Chunks Analyzed": result["analysis"]["metadata"]["total_chunks"]
                    })
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    show()