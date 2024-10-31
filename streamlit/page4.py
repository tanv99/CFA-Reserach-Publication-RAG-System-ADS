# pages/page4.py

import streamlit as st
import requests
import logging
from datetime import datetime
from typing import Dict, List
import json
import os

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi-app:8000")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_page_state():
    """Initialize session state variables for the page"""
    if 'selected_document' not in st.session_state:
        st.session_state.selected_document = None
    if 'current_notes' not in st.session_state:
        st.session_state.current_notes = []
    if 'sort_order' not in st.session_state:
        st.session_state.sort_order = "Newest First"

def fetch_pdfs() -> List[Dict]:
    """Fetch list of available PDFs"""
    try:
        response = requests.get(f"{FASTAPI_URL}/pdfs/all")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to fetch PDFs: Status code {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching PDFs: {str(e)}")
        return []

def fetch_document_notes(document_id: str) -> List[Dict]:
    """Fetch research notes for a specific document"""
    try:
        with st.spinner("Loading notes..."):
            response = requests.get(
                f"{FASTAPI_URL}/pdfs/{document_id}/notes"
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("status") == "success":
                    notes = result.get("notes", [])
                    logger.info(f"Retrieved {len(notes)} notes for document {document_id}")
                    return notes
            logger.error(f"Failed to fetch notes: Status code {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error fetching notes: {str(e)}")
        return []

def format_timestamp(timestamp: str) -> str:
    """Format ISO timestamp to readable date/time"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%B %d, %Y %I:%M %p")
    except Exception as e:
        logger.error(f"Error formatting timestamp: {str(e)}")
        return timestamp

def render_image(image_path: str):
    """Render an image from a given path"""
    try:
        image_url = f"{FASTAPI_URL}{image_path}"
        logger.info(f"Loading image from: {image_url}")
        
        response = requests.get(image_url)
        if response.status_code == 200:
            st.image(
                response.content,
                use_column_width=True,
                caption="Document Image"
            )
        else:
            st.warning(f"Could not load image (Status: {response.status_code})")
    except Exception as e:
        logger.error(f"Error rendering image: {str(e)}")
        st.warning("Failed to load image")

def render_note_card(note: Dict):
    """Render a single note card"""
    try:
        with st.container():
            # Apply custom styling
            st.markdown("""
                <style>
                .note-card {
                    border-left: 4px solid #0066cc;
                    padding: 1rem;
                    margin: 1rem 0;
                    background-color: #f8f9fa;
                    border-radius: 4px;
                }
                .note-metadata {
                    color: #666;
                    font-size: 0.9rem;
                    margin-bottom: 0.5rem;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Note header
            st.markdown(f"""
                <div class="note-card">
                    <div class="note-metadata">
                        📝 Created: {format_timestamp(note['timestamp'])}<br>
                        🔍 Query: "{note['query']}"
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Note content
            with st.expander("View Note Content", expanded=False):
                if note.get('content'):
                    st.markdown(note['content'])
                else:
                    st.info("No text content available")
                
                # Display images if available
                if note.get('image_paths'):
                    st.markdown("### Related Images")
                    for image_path in note['image_paths']:
                        render_image(image_path)
                
            st.markdown("---")
    except Exception as e:
        logger.error(f"Error rendering note card: {str(e)}")
        st.error("Error displaying note")

def show():
    """Main function to display the research notes page"""
    try:
        st.title("Research Notes")
        
        # Initialize page state
        init_page_state()
        
        # Add description
        st.markdown("""
            View and manage your saved research notes for each document.
            Select a document to see its associated notes.
        """)
        
        # Create two columns for document selection and filters
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Document selector
            pdfs = fetch_pdfs()
            if pdfs:
                options = [""] + [pdf['title'] for pdf in pdfs]
                selected_doc = st.selectbox(
                    "Select a document",
                    options=options,
                    index=0,
                    key="notes_doc_selector"
                )
                
                if selected_doc:
                    st.session_state.selected_document = selected_doc
            else:
                st.warning("No documents available")
                return
        
        with col2:
            # Sort order selector
            st.session_state.sort_order = st.selectbox(
                "Sort by",
                ["Newest First", "Oldest First"],
                key="notes_sort_order"
            )
        
        # Display notes if document is selected
        if st.session_state.selected_document:
            # Fetch notes
            notes = fetch_document_notes(st.session_state.selected_document)
            
            if not notes:
                st.info(f"No research notes found for {st.session_state.selected_document}")
                return
            
            # Sort notes
            notes.sort(
                key=lambda x: datetime.fromisoformat(x["timestamp"].replace('Z', '+00:00')),
                reverse=(st.session_state.sort_order == "Newest First")
            )
            
            # Display total count
            st.markdown(f"### Found {len(notes)} notes")
            
            # Display notes
            for note in notes:
                render_note_card(note)
                
    except Exception as e:
        logger.error(f"Error in research notes page: {str(e)}")
        st.error("An error occurred while loading the research notes page")
        st.error(str(e))

if __name__ == "__main__":
    show()