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
    if 'selected_note' not in st.session_state:
        st.session_state.selected_note = None
    if 'current_notes' not in st.session_state:
        st.session_state.current_notes = []
    if 'sort_order' not in st.session_state:
        st.session_state.sort_order = "Newest First"
    if 'question' not in st.session_state:
        st.session_state.question = ""

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
    """Format timestamp to readable date/time with fallback"""
    if not timestamp:
        return "No date available"
    
    try:
        # Try parsing as ISO format
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%B %d, %Y %I:%M %p")
    except ValueError:
        try:
            # Try parsing as Unix timestamp if it's a number
            if timestamp.isdigit():
                dt = datetime.fromtimestamp(int(timestamp))
                return dt.strftime("%B %d, %Y %I:%M %p")
        except:
            pass
        
        # If all parsing fails, return the original string
        return str(timestamp)

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
            
            # Note header with note_id
            st.markdown(f"""
                <div class="note-card">
                    <div class="note-metadata">
                        <strong>Note ID: {note.get('note_id', 'Unknown')}</strong><br>
                        📝 Created: {format_timestamp(note.get('timestamp', ''))}<br>
                        🔍 Query: "{note.get('query', 'No query available')}"
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
                    for image_path in note['image_paths']:
                        render_image(image_path)
            
            st.markdown("---")
    except Exception as e:
        logger.error(f"Error rendering note card: {str(e)}")
        st.error("Error displaying note")

def sort_notes(notes: List[Dict], sort_order: str) -> List[Dict]:
    """Sort notes by timestamp with error handling"""
    try:
        def get_sort_key(note):
            timestamp = note.get('timestamp', '')
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                try:
                    if str(timestamp).isdigit():
                        return datetime.fromtimestamp(int(timestamp))
                except:
                    pass
                return datetime.min  # Fallback for invalid dates
        
        return sorted(
            notes,
            key=get_sort_key,
            reverse=(sort_order == "Newest First")
        )
    except Exception as e:
        logger.error(f"Error sorting notes: {str(e)}")
        return notes

def show():
    """Main function to display the research notes page"""
    try:
        st.title("Research Notes")
        
        # Initialize page state
        init_page_state()
        
        # Add description
        st.markdown("""
            View and manage your saved research notes for each document.
            Select a document and a specific research note to view its content.
        """)
        
        # Create three columns for document selection, note selection, and filters
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Document selector
            pdfs = fetch_pdfs()
            if pdfs:
                options = ["Select a document"] + [pdf['title'] for pdf in pdfs]
                selected_doc = st.selectbox(
                    "Select a document",
                    options=options,
                    index=0,
                    key="notes_doc_selector"
                )
                
                if selected_doc != "Select a document":
                    st.session_state.selected_document = selected_doc
                else:
                    st.session_state.selected_document = None
            else:
                st.warning("No documents available")
                return
        
        # Only show note selection and content if a document is selected
        if st.session_state.selected_document:
            notes = fetch_document_notes(st.session_state.selected_document)
            
            if notes:
                with col2:
                    # Create note options using note_ids
                    note_options = ["Select a note", "ALL"] + [
                        note.get('note_id', f"note_{st.session_state.selected_document}_{idx}")
                        for idx, note in enumerate(notes)
                    ]
                    selected_note = st.selectbox(
                        "Select research note",
                        options=note_options,
                        key="note_selector"
                    )
                    if selected_note != "Select a note":
                        st.session_state.selected_note = selected_note
                    else:
                        st.session_state.selected_note = None
                
                with col3:
                    # Sort order selector
                    st.session_state.sort_order = st.selectbox(
                        "Sort by",
                        ["Newest First", "Oldest First"],
                        key="notes_sort_order"
                    )
                
                # Only display notes if both document and note are selected
                if st.session_state.selected_note:
                    # Sort notes with error handling
                    sorted_notes = sort_notes(notes, st.session_state.sort_order)
                    
                    if st.session_state.selected_note == "ALL":
                        for note in sorted_notes:
                            render_note_card(note)
                    else:
                        # Find the specific note by note_id
                        selected_note_data = next(
                            (note for note in sorted_notes if note.get('note_id') == st.session_state.selected_note),
                            None
                        )
                        if selected_note_data:
                            render_note_card(selected_note_data)
                        else:
                            st.error("Selected note not found")
                    
                    # Question input section
                    st.markdown("### Ask a Question")
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        question = st.text_input(
                            "Enter your question about this document:",
                            key="question",
                            placeholder="Type your question here..."
                        )
                    with col2:
                        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align with text input
                        if st.button("Submit Question"):
                            submit_question()
                
            else:
                st.info(f"No research notes found for {st.session_state.selected_document}")
                
    except Exception as e:
        logger.error(f"Error in research notes page: {str(e)}")
        st.error("An error occurred while loading the research notes page")
        st.error(str(e))

if __name__ == "__main__":
    show()