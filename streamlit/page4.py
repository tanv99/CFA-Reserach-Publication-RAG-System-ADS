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
    """Render an image with consistent styling"""
    try:
        image_url = f"{FASTAPI_URL}{image_path}"
        response = requests.get(image_url)
        if response.status_code == 200:
            st.image(
                response.content,
                use_column_width=True,
                caption="Document Image",
                output_format="PNG"
            )
    except Exception as e:
        logger.error(f"Error rendering image: {str(e)}")
        st.warning("Failed to load image")

def render_note_card(note: Dict):
    """Render a note card for selected notes section"""
    try:
        with st.container():
            st.markdown(f"""
                <div style="padding: 1rem; background-color: #f8f9fa; border-radius: 4px; margin-bottom: 1rem;">
                    <div style="margin-bottom: 0.5rem;">
                        <strong>Note ID:</strong> {note.get('note_id', 'Unknown')}<br>
                        <strong>Created:</strong> {format_timestamp(note.get('timestamp', ''))}<br>
                        <strong>Query:</strong> {note.get('query', 'No query available')}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Note content in collapsed state by default
            with st.expander("View Note Content", expanded=False):
                if note.get('content'):
                    st.markdown(note['content'])
                else:
                    st.info("No text content available")
                
                if note.get('image_paths'):
                    for image_path in note['image_paths']:
                        if image_path.strip():
                            render_image(image_path)
    except Exception as e:
        logger.error(f"Error rendering note card: {str(e)}")
        st.error("Error displaying note")

def render_search_result(note: Dict):
    """Render a search result with different styles for exact and semantic matches"""
    try:
        match_type = note.get('match_type', 'exact')
        
        # Different styling for exact vs semantic matches
        if match_type == 'exact':
            border_color = "#28a745"  # Green for exact matches
            match_label = "Exact Match"
        else:
            border_color = "#17a2b8"  # Blue for semantic matches
            match_label = "Semantic Match"
        
        st.markdown(f"""
            <style>
            .search-result-{match_type} {{
                border-left: 4px solid {border_color};
                padding: 1rem;
                margin: 1rem 0;
                background-color: #f8f9fa;
                border-radius: 4px;
            }}
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="search-result-{match_type}">
                <div style="margin-bottom: 0.5rem;">
                    <strong>{match_label}</strong><br>
                    <strong>Your Question:</strong> {note.get('query', 'No query available')}
                    {f'<br><strong>Original Note Query:</strong> {note.get("original_query", "")}' if match_type == 'semantic' else ''}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Show search results in expanded state by default
        with st.expander("View Content", expanded=True):
            if note.get('content'):
                st.markdown(note['content'])
            else:
                st.info("No text content available")
            
            if note.get('image_paths'):
                st.markdown("### Supporting Images")
                for image_path in note['image_paths']:
                    if image_path.strip():
                        render_image(image_path)
        
        st.markdown("---")
                
    except Exception as e:
        logger.error(f"Error rendering search result: {str(e)}")
        st.error("Error displaying search result")

def sort_notes(notes: List[Dict], sort_order: str) -> List[Dict]:
    """Sort notes by timestamp"""
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
                return datetime.min
        
        return sorted(
            notes,
            key=get_sort_key,
            reverse=(sort_order == "Newest First")
        )
    except Exception as e:
        logger.error(f"Error sorting notes: {str(e)}")
        return notes

def submit_question():
    """Handle question submission and search through notes"""
    try:
        if not st.session_state.question.strip():
            st.warning("Please enter a question")
            return
            
        if not st.session_state.selected_document:
            st.warning("Please select a document first")
            return
            
        with st.spinner("Searching through notes..."):
            # Call the search endpoint
            response = requests.post(
                f"{FASTAPI_URL}/pdfs/{st.session_state.selected_document}/search-notes",
                json={
                    "query": st.session_state.question,
                    "top_k": 5,
                    "pdf_id": st.session_state.selected_document
                }
            )
            
            if response.status_code == 200:
                results = response.json()
                matches = results.get("matches", [])
                
                if matches:
                    st.success(f"Found {len(matches)} matching notes!")
                    
                    # Display matching notes
                    st.markdown("### Matching Notes")
                    for note in matches:
                        with st.expander(f"Note from {format_timestamp(note['timestamp'])}", expanded=True):
                            # Display original query
                            st.markdown(f"**Original Query:** {note['query']}")
                            
                            # Display content
                            if note.get('content'):
                                st.markdown(note['content'])
                            
                            # Display images if available
                            if note.get('image_paths'):
                                for image_path in note['image_paths']:
                                    render_image(image_path)
                else:
                    st.info("No matching notes found for your question.")
            else:
                st.error("Failed to search notes. Please try again.")
                
    except Exception as e:
        logger.error(f"Error in submit_question: {str(e)}")
        st.error("An error occurred while processing your question")


def normalize_query(query: str) -> str:
    """Normalize query string to match server-side normalization"""
    return ' '.join(query.split())

def show():
    """Main function to display the research notes page with enhanced search functionality"""
    try:
        st.title("Research Notes")
        
        # Initialize page state
        if 'selected_document' not in st.session_state:
            st.session_state.selected_document = None
        if 'selected_note' not in st.session_state:
            st.session_state.selected_note = None
        if 'question' not in st.session_state:
            st.session_state.question = ""
        
        # Create three columns for document selection, note selection, and sort order
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
                    st.session_state.selected_note = None
            else:
                st.warning("No documents available")
                return
        
        # Only show content if a document is selected
        if st.session_state.selected_document:
            # Fetch notes for the selected document
            notes = fetch_document_notes(st.session_state.selected_document)
            
            if notes:
                with col2:
                    # Note selector dropdown
                    note_options = ["Select a note", "ALL"] + [
                        f"Note {note.get('note_id', idx)}"
                        for idx, note in enumerate(notes, 1)
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
                    sort_order = st.selectbox(
                        "Sort by",
                        ["Newest First", "Oldest First"],
                        key="notes_sort_order"
                    )
                
                # Display selected notes section
                if st.session_state.selected_note:
                    st.markdown("### Selected Research Notes")
                    sorted_notes = sort_notes(notes, sort_order)
                    
                    if st.session_state.selected_note == "ALL":
                        for note in sorted_notes:
                            render_note_card(note)
                    else:
                        note_id = st.session_state.selected_note.split("Note ")[-1]
                        selected_note_data = next(
                            (note for note in sorted_notes if str(note.get('note_id')) == note_id),
                            None
                        )
                        if selected_note_data:
                            render_note_card(selected_note_data)
                
                # Add separator before search section
                st.markdown("---")
                
                # Search section
                st.markdown("### Search Notes")
                search_col1, search_col2 = st.columns([4, 1])
                
                with search_col1:
                    question = st.text_input(
                        "Search by question:",
                        key="question",
                        placeholder="Enter your question to find matching research notes..."
                    )
                
                with search_col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    search_clicked = st.button("Search", type="primary", use_container_width=True)
                
                # Handle search
                if search_clicked and question.strip():
                    st.markdown("### Search Results")
                    with st.spinner("Searching for relevant information..."):
                        try:
                            response = requests.post(
                                f"{FASTAPI_URL}/pdfs/{st.session_state.selected_document}/search-notes",
                                json={
                                    "query": question,
                                    "top_k": 5,
                                    "pdf_id": st.session_state.selected_document
                                }
                            )
                            
                            if response.status_code == 200:
                                results = response.json()
                                matches = results.get("matches", [])
                                match_type = results.get("match_type", "none")
                                
                                if matches:
                                    if match_type == "exact":
                                        st.success("")
                                    else:
                                        st.success("")
                                        
                                    for match in matches:
                                        render_search_result(match)
                                else:
                                    st.info("No relevant information found in the research notes.")
                                    
                                    # Additional context for no matches
                                    if st.session_state.selected_note != "ALL":
                                        st.markdown("""
                                            **Note:** Currently searching only the selected note. 
                                            Select "ALL" from the dropdown to search across all notes.
                                        """)
                            else:
                                st.error(f"Failed to search notes. Status code: {response.status_code}")
                                
                        except Exception as e:
                            logger.error(f"Search error: {str(e)}")
                            st.error("An error occurred while processing your search.")
            else:
                st.info(f"No research notes found for {st.session_state.selected_document}")
                
    except Exception as e:
        logger.error(f"Error in research notes page: {str(e)}")
        st.error("An error occurred while loading the page")
        if os.getenv("DEBUG"):
            st.error(f"Debug info: {str(e)}")