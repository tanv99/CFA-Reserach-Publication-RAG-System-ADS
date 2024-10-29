from typing import List, Dict, Any
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, PodSpec
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        """Initialize the text processor with necessary clients and configurations"""
        # Initialize NVIDIA embeddings client
        self.embeddings_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv('NVIDIA_API_KEY')
        )
        
        # Initialize Pinecone
        self.pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY'),
            environment="gcp-starter"
        )
        
        # Get or create Pinecone index
        self.index_name = "pdf-embeddings"
        self.create_pinecone_index()
        
        # Get Pinecone index
        self.index = self.pc.Index(self.index_name)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def create_pinecone_index(self) -> None:
        """Create Pinecone index if it doesn't exist"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            if self.index_name not in existing_indexes:
                # Create the index for gcp-starter environment
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1024,  # E5 model dimension
                    metric='cosine',
                    spec=PodSpec(
                        environment="gcp-starter",
                        pod_type="starter"
                    )
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error creating/checking Pinecone index: {str(e)}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            raise

    def create_embedding(self, text: str, input_type: str = 'passage') -> List[float]:
        """Create embedding for a single text chunk"""
        try:
            response = self.embeddings_client.embeddings.create(
                input=[text],
                model="nvidia/nv-embedqa-e5-v5",
                encoding_format="float",
                extra_body={
                    "input_type": input_type,
                    "truncate": "NONE"
                }
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            raise

    def process_and_store(self, text: str, metadata: Dict[str, Any]) -> None:
        """Process text, create embeddings, and store in Pinecone"""
        try:
            # Split text into chunks
            chunks = self.chunk_text(text)
            
            # Process chunks in batches
            batch_size = 50
            for i in tqdm(range(0, len(chunks), batch_size)):
                batch = chunks[i:i + batch_size]
                
                # Create embeddings for batch
                vectors = []
                for idx, chunk in enumerate(batch):
                    # Create embedding
                    embedding = self.create_embedding(chunk, input_type='passage')
                    
                    # Prepare metadata
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": i + idx,
                        "text": chunk
                    }
                    
                    # Prepare vector
                    vectors.append((
                        f"{metadata.get('pdf_id', 'unknown')}_{i + idx}",
                        embedding,
                        chunk_metadata
                    ))
                
                # Upsert batch to Pinecone
                self.index.upsert(vectors=vectors)
                
                # Rate limiting
                time.sleep(0.5)
            
            logger.info(f"Successfully processed and stored {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error in process_and_store: {str(e)}")
            raise

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar text chunks"""
        try:
            # Create query embedding
            query_embedding = self.create_embedding(query, input_type='query')
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return results
        except Exception as e:
            logger.error(f"Error in search_similar: {str(e)}")
            raise

# Export the class
__all__ = ['TextProcessor']