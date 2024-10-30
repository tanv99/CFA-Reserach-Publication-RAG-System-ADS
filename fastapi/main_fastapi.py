from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.responses import StreamingResponse, JSONResponse
import boto3
import json
from botocore.exceptions import ClientError
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, constr, validator
from typing import List, Dict, Optional
import io
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta
import logging
import bcrypt
import pymysql
from google.cloud import storage
from google.oauth2 import service_account
import openai
import os, pathlib
from pathlib import Path
from pdf_processor import PDFProcessor
import requests
from openai import OpenAI
from typing import Any
 
 
# Load environment variables
load_dotenv()
 
env_path = pathlib.Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
 
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
app = FastAPI()
 
pdf_processor = PDFProcessor()
 
# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
 
# OpenAI settings
openai.api_key = os.getenv("OPENAI_API_KEY")
 
# GCP bucket settings
TXT_BUCKET_NAME = os.getenv("TXT_BUCKET_NAME")
 
# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
 
# Database connection
def load_sql_db_config():
    try:
        connection = pymysql.connect(
            user=os.getenv("GCP_SQL_USER"),
            password=os.getenv("GCP_SQL_PASSWORD"),
            host=os.getenv("GCP_SQL_HOST"),
            database=os.getenv("GCP_SQL_DATABASE"),
            cursorclass=pymysql.cursors.DictCursor
        )
        return connection
    except pymysql.Error as e:
        logger.error(f"Error connecting to Cloud SQL: {e}")
        return None
 
# Google Cloud Storage client setup
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if credentials_path:
    credentials = service_account.Credentials.from_service_account_file(credentials_path)
    storage_client = storage.Client(credentials=credentials)
else:
    storage_client = storage.Client()
 
# Models
class UserRegister(BaseModel):
    email: EmailStr
    password: constr(min_length=8)
 
    @validator('password')
    def validate_password(cls, value):
        if len(value) < 8:
            raise ValueError('Password should be at least 8 characters long')
        if not any(char.islower() for char in value):
            raise ValueError('Password should contain at least one lowercase letter')
        if not any(char.isupper() for char in value):
            raise ValueError('Password should contain at least one uppercase letter')
        return value
 
class UserLogin(BaseModel):
    email: EmailStr
    password: str
 
 
 
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
 
def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except ValueError:
        logger.error(f"Invalid password hash encountered")
        return False
 
def create_jwt_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
 
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return email
 
 
    connection = load_sql_db_config()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with connection.cursor() as cursor:
            check_user_sql = "SELECT * FROM users WHERE email = %s"
            cursor.execute(check_user_sql, (user.email,))
            existing_user = cursor.fetchone()
            if existing_user:
                raise HTTPException(status_code=400, detail="Email already registered")
 
            hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
 
            sql = "INSERT INTO users (email, password) VALUES (%s, %s)"
            cursor.execute(sql, (user.email, hashed_password))
        connection.commit()
        return {"message": "User registered successfully"}
 
    except pymysql.Error as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")
 
    finally:
        connection.close()
 
@app.post("/register")
def register_user(user: UserRegister):
    connection = load_sql_db_config()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with connection.cursor() as cursor:
            check_user_sql = "SELECT * FROM users WHERE email = %s"
            cursor.execute(check_user_sql, (user.email,))
            existing_user = cursor.fetchone()
            if existing_user:
                raise HTTPException(status_code=400, detail="Email already registered")
 
            # Check if password meets the minimum length requirement
            if len(user.password) < 8:
                raise HTTPException(status_code=400, detail="Password must be at least 8 characters long and contain at least one uppercase letter and one lowercase letter.")
 
            hashed_password = hash_password(user.password)
 
            sql = "INSERT INTO users (email, password) VALUES (%s, %s)"
            cursor.execute(sql, (user.email, hashed_password))
        connection.commit()
        return {"message": "User registered successfully"}
 
    except pymysql.Error as e:
        raise HTTPException(status_code=400, detail=f"Registration failed: {str(e)}")
 
    finally:
        connection.close()
 
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    connection = load_sql_db_config()
    if not connection:
        logger.error("Database connection failed")
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM users WHERE email = %s"
            cursor.execute(sql, (form_data.username,))
            user = cursor.fetchone()
            if not user:
                logger.warning(f"Login attempt failed: User not found - {form_data.username}")
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect email. Please check your login credentials and try again.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
           
            # Verify the password
            if not verify_password(form_data.password, user['password']):
                logger.warning(f"Login attempt failed: Incorrect password - {form_data.username}")
                raise HTTPException(
                    status_code=401,
                    detail="Incorrect password. Please check your login credentials and try again.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
           
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_jwt_token(
                data={"sub": user['email']}, expires_delta=access_token_expires
            )
            logger.info(f"Login successful: {form_data.username}")
            return {"access_token": access_token, "token_type": "bearer"}
    except pymysql.Error as e:
        logger.error(f"Database error during login: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during login: {str(e)}")
        raise HTTPException(status_code=500, detail="Please check your login credentials and try again")
    finally:
        connection.close()
@app.get("/users/me")
async def read_users_me(current_user: str = Depends(get_current_user)):
    return {"email": current_user}
 
@app.get("/test-db-connection")
async def test_db_connection():
    connection = load_sql_db_config()
    if not connection:
        raise HTTPException(status_code=500, detail="Failed to establish database connection")
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
        return {"message": "Database connection successful", "test_query_result": result}
    except pymysql.Error as e:
        logger.error(f"Database connection test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection test failed: {str(e)}")
    finally:
        connection.close()
 
 
###########################AWS  streamlit show documents ############
class PDFMetadata(BaseModel):
    title: str
    metadata: Dict
    summary: Optional[str]
 
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
 
def get_s3_client():
    """Create and return an S3 client"""
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
 
@app.get("/pdfs/all", response_model=List[PDFMetadata])
async def get_all_pdfs():
    """Get all PDFs with their metadata and cover image URLs"""
    try:
        s3_client = get_s3_client()
        folders = set()
       
        # List all objects in the bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=BUCKET_NAME):
            if 'Contents' in page:
                for obj in page['Contents']:
                    parts = obj['Key'].split('/')
                    if len(parts) > 1:
                        folders.add(parts[0])
       
        # Get metadata and cover URLs for each PDF
        pdfs = []
        for folder in sorted(folders):
            try:
                # Get metadata
                metadata_key = f"{folder}/metadata.json"
                metadata_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=metadata_key)
                metadata = json.loads(metadata_obj['Body'].read().decode('utf-8'))
            except ClientError:
                metadata = {}
           
            try:
                # Get summary
                summary_key = f"{folder}/summary.txt"
                summary_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=summary_key)
                summary = summary_obj['Body'].read().decode('utf-8')
            except ClientError:
                summary = "No summary available"
           
            # Create cover image URL
            cover_url = f"/pdfs/{folder}/cover"
           
            pdfs.append(PDFMetadata(
                title=folder,
                metadata=metadata,
                summary=summary,
                cover_url=cover_url
            ))
       
        return pdfs
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))
 
@app.get("/pdfs/{folder_name}/document")
async def get_pdf_document(folder_name: str):
    """Stream the PDF document"""
    try:
        s3_client = get_s3_client()
        pdf_key = f"{folder_name}/document.pdf"
       
        try:
            pdf_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=pdf_key)
            pdf_content = pdf_obj['Body'].read()
           
            return StreamingResponse(
                io.BytesIO(pdf_content),
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'inline; filename="{folder_name}.pdf"'
                }
            )
        except ClientError:
            raise HTTPException(status_code=404, detail="PDF not found")
           
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))
 
@app.get("/pdfs/{folder_name}/cover")
async def get_cover_image(folder_name: str):
    """Stream the cover image"""
    try:
        s3_client = get_s3_client()
        image_key = f"{folder_name}/image.jpg"
       
        try:
            image_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=image_key)
            image_content = image_obj['Body'].read()
           
            return StreamingResponse(
                io.BytesIO(image_content),
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": f'inline; filename="{folder_name}_cover.jpg"'
                }
            )
        except ClientError:
            raise HTTPException(status_code=404, detail="Cover image not found")
           
    except ClientError as e:
        raise HTTPException(status_code=500, detail=str(e))
   
#################################################generte summay################
# NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
# NVIDIA_API_URL = os.getenv("NVIDIA_API_URL")
# Initialize OpenAI client with NVIDIA endpoint
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv('NVIDIA_API_KEY')
)
 
@app.get("/pdfs/{folder_name}/process")
async def process_pdf_content(folder_name: str):
    try:
        # Get PDF from S3
        s3_client = get_s3_client()
        try:
            response = s3_client.get_object(
                Bucket=os.getenv("BUCKET_NAME"),
                Key=f"{folder_name}/document.pdf"
            )
            pdf_content = response['Body'].read()
        except Exception as e:
            logger.error(f"Error reading PDF from S3: {str(e)}")
            raise HTTPException(status_code=404, detail="PDF not found")
       
        # Process PDF and get extracted text
        extracted_text = pdf_processor.process_pdf(pdf_content)
       
        # Truncate text if too long (add this section)
        max_chars = 25000  # Adjust this based on testing
        full_text = extracted_text
        if len(extracted_text) > max_chars:
            extracted_text = extracted_text[:max_chars] + "\n\n[Text truncated due to length...]"
       
        # Prepare prompt for summary (your existing prompt)
        prompt = f"""Please analyze this document and provide a structured summary following this exact format:
 
Document text:
{extracted_text}
 
Please structure your response exactly as follows:
 
Key Points:
- [First key point]
- [Second key point]
- [Third key point]
(provide 3-5 key points)
 
Main Topics:
- [First main topic]
- [Second main topic]
- [Third main topic]
(provide 2-3 main topics)
 
Summary:
[Provide a 2-3 paragraph summary here]
 
Important: Please maintain this exact structure and format in your response, including the headers 'Key Points:', 'Main Topics:', and 'Summary:'."""
 
        try:
            # Generate summary using NVIDIA's API
            completion = client.chat.completions.create(
                model="mistralai/mixtral-8x7b-instruct-v0.1",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
           
            summary_text = completion.choices[0].message.content
           
            # Process the response into structured format
            sections = summary_text.split('\n\n')
            key_points = []
            main_topics = []
            detailed_summary = ""
           
            for section in sections:
                section = section.strip()
                if "key points:" in section.lower():
                    points = [p.strip('- ').strip() for p in section.split('\n')[1:]]
                    key_points = [p for p in points if p]
                elif "main topics:" in section.lower():
                    topics = [t.strip('- ').strip() for t in section.split('\n')[1:]]
                    main_topics = [t for t in topics if t]
                elif "summary:" in section.lower():
                    detailed_summary = section.split('Summary:', 1)[-1].strip()
           
            structured_summary = {
                "key_points": key_points,
                "main_topics": main_topics,
                "summary": detailed_summary or summary_text
            }
           
            return {
                "extracted_text": full_text,  # Return full text but use truncated for processing
                "summary": structured_summary
            }
           
        except Exception as e:
            logger.error(f"Error with NVIDIA API: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")
           
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
 
# Test endpoint
@app.get("/test-llama")
async def test_llama():
    """Test NVIDIA Llama API connection"""
    try:
        completion = client.chat.completions.create(
            model="mistralai/mixtral-8x7b-instruct-v0.1",
            messages=[
                {"role": "user", "content": "Say hello and confirm you're working!"}
            ],
            temperature=0.5,
            max_tokens=50
        )
       
        return {
            "status": "success",
            "response": completion.choices[0].message.content,
            "model": "mistralai/mixtral-8x7b-instruct-v0.1"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "model": "mistralai/mixtral-8x7b-instruct-v0.1"
        }
 
############################ Page3 view ppdf######################
import nest_asyncio
nest_asyncio.apply()
 
from fastapi import FastAPI, HTTPException
from llama_parse import LlamaParse
from typing import List, Dict, Any
import os
from pathlib import Path
import base64
import json
import logging
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from PIL import Image
import io
from dotenv import load_dotenv
from llama_index.core.schema import TextNode
import re
# Load environment variables
load_dotenv()
 
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
 
# Initialize environment variables
LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")
 
# Define base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
EXTRACTION_DIR = os.path.join(STORAGE_DIR, "extractions")
IMAGES_DIR = os.path.join(STORAGE_DIR, "images")
 
# Create directories if they don't exist
os.makedirs(EXTRACTION_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
 
def get_s3_client():
    """Create and return an S3 client"""
    try:
        client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create S3 client: {str(e)}")
        raise
 
def get_page_number(file_name: str) -> int:
    """Extract page number from image filename."""
    match = re.search(r"-page-(\d+)\.jpg$", str(file_name))
    if match:
        return int(match.group(1))
    return 0

def _get_sorted_image_files(image_dir: str) -> List[Path]:
    """Get image files sorted by page number."""
    raw_files = [f for f in list(Path(image_dir).iterdir()) if f.is_file()]
    sorted_files = sorted(raw_files, key=get_page_number)
    return sorted_files

class PDFProcessor_llama:
    def __init__(self, api_key: str):
        """Initialize LlamaParse with enhanced multimodal configuration"""
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="anthropic-sonnet-3.5"
        )
    
    def get_text_nodes(self, json_dicts: List[Dict], image_dir: Optional[str] = None) -> List[TextNode]:
        """Create text nodes from parsed PDF content with image metadata."""
        nodes = []
        
        image_files = _get_sorted_image_files(image_dir) if image_dir is not None else None
        md_texts = [d.get("md", "") for d in json_dicts]
        
        for idx, md_text in enumerate(md_texts):
            chunk_metadata = {"page_num": idx + 1}
            if image_files is not None and idx < len(image_files):
                image_file = image_files[idx]
                chunk_metadata["image_path"] = str(image_file)
            chunk_metadata["parsed_text_markdown"] = md_text
            
            node = TextNode(
                text="",
                metadata=chunk_metadata,
            )
            nodes.append(node)
        
        return nodes
    
    def process_pdf(self, pdf_path: str, folder_name: str) -> Dict[str, Any]:
        """Process PDF and extract content using LlamaParse with indexing"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # First get the markdown content
            logger.info("Getting JSON result...")
            md_json_objs = self.parser.get_json_result(pdf_path)
            
            # Check if we got valid results
            if not md_json_objs or not isinstance(md_json_objs, list):
                raise ValueError("Invalid JSON result from parser")
            
            md_json_list = md_json_objs[0]["pages"]
            logger.info(f"Successfully processed {len(md_json_list)} pages")
            
            # Create image directory with absolute path
            image_dir = os.path.abspath(os.path.join(IMAGES_DIR, folder_name))
            os.makedirs(image_dir, exist_ok=True)
            logger.info(f"Created image directory at: {image_dir}")
            
            # Extract images with explicit path
            logger.info("Extracting images...")
            image_dicts = self.parser.get_images(
                md_json_objs,
                image_dir
            )
            logger.info(f"Found {len(image_dicts) if image_dicts else 0} images")
            
            # Create text nodes with image metadata
            text_nodes = self.get_text_nodes(md_json_list, image_dir)
            logger.info(f"Created {len(text_nodes)} text nodes")
            
            # Structure the content
            extracted_content = {
                "pages": [],
                "images": [],
                "nodes": [],
                "metadata": {
                    "total_pages": len(md_json_list),
                    "file_name": os.path.basename(pdf_path),
                    "extraction_time": datetime.now().isoformat()
                }
            }
            
            # Process each page's markdown content
            for page in md_json_list:
                extracted_content["pages"].append({
                    "page_num": page.get("page", 0),
                    "content": page.get("md", ""),
                    "has_images": bool(page.get("images", []))
                })
            
            # Process image information if any images were found
            if image_dicts:
                for idx, img in enumerate(image_dicts):
                    try:
                        image_data = {
                            "file_name": f"image_{idx}.png" if not img.get("file_path") else os.path.basename(img["file_path"]),
                            "local_path": os.path.relpath(img.get("file_path", ""), BASE_DIR) if img.get("file_path") else "",
                            "page_number": img.get("page_number", 0),
                            "caption": img.get("caption", "")
                        }
                        extracted_content["images"].append(image_data)
                    except Exception as e:
                        logger.error(f"Error processing image {idx}: {str(e)}")
            
            # Add text nodes information
            for node in text_nodes:
                extracted_content["nodes"].append({
                    "page_num": node.metadata.get("page_num"),
                    "image_path": node.metadata.get("image_path"),
                    "content": node.metadata.get("parsed_text_markdown")
                })
            
            # Save the extracted content
            output_dir = os.path.join(EXTRACTION_DIR, folder_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save markdown content
            markdown_path = os.path.join(output_dir, "extracted_content.md")
            markdown_content = "\n\n".join(page["content"] for page in extracted_content["pages"])
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            # Save nodes data
            nodes_path = os.path.join(output_dir, "text_nodes.json")
            with open(nodes_path, "w", encoding="utf-8") as f:
                json.dump([{
                    "page_num": node.metadata.get("page_num"),
                    "image_path": node.metadata.get("image_path"),
                    "content": node.metadata.get("parsed_text_markdown")
                } for node in text_nodes], f, indent=2)
            
            return extracted_content
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
        
        
# Add this test endpoint to verify paths and processing
@app.post("/pdfs/{folder_name}/test-extract")
async def test_extraction(folder_name: str):
    """Test extraction process with detailed logging"""
    try:
        # Log environment setup
        logger.info("Testing extraction process...")
        logger.info(f"Base directory: {BASE_DIR}")
        logger.info(f"Storage directory: {STORAGE_DIR}")
        logger.info(f"Images directory: {IMAGES_DIR}")
        
        # Verify API key
        if not LLAMAPARSE_API_KEY:
            return {
                "status": "error",
                "detail": "LLAMAPARSE_API_KEY not configured"
            }
        logger.info("API key found")
        
        # Get PDF from S3
        s3_client = get_s3_client()
        pdf_key = f"{folder_name}/document.pdf"
        
        try:
            # Get PDF content
            pdf_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=pdf_key)
            pdf_content = pdf_obj['Body'].read()
            logger.info(f"Retrieved PDF from S3: {len(pdf_content)} bytes")
            
            # Create folder for extraction
            temp_dir = os.path.join(EXTRACTION_DIR, folder_name)
            os.makedirs(temp_dir, exist_ok=True)
            
            # Save PDF temporarily
            temp_path = os.path.join(temp_dir, "document.pdf")
            with open(temp_path, "wb") as f:
                f.write(pdf_content)
            logger.info(f"Saved PDF to: {temp_path}")
            
            # Initialize processor
            processor = PDFProcessor_llama(api_key=LLAMAPARSE_API_KEY)
            
            # Process content
            extracted_content = processor.process_pdf(temp_path, folder_name)
            
            # Save extracted content as markdown
            markdown_path = os.path.join(temp_dir, "extracted_content.md")
            markdown_content = ""
            
            # Convert extracted content to markdown format
            for page in extracted_content["pages"]:
                markdown_content += f"\n\n## Page {page['page_num']}\n\n"
                markdown_content += page["content"]
            
            # Save markdown content
            with open(markdown_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            logger.info(f"Saved markdown content to: {markdown_path}")
            
            # Save image index if there are images
            if extracted_content["images"]:
                image_index_path = os.path.join(temp_dir, "image_index.json")
                with open(image_index_path, "w") as f:
                    json.dump(extracted_content["images"], f, indent=2)
                logger.info(f"Saved image index to: {image_index_path}")
            
            # Clean up temporary PDF
            os.remove(temp_path)
            
            return {
                "status": "success",
                "paths": {
                    "base_dir": BASE_DIR,
                    "storage_dir": STORAGE_DIR,
                    "images_dir": IMAGES_DIR,
                    "markdown_path": markdown_path,
                    "image_index_path": image_index_path if extracted_content["images"] else None
                },
                "content": extracted_content
            }
            
        except ClientError as e:
            return {
                "status": "error",
                "detail": str(e),
                "error_code": e.response['Error']['Code'] if hasattr(e, 'response') else None
            }
            
    except Exception as e:
        logger.error(f"Error in test extraction: {str(e)}")
        return {
            "status": "error",
            "detail": str(e)
        }


@app.post("/pdfs/{folder_name}/get-nodes")
async def get_pdf_nodes(folder_name: str):
    """Get text nodes for a processed PDF"""
    try:
        nodes_path = os.path.join(EXTRACTION_DIR, folder_name, "text_nodes.json")
        
        if not os.path.exists(nodes_path):
            raise HTTPException(
                status_code=404,
                detail="Nodes data not found. Please process the PDF first."
            )
        
        with open(nodes_path, "r", encoding="utf-8") as f:
            nodes_data = json.load(f)
        
        return {
            "status": "success",
            "folder_name": folder_name,
            "total_nodes": len(nodes_data),
            "nodes": nodes_data
        }
        
    except Exception as e:
        logger.error(f"Error retrieving nodes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

######################Embedding#######################
from text_processor import TextProcessor  
text_processor = TextProcessor()
class SearchQuery(BaseModel):
    query: str
    top_k: Optional[int] = 5
 
@app.post("/pdfs/{folder_name}/process-embeddings")
async def process_pdf_embeddings(folder_name: str):
    """Process PDF content and store embeddings"""
    try:
        # Get markdown content from previous extraction using os.path.join
        markdown_path = os.path.join(EXTRACTION_DIR, folder_name, "extracted_content.md")
        
        if not os.path.exists(markdown_path):
            raise HTTPException(
                status_code=404,
                detail="Markdown content not found. Please process the PDF first."
            )
        
        # Read markdown content
        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Create metadata
        metadata = {
            "pdf_id": folder_name,
            "source": "pdf",
            "processed_date": datetime.now().isoformat()
        }
        
        # Process and store embeddings
        text_processor.process_and_store(content, metadata)
        
        return {
            "status": "success",
            "message": f"Successfully processed and stored embeddings for {folder_name}"
        }
        
    except Exception as e:
        logger.error(f"Error processing embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_image_info(folder_name: str):
    """Get information about extracted images"""
    try:
        image_index_path = os.path.join(EXTRACTION_DIR, folder_name, "image_index.json")
        
        if not os.path.exists(image_index_path):
            raise HTTPException(
                status_code=404,
                detail="Image index not found. Please process the PDF first."
            )
            
        with open(image_index_path, "r") as f:
            image_info = json.load(f)
            
        return {
            "folder_name": folder_name,
            "total_images": len(image_info),
            "images": image_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/pdfs/search")
async def search_pdfs(query: SearchQuery):
    """Search through PDF content using embeddings"""
    try:
        results = text_processor.search_similar(query.query, query.top_k)
       
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "score": match.score,
                "pdf_id": match.metadata.get("pdf_id"),
                "chunk_index": match.metadata.get("chunk_index"),
                "text": match.metadata.get("text")
            })
       
        return {
            "query": query.query,
            "results": formatted_results
        }
       
    except Exception as e:
        logger.error(f"Error searching PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pdfs/search")
async def search_pdfs(query: SearchQuery):
    """Search through PDF content using embeddings"""
    try:
        results = text_processor.search_similar(query.query, query.top_k)
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "score": match.score,
                "pdf_id": match.metadata.get("pdf_id"),
                "chunk_index": match.metadata.get("chunk_index"),
                "text": match.metadata.get("text")
            })
        
        return {
            "query": query.query,
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Error searching PDFs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class QuestionRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    max_tokens: Optional[int] = 1000

@app.post("/pdfs/qa")
async def question_answer(request: QuestionRequest):
    """Search through PDFs and generate an answer to the question"""
    try:
        result = await text_processor.search_and_answer(
            query=request.query,
            top_k=request.top_k
        )
        
        return {
            "status": "success",
            "query": request.query,
            "answer": result["answer"],
            "supporting_evidence": {
                "chunks": [
                    {
                        "text": chunk["metadata"]["text"],
                        "relevance_score": chunk["score"],
                        "source": {
                            "pdf_id": chunk["metadata"].get("pdf_id"),
                            "chunk_index": chunk["metadata"].get("chunk_index")
                        }
                    }
                    for chunk in result["supporting_chunks"]
                ],
                "total_chunks": result["total_chunks"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing QA request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    