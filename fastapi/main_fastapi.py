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


# Load environment variables
load_dotenv()

env_path = pathlib.Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

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