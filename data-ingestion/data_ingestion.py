import snowflake.connector
import json
import os
from dotenv import load_dotenv
import logging
import sys
from typing import Dict, Any
import boto3

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('snowflake_loader.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)  # This will handle console output
    ]
)

def setup_snowflake_connection():
    """Set up connection to Snowflake."""
    try:
        conn = snowflake.connector.connect(
            user=os.environ['SNOWFLAKE_USER'],            
            password=os.environ['SNOWFLAKE_PASSWORD'],    
            account=os.environ['SNOWFLAKE_ACCOUNT'],      
            warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],  
            database=os.environ['SNOWFLAKE_DATABASE'],    
            schema=os.environ['SNOWFLAKE_SCHEMA'],        
            role=os.environ['SNOWFLAKE_ROLE']          
        )
        logging.info("Successfully connected to Snowflake")
        
        # Test the connection and role
        cursor = conn.cursor()
        cursor.execute("SELECT CURRENT_ROLE()")
        role = cursor.fetchone()[0]
        logging.info(f"Connected with role: {role}")
        cursor.close()
        
        return conn
    except Exception as e:
        logging.error(f"Error connecting to Snowflake: {e}")
        raise

def create_publications_table(conn):
    """Create the publications table if it doesn't exist."""
    try:
        cursor = conn.cursor()
        
        # First, ensure we're using the correct database and schema
        cursor.execute(f"USE DATABASE {os.environ['SNOWFLAKE_DATABASE']}")
        cursor.execute(f"USE SCHEMA {os.environ['SNOWFLAKE_SCHEMA']}")
        
        # Create the table with VARCHAR instead of NVARCHAR(MAX)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS CFA_PUBLICATIONS (
            ID NUMBER AUTOINCREMENT,
            TITLE VARCHAR(500),
            SUMMARY TEXT,
            IMAGE_URL VARCHAR(1000),
            PDF_URL VARCHAR(1000),
            S3_BUCKET VARCHAR(100),
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (ID)
        )
        """)
        logging.info("CFA_PUBLICATIONS table created or verified")
    except Exception as e:
        logging.error(f"Error creating table: {e}")
        raise
    finally:
        cursor.close()
        
def setup_s3_client():
    """Set up S3 client and return client and bucket name."""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region_name=os.environ['AWS_REGION']
        )
        bucket = os.environ['AWS_BUCKET_NAME']  # Changed from S3_BUCKET_NAME
        logging.info(f"Successfully set up S3 client for bucket: {bucket}")
        return s3_client, bucket
    except Exception as e:
        logging.error(f"Error setting up S3 client: {e}")
        raise

def process_metadata_file(bucket: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Process metadata and construct S3 URLs."""
    try:
        # Extract relevant fields from metadata
        publication_data = {
            'TITLE': metadata.get('title', ''),
            'SUMMARY': metadata.get('summary', ''),
            'IMAGE_URL': metadata.get('image_url', ''),
            'PDF_URL': metadata.get('pdf_url', ''),
            'S3_BUCKET': bucket
        }
        return publication_data
    except Exception as e:
        logging.error(f"Error processing metadata: {e}")
        raise

def insert_publication_data(conn, publication_data: Dict[str, Any]):
    """Insert publication data into Snowflake."""
    try:
        cursor = conn.cursor()
        
        # Insert data into the table
        cursor.execute("""
        INSERT INTO CFA_PUBLICATIONS (
            TITLE, SUMMARY, IMAGE_URL, PDF_URL, S3_BUCKET
        ) VALUES (
            %s, %s, %s, %s, %s
        )
        """, (
            publication_data['TITLE'],
            publication_data['SUMMARY'],
            publication_data['IMAGE_URL'],
            publication_data['PDF_URL'],
            publication_data['S3_BUCKET']
        ))
        
        logging.info(f"Inserted publication: {publication_data['TITLE']}")
    except Exception as e:
        logging.error(f"Error inserting publication data: {e}")
        raise
    finally:
        cursor.close()

def load_data_to_snowflake(s3_client, bucket: str):
    """Main function to load data from S3 to Snowflake."""
    conn = None
    try:
        # Set up Snowflake connection
        conn = setup_snowflake_connection()
        
        # Create table if it doesn't exist
        create_publications_table(conn)
        
        # List all objects in the bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        page_number = 1
        total_files_processed = 0
        
        logging.info(f"Starting to process files from bucket: {bucket}")
        
        for page in paginator.paginate(Bucket=bucket):
            logging.info(f"Processing page {page_number} of S3 bucket")
            file_count = 0
            
            for obj in page.get('Contents', []):
                if obj['Key'].endswith('metadata.json'):
                    logging.info(f"Processing file: {obj['Key']}")
                    try:
                        # Get metadata file content
                        response = s3_client.get_object(Bucket=bucket, Key=obj['Key'])
                        metadata = json.loads(response['Body'].read().decode('utf-8'))
                        
                        # Process metadata and construct S3 URLs
                        publication_data = process_metadata_file(bucket, metadata)
                        
                        # Insert data into Snowflake
                        insert_publication_data(conn, publication_data)
                        file_count += 1
                        total_files_processed += 1
                        
                        logging.info(f"Successfully processed file {obj['Key']}")
                        
                    except Exception as e:
                        logging.error(f"Error processing metadata file {obj['Key']}: {e}")
                        continue
            
            logging.info(f"Completed page {page_number}. Processed {file_count} metadata files on this page.")
            logging.info(f"Total files processed so far: {total_files_processed}")
            page_number += 1
        
        conn.commit()
        logging.info(f"Data loading completed successfully. Total files processed: {total_files_processed}")
        
    except Exception as e:
        logging.error(f"Error in load_data_to_snowflake: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
            logging.info("Snowflake connection closed")

def main():
    """Main execution function."""
    try:
        # Configure console output to handle Unicode
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
            
        # Load environment variables
        load_dotenv()
        logging.info("Environment variables loaded")
        
        # Set up S3 client
        s3, bucket = setup_s3_client()
        
        # Load data from S3 to Snowflake
        load_data_to_snowflake(s3, bucket)
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise
    finally:
        logging.info("Script execution completed")

if __name__ == "__main__":
    main()