import snowflake.connector
import json
import os
from dotenv import load_dotenv
import logging
import sys
from typing import Dict, Any, Optional, Tuple
import boto3
from botocore.exceptions import ClientError
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class SnowflakeConfig:
    user: str
    password: str
    account: str
    warehouse: str
    database: str
    schema: str
    role: str

@dataclass
class AWSConfig:
    access_key_id: str
    secret_access_key: str
    region: str
    bucket: str

class SnowflakeLoader:
    def __init__(self, snowflake_config: SnowflakeConfig, aws_config: AWSConfig):
        self.snowflake_config = snowflake_config
        self.aws_config = aws_config
        self.logger = self._setup_logging()
        
    @staticmethod
    def _setup_logging() -> logging.Logger:
        """Configure logging with proper encoding."""
        logger = logging.getLogger('SnowflakeLoader')
        logger.setLevel(logging.INFO)
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler('snowflake_loader.log', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    @contextmanager
    def snowflake_connection(self):
        """Context manager for Snowflake connection."""
        conn = None
        try:
            conn = snowflake.connector.connect(
                user=self.snowflake_config.user,
                password=self.snowflake_config.password,
                account=self.snowflake_config.account,
                warehouse=self.snowflake_config.warehouse,
                database=self.snowflake_config.database,
                schema=self.snowflake_config.schema,
                role=self.snowflake_config.role
            )
            self.logger.info("Successfully connected to Snowflake")
            
            # Test the connection and roles
            with conn.cursor() as cursor:
                cursor.execute("SELECT CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_SCHEMA()")
                role, database, schema = cursor.fetchone()
                self.logger.info(f"Connected with role: {role}, database: {database}, schema: {schema}")
            
            yield conn
        except Exception as e:
            self.logger.error(f"Error connecting to Snowflake: {e}")
            raise
        finally:
            if conn:
                conn.close()
                self.logger.info("Snowflake connection closed")

    def setup_s3_client(self) -> boto3.client:
        """Set up and return S3 client."""
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_config.access_key_id,
                aws_secret_access_key=self.aws_config.secret_access_key,
                region_name=self.aws_config.region
            )
            self.logger.info(f"Successfully set up S3 client for bucket: {self.aws_config.bucket}")
            return s3_client
        except Exception as e:
            self.logger.error(f"Error setting up S3 client: {e}")
            raise

    def create_publications_table(self, conn: snowflake.connector.SnowflakeConnection) -> None:
        """Create the publications table if it doesn't exist."""
        with conn.cursor() as cursor:
            try:
                # First, ensure we're using the correct database and schema
                cursor.execute(f"USE DATABASE {self.snowflake_config.database}")
                cursor.execute(f"USE SCHEMA {self.snowflake_config.schema}")
                
                # Create table with explicit schema reference
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {self.snowflake_config.database}.{self.snowflake_config.schema}.CFA_PUBLICATIONS (
                    ID NUMBER AUTOINCREMENT,
                    TITLE VARCHAR(500),
                    SUMMARY TEXT,
                    IMAGE_URL VARCHAR(1000),
                    PDF_URL VARCHAR(1000),
                    S3_BUCKET VARCHAR(100),
                    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                    UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                    PRIMARY KEY (ID)
                )
                """
                cursor.execute(create_table_sql)
                
                # Important: Commit the table creation
                conn.commit()
                
                # Verify table exists
                verify_sql = f"""
                SELECT COUNT(*)
                FROM {self.snowflake_config.database}.INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = '{self.snowflake_config.schema}'
                AND TABLE_NAME = 'CFA_PUBLICATIONS'
                """
                cursor.execute(verify_sql)
                if cursor.fetchone()[0] == 0:
                    raise Exception("Table was not created successfully")
                    
                self.logger.info("CFA_PUBLICATIONS table created or verified successfully")
                
            except Exception as e:
                self.logger.error(f"Error creating table: {e}")
                conn.rollback()
                raise

    @staticmethod
    def process_metadata(metadata: Dict[str, Any], bucket: str) -> Dict[str, Any]:
        """Process metadata and validate required fields."""
        required_fields = ['title', 'summary', 'image_url', 'pdf_url']
        missing_fields = [field for field in required_fields if not metadata.get(field)]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
            
        return {
            'TITLE': metadata['title'],
            'SUMMARY': metadata['summary'],
            'IMAGE_URL': metadata['image_url'],
            'PDF_URL': metadata['pdf_url'],
            'S3_BUCKET': bucket
        }

    def insert_publication_data(self, conn: snowflake.connector.SnowflakeConnection, 
                              publication_data: Dict[str, Any]) -> None:
        """Insert publication data with error handling and validation."""
        with conn.cursor() as cursor:
            try:
                # Use fully qualified table name
                insert_sql = f"""
                INSERT INTO {self.snowflake_config.database}.{self.snowflake_config.schema}.CFA_PUBLICATIONS (
                    TITLE, SUMMARY, IMAGE_URL, PDF_URL, S3_BUCKET
                ) VALUES (
                    %s, %s, %s, %s, %s
                )
                """
                cursor.execute(insert_sql, (
                    publication_data['TITLE'],
                    publication_data['SUMMARY'],
                    publication_data['IMAGE_URL'],
                    publication_data['PDF_URL'],
                    publication_data['S3_BUCKET']
                ))
                self.logger.info(f"Inserted publication: {publication_data['TITLE']}")
            except Exception as e:
                self.logger.error(f"Error inserting publication data: {e}")
                raise

    def load_data(self) -> None:
        """Main data loading process with improved error handling and batching."""
        s3_client = self.setup_s3_client()
        batch_size = 100
        current_batch = []

        with self.snowflake_connection() as conn:
            self.create_publications_table(conn)
            
            try:
                paginator = s3_client.get_paginator('list_objects_v2')
                total_processed = 0
                
                for page in paginator.paginate(Bucket=self.aws_config.bucket):
                    for obj in page.get('Contents', []):
                        if not obj['Key'].endswith('metadata.json'):
                            continue
                            
                        try:
                            self.logger.info(f"Processing file: {obj['Key']}")
                            response = s3_client.get_object(
                                Bucket=self.aws_config.bucket, 
                                Key=obj['Key']
                            )
                            metadata = json.loads(response['Body'].read().decode('utf-8'))
                            publication_data = self.process_metadata(
                                metadata, 
                                self.aws_config.bucket
                            )
                            current_batch.append(publication_data)
                            
                            if len(current_batch) >= batch_size:
                                self._process_batch(conn, current_batch)
                                current_batch = []
                                
                            total_processed += 1
                            
                        except ClientError as e:
                            self.logger.error(f"S3 error processing {obj['Key']}: {e}")
                            continue
                        except json.JSONDecodeError as e:
                            self.logger.error(f"JSON parsing error in {obj['Key']}: {e}")
                            continue
                        except Exception as e:
                            self.logger.error(f"Unexpected error processing {obj['Key']}: {e}")
                            continue
                
                # Process any remaining items in the final batch
                if current_batch:
                    self._process_batch(conn, current_batch)
                
                self.logger.info(f"Data loading completed. Total files processed: {total_processed}")
                
            except Exception as e:
                self.logger.error(f"Error in load_data: {e}")
                conn.rollback()
                raise

    def _process_batch(self, conn: snowflake.connector.SnowflakeConnection, 
                      batch: list[Dict[str, Any]]) -> None:
        """Process a batch of publications."""
        try:
            for publication_data in batch:
                self.insert_publication_data(conn, publication_data)
            conn.commit()
            self.logger.info(f"Successfully processed batch of {len(batch)} items")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error processing batch: {e}")
            raise

def main():
    """Main execution function with configuration management."""
    try:
        # Ensure UTF-8 encoding
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
            
        load_dotenv()
        
        # Configure Snowflake
        snowflake_config = SnowflakeConfig(
            user=os.environ['SNOWFLAKE_USER'],
            password=os.environ['SNOWFLAKE_PASSWORD'],
            account=os.environ['SNOWFLAKE_ACCOUNT'],
            warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
            database=os.environ['SNOWFLAKE_DATABASE'],
            schema=os.environ['SNOWFLAKE_SCHEMA'],
            role=os.environ['SNOWFLAKE_ROLE']
        )
        
        # Configure AWS
        aws_config = AWSConfig(
            access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
            secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
            region=os.environ['AWS_REGION'],
            bucket=os.environ['AWS_BUCKET_NAME']
        )
        
        # Initialize and run loader
        loader = SnowflakeLoader(snowflake_config, aws_config)
        loader.load_data()
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()