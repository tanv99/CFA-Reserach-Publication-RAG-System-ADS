import os
import json
import subprocess
import shutil
import pymysql
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import storage
from PyPDF2 import PdfReader
from datetime import datetime
from airflow.utils.dates import days_ago
from env_var import GIT_USERNAME, GIT_TOKEN, GIT_REPO_URL, GCP_BUCKET_NAME, GCP_SERVICE_ACCOUNT_FILE, GCP_SQL_USER, GCP_SQL_PASSWORD, GCP_SQL_HOST, GCP_SQL_DATABASE

# New imports for enhanced PDF processing
import tabula

# Default arguments for the DAG
default_args = {
    'start_date': days_ago(0),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    dag_id='pypdf_processing_dag',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)

# Step 1: Clone the dataset from Hugging Face including LFS files
def clone_repository(**kwargs):
    LOCAL_CLONE_DIR = "./GAIA"
    git_url_with_credentials = GIT_REPO_URL.replace("https://", f"https://{GIT_USERNAME}:{GIT_TOKEN}@")

    if os.path.exists(LOCAL_CLONE_DIR):
        try:
            print(f"Directory {LOCAL_CLONE_DIR} exists. Deleting it...")
            shutil.rmtree(LOCAL_CLONE_DIR)
        except Exception as e:
            print(f"Error deleting directory {LOCAL_CLONE_DIR}: {e}")
            return None

    try:
        print("Cloning the repository with Git LFS support...")
        subprocess.run(["git", "clone", git_url_with_credentials, LOCAL_CLONE_DIR], check=True)
        os.chdir(LOCAL_CLONE_DIR)
        subprocess.run(["git", "lfs", "install"], check=True)
        subprocess.run(["git", "lfs", "pull"], check=True)
        print(f"Successfully cloned repository into {LOCAL_CLONE_DIR} and downloaded all LFS files.")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository or downloading LFS files: {e}")
        return None

    return LOCAL_CLONE_DIR

# Step 2: Filter only the PDF files from both validation and test metadata.jsonl
def filter_pdf_files(**kwargs):
    local_clone_dir = kwargs['ti'].xcom_pull(task_ids='clone_repo')
    datasets = ['validation', 'test']
    pdf_files = []
    dataset_counts = {}

    for dataset in datasets:
        metadata_file = os.path.join(local_clone_dir, '2023', dataset, 'metadata.jsonl')
        count = 0
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    if data.get('file_name', '').endswith('.pdf'):
                        data['dataset'] = dataset
                        pdf_files.append(data)
                        count += 1
        else:
            print(f"Metadata file not found for {dataset}")

        dataset_counts[dataset] = count
    
    for dataset, count in dataset_counts.items():
        print(f"Found {count} PDF files in {dataset} dataset.")

    return pdf_files

# Step 3: Process each PDF (extract text, images, tables)
def process_pdf(pdf_file, **kwargs):
    dataset = pdf_file['dataset']
    pdf_path = os.path.join(f"./GAIA/2023/{dataset}/", pdf_file['file_name'])
    
    print(f"Processing PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        return None

    output_txt_path = pdf_path.replace('.pdf', '.txt')
    
    try:
        with open(pdf_path, 'rb') as pdf_file_obj:
            reader = PdfReader(pdf_file_obj)
            basic_text = ''
            for page in reader.pages:
                basic_text += page.extract_text() + '\n'

        try:
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
        except Exception as e:
            tables = []
            print(f"Error extracting tables from {pdf_path}: {e}")

        table_text = ''
        for i, table in enumerate(tables):
            table_text += f"\n--- Table {i+1} ---\n"
            table_text += table.to_csv(index=False) + '\n'

        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(f"--- Basic Text ---\n{basic_text}\n\n")
            txt_file.write(f"--- Table Text ---\n{table_text}")
        
        print(f"Successfully processed: {pdf_path}")
        return output_txt_path

    except Exception as e:
        print(f"Error processing PDF: {pdf_path}")
        print(f"Error details: {str(e)}")
        return None

# Function to ensure the SQL table exists
def ensure_table_exists():
    try:
        connection = pymysql.connect(
            host=GCP_SQL_HOST,
            user=GCP_SQL_USER,
            password=GCP_SQL_PASSWORD,
            database=GCP_SQL_DATABASE,
        )
        cursor = connection.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS files_pypdf (
            id INT AUTO_INCREMENT PRIMARY KEY,
            file_name VARCHAR(255) NOT NULL,
            processed_file_name VARCHAR(255) NOT NULL,
            dataset VARCHAR(255) NOT NULL,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        cursor.close()
        connection.close()
        print("Ensured that files_pypdf table exists with processed_file_name column.")
    except Exception as e:
        print(f"Error ensuring table exists: {str(e)}")

# Function to insert processed file names into the GCP SQL database
def insert_file_name_to_sql(file_name, processed_file_name, dataset):
    try:
        ensure_table_exists()  # Make sure the table exists before inserting
        connection = pymysql.connect(
            host=GCP_SQL_HOST,
            user=GCP_SQL_USER,
            password=GCP_SQL_PASSWORD,
            database=GCP_SQL_DATABASE,
        )
        
        cursor = connection.cursor()
        sql_query = """
        INSERT INTO files_pypdf (file_name, processed_file_name, dataset)
        VALUES (%s, %s, %s)
        """
        cursor.execute(sql_query, (file_name, processed_file_name, dataset))
        connection.commit()
        cursor.close()
        connection.close()
        print(f"Inserted {file_name} and {processed_file_name} into files_pypdf table.")
        
    except Exception as e:
        print(f"Error inserting {file_name} into SQL: {str(e)}")


# Updated function to process all PDFs and insert into SQL
def process_all_pdfs(**kwargs):
    pdf_files = kwargs['ti'].xcom_pull(task_ids='filter_pdfs')
    processed_files = []
    failed_files = []
    
    for pdf_file in pdf_files:
        txt_file_path = process_pdf(pdf_file)
        if txt_file_path:
            processed_files.append({
                'file_path': txt_file_path,
                'dataset': pdf_file['dataset']
            })
            # Use the .txt filename along with the original .pdf filename
            txt_file_name = os.path.basename(txt_file_path)
            insert_file_name_to_sql(pdf_file['file_name'], txt_file_name, pdf_file['dataset'])
        else:
            failed_files.append(pdf_file['file_name'])
    
    print(f"Processing complete. Processed: {len(processed_files)}, Failed: {len(failed_files)}")
    
    kwargs['ti'].xcom_push(key='processed_files', value=processed_files)
    kwargs['ti'].xcom_push(key='failed_files', value=failed_files)

# Step 4: Upload the .txt files to GCP, storing them in 'test' or 'validation' folders based on the dataset
def upload_to_gcp(txt_file_path, dataset, **kwargs):
    try:
        storage_client = storage.Client.from_service_account_json(GCP_SERVICE_ACCOUNT_FILE)
        bucket = storage_client.bucket(GCP_BUCKET_NAME)
        destination_blob_name = f"{dataset}/{os.path.basename(txt_file_path)}"
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(txt_file_path)
        print(f"{txt_file_path} uploaded to {GCP_BUCKET_NAME}/{dataset}.")
        return True
    except Exception as e:
        print(f"Error uploading to GCP: {txt_file_path}")
        print(f"Error details: {str(e)}")
        return False

# Function to upload all processed files
def upload_all_files(**kwargs):
    processed_files = kwargs['ti'].xcom_pull(task_ids='process_pdfs', key='processed_files')
    uploaded_count = 0
    failed_count = 0
    
    for txt_file_path in processed_files:
        dataset = txt_file_path['dataset']
        txt_file_full_path = txt_file_path['file_path']
        
        if upload_to_gcp(txt_file_full_path, dataset):
            uploaded_count += 1
        else:
            failed_count += 1
    
    print(f"Upload complete. Uploaded: {uploaded_count}, Failed: {failed_count}")

# Step 5: Define tasks in the DAG
with dag:
    # Clone the dataset
    clone_repo = PythonOperator(
        task_id='clone_repo',
        python_callable=clone_repository
    )

    # Filter only the PDFs from both validation and test datasets
    filter_pdfs = PythonOperator(
        task_id='filter_pdfs',
        python_callable=filter_pdf_files,
        provide_context=True
    )

    # Process all PDF files and insert the file names into GCP SQL
    process_pdfs = PythonOperator(
        task_id='process_pdfs',
        python_callable=process_all_pdfs,
        provide_context=True
    )

    # Upload all processed files
    upload_files = PythonOperator(
        task_id='upload_files',
        python_callable=upload_all_files,
        provide_context=True
    )

    # Task flow
    clone_repo >> filter_pdfs >> process_pdfs >> upload_files
    