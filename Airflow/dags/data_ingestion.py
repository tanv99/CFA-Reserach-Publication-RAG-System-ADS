from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import boto3
import requests
import time
import json
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# Import AWS credentials and other variables from env_var.py
import env_var  # Make sure this is in the same directory or in PYTHONPATH

# Set default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'web_scraping_pipeline',
    default_args=default_args,
    description='A web scraping pipeline to ingest data to S3',
    schedule_interval=None,  # You can set a schedule later
)

def setup_s3_client():
    """Sets up AWS S3 client using environment variables from env_var.py."""
    s3 = boto3.client(
        's3',
        aws_access_key_id=env_var.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=env_var.AWS_SECRET_ACCESS_KEY,
        region_name=env_var.AWS_REGION
    )
    bucket_name = env_var.AWS_BUCKET_NAME
    return s3, bucket_name

# def setup_webdriver():
#     """Sets up Chrome webdriver with appropriate options."""
#     chrome_options = Options()
#     # chrome_options.add_argument("--headless")
#     chrome_options.add_argument("--disable-gpu")
#     chrome_options.add_argument("--no-sandbox")
#     chrome_options.add_argument("--disable-dev-shm-usage")
#     chrome_options.add_argument("--window-size=1920,1080")
#     chrome_options.add_argument("--remote-debugging-port=9222")
#     return webdriver.Chrome(options=chrome_options)

def setup_webdriver():
    """Sets up Chrome webdriver with appropriate options."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--remote-debugging-port=9222")  # Debugging port
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--window-size=1920,1080")
    return webdriver.Chrome(options=chrome_options)

def clean_filename(title):
    """Create a safe filename from title."""
    return "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()

def scrape_publication_page(publication_url, s3, bucket):
    """Scrapes the publication page using Selenium for dynamic content."""
    driver = setup_webdriver()
    driver.get(publication_url)

    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "h1")))

        title_element = driver.find_element(By.CLASS_NAME, "spotlight-hero__title")
        title = title_element.text.strip() if title_element else None

        if not title:
            print(f"No title found for {publication_url}")
            return

        safe_title = clean_filename(title)
        print(f"Processing: {safe_title}")

        # Create metadata dictionary
        metadata = {
            'title': title,
            'url': publication_url,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'has_image': False,
            'has_summary': False,
            'has_pdf': False
        }

        # Extract and upload summary
        try:
            summary_element = driver.find_element(By.CSS_SELECTOR, 'meta[name="description"]')
            if summary_element:
                summary_content = summary_element.get_attribute('content').strip()
                if summary_content:
                    metadata['has_summary'] = True
                    metadata['summary'] = summary_content
        except Exception:
            pass

        # Upload metadata to S3
        s3.put_object(
            Bucket=bucket,
            Key=f"{safe_title}/metadata.json",
            Body=json.dumps(metadata, indent=2).encode('utf-8'),
            ContentType='application/json'
        )
        print(f"Uploaded metadata for {safe_title}")
    except Exception as e:
        print(f"Error processing {publication_url}: {e}")
    finally:
        driver.quit()

def get_publication_links():
    """Extracts publication links using Selenium."""
    driver = setup_webdriver()
    driver.get('https://rpc.cfainstitute.org/en/research-foundation/publications')

    publications = set()
    try:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".CoveoSearchInterface")))

        elements = driver.find_elements(By.TAG_NAME, "a")
        for element in elements:
            href = element.get_attribute('href')
            if href and '/research/foundation/' in href:
                publications.add(href)
    except Exception as e:
        print(f"Error getting publication links: {e}")
    finally:
        driver.quit()

    return list(publications)

def scrape_main_page(**kwargs):
    """Scrapes the main publications page and processes each publication."""
    s3, bucket = setup_s3_client()
    publication_links = get_publication_links()

    for pub_url in publication_links:
        scrape_publication_page(pub_url, s3, bucket)

# Define PythonOperators for each task
scrape_task = PythonOperator(
    task_id='scrape_main_page',
    python_callable=scrape_main_page,
    dag=dag,
)

scrape_task