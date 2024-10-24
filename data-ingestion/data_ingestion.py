import boto3
import requests
import time
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urljoin
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('scraper.log'), logging.StreamHandler()]
)

def setup_s3_client():
    """Sets up AWS S3 client."""
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.environ["AWS_REGION"]
        )
        bucket_name = os.environ["AWS_BUCKET_NAME"]
        logging.info("Successfully connected to AWS S3")
        return s3, bucket_name
    except Exception as e:
        logging.error(f"Error setting up S3 client: {e}")
        raise

def upload_to_s3(s3, bucket, key_name, content, content_type):
    # logging.info("upload:", s3, bucket, key_name, content, content_type)
    s3.put_object(Bucket=bucket, Key=key_name, Body=content, ContentType=content_type)
    logging.info(f"Successfully uploaded {key_name}")

def scrape_publication_page(publication_url, s3, bucket):
    """Scrapes the publication page for title, image, summary, and PDF."""
    try:
        response = requests.get(publication_url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title_element = soup.find('h1', class_='spotlight-hero__title')
        title = title_element.get_text(strip=True) if title_element else None

        if not title:
            logging.warning(f"No title found for {publication_url}")
            return

        logging.info(f"Processing: {title}")
        print(f"Title: {title} (From {publication_url})")

        # Initialize content variables
        image_content = None
        summary_content = None
        pdf_content = None

        # Extract and upload image
        image_section = soup.find('picture', class_='spotlight-hero__image')
        if image_section and (img_tag := image_section.find('img')):
            image_url = urljoin('https://rpc.cfainstitute.org', img_tag['src'])
            print(f"Image URL: {image_url} (From {publication_url})")
            image_response = requests.get(image_url, timeout=30)
            if image_response.status_code == 200:
                image_content = image_response.content
                print(f"Image content: {len(image_content)} bytes")  # Show image size
                if upload_to_s3(s3, bucket, f"{title}.jpg", image_content, 'image/jpeg'):
                    logging.info(f"Image uploaded: {title}.jpg")
            else:
                logging.warning(f"Failed to fetch image: {image_url}")
        else:
            logging.warning(f"No image found for {title}")

        # Extract and upload summary
        summary_meta = soup.find('meta', {'name': 'description'})
        summary_content = summary_meta['content'].strip() if summary_meta else None
        if summary_content:
            print(f"Summary: {summary_content} (From {publication_url})")
            print(f"Summary content being uploaded: {summary_content}")
            if upload_to_s3(s3, bucket, f"{title}_summary.txt", summary_content, 'text/plain'):
                logging.info(f"Summary uploaded: {title}_summary.txt")
        else:
            logging.warning(f"No summary found for {title}")

        # Extract and upload PDF
        pdf_link = (
            soup.find('a', string='Download Full PDF') or
            soup.find('a', href=lambda x: x and '.pdf' in x)
        )
        if pdf_link and 'href' in pdf_link.attrs:
            pdf_url = urljoin('https://rpc.cfainstitute.org', pdf_link['href'])
            print(f"PDF URL: {pdf_url} (From {publication_url})")
            pdf_response = requests.get(pdf_url, timeout=30)
            if pdf_response.status_code == 200:
                pdf_content = pdf_response.content
                print(f"PDF content: {len(pdf_content)} bytes")  # Show PDF size
                if upload_to_s3(s3, bucket, f"{title}.pdf", pdf_content, 'application/pdf'):
                    logging.info(f"PDF uploaded: {title}.pdf")
            else:
                logging.warning(f"Failed to fetch PDF: {pdf_url}")
        else:
            logging.info(f"No PDF found for: {title}")

    except Exception as e:
        logging.exception(f"Error processing {publication_url}")

def get_publication_links(soup):
    """Extracts publication links from the page."""
    publications = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/en/research-foundation/' in href and not any(x in href for x in ['/donate', '/rf-review-board']):
            publications.append(href)
    return list(set(publications))

def scrape_main_page(main_url, s3, bucket, max_pages=100):
    """Scrapes the main publications page with robust pagination handling."""
    base_url = 'https://rpc.cfainstitute.org'
    page = 1

    while page <= max_pages:
        try:
            logging.info(f"Scraping page {page}")
            url = f"{main_url}?page={page}"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Get publication links
            publications = get_publication_links(soup)
            if not publications:
                logging.info(f"No publications found on page {page}. Stopping.")
                break  # Stop if no new publications are found

            # Scrape each publication
            for pub_url in publications:
                full_pub_url = urljoin(base_url, pub_url)
                scrape_publication_page(full_pub_url, s3, bucket)
                time.sleep(2)  # Polite delay

            # Check for the "Next" button on the main page
            next_page = soup.find('li', class_='coveo-pager-next')
            if not next_page:
                logging.info("No more pages to scrape")
                break  # Stop if no "Next" button is found

            page += 1  # Move to the next page

        except requests.exceptions.RequestException as e:
            logging.error(f"Error scraping page {page}: {e}")
            if "404" in str(e):
                break
            time.sleep(5)  # Delay on error

def main():
    s3, bucket = setup_s3_client()
    main_url = 'https://rpc.cfainstitute.org/en/research-foundation/publications'
    scrape_main_page(main_url, s3, bucket)
    logging.info("Scraping completed successfully")

if __name__ == "__main__":
    load_dotenv()
    main()
