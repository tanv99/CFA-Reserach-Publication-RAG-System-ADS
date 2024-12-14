# CFA Institute Research Foundation Publications Analysis System

## Project Resources

Google codelab: https://codelabs-preview.appspot.com/?file_id=1jphJPxx12pEU5ZX9iGk58KrWBYvTFB6bMzoFP0cu97k#0

Streamlit App link (hosted on GCP): http://35.225.250.5:8501/

Airflow Link (hosted on GCP): http://35.225.250.5:8080/home

FastAPI Link (hosted on GCP): http://35.225.250.5:8000

Demo Video URL: https://drive.google.com/file/d/1kOeyb4jt2pzpOXBe0-YgoAaBJ-YVFK21/view?usp=sharing

Google collab notebook: https://colab.research.google.com/drive/1-jqj7SBSEZs0Yy58V54IdGMA9b2gZDYd?usp=sharing

### Goal of the Project
This project aims to build an automated system for extracting, processing, and analyzing research publications from the CFA Institute Research Foundation. The system combines web scraping, document processing, and AI-driven search and retrieval to create a user-friendly platform for exploring and analyzing financial research documents.

### Project Overview
The project automates the data ingestion and processing pipeline for CFA publications, transforming raw documents into structured data and summaries. With an interactive interface, users can explore documents, ask questions, and generate research notes. The backend supports complex queries using vector embeddings for semantic search, while the frontend allows seamless navigation of documents.

### Key Technologies Involved

- **Streamlit**: A user-friendly UI framework for browsing documents, generating summaries, and performing Q&A interactions.
- **FastAPI**: Handles backend API services, managing document retrieval, summary generation, and question-answering functionalities.
- **Selenium**: Web scraping framework for extracting data from the CFA Institute Research Foundation website.
- **PyMuPDF**: Processes PDF files, extracting text and images to facilitate further analysis.
- **LlamaParser**: Extracts structured content, organizing text for seamless integration with the backend.
- **Pinecone**: A vector database that stores document embeddings, enabling fast and accurate semantic searches across content.
- **NVIDIA API**: Summarizes documents and generates embeddings, aiding in search functionality and content analysis.
- **Airflow**: Orchestrates the pipeline tasks, automating web scraping, data processing, and loading to ensure the pipeline is consistent and up-to-date.
- **AWS S3**: Stores PDF files, images, and metadata securely, ensuring easy retrieval and backup of ingested data.
- **Snowflake**: A cloud database for structured data storage, handling metadata and document details for easy querying.

### Architecture diagram ###

![image](Architecture/images/cfa_architecture_diag.png)

### System Workflow

1. **Data Ingestion**: Airflow manages the ingestion pipeline, scraping CFA publications, and uploading data (PDFs, images) to AWS S3. Metadata is stored in Snowflake for structured access.
2. **Backend Processing**: FastAPI serves as the API layer, handling queries, generating summaries, and storing research notes.
3. **User Interaction**: Streamlit provides the interactive UI where users can browse publications, generate summaries, and query content.
4. **Search and Retrieval**: Pinecone indexes document embeddings, enabling semantic search across documents and research notes.

### Deployment
The system is deployed on **Google Cloud Platform (GCP)**, using Docker for containerized deployment:
- **Docker**: Containers manage FastAPI and Streamlit, with Docker Compose orchestrating the components for consistent deployment.
- **GCP**: Ensures public access to the application and scalable infrastructure to accommodate user demand.

### How to Use
1. **Register/Login**: Users can sign up and log in to access the platform.
2. **Browse Publications**: Use the grid or dropdown view to select documents for analysis.
3. **Generate Summaries**: Select a document to view a generated summary or key points.
4. **Q&A Interaction**: Ask questions directly from the document, and receive relevant answers powered by AI.
5. **Save Research Notes**: Generate and save insights for future reference or further analysis.
6. **Search Research Notes**: Search across notes for specific content, with search results differentiated by source (document or notes).

### Repository Structure
## Project Structure

```plaintext
ASSIGNMENT-3/
├── Airflow/
│   ├── dags/
│   │   ├── aws_ingestion_pipeline.py
│   │   └── snowflake_ingestion_pipeline.py
│   ├── config/
│   ├── credentials/
│   ├── logs/
│   ├── plugins/
│   ├── .gitignore
│   ├── docker-compose.yaml
│   └── Dockerfile
├── fastapi/
│   ├── credentials/
│   ├── storage/
│   ├── Dockerfile
│   ├── main_fastapi.py
│   ├── pdf_processor.py
│   ├── poetry.lock
│   ├── pyproject.toml
│   ├── text_processor.py
│   ├── utils.py
│   └── docker-compose.yaml
└── streamlit/
    ├── Dockerfile
    ├── index.py
    ├── page1.py
    ├── page2.py
    ├── page3.py
    ├── page4.py
    ├── poetry.lock
    └── pyproject.toml
```

### Prerequisites
- Docker and Docker Compose
- Access to GCP and AWS credentials
- Snowflake and Pinecone setup

### Environment Variables
Each component has its own .env file.

Airflow: Airflow/.env

FastAPI: fastapi/.env

Streamlit: streamlit/.env

### Steps to run the application
**Step 1: Clone the Repository**

If you haven’t done so already, clone the repository to your local machine:

git clone https://github.com/BigData-Fall2024-Team4/Assignment-3

cd Assignment-3

**Step 2: Initialize Airflow**

- cd Airflow

- docker compose up airflow-init

- docker compose up -d

- docker ps

**Step 3: Running Application:**
   
- docker compose up --build -d

- Streamlit application should now be accessible at http://localhost:8501

### Contributions

| Name                            | Percentage Contribution |
| ------------------------------- | ----------------------- |
| Sarthak Somvanshi               | 33.33% (User login, Q/A System, Report Generation)                    |
| Yuga Kanse                      | 33.33%  (Summary Generation, Search & Research note indexing functionality)                  |
| Tanvi Inchanalkar               | 33.33% (Airflow Pipelines-AWS & Snowflake, GCP Deployment, Documentation)                   |

### Additional Notes
WE ATTEST THAT WE HAVEN’T USED ANY OTHER STUDENTS’ WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK. 







