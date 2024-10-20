FROM apache/airflow:2.10.2

USER root

# Install git and other required system dependencies
RUN apt-get update && apt-get install -y git git-lfs \
    libgl1-mesa-glx libglib2.0-0 tesseract-ocr libtesseract-dev poppler-utils default-jre

# Switch to the airflow user for installing Python packages
USER airflow

# Install Python packages, including PyMySQL
RUN pip install --no-cache-dir pymysql pandas streamlit numpy PyPDF2[full]==3.0.1 \
    pytesseract==0.3.13 pdf2image==1.17.0 opencv-python tabula-py azure-ai-formrecognizer==3.2.0

# Copy the contents of the dags, plugins, logs, and credentials folders into the image
COPY dags /opt/airflow/dags
COPY plugins /opt/airflow/plugins
COPY credentials /opt/airflow/credentials
COPY config /opt/airflow/config

# Switch back to root for any additional commands, if needed
USER root