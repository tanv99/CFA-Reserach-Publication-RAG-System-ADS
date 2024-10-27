# FROM apache/airflow:2.10.2

# USER root

# # Install git and other required system dependencies
# RUN apt-get update && apt-get install -y \
#     git \
#     git-lfs \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     tesseract-ocr \
#     libtesseract-dev \
#     poppler-utils \
#     default-jre \
#     wget \
#     gnupg2 \
#     apt-transport-https \
#     ca-certificates \
#     chromium \
#     chromium-driver \
#     # Add these additional dependencies
#     libnss3 \
#     libgconf-2-4 \
#     libfontconfig1

# # Create directory for Chrome if it doesn't exist
# RUN mkdir -p /opt/chrome && chmod -R 777 /opt/chrome

# # Create and set permissions for airflow directories
# RUN mkdir -p /opt/airflow/dags /opt/airflow/plugins /opt/airflow/credentials /opt/airflow/config && \
#     chown -R airflow:root /opt/airflow && \
#     chmod -R 777 /opt/airflow

# # Switch to the airflow user for installing Python packages
# USER airflow

# # Install Python packages
# RUN pip install --no-cache-dir \
#     pymysql \
#     pandas \
#     streamlit \
#     numpy \
#     PyPDF2[full]==3.0.1 \
#     pytesseract==0.3.13 \
#     pdf2image==1.17.0 \
#     opencv-python \
#     tabula-py \
#     azure-ai-formrecognizer==3.2.0 \
#     selenium \
#     webdriver-manager \
#     # Add these packages
#     boto3 \
#     requests \
#     PyMuPDF \
#     pillow

# # Copy the contents of the folders into the image
# # Note: COPY instructions automatically create directories if they don't exist
# COPY --chown=airflow:root dags /opt/airflow/dags/
# COPY --chown=airflow:root plugins /opt/airflow/plugins/
# COPY --chown=airflow:root credentials /opt/airflow/credentials/
# COPY --chown=airflow:root config /opt/airflow/config/

# # Set environment variables
# ENV DISPLAY=:99
# ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
# ENV PYTHONPATH=/opt/airflow/dags

# # Keep as airflow user for security
# USER airflow


# Base Image
FROM apache/airflow:2.10.2

# Switch to root for system installations
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    default-jre \
    wget \
    gnupg2 \
    apt-transport-https \
    ca-certificates \
    chromium \
    chromium-driver \
    libnss3 \
    libfontconfig1 \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories and set permissions
RUN mkdir -p /opt/chrome \
    /opt/airflow/dags \
    /opt/airflow/plugins \
    /opt/airflow/credentials \
    /opt/airflow/config \
    /opt/airflow/logs \
    /opt/airflow/logs/selenium && \
    chmod -R 777 /opt/chrome && \
    chown -R airflow:root /opt/airflow && \
    chmod -R 777 /opt/airflow

# Verify Chrome installations
RUN chromium --version && \
    chromedriver --version

# Switch to airflow user for Python package installations
USER airflow

# Install Python packages
RUN pip install --no-cache-dir \
    pymysql \
    pandas \
    streamlit \
    numpy \
    PyPDF2[full]==3.0.1 \
    pytesseract==0.3.13 \
    pdf2image==1.17.0 \
    opencv-python \
    tabula-py \
    azure-ai-formrecognizer==3.2.0 \
    selenium \
    boto3 \
    requests \
    PyMuPDF \
    pillow \
    beautifulsoup4

# Copy application files
COPY --chown=airflow:root dags /opt/airflow/dags/
COPY --chown=airflow:root plugins /opt/airflow/plugins/
COPY --chown=airflow:root credentials /opt/airflow/credentials/
COPY --chown=airflow:root config /opt/airflow/config/

# Set environment variables
ENV PYTHONPATH=/opt/airflow/dags
ENV DISPLAY=:99
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV CHROME_BINARY_PATH=/usr/bin/chromium
ENV SELENIUM_DRIVER_PATH=/usr/bin/chromedriver

# Start Xvfb
RUN Xvfb :99 -screen 0 1920x1080x24 > /dev/null 2>&1 &

# Final verification of installations and permissions
RUN echo "=== Verifying installations ===" && \
    echo "Python packages:" && pip list | grep -E "selenium|boto3|PyMuPDF" && \
    echo -e "\nChrome installations:" && \
    echo "Chromium: $(chromium --version)" && \
    echo "ChromeDriver: $(chromedriver --version)" && \
    echo -e "\nFile permissions:" && \
    ls -la /usr/bin/chromium /usr/bin/chromedriver

# Stay as airflow user for security
USER airflow

# Set working directory
WORKDIR /opt/airflow