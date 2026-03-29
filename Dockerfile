# Use the latest stable Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# - build-essential: for compiling local packages
# - libgomp1: REQUIRED for FAISS (Vector DB) to function in Docker
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose ports for both the API (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default command runs the UI, but we override this in docker-compose for the backend
CMD ["streamlit", "run", "frontend/ui_main.py", "--server.port=8501", "--server.address=0.0.0.0"]