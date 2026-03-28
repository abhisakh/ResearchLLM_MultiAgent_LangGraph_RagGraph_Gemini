# Use the latest stable Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (required for FAISS and SQLite)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
# Increase timeout and point to the CPU-only repository for heavy ML libs
RUN pip install --no-cache-dir --default-timeout=1000 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose ports for the Backend (8000) and Frontend/Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Command to run the application (Adjust to your primary entry point)
CMD ["streamlit", "run", "frontend/ui_main.py", "--server.port=8501", "--server.address=0.0.0.0"]