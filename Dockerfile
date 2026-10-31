# Python 3.12 slim base image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# The FAISS index and its metadata are written here (see backend/core/utilities.py).
ENV VECTOR_DATA_DIR=/app/data

# System dependencies
# - libgomp1: REQUIRED for FAISS (Vector DB) to function in Docker
# No compiler is installed: the Python dependencies install from prebuilt wheels.
# If a dependency ever needs one, add build-essential back here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run the app as an unprivileged user (its home folder holds the model download cache)
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

# Set working directory
WORKDIR /app

# Install Python dependencies (own layer: cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy the application code, then make /app and the data folder writable for the app user
COPY --chown=appuser:appuser . .
RUN mkdir -p /app/data \
    && chown appuser:appuser /app /app/data \
    && chmod +x /app/start.sh

USER appuser

# Expose ports for both the API (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Force execution using an absolute path shell call
CMD ["/bin/bash", "/app/start.sh"]
