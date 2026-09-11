#!/bin/bash

# FIX: Make the absolute root app directory visible to both backend and frontend execution loops
export PYTHONPATH=/app

# 1. Start FastAPI backend in the background on port 8000
uvicorn backend.backend:app --host 0.0.0.0 --port 8000 &

# 2. Wait a brief moment for the backend to spin up safely
sleep 3

# 3. Start Streamlit frontend on the cloud platform's assigned $PORT
streamlit run frontend/ui_main.py --server.port $PORT --server.address 0.0.0.0
