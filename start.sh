#!/bin/bash
# Start FastAPI backend in the background on port 8000
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit frontend on Hugging Face's required port 7860
streamlit run frontend/ui_main.py --server.port 7860 --server.address 0.0.0.0