#!/bin/bash
export PYTHONPATH=/app

echo "🚀 Starting FastAPI Backend..."
uvicorn backend.backend:app --host 127.0.0.1 --port 8000 &

echo "⏳ Waiting for FastAPI backend to initialize..."
# Ensure this exact python command is what is saved locally on line 11:
while ! python3 -c "import socket; s = socket.socket(); s.connect(('127.0.0.1', 8000))" 2>/dev/null; do
  sleep 1
done
echo "✅ Backend is up! Starting Streamlit frontend..."

streamlit run frontend/ui_main.py --server.port $PORT --server.address 0.0.0.0
