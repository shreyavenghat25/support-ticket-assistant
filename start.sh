#!/bin/sh
# Backend (internal only) and frontend (public, port 7860) in one container
python -m uvicorn services.api:app --host 127.0.0.1 --port 8000 &
exec streamlit run app/streamlit_app.py --server.port 7860 --server.address 0.0.0.0 \
     --server.headless true --server.enableCORS false --server.enableXsrfProtection false
