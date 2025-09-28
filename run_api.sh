#!/bin/sh
# Simple script to run the Flask API using Gunicorn

echo "Starting Gunicorn server..."

# gunicorn is the production HTTP server
# -w 4: 4 worker processes (good for multi-core systems)
# -b 0.0.0.0:5000: Binds to all network interfaces on port 5000
# src.rag_api:app: Targets the 'app' object inside src/rag_api.py
exec gunicorn -w 4 -b 0.0.0.0:5000 src.rag_api:app
