#!/bin/bash
echo "Starting gunicorn server..."
NUM_WORKERS=$(nproc)
echo "Number of workers: $NUM_WORKERS" 
# gunicorn -w ${NUM_WORKERS} -b :8080 -t 300 -k uvicorn.workers.UvicornWorker --log-config log.ini --reload main:app
gunicorn -w ${NUM_WORKERS} -b :8080 -t 300 -k uvicorn.workers.UvicornWorker --reload main:app
