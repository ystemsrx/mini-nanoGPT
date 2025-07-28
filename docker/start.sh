#!/bin/bash
set -e

echo "Starting Mini-NanoGPT Docker Container..."
echo "================================================"

# Detect CUDA and get appropriate Python path
PYTHON_PATH=$(python3 /detect_cuda.py)

if [[ $PYTHON_PATH == *"cuda"* ]]; then
    echo "Environment: CUDA"
    ENV_TYPE="CUDA"
else
    echo "Environment: CPU"
    ENV_TYPE="CPU"
fi

echo "Python path: $PYTHON_PATH"
echo "================================================"

# Set environment variable for the application to know which environment is being used
export MINI_NANOGPT_ENV_TYPE="$ENV_TYPE"

# Change to app directory
cd /app

# Start the application
echo "Starting Mini-NanoGPT application..."
exec "$PYTHON_PATH" app.py