#!/bin/bash

# Script to build Docker container with T5 model

set -e

echo "🐳 Building RAG QA System Docker Container"
echo "=========================================="

# Check if the source model directory exists
SOURCE_MODEL_DIR="/home/fg12/.cache/huggingface/hub/models--t5-small"
if [ ! -d "$SOURCE_MODEL_DIR" ]; then
    echo "❌ Error: Source model directory not found: $SOURCE_MODEL_DIR"
    echo "Please make sure the T5 model is downloaded to the correct location."
    exit 1
fi

# Create models directory if it doesn't exist
echo "📁 Creating models directory..."
mkdir -p models

# Copy the T5 model to the local models directory
echo "📋 Copying T5 model to local models directory..."
cp -r "$SOURCE_MODEL_DIR" models/

# Verify the model was copied
if [ ! -d "models/models--t5-small" ]; then
    echo "❌ Error: Failed to copy model to models directory"
    exit 1
fi

echo "✅ Model copied successfully"

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t rag-qa-system .

echo "✅ Docker image built successfully!"
echo ""
echo "🚀 To run the container:"
echo "   docker run -p 8501:8501 rag-qa-system"
echo ""
echo "   Or use docker-compose:"
echo "   docker-compose up"
echo ""
echo "🌐 The application will be available at: http://localhost:8501"