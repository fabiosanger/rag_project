#!/bin/bash

# RAG QA System - Streamlit Launcher
echo "🤖 Starting RAG QA System with Streamlit..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first."
    echo "   pip install uv"
    exit 1
fi

# Set uv virtual environment path
export UV_VENV_PATH="/home/fg12/envs/"

# Install dependencies if pyproject.toml exists
if [ -f "pyproject.toml" ]; then
    echo "📦 Installing dependencies with uv..."
    uv sync
else
    echo "⚠️  pyproject.toml not found. Installing basic dependencies..."
    uv add streamlit torch transformers sentence-transformers scikit-learn numpy pandas
fi

# Check if streamlit app exists
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ streamlit_app.py not found. Please ensure the file exists in the current directory."
    exit 1
fi

# Run Streamlit
echo "🚀 Launching Streamlit application..."
echo "📱 The app will open in your browser at http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

uv run streamlit run streamlit_app.py --server.port 8501 --server.address localhost