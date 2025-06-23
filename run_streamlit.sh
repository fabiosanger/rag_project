#!/bin/bash

# RAG QA System - Streamlit Launcher
echo "🤖 Starting RAG QA System with Streamlit..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

# Install dependencies if requirements file exists
if [ -f "requirements_streamlit.txt" ]; then
    echo "📦 Installing dependencies..."
    pip3 install -r requirements_streamlit.txt
else
    echo "⚠️  requirements_streamlit.txt not found. Installing basic dependencies..."
    pip3 install streamlit torch transformers sentence-transformers scikit-learn numpy pandas
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

streamlit run streamlit_app.py --server.port 8501 --server.address localhost