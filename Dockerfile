# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy uv configuration files
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN uv venv && \
    uv pip compile pyproject.toml --extra dev -o requirements.txt && \
    uv pip install --system -r requirements.txt

# Copy the local T5 model to the container
COPY models/models-t5-small /app/models/models-t5-small

# Copy application code
COPY rag_project/ ./rag_project/
COPY streamlit_app.py ./
COPY sample_data.json ./
COPY demo.py ./
COPY main.py ./

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose Streamlit port
EXPOSE 8501

# Set the default command to run Streamlit
CMD ["uv", "run", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]