# Docker Setup for RAG QA System

This guide explains how to dockerize and run the RAG QA System using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier management)
- The T5 model downloaded to `/home/fg12/.cache/huggingface/hub/models--t5-small`

## Quick Start

### 1. Build and Run with Script

```bash
# Make the build script executable (if not already done)
chmod +x build_docker.sh

# Run the build script
./build_docker.sh
```

This script will:
- Copy the T5 model from your local cache to the project
- Build the Docker image
- Provide instructions for running the container

### 2. Run the Container

After building, you can run the container in several ways:

#### Using Docker directly:
```bash
docker run -p 8501:8501 rag-qa-system
```

#### Using Docker Compose:
```bash
docker-compose up
```

#### Using Docker Compose in detached mode:
```bash
docker-compose up -d
```

### 3. Access the Application

Once running, the Streamlit application will be available at:
- **Local**: http://localhost:8501
- **Network**: http://your-server-ip:8501

## Manual Build Process

If you prefer to build manually:

### 1. Copy the Model

```bash
# Create models directory
mkdir -p models

# Copy the T5 model
cp -r /home/fg12/.cache/huggingface/hub/models--t5-small models/
```

### 2. Build the Image

```bash
docker build -t rag-qa-system .
```

### 3. Run the Container

```bash
docker run -p 8501:8501 rag-qa-system
```

## Docker Compose Features

The `docker-compose.yml` file includes:

- **Port mapping**: Maps container port 8501 to host port 8501
- **Volume mounting**: Mounts `sample_data.json` for easy updates
- **Health checks**: Monitors application health
- **Restart policy**: Automatically restarts on failure

## Configuration

### Environment Variables

You can customize the container behavior with environment variables:

```bash
docker run -p 8501:8501 \
  -e PYTHONUNBUFFERED=1 \
  -e STREAMLIT_SERVER_PORT=8501 \
  rag-qa-system
```

### Volume Mounts

For development, you can mount additional volumes:

```bash
docker run -p 8501:8501 \
  -v $(pwd)/sample_data.json:/app/sample_data.json:ro \
  -v $(pwd)/custom_data:/app/custom_data:ro \
  rag-qa-system
```

## Troubleshooting

### Model Loading Issues

If the model fails to load:

1. **Check model path**: Ensure the model is copied to `models/` directory
2. **Verify model structure**: The model should be in `models/models--t5-small/snapshots/...`
3. **Check permissions**: Ensure the model files are readable

### Port Conflicts

If port 8501 is already in use:

```bash
# Use a different port
docker run -p 8502:8501 rag-qa-system
```

### Memory Issues

For large models, you might need to increase Docker memory limits:

```bash
docker run -p 8501:8501 \
  --memory=4g \
  --memory-swap=4g \
  rag-qa-system
```

### GPU Support

To enable GPU support (if available):

```bash
docker run -p 8501:8501 \
  --gpus all \
  rag-qa-system
```

## Development

### Rebuilding the Image

After making changes to the code:

```bash
# Rebuild the image
docker build -t rag-qa-system .

# Or force rebuild without cache
docker build --no-cache -t rag-qa-system .
```

### Viewing Logs

```bash
# View container logs
docker logs <container_id>

# Follow logs in real-time
docker logs -f <container_id>
```

### Accessing the Container

```bash
# Execute commands inside the container
docker exec -it <container_id> /bin/bash

# Run Python commands
docker exec -it <container_id> uv run python -c "print('Hello from container')"
```

## Production Deployment

For production deployment:

1. **Use a production base image**: Consider using `python:3.11-slim` or `python:3.11-alpine`
2. **Add security**: Run as non-root user (already implemented)
3. **Optimize size**: Use multi-stage builds to reduce image size
4. **Add monitoring**: Include health checks and logging
5. **Use secrets**: Store sensitive data in Docker secrets or environment variables

## File Structure

```
rag_project/
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore          # Files to exclude from build
├── build_docker.sh        # Build script
├── models/                # Local model storage
│   └── models--t5-small/  # T5 model files
├── rag_project/           # Application code
├── streamlit_app.py       # Streamlit application
├── sample_data.json       # Sample Q&A data
└── pyproject.toml         # Project dependencies
```