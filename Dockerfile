# Multi-stage build for perceptra-detector API service

# Stage 1: Base image with Python and system dependencies
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: Dependencies
FROM base as dependencies

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional backends
RUN pip install --no-cache-dir ultralytics transformers

# Stage 3: Application
FROM dependencies as application

# Copy package
COPY . /app

# Install package
RUN pip install --no-cache-dir -e .

# Create directory for models
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "perceptra_detector.api.server:create_app", "--host", "0.0.0.0", "--port", "8000", "--factory"]

# For GPU support, use nvidia/cuda base image:
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base