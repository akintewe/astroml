# Multi-stage Dockerfile for AstroML
# This Dockerfile creates optimized images for both ingestion and training environments

# ============================================================================
# BASE STAGE - Common dependencies and Python environment
# ============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r astroml && useradd -r -g astroml astroml

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ============================================================================
# INGESTION STAGE - Optimized for data ingestion and streaming
# ============================================================================
FROM base as ingestion

# Install additional dependencies for ingestion
RUN apt-get update && apt-get install -y \
    jq \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY --chown=astroml:astroml astroml/ ./astroml/
COPY --chown=astroml:astroml migrations/ ./migrations/

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R astroml:astroml /app

# Switch to non-root user
USER astroml

# Expose ports for health checks and monitoring
EXPOSE 8000 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import astroml.ingestion" || exit 1

# Default command for ingestion
CMD ["python", "-m", "astroml.ingestion"]

# ============================================================================
# TRAINING STAGE - Optimized for ML training with GPU support
# ============================================================================
FROM nvidia/cuda:12.1-runtime-base-ubuntu22.04 as training-base

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-pip \
    python3.11-dev \
    build-essential \
    curl \
    git \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=0

# Create app user
RUN groupadd -r astroml && useradd -r -g astroml astroml

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric with CUDA support
RUN pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Copy application code
COPY --chown=astroml:astroml astroml/ ./astroml/

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R astroml:astroml /app

# Switch to non-root user
USER astroml

# Expose port for monitoring (TensorBoard, etc.)
EXPOSE 6006

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import torch_geometric" || exit 1

# Default command for training
CMD ["python", "-m", "astroml.training.train_gcn"]

# ============================================================================
# CPU-ONLY TRAINING STAGE - For environments without GPU
# ============================================================================
FROM base as training-cpu

# Copy application code
COPY --chown=astroml:astroml astroml/ ./astroml/

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R astroml:astroml /app

# Switch to non-root user
USER astroml

# Expose port for monitoring
EXPOSE 6006

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import torch_geometric" || exit 1

# Default command for training
CMD ["python", "-m", "astroml.training.train_gcn"]

# ============================================================================
# DEVELOPMENT STAGE - Includes development tools and testing
# ============================================================================
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-cov black flake8 mypy jupyter

# Copy application code
COPY --chown=astroml:astroml astroml/ ./astroml/
COPY --chown=astroml:astroml tests/ ./tests/
COPY --chown=astroml:astroml migrations/ ./migrations/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/notebooks && \
    chown -R astroml:astroml /app

# Switch to non-root user
USER astroml

# Expose ports for development
EXPOSE 8000 8080 8888 6006

# Default command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]

# ============================================================================
# PRODUCTION STAGE - Minimal production image
# ============================================================================
FROM base as production

# Copy only necessary files for production
COPY --chown=astroml:astroml astroml/ ./astroml/

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R astroml:astroml /app

# Switch to non-root user
USER astroml

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import astroml" || exit 1

# Default production command (can be overridden)
CMD ["python", "-m", "astroml.ingestion"]
