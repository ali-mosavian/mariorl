# Dockerfile for MadMario training on RunPod
# 
# This creates a container with all dependencies for training the world model agent
#
# Build:
#   docker build -t madmario .
#
# Run locally:
#   docker run --gpus all -v $(pwd)/checkpoints:/workspace/checkpoints madmario
#
# For RunPod: Use this as a custom container or build from this Dockerfile

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    curl \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfcitx-bin \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install Python dependencies using uv
RUN uv sync

# Make training script executable
RUN chmod +x train_runpod.sh

# Create checkpoints directory
RUN mkdir -p checkpoints

# Default command runs training
# Can override with custom environment variables
CMD ["bash", "./train_runpod.sh"]

