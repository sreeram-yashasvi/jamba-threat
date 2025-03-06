FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies - RunPod package first to avoid conflicts
RUN pip install --no-cache-dir runpod==0.10.0

# Copy requirements file
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install debugging tools
RUN pip install --no-cache-dir fastapi uvicorn python-dotenv

# Create a proper Python module structure
RUN mkdir -p /app/jamba_model

# Copy model and handler files first and verify imports work
COPY src/jamba_model.py /app/jamba_model/model.py 
COPY src/handler.py /app/handler.py

# Create proper __init__.py for the module
RUN echo "from .model import JambaThreatModel, ThreatDataset" > /app/jamba_model/__init__.py

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create a startup verification script
RUN echo '#!/bin/bash' > /app/startup_check.sh && \
    echo 'echo "Starting container validation checks..."' >> /app/startup_check.sh && \
    echo 'echo "Checking Python imports..."' >> /app/startup_check.sh && \
    echo 'python -c "from jamba_model import JambaThreatModel, ThreatDataset; print(\"✓ Imports successful\")"' >> /app/startup_check.sh && \
    echo 'if [ $? -ne 0 ]; then' >> /app/startup_check.sh && \
    echo '  echo "✗ Import check failed - container may not function correctly"' >> /app/startup_check.sh && \
    echo '  exit 1' >> /app/startup_check.sh && \
    echo 'fi' >> /app/startup_check.sh && \
    echo 'echo "Checking environment..."' >> /app/startup_check.sh && \
    echo 'echo "PYTHONPATH: $PYTHONPATH"' >> /app/startup_check.sh && \
    echo 'echo "Current directory: $(pwd)"' >> /app/startup_check.sh && \
    echo 'echo "Files in jamba_model: $(ls -la /app/jamba_model)"' >> /app/startup_check.sh && \
    echo 'echo "✓ All validation checks passed"' >> /app/startup_check.sh && \
    chmod +x /app/startup_check.sh

# Copy the startup wrapper script
COPY src/runpod_entry.sh /app/runpod_entry.sh
RUN chmod +x /app/runpod_entry.sh

# RunPod specific settings
ENV RUNPOD_DEBUG_LEVEL=DEBUG
ENV RUNPOD_STOP_SIGNAL=SIGINT
ENV RUNPOD_TIMEOUT_SECONDS=900

# Set healthcheck to verify the container is working properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; from jamba_model import JambaThreatModel; print('Health check passed')" || exit 1

# RunPod entrypoint - Use our wrapper script to ensure proper startup
CMD ["/app/runpod_entry.sh"] 