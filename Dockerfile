FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies - RunPod package first to avoid conflicts
RUN pip install --no-cache-dir runpod==0.10.0

# Install other dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Install debugging tools
RUN pip install --no-cache-dir fastapi uvicorn

# Create a proper Python module structure
RUN mkdir -p /app/jamba_model
COPY src/jamba_model.py /app/jamba_model/model.py
RUN echo "from jamba_model.model import JambaThreatModel, ThreatDataset" > /app/jamba_model/__init__.py

# Copy handler and other files
COPY src/handler.py /app/handler.py

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Test that imports work correctly - fail the build if they don't
RUN python -c "from jamba_model import JambaThreatModel, ThreatDataset; print('Import test successful!')"

# RunPod specific settings
ENV RUNPOD_DEBUG_LEVEL=DEBUG
ENV RUNPOD_STOP_SIGNAL=SIGINT
ENV RUNPOD_TIMEOUT_SECONDS=900

# Set healthcheck to verify the container is working properly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; from jamba_model import JambaThreatModel; print('Health check passed')" || exit 1

# RunPod entrypoint - needs to use the runpod-handler format
CMD [ "python", "-m", "runpod.serverless.start" ] 