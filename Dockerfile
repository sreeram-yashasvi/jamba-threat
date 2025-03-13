FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set labels
LABEL maintainer="Jamba Threat Detection Team"
LABEL version="2.0.0" 
LABEL description="Jamba Threat Detection with RunPod GPU Training"

# Install system dependencies in one RUN
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Upgrade pip and install Python dependencies in one layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir runpod==0.10.0 && \
    pip install --no-cache-dir --ignore-installed \
         requests==2.31.0 \
         python-dotenv==1.0.1 \
         pandas==2.0.3 \
         numpy==1.24.3 \
         tqdm==4.65.0 \
         fastapi uvicorn

# Create necessary directories in one command
RUN mkdir -p /app/src/jamba /app/utils /app/docs /app/models /app/data /app/logs

# Copy the entire jamba package
COPY src/jamba /app/src/jamba/
COPY src/handler.py /app/handler.py

# Copy source and utility files
COPY src/*.py /app/src/
COPY src/utils/*.py /app/utils/
COPY src/README_RUNPOD_TRAINING.md /app/docs/
COPY setup.py /app/setup.py

# Install the jamba package
RUN pip install -e .

# Make scripts executable
RUN chmod +x /app/src/*.py

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create startup verification script in a single RUN
RUN echo '#!/bin/bash' > /app/startup_check.sh && \
    echo 'echo "Starting container validation checks..."' >> /app/startup_check.sh && \
    echo 'echo "Checking Python imports..."' >> /app/startup_check.sh && \
    echo 'python -c "from jamba.jamba_model import JambaThreatModel, ThreatDataset; print(\"✓ Imports successful\")"' >> /app/startup_check.sh && \
    echo 'if [ $? -ne 0 ]; then' >> /app/startup_check.sh && \
    echo '  echo "✗ Import check failed - container may not function correctly"' >> /app/startup_check.sh && \
    echo '  exit 1' >> /app/startup_check.sh && \
    echo 'fi' >> /app/startup_check.sh && \
    echo 'echo "Checking environment..."' >> /app/startup_check.sh && \
    echo 'echo "PYTHONPATH: $PYTHONPATH"' >> /app/startup_check.sh && \
    echo 'echo "Current directory: $(pwd)"' >> /app/startup_check.sh && \
    echo 'echo "Files in jamba package: $(ls -la /app/src/jamba)"' >> /app/startup_check.sh && \
    echo 'echo "✓ All validation checks passed"' >> /app/startup_check.sh && \
    chmod +x /app/startup_check.sh

# Copy the startup wrapper script and make it executable
COPY src/runpod_entry.sh /app/runpod_entry.sh
RUN chmod +x /app/runpod_entry.sh

# Set RunPod specific environment variables
ENV RUNPOD_DEBUG_LEVEL=DEBUG
ENV RUNPOD_STOP_SIGNAL=SIGINT
ENV RUNPOD_TIMEOUT_SECONDS=900

# Healthcheck to verify container status
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; from jamba.jamba_model import JambaThreatModel; print('Health check passed')" || exit 1

# Use the startup script as the entrypoint
CMD ["/app/runpod_entry.sh"] 