FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set labels
LABEL maintainer="Jamba Threat Detection Team"
LABEL version="2.0.0" 
LABEL description="Jamba Threat Detection with GPU Support and Advanced Analytics"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    vim \
    graphviz \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create app directory structure
WORKDIR /app
RUN mkdir -p /app/src/jamba /app/utils /app/docs /app/models /app/data /app/logs \
    /app/data/raw /app/data/processed /app/data/balanced /app/data/experiments \
    /app/experiments/training_logs /app/experiments/metrics /app/experiments/plots \
    /app/models/gguf

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.0.1 \
        numpy==1.24.3 \
        pandas==2.0.3 \
        scikit-learn==1.3.0 \
        matplotlib==3.7.1 \
        seaborn==0.12.2 \
        networkx==3.1 \
        azure-storage-blob==12.17.0 \
        requests==2.31.0 \
        runpod==0.10.0 \
        fastapi==0.100.0 \
        uvicorn==0.23.0 \
        python-dotenv==1.0.0 \
        tensorboard==2.13.0 \
        ray==2.5.1 \
        pytest==7.4.0 \
        pytest-cov==4.1.0 \
        black==23.7.0 \
        mypy==1.4.1 \
        jupyter==1.0.0 \
        tqdm==4.65.0 \
        ctransformers>=0.2.27 \
        llama-cpp-python>=0.2.11 \
        sentencepiece>=0.1.99 \
        transformers>=4.34.0 \
        huggingface-hub>=0.19.0

# Install GGUF conversion tools
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    make && \
    cp convert-pytorch-to-gguf /usr/local/bin/ && \
    cd .. && \
    rm -rf llama.cpp

# Copy the entire jamba package
COPY src/jamba /app/src/jamba/
COPY src/handler.py /app/handler.py
COPY src/runpod_entry.sh /app/runpod_entry.sh
COPY src/startup_check.sh /app/startup_check.sh
COPY setup.py /app/setup.py
COPY README.md /app/README.md
COPY requirements.txt /app/requirements.txt
COPY requirements-optional.txt /app/requirements-optional.txt

# Make scripts executable
RUN chmod +x /app/runpod_entry.sh /app/startup_check.sh

# Install the jamba package
RUN pip install -e .

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_DIR=/app \
    MODEL_DIR=/app/models \
    LOGS_DIR=/app/logs \
    DATA_DIR=/app/data \
    DEBUG_MODE=false \
    GGUF_MODELS_DIR=/app/models/gguf

# Create directories for model artifacts
RUN mkdir -p /app/models/checkpoints \
    /app/models/exports \
    /app/models/configs \
    /app/models/gguf/q4_k_m \
    /app/models/gguf/q5_k_m \
    /app/models/gguf/q8_0

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; from jamba.jamba_model import JambaThreatModel; print('Health check passed')" || exit 1

# Default command
CMD ["/app/runpod_entry.sh"] 