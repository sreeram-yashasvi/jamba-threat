FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies in two steps
RUN pip install --no-cache-dir runpod==0.10.0
RUN pip install --no-cache-dir \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    pyarrow==12.0.1 \
    fastparquet==2023.7.0 \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    tqdm==4.65.0

# Create a Python module structure
RUN mkdir -p /app/jamba_model
RUN touch /app/jamba_model/__init__.py

# Copy model files
COPY src/jamba_model.py /app/jamba_model/model.py
COPY src/handler.py /app/handler.py

# Set environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set entrypoint
ENTRYPOINT ["python", "-u", "handler.py"] 