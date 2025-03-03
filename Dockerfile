FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ .

# Set environment variables
ENV AZURE_SUBSCRIPTION_ID=""
ENV AZURE_RESOURCE_GROUP=""
ENV AZURE_ML_WORKSPACE=""

# Run the application
CMD ["python", "data_ingestion.py"] 