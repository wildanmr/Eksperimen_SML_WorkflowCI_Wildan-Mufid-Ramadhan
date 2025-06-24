# Dockerfile for Diabetes Prediction Model
FROM python:3.9-slim

LABEL maintainer="wildanmr"
LABEL description="Diabetes Prediction Model with MLflow"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY conda.yaml .
COPY requirements.txt .

# Install MLflow and dependencies
RUN pip install --no-cache-dir \
    -r requirements.txt

# Copy model artifacts
COPY saved_models/ ./saved_models/
COPY artifacts/ ./artifacts/

# Copy prediction script
COPY predict.py .

# Create health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8080/health || exit 1' > /health_check.sh && \
    chmod +x /health_check.sh

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /health_check.sh

# Run the prediction service
CMD ["python", "predict.py"]