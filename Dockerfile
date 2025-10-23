# Base image untuk dependensi Python umum
FROM python:3.9-slim as base
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# First install core numeric libraries and ML dependencies
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==1.5.3 \
    scikit-learn==1.2.2 \
    xgboost==1.7.3 \
    mlflow==2.8.0

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Stage untuk FastAPI app
FROM base as fastapi
WORKDIR /app

COPY src/api/ ./fastapi_app/
RUN touch fastapi_app/__init__.py

# Create necessary directories
RUN mkdir -p /app/fastapi_app/models/trained

# MLflow setup terintegrasi
RUN mkdir -p /mlflow && \
    ln -s /mlflow/mlflow.db /app/mlflow.db && \
    ln -s /mlflow/artifacts /app/mlruns

ENV PYTHONPATH=/app
EXPOSE 8000

# Default command untuk FastAPI
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
