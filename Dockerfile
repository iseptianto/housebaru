# Base image untuk dependensi Python umum
FROM python:3.11-slim as base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install numpy first to ensure correct version
RUN pip install numpy==1.26.0

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
