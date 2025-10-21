FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy source sebagai package "fastapi_app"
COPY src/api/ ./fastapi_app/
RUN touch fastapi_app/__init__.py

# opsional: copy model kalau mau dibundel (lebih baik pakai volume)
# COPY models/trained/ ./fastapi_app/models/trained/

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn","fastapi_app.main:app","--host","0.0.0.0","--port","8000"]
