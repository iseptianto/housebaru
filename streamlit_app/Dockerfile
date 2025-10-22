FROM python:3.11-slim
WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
 && pip install --timeout 300 --retries 10 -r requirements.txt

COPY app.py translations.py ./
EXPOSE 8501
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501"]
