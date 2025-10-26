FROM python:3.11-slim

WORKDIR /app

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.sh /usr/local/bin/download_model.sh
RUN chmod +x /usr/local/bin/download_model.sh
ENTRYPOINT ["download_model.sh"]
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# copy source
COPY . .

# copy model (kalau folder models ada di root repo, ganti pathnya)
# Jika model berada di root repo: gunakan COPY ../../models ... (kurang nyaman)
# Saran: pindahkan folder models ke dalam streamlit_app/ agar COPY ini simpel:
# repo/
# ├─ streamlit_app/
# │   ├─ app.py
# │   ├─ requirements.txt
# │   ├─ ...
# │   └─ models/  <-- taruh di sini
# └─ models/ (hapus / pindahkan)

# contoh jika models sudah diletakkan di dalam streamlit_app/
RUN mkdir -p /app/models
COPY models/* /app/models/

ENV MODEL_PATH=/app/models/modelbaru.pkl
ENV PREPROCESSOR_PATH=/app/models/barupreprocessor.pkl

EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

