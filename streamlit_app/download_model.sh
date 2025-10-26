#!/usr/bin/env bash
set -e
mkdir -p /app/models
if [ ! -f /app/models/modelbaru.pkl ]; then
  echo "Downloading model..."
  curl -L "$MODEL_URL" -o /app/models/modelbaru.pkl
fi
if [ ! -f /app/models/barupreprocessor.pkl ]; then
  echo "Downloading preprocessor..."
  curl -L "$PREPROCESSOR_URL" -o /app/models/barupreprocessor.pkl
fi
exec "$@"
