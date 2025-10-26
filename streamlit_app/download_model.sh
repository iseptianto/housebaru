#!/usr/bin/env bash
set -e

echo "[init] preparing /app/models"
mkdir -p /app/models

download() {
  local url="$1"
  local out="$2"
  if [ -n "$url" ] && [ ! -f "$out" ]; then
    echo "[init] downloading $out ..."
    # -f: fail on http errors, -L: follow redirect, --retry: robust
    curl -fL --retry 3 --retry-delay 2 "$url" -o "$out"
  fi
}

download "$MODEL_URL" "/app/models/modelbaru.pkl"
download "$PREPROCESSOR_URL" "/app/models/barupreprocessor.pkl"

echo "[init] starting streamlit..."
exec "$@"
