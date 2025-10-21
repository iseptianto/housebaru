import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi_app.schemas import OLXPredictionRequest, PredictionResponse

# Default path (bisa override via ENV)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "trained" / "house_price_best.pkl"
DEFAULT_PREP_PATH  = BASE_DIR / "models" / "trained" / "preprocessor.pkl"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
PREP_PATH  = Path(os.getenv("PREPROCESSOR_PATH", str(DEFAULT_PREP_PATH)))

_model = None
_preproc = None

def _ensure_loaded():
    """Lazy load model & preprocessor sekali saja."""
    global _model, _preproc
    if _model is not None and _preproc is not None:
        return
    try:
        _preproc = joblib.load(PREP_PATH)
        _model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Gagal load model/preprocessor: {e}")

# Skema kolom yang diharapkan oleh preprocessor/model
CSV_COLS = [
    "LB","LT","KM","KT","Kota/Kab","Provensi",
    "harga_per_m2","ratio_bangunan ruma","tyype"
]
NUM_COLS = {"LB","LT","KM","KT","harga_per_m2","ratio_bangunan ruma"}
CAT_COLS = {"Kota/Kab","Provensi","tyype"}

def _to_row(req: OLXPredictionRequest) -> Dict[str, Any]:
    # pakai alias agar key persis dengan nama kolom CSV
    r = req.dict(by_alias=True)
    out: Dict[str, Any] = {}
    for c in CSV_COLS:
        v = r.get(c, None)
        if c in NUM_COLS:
            if v is None or v == "":
                out[c] = np.nan
            else:
                out[c] = float(v) if c not in {"KM","KT"} else int(v)
        elif c in CAT_COLS:
            out[c] = "" if v is None else str(v).strip()
        else:
            out[c] = v
    return out

def predict_price(request: OLXPredictionRequest) -> PredictionResponse:
    _ensure_loaded()
    row = _to_row(request)
    df = pd.DataFrame([row], columns=CSV_COLS)

    X = _preproc.transform(df)
    y = _model.predict(X)

    # kalau model kamu train di log1p, ubah ke expm1:
    # price = float(np.expm1(y[0]))
    price = float(y[0])

    return PredictionResponse(
        prediction=round(price, 2),
        prediction_time=datetime.utcnow().isoformat() + "Z"
    )
