import os
import joblib
from pathlib import Path
from datetime import datetime
import pandas as pd

from fastapi_app.schemas import OLXPredictionRequest, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "trained" / "house_price_best.pkl"
DEFAULT_PREP_PATH  = BASE_DIR / "models" / "trained" / "preprocessor.pkl"

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", str(DEFAULT_PREP_PATH)))

_model = None
_preproc = None

def _ensure_loaded():
    global _model, _preproc
    if _model is None or _preproc is None:
        _preproc = joblib.load(PREPROCESSOR_PATH)
        _model = joblib.load(MODEL_PATH)

CSV_COLS = [
    "LB","LT","KM","KT","Kota/Kab","Provensi",
    "harga_per_m2","ratio_bangunan ruma","tyype"
]

def _to_row(req: OLXPredictionRequest) -> dict:
    # pakai alias agar nama kolom persis
    r = req.dict(by_alias=True)
    row = {c: r.get(c, None) for c in CSV_COLS}
    return row

def predict_price(req: OLXPredictionRequest) -> PredictionResponse:
    _ensure_loaded()
    df = pd.DataFrame([_to_row(req)], columns=CSV_COLS)
    X = _preproc.transform(df)
    y = _model.predict(X)
    price = float(y[0])
    return PredictionResponse(
        prediction=round(price, 2),
        prediction_time=datetime.utcnow().isoformat() + "Z"
    )
