# fastapi_app/inference.py
import os
import joblib
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import sys

from .schemas import OLXPredictionRequest, PredictionResponse

BASE_DIR = Path(__file__).resolve().parent
# Primary expected location (when container copies files to fastapi_app/models/trained)
package_model_path = BASE_DIR / "models" / "trained" / "house_price_best.pkl"
package_prep_path = BASE_DIR / "models" / "trained" / "preprocessor.pkl"

# Fallback to repository root (project-level models/trained)
repo_root = BASE_DIR.parent.parent
repo_model_path = repo_root / "models" / "trained" / "house_price_best.pkl"
repo_prep_path = repo_root / "models" / "trained" / "preprocessor.pkl"

# Choose default paths: prefer package-local, else repo-level
DEFAULT_MODEL_PATH = package_model_path if package_model_path.exists() else repo_model_path
DEFAULT_PREP_PATH = package_prep_path if package_prep_path.exists() else repo_prep_path

# Allow overriding via env vars
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
PREPROCESSOR_PATH = Path(os.getenv("PREPROCESSOR_PATH", str(DEFAULT_PREP_PATH)))

_model = None
_preproc = None

def _ensure_loaded():
    global _model, _preproc
    if _model is None or _preproc is None:
        # Some preprocessor objects may have been pickled when a helper
        # function (_make_interactions) was defined in a training script
        # run as __main__. During unpickling we must ensure that symbol
        # exists on the same module. Provide a safe fallback here.
        def _make_interactions(df):
            try:
                df = df.copy()
                if "LB" in df.columns and "LT" in df.columns:
                    df["LBxLT"] = df["LB"] * df["LT"]
            except Exception:
                pass
            return df

        # Inject into __main__ so pickle can find it if it was saved from a
        # script executed as __main__ previously.
        main_mod = sys.modules.get("__main__")
        if main_mod is not None and not hasattr(main_mod, "_make_interactions"):
            setattr(main_mod, "_make_interactions", _make_interactions)
        # Provide helpful errors when model files are missing
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        try:
            _preproc = joblib.load(PREPROCESSOR_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load preprocessor from {PREPROCESSOR_PATH}: {e}")

        try:
            _model = joblib.load(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

CSV_COLS = [
    "LB","LT","KM","KT","Kota/Kab","Provinsi",
    "harga_per_m2","ratio_bangunan_rumah","Type"
]

def _to_row(req: OLXPredictionRequest) -> dict:
    r = req.dict(by_alias=True)
    base = {c: r.get(c, None) for c in CSV_COLS}
    
    # Create features matching the training pipeline
    df = pd.DataFrame([base])
    
    # Basic features
    df['LBxLT'] = df['LB'] * df['LT']
    df['log_LB'] = np.log1p(df['LB'])
    df['log_LT'] = np.log1p(df['LT'])
    df['lb_x_km'] = df['LB'] * df['KM']
    df['lt_x_kt'] = df['LT'] * df['KT']
    df['ratio_lb_lt'] = df['LB'] / df['LT'].replace(0, np.nan)
    
    # Additional features that might be used by the model
    df['price_per_m2'] = df['harga_per_m2']
    df['ratio_bangunan_rumah'] = df['ratio_bangunan_rumah']
    
    return df.iloc[0].to_dict()

def predict_price(req: OLXPredictionRequest) -> PredictionResponse:
    _ensure_loaded()
    row_dict = _to_row(req)
    
    # Create DataFrame with all required columns
    df = pd.DataFrame([row_dict])
    
    # Make sure we have all the engineered features
    df['LBxLT'] = df['LB'] * df['LT']  # This matches the preprocessor's expected features
    df['log_LB'] = np.log1p(df['LB'])
    df['log_LT'] = np.log1p(df['LT'])
    df['lb_x_km'] = df['LB'] * df['KM']
    df['lt_x_kt'] = df['LT'] * df['KT']
    df['ratio_lb_lt'] = df['LB'] / df['LT'].replace(0, np.nan)
    
    # Drop raw features that aren't used by the model
    # but keep those needed by preprocessor
    cols_to_keep = ['LB', 'LT', 'KM', 'KT', 'Kota/Kab', 'Provinsi', 'Type',
                    'LBxLT', 'log_LB', 'log_LT', 'lb_x_km', 'lt_x_kt', 'ratio_lb_lt']
    df = df[cols_to_keep]
    
    X = _preproc.transform(df)
    y = _model.predict(X)
    price = float(y[0])
    return PredictionResponse(
        prediction=round(price, 2),
        prediction_time=datetime.utcnow().isoformat() + "Z"
    )
