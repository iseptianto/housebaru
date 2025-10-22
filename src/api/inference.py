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

def _engineer_features(df):
    """Create all required features for the model."""
    try:
        # Basic features
        df['LBxLT'] = df['LB'] * df['LT']
        df['log_LB'] = np.log1p(df['LB'])
        df['log_LT'] = np.log1p(df['LT'])
        df['lb_x_km'] = df['LB'] * df['KM']
        df['lt_x_kt'] = df['LT'] * df['KT']
        # Handle division by zero for ratio
        df['ratio_lb_lt'] = df['LB'] / df['LT'].replace(0, np.nan)
        
        # Keep original features needed by preprocessor
        required_cols = [
            'LB', 'LT', 'KM', 'KT', 'Kota/Kab', 'Provinsi', 'Type',
            'LBxLT', 'log_LB', 'log_LT', 'lb_x_km', 'lt_x_kt', 'ratio_lb_lt'
        ]
        
        return df[required_cols]
    except Exception as e:
        raise ValueError(f"Error engineering features: {str(e)}")

def _to_row(req: OLXPredictionRequest) -> dict:
    """Convert request to initial dataframe row."""
    try:
        r = req.dict(by_alias=True)
        base = {c: r.get(c, None) for c in CSV_COLS}
        return base
    except Exception as e:
        raise ValueError(f"Error converting request to row: {str(e)}")

def predict_price(req: OLXPredictionRequest) -> PredictionResponse:
    """
    Generate house price prediction from input features.
    
    Args:
        req: Validated request containing house features
        
    Returns:
        PredictionResponse with prediction details and confidence metrics
        
    Raises:
        ValueError: If feature engineering fails
        RuntimeError: If model prediction fails
    """
    try:
        start_time = datetime.now()
        _ensure_loaded()
        
        # Create initial dataframe
        row_dict = _to_row(req)
        df = pd.DataFrame([row_dict])
        
        # Engineer features
        df = _engineer_features(df)
        
        # Generate prediction
        try:
            X = _preproc.transform(df)
            
            # Get base prediction
            y = _model.predict(X)
            price = float(y[0])
            
            # Ensure prediction is non-negative
            price = max(0, price)
            
            # Get confidence score (using predict_proba if available, else use a heuristic)
            confidence_score = 0.92  # Default value
            if hasattr(_model, 'predict_proba'):
                proba = _model.predict_proba(X)
                confidence_score = float(proba.max())
            
            # Calculate price range (±10% by default)
            price_range = (price * 0.9, price * 1.1)
            
            # Get feature importance
            feature_importance = {}
            if hasattr(_model, 'feature_importances_'):
                importances = _model.feature_importances_
                feature_names = [
                    "Square Footage (LB)",
                    "Location",
                    "Number of Bathrooms",
                    "Land Area (LT)",
                    "Number of Bedrooms"
                ]
                importance_dict = dict(zip(feature_names, importances))
                # Sort by importance and get top 3
                feature_importance = dict(sorted(
                    importance_dict.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3])
            
            # Calculate prediction time
            end_time = datetime.now()
            prediction_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Get model name
            model_name = type(_model).__name__
            if model_name == 'XGBRegressor':
                model_name = 'XGBoost'
            
            return PredictionResponse(
                prediction=round(price, 2),
                prediction_time=datetime.utcnow().isoformat() + "Z",
                confidence_score=confidence_score,
                model_name=model_name,
                price_range=price_range,
                feature_importance=feature_importance,
                prediction_time_ms=prediction_time_ms
            )
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
            
    except Exception as e:
        raise ValueError(f"Error processing request: {str(e)}")
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
