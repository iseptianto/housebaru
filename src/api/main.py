# fastapi_app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ---- Fallback utk pipeline lama yang expect fastapi_app.main._make_interactions
def _make_interactions(df):
    """
    Implementasi minimal agar unpickle FunctionTransformer tidak error.
    Ganti sesuai logika training aslinya bila ada.
    """
    try:
        df = df.copy()
        if "LB" in df.columns and "LT" in df.columns:
            df["LBxLT"] = df["LB"] * df["LT"]     # contoh interaksi umum
            # jika dulu ada rasio: df["LB_per_LT"] = df["LB"] / df["LT"].replace(0, pd.NA)
    except Exception:
        pass
    return df
# ---- end fallback

from .schemas import OLXPredictionRequest, PredictionResponse
from .inference import predict_price

app = FastAPI(title="House Price Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: OLXPredictionRequest):
    try:
        return predict_price(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app.main:app", host="0.0.0.0", port=8000, reload=False)
