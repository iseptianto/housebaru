from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fastapi_app.schemas import OLXPredictionRequest, PredictionResponse
from fastapi_app.inference import predict_price

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
