# fastapi_app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import io
import numpy as np
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# =========================
# Load bundled model
# =========================
MODEL_URL = "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models/xgb_model_with_encoders.pkl"

def load_from_url(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return joblib.load(io.BytesIO(resp.content))

bundle = load_from_url(MODEL_URL)
model = bundle["model"]
encoders = bundle.get("encoders", {})
scaler = bundle.get("scaler", None)

# =========================
# FastAPI setup
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic model for URL input
# =========================
class UrlInput(BaseModel):
    csv_url: str

# =========================
# Common prediction function
# =========================
def run_prediction(df: pd.DataFrame):
    # Apply SAME preprocessing as training
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str).fillna("Unknown"))

    if "sales" in df.columns:
        df["lag_1"] = df["sales"].shift(1).fillna(0)
        df["rolling_3"] = df["sales"].rolling(window=3).mean().fillna(0)

    if scaler:
        df[df.columns] = scaler.transform(df[df.columns])

    expected_features = model.get_booster().feature_names
    X = df[expected_features]

    preds = model.predict(X)

    # Metrics
    metrics = {}
    if "sales" in df.columns:
        y_true = df["sales"].values
        metrics = {
            "mae": float(mean_absolute_error(y_true, preds)),
            "mse": float(mean_squared_error(y_true, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
            "r2": float(r2_score(y_true, preds)),
            "mape": float(mean_absolute_percentage_error(y_true, preds) * 100),
        }

    return {
        "total_predictions": len(preds),
        "predictions": preds.tolist(),
        "performance_metrics": metrics,
    }

# =========================
# Endpoints
# =========================

@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        return run_prediction(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/url")
async def predict_url(input_data: UrlInput):
    try:
        resp = requests.get(input_data.csv_url)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content))
        return run_prediction(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
