# fastapi_app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# Unpack the bundle (model + encoders + scaler)
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
# Prediction endpoint
# =========================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load dataset
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # === Apply SAME preprocessing as training ===
        # Categorical encoding
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str).fillna("Unknown"))

        # Lag features / rolling mean (example)
        if "sales" in df.columns:
            df["lag_1"] = df["sales"].shift(1).fillna(0)
            df["rolling_3"] = df["sales"].rolling(window=3).mean().fillna(0)

        # Scaling (if scaler exists)
        if scaler:
            df[df.columns] = scaler.transform(df[df.columns])

        # Ensure correct feature order
        expected_features = model.get_booster().feature_names
        X = df[expected_features]

        # Predictions
        preds = model.predict(X)

        # Metrics (if ground truth available)
        metrics = {}
        if "sales" in df.columns:  # your target column
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

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
