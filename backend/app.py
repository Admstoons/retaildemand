# fastapi_app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import io
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Load model + preprocessing objects
model = joblib.load("xgb_model_with_encoders.pkl")
encoders = joblib.load("encoders.pkl")   # save these from training
scaler = joblib.load("scaler.pkl")       # if you used scaling

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Load dataset
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # === Apply SAME preprocessing as training ===
        # Example: categorical encoding
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col].astype(str).fillna("Unknown"))

        # Example: lag features / rolling mean
        if "sales" in df.columns:
            df["lag_1"] = df["sales"].shift(1).fillna(0)
            df["rolling_3"] = df["sales"].rolling(window=3).mean().fillna(0)

        # Example: scaling
        if scaler:
            df[df.columns] = scaler.transform(df[df.columns])

        # Ensure same feature order
        expected_features = model.get_booster().feature_names
        X = df[expected_features]

        # Make predictions
        preds = model.predict(X)

        # Compute metrics (if actual values available)
        metrics = {}
        if "sales" in df.columns:  # replace with your target column
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
