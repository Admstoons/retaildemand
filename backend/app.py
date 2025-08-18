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
import traceback

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
# Feature engineering function
# =========================
def preprocess_features(df: pd.DataFrame):
    # Ensure Date column exists
    if "Date" not in df.columns:
        if {"Year", "Month", "Day"}.issubset(df.columns):
            df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
        else:
            raise ValueError("CSV must contain either 'Date' column or Year/Month/Day columns.")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Time-based features
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Weekday"] = df["Date"].dt.weekday
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # Cyclical encoding
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
    df["Weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)

    # Period markers
    df["Is_Month_Start"] = df["Date"].dt.is_month_start.astype(int)
    df["Is_Month_End"] = df["Date"].dt.is_month_end.astype(int)
    df["Is_Quarter_Start"] = df["Date"].dt.is_quarter_start.astype(int)
    df["Is_Quarter_End"] = df["Date"].dt.is_quarter_end.astype(int)
    df["Is_Year_Start"] = df["Date"].dt.is_year_start.astype(int)
    df["Is_Year_End"] = df["Date"].dt.is_year_end.astype(int)

    # Lag & rolling features (require "sales")
    if "sales" in df.columns:
        df["lag_1"] = df["sales"].shift(1).fillna(0)
        df["lag_2"] = df["sales"].shift(2).fillna(0)
        df["lag_3"] = df["sales"].shift(3).fillna(0)
        df["lag_7"] = df["sales"].shift(7).fillna(0)

        df["rolling_mean_3"] = df["sales"].rolling(window=3).mean().fillna(0)
        df["rolling_mean_7"] = df["sales"].rolling(window=7).mean().fillna(0)
        df["rolling_mean_14"] = df["sales"].rolling(window=14).mean().fillna(0)
        df["rolling_mean_28"] = df["sales"].rolling(window=28).mean().fillna(0)

        df["rolling_std_7"] = df["sales"].rolling(window=7).std().fillna(0)

    return df

# =========================
# Common prediction function
# =========================
def run_prediction(df: pd.DataFrame):
    # Preprocess
    df = preprocess_features(df)

    # Apply encoders
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str).fillna("Unknown"))

    # Apply scaler
    if scaler:
        df[df.columns] = scaler.transform(df[df.columns])

    # Match training features
    expected_features = model.get_booster().feature_names
    X = df[expected_features]

    preds = model.predict(X)

    # Metrics if ground truth exists
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
@app.post("/predict")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        return run_prediction(df)
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


@app.post("/predict")
async def predict_url(input_data: UrlInput):
    try:
        resp = requests.get(input_data.csv_url)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content))
        return run_prediction(df)
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
