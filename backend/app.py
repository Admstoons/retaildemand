# fastapi_app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import io
import numpy as np
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
# Robust preprocessing function
# =========================
def preprocess_features(df: pd.DataFrame):
    df = df.copy()
    
    # -------------------------
    # Identify Date column
    # -------------------------
    date_column = None
    for col in df.columns:
        if col.lower() in ['date', 'invoicedate', 'unnamed: 0', 'unnamed_column'] or \
           df[col].astype(str).str.match(r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}').any() or \
           df[col].astype(str).str.match(r'\d{4}-\d{2}-\d{2}').any() or \
           df[col].astype(str).str.match(r'\d{2}-\d{2}-\d{4}').any():
            date_column = col
            break

    if date_column:
        df = df.rename(columns={date_column: 'Date'})
    else:
        if {"Year", "Month", "Day"}.issubset(df.columns):
            df["Date"] = pd.to_datetime(df[["Year","Month","Day"]], errors="coerce")
        else:
            raise ValueError("CSV must contain 'Date' or Year/Month/Day columns.")

    # -------------------------
    # Robust Date Parsing
    # -------------------------
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    date_formats = [
        "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%Y-%m-%dT%H:%M:%S.%f"
    ]
    for fmt in date_formats:
        mask = df['Date'].isnull()
        if mask.any():
            df.loc[mask, 'Date'] = pd.to_datetime(df.loc[mask, 'Date'], format=fmt, errors='coerce')

    if df['Date'].isnull().any():
        df = df.dropna(subset=['Date']).reset_index(drop=True)

    # -------------------------
    # Time-based features
    # -------------------------
    df['Year'] = df['Date'].dt.year.astype("Int64")
    df['Month'] = df['Date'].dt.month.astype("Int64")
    df['Day'] = df['Date'].dt.day.astype("Int64")
    df['Weekday'] = df['Date'].dt.weekday.astype("Int64")
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype("Int64")
    df['DayOfYear'] = df['Date'].dt.dayofyear.astype("Int64")

    # Cyclical features
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
    df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

    # Period markers
    df['Is_Month_Start'] = df['Date'].dt.is_month_start.astype(int)
    df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
    df['Is_Quarter_Start'] = df['Date'].dt.is_quarter_start.astype(int)
    df['Is_Quarter_End'] = df['Date'].dt.is_quarter_end.astype(int)
    df['Is_Year_Start'] = df['Date'].dt.is_year_start.astype(int)
    df['Is_Year_End'] = df['Date'].dt.is_year_end.astype(int)

    # -------------------------
    # Lag & rolling features
    # -------------------------
    lag_cols = ["lag_1","lag_2","lag_3","lag_7"]
    roll_cols = ["rolling_mean_3","rolling_mean_7","rolling_mean_14","rolling_mean_28","rolling_std_7"]

    if "Weekly_Sales" in df.columns:
        df["lag_1"] = df["Weekly_Sales"].shift(1).fillna(0)
        df["lag_2"] = df["Weekly_Sales"].shift(2).fillna(0)
        df["lag_3"] = df["Weekly_Sales"].shift(3).fillna(0)
        df["lag_7"] = df["Weekly_Sales"].shift(7).fillna(0)

        df["rolling_mean_3"] = df["Weekly_Sales"].rolling(window=3).mean().fillna(0)
        df["rolling_mean_7"] = df["Weekly_Sales"].rolling(window=7).mean().fillna(0)
        df["rolling_mean_14"] = df["Weekly_Sales"].rolling(window=14).mean().fillna(0)
        df["rolling_mean_28"] = df["Weekly_Sales"].rolling(window=28).mean().fillna(0)
        df["rolling_std_7"] = df["Weekly_Sales"].rolling(window=7).std().fillna(0)
    else:
        for col in lag_cols + roll_cols:
            df[col] = 0

    return df

# =========================
# Common prediction function
# =========================
def run_prediction(df: pd.DataFrame):
    df = preprocess_features(df)

    # Apply encoders
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str).fillna("Unknown"))

    # Apply scaler if exists
    if scaler:
        df[df.columns] = scaler.transform(df[df.columns])

    # Ensure all model features exist
    expected_features = model.get_booster().feature_names
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    X = df[expected_features]
    preds = model.predict(X)

    # Metrics if ground truth exists
    metrics = {}
    if "Weekly_Sales" in df.columns:
        y_true = df["Weekly_Sales"].values
        metrics = {
            "mae": float(mean_absolute_error(y_true, preds)),
            "mse": float(mean_squared_error(y_true, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
            "r2": float(r2_score(y_true, preds)),
            "mape": float(np.mean(np.abs((y_true - preds)/y_true)) * 100),
        }

    # Prepare output with actual and predicted
    output_df = df.copy()
    output_df["Actual_Weekly_Sales"] = df["Weekly_Sales"] if "Weekly_Sales" in df.columns else None
    output_df["Predicted_Weekly_Sales"] = preds

    return {
        "total_predictions": len(preds),
        "predictions": output_df.to_dict(orient="records"),
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
        return {"error": str(e), "trace": traceback.format_exc()}

@app.post("/predict/url")
async def predict_url(input_data: UrlInput):
    try:
        resp = requests.get(input_data.csv_url)
        resp.raise_for_status()
        df = pd.read_csv(io.BytesIO(resp.content))
        return run_prediction(df)
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
