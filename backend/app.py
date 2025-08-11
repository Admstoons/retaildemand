from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import requests
from io import StringIO, BytesIO
import numpy as np
import traceback

MODEL_AND_ENCODERS_URL = (
    "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models/xgb_model.pkl"
)

model = None
encoders = {}
TARGET_COLUMN = "Weekly_Sales"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoders
    try:
        print("📦 Downloading model & encoders...")
        response = requests.get(MODEL_AND_ENCODERS_URL)
        response.raise_for_status()
        data = joblib.load(BytesIO(response.content))
        model = data.get("model")
        encoders = data.get("encoders", {})
        if model is None or not encoders:
            raise ValueError("Model or encoders missing in loaded file")
        print("✅ Model and encoders loaded successfully")
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        traceback.print_exc()
        model, encoders = None, {}
    yield

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FileURLInput(BaseModel):
    file_url: str

@app.get("/")
def root():
    return {"message": "Retail Demand Forecast API is running"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

def create_lag_features(df: pd.DataFrame, lags=[1, 2, 3]) -> pd.DataFrame:
    for lag in lags:
        df[f"{TARGET_COLUMN}_lag_{lag}"] = df[TARGET_COLUMN].shift(lag)
    df[f"{TARGET_COLUMN}_rolling_mean_3"] = df[TARGET_COLUMN].rolling(window=3).mean()
    df[f"{TARGET_COLUMN}_rolling_mean_7"] = df[TARGET_COLUMN].rolling(window=7).mean()
    return df

@app.post("/predict")
def predict_from_file(data: FileURLInput):
    if model is None or not encoders:
        raise HTTPException(status_code=503, detail="Model or encoders not loaded")

    try:
        print(f"📂 Fetching CSV from {data.file_url}...")
        response = requests.get(data.file_url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        print(f"✅ CSV loaded with {len(df)} rows and {len(df.columns)} columns")

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")
        if "Date" not in df.columns:
            raise HTTPException(status_code=400, detail="'Date' column is required")
        if TARGET_COLUMN not in df.columns:
            raise HTTPException(status_code=400, detail=f"'{TARGET_COLUMN}' column is required")

        # Dates
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

        # Date features
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["Weekday"] = df["Date"].dt.weekday

        # Cyclical features
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
        df["Weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
        df["Weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)

        # Lag & rolling
        df = create_lag_features(df)

        # Encode categoricals
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
                df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else "Unknown")
                if "Unknown" not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, "Unknown")
                df[col] = encoder.transform(df[col].astype(str))

        # Drop rows with NaNs from lagging
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        print(f"🧹 Dropped {before - len(df)} rows due to NaNs from lag features")

        if df.empty:
            raise HTTPException(status_code=400, detail="No valid rows after preprocessing")

        output_dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()

        X = df.drop(columns=[TARGET_COLUMN, "Date"], errors="ignore")
        print(f"📊 Predicting on {X.shape[0]} rows and {X.shape[1]} features...")

        predictions = model.predict(X)
        actual_values = df[TARGET_COLUMN].astype(float).tolist()

        print("✅ Prediction complete")

        return {
            "actual_values": [float(v) for v in actual_values],
            "predicted_values": [float(v) for v in predictions],
            "dates": {str(i): output_dates[i] for i in range(len(predictions))}
        }

    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Request error: {e}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Server error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
