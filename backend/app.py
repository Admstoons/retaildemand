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

# URL of combined model + encoders file in Supabase
MODEL_AND_ENCODERS_URL = (
    "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models/xgb_model.pkl"
)

# Global variables
model = None
encoders = {}
TARGET_COLUMN = "Weekly_Sales"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model & encoders when API starts."""
    global model, encoders
    try:
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
        model, encoders = None, {}
    yield

app = FastAPI(lifespan=lifespan)

# Enable CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class FileURLInput(BaseModel):
    file_url: str

@app.get("/")
def root():
    return {"message": "Retail Demand Forecast API is running"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

def create_lag_features(df: pd.DataFrame, lags=[1, 2, 3]) -> pd.DataFrame:
    """Create lag & rolling mean features for the target column."""
    for lag in lags:
        df[f"{TARGET_COLUMN}_lag_{lag}"] = df[TARGET_COLUMN].shift(lag)
    df[f"{TARGET_COLUMN}_rolling_mean_3"] = df[TARGET_COLUMN].rolling(window=3).mean()
    df[f"{TARGET_COLUMN}_rolling_mean_7"] = df[TARGET_COLUMN].rolling(window=7).mean()
    return df

@app.post("/predict")
def predict_from_file(data: FileURLInput):
    """Predict sales given a CSV file URL."""
    if model is None or not encoders:
        raise HTTPException(status_code=503, detail="Model or encoders not loaded")

    try:
        # Load CSV from provided URL
        response = requests.get(data.file_url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")
        if "Date" not in df.columns:
            raise HTTPException(status_code=400, detail="'Date' column is required")

        # Convert and sort by Date
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

        # Extract date features
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["Weekday"] = df["Date"].dt.weekday

        # Add cyclical seasonality features
        df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
        df["Weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
        df["Weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)

        # Verify target column exists for lagging
        if TARGET_COLUMN not in df.columns:
            raise HTTPException(status_code=400, detail=f"'{TARGET_COLUMN}' column is required")

        # Lag & rolling features
        df = create_lag_features(df)

        # Encode categorical columns with saved encoders
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
                df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else "Unknown")
                if "Unknown" not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, "Unknown")
                df[col] = encoder.transform(df[col].astype(str))

        # Drop rows with NaNs from lag features
        df = df.dropna().reset_index(drop=True)
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid rows after preprocessing")

        # Keep Date for output
        output_dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()

        # Prepare features for model
        X = df.drop(columns=[TARGET_COLUMN, "Date"], errors="ignore")

        # Predict
        predictions = model.predict(X)
        actual_values = df[TARGET_COLUMN].astype(float).tolist()

        return {
            "actual_values": [float(v) for v in actual_values],
            "predicted_values": [float(v) for v in predictions],
            "dates": {str(i): output_dates[i] for i in range(len(predictions))}
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Request error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
