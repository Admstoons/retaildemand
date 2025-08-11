from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
import requests
import io
import uvicorn
import traceback

# === CONFIG ===
MODEL_AND_ENCODERS_URL = (
    "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models/xgb_model.pkl"
)

model = None
encoders = {}
feature_columns = None  # Keep feature order


# === Preprocessing function with optional engineered features ===
def preprocess_optional_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Date features
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')

        if not {'Month_sin', 'Month_cos'}.issubset(df.columns):
            df['Month'] = df['Date'].dt.month
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

        if not {'Weekday_sin', 'Weekday_cos'}.issubset(df.columns):
            df['Weekday'] = df['Date'].dt.weekday
            df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
            df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)
    else:
        # If no Date column, set dummy features (optional)
        if not {'Month_sin', 'Month_cos'}.issubset(df.columns):
            df['Month_sin'] = 0
            df['Month_cos'] = 0
        if not {'Weekday_sin', 'Weekday_cos'}.issubset(df.columns):
            df['Weekday_sin'] = 0
            df['Weekday_cos'] = 0

    # Clean numeric columns
    for col in ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'Weekly_Sales']:
        if col in df.columns:
            df[col] = df[col].replace({',': '', '₹': '', '$': ''}, regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Lag and rolling features for Weekly_Sales
    if 'Weekly_Sales' in df.columns:
        df = df.sort_values(by='Date')
        if 'lag_1' not in df.columns:
            df['lag_1'] = df['Weekly_Sales'].shift(1)
        if 'lag_2' not in df.columns:
            df['lag_2'] = df['Weekly_Sales'].shift(2)
        if 'lag_3' not in df.columns:
            df['lag_3'] = df['Weekly_Sales'].shift(3)
        if 'rolling_mean_3' not in df.columns:
            df['rolling_mean_3'] = df['Weekly_Sales'].rolling(window=3).mean()
        if 'rolling_mean_7' not in df.columns:
            df['rolling_mean_7'] = df['Weekly_Sales'].rolling(window=7).mean()

    # Interaction feature example
    if 'discount_percentage' in df.columns and 'rating' in df.columns:
        if 'discount_rating_interaction' not in df.columns:
            df['discount_rating_interaction'] = df['discount_percentage'] * df['rating']

    # Drop irrelevant text columns (optional)
    irrelevant_cols = [
        'product_id', 'product_name', 'category', 'about_product',
        'user_id', 'user_name', 'review_id', 'review_title', 'review_content',
        'img_link', 'product_link'
    ]
    df.drop(columns=[col for col in irrelevant_cols if col in df.columns], inplace=True)

    df.dropna(inplace=True)  # Drop rows with NaNs created by lag/rolling shifts etc.

    return df


# === LIFESPAN: Load Model + Encoders ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoders, feature_columns
    print("🚀 Starting FastAPI and loading model + encoders...")
    try:
        # Download model file from Supabase
        print(f"Fetching model from: {MODEL_AND_ENCODERS_URL}")
        response = requests.get(MODEL_AND_ENCODERS_URL)

        if response.status_code != 200 or not response.content:
            raise RuntimeError(
                f"Failed to fetch model file (status {response.status_code})."
            )

        # Load dict containing model + encoders + features
        obj = joblib.load(io.BytesIO(response.content))

        if not isinstance(obj, dict):
            raise TypeError("Downloaded file is not a dict containing model & encoders.")

        model = obj.get("model")
        encoders = obj.get("encoders", {})
        feature_columns = obj.get("features", None)

        if model is None or not encoders:
            raise ValueError("Model or encoders missing in loaded file.")

        print(f"✅ Model loaded: {type(model)}")
        print(f"✅ Encoders loaded for columns: {list(encoders.keys())}")

        if feature_columns:
            print(f"✅ Feature columns preserved ({len(feature_columns)} features)")

    except Exception as e:
        print("❌ Failed to load model/encoders:", e)
        traceback.print_exc()
        model = None
        encoders = {}
        feature_columns = None

    yield
    print("🛑 Shutting down FastAPI")


app = FastAPI(lifespan=lifespan)

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Prediction Endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, encoders, feature_columns
    if model is None or not encoders:
        raise HTTPException(
            status_code=503, detail="Model or encoders not loaded. Check server logs."
        )

    try:
        # Read CSV
        df = pd.read_csv(file.file)

        # Preprocess input with optional engineered features
        df = preprocess_optional_engineered_features(df)

        # Apply label encoders
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except Exception as enc_err:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Encoding failed for column '{col}': {str(enc_err)}",
                    )
            else:
                raise HTTPException(
                    status_code=400, detail=f"Missing column in input: {col}"
                )

        # Match training column order
        if feature_columns:
            missing_cols = [c for c in feature_columns if c not in df.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required feature columns after preprocessing: {missing_cols}",
                )
            df = df[feature_columns]

        # Make predictions
        predictions = model.predict(df)

        return JSONResponse(
            content={
                "predictions": predictions.tolist(),
                "total_predictions": len(predictions)
            }
        )

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
