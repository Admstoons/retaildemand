from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import joblib
import requests
import io
import traceback

# === CONFIG ===
MODEL_AND_ENCODERS_URL = (
    "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models/xgb_model_with_encoders.pkl"
)

model = None
encoders = {}
feature_columns = None  # Keep feature order

# === Preprocessing function with optional engineered features ===
def preprocess_optional_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("Starting preprocessing for prediction...")

    # Define required features with default values (matching training)
    required_features = {
        'Year': 2023,
        'Month': 1,
        'Day': 1,
        'Weekday': 0,
        'Month_sin': 0,
        'Month_cos': 0,
        'Weekday_sin': 0,
        'Weekday_cos': 0,
        'lag_1': 0,
        'lag_2': 0,
        'lag_3': 0,
        'rolling_mean_3': 0,
        'rolling_mean_7': 0,
        'discount_rating_interaction': 0
    }

    # Date features
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.weekday
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
        df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)
    else:
        print("No 'Date' column found, setting default date features")
        for feature, default_value in required_features.items():
            if feature not in df.columns:
                df[feature] = default_value

    # Clean numeric columns
    for col in ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'Weekly_Sales']:
        if col in df.columns:
            df[col] = df[col].replace({',': '', '₹': '', '$': ''}, regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Cleaned column: {col}")

    # Lag and rolling features for Weekly_Sales
    if 'Weekly_Sales' in df.columns and 'Date' in df.columns:
        df = df.sort_values(by='Date')
        df['lag_1'] = df['Weekly_Sales'].shift(1).fillna(0)
        df['lag_2'] = df['Weekly_Sales'].shift(2).fillna(0)
        df['lag_3'] = df['Weekly_Sales'].shift(3).fillna(0)
        df['rolling_mean_3'] = df['Weekly_Sales'].rolling(window=3).mean().fillna(0)
        df['rolling_mean_7'] = df['Weekly_Sales'].rolling(window=7).mean().fillna(0)
        print("Generated lag and rolling features")
    else:
        print("No 'Weekly_Sales' or 'Date' column, setting lag/rolling features to 0")
        for feature in ['lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_mean_7']:
            df[feature] = 0

    # Interaction feature
    if 'discount_percentage' in df.columns and 'rating' in df.columns:
        df['discount_rating_interaction'] = df['discount_percentage'] * df['rating']
        print("Generated interaction feature: discount_rating_interaction")
    else:
        df['discount_rating_interaction'] = 0

    # Drop irrelevant text columns
    irrelevant_cols = [
        'product_id', 'product_name', 'category', 'about_product',
        'user_id', 'user_name', 'review_id', 'review_title', 'review_content',
        'img_link', 'product_link'
    ]
    df.drop(columns=[col for col in irrelevant_cols if col in df.columns], inplace=True)
    print(f"Dropped columns: {[col for col in irrelevant_cols if col in df.columns]}")

    # Fill NaNs in engineered features
    for feature in required_features:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0)

    # Drop rows only if critical features are missing
    if feature_columns:
        critical_cols = [col for col in feature_columns if col in df.columns]
        df.dropna(subset=critical_cols, inplace=True)
        print(f"Dropped {len(df) - len(df.dropna(subset=critical_cols))} rows with NaNs in critical features")

    if df.empty:
        raise ValueError("Dataframe is empty after preprocessing")

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
        print(f"✅ Feature columns: {feature_columns}")

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
        print(f"Loaded input CSV with {len(df)} rows and columns: {list(df.columns)}")

        # Preprocess input with optional engineered features
        df = preprocess_optional_engineered_features(df)

        # Apply label encoders
        for col, encoder in encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                    print(f"Encoded column: {col}")
                except Exception as enc_err:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Encoding failed for column '{col}': {str(enc_err)}",
                    )
            elif encoder is not None:  # Only fill default if column was used in training
                df[col] = 0
                print(f"Column {col} not found, set to default value 0")

        # Match training column order
        if feature_columns:
            missing_cols = [c for c in feature_columns if c not in df.columns]
            if missing_cols:
                # Add missing columns with zeros
                for col in missing_cols:
                    df[col] = 0
                print(f"Added missing columns with zeros: {missing_cols}")
            df = df[feature_columns]
            print(f"Reordered columns to match training: {list(df.columns)}")

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
