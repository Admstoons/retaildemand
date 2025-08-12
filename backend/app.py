from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import requests
import io
import traceback
import uvicorn

# === CONFIG ===
MODEL_AND_ENCODERS_URL = (
    "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models/xgb_model_with_encoders.pkl"
)

model = None
encoders = {}
feature_columns = None  # Keep feature order

# === Request / Response Schemas ===
class PredictRequest(BaseModel):
    file_url: Optional[str] = None  # accept file_url in JSON; optional because multipart is also supported

class PredictResponse(BaseModel):
    dates: List[str]
    actual_price: List[Optional[float]]
    predicted_price: List[float]
    total_predictions: int


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
        # Accept a variety of date formats but try the project's expected format first
        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
        # fallback: try parsing any format for rows that stayed NaT
        if df['Date'].isna().any():
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year.fillna(required_features['Year']).astype(int)
        df['Month'] = df['Date'].dt.month.fillna(required_features['Month']).astype(int)
        df['Day'] = df['Date'].dt.day.fillna(required_features['Day']).astype(int)
        df['Weekday'] = df['Date'].dt.weekday.fillna(required_features['Weekday']).astype(int)
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
        df = df.sort_values(by='Date').reset_index(drop=True)
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
        if critical_cols:
            before = len(df)
            df.dropna(subset=critical_cols, inplace=True)
            after = len(df)
            print(f"Dropped {before - after} rows with NaNs in critical features")

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
        response = requests.get(MODEL_AND_ENCODERS_URL, timeout=30)

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

        if model is None or encoders is None:
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
@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(None)):
    """
    Supports:
      - multipart/form-data with a 'file' CSV upload (UploadFile)
      - JSON body { "file_url": "<public_supabase_url>" }
    """
    global model, encoders, feature_columns

    if model is None or not encoders:
        raise HTTPException(
            status_code=503, detail="Model or encoders not loaded. Check server logs."
        )

    try:
        # ---------- 1) Obtain dataframe either from upload or from URL ----------
        if file is not None:
            # Received multipart upload
            print("Received multipart file upload")
            df = pd.read_csv(file.file)
        else:
            # Try reading JSON body for file_url
            try:
                body = await request.json()
            except Exception:
                body = {}
            file_url = body.get("file_url") if isinstance(body, dict) else None
            if not file_url:
                raise HTTPException(
                    status_code=400,
                    detail="No file uploaded and no 'file_url' provided in request body."
                )
            print(f"Fetching CSV from URL: {file_url}")
            csv_resp = requests.get(file_url, timeout=30)
            if csv_resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download CSV from URL (status {csv_resp.status_code})")
            df = pd.read_csv(io.StringIO(csv_resp.text))

        print(f"Loaded input CSV with {len(df)} rows and columns: {list(df.columns)}")

        # ---------- 2) Save original date and actual_price columns for output ----------
        # date candidates
        date_col = None
        for candidate in ["Date", "date", "DATE"]:
            if candidate in df.columns:
                date_col = candidate
                break
        dates = df[date_col].astype(str).tolist() if date_col else [str(i) for i in range(len(df))]

        # actual price column candidates
        actual_col = None
        for candidate in ["actual_price", "Actual", "actual", "Weekly_Sales", "weekly_sales"]:
            if candidate in df.columns:
                actual_col = candidate
                break
        if actual_col:
            # coerce to numeric, keep NAs as None
            actual_prices = pd.to_numeric(df[actual_col].replace({',': ''}, regex=True), errors='coerce').tolist()
            # convert nan -> None for JSON friendly
            actual_prices = [None if (pd.isna(x)) else float(x) for x in actual_prices]
        else:
            actual_prices = [None] * len(df)

        # ---------- 3) Preprocess ----------
        df_processed = preprocess_optional_engineered_features(df)

        # ---------- 4) Apply label encoders ----------
        # encoders expected to be a dict: { column_name: encoder_obj }
        for col, encoder in encoders.items():
            if col in df_processed.columns:
                try:
                    # convert to string before transform to match training pipeline
                    df_processed[col] = encoder.transform(df_processed[col].astype(str))
                    print(f"Encoded column: {col}")
                except Exception as enc_err:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Encoding failed for column '{col}': {str(enc_err)}",
                    )
            else:
                # Only add a zero column if the encoder existed in training
                if encoder is not None:
                    df_processed[col] = 0
                    print(f"Column {col} not found, set to default value 0")

        # ---------- 5) Match training column order ----------
        if feature_columns:
            missing_cols = [c for c in feature_columns if c not in df_processed.columns]
            if missing_cols:
                for c in missing_cols:
                    df_processed[c] = 0
                print(f"Added missing columns with zeros: {missing_cols}")
            # Reorder: select only feature_columns (this will drop extra columns)
            df_processed = df_processed[feature_columns]
            print(f"Reordered columns to match training: {list(df_processed.columns)}")

        # ---------- 6) Predict ----------
        predictions = model.predict(df_processed)
        # Ensure predictions are plain Python floats
        predicted_list = [float(x) for x in predictions]

        # ---------- 7) Return result ----------
        return PredictResponse(
            dates=[str(d) for d in dates],
            actual_price=actual_prices,
            predicted_price=predicted_list,
            total_predictions=len(predicted_list)
        )

    except HTTPException:
        # re-raise HTTPExceptions unchanged
        raise
    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


# === Feature Importance Endpoint ===
@app.get("/feature_importance")
async def get_feature_importance():
    if model is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        fi = dict(zip(feature_columns, model.feature_importances_.tolist()))
        return JSONResponse(content={"feature_importance": fi})
    except Exception as e:
        print("❌ Feature importance error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
