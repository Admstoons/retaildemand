from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib
import requests
import io
import traceback
import uvicorn
import logging

# ========= CONFIG =========
MODEL_AND_ENCODERS_URL = (
    "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models/xgb_model_with_encoders.pkl"
)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/vnd.ms-excel",
    "application/octet-stream",  # some clients send this
}

# ========= GLOBALS =========
model = None
encoders: Dict[str, object] = {}
feature_columns: Optional[List[str]] = None  # Keep training feature order

# ========= LOGGING =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("inference")

# ========= SCHEMAS =========
class PredictRequest(BaseModel):
    file_url: Optional[str] = None  # also support multipart 'file'

class PredictResponse(BaseModel):
    dates: List[str]
    actual_price: List[Optional[float]]
    predicted_price: List[float]
    total_predictions: int
    performance_metrics: Optional[Dict[str, float]] = None


# ========= UTILITIES =========
def read_csv_content(content: bytes) -> pd.DataFrame:
    """Robust CSV reader for uploaded bytes or downloaded content."""
    if len(content) == 0:
        raise ValueError("Uploaded/Downloaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError(f"File too large. Max allowed is {MAX_UPLOAD_BYTES} bytes.")

    # Try a few encodings
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(io.StringIO(content.decode(enc)))
        except Exception:
            continue
    # Last resort: pandas can sometimes handle BytesIO directly
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")


def compute_metrics(actuals: List[Optional[float]], preds: List[float]) -> Optional[Dict[str, float]]:
    """Compute MAE, MSE, RMSE, R2, MAPE for rows where actuals are present."""
    valid_idx = [i for i, x in enumerate(actuals) if x is not None]
    if not valid_idx:
        return None

    y_true = np.array([actuals[i] for i in valid_idx], dtype=float)
    y_pred = np.array([preds[i] for i in valid_idx], dtype=float)

    abs_err = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_err))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - (np.sum((y_true - y_pred) ** 2) / ss_tot)) if ss_tot > 0 else 0.0

    # MAPE (ignoring zeros in actual)
    mask_nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100) if np.any(mask_nonzero) else 0.0

    return {
        "mae": round(mae, 2),
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
    }


def fast_label_encode(series: pd.Series, encoder) -> pd.Series:
    """
    Encode using sklearn LabelEncoder-like object.
    Unknown classes -> -1. Works vectorized.
    """
    classes = getattr(encoder, "classes_", None)
    if classes is None:
        # No classes_ -> fallback to zero
        return pd.Series(0, index=series.index, dtype="int64")

    mapping = {cls: int(i) for i, cls in enumerate(classes)}
    out = series.astype(str).map(mapping)
    return out.fillna(-1).astype("int64")


# ========= PREPROCESS =========
def preprocess_optional_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_features_defaults = {
        'Year': 2023, 'Month': 1, 'Day': 1, 'Weekday': 0,
        'WeekOfYear': 1, 'DayOfYear': 1,
        'Month_sin': 0, 'Month_cos': 0, 'Weekday_sin': 0, 'Weekday_cos': 0,
        'Is_Month_Start': 0, 'Is_Month_End': 0, 'Is_Quarter_Start': 0,
        'Is_Quarter_End': 0, 'Is_Year_Start': 0, 'Is_Year_End': 0,
        'lag_1': 0, 'lag_2': 0, 'lag_3': 0, 'lag_7': 0,
        'rolling_mean_3': 0, 'rolling_mean_7': 0, 'rolling_mean_14': 0,
        'rolling_mean_28': 0, 'rolling_std_7': 0
    }

    # Ensure required cols exist
    for feature, default_value in required_features_defaults.items():
        if feature not in df.columns:
            df[feature] = default_value

    # Date detection
    date_col = next((c for c in ["Date", "date", "DATE"] if c in df.columns), None)
    if date_col:
        try:
            # Prefer %d-%m-%Y, fallback to any
            parsed = pd.to_datetime(df[date_col], format="%d-%m-%Y", errors="coerce")
            if parsed.isna().any():
                parsed = pd.to_datetime(df[date_col], errors="coerce")

            if parsed.isna().all():
                raise ValueError("All dates failed to parse after attempts.")

            df['Year'] = parsed.dt.year.fillna(0).astype(int)
            df['Month'] = parsed.dt.month.fillna(0).astype(int)
            df['Day'] = parsed.dt.day.fillna(0).astype(int)
            df['Weekday'] = parsed.dt.weekday.fillna(0).astype(int)
            df['WeekOfYear'] = parsed.dt.isocalendar().week.fillna(0).astype(int)
            df['DayOfYear'] = parsed.dt.dayofyear.fillna(0).astype(int)

            # Cyclic
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
            df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

            # Flags
            df['Is_Month_Start'] = parsed.dt.is_month_start.astype(int)
            df['Is_Month_End'] = parsed.dt.is_month_end.astype(int)
            df['Is_Quarter_Start'] = parsed.dt.is_quarter_start.astype(int)
            df['Is_Quarter_End'] = parsed.dt.is_quarter_end.astype(int)
            df['Is_Year_Start'] = parsed.dt.is_year_start.astype(int)
            df['Is_Year_End'] = parsed.dt.is_year_end.astype(int)

            df.drop(columns=[date_col], inplace=True, errors='ignore')
        except Exception as e:
            log.warning(f"Date parsing failed: {e}. Using defaults and dropping '{date_col}'.")
            df.drop(columns=[date_col], inplace=True, errors='ignore')

    # Clean numeric-like text
    for col in ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'Weekly_Sales']:
        if col in df.columns:
            df[col] = df[col].replace({',': '', '₹': '', '$': ''}, regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Time-series lags/rollings if Weekly_Sales exists
    if 'Weekly_Sales' in df.columns:
        if set(['Year', 'Month', 'Day']).issubset(df.columns):
            tmp = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce')
            if not tmp.isna().all():
                df = df.assign(_tmpdate=tmp).sort_values('_tmpdate').drop(columns=['_tmpdate']).reset_index(drop=True)
        else:
            df = df.sort_index().reset_index(drop=True)

        df['lag_1'] = df['Weekly_Sales'].shift(1).fillna(0)
        df['lag_2'] = df['Weekly_Sales'].shift(2).fillna(0)
        df['lag_3'] = df['Weekly_Sales'].shift(3).fillna(0)
        df['lag_7'] = df['Weekly_Sales'].shift(7).fillna(0)
        df['rolling_mean_3'] = df['Weekly_Sales'].rolling(window=3).mean().fillna(0)
        df['rolling_mean_7'] = df['Weekly_Sales'].rolling(window=7).mean().fillna(0)
        df['rolling_mean_14'] = df['Weekly_Sales'].rolling(window=14).mean().fillna(0)
        df['rolling_mean_28'] = df['Weekly_Sales'].rolling(window=28).mean().fillna(0)
        df['rolling_std_7'] = df['Weekly_Sales'].rolling(window=7).std().fillna(0)

    # Drop irrelevant text columns
    irrelevant_cols = [
        'product_id', 'product_name', 'category', 'about_product',
        'user_id', 'user_name', 'review_id', 'review_title', 'review_content',
        'img_link', 'product_link'
    ]
    df.drop(columns=[c for c in irrelevant_cols if c in df.columns], inplace=True, errors='ignore')

    # Fill remaining NaNs
    df.fillna(0, inplace=True)

    if df.empty:
        raise ValueError("Dataframe is empty after preprocessing")

    return df


# ========= LIFESPAN: Load Model + Encoders =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoders, feature_columns
    log.info("Starting FastAPI and loading model + encoders...")
    try:
        log.info(f"Fetching model from: {MODEL_AND_ENCODERS_URL}")
        response = requests.get(MODEL_AND_ENCODERS_URL, timeout=30)

        if response.status_code != 200 or not response.content:
            raise RuntimeError(f"Failed to fetch model file (status {response.status_code}).")

        obj = joblib.load(io.BytesIO(response.content))
        if not isinstance(obj, dict):
            raise TypeError("Downloaded file is not a dict containing model & encoders.")

        model = obj.get("model")
        encoders = obj.get("encoders", {})
        feature_columns = obj.get("features", None)

        if model is None or encoders is None:
            raise ValueError("Model or encoders missing in loaded file.")

        log.info(f"Model loaded: {type(model)}")
        log.info(f"Encoders for columns: {list(encoders.keys())}")
        log.info(f"Feature columns: {feature_columns}")
    except Exception as e:
        log.exception("Failed to load model/encoders: %s", e)
        model = None
        encoders = {}
        feature_columns = None

    yield
    log.info("Shutting down FastAPI")


app = FastAPI(lifespan=lifespan)

# ========= CORS =========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= SIMPLE HEALTH =========
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "n_features": len(feature_columns) if feature_columns else 0,
        "encoder_columns": list(encoders.keys()) if encoders else [],
    }

# ========= SCHEMA / SAMPLE HELPERS =========
@app.get("/schema")
def schema():
    return {
        "expected_features": feature_columns,
        "input_hints": {
            "csv_columns": [
                "Date (dd-mm-YYYY) – preferred",
                "Weekly_Sales (optional, for metrics)",
                "Other columns allowed; unknown text columns are dropped.",
            ],
            "note": "If 'Date' is missing, defaults are used. For metrics, include 'Weekly_Sales' or 'actual_price'.",
        },
    }

@app.get("/sample_csv", response_class=PlainTextResponse)
def sample_csv():
    df = pd.DataFrame({
        "Date": ["01-01-2023", "02-01-2023", "03-01-2023"],
        "Weekly_Sales": [12000.0, 12550.5, 13100.0],
    })
    csv_text = df.to_csv(index=False)
    return PlainTextResponse(csv_text, media_type="text/csv")


# ========= PREDICT =========
@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request, file: UploadFile = File(None)):
    global model, encoders, feature_columns

    if model is None or not encoders:
        raise HTTPException(status_code=503, detail="Model or encoders not loaded. Check server logs.")

    try:
        # 1) Load dataframe from multipart or from URL
        if file is not None:
            if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
                log.warning(f"Received content_type {file.content_type}, attempting to parse anyway.")
            content = await file.read()
            df = read_csv_content(content)
            log.info(f"Loaded multipart CSV: rows={len(df)} cols={list(df.columns)}")
        else:
            try:
                body = await request.json()
            except Exception:
                body = {}
            file_url = body.get("file_url") if isinstance(body, dict) else None
            if not file_url:
                raise HTTPException(status_code=400, detail="No file uploaded and no 'file_url' provided.")
            resp = requests.get(file_url, timeout=30)
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download CSV (status {resp.status_code})")
            df = read_csv_content(resp.content)
            log.info(f"Loaded URL CSV: rows={len(df)} cols={list(df.columns)}")

        # 2) Capture dates & actuals before preprocessing
        date_col = next((c for c in ["Date", "date", "DATE"] if c in df.columns), None)
        original_dates = df[date_col].copy() if date_col else None

        actual_col = next(
            (c for c in ["actual_price", "Actual", "actual", "Weekly_Sales", "weekly_sales"] if c in df.columns),
            None
        )
        if actual_col:
            actual_prices = pd.to_numeric(df[actual_col].replace({',': ''}, regex=True), errors='coerce').tolist()
            actual_prices = [None if pd.isna(x) else float(x) for x in actual_prices]
        else:
            actual_prices = [None] * len(df)

        # 3) Preprocess
        df_processed = preprocess_optional_engineered_features(df)

        # 4) Label encoders
        for col, encoder in encoders.items():
            if col in df_processed.columns:
                try:
                    df_processed[col] = fast_label_encode(df_processed[col], encoder)
                except Exception as e:
                    log.warning(f"Encoding failed for '{col}' -> set 0. Error: {e}")
                    df_processed[col] = 0
            else:
                # Feature existed at train time; ensure column exists for model
                df_processed[col] = 0

        # Ensure numeric types
        df_processed = df_processed.apply(pd.to_numeric, errors='coerce').fillna(0)

        # 5) Align to training feature order
        if not feature_columns:
            raise ValueError("Feature columns not loaded. Cannot align input to model.")

        # Add missing & reorder
        missing_cols = [c for c in feature_columns if c not in df_processed.columns]
        for c in missing_cols:
            df_processed[c] = 0
        # Drop extras & reorder
        df_processed = df_processed[feature_columns]

        # 6) Predict
        preds = model.predict(df_processed)
        predicted_list = [float(x) for x in preds]

        # 7) Metrics (if actuals present)
        performance_metrics = compute_metrics(actual_prices, predicted_list)

        # 8) Response dates
        if original_dates is not None and not original_dates.empty:
            response_dates = original_dates.astype(str).tolist()
        elif set(["Year", "Month", "Day"]).issubset(df_processed.columns):
            # reconstruct (best-effort)
            yr = df_processed["Year"].astype(int).astype(str)
            mo = df_processed["Month"].astype(int).astype(str)
            dy = df_processed["Day"].astype(int).astype(str)
            response_dates = (yr + "-" + mo + "-" + dy).tolist()
        else:
            response_dates = [str(i) for i in range(len(predicted_list))]

        return PredictResponse(
            dates=[str(d) for d in response_dates],
            actual_price=actual_prices,
            predicted_price=predicted_list,
            total_predictions=len(predicted_list),
            performance_metrics=performance_metrics,
        )

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Prediction error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


# ========= FEATURE IMPORTANCE =========
@app.get("/feature_importance")
async def get_feature_importance():
    if model is None or feature_columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        fi = dict(zip(feature_columns, getattr(model, "feature_importances_", []).tolist()))
        return JSONResponse(content={"feature_importance": fi})
    except Exception as e:
        log.exception("Feature importance error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
