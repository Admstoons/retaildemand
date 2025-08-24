# fastapi_app.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import ssl
from urllib3.util.ssl_ import create_urllib3_context
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from typing import Optional, Dict, Any
import logging

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------
# FastAPI setup
# ---------------------------
app = FastAPI(title="On-demand Train & Predict (Supabase CSV)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Metrics
# ---------------------------
def mean_absolute_percentage_error(y_true, y_pred):  
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero_idx = y_true != 0
    if not np.any(nonzero_idx):
        return np.nan
    return float(np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100)

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    nonzero_idx = denom != 0
    if not np.any(nonzero_idx):
        return np.nan
    return float(np.mean(np.abs(y_true[nonzero_idx] - y_pred[nonzero_idx]) / denom[nonzero_idx]) * 100)

# ---------------------------
# Load dataset from Supabase
# ---------------------------
def load_data_from_csv(csv_url: str, api_token: Optional[str] = None) -> pd.DataFrame:
    logger.info(f"Loading CSV from URL: {csv_url}")
    _ = create_urllib3_context(ssl_minimum_version=ssl.TLSVersion.TLSv1_2)
    session = requests.Session()
    retries = Retry(total=10, backoff_factor=1, status_forcelist=[502, 503, 504, 408])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    resp = session.get(csv_url, headers=headers, timeout=30, verify=True)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    logger.info(f"CSV loaded with shape: {df.shape}")

    # Clean numeric columns (if present)
    for col in [
        'Quantity', 'Weekly_Sales', 'CustomerID', 'discounted_price', 'actual_price',
        'discount_percentage', 'rating', 'n_reviews', 'Store', 'Holiday_Flag',
        'Temperature', 'Fuel_Price', 'CPI', 'Unemployment'
    ]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True).replace('', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if col == 'Weekly_Sales' and df[col].isnull().any():
                df = df.dropna(subset=[col])
            else:
                df[col] = df[col].fillna(0)

    # Fill remaining NaNs
    num_cols = df.select_dtypes(include=['number']).columns
    num_cols = [c for c in num_cols if c != 'Weekly_Sales']
    cat_cols = df.select_dtypes(include=['object']).columns
    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    logger.info(f"CSV preprocessing complete, final shape: {df.shape}")
    return df

# ---------------------------
# Robust preprocessing
# ---------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Starting preprocessing, initial shape: {df.shape}")
    df = df.copy()
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
        logger.info(f"Date column detected: {date_column}")

    if 'Date' in df.columns:
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

        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        logger.info(f"Date parsing complete, date range: {df['Date'].min()} -> {df['Date'].max()}")

        # Calendar features
        df['Year'] = df['Date'].dt.year.astype("Int64")
        df['Month'] = df['Date'].dt.month.astype("Int64")
        df['Day'] = df['Date'].dt.day.astype("Int64")
        df['Weekday'] = df['Date'].dt.weekday.astype("Int64")
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype("Int64")
        df['DayOfYear'] = df['Date'].dt.dayofyear.astype("Int64")

        # Cyclic encodings
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
        df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

        # Period boundaries
        df['Is_Month_Start']   = df['Date'].dt.is_month_start.astype(int)
        df['Is_Month_End']     = df['Date'].dt.is_month_end.astype(int)
        df['Is_Quarter_Start'] = df['Date'].dt.is_quarter_start.astype(int)
        df['Is_Quarter_End']   = df['Date'].dt.is_quarter_end.astype(int)
        df['Is_Year_Start']    = df['Date'].dt.is_year_start.astype(int)
        df['Is_Year_End']      = df['Date'].dt.is_year_end.astype(int)

    if 'Weekly_Sales' in df.columns:
        df['Weekly_Sales'] = pd.to_numeric(df['Weekly_Sales'], errors='coerce')
        df = df.dropna(subset=['Weekly_Sales'])
        for lag in [1, 2, 3, 7]:
            df[f'lag_{lag}'] = df['Weekly_Sales'].shift(lag).fillna(0)
        for window in [3, 7, 14, 28]:
            df[f'rolling_mean_{window}'] = df['Weekly_Sales'].rolling(window).mean().fillna(0)
        df['rolling_std_7'] = df['Weekly_Sales'].rolling(7).std().fillna(0)
        logger.info(f"Lag and rolling features created for Weekly_Sales")

    irrelevant_cols = [
        'InvoiceNo', 'StockCode', 'Description',
        'product_id', 'product_name', 'category', 'about_product',
        'user_id', 'user_name', 'review_id', 'review_title', 'review_content',
        'img_link', 'product_link', 'name'
    ]
    df.drop(columns=[c for c in irrelevant_cols if c in df.columns], inplace=True, errors='ignore')

    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    logger.info(f"Preprocessing finished, final shape: {df.shape}")
    return df

# ---------------------------
# Feature prep
# ---------------------------
def prepare_features_and_target(df: pd.DataFrame, target_column: str = 'Weekly_Sales'):
    logger.info(f"Preparing features and target: {target_column}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    df = df.copy()
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
    df = df.dropna(subset=[target_column])

    dates = df['Date'] if 'Date' in df.columns else pd.date_range(start='2020-01-01', periods=len(df))
    if 'Date' in df.columns:
        df = df.drop(columns=['Date'])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype(str).fillna("Missing")

    logger.info(f"Feature shape: {X.shape}, Target length: {len(y)}")
    return X, y, dates

# ---------------------------
# Hyperparameter tuning
# ---------------------------
def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series):
    logger.info("Tuning hyperparameters for XGBoost")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=0)
    grid.fit(X_train, y_train)
    logger.info(f"Best hyperparameters found: {grid.best_params_}")
    return grid.best_estimator_

# ---------------------------
# Train + Predict
# ---------------------------
def train_and_predict(csv_url: str, api_token: Optional[str] = None) -> Dict[str, Any]:
    logger.info("Starting train_and_predict pipeline")
    df = load_data_from_csv(csv_url, api_token)
    df = preprocess_data(df)
    X, y, dates = prepare_features_and_target(df)

    logger.info("Splitting data into train/test sets")
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, test_size=0.2, random_state=42
    )

    # Label encode categorical columns
    for col in X_train.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = np.where(X_test[col].isin(le.classes_), le.transform(X_test[col]), -1)

    model = tune_hyperparameters(X_train, y_train)
    model.fit(X_train, y_train)
    logger.info("Model training complete")

    y_pred = model.predict(X_test)
    logger.info("Predictions generated")

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "mape": mean_absolute_percentage_error(y_test, y_pred),
        "smape": symmetric_mean_absolute_percentage_error(y_test, y_pred),
    }

    predictions_list = []
    for dt, actual, pred in zip(dates_test, y_test, y_pred):
        predictions_list.append({
            "Date": str(dt.date()) if isinstance(dt, pd.Timestamp) else str(dt),
            "Actual_Weekly_Sales": float(actual),
            "Predicted_Weekly_Sales": float(pred)
        })

    return {
        "dataset_url": csv_url,
        "total_predictions": len(predictions_list),
        "predictions": predictions_list,
        "performance_metrics": metrics
    }

# ---------------------------
# API Endpoints
# ---------------------------
@app.get("/predict")
def predict(
    file_url: str = Query(..., description="Supabase public CSV URL"),
    api_token: Optional[str] = Query(None, description="Optional Bearer token for private buckets")
):
    try:
        return train_and_predict(file_url, api_token)
    except Exception as e:
        logger.error(f"Error in /predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/url")
def predict_url(payload: Dict[str, Optional[str]]):
    """
    Accepts JSON like: {"csv_url": "...", "api_token": "..."}
    """
    try:
        url = payload.get("csv_url")
        if not url:
            raise ValueError("Field 'csv_url' is required.")
        token = payload.get("api_token")
        return train_and_predict(url, token)
    except Exception as e:
        logger.error(f"Error in /predict/url: {e}")
        raise HTTPException(status_code=500, detail=str(e))
