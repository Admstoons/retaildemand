from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Optional, Dict # Added Dict for performance_metrics
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
    performance_metrics: Optional[Dict[str, float]] = None # Added for model metrics


# === Preprocessing function with optional engineered features ===
def preprocess_optional_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    print("Starting preprocessing for prediction...")
    print(f"Input DataFrame columns at start of preprocessing: {list(df.columns)}")

    # Define required features with default values (matching training)
    # These defaults are primarily for handling cases where input CSV might lack certain columns
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

    # Ensure all required features are present, fill with default values if missing
    for feature, default_value in required_features_defaults.items():
        if feature not in df.columns:
            df[feature] = default_value
            print(f"Added missing column '{feature}' with default value {default_value}")

    # Date features - matching training notebook's logic
    # Find the correct date column, case-insensitively
    date_col = None
    for col_name in ["Date", "date", "DATE"]:
        if col_name in df.columns:
            date_col = col_name
            break

    if date_col:
        print(f"Date column identified: {date_col}")
        # Accept a variety of date formats but try the project's expected format first
        try:
            df[date_col] = pd.to_datetime(df[date_col], format="%d-%m-%Y", errors='coerce')
            # fallback: try parsing any format for rows that stayed NaT
            if df[date_col].isna().any():
                print(f"Some dates failed with %d-%m-%Y, trying generic parse for NaT values.")
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Check for NaT values after all parsing attempts
            if df[date_col].isna().all():
                raise ValueError("All dates failed to parse after all attempts.")
            elif df[date_col].isna().any():
                print("Warning: Some dates are still NaT after parsing. These rows will get default date features.")

            df['Year'] = df[date_col].dt.year.fillna(0).astype(int) # Change fillna to 0
            df['Month'] = df[date_col].dt.month.fillna(0).astype(int) # Change fillna to 0
            df['Day'] = df[date_col].dt.day.fillna(0).astype(int) # Change fillna to 0
            df['Weekday'] = df[date_col].dt.weekday.fillna(0).astype(int) # Change fillna to 0
            
            df['WeekOfYear'] = df[date_col].dt.isocalendar().week.fillna(0).astype(int) # Change fillna to 0
            df['DayOfYear'] = df[date_col].dt.dayofyear.fillna(0).astype(int) # Change fillna to 0

            # Cyclic encoding for Month and Weekday
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
            df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

            # Boolean flags for start/end of month, quarter, and year
            df['Is_Month_Start'] = df[date_col].dt.is_month_start.astype(int)
            df['Is_Month_End'] = df[date_col].dt.is_month_end.astype(int)
            df['Is_Quarter_Start'] = df[date_col].dt.is_quarter_start.astype(int)
            df['Is_Quarter_End'] = df[date_col].dt.is_quarter_end.astype(int)
            df['Is_Year_Start'] = df[date_col].dt.is_year_start.astype(int)
            df['Is_Year_End'] = df[date_col].dt.is_year_end.astype(int)
            
            print(f"Derived date features: Year={df['Year'].iloc[0]}, Month={df['Month'].iloc[0]}, WeekOfYear={df['WeekOfYear'].iloc[0]}")
            
            df.drop(columns=[date_col], inplace=True, errors='ignore') # Drop original Date column
        except Exception as e:
            print(f"Error during date feature engineering: {e}")
            traceback.print_exc()
            print("Proceeding with default date features due to error.")
            # Set all date-related columns to defaults if an error occurred during parsing or derivation
            for f in ['Year', 'Month', 'Day', 'Weekday', 'WeekOfYear', 'DayOfYear',
                      'Month_sin', 'Month_cos', 'Weekday_sin', 'Weekday_cos',
                      'Is_Month_Start', 'Is_Month_End', 'Is_Quarter_Start', 'Is_Quarter_End',
                      'Is_Year_Start', 'Is_Year_End']:
                if f not in df.columns:
                    df[f] = 0 # Change default value from required_features_defaults.get(f, 0) to just 0
            
            if date_col in df.columns:
                df.drop(columns=[date_col], inplace=True, errors='ignore')

    else:
        print("No 'Date' column found, using default date features and cyclic/boolean values.")
        # Defaults for date components are already set at the beginning of the function (now 0, not from required_features_defaults)
        # No need to drop date_col as it was never present
            
    # Clean numeric columns (price, rating related, and Weekly_Sales)
    for col in ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'Weekly_Sales']:
        if col in df.columns:
            df[col] = df[col].replace({',': '', '₹': '', '$': ''}, regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Cleaned column: {col}")

    # Lag and rolling features for Weekly_Sales - matching training notebook's logic
    if 'Weekly_Sales' in df.columns:
        # Sort by date before creating time-series features
        # Ensure 'Year', 'Month', 'Day' are available for sorting if original 'Date' was dropped
        if 'Year' in df.columns and 'Month' in df.columns and 'Day' in df.columns:
            # Create a temporary datetime column for sorting if original 'Date' was dropped
            df['_temp_date_for_sort'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce')
            # Only sort if _temp_date_for_sort column is not all NaT (i.e. at least some dates parsed correctly)
            if not df['_temp_date_for_sort'].isna().all():
                df = df.sort_values(by='_temp_date_for_sort').reset_index(drop=True)
            df.drop(columns=['_temp_date_for_sort'], inplace=True, errors='ignore')
        else: # If no date components at all, still try to sort by index as a fallback
            df = df.sort_index().reset_index(drop=True)

        df['lag_1'] = df['Weekly_Sales'].shift(1).fillna(0)
        df['lag_2'] = df['Weekly_Sales'].shift(2).fillna(0)
        df['lag_3'] = df['Weekly_Sales'].shift(3).fillna(0)
        df['lag_7'] = df['Weekly_Sales'].shift(7).fillna(0) # Added lag_7
        df['rolling_mean_3'] = df['Weekly_Sales'].rolling(window=3).mean().fillna(0)
        df['rolling_mean_7'] = df['Weekly_Sales'].rolling(window=7).mean().fillna(0)
        df['rolling_mean_14'] = df['Weekly_Sales'].rolling(window=14).mean().fillna(0) # Added rolling_mean_14
        df['rolling_mean_28'] = df['Weekly_Sales'].rolling(window=28).mean().fillna(0) # Added rolling_mean_28
        df['rolling_std_7'] = df['Weekly_Sales'].rolling(window=7).std().fillna(0) # Added rolling_std_7
        print("Generated lag and rolling features")
    else:
        print("No 'Weekly_Sales' column, setting lag/rolling features to default values (0)")
        # Defaults are already set for these at the beginning of the function
            
    # Interaction feature
    # No interaction feature as discount_percentage and rating are not in the training data
    pass

    # Drop irrelevant text columns - matching training notebook's logic
    irrelevant_cols = [
        'product_id', 'product_name', 'category', 'about_product',
        'user_id', 'user_name', 'review_id', 'review_title', 'review_content',
        'img_link', 'product_link'
    ]
    df.drop(columns=[col for col in irrelevant_cols if col in df.columns], inplace=True, errors='ignore')
    print(f"Dropped columns: {[col for col in irrelevant_cols if col in df.columns]}")

    # Final fill NaNs for any remaining numeric columns not covered (e.g., Temperature, Fuel_Price if missing)
    # This also catches any NaNs introduced by operations if default wasn't sufficient or new NaN occurred.
    df.fillna(0, inplace=True) 
    
    # Do not drop rows at this stage, as it will be handled by filtering with feature_columns later
    # and dropping rows here might remove data needed for actual_price and dates in the response.

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
        print("Attempting to load data...")
        if file is not None:
            # Received multipart upload
            print(f"Received multipart file upload: {file.filename}")
            df = pd.read_csv(file.file)
        else:
            # Try reading JSON body for file_url
            try:
                body = await request.json()
            except Exception as json_err:
                print(f"Could not parse JSON body: {json_err}")
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
        print(f"Successfully loaded input data with {len(df)} rows and columns: {list(df.columns)}")

        # ---------- 2) Save original date and actual_price columns for output ----------
        # date candidates
        date_col_name_original = None
        for candidate in ["Date", "date", "DATE"]:
            if candidate in df.columns:
                date_col_name_original = candidate
                break
        
        # Capture dates before preprocessing potentially drops the column or changes its format
        if date_col_name_original:
            original_dates = df[date_col_name_original].copy() # Capture as Series
        else:
            original_dates = None # No date column in original input
            
        # actual price column candidates
        actual_col_name_original = None
        for candidate in ["actual_price", "Actual", "actual", "Weekly_Sales", "weekly_sales"]:
            if candidate in df.columns:
                actual_col_name_original = candidate
                break
        
        if actual_col_name_original:
            # coerce to numeric, keep NAs as None, convert commas/symbols
            actual_prices = pd.to_numeric(df[actual_col_name_original].replace({',': ''}, regex=True), errors='coerce').tolist()
            # convert nan -> None for JSON friendly
            actual_prices = [None if (pd.isna(x)) else float(x) for x in actual_prices]
        else:
            actual_prices = [None] * len(df)

        print(f"Original dates captured: {original_dates.iloc[0] if original_dates is not None and not original_dates.empty else 'N/A'}")
        print(f"Actual prices captured: {actual_prices[0] if actual_prices else 'N/A'}")


        # ---------- 3) Preprocess ----------
        print("Starting preprocessing with preprocess_optional_engineered_features...")
        df_processed = preprocess_optional_engineered_features(df.copy()) # Pass a copy to avoid modifying original df for actual_prices/dates
        print(f"DataFrame after preprocessing (shape: {df_processed.shape}, columns: {list(df_processed.columns)})")
        # Add more specific logging for 'Year' column here
        if 'Year' in df_processed.columns:
            print(f"Year column present after preprocessing. First few values: {df_processed['Year'].head().tolist()}, dtype: {df_processed['Year'].dtype}")
        else:
            print("Year column NOT present after preprocessing.")


        # ---------- 4) Apply label encoders ----------
        # encoders expected to be a dict: { column_name: encoder_obj }
        for col, encoder in encoders.items():
            if col in df_processed.columns:
                try:
                    # convert to string before transform to match training pipeline
                    df_processed[col] = df_processed[col].astype(str)
                    # Handle unseen labels during inference by converting to string and then encoding
                    # If a label is not seen, encoder.transform will raise an error,
                    # so we map known labels and use a placeholder for unknown ones.
                    df_processed[col] = df_processed[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1) # Use -1 for unseen
                    print(f"Encoded column: {col}")
                except Exception as enc_err:
                    print(f"Warning: Encoding failed for column '{col}'. Setting to 0. Error: {str(enc_err)}")
                    df_processed[col] = 0 # Fallback to 0 if encoding fails
            else:
                # Only add a zero column if the encoder existed in training (i.e. if it was a feature)
                if encoder is not None: # Check if encoder actually exists in the dict
                    df_processed[col] = 0
                    print(f"Column {col} not found in processed data, set to default value 0 for encoding.")

        print(f"DataFrame after label encoding (shape: {df_processed.shape}, columns: {list(df_processed.columns)})")
        # Add more specific logging for 'Year' column here
        if 'Year' in df_processed.columns:
            print(f"Year column present after encoding. First few values: {df_processed['Year'].head().tolist()}, dtype: {df_processed['Year'].dtype}")
        else:
            print("Year column NOT present after encoding.")

        # Ensure all columns are numeric after encoding
        for col in df_processed.columns:
            # IMPORTANT: Use df_processed[col] here, not df[col] as df is the original
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(0)
            
        print(f"DataFrame after final numeric conversion (shape: {df_processed.shape}, columns: {list(df_processed.columns)})")
        if 'Year' in df_processed.columns:
            print(f"Year column present after numeric conversion. First few values: {df_processed['Year'].head().tolist()}, dtype: {df_processed['Year'].dtype}")
        else:
            print("Year column NOT present after numeric conversion.")


        # ---------- 5) Match training column order ----------
        if feature_columns:
            # Add any missing columns that the model expects, fill with 0
            missing_cols = [c for c in feature_columns if c not in df_processed.columns]
            if missing_cols:
                for c in missing_cols:
                    df_processed[c] = 0
                print(f"Added missing columns with zeros: {missing_cols}")
            
            # Drop any extra columns that the model does not expect, and reorder columns
            # This is where the KeyError for 'Year' might happen if df_processed does not have 'Year'
            print(f"Before final column reorder. df_processed columns: {list(df_processed.columns)}, feature_columns: {feature_columns}")
            df_processed = df_processed[feature_columns]
            print(f"Reordered columns to match training: {list(df_processed.columns)}")
        else:
            raise ValueError("Feature columns not loaded. Cannot match input to model.")
        
        print(f"DataFrame ready for prediction (shape: {df_processed.shape}, columns: {list(df_processed.columns)})")
        if 'Year' in df_processed.columns:
            print(f"Year column present before prediction. First few values: {df_processed['Year'].head().tolist()}, dtype: {df_processed['Year'].dtype}")
        else:
            print("Year column NOT present before prediction. THIS IS CRITICAL.")


        # ---------- 6) Predict ----------
        predictions = model.predict(df_processed)
        # Ensure predictions are plain Python floats
        predicted_list = [float(x) for x in predictions]

        # ---------- 7) Calculate Performance Metrics (if actual_price available) ----------
        performance_metrics = {}
        if actual_col_name_original and actual_prices:
            # Filter out None values from actual_prices for metric calculation
            valid_actuals = np.array([x for x in actual_prices if x is not None])
            valid_predictions = np.array([predicted_list[i] for i, x in enumerate(actual_prices) if x is not None])

            if len(valid_actuals) > 0:
                errors = np.abs(valid_actuals - valid_predictions)

                mae = np.mean(errors)
                mse = np.mean(errors**2)
                rmse = np.sqrt(mse)
                
                # Calculate R2 score only if actual_values has variance
                if np.sum((valid_actuals - np.mean(valid_actuals))**2) > 0:
                    r2 = 1 - (np.sum((errors)**2) / np.sum((valid_actuals - np.mean(valid_actuals))**2))
                else:
                    r2 = 0.0 # If actual values are constant, R2 is undefined or 0
                
                # Calculate MAPE (Mean Absolute Percentage Error)
                mape_errors = []
                for i in range(len(valid_actuals)):
                    if valid_actuals[i] != 0: # Avoid division by zero
                        mape_errors.append(np.abs((valid_actuals[i] - valid_predictions[i]) / valid_actuals[i]))
                
                if len(mape_errors) > 0:
                    mape = np.mean(mape_errors) * 100
                else:
                    mape = 0.0 # No valid actual values to calculate MAPE

                performance_metrics = {
                    "mae": round(float(mae), 2),
                    "mse": round(float(mse), 2),
                    "rmse": round(float(rmse), 2),
                    "r2": round(float(r2), 4),
                    "mape": round(float(mape), 2)
                }
                print("Calculated performance metrics:", performance_metrics)
            else:
                print("⚠️ No valid actual values for performance metrics calculation.")
        else:
            print("⚠️ 'Weekly_Sales' or 'actual_price' column not found or no valid actuals, performance metrics not available.")


        # ---------- 8) Prepare dates for response ----------
        # Use the original dates captured earlier, or default to generic if none existed
        if original_dates is not None and not original_dates.empty:
            response_dates = original_dates.astype(str).tolist()
        elif 'Year' in df_processed.columns and 'Month' in df_processed.columns and 'Day' in df_processed.columns:
            # Fallback to reconstructed dates from preprocessed df if original date column was dropped/missing
            # Ensure these are integer types before converting to string for dates
            df_processed['Year'] = df_processed['Year'].astype(int)
            df_processed['Month'] = df_processed['Month'].astype(int)
            df_processed['Day'] = df_processed['Day'].astype(int)
            response_dates = (df_processed['Year'].astype(str) + '-' + 
                              df_processed['Month'].astype(str) + '-' + 
                              df_processed['Day'].astype(str)).tolist()
        else:
            response_dates = [str(i) for i in range(len(predicted_list))] # Generic numbering if no date info at all

        # ---------- 9) Return result ----------
        return PredictResponse(
            dates=[str(d) for d in response_dates], # Ensure all dates are strings
            actual_price=actual_prices,
            predicted_price=predicted_list,
            total_predictions=len(predicted_list),
            performance_metrics=performance_metrics if performance_metrics else None # Include metrics if calculated
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
