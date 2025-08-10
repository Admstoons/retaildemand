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

MODEL_AND_ENCODERS_URL = "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models/xgb_model.pkl"

model = None
encoders = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoders
    try:
        response = requests.get(MODEL_AND_ENCODERS_URL)
        response.raise_for_status()
        data = joblib.load(BytesIO(response.content))
        model = data['model']
        encoders = data['encoders']
        print("✅ Model and encoders loaded successfully")
    except Exception as e:
        print(f"❌ Loading failed: {e}")
        model = None
        encoders = {}
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or restrict as needed
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

@app.post("/predict")
def predict_from_file(data: FileURLInput):
    if model is None or not encoders:
        raise HTTPException(status_code=503, detail="Model or encoders not loaded")

    try:
        response = requests.get(data.file_url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")

        # Required columns, fill missing numeric with 0, categorical with 'Unknown'
        required_columns = [
            'Store', 'Source', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
            'CPI', 'Unemployment', 'Year', 'Month', 'Day', 'Weekday',
            'Date', 'Weekly_Sales'
        ]
        for col in required_columns:
            if col not in df.columns:
                if col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Year', 'Month', 'Day', 'Weekday']:
                    df[col] = 0
                else:
                    df[col] = 'Unknown'

        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
        df['Year'] = df['Date'].dt.year.fillna(2023).astype(int)
        df['Month'] = df['Date'].dt.month.fillna(1).astype(int)
        df['Day'] = df['Date'].dt.day.fillna(1).astype(int)
        df['Weekday'] = df['Date'].dt.weekday.fillna(0).astype(int)

        # Cyclic seasonal features
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
        df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

        # Use encoders to transform categorical columns without refitting
        for col in ['Store', 'Source', 'Holiday_Flag']:
            if col in df.columns:
                encoder = encoders.get(col)
                if encoder is None:
                    raise HTTPException(status_code=500, detail=f"Encoder for {col} not found")

                # Map unknown labels to 'Unknown' or a default known class
                df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else 'Unknown')

                # Add 'Unknown' class if missing to avoid errors
                if 'Unknown' not in encoder.classes_:
                    encoder.classes_ = np.append(encoder.classes_, 'Unknown')

                df[col] = encoder.transform(df[col].astype(str))

        # Clean numeric columns with commas, currency symbols
        for col in ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'Weekly_Sales']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '').str.replace('₹', '').str.replace('$', '')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df = df.sort_values(by='Date')

        # Lag and rolling features on Weekly_Sales
        if 'Weekly_Sales' in df.columns:
            df['lag_1'] = df['Weekly_Sales'].shift(1)
            df['lag_2'] = df['Weekly_Sales'].shift(2)
            df['lag_3'] = df['Weekly_Sales'].shift(3)
            df['rolling_mean_3'] = df['Weekly_Sales'].rolling(window=3).mean()
            df['rolling_mean_7'] = df['Weekly_Sales'].rolling(window=7).mean()

            df.dropna(inplace=True)
            actual_values = df['Weekly_Sales'].astype(float).tolist()
        else:
            raise HTTPException(status_code=400, detail="Weekly_Sales column required for lag features")

        # Drop unused columns including 'Date'
        non_numeric_cols = [
            'product_id', 'product_name', 'category', 'about_product',
            'user_id', 'user_name', 'review_id', 'review_title', 'review_content',
            'img_link', 'product_link', 'Date'
        ]
        df.drop(columns=[col for col in non_numeric_cols if col in df.columns], inplace=True)

        df.fillna(0, inplace=True)

        if df.empty:
            raise HTTPException(status_code=400, detail="No valid rows for prediction after preprocessing")

        predictions = model.predict(df)

        formatted_dates = df['Date'].dt.strftime('%Y-%m-%d').tolist() if 'Date' in df.columns else [''] * len(predictions)

        results = {
            "actual_values": [float(x) for x in actual_values],
            "predicted_values": [float(x) for x in predictions],
            "dates": {str(i): formatted_dates[i] for i in range(len(predictions))}
        }

        if not results["actual_values"] or not results["predicted_values"]:
            raise HTTPException(status_code=204, detail="No valid predictions generated")

        return results

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Request error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
