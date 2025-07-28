from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware  # ✅ CORS import
import joblib
import pandas as pd
import requests
from io import StringIO, BytesIO
from sklearn.preprocessing import LabelEncoder

MODEL_URL = "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models/xgb_model.pkl"
model = None  # Global model

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        model = joblib.load(BytesIO(response.content))
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        model = None
    yield

app = FastAPI(lifespan=lifespan)

# ✅ CORS middleware setup
origins = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "https://your-flutter-web-url.web.app",  # 🔁 Replace with your real domain
    "*",  # ⚠️ Allow all for dev testing, remove in production
]

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

@app.post("/predict")
def predict_from_file(data: FileURLInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Download CSV file
        response = requests.get(data.file_url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))

        if df.empty:
            raise HTTPException(status_code=400, detail="CSV is empty")

        required_columns = [
            'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price',
            'CPI', 'Unemployment', 'Year', 'Month', 'Day', 'Weekday',
            'Date', 'Weekly_Sales'
        ]

        for col in required_columns:
            if col not in df.columns:
                if col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
                    df[col] = 0
                else:
                    df[col] = 'Unknown'

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year.fillna(2025).astype(int)
        df['Month'] = df['Date'].dt.month.fillna(1).astype(int)
        df['Day'] = df['Date'].dt.day.fillna(1).astype(int)
        df['Weekday'] = df['Date'].dt.weekday.fillna(0).astype(int)

        df['FormattedDate'] = df['Date'].dt.strftime('%Y-%m-%d').fillna('2025-01-01')

        encoder = LabelEncoder()
        df['Store'] = encoder.fit_transform(df['Store'].astype(str))
        df['Holiday_Flag'] = encoder.fit_transform(df['Holiday_Flag'].astype(str))

        if 'Weekly_Sales' in df.columns:
            actual_values = df['Weekly_Sales'].astype(float).tolist()
        else:
            actual_values = [0.0] * len(df)

        formatted_dates = df['FormattedDate'].tolist()

        df.drop(columns=['Weekly_Sales', 'Date', 'DateStr', 'FormattedDate'], errors='ignore', inplace=True)
        df.fillna(0, inplace=True)

        if df.empty:
            raise HTTPException(status_code=400, detail="No valid rows for prediction")

        predictions = model.predict(df)

        results = {
            "actual_values": [],
            "predicted_values": [],
            "dates": {}
        }

        for i in range(len(predictions)):
            results["actual_values"].append(float(actual_values[i]))
            results["predicted_values"].append(float(predictions[i]))
            results["dates"][str(i)] = formatted_dates[i]

        if not results["actual_values"] or not results["predicted_values"]:
            raise HTTPException(status_code=204, detail="No valid predictions generated")

        return results

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Request error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
