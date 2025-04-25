from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
from io import BytesIO, StringIO
from contextlib import asynccontextmanager
from fastapi.responses import Response

MODEL_URL = "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models//xgb_model.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    model = joblib.load(BytesIO(response.content))
    yield

app = FastAPI(lifespan=lifespan)

class FileURLInput(BaseModel):
    file_url: str

@app.get("/")
def root():
    return {"message": "Retail Demand Forecast API is running"}

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)  # Prevents 404 error for favicon

@app.post("/predict")
def predict_from_file(data: FileURLInput):
    try:
        response = requests.get(data.file_url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))

        if df.empty:
            return {"error": "Downloaded CSV is empty"}

        required_columns = ['Holiday_Flag', 'Date', 'Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {"error": f"Missing columns in the input CSV: {', '.join(missing_columns)}"}

        # Process 'Date' column
        if 'Date' in df.columns:
            try:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day
                df['Weekday'] = df['Date'].dt.weekday
                df.drop(columns=['Date'], inplace=True)
            except Exception as e:
                return {"error": f"Date processing failed: {str(e)}"}

        # Drop target column if present
        if 'Weekly_Sales' in df.columns:
            df.drop(columns=['Weekly_Sales'], inplace=True)

        # Drop non-numeric columns
        non_numeric = df.select_dtypes(exclude=['int', 'float', 'bool']).columns
        df.drop(columns=non_numeric, inplace=True)

        if df.empty:
            return {"error": "No numeric data left for prediction after preprocessing."}

        # Make prediction
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
