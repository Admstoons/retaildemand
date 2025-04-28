from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import requests
from io import StringIO
from contextlib import asynccontextmanager
from fastapi.responses import Response
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

MODEL_URL = "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models/xgb_model.pkl"

model = None  # Initialize model globally

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        model = joblib.load(BytesIO(response.content))
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        model = None  # Allow app to run without crashing
    yield

app = FastAPI(lifespan=lifespan)

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
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    try:
        # Fetch the CSV file
        response = requests.get(data.file_url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))

        if df.empty:
            raise HTTPException(status_code=400, detail="Downloaded CSV is empty")

        # === Start of Preprocessing ===

        # List of required columns for prediction (match the columns the model expects)
        required_columns = [
            'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 
            'CPI', 'Unemployment', 'Year', 'Month', 'Day', 'Weekday', 'Date', 'Weekly_Sales'
        ]
        
        # Handle missing columns by adding them with default values (like 0 or the mean of the column)
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: Missing column {col}. Filling with default values.")
                # Default values or mean of the column (if numeric)
                if col in ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']:
                    df[col] = 0  # Fill with 0 for numeric columns
                else:
                    df[col] = 'Unknown'  # Fill with 'Unknown' for categorical columns like 'Store'

        # Remove any extra columns that might have been included in the data
        extra_columns = [col for col in df.columns if col not in required_columns]
        df.drop(columns=extra_columns, inplace=True)

        # Process 'Date' column only if it exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Day'] = df['Date'].dt.day
            df['Weekday'] = df['Date'].dt.weekday
            df.drop(columns=['Date'], inplace=True)
        else:
            # If 'Date' column is missing, create a default column for dates (e.g., range of integers)
            df['Year'] = 2025  # Default year, can be adjusted
            df['Month'] = 1  # Default month
            df['Day'] = 1  # Default day
            df['Weekday'] = 0  # Default to Monday (0)

        # Convert Year, Month, Day, and Weekday to numeric types (ensure they're integers)
        for col in ['Year', 'Month', 'Day', 'Weekday']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # Label Encode certain columns (e.g., 'Store')
        encoder = LabelEncoder()
        for col in ['Store', 'Holiday_Flag']:
            if col in df.columns:
                df[col] = encoder.fit_transform(df[col].astype(str))

        # Fill missing values with 0 or appropriate default
        df.fillna(0, inplace=True)

        # Drop target column 'Weekly_Sales' if present
        if 'Weekly_Sales' in df.columns:
            actual_values = df['Weekly_Sales']
            df.drop(columns=['Weekly_Sales'], inplace=True)
        else:
            actual_values = [0] * len(df)  # Default to 0 if 'Weekly_Sales' is missing

        if df.empty:
            raise HTTPException(status_code=400, detail="No data left for prediction after preprocessing.")

        # === End of Preprocessing ===

        # Make predictions
        predictions = model.predict(df)

        # Calculate errors (absolute difference between actual and predicted)
        errors = abs(actual_values - predictions)

        # Prepare final response with Dates, Actual Values, Predictions, and Errors
        response_data = {
            "dates": df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-' + df['Day'].astype(str),
            "actual_values": actual_values.tolist(),
            "predictions": predictions.tolist(),
            "errors": errors.tolist()
        }

        return response_data

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Request failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")






