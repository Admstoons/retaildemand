from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import requests
import io
import uvicorn
import traceback

MODEL_AND_ENCODERS_URL = "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models/xgb_model.pkl"

model = None
encoders = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoders
    print("=== Starting FastAPI and loading model/encoders ===")
    try:
        print(f"Fetching model from: {MODEL_AND_ENCODERS_URL}")
        response = requests.get(MODEL_AND_ENCODERS_URL)
        print(f"HTTP status: {response.status_code}, size: {len(response.content)} bytes")

        if response.status_code != 200 or not response.content:
            raise RuntimeError(f"Failed to fetch model file (status {response.status_code}).")

        obj = joblib.load(io.BytesIO(response.content))

        if not isinstance(obj, dict):
            raise TypeError("Downloaded file is not a dictionary containing model and encoders.")

        model = obj.get("model")
        encoders = obj.get("encoders", {})

        if model is None or not encoders:
            raise ValueError("Model or encoders missing in loaded file.")

        print("✅ Model and encoders loaded successfully.")

    except Exception as e:
        print("❌ Failed to load model/encoders:", e)
        traceback.print_exc()
        model = None
        encoders = {}

    yield
    print("=== Shutting down FastAPI ===")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model, encoders
    if model is None or not encoders:
        raise HTTPException(status_code=503, detail="Model or encoders not loaded. Check server logs.")

    try:
        df = pd.read_csv(file.file)
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])

        predictions = model.predict(df)
        return JSONResponse(content={"predictions": predictions.tolist()})

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
