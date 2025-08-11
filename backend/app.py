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

# === CONFIG ===
MODEL_AND_ENCODERS_URL = (
    "https://tjdagsnqjofpssegmczw.supabase.co/storage/v1/object/public/models/xgb_model.pkl"
)

model = None
encoders = {}
feature_columns = None  # To keep consistent features


# === LIFESPAN: Load Model + Encoders ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoders, feature_columns
    print("🚀 Starting FastAPI and loading model + encoders...")
    try:
        # Download model file
        print(f"Fetching model from: {MODEL_AND_ENCODERS_URL}")
        response = requests.get(MODEL_AND_ENCODERS_URL)

        if response.status_code != 200 or not response.content:
            raise RuntimeError(
                f"Failed to fetch model file (status {response.status_code})."
            )

        # Load the dict with model + encoders
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

        if feature_columns:
            print(f"✅ Feature columns order preserved ({len(feature_columns)} features)")

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
    allow_origins=["*"],  # Adjust for production
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

        # Apply same preprocessing as training
        for col, encoder in encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
            else:
                raise HTTPException(
                    status_code=400, detail=f"Missing column in input: {col}"
                )

        # Reorder columns to match training
        if feature_columns:
            missing_cols = [c for c in feature_columns if c not in df.columns]
            if missing_cols:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required feature columns: {missing_cols}",
                )
            df = df[feature_columns]

        # Predict
        predictions = model.predict(df)

        return JSONResponse(content={"predictions": predictions.tolist()})

    except Exception as e:
        print("❌ Prediction error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
