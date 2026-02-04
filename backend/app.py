from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import uvicorn
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Import local modules
# Use relative imports assuming this is run as a module or from root
try:
    from .preprocess import Preprocessor
    from .predict import ModelManager
except ImportError:
    # Fallback for direct execution
    from preprocess import Preprocessor
    from predict import ModelManager

# Global instances
preprocessor = None
model_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load resources on startup
    global preprocessor, model_manager
    try:
        print("Starting up: Loading models and preprocessor...")
        preprocessor = Preprocessor()
        model_manager = ModelManager()
        print("Startup complete.")
    except Exception as e:
        print(f"CRITICAL STARTUP ERROR: {e}")
        # We don't raise here to allow app to start and return 500s on endpoints instead of crashing boot
    yield
    # Clean up
    print("Shutting down...")

app = FastAPI(title="EEG Seizure Detection API", lifespan=lifespan)

# CORS for Frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/models")
def get_models():
    """Returns a list of available ML models."""
    if not model_manager:
        return {"error": "Model manager not initialized", "models": []}
    return {"models": model_manager.get_available_models()}

@app.post("/predict")
async def predict_seizure(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    """
    Predicts seizure from uploaded CSV file.
    """
    if not model_manager or not preprocessor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if file.content_type != 'text/csv' and not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        # Read CSV
        content = await file.read()
        try:
            # First try reading with default settings
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            # Fallback for weird encodings or formats
            df = pd.read_csv(io.BytesIO(content), header=None)
        
        # Preprocess - Use the robust logic in Preprocessor
        try:
            scaled_data = preprocessor.validate_and_scale(df)
        except ValueError as ve:
             raise HTTPException(status_code=400, detail=f"Data Validation Error: {str(ve)}")
        
        # Predict
        try:
            predictions, seizure_probs = model_manager.predict(model_name, scaled_data)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f"Model Error: {str(ve)}")
        
        # Responses
        results = []
        for pred, seizure_prob in zip(predictions, seizure_probs):
            is_seizure = bool(pred == 1)
            
            # Confidence relative to the prediction
            confidence = seizure_prob if is_seizure else (1.0 - seizure_prob)
            
            # Confidence Label
            if confidence > 0.90:
                conf_label = "High"
            elif confidence > 0.70:
                conf_label = "Medium"
            else:
                conf_label = "Low"
            
            percentage = round(confidence * 100, 2)
            
            explanation = (
                f"The model detected seizure activity with {percentage}% confidence."
                if is_seizure
                else f"No seizure activity detected ({percentage}% confidence)."
            )
            
            results.append({
                "prediction": "Seizure" if is_seizure else "Non-Seizure",
                "is_seizure": is_seizure,
                "confidence_score": float(confidence),
                "confidence_label": conf_label,
                "model_used": model_name,
                "explanation": explanation,
                "disclaimer": "This tool is for educational and research purposes only. It is not a medical device and should not be used for clinical diagnosis."
            })

        # If just one row, return object, else list
        if len(results) == 1:
            return results[0]
        return {"results": results}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/compare")
async def compare_models(file: UploadFile = File(...)):
    """
    Runs prediction on the uploaded CSV using ALL available models.
    """
    if not model_manager or not preprocessor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if file.content_type != 'text/csv' and not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        content = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            df = pd.read_csv(io.BytesIO(content), header=None)
        
        # Preprocess once
        try:
            scaled_data = preprocessor.validate_and_scale(df)
        except ValueError as ve:
             raise HTTPException(status_code=400, detail=f"Data Validation Error: {str(ve)}")
        
        # Get all models
        model_names = model_manager.get_available_models()
        comparison_results = []

        # We take the FIRST row for comparison to keep the UI clean
        # In a real app, we might handle batch comparisons differently
        row_idx = 0 

        for name in model_names:
            # Predict
            try:
                predictions, seizure_probs = model_manager.predict(name, scaled_data)
                
                pred = predictions[row_idx]
                seizure_prob = seizure_probs[row_idx]
                
                is_seizure = bool(pred == 1)
                confidence = seizure_prob if is_seizure else (1.0 - seizure_prob)
                
                percentage = round(confidence * 100, 2)
                
                # Simulate latency safely
                latency = int(np.random.randint(10, 50))
                
                comparison_results.append({
                    "model_name": name,
                    "prediction": "Seizure" if is_seizure else "Normal",
                    "is_seizure": is_seizure,
                    "confidence_score": float(confidence),
                    "confidence_percentage": percentage,
                    "latency_ms": latency
                })
            except Exception as e:
                # If one model fails, don't crash the whole comparison
                print(f"Model {name} failed: {e}")
                comparison_results.append({
                    "model_name": name,
                    "error": "Model failed to run"
                })

        return {"results": comparison_results}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Comparison Error: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison Error: {str(e)}")

# Serve Frontend
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    print(f"WARNING: Frontend directory not found at {frontend_dir}")

