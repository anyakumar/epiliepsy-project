from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
import time
import uvicorn
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Import local modules
try:
    from .preprocess import Preprocessor
    from .predict import ModelManager
except ImportError:
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
    yield
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

def _parse_csv_content(content: bytes) -> pd.DataFrame:
    """Robustly parse CSV bytes into a DataFrame handling headerless or headed data."""
    try:
        df = pd.read_csv(io.BytesIO(content))
        if len(df) == 0 or pd.to_numeric(df.columns, errors='coerce').notnull().sum() > 10:
            return pd.read_csv(io.BytesIO(content), header=None)
        return df
    except Exception:
        return pd.read_csv(io.BytesIO(content), header=None)

@app.get("/models")
def get_models():
    """Returns a list of available ML & DL models with metadata."""
    if not model_manager:
        return {"error": "Model manager not initialized", "models": []}
    return {
        "models": model_manager.get_available_models(),
        "metadata": model_manager.get_models_metadata()
    }

@app.post("/predict")
async def predict_seizure(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    """
    Predicts seizure from uploaded CSV file with support for classical ML,
    1D CNN, and BiLSTM deep learning models, plus Grad-CAM explainability.
    """
    if not model_manager or not preprocessor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if file.content_type != 'text/csv' and not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        content = await file.read()
        df = _parse_csv_content(content)

        # Preprocess
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
        for i, (pred, seizure_prob) in enumerate(zip(predictions, seizure_probs)):
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

            percentage = round(float(confidence) * 100, 2)
            pred_label = "Seizure" if is_seizure else "Non-Seizure"
            model_type = "Deep Learning" if model_manager.is_deep_learning(model_name) else "Classical ML"

            explanation = (
                f"The {model_type} model ({model_name}) detected seizure activity with {percentage}% confidence."
                if is_seizure
                else f"No seizure activity detected by {model_type} model ({model_name}) ({percentage}% confidence)."
            )

            result_item = {
                "prediction": pred_label,
                "is_seizure": is_seizure,
                "confidence_score": float(confidence),
                "confidence_label": conf_label,
                "model_used": model_name,
                "model_type": model_type,
                "explanation": explanation,
                "disclaimer": "This tool is for educational and research purposes only. It is not a medical device and should not be used for clinical diagnosis."
            }

            # Generate Grad-CAM payload for 1D CNN
            if model_manager.supports_gradcam(model_name) and len(predictions) <= 10:
                try:
                    exp_payload = model_manager.explain(
                        model_name,
                        scaled_data[i],
                        pred_label,
                        percentage
                    )
                    if exp_payload:
                        result_item["explainability"] = exp_payload
                except Exception as exp_err:
                    print(f"Explainability warning: {exp_err}")
                    result_item["explainability_error"] = str(exp_err)

            results.append(result_item)

        if len(results) == 1:
            return results[0]
        return {"results": results}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/explain")
async def explain_eeg(
    file: UploadFile = File(...),
    model_name: str = Form("cnn")
):
    """
    Generates Grad-CAM temporal explainability heatmap and visualization for uploaded EEG sample.
    """
    if not model_manager or not preprocessor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not model_manager.supports_gradcam(model_name):
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' does not support Grad-CAM explainability.")

    try:
        content = await file.read()
        df = _parse_csv_content(content)

        scaled_data = preprocessor.validate_and_scale(df)
        predictions, seizure_probs = model_manager.predict(model_name, scaled_data)

        pred = predictions[0]
        seizure_prob = seizure_probs[0]
        is_seizure = bool(pred == 1)
        confidence = seizure_prob if is_seizure else (1.0 - seizure_prob)
        percentage = round(float(confidence) * 100, 2)
        pred_label = "Seizure" if is_seizure else "Non-Seizure"

        payload = model_manager.explain(model_name, scaled_data[0], pred_label, percentage)
        return {
            "model_used": model_name,
            "prediction": pred_label,
            "confidence_percentage": percentage,
            "explainability": payload
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Explainability Error: {e}")
        raise HTTPException(status_code=500, detail=f"Explainability Error: {str(e)}")


@app.post("/compare")
async def compare_models(file: UploadFile = File(...)):
    """
    Runs prediction on the uploaded CSV using ALL available classical ML and deep learning models.
    """
    if not model_manager or not preprocessor:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if file.content_type != 'text/csv' and not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")

    try:
        content = await file.read()
        df = _parse_csv_content(content)

        # Preprocess once
        try:
            scaled_data = preprocessor.validate_and_scale(df)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=f"Data Validation Error: {str(ve)}")

        model_names = model_manager.get_available_models()
        comparison_results = []
        row_idx = 0

        for name in model_names:
            try:
                t0 = time.perf_counter()
                predictions, seizure_probs = model_manager.predict(name, scaled_data)
                latency = round((time.perf_counter() - t0) * 1000, 2)

                pred = predictions[row_idx]
                seizure_prob = seizure_probs[row_idx]

                is_seizure = bool(pred == 1)
                confidence = seizure_prob if is_seizure else (1.0 - seizure_prob)
                percentage = round(float(confidence) * 100, 2)

                comparison_results.append({
                    "model_name": name,
                    "model_type": "Deep Learning" if model_manager.is_deep_learning(name) else "Classical ML",
                    "supports_gradcam": model_manager.supports_gradcam(name),
                    "prediction": "Seizure" if is_seizure else "Normal",
                    "is_seizure": is_seizure,
                    "confidence_score": float(confidence),
                    "confidence_percentage": percentage,
                    "latency_ms": latency
                })
            except Exception as e:
                print(f"Model {name} failed: {e}")
                comparison_results.append({
                    "model_name": name,
                    "model_type": "Deep Learning" if model_manager.is_deep_learning(name) else "Classical ML",
                    "error": f"Model failed: {str(e)}"
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
