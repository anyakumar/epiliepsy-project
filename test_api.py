import pandas as pd
import numpy as np
import io
import sys
import os

# Set UTF-8 encoding for standard output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from backend.app import app

def make_dummy_csv_buffer(with_header=True):
    np.random.seed(42)
    dummy_data = np.random.randn(1, 178)
    cols = [f"X{i}" for i in range(1, 179)] if with_header else None
    df = pd.DataFrame(dummy_data, columns=cols)
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False, header=with_header)
    csv_buffer.seek(0)
    return csv_buffer.getvalue()

def run_tests():
    print("Initializing TestClient with FastAPI lifespan...")
    with TestClient(app) as client:
        # 1. Test /models
        print("\n[1/5] Testing GET /models...")
        resp = client.get("/models")
        assert resp.status_code == 200, f"/models failed: {resp.text}"
        data = resp.json()
        models = data.get("models", [])
        print(f"[OK] /models returned {len(models)} models: {models}")
        assert "cnn" in models and "bilstm" in models and "xgboost" in models

        csv_bytes = make_dummy_csv_buffer()

        # 2. Test /predict with Classical ML
        print("\n[2/5] Testing POST /predict with Random Forest...")
        resp = client.post(
            "/predict",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            data={"model_name": "random_forest"}
        )
        assert resp.status_code == 200, f"/predict RF failed: {resp.text}"
        rf_res = resp.json()
        print(f"[OK] RF Prediction: {rf_res['prediction']} ({rf_res['confidence_score']:.2%})")
        assert "prediction" in rf_res and "confidence_score" in rf_res

        # 3. Test /predict with 1D CNN (Deep Learning + Grad-CAM)
        print("\n[3/5] Testing POST /predict with 1D CNN (Grad-CAM)...")
        resp = client.post(
            "/predict",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            data={"model_name": "cnn"}
        )
        assert resp.status_code == 200, f"/predict CNN failed: {resp.text}"
        cnn_res = resp.json()
        print(f"[OK] CNN Prediction: {cnn_res['prediction']} ({cnn_res['confidence_score']:.2%})")
        assert cnn_res.get("model_type") == "Deep Learning"
        assert "explainability" in cnn_res, "Expected explainability in CNN prediction"
        exp = cnn_res["explainability"]
        assert "image_base64" in exp and exp["image_base64"].startswith("data:image/png;base64,")
        assert len(exp["heatmap"]) == 178
        print(f"[OK] Grad-CAM explainability payload verified: heatmap points={len(exp['heatmap'])}, image_b64={len(exp['image_base64'])} chars")

        # 4. Test /predict with BiLSTM
        print("\n[4/5] Testing POST /predict with BiLSTM...")
        resp = client.post(
            "/predict",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            data={"model_name": "bilstm"}
        )
        assert resp.status_code == 200, f"/predict BiLSTM failed: {resp.text}"
        bilstm_res = resp.json()
        print(f"[OK] BiLSTM Prediction: {bilstm_res['prediction']} ({bilstm_res['confidence_score']:.2%})")
        assert bilstm_res.get("model_type") == "Deep Learning"

        # 5. Test /compare (All Models)
        print("\n[5/5] Testing POST /compare across all models...")
        resp = client.post(
            "/compare",
            files={"file": ("test.csv", csv_bytes, "text/csv")}
        )
        assert resp.status_code == 200, f"/compare failed: {resp.text}"
        comp_res = resp.json()
        results = comp_res.get("results", [])
        print(f"[OK] Comparison complete for {len(results)} models:")
        for r in results:
            print(f"   • {r['model_name']} ({r.get('model_type')}): {r.get('prediction')} - {r.get('confidence_percentage')}% (latency: {r.get('latency_ms')}ms)")
        assert len(results) >= 6

    print("\n[SUCCESS] ALL API & DEEP LEARNING INTEGRATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
