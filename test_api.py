import requests
import pandas as pd
import numpy as np
import io
import time
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    # 1. Check health/models
    print("Checking /models...")
    try:
        resp = requests.get(f"{BASE_URL}/models")
        if resp.status_code == 200:
            print("âœ… /models working:", resp.json())
        else:
            print("âŒ /models failed:", resp.text)
            return
    except requests.exceptions.ConnectionError:
        print("âŒ Could not connect to server. Is it running?")
        return

    # 2. Create dummy data
    print("\nCreating dummy EEG data (1 row, 178 features)...")
    # Random data resembling normalized EEG
    dummy_data = np.random.randn(1, 178)
    df = pd.DataFrame(dummy_data)
    
    # Save to memory buffer
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, header=False)
    csv_buffer.seek(0)
    
    # 3. Predict
    model = "random_forest"
    print(f"\nRequesting prediction using {model}...")
    
    files = {'file': ('test.csv', csv_buffer.getvalue(), 'text/csv')}
    data = {'model_name': model}
    
    resp = requests.post(f"{BASE_URL}/predict", files=files, data=data)
    
    if resp.status_code == 200:
        print("✅ /predict success!")
        data = resp.json()
        print(data)
        
        # Validation
        required_fields = ["prediction", "confidence_score", "confidence_label", "explanation", "disclaimer"]
        missing = [f for f in required_fields if f not in data]
        if missing:
             print(f"❌ Missing fields: {missing}")
        else:
             print("✅ All required fields present.")
             print(f"Confidence Label: {data['confidence_label']}")
             print(f"Disclaimer: {data['disclaimer']}")
    else:
        print("❌ /predict failed:", resp.text)

if __name__ == "__main__":
    test_api()
