import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # correct path
    DATA_PATH = 'data/data.csv'
    MODELS_DIR = 'models'
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {DATA_PATH}")
        return

    # Preprocessing
    print("Preprocessing data...")
    
    # The dataset usually has the label in the last column, often named 'y' or similar.
    # We'll drop the first column if it's an index (often named "Unnamed" or "X")
    # Features are X1 to X178
    
    # Let's inspect columns briefly (conceptually)
    # The UCI dataset typically has 179 columns: X1..X178, y
    
    X = df.iloc[:, 1:-1] # Features: Exclude first col (name/index) and last col (label)
    y = df.iloc[:, -1]   # Label: Last column
    
    # Binary Classification:
    # 1 - Seizure
    # 2,3,4,5 - Non-Seizure
    # We want 1 -> 1, others -> 0
    y_binary = (y == 1).astype(int)
    
    print(f"Class balance: {y_binary.value_counts().to_dict()}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)
    
    # Scaling
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    print("Scaler saved.")

    # Dictionary of models
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'svm_rbf': SVC(kernel='rbf', probability=True, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Training loop
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        
        # Save
        save_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, save_path)
        print(f"Saved to {save_path}")

    print("\nAll models trained and saved successfully.")

if __name__ == "__main__":
    main()
