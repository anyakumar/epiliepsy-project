import joblib
import pandas as pd
import os
import numpy as np

# Path to the scaler relative to this file
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.joblib')

class Preprocessor:
    def __init__(self):
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
        self.scaler = joblib.load(SCALER_PATH)

    def validate_and_scale(self, df: pd.DataFrame):
        """
        Validates the input dataframe and scales it.
        Robustly handles CSVs with index columns, 'y' labels, or non-numeric metadata.
        Expected result: 178 numerical features.
        """
        # 1. Drop 'y' column if present (target label)
        if 'y' in df.columns:
            df = df.drop(columns=['y'])
            
        # 2. Rename columns to generic if they look like the UCI dataset "X1...X178" style
        # mixed with "Unnamed: 0" or "id"
        
        # 3. Select only numeric columns
        features = df.select_dtypes(include=[np.number])
        
        # 4. Check for potential index column (often the first column if it's just sequential ints)
        # Heuristic: If we have 179 or 180 cols, and the first one is just 1..N or "Unnamed: 0"
        if features.shape[1] > 178:
            # Common pattern: First col is index
            # If removing first col gives us exactly 178, assume it was index
            if features.shape[1] == 179:
                features = features.iloc[:, 1:]
            # If removing first AND last gives 178, done (but we already dropped 'y')
        
        # 5. Strict 178 check final pass
        if features.shape[1] != 178:
             raise ValueError(f"Input data must have exactly 178 numerical features. Found {features.shape[1]}.\n"
                              f"Please upload a clean CSV containing only the EEG signal values.")

        # 6. Scale
        scaled_data = self.scaler.transform(features)
        return scaled_data
