import joblib
import os
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

MODEL_FILES = {
    'logistic_regression': 'logistic_regression.joblib',
    'svm': 'svm_rbf.joblib',
    'random_forest': 'random_forest.joblib',
    'xgboost': 'xgboost.joblib'
}

class ModelManager:
    def __init__(self):
        self.loaded_models = {}

    def get_available_models(self):
        return list(MODEL_FILES.keys())

    def load_model(self, model_name: str):
        if model_name not in MODEL_FILES:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(MODEL_FILES.keys())}")
        
        if model_name not in self.loaded_models:
            model_path = os.path.join(MODELS_DIR, MODEL_FILES[model_name])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            
            print(f"Loading model: {model_name}...")
            self.loaded_models[model_name] = joblib.load(model_path)
            
        return self.loaded_models[model_name]

    def predict(self, model_name: str, data: np.ndarray):
        model = self.load_model(model_name)
        
        # Binary classification: 0 = Non-Seizure, 1 = Seizure
        prediction = model.predict(data)
        
        # Get probability
        # All our trained models support predict_proba (SVM has probability=True)
        try:
            probs = model.predict_proba(data)
            # Probability of class 1 (Seizure)
            seizure_prob = probs[:, 1]
        except AttributeError:
            # Fallback for models that might not support it (though ours do)
            seizure_prob = prediction.astype(float)
            
        return prediction, seizure_prob
