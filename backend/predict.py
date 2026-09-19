import joblib
import os
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

MODEL_FILES = {
    'logistic_regression': 'logistic_regression.joblib',
    'svm': 'svm_rbf.joblib',
    'svm_rbf': 'svm_rbf.joblib',
    'random_forest': 'random_forest.joblib',
    'xgboost': 'xgboost.joblib',
    'cnn': 'cnn_best.keras',
    'bilstm': 'bilstm_best.keras'
}

DEEP_LEARNING_MODELS = {'cnn', 'bilstm'}

DISPLAY_MODELS = [
    {
        'id': 'logistic_regression',
        'name': 'Logistic Regression',
        'type': 'Classical ML',
        'supports_gradcam': False
    },
    {
        'id': 'svm_rbf',
        'name': 'SVM (RBF Kernel)',
        'type': 'Classical ML',
        'supports_gradcam': False
    },
    {
        'id': 'random_forest',
        'name': 'Random Forest',
        'type': 'Classical ML',
        'supports_gradcam': False
    },
    {
        'id': 'xgboost',
        'name': 'XGBoost',
        'type': 'Classical ML',
        'supports_gradcam': False
    },
    {
        'id': 'cnn',
        'name': '1D Convolutional Neural Network (CNN)',
        'type': 'Deep Learning',
        'supports_gradcam': True
    },
    {
        'id': 'bilstm',
        'name': 'Bidirectional LSTM (BiLSTM)',
        'type': 'Deep Learning',
        'supports_gradcam': False
    }
]

class ModelManager:
    def __init__(self):
        self.loaded_models = {}

    def get_available_models(self):
        return [m['id'] for m in DISPLAY_MODELS]

    def get_models_metadata(self):
        return DISPLAY_MODELS

    def is_deep_learning(self, model_name: str) -> bool:
        return model_name in DEEP_LEARNING_MODELS

    def supports_gradcam(self, model_name: str) -> bool:
        return model_name == 'cnn'

    def load_model(self, model_name: str):
        # Support alias
        if model_name == 'svm':
            model_name = 'svm_rbf'

        if model_name not in MODEL_FILES:
            raise ValueError(f"Model '{model_name}' not found. Available: {self.get_available_models()}")

        if model_name not in self.loaded_models:
            model_path = os.path.join(MODELS_DIR, MODEL_FILES[model_name])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            print(f"Loading model: {model_name} from {model_path}...")
            if model_name in DEEP_LEARNING_MODELS:
                import tensorflow as tf
                self.loaded_models[model_name] = tf.keras.models.load_model(model_path)
            else:
                self.loaded_models[model_name] = joblib.load(model_path)

        return self.loaded_models[model_name]

    def predict(self, model_name: str, data: np.ndarray):
        model = self.load_model(model_name)

        if model_name in DEEP_LEARNING_MODELS:
            # Reshape input for 1D CNN / BiLSTM: (samples, 178, 1)
            data_dl = data.reshape(data.shape[0], data.shape[1], 1).astype(np.float32)
            raw_preds = model.predict(data_dl, verbose=0)
            seizure_prob = raw_preds.flatten()
            prediction = (seizure_prob >= 0.5).astype(int)
            return prediction, seizure_prob

        # Binary classification for classical ML: 0 = Non-Seizure, 1 = Seizure
        prediction = model.predict(data)

        # Get probability
        try:
            probs = model.predict_proba(data)
            seizure_prob = probs[:, 1]
        except AttributeError:
            seizure_prob = prediction.astype(float)

        return prediction, seizure_prob

    def explain(self, model_name: str, sample: np.ndarray, prediction_label: str = "", confidence_pct: float = 0.0):
        """
        Generates Grad-CAM explainability payload if the model supports it.
        """
        if not self.supports_gradcam(model_name):
            return None

        from deep_learning.explainability import generate_gradcam_payload
        model = self.load_model(model_name)
        return generate_gradcam_payload(model, sample, prediction_label, confidence_pct)
