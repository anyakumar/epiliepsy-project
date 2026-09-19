import unittest
import numpy as np
import pandas as pd
import os
import sys

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.preprocess import Preprocessor
from backend.predict import ModelManager
from deep_learning.explainability import grad_cam_1d, generate_gradcam_payload


class TestModelPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preprocessor = Preprocessor()
        cls.model_manager = ModelManager()
        # Create deterministic synthetic sample: 1 sample, 178 features
        np.random.seed(42)
        cls.dummy_raw = np.random.randn(2, 178)
        cls.dummy_df = pd.DataFrame(cls.dummy_raw)

    def test_preprocessor_validation_and_scale(self):
        scaled = self.preprocessor.validate_and_scale(self.dummy_df)
        self.assertEqual(scaled.shape, (2, 178))
        self.assertTrue(np.all(np.isfinite(scaled)))

    def test_available_models(self):
        models = self.model_manager.get_available_models()
        self.assertIn('cnn', models)
        self.assertIn('bilstm', models)
        self.assertIn('xgboost', models)
        self.assertIn('random_forest', models)
        self.assertIn('svm_rbf', models)
        self.assertIn('logistic_regression', models)

    def test_predict_classical_models(self):
        scaled = self.preprocessor.validate_and_scale(self.dummy_df)
        for model_name in ['xgboost', 'random_forest', 'svm_rbf', 'logistic_regression']:
            preds, probs = self.model_manager.predict(model_name, scaled)
            self.assertEqual(len(preds), 2)
            self.assertEqual(len(probs), 2)
            self.assertTrue(all(p in (0, 1) for p in preds))
            self.assertTrue(all(0.0 <= prob <= 1.0 for prob in probs))

    def test_predict_deep_learning_models(self):
        scaled = self.preprocessor.validate_and_scale(self.dummy_df)
        for model_name in ['cnn', 'bilstm']:
            preds, probs = self.model_manager.predict(model_name, scaled)
            self.assertEqual(len(preds), 2)
            self.assertEqual(len(probs), 2)
            self.assertTrue(all(p in (0, 1) for p in preds))
            self.assertTrue(all(0.0 <= prob <= 1.0 for prob in probs))

    def test_gradcam_explainability_cnn(self):
        scaled = self.preprocessor.validate_and_scale(self.dummy_df)
        payload = self.model_manager.explain('cnn', scaled[0], 'Seizure', 96.5)
        self.assertIsNotNone(payload)
        self.assertIn('heatmap', payload)
        self.assertIn('image_base64', payload)
        self.assertEqual(len(payload['heatmap']), 178)
        self.assertTrue(payload['image_base64'].startswith('data:image/png;base64,'))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in payload['heatmap']))

    def test_explainability_unsupported_model(self):
        scaled = self.preprocessor.validate_and_scale(self.dummy_df)
        payload = self.model_manager.explain('xgboost', scaled[0], 'Seizure', 90.0)
        self.assertIsNone(payload)


if __name__ == '__main__':
    unittest.main()
