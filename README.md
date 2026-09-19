# EEG-Based Epileptic Seizure Detection (ML + DL)

This project builds an automated seizure detection system from EEG signals using:

1. Signal processing
2. Traditional machine learning
3. Deep learning (1D CNN + BiLSTM)
4. Explainability (Grad-CAM and optional SHAP)
5. Real-time sliding-window simulation

It is designed as a final year research project and is suitable for experimentation, benchmarking, and paper-style reporting.

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Enabled-green)
![Deep%20Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)

## What Is Implemented

### Traditional ML Pipeline
1. Data loading and cleaning
2. Standard scaling
3. Binary mapping: seizure vs non-seizure
4. Models: Logistic Regression, SVM (RBF), Random Forest, XGBoost
5. FastAPI inference endpoints for prediction and model comparison

### New Deep Learning and Research Pipeline
1. EEG filtering
2. Wavelet feature extraction
3. Class balancing with SMOTE
4. 1D CNN model
5. BiLSTM model
6. Metrics and comparison reporting
7. Plots for confusion matrix, ROC, and training curves
8. Explainability with Grad-CAM (and optional SHAP)
9. Real-time simulation with sliding windows
10. Ablation study for component importance

## Project Structure

```text
.
|-- backend/
|   |-- app.py
|   |-- predict.py
|   `-- preprocess.py
|-- deep_learning/
|   |-- __init__.py
|   |-- config.py
|   |-- signal_processing.py
|   |-- wavelet_features.py
|   |-- data_pipeline.py
|   |-- cnn_model.py
|   |-- lstm_model.py
|   |-- evaluate.py
|   |-- visualize.py
|   |-- explainability.py
|   |-- realtime_simulation.py
|   |-- ablation_study.py
|   `-- run_pipeline.py
|-- frontend/
|-- models/
|-- data/
|-- train_models.py
|-- test_api.py
`-- requirements.txt
```

## Dataset

The project uses the UCI Epileptic Seizure Recognition dataset.

1. Total samples: 11,500
2. Signal length: 178 points per sample (about 1 second)
3. Original labels: 1 to 5
4. Binary task mapping:
   1. Label 1 = Seizure
   2. Labels 2, 3, 4, 5 = Non-seizure

## Installation

1. Clone the repository

```bash
git clone https://github.com/anyakumar/epiliepsy-project.git
cd epiliepsy-project
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### A) Run Web Interface (ML + Deep Learning + Grad-CAM)

1. Start FastAPI backend:

```bash
uvicorn backend.app:app --reload
```

2. Open the web interface in your browser:

- **EEG Analysis & Grad-CAM Heatmap**: `http://127.0.0.1:8000/analysis.html`
  - Select **1D CNN** to view the live EEG signal with an overlay of the Grad-CAM temporal attention heatmap and critical time windows.
  - Select **BiLSTM** or classical models (**XGBoost**, **Random Forest**, **SVM**, **Logistic Regression**).
- **Multi-Model Comparison**: `http://127.0.0.1:8000/compare.html`
  - Runs all 6 classical and deep learning models concurrently with real-time latency and confidence benchmarking.

### B) Run Automated Tests

```bash
# Run unit tests for Preprocessor, ModelManager, and Grad-CAM:
python -m unittest tests/test_predict.py

# Run end-to-end FastAPI integration tests:
python test_api.py
```

### C) Train baseline ML models

```bash
python train_models.py
```

### C) Run full deep learning pipeline

```bash
python -m deep_learning.run_pipeline
```

### D) Faster run (skip heavy steps)

```bash
python -m deep_learning.run_pipeline --skip-shap --skip-ablation
```

## Deep Learning Modules (Simple Explanation)

### 1) signal_processing.py
Applies notch filter (50 Hz) and band-pass filter (0.5 to 40 Hz) to remove noise and keep relevant EEG frequencies.

### 2) wavelet_features.py
Uses DWT with db4 and extracts statistical features (mean, std, min, max, energy, entropy, kurtosis, skewness) from each wavelet sub-band.

### 3) data_pipeline.py
Handles split, scaling, optional filtering, optional wavelets, and SMOTE on training data only.

### 4) cnn_model.py
Defines and trains a 1D CNN for EEG classification.

### 5) lstm_model.py
Defines and trains an LSTM/BiLSTM model for temporal dependencies.

### 6) evaluate.py
Calculates accuracy, precision, recall, F1, confusion matrix, and ROC-AUC. Saves JSON summaries.

### 7) visualize.py
Creates publication-style plots for confusion matrix, ROC, model comparison, and training history.

### 8) explainability.py
Runs Grad-CAM for CNN windows and optional SHAP for model interpretability.

### 9) realtime_simulation.py
Simulates continuous EEG monitoring with sliding-window inference.

### 10) ablation_study.py
Compares settings such as with/without filtering, with/without SMOTE, and architecture variants.

### 11) run_pipeline.py
Master runner that executes all steps end-to-end.

## Outputs You Get

After running the deep learning pipeline, outputs are saved automatically.

1. Trained models
   1. `models/cnn_best.keras`
   2. `models/bilstm_best.keras`
   3. `models/cnn_final.keras`
   4. `models/bilstm_final.keras`
2. Metrics JSON files
   1. `results/*_results.json`
   2. `results/model_comparison.json`
3. Plots
   1. `results/plots/confusion_matrix_*.png`
   2. `results/plots/roc_curve_*.png`
   3. `results/plots/training_history_*.png`
   4. `results/plots/model_comparison_bar.png`
   5. `results/plots/realtime_simulation.png`
   6. `results/plots/gradcam_*.png`

## Recommended Workflow for Final Year Project

1. Train baseline ML (`train_models.py`)
2. Run DL pipeline with quick flags first
3. Verify plots and metrics
4. Run full pipeline without skip flags for final results
5. Use ablation outputs and ROC/F1 tables in report
6. Include Grad-CAM visualizations in explainability section

## Common Commands

```bash
# Re-run quick DL experiment
python -m deep_learning.run_pipeline --skip-shap --skip-ablation

# Run only API check
python test_api.py

# Check current branch and status
git status
git branch --show-current
```

## Disclaimer

This system is for academic and research use only.
It is not a medical device and must not be used for clinical diagnosis.
Always consult qualified medical professionals for clinical decisions.

## License

This project is licensed under the MIT License.
