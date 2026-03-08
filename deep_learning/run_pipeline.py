"""
═══════════════════════════════════════════════════════════════════════════════
  MAIN DEEP LEARNING PIPELINE
  EEG-based Epileptic Seizure Detection
═══════════════════════════════════════════════════════════════════════════════

This script runs the complete pipeline:

  Step 1 : Load data & preprocess (filter + scale + SMOTE)
  Step 2 : Visualize raw vs filtered EEG signals
  Step 3 : Train 1D CNN model
  Step 4 : Train BiLSTM model
  Step 5 : Evaluate all models (Accuracy, Precision, Recall, F1, ROC-AUC)
  Step 6 : Generate all visualization plots
  Step 7 : Run Grad-CAM explainability on CNN
  Step 8 : Run SHAP explainability (optional, slower)
  Step 9 : Simulate real-time seizure detection
  Step 10: Run ablation study (optional)

Usage:
  cd "Epileptic Seizure Recognition"
  python -m deep_learning.run_pipeline

Or selectively:
  python -m deep_learning.run_pipeline --skip-ablation --skip-shap
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import joblib

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_learning import config
from deep_learning.data_pipeline import (
    load_dataset, binarize_labels, prepare_data,
    reshape_for_cnn, reshape_for_lstm
)
from deep_learning.signal_processing import filter_eeg_signals
from deep_learning.wavelet_features import extract_wavelet_features_batch, get_wavelet_feature_names
from deep_learning.cnn_model import train_cnn, build_cnn_model
from deep_learning.lstm_model import train_lstm, build_lstm_model
from deep_learning.evaluate import evaluate_model, compare_models
from deep_learning.visualize import (
    plot_confusion_matrix, plot_roc_curve, plot_roc_curves_comparison,
    plot_training_history, plot_class_distribution, plot_model_comparison_bar,
    plot_eeg_signal
)
from deep_learning.explainability import grad_cam_1d, plot_grad_cam
from deep_learning.realtime_simulation import run_realtime_simulation, plot_realtime_results


def main(args):
    print("\n" + "=" * 70)
    print("  EEG-BASED EPILEPTIC SEIZURE DETECTION — DEEP LEARNING PIPELINE")
    print("=" * 70)
    start_total = time.time()

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1: Data Loading & Preprocessing
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 1: Data Loading & Preprocessing")
    print("─" * 70)

    # Load raw data for visualization
    X_raw, y_raw = load_dataset()

    # Prepare filtered + SMOTE data for DL
    data = prepare_data(
        apply_filter=True,
        apply_wavelet=False,  # DL models work on raw time-series
        apply_smote=True,
        binary=True,
    )

    # Also prepare wavelet data for ML comparison (optional)
    if not args.skip_wavelet_ml:
        data_wavelet = prepare_data(
            apply_filter=True,
            apply_wavelet=True,
            apply_smote=True,
            binary=True,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2: Visualize EEG Signals
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 2: Visualize EEG Signals")
    print("─" * 70)

    y_binary = binarize_labels(y_raw)
    plot_class_distribution(y_binary, "Binary Class Distribution (Before SMOTE)",
                            "class_dist_binary.png")
    plot_class_distribution(y_raw, "Original 5-Class Distribution", "class_dist_5class.png")

    # Plot a seizure and a non-seizure sample
    seizure_idx = np.where(y_binary == 1)[0][0]
    normal_idx = np.where(y_binary == 0)[0][0]

    raw_seizure = X_raw[seizure_idx]
    filtered_seizure = filter_eeg_signals(raw_seizure.reshape(1, -1))[0]
    plot_eeg_signal(raw_seizure, filtered_seizure, sample_idx=seizure_idx,
                    title="Seizure EEG Signal")

    raw_normal = X_raw[normal_idx]
    filtered_normal = filter_eeg_signals(raw_normal.reshape(1, -1))[0]
    plot_eeg_signal(raw_normal, filtered_normal, sample_idx=normal_idx,
                    title="Non-Seizure EEG Signal")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3: Train 1D CNN
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 3: Train 1D CNN Model")
    print("─" * 70)

    X_train_cnn = reshape_for_cnn(data["X_train"])
    X_val_cnn = reshape_for_cnn(data["X_val"])
    X_test_cnn = reshape_for_cnn(data["X_test"])

    cnn_model, cnn_history = train_cnn(
        X_train_cnn, data["y_train"],
        X_val_cnn, data["y_val"],
        num_classes=2
    )
    plot_training_history(cnn_history, "1D_CNN")

    # Save CNN model
    cnn_save_path = os.path.join(config.MODELS_DIR, "cnn_final.keras")
    cnn_model.save(cnn_save_path)
    print(f"[MODEL] CNN saved to {cnn_save_path}")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4: Train BiLSTM
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 4: Train BiLSTM Model")
    print("─" * 70)

    X_train_lstm = reshape_for_lstm(data["X_train"])
    X_val_lstm = reshape_for_lstm(data["X_val"])
    X_test_lstm = reshape_for_lstm(data["X_test"])

    lstm_model, lstm_history = train_lstm(
        X_train_lstm, data["y_train"],
        X_val_lstm, data["y_val"],
        num_classes=2, bidirectional=True
    )
    plot_training_history(lstm_history, "BiLSTM")

    lstm_save_path = os.path.join(config.MODELS_DIR, "bilstm_final.keras")
    lstm_model.save(lstm_save_path)
    print(f"[MODEL] BiLSTM saved to {lstm_save_path}")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 5: Evaluate All Models
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 5: Evaluate All Models")
    print("─" * 70)

    all_results = []

    # --- CNN Evaluation ---
    y_prob_cnn = cnn_model.predict(X_test_cnn, verbose=0).flatten()
    y_pred_cnn = (y_prob_cnn > 0.5).astype(int)
    cnn_results = evaluate_model(data["y_test"], y_pred_cnn, y_prob_cnn,
                                 model_name="1D_CNN", num_classes=2)
    all_results.append(cnn_results)

    # --- BiLSTM Evaluation ---
    y_prob_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    y_pred_lstm = (y_prob_lstm > 0.5).astype(int)
    lstm_results = evaluate_model(data["y_test"], y_pred_lstm, y_prob_lstm,
                                  model_name="BiLSTM", num_classes=2)
    all_results.append(lstm_results)

    # --- Existing ML Models (if available) ---
    ml_models_to_eval = {
        "Logistic_Regression": "logistic_regression.joblib",
        "SVM_RBF": "svm_rbf.joblib",
        "Random_Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
    }

    scaler_path = os.path.join(config.MODELS_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        ml_scaler = joblib.load(scaler_path)
        # Use unfiltered raw data for ML models (as they were originally trained)
        X_raw_all, y_raw_all = load_dataset()
        y_binary_all = binarize_labels(y_raw_all)
        from sklearn.model_selection import train_test_split
        _, X_test_ml, _, y_test_ml = train_test_split(
            X_raw_all, y_binary_all, test_size=0.2, random_state=42, stratify=y_binary_all
        )
        # Drop index column same as train_models.py
        if X_test_ml.shape[1] > 178:
            X_test_ml = X_test_ml[:, 1:]
        X_test_ml_scaled = ml_scaler.transform(X_test_ml)

        for name, filename in ml_models_to_eval.items():
            model_path = os.path.join(config.MODELS_DIR, filename)
            if os.path.exists(model_path):
                print(f"\n[EVAL] Evaluating existing ML model: {name}")
                ml_model = joblib.load(model_path)
                y_pred_ml = ml_model.predict(X_test_ml_scaled)
                try:
                    y_prob_ml = ml_model.predict_proba(X_test_ml_scaled)[:, 1]
                except AttributeError:
                    y_prob_ml = None
                ml_result = evaluate_model(y_test_ml, y_pred_ml, y_prob_ml,
                                           model_name=name, num_classes=2)
                all_results.append(ml_result)

    # Compare all models
    compare_models(all_results)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 6: Generate All Visualizations
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 6: Generate Visualization Plots")
    print("─" * 70)

    # Confusion matrices
    plot_confusion_matrix(data["y_test"], y_pred_cnn, "1D_CNN")
    plot_confusion_matrix(data["y_test"], y_pred_lstm, "BiLSTM")

    # Individual ROC curves
    plot_roc_curve(data["y_test"], y_prob_cnn, "1D_CNN")
    plot_roc_curve(data["y_test"], y_prob_lstm, "BiLSTM")

    # Overlay ROC comparison
    roc_data = {}
    for r in all_results:
        if r.get("roc_curve"):
            roc_data[r["model_name"]] = {
                "fpr": r["roc_curve"]["fpr"],
                "tpr": r["roc_curve"]["tpr"],
                "auc": r.get("roc_auc", 0),
            }
    if roc_data:
        plot_roc_curves_comparison(roc_data)

    # Model comparison bar chart
    plot_model_comparison_bar(all_results)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 7: Grad-CAM Explainability (CNN)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 7: Grad-CAM Explainability")
    print("─" * 70)

    # Explain a few seizure and non-seizure samples
    seizure_test_indices = np.where(data["y_test"] == 1)[0][:3]
    normal_test_indices = np.where(data["y_test"] == 0)[0][:3]

    for idx in list(seizure_test_indices) + list(normal_test_indices):
        sample = X_test_cnn[idx:idx + 1]
        heatmap = grad_cam_1d(cnn_model, sample)
        pred_label = "Seizure" if y_pred_cnn[idx] == 1 else "Non-Seizure"
        original_signal = data["X_test"][idx]
        plot_grad_cam(original_signal, heatmap, prediction=pred_label, sample_idx=idx)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 8: SHAP Explainability (Optional)
    # ═════════════════════════════════════════════════════════════════════════
    if not args.skip_shap:
        print("\n" + "─" * 70)
        print("  STEP 8: SHAP Explainability")
        print("─" * 70)
        try:
            from deep_learning.explainability import explain_with_shap

            # SHAP for Random Forest (tree-based, fast)
            rf_path = os.path.join(config.MODELS_DIR, "random_forest.joblib")
            if os.path.exists(rf_path) and os.path.exists(scaler_path):
                rf_model = joblib.load(rf_path)
                feature_names = [f"X{i+1}" for i in range(X_test_ml_scaled.shape[1])]
                explain_with_shap(rf_model, X_test_ml_scaled[:500],
                                  feature_names=feature_names,
                                  model_name="Random_Forest", model_type="tree")
        except ImportError:
            print("[SHAP] shap library not installed. Skipping. Install with: pip install shap")
        except Exception as e:
            print(f"[SHAP] Error: {e}")
    else:
        print("\n[SKIP] SHAP explainability skipped (use --no-skip-shap to enable)")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 9: Real-Time Simulation
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 9: Real-Time Seizure Detection Simulation")
    print("─" * 70)

    # Create a simulated continuous EEG stream by concatenating test samples
    n_samples_sim = min(50, len(data["X_test"]))
    continuous_signal = data["X_test"][:n_samples_sim].flatten()
    # Note: data is already scaled, we need the scaler for the simulation
    # We'll use the scaler from data preparation
    sim_results = run_realtime_simulation(
        model=cnn_model,
        scaler=data["scaler"],
        signal=continuous_signal,
        model_type="dl",
        simulate_delay=False  # Skip artificial delay for testing
    )
    plot_realtime_results(sim_results, "CNN Real-Time Simulation")

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 10: Ablation Study (Optional)
    # ═════════════════════════════════════════════════════════════════════════
    if not args.skip_ablation:
        print("\n" + "─" * 70)
        print("  STEP 10: Ablation Study")
        print("─" * 70)
        from deep_learning.ablation_study import run_ablation_study
        ablation_results = run_ablation_study(quick_mode=True)
    else:
        print("\n[SKIP] Ablation study skipped (use --no-skip-ablation to enable)")

    # ═════════════════════════════════════════════════════════════════════════
    # DONE
    # ═════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE! Total time: {elapsed / 60:.1f} minutes")
    print(f"  Results saved to: {config.RESULTS_DIR}")
    print(f"  Plots saved to:   {config.PLOTS_DIR}")
    print(f"  Models saved to:  {config.MODELS_DIR}")
    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the EEG Deep Learning Seizure Detection Pipeline"
    )
    parser.add_argument("--skip-shap", action="store_true", default=False,
                        help="Skip SHAP explainability (faster)")
    parser.add_argument("--skip-ablation", action="store_true", default=False,
                        help="Skip ablation study (faster)")
    parser.add_argument("--skip-wavelet-ml", action="store_true", default=False,
                        help="Skip wavelet feature ML comparison")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
