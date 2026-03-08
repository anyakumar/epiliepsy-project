"""
Ablation Study Module
---------------------
Systematically evaluates the contribution of individual components:

Experiments:
  1. Raw signal vs Filtered signal
  2. Raw features vs Wavelet features vs Raw+Wavelet (combined)
  3. Without SMOTE vs With SMOTE
  4. CNN-only vs LSTM-only vs CNN+LSTM ensemble
  5. Feature importance ranking

Results are stored in a structured JSON for easy paper reporting.
"""

import os
import json
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from . import config
from .data_pipeline import prepare_data, reshape_for_cnn
from .cnn_model import build_cnn_model
from .evaluate import evaluate_model


def run_ablation_study(quick_mode: bool = True):
    """
    Run a comprehensive ablation study.

    Parameters
    ----------
    quick_mode : If True, uses fewer epochs and a simpler model for speed.
                 Set False for full paper-quality results.

    Returns
    -------
    results : dict with all ablation experiment results
    """
    epochs = 10 if quick_mode else config.CNN_EPOCHS
    results = {}

    print("\n" + "=" * 70)
    print("  ABLATION STUDY")
    print("=" * 70)

    # ─── Experiment 1: Impact of Signal Filtering ────────────────────────────
    print("\n[ABLATION] Experiment 1: Raw vs Filtered signals ...")

    # 1a: No filtering
    data_raw = prepare_data(apply_filter=False, apply_wavelet=False,
                            apply_smote=True, binary=True)
    acc_raw = _quick_eval(data_raw, "RF")
    results["no_filter"] = acc_raw

    # 1b: With filtering
    data_filt = prepare_data(apply_filter=True, apply_wavelet=False,
                             apply_smote=True, binary=True)
    acc_filt = _quick_eval(data_filt, "RF")
    results["with_filter"] = acc_filt

    print(f"  Raw signals:      Acc={acc_raw['accuracy']:.4f}, F1={acc_raw['f1']:.4f}")
    print(f"  Filtered signals: Acc={acc_filt['accuracy']:.4f}, F1={acc_filt['f1']:.4f}")

    # ─── Experiment 2: Feature Type Comparison ───────────────────────────────
    print("\n[ABLATION] Experiment 2: Raw vs Wavelet features ...")

    # 2a: Raw features (already computed as data_filt)
    results["raw_features"] = acc_filt

    # 2b: Wavelet features
    data_wav = prepare_data(apply_filter=True, apply_wavelet=True,
                            apply_smote=True, binary=True)
    acc_wav = _quick_eval(data_wav, "RF")
    results["wavelet_features"] = acc_wav

    print(f"  Raw features:     Acc={acc_wav['accuracy']:.4f}, F1={acc_wav['f1']:.4f}")
    print(f"  Wavelet features: Acc={acc_wav['accuracy']:.4f}, F1={acc_wav['f1']:.4f}")

    # ─── Experiment 3: Impact of SMOTE ───────────────────────────────────────
    print("\n[ABLATION] Experiment 3: Without SMOTE vs With SMOTE ...")

    data_no_smote = prepare_data(apply_filter=True, apply_wavelet=False,
                                  apply_smote=False, binary=True)
    acc_no_smote = _quick_eval(data_no_smote, "RF")
    results["no_smote"] = acc_no_smote
    results["with_smote"] = acc_filt  # Already computed with SMOTE

    print(f"  Without SMOTE: Acc={acc_no_smote['accuracy']:.4f}, F1={acc_no_smote['f1']:.4f}")
    print(f"  With SMOTE:    Acc={acc_filt['accuracy']:.4f}, F1={acc_filt['f1']:.4f}")

    # ─── Experiment 4: Model Architecture Comparison ─────────────────────────
    print("\n[ABLATION] Experiment 4: CNN depth comparison ...")

    data = prepare_data(apply_filter=True, apply_wavelet=False,
                        apply_smote=True, binary=True)
    X_train = reshape_for_cnn(data["X_train"])
    X_val = reshape_for_cnn(data["X_val"])
    X_test = reshape_for_cnn(data["X_test"])

    # 4a: Shallow CNN (1 conv layer)
    model_shallow = build_cnn_model(
        input_shape=(X_train.shape[1], 1),
        filters=[64], num_classes=2
    )
    model_shallow.fit(X_train, data["y_train"], validation_data=(X_val, data["y_val"]),
                      epochs=epochs, batch_size=64, verbose=0)
    y_pred_shallow = (model_shallow.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    y_prob_shallow = model_shallow.predict(X_test, verbose=0).flatten()
    results["cnn_1_layer"] = {
        "accuracy": float(accuracy_score(data["y_test"], y_pred_shallow)),
        "f1": float(f1_score(data["y_test"], y_pred_shallow)),
        "roc_auc": float(roc_auc_score(data["y_test"], y_prob_shallow)),
    }

    # 4b: Deep CNN (3 conv layers - default)
    model_deep = build_cnn_model(
        input_shape=(X_train.shape[1], 1),
        filters=[64, 128, 64], num_classes=2
    )
    model_deep.fit(X_train, data["y_train"], validation_data=(X_val, data["y_val"]),
                   epochs=epochs, batch_size=64, verbose=0)
    y_pred_deep = (model_deep.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    y_prob_deep = model_deep.predict(X_test, verbose=0).flatten()
    results["cnn_3_layers"] = {
        "accuracy": float(accuracy_score(data["y_test"], y_pred_deep)),
        "f1": float(f1_score(data["y_test"], y_pred_deep)),
        "roc_auc": float(roc_auc_score(data["y_test"], y_prob_deep)),
    }

    print(f"  CNN (1 layer): Acc={results['cnn_1_layer']['accuracy']:.4f}")
    print(f"  CNN (3 layers): Acc={results['cnn_3_layers']['accuracy']:.4f}")

    # ─── Save Results ────────────────────────────────────────────────────────
    print("\n[ABLATION] Summary:")
    for key, val in results.items():
        if isinstance(val, dict):
            print(f"  {key:25s} → Acc={val.get('accuracy', 'N/A'):.4f}, F1={val.get('f1', 'N/A'):.4f}")

    save_path = os.path.join(config.RESULTS_DIR, "ablation_study_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[ABLATION] Results saved to {save_path}")

    return results


def _quick_eval(data: dict, model_type: str = "RF") -> dict:
    """
    Quick evaluation using a Random Forest for ablation speed.
    """
    if model_type == "RF":
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(data["X_train"], data["y_train"])
    y_pred = model.predict(data["X_test"])
    y_prob = model.predict_proba(data["X_test"])[:, 1]

    return {
        "accuracy": float(accuracy_score(data["y_test"], y_pred)),
        "f1": float(f1_score(data["y_test"], y_pred)),
        "roc_auc": float(roc_auc_score(data["y_test"], y_prob)),
    }


# ─── Run directly ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_ablation_study(quick_mode=True)
