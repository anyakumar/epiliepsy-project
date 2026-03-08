"""
Evaluation Module
-----------------
Comprehensive model evaluation with:
  - Accuracy, Precision, Recall, F1-score
  - Classification report
  - Confusion matrix
  - ROC-AUC (binary and multi-class OvR)
  - Results saved to JSON for paper reporting
"""

import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from . import config


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   y_prob: np.ndarray = None,
                   model_name: str = "model",
                   num_classes: int = 2,
                   save: bool = True) -> dict:
    """
    Compute all evaluation metrics for a model.

    Parameters
    ----------
    y_true      : Ground truth labels
    y_pred      : Predicted labels (argmax for multi-class)
    y_prob      : Predicted probabilities (for ROC-AUC)
                  - Binary: shape (n,) or (n, 1) — probability of class 1
                  - Multi:  shape (n, num_classes)
    model_name  : Name for saving results
    num_classes : 2 or 5
    save        : Save results to JSON

    Returns
    -------
    results : dict with all metrics
    """
    results = {"model_name": model_name}

    # Basic metrics
    avg = "binary" if num_classes == 2 else "weighted"
    results["accuracy"] = float(accuracy_score(y_true, y_pred))
    results["precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    results["recall"] = float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    results["f1_score"] = float(f1_score(y_true, y_pred, average=avg, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results["confusion_matrix"] = cm.tolist()

    # Classification report (dict form)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    results["classification_report"] = report

    # ROC-AUC
    if y_prob is not None:
        try:
            if num_classes == 2:
                # Binary ROC-AUC
                if y_prob.ndim == 2:
                    prob_pos = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob[:, 0]
                else:
                    prob_pos = y_prob
                results["roc_auc"] = float(roc_auc_score(y_true, prob_pos))

                # ROC curve data
                fpr, tpr, thresholds = roc_curve(y_true, prob_pos)
                results["roc_curve"] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                }
            else:
                # Multi-class: One-vs-Rest
                y_bin = label_binarize(y_true, classes=list(range(num_classes)))
                results["roc_auc"] = float(roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr"))

                # Per-class ROC
                roc_data = {}
                for i in range(num_classes):
                    fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                    roc_data[f"class_{i}"] = {
                        "fpr": fpr_i.tolist(),
                        "tpr": tpr_i.tolist(),
                        "auc": float(auc(fpr_i, tpr_i)),
                    }
                results["roc_curves"] = roc_data

        except Exception as e:
            results["roc_auc"] = None
            results["roc_auc_error"] = str(e)
    else:
        results["roc_auc"] = None

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Evaluation Results — {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1_score']:.4f}")
    if results["roc_auc"] is not None:
        print(f"  ROC-AUC   : {results['roc_auc']:.4f}")
    print(f"{'='*60}")
    print(f"  Confusion Matrix:\n{np.array(cm)}")
    print(f"{'='*60}\n")

    # Save to JSON
    if save:
        save_path = os.path.join(config.RESULTS_DIR, f"{model_name}_results.json")
        # Convert numpy types for JSON serialization
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[EVAL] Results saved to {save_path}")

    return results


def compare_models(results_list: list) -> None:
    """
    Print a comparison table of all models.

    Parameters
    ----------
    results_list : List of result dicts from evaluate_model()
    """
    print(f"\n{'='*80}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*80}")
    header = f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
    print(header)
    print(f"  {'-'*70}")
    for r in results_list:
        roc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "N/A"
        print(f"  {r['model_name']:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1_score']:>10.4f} {roc:>10}")
    print(f"{'='*80}\n")

    # Save comparison
    save_path = os.path.join(config.RESULTS_DIR, "model_comparison.json")
    comparison = []
    for r in results_list:
        comparison.append({
            "model": r["model_name"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1_score": r["f1_score"],
            "roc_auc": r.get("roc_auc"),
        })
    with open(save_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[EVAL] Comparison saved to {save_path}")
