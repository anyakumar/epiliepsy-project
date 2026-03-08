"""
Visualization Module
--------------------
Generates publication-quality plots for:
  1. Confusion matrix (heatmap)
  2. ROC curves (single & multi-model overlay)
  3. Training vs validation accuracy/loss curves
  4. Class distribution bar chart
  5. Model comparison bar chart
  6. EEG signal plot (raw vs filtered)

All plots saved to results/plots/.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from . import config

# Plotting style
sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = sns.color_palette("Set2", 10)


def plot_confusion_matrix(y_true, y_pred, model_name: str = "model",
                          class_names: list = None, normalize: bool = True):
    """
    Plot and save a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
        title = f"Confusion Matrix (Normalized) — {model_name}"
    else:
        cm_display = cm
        fmt = "d"
        title = f"Confusion Matrix — {model_name}"

    if class_names is None:
        class_names = ["Non-Seizure", "Seizure"] if cm.shape[0] == 2 else [f"Class {i}" for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, square=True)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Confusion matrix saved to {path}")


def plot_roc_curve(y_true, y_prob, model_name: str = "model"):
    """
    Plot ROC curve for a single binary classifier.

    Parameters
    ----------
    y_prob : Probability of the positive class (seizure).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=COLORS[0], lw=2,
            label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, f"roc_curve_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] ROC curve saved to {path}")


def plot_roc_curves_comparison(roc_data: dict):
    """
    Overlay ROC curves from multiple models on one plot.

    Parameters
    ----------
    roc_data : dict of {model_name: {"fpr": [...], "tpr": [...], "auc": float}}
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, data) in enumerate(roc_data.items()):
        fpr = np.array(data["fpr"])
        tpr = np.array(data["tpr"])
        roc_auc = data.get("auc", auc(fpr, tpr))
        ax.plot(fpr, tpr, color=COLORS[i % len(COLORS)], lw=2,
                label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison — All Models")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "roc_curves_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] ROC comparison saved to {path}")


def plot_training_history(history, model_name: str = "model"):
    """
    Plot training vs validation accuracy and loss from a Keras History object.

    Parameters
    ----------
    history : keras.callbacks.History or dict with keys 'accuracy', 'val_accuracy', 'loss', 'val_loss'
    """
    if hasattr(history, "history"):
        history = history.history

    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(epochs, history["accuracy"], "-o", color=COLORS[0], label="Train Accuracy", markersize=3)
    axes[0].plot(epochs, history["val_accuracy"], "-o", color=COLORS[1], label="Val Accuracy", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"Accuracy — {model_name}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(epochs, history["loss"], "-o", color=COLORS[2], label="Train Loss", markersize=3)
    axes[1].plot(epochs, history["val_loss"], "-o", color=COLORS[3], label="Val Loss", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"Loss — {model_name}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"training_history_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Training history saved to {path}")


def plot_class_distribution(y, title: str = "Class Distribution",
                            filename: str = "class_distribution.png"):
    """
    Bar chart of class label counts.
    """
    unique, counts = np.unique(y, return_counts=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(unique.astype(str), counts, color=COLORS[:len(unique)], edgecolor="black")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Class distribution saved to {path}")


def plot_model_comparison_bar(results_list: list):
    """
    Grouped bar chart comparing models on Accuracy, Precision, Recall, F1, ROC-AUC.
    """
    model_names = [r["model_name"] for r in results_list]
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    values = {m: [] for m in metrics}
    for r in results_list:
        for m in metrics:
            val = r.get(m)
            values[m].append(val if val is not None else 0)

    x = np.arange(len(model_names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        offset = (i - len(metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values[metric], width, label=label, color=COLORS[i])

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "model_comparison_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] Model comparison bar chart saved to {path}")


def plot_eeg_signal(raw_signal, filtered_signal=None, sample_idx: int = 0,
                    title: str = "EEG Signal"):
    """
    Plot a single raw vs filtered EEG segment.
    """
    t = np.arange(len(raw_signal)) / config.SAMPLING_RATE

    fig, axes = plt.subplots(2 if filtered_signal is not None else 1, 1,
                              figsize=(14, 5), sharex=True)
    if filtered_signal is None:
        axes = [axes]

    axes[0].plot(t, raw_signal, color="black", alpha=0.7, linewidth=0.8)
    axes[0].set_title(f"{title} — Raw (Sample #{sample_idx})")
    axes[0].set_ylabel("Amplitude")

    if filtered_signal is not None:
        axes[1].plot(t, filtered_signal, color="blue", alpha=0.7, linewidth=0.8)
        axes[1].set_title(f"{title} — Filtered (Sample #{sample_idx})")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_xlabel("Time (s)")
    else:
        axes[0].set_xlabel("Time (s)")

    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"eeg_signal_sample_{sample_idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIZ] EEG signal plot saved to {path}")
