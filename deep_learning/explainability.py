"""
Model Explainability Module
---------------------------
Provides:
  1. SHAP (SHapley Additive exPlanations) for ML and DL models
  2. Grad-CAM for the 1D CNN model

These help interpret *why* the model made a particular prediction,
which is critical for medical/clinical AI systems.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


# ═══════════════════════════════════════════════════════════════════════════════
# SHAP Explainability
# ═══════════════════════════════════════════════════════════════════════════════

def explain_with_shap(model, X_sample: np.ndarray, feature_names: list = None,
                      model_name: str = "model", max_display: int = 20,
                      model_type: str = "tree"):
    """
    Generate SHAP explanations for a trained model.

    Parameters
    ----------
    model         : Trained sklearn/xgboost model or keras model
    X_sample      : Sample data for explanation, shape (n_samples, n_features)
    feature_names : List of feature names
    model_name    : For saving plots
    max_display   : Max features to show in summary plot
    model_type    : 'tree' for tree-based models (RF, XGBoost),
                    'kernel' for any model (slower),
                    'deep' for neural networks (Keras/TF)

    Returns
    -------
    shap_values : SHAP values array
    """
    import shap  # Import here to make it optional

    print(f"[SHAP] Computing SHAP values for {model_name} (type={model_type}) ...")

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    elif model_type == "kernel":
        # Use a small background dataset for KernelExplainer
        background = shap.kmeans(X_sample, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample[:100])  # Limit for speed
    elif model_type == "deep":
        # For Keras models — expects 3D input for CNN/LSTM
        explainer = shap.GradientExplainer(model, X_sample[:200])
        shap_values = explainer.shap_values(X_sample[:100])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Summary plot (bar)
    fig = plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list):
        # Multi-class: use class 1 (seizure)
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    if sv.ndim == 3:
        # CNN/LSTM: (samples, timesteps, channels) → flatten last dims
        sv = sv.reshape(sv.shape[0], -1)
        X_plot = X_sample[:sv.shape[0]].reshape(sv.shape[0], -1)
    else:
        X_plot = X_sample[:sv.shape[0]]

    shap.summary_plot(sv, X_plot, feature_names=feature_names,
                      max_display=max_display, show=False)
    path = os.path.join(config.PLOTS_DIR, f"shap_summary_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Summary plot saved to {path}")

    # Bar plot (mean absolute SHAP)
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(sv, X_plot, feature_names=feature_names,
                      max_display=max_display, plot_type="bar", show=False)
    path_bar = os.path.join(config.PLOTS_DIR, f"shap_bar_{model_name}.png")
    plt.savefig(path_bar, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Bar plot saved to {path_bar}")

    return shap_values


# ═══════════════════════════════════════════════════════════════════════════════
# Grad-CAM for 1D CNN
# ═══════════════════════════════════════════════════════════════════════════════

def grad_cam_1d(model, input_signal: np.ndarray, target_class: int = None,
                last_conv_layer_name: str = None) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a 1D CNN model.

    This highlights which temporal regions of the EEG signal
    are most important for the model's prediction.

    Parameters
    ----------
    model               : Trained Keras 1D CNN model
    input_signal        : Single sample, shape (1, timesteps, 1)
    target_class        : Class to explain (default: predicted class)
    last_conv_layer_name: Name of the last Conv1D layer

    Returns
    -------
    heatmap : 1D array of same length as input, normalized [0, 1]
    """
    import tensorflow as tf

    # Auto-detect last conv layer
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No Conv1D layer found in model.")

    # Build gradient model
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(input_signal)
        if target_class is None:
            if predictions.shape[-1] == 1:
                # Binary: class is determined by > 0.5
                target_class = int(predictions.numpy()[0, 0] > 0.5)
                loss = predictions[0, 0] if target_class == 1 else (1 - predictions[0, 0])
            else:
                target_class = tf.argmax(predictions[0])
                loss = predictions[0, target_class]
        else:
            if predictions.shape[-1] == 1:
                loss = predictions[0, 0] if target_class == 1 else (1 - predictions[0, 0])
            else:
                loss = predictions[0, target_class]

    grads = tape.gradient(loss, conv_output)

    # Global average pooling over the spatial/temporal dimension
    weights = tf.reduce_mean(grads, axis=1)  # shape: (1, n_filters)

    # Weighted combination of feature maps
    cam = tf.reduce_sum(conv_output[0] * weights[0], axis=-1)  # shape: (temporal_dim,)
    cam = tf.nn.relu(cam)  # Keep only positive contributions
    cam = cam.numpy()

    # Resize to input length via interpolation
    input_length = input_signal.shape[1]
    if len(cam) != input_length:
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(cam))
        x_new = np.linspace(0, 1, input_length)
        cam = interp1d(x_old, cam, kind="linear")(x_new)

    # Normalize to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam


def plot_grad_cam(signal: np.ndarray, heatmap: np.ndarray,
                  prediction: str = "", sample_idx: int = 0):
    """
    Overlay Grad-CAM heatmap on the EEG signal.

    Parameters
    ----------
    signal   : Original EEG signal, 1D array (178,)
    heatmap  : Grad-CAM heatmap, 1D array (178,), values in [0, 1]
    prediction : Prediction label for title
    sample_idx : Sample index for filename
    """
    t = np.arange(len(signal)) / config.SAMPLING_RATE

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, signal, color="black", linewidth=0.8, alpha=0.8, label="EEG Signal")

    # Color the background based on heatmap intensity
    for i in range(len(t) - 1):
        ax.axvspan(t[i], t[i + 1], alpha=heatmap[i] * 0.6,
                   color="red", linewidth=0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Grad-CAM Explanation — Sample #{sample_idx} ({prediction})")
    ax.legend(loc="upper right")
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, f"gradcam_sample_{sample_idx}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[GRADCAM] Plot saved to {path}")

    return path
