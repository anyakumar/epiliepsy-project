"""
1D Convolutional Neural Network for EEG Seizure Detection
---------------------------------------------------------
Architecture:
  Input (178, 1)
    → Conv1D(64, 5) → BatchNorm → ReLU → MaxPool(2)
    → Conv1D(128, 5) → BatchNorm → ReLU → MaxPool(2)
    → Conv1D(64, 5) → BatchNorm → ReLU → MaxPool(2)
    → GlobalAveragePooling1D
    → Dense(128) → Dropout(0.5)
    → Dense(num_classes, softmax)

Uses TensorFlow / Keras.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers

from . import config


def build_cnn_model(input_shape: tuple = None, num_classes: int = 2,
                    filters: list = None, kernel_size: int = None,
                    pool_size: int = None, dense_units: int = None,
                    dropout: float = None, learning_rate: float = None) -> keras.Model:
    """
    Build and compile a 1D CNN model for EEG classification.

    Parameters
    ----------
    input_shape   : Tuple (timesteps, channels), default (178, 1)
    num_classes   : Number of output classes (2 for binary)
    filters       : List of filter counts per conv block
    kernel_size   : Kernel size for Conv1D layers
    pool_size     : MaxPooling size
    dense_units   : Units in the Dense layer before output
    dropout       : Dropout rate
    learning_rate : Adam optimizer learning rate

    Returns
    -------
    model : keras.Model (compiled)
    """
    input_shape = input_shape or (config.NUM_FEATURES, 1)
    filters = filters or config.CNN_FILTERS
    kernel_size = kernel_size or config.CNN_KERNEL_SIZE
    pool_size = pool_size or config.CNN_POOL_SIZE
    dense_units = dense_units or config.CNN_DENSE_UNITS
    dropout = dropout or config.CNN_DROPOUT
    learning_rate = learning_rate or config.CNN_LEARNING_RATE

    model = keras.Sequential(name="EEG_1D_CNN")

    # Convolutional blocks
    for i, n_filters in enumerate(filters):
        if i == 0:
            model.add(layers.Conv1D(n_filters, kernel_size, padding="same",
                                    input_shape=input_shape, name=f"conv1d_{i+1}"))
        else:
            model.add(layers.Conv1D(n_filters, kernel_size, padding="same",
                                    name=f"conv1d_{i+1}"))
        model.add(layers.BatchNormalization(name=f"bn_{i+1}"))
        model.add(layers.Activation("relu", name=f"relu_{i+1}"))
        model.add(layers.MaxPooling1D(pool_size=pool_size, name=f"maxpool_{i+1}"))

    # Global pooling to collapse temporal dimension
    model.add(layers.GlobalAveragePooling1D(name="global_avg_pool"))

    # Classifier head
    model.add(layers.Dense(dense_units, activation="relu", name="dense_1"))
    model.add(layers.Dropout(dropout, name="dropout_1"))

    # Output layer
    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid", name="output"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        model.add(layers.Dense(num_classes, activation="softmax", name="output"))
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics
    )

    return model


def get_cnn_callbacks(model_name: str = "cnn_best") -> list:
    """
    Standard training callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
    """
    checkpoint_path = os.path.join(config.MODELS_DIR, f"{model_name}.keras")
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.CNN_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]


def train_cnn(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              num_classes: int = 2,
              epochs: int = None, batch_size: int = None) -> tuple:
    """
    Build, train, and return the CNN model + training history.

    Parameters
    ----------
    X_train, y_train : Training data (already shaped for Conv1D)
    X_val, y_val     : Validation data
    num_classes      : 2 (binary) or 5 (multi-class)
    epochs           : Max training epochs
    batch_size       : Batch size

    Returns
    -------
    model   : Trained keras.Model
    history : keras History object
    """
    epochs = epochs or config.CNN_EPOCHS
    batch_size = batch_size or config.CNN_BATCH_SIZE

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    cbs = get_cnn_callbacks("cnn_best")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1
    )

    return model, history


# ─── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()
    print("\n✅ CNN model built successfully.")
