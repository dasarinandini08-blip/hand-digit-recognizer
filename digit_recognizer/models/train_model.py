"""
train_model.py
==============
Trains a Convolutional Neural Network (CNN) on the MNIST dataset.
Saves the trained model and generates training visualizations.

Run:
    python models/train_model.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────
IMG_SIZE    = 28          # MNIST images are 28×28 pixels
NUM_CLASSES = 10          # Digits 0-9
BATCH_SIZE  = 128
EPOCHS      = 20
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "cnn_mnist.keras")
PLOTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "dataset")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 3. LOAD & PREPROCESS DATA
# ─────────────────────────────────────────────
def load_and_preprocess():
    """
    Loads MNIST, normalises pixel values to [0, 1], and reshapes
    images to (28, 28, 1) for the CNN.
    """
    print("📥  Loading MNIST dataset …")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalise: uint8 [0,255] → float32 [0,1]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0

    # Add channel dimension: (N,28,28) → (N,28,28,1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test  = np.expand_dims(x_test,  axis=-1)

    # One-hot encode labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  NUM_CLASSES)

    print(f"   Train: {x_train.shape}  |  Test: {x_test.shape}")
    return x_train, y_train, y_train_oh, x_test, y_test, y_test_oh


# ─────────────────────────────────────────────
# 4. BUILD CNN MODEL
# ─────────────────────────────────────────────
def build_model():
    """
    Architecture:
      Conv(32) → Conv(32) → MaxPool → Dropout
      Conv(64) → Conv(64) → MaxPool → Dropout
      Flatten → Dense(256) → Dropout → Dense(10, softmax)
    """
    model = models.Sequential([
        # ── Block 1 ──────────────────────────
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # ── Block 2 ──────────────────────────
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        # ── Classifier ───────────────────────
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="CNN_MNIST")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ─────────────────────────────────────────────
# 5. DATA AUGMENTATION
# ─────────────────────────────────────────────
def get_augmentor():
    """Light augmentation to improve generalisation."""
    return ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
    )


# ─────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────
def train(model, x_train, y_train_oh, x_test, y_test_oh):
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    datagen = get_augmentor()
    datagen.fit(x_train)

    print("\n🚀  Training …")
    history = model.fit(
        datagen.flow(x_train, y_train_oh, batch_size=BATCH_SIZE),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test_oh),
        callbacks=callbacks,
    )
    return history


# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, x_test, y_test, y_test_oh):
    loss, acc = model.evaluate(x_test, y_test_oh, verbose=0)
    print(f"\n✅  Test Accuracy : {acc * 100:.2f}%")
    print(f"   Test Loss     : {loss:.4f}")

    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    print("\n📊  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
    return y_pred


# ─────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────
def plot_training_history(history):
    """Accuracy and Loss curves saved to dataset/."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN Training History – MNIST", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train Accuracy",      linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy", linewidth=2, linestyle="--")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train Loss",      linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validation Loss", linewidth=2, linestyle="--")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    print(f"   Saved → {path}")
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.title("Confusion Matrix – MNIST CNN", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label"); plt.ylabel("True Label")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    print(f"   Saved → {path}")
    plt.show()


def plot_sample_predictions(x_test, y_test, y_pred, n=20):
    """Show a grid of sample predictions (green = correct, red = wrong)."""
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle("Sample Predictions (Green=Correct / Red=Wrong)", fontsize=13, fontweight="bold")

    indices = np.random.choice(len(x_test), n, replace=False)
    for ax, idx in zip(axes.flat, indices):
        ax.imshow(x_test[idx].squeeze(), cmap="gray")
        color = "green" if y_pred[idx] == y_test[idx] else "red"
        ax.set_title(f"P:{y_pred[idx]}  T:{y_test[idx]}", color=color, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "sample_predictions.png")
    plt.savefig(path, dpi=150)
    print(f"   Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    x_train, y_train, y_train_oh, x_test, y_test, y_test_oh = load_and_preprocess()

    model   = build_model()
    history = train(model, x_train, y_train_oh, x_test, y_test_oh)
    y_pred  = evaluate(model, x_test, y_test, y_test_oh)

    print("\n📈  Generating plots …")
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    plot_sample_predictions(x_test, y_test, y_pred)

    print(f"\n💾  Model saved → {MODEL_PATH}")
