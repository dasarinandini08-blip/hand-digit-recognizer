"""
predict.py  –  Predict a single custom image from the command line
==================================================================
Usage:
    python app/predict.py path/to/digit_image.png
"""

import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_mnist.keras")


def preprocess(img_path: str) -> np.ndarray:
    """Load image → grayscale → 28×28 → normalise → add batch+channel dims."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    # Invert if white background
    if img.mean() > 127:
        img = 255 - img

    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return img.reshape(1, 28, 28, 1)


def predict(img_path: str):
    if not os.path.exists(MODEL_PATH):
        print("❌  Model not found. Run: python models/train_model.py")
        sys.exit(1)

    model  = tf.keras.models.load_model(MODEL_PATH)
    tensor = preprocess(img_path)
    probs  = model.predict(tensor, verbose=0)[0]
    digit  = int(np.argmax(probs))

    print(f"\n🔢  Predicted Digit : {digit}")
    print(f"   Confidence      : {probs[digit]*100:.2f}%\n")
    print("   Full probabilities:")
    for i, p in enumerate(probs):
        bar = "█" * int(p * 40)
        print(f"   {i}: {bar:<40} {p*100:5.1f}%")

    # Visual
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(tensor.squeeze(), cmap="gray")
    axes[0].set_title(f"Input Image → Predicted: {digit}", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    colors = ["#e63946" if i == digit else "#457b9d" for i in range(10)]
    axes[1].barh(range(10), probs * 100, color=colors)
    axes[1].set_yticks(range(10))
    axes[1].set_xlabel("Confidence (%)")
    axes[1].set_title("Class Probabilities")
    axes[1].grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app/predict.py <image_path>")
        sys.exit(1)
    predict(sys.argv[1])
