"""
explore.py  –  Digit Recognizer: Data Exploration & Visualisation
=================================================================
Run cell-by-cell in VS Code (# %% cells) or as a plain script.

Usage:
    python notebooks/explore.py
"""

# %% ── Imports ────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import mnist

print("TensorFlow version:", tf.__version__)

# %% ── 1. Load Data ───────────────────────────────────────────────
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Train set  :  {x_train.shape}  labels {y_train.shape}")
print(f"Test  set  :  {x_test.shape}   labels {y_test.shape}")
print(f"Pixel range:  [{x_train.min()}, {x_train.max()}]")

# %% ── 2. Class Distribution ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle("Class Distribution in MNIST", fontsize=13, fontweight="bold")

for ax, data, split in zip(axes, [y_train, y_test], ["Train", "Test"]):
    counts = np.bincount(data)
    bars = ax.bar(range(10), counts, color=plt.cm.tab10.colors, edgecolor="white", linewidth=0.8)
    ax.set_title(f"{split} Set ({len(data):,} samples)")
    ax.set_xlabel("Digit"); ax.set_ylabel("Count")
    ax.set_xticks(range(10))
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                str(count), ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("dataset/class_distribution.png", dpi=150)
plt.show()

# %% ── 3. Sample Images Grid ──────────────────────────────────────
fig = plt.figure(figsize=(15, 6))
fig.suptitle("Sample MNIST Images (5 per digit)", fontsize=14, fontweight="bold")
gs  = gridspec.GridSpec(10, 5, figure=fig, hspace=0.05, wspace=0.05)

for digit in range(10):
    indices = np.where(y_train == digit)[0][:5]
    for col, idx in enumerate(indices):
        ax = fig.add_subplot(gs[digit, col])
        ax.imshow(x_train[idx], cmap="gray", aspect="equal")
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(str(digit), rotation=0, labelpad=15, fontsize=11, va="center")

plt.savefig("dataset/sample_images.png", dpi=150, bbox_inches="tight")
plt.show()

# %% ── 4. Pixel Intensity Distribution ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(x_train.flatten(), bins=50, color="#457b9d", edgecolor="white", linewidth=0.5)
ax.set_title("Pixel Intensity Distribution (Train Set)", fontsize=13, fontweight="bold")
ax.set_xlabel("Pixel Value (0-255)"); ax.set_ylabel("Frequency")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("dataset/pixel_distribution.png", dpi=150)
plt.show()

# %% ── 5. Mean Image per Digit ────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Mean Pixel Image per Digit", fontsize=13, fontweight="bold")

for digit, ax in enumerate(axes.flat):
    mean_img = x_train[y_train == digit].mean(axis=0)
    im = ax.imshow(mean_img, cmap="hot")
    ax.set_title(f"Digit {digit}", fontsize=10)
    ax.axis("off")

plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Mean Pixel Value")
plt.savefig("dataset/mean_digit_images.png", dpi=150)
plt.show()

# %% ── 6. Single-Image Preprocessing Demo ────────────────────────
sample = x_train[0]
print(f"Original shape : {sample.shape}   dtype: {sample.dtype}")

normalised  = sample.astype("float32") / 255.0
reshaped    = normalised.reshape(1, 28, 28, 1)
print(f"After normalise: min={normalised.min():.2f}  max={normalised.max():.2f}")
print(f"Model input    : {reshaped.shape}")

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
fig.suptitle("Preprocessing Pipeline", fontsize=12, fontweight="bold")

axes[0].imshow(sample, cmap="gray"); axes[0].set_title("1. Raw (uint8)"); axes[0].axis("off")
axes[1].imshow(normalised, cmap="gray"); axes[1].set_title("2. Normalised (float32)"); axes[1].axis("off")
axes[2].imshow(reshaped.squeeze(), cmap="hot"); axes[2].set_title("3. Model Input"); axes[2].axis("off")

plt.tight_layout()
plt.savefig("dataset/preprocessing_demo.png", dpi=150)
plt.show()

print("\n✅  All exploration plots saved to dataset/")
