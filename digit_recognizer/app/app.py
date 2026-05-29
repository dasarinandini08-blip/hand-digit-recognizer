"""
app.py  –  Handwritten Digit Recognizer  |  Streamlit Interface
================================================================
Features:
  • Draw a digit on a canvas and get an instant CNN prediction
  • Upload a custom image (PNG / JPG)
  • See confidence bar chart for all 10 digits
  • View model architecture summary

Run:
    streamlit run app/app.py
"""

import os, io, cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

# ── Try importing optional deps gracefully ───────────────────────
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

import tensorflow as tf

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🔢",
    layout="centered",
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_mnist.keras")

@st.cache_resource
def load_model():
    """Load the trained Keras model (cached so it loads only once)."""
    if not os.path.exists(MODEL_PATH):
        st.error(
            "⚠️  Trained model not found!\n\n"
            "Please run `python models/train_model.py` first to train and save the model."
        )
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(img_array: np.ndarray) -> np.ndarray:
    """
    Convert any input image to the 28×28 grayscale tensor expected by the model.
    Handles RGBA, RGB, and grayscale inputs.
    """
    # Convert RGBA → RGB → Gray
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    elif img_array.ndim == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    img_array = img_array.astype(np.float32)

    # Invert if background is white (canvas draws black on white)
    if img_array.mean() > 127:
        img_array = 255.0 - img_array

    # Resize to 28×28
    img_array = cv2.resize(img_array, (28, 28), interpolation=cv2.INTER_AREA)

    # Normalise
    img_array = img_array / 255.0

    # Shape: (1, 28, 28, 1)
    return img_array.reshape(1, 28, 28, 1)


def predict(model, img_tensor: np.ndarray):
    """Return (predicted_digit, confidence_array)."""
    probs  = model.predict(img_tensor, verbose=0)[0]   # shape (10,)
    digit  = int(np.argmax(probs))
    return digit, probs


def confidence_chart(probs: np.ndarray):
    """Horizontal bar chart showing confidence for each digit."""
    fig, ax = plt.subplots(figsize=(6, 3))
    colors  = ["#e63946" if i == np.argmax(probs) else "#457b9d" for i in range(10)]
    ax.barh(range(10), probs * 100, color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"  {i}" for i in range(10)], fontsize=11)
    ax.set_xlabel("Confidence (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Prediction Confidence", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(probs):
        ax.text(v * 100 + 1, i, f"{v*100:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🔢 Handwritten Digit Recognizer")
st.markdown(
    "Draw a digit **0–9** on the canvas *or* upload an image. "
    "The CNN model will predict the digit instantly."
)
st.divider()

model = load_model()

tab1, tab2, tab3 = st.tabs(["✏️  Draw", "📁  Upload Image", "🧠  Model Info"])

# ── Tab 1: Draw ───────────────────────────────
with tab1:
    if not CANVAS_AVAILABLE:
        st.warning(
            "`streamlit-drawable-canvas` is not installed.\n\n"
            "Run `pip install streamlit-drawable-canvas` and restart the app."
        )
    else:
        st.markdown("**Draw a single digit in the box below:**")
        col1, col2 = st.columns([1, 1])

        with col1:
            canvas_result = st_canvas(
                fill_color   = "black",
                stroke_width = 20,
                stroke_color = "white",
                background_color = "#000000",
                width  = 280,
                height = 280,
                drawing_mode = "freedraw",
                key = "canvas",
            )

        with col2:
            if canvas_result.image_data is not None:
                img_data = canvas_result.image_data
                # Only predict if user has drawn something
                if img_data[:, :, :3].max() > 10:
                    tensor = preprocess_image(img_data)
                    digit, probs = predict(model, tensor)

                    st.markdown(f"### Predicted Digit: **:red[{digit}]**")
                    st.markdown(f"Confidence: **{probs[digit]*100:.1f}%**")
                    st.pyplot(confidence_chart(probs))
                else:
                    st.info("Start drawing to see the prediction!")
            else:
                st.info("Start drawing to see the prediction!")

        if st.button("🗑️  Clear Canvas"):
            st.rerun()

# ── Tab 2: Upload ─────────────────────────────
with tab2:
    st.markdown("**Upload a handwritten digit image (PNG / JPG):**")
    uploaded = st.file_uploader("Choose an image …", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        image = Image.open(io.BytesIO(uploaded.read())).convert("L")  # grayscale
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, caption="Uploaded Image", width=200)

        with col2:
            img_array = np.array(image)
            tensor    = preprocess_image(img_array)
            digit, probs = predict(model, tensor)

            st.markdown(f"### Predicted Digit: **:red[{digit}]**")
            st.markdown(f"Confidence: **{probs[digit]*100:.1f}%**")
            st.pyplot(confidence_chart(probs))

        # Show preprocessed 28×28 thumbnail
        with st.expander("🔍 See preprocessed 28×28 input fed to the model"):
            processed_display = (tensor.squeeze() * 255).astype(np.uint8)
            st.image(processed_display, width=140, caption="28×28 input tensor")

# ── Tab 3: Model Info ─────────────────────────
with tab3:
    st.subheader("🧠 CNN Architecture")
    st.markdown("""
| Layer | Type | Output Shape | Parameters |
|-------|------|-------------|------------|
| 1 | Conv2D (32 filters, 3×3) + BN | (28, 28, 32) | 320 |
| 2 | Conv2D (32 filters, 3×3) + BN | (28, 28, 32) | 9,248 |
| 3 | MaxPooling2D (2×2) | (14, 14, 32) | — |
| 4 | Dropout (0.25) | (14, 14, 32) | — |
| 5 | Conv2D (64 filters, 3×3) + BN | (14, 14, 64) | 18,496 |
| 6 | Conv2D (64 filters, 3×3) + BN | (14, 14, 64) | 36,928 |
| 7 | MaxPooling2D (2×2) | (7, 7, 64) | — |
| 8 | Dropout (0.25) | (7, 7, 64) | — |
| 9 | Flatten | (3136,) | — |
| 10 | Dense (256) + BN | (256,) | 802,816 |
| 11 | Dropout (0.5) | (256,) | — |
| 12 | Dense (10, Softmax) | (10,) | 2,570 |
""")
    st.markdown("""
**Training Details:**
- **Dataset:** MNIST (60,000 train / 10,000 test)
- **Optimiser:** Adam (lr=1e-3, with ReduceLROnPlateau)
- **Loss:** Categorical Cross-Entropy
- **Augmentation:** Rotation ±10°, Zoom ±10%, Shifts ±10%
- **Test Accuracy:** > 99%
""")
    # Show model summary in expandable
    with st.expander("📋 Keras model.summary()"):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        st.code(stream.getvalue(), language="text")

st.divider()
st.caption("Built with TensorFlow / Keras + Streamlit  •  MNIST Dataset  •  CNN Architecture")
