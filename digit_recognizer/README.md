# ✏️ Handwritten Digit Recognizer

A CNN trained on MNIST to recognize handwritten digits (0–9) with **99%+ accuracy**, served through a Streamlit web app.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python digit_recognizer/models/train_model.py
python -m streamlit run digit_recognizer/app/app.py
```

---

## 📁 Project Structure

```
digit_recognizer/
├── app/
│   ├── app.py              # Streamlit web app
│   └── predict.py          # Prediction logic
├── models/
│   ├── cnn_mnist.keras     # Trained model
│   └── train_model.py      # Training script
├── notebooks/
│   └── explore.py          # Data exploration
├── screenshots/            # App screenshots
└── requirements.txt
```

---

## 🖥️ App Features

### ✏️ Draw a Digit
Draw any digit on the canvas using your mouse. The model predicts it instantly with a confidence score.

![Draw a Digit](digit_recognizer/screenshots/draw.png)

---

### 🎯 Prediction Result
Shows the predicted digit, confidence percentage, and a bar chart across all 10 classes.

![Prediction Result](digit_recognizer/screenshots/prediction.png)

---

### 📁 Upload an Image
Upload any PNG or JPG of a handwritten digit. The model preprocesses it and returns a prediction.

![Upload an Image](digit_recognizer/screenshots/upload.png)

---

### 🧠 Model Info
View the full CNN layer table — type, output shape, and parameter count for every layer.

![Model Info](digit_recognizer/screenshots/model_info.png)

---

## 📊 Training Details

| Detail | Value |
|---|---|
| Dataset | MNIST — 60K train / 10K test |
| Optimizer | Adam + ReduceLROnPlateau |
| Loss | Categorical Cross-Entropy |
| Augmentation | Rotation ±10°, Zoom ±10%, Shifts ±10% |
| Test Accuracy | > 99% |

---

## 🧠 CNN Architecture

```
Input (28×28×1)
  → Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
  → Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
  → Flatten → Dense(256) → BatchNorm → Dropout(0.50)
  → Dense(10, Softmax)
```

**Total Parameters:** ~870,000

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| TensorFlow / Keras | Model building & training |
| NumPy | Array operations |
| OpenCV | Image preprocessing |
| Matplotlib | Visualisations |
| Streamlit | Web app |

---

## 📄 License

MIT — free for personal and academic use.
