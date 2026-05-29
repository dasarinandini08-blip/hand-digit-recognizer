# ✏️ Handwritten Digit Recognizer

A CNN trained on MNIST to recognize handwritten digits (0–9) with **99%+ accuracy**, served through a Streamlit web app.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python train.py
python -m streamlit run app/app.py
```

---

## 📁 Project Structure

```
digit_recognizer/
├── app/
│   └── app.py          ← Streamlit web app
├── models/
│   └── cnn_model.py    ← CNN architecture
├── screenshots/        ← App screenshots
├── train.py            ← Training pipeline
└── requirements.txt
```

---

## 🌐 App Features

### ✏️ Draw a Digit

Draw any digit on the canvas using your mouse. The model predicts it instantly with a confidence score.

![Draw Tab](screenshots/app_draw.png)

---

### 🎨 Drawing Canvas

Draw on the black canvas — the model reads your stroke and classifies it as one of digits 0–9.

![Canvas](screenshots/canvas_draw.jpg)

---

### 🎯 Prediction Result

Shows the predicted digit, confidence percentage, and a bar chart across all 10 classes.

![Prediction](screenshots/prediction_result.jpg)

---

### 📁 Upload an Image

Upload any PNG or JPG of a handwritten digit. The model preprocesses it and returns a prediction.

![Upload Tab](screenshots/app_upload.png)

---

### 🧠 Model Architecture

View the full CNN layer table — type, output shape, and parameter count for every layer.

![Model Info](screenshots/app_model_info.png)

---

### 📋 Training Details

Shows the complete 12-layer architecture and training configuration.

![Model Details](screenshots/app_model_details.png)

| Detail | Value |
|---|---|
| Dataset | MNIST — 60K train / 10K test |
| Optimizer | Adam + ReduceLROnPlateau |
| Loss | Categorical Cross-Entropy |
| Test Accuracy | > 99% |

---

## 🧠 CNN Architecture

```
Input (28×28×1)
  → Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)
  → Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)
  → Conv2D(128) → BatchNorm → Dropout(0.25)
  → Flatten → Dense(256) → BatchNorm → Dropout(0.50)
  → Dense(10, Softmax)
```

Total Parameters: ~870,000

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
