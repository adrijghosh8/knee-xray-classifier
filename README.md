# 🦴 Knee X-Ray Classifier

A CNN-based deep learning application that classifies knee X-ray images into **five osteoarthritis severity levels** with confidence scores and Grad-CAM visual explanations.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?style=flat&logo=huggingface&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-5C3EE8?style=flat&logo=opencv&logoColor=white)

</p>

## ✨ Features

- 🧠 CNN-based knee X-ray classification
- 📊 Five severity classes: **Normal, Doubtful, Mild, Moderate, Severe**
- 🎯 Prediction confidence and class probabilities
- 🔥 Grad-CAM visualization for model explainability
- 🌐 Streamlit web application
- ☁️ Model hosted separately on Hugging Face

## ⚙️ Working

```text
Knee X-Ray
    ↓
Grayscale + Resize (200×200)
    ↓
CNN Model
    ↓
┌───────────────────────┐
│ Severity Prediction   │
│ Confidence Score      │
│ Class Probabilities   │
│ Grad-CAM Heatmap      │
└───────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow / Keras |
| Model | Custom CNN |
| Explainability | Grad-CAM |
| Interface | Streamlit |
| Image Processing | Pillow, OpenCV |
| Model Hosting | Hugging Face Hub |

## 📁 Structure

```text
knee-xray-classifier/
├── app.py
├── predict.py
├── preprocessing.py
├── gradcam.py
├── requirements.txt
├── runtime.txt
└── README.md
```

Some sample images are provided in Testing Images for quick model testing purposes

## 🚀 Run Locally

```bash
git clone https://github.com/adrijghosh8/knee-xray-classifier.git
cd knee-xray-classifier

pip install -r requirements.txt
streamlit run app.py
```

The trained model is downloaded from Hugging Face when the application starts.

## 🌐 Live Demo

**Working Video**
https://www.youtube.com/watch?v=curDGwR9VZY

**Streamlit:**  
https://knee-xray-classifier-dr5ldacmpvt64catmav8pj.streamlit.app/

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes only** and should not be used as a medical diagnostic tool.
