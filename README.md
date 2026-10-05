# Knee X-Ray Classifier

A deep learning web application that classifies knee X-ray images into five osteoarthritis severity levels using a Convolutional Neural Network (CNN).

## Classes

- Normal
- Doubtful
- Mild
- Moderate
- Severe

## Tech Stack

- Python
- TensorFlow / Keras
- CNN
- Streamlit
- NumPy
- Pillow
- Hugging Face Hub

## How It Works

1. Upload a knee X-ray image.
2. The image is converted to grayscale and resized to `200 × 200`.
3. The trained CNN processes the image.
4. The application predicts the severity class and confidence score.
5. Prediction probabilities for all five classes are displayed.

## Project Structure

```text
knee-xray-classifier/
├── app.py
├── predict.py
├── preprocessing.py
├── requirements.txt
├── runtime.txt
└── README.md
```

## Run Locally

```bash
git clone https://github.com/adrijghosh8/knee-xray-classifier.git
cd knee-xray-classifier

pip install -r requirements.txt
streamlit run app.py
```

The trained model is hosted separately on Hugging Face and downloaded by the application when required.

## Deployment

The application is deployed using **Streamlit Community Cloud**, while the trained model is hosted on **Hugging Face Hub**.

## Disclaimer

This project is intended for educational and demonstration purposes only. It is not a medical diagnostic tool.
