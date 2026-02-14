# Psoriasis Severity Detection System

A machine learning–based web application that predicts **psoriasis severity** from skin images and provides basic recommendations.

This project uses **deep learning with transfer learning** to classify psoriasis severity levels and provides an interactive interface for users to upload images and receive predictions.

---

## Project Overview
Psoriasis is a chronic skin condition that varies in severity. Early assessment helps guide treatment decisions.

This system:
- Accepts a skin image input
- Predicts psoriasis severity
- Provides severity-based suggestions
- Runs via a simple web interface

---

## Features
- Image-based psoriasis severity prediction
- Deep learning model using transfer learning
- Simple and interactive web interface
- Severity-based recommendations
- Easy deployment and extension

---

## Model Details
- Base Model: Transfer Learning (MobileNetV2)
- Framework: TensorFlow / Keras
- Input: Skin lesion image
- Output: Severity category prediction

---

## Project Structure
skincare/
│
├── app.py # Main application
├── psoriasis_severity_model.h5 # Trained ML model
├── templates/ # HTML templates
├── static/ # CSS/JS files
├── uploads/ # Uploaded images
├── README.md
└── requirements.txt



---

## Installation

### 1. Clone repository
```bash
git clone https://github.com/<your-username>/psoriasis-detection-project.git
cd psoriasis-detection-project




2. Install dependencies
pip install -r requirements.txt

3. Run application
python app.py


Open browser:

http://127.0.0.1:5000

How It Works

User uploads a skin image.

Image is preprocessed.

Model predicts severity.

Result and recommendation are displayed.

Dataset and Training

Images labeled into severity classes

Data augmentation used

Train/test split applied

Transfer learning used for faster convergence

Limitations

Model accuracy depends on dataset quality

Not a replacement for medical diagnosis

Needs larger clinical dataset for production use

Future Improvements

Larger medical dataset integration

Mobile app deployment

Doctor recommendation system

Multi-disease skin detection

Real-time camera prediction

Requirements

Main libraries:

Python

TensorFlow / Keras

Flask or Streamlit

OpenCV

NumPy

Author

Shruti Kukreti