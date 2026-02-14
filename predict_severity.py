import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2
import os

# Configuration
MODEL_PATH = r"c:\Users\HP\Desktop\skincare\psoriasis_severity_model.h5"
CLASSES = ['mild', 'moderate', 'severe'] # Must match generator class_indices alphabetically usually

def predict_image(image_path, model):
    try:
        # Load and preprocess
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0 # Normalize
        
        # Predict
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) # Softmax again? model output is softmax already usually.
        # If output is softmax, predictions[0] sums to 1.
        
        class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return CLASSES[class_idx], confidence, predictions[0]
    except Exception as e:
        print(f"Error: {e}")
        return None, 0, []

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please run training first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    # Interactive loop or hardcoded test
    print("Model Loaded. Ready to predict.")
    print("Enter image path (or 'q' to quit):")
    
    while True:
        path = input("Path: ").strip()
        if path.lower() == 'q':
            break
            
        path = path.strip('"').strip("'") # Clean quotes
        
        if not os.path.exists(path):
            print("File does not exist.")
            continue
            
        label, conf, probs = predict_image(path, model)
        
        if label:
            print("-" * 30)
            print(f"Prediction: {label.upper()}")
            print(f"Confidence: {conf:.2%}")
            print("Probabilities:")
            for i, c in enumerate(CLASSES):
                print(f"  {c}: {probs[i]:.4f}")
            print("-" * 30)

if __name__ == "__main__":
    main()
