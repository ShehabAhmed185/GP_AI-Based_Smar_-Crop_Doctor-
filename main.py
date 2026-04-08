import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from collections import Counter

# =========================
# Class Mapping (38 Classes)
# =========================
class_indices = {
    'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 'Apple___healthy': 3,
    'Blueberry___healthy': 4, 'Cherry_(including_sour)___Powdery_mildew': 5, 'Cherry_(including_sour)___healthy': 6,
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 'Corn_(maize)___Common_rust_': 8,
    'Corn_(maize)___Northern_Leaf_Blight': 9, 'Corn_(maize)___healthy': 10, 'Grape___Black_rot': 11,
    'Grape___Esca_(Black_Measles)': 12, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 'Grape___healthy': 14,
    'Orange___Haunglongbing_(Citrus_greening)': 15, 'Peach___Bacterial_spot': 16, 'Peach___healthy': 17,
    'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19, 'Potato___Early_blight': 20,
    'Potato___Late_blight': 21, 'Potato___healthy': 22, 'Raspberry___healthy': 23, 'Soybean___healthy': 24,
    'Squash___Powdery_mildew': 25, 'Strawberry___Leaf_scorch': 26, 'Strawberry___healthy': 27,
    'Tomato___Bacterial_spot': 28, 'Tomato___Early_blight': 29, 'Tomato___Late_blight': 30,
    'Tomato___Leaf_Mold': 31, 'Tomato___Septoria_leaf_spot': 32,
    'Tomato___Spider_mites Two-spotted_spider_mite': 33, 'Tomato___Target_Spot': 34,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato___Tomato_mosaic_virus': 36, 'Tomato___healthy': 37
}

idx_to_class = {v: k for k, v in class_indices.items()}

# =========================
# Load Model Function
# =========================
def load_models_func(model_filename):
    model_path = os.path.join("..",model_filename) 

    print(f"Loading model from: {model_path}...")
    try:
        model = load_model(model_path, compile=False)
        print(f"Model {model_filename} loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading {model_filename}: {e}")
        return None

# =========================
# Prediction Function
# =========================
def get_prediction(img_path, model):
    if model is None: return None
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x, verbose=0)
    return preds[0]

# =========================
# Browse Image
# =========================
def browse_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Image")
    root.destroy()
    return file_path

# =========================
# Main Execution
# =========================

vgg19_m = load_models_func("VGG19.h5")
vgg16_m = load_models_func("VGG16.h5")
resnet_m = load_models_func("resnet101v2.h5")

selected_img = browse_image()

if selected_img and all([vgg19_m, vgg16_m, resnet_m]):
    p1 = get_prediction(selected_img, vgg19_m)
    p2 = get_prediction(selected_img, vgg16_m)
    p3 = get_prediction(selected_img, resnet_m)

    c1 = np.argmax(p1)
    c2 = np.argmax(p2)
    c3 = np.argmax(p3)

    results = [c1, c2, c3]
    confidences = [p1[c1], p2[c2], p3[c3]]
    
    occurence_count = Counter(results)
    most_common_class, count = occurence_count.most_common(1)[0]

    print("\n" + "="*60)
    print(f"File: {os.path.basename(selected_img)}")
    print("="*60)
    print(f"VGG19 Predicted    : {idx_to_class[c1]} ({p1[c1]*100:.2f}%)")
    print(f"VGG16 Predicted    : {idx_to_class[c2]} ({p2[c2]*100:.2f}%)")
    print(f"ResNet101 Predicted: {idx_to_class[c3]} ({p3[c3]*100:.2f}%)")
    print("-" * 60)

    if count >= 2:
        final_idx = most_common_class
        reason = f"Majority Vote ({count}/3 models agreed)"
    else:
        final_idx = results[np.argmax(confidences)]
        reason = "Highest Confidence (No majority agreement)"

    final_class_name = idx_to_class[final_idx]
    final_conf = confidences[results.index(final_idx)]

    print(f"FINAL DECISION: {final_class_name}")
    print(f"Confidence    : {final_conf*100:.2f}%")
    print(f"Based on      : {reason}")
    print("="*60)

else:
    print("Execution failed: Image not selected or models not loaded.")