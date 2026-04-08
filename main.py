import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

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
# Load Model
# =========================
def load_models(modelPath):
    model_path = os.path.join("..", modelPath) 

    print(f"Loading model from: {model_path}...")
    try:
        model = load_model(model_path, compile=False)
        print("Model loaded successfully!\n")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

# =========================
# Browse Image
# =========================
def browse_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    root.destroy()
    return file_path

# =========================
# Prediction
# =========================
def predict_vgg19(img_path, model):
    # 1. Load image with target size matching training (224, 224)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)

    x = x / 255.0

    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    return preds

# =========================
# Main
# =========================

# call models
VGG19_MODEL = "VGG19.h5"
VGG16_MODEL = "VGG16.h5"
RESNET101V2_MODEL = "resnet101v2.h5"
VGG19model = load_models(VGG19_MODEL)
VGG16model = load_models(VGG16_MODEL)
resnet101v2Model = load_models(RESNET101V2_MODEL)

selected_img = browse_image()

if selected_img:
    predictions = predict_vgg19(selected_img, resnet101v2Model)[0]

    top3_indices = np.argsort(predictions)[-3:][::-1]

    print("\n" + "="*60)
    print(f"File: {os.path.basename(selected_img)}")
    print("="*60)

    for i, idx in enumerate(top3_indices):
        class_name = idx_to_class.get(idx, "Unknown")
        confidence = predictions[idx] * 100

        if "___" in class_name:
            plant, disease = class_name.split("___")
        else:
            plant, disease = class_name, ""

        print(f"Top {i+1}:")
        print(f"  Plant      : {plant}")
        print(f"  Disease    : {disease.replace('_', ' ')}")
        print(f"  Confidence : {confidence:.2f}%")
        print("-"*40)

    best_idx = top3_indices[0]
    best_conf = predictions[best_idx]

    print("="*60)
    if best_conf < 0.5:
        print("Warning: Model is not confident. Please use a clearer image.")
    else:
        print(f"Final Prediction: {idx_to_class[best_idx]} ({best_conf*100:.2f}%)")
    print("="*60)

else:
    print("No file selected.")