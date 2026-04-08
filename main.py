import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input 


def load_models():
    model_path = os.path.join("..", "VGG19.h5") 

    print("Loading VGG19 model from H5 file...")
    try:
        VGG19Model = load_model(model_path, compile=False)
        print(" VGG19 Model loaded successfully!")
        return VGG19Model
    except Exception as e:
        print(f" Error: {e}")
        exit()

def browse_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Image for Testing",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    root.destroy()
    return file_path

def predict_vgg19(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x)
    
    preds = model.predict(x)
    return preds

VGG19Model= load_models()
selected_img = browse_image()

if selected_img:
    output = predict_vgg19(selected_img,VGG19Model)
    predicted_idx = np.argmax(output, axis=1)[0]
    confidence = np.max(output) * 100
    
    print("\n" + "="*40)
    print(f"File: {os.path.basename(selected_img)}")
    print(f"Predicted Class Index: {predicted_idx}")
    print(f"Confidence: {confidence:.2f}%")
    print("="*40)
else:
    print("No file selected.")