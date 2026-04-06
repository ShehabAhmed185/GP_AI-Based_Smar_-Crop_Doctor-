import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# 1. Define the path to your saved model
# Adjust the path based on your exact folder structure
model_path = os.path.join("..","best_resnet101v2.keras")

# 2. Load the model
print("Loading model...")
model = load_model(model_path)

# 3. Setup File Browser
def browse_image():
    root = tk.Tk()
    root.withdraw() # Hide the main tkinter window
    file_path = filedialog.askopenfilename(
        title="Select Image for Prediction",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    root.destroy()
    return file_path

# 4. Image Preprocessing & Prediction
def predict_image(img_path):
    # Load and resize image to (224, 224) as defined in your training script
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Convert to array and add batch dimension
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply ResNetV2 specific preprocessing
    img_array = preprocess_input(img_array)
    
    # Run prediction
    predictions = model.predict(img_array)
    return predictions

# --- Main Execution ---
selected_img = browse_image()

if selected_img:
    print(f"Selected: {selected_img}")
    
    # Save output of model in a variable
    output_probs = predict_image(selected_img)
    
    # Get the class index with highest probability
    predicted_class_index = np.argmax(output_probs, axis=1)[0]
    
    # Print the raw output and the predicted class index
    print("-" * 30)
    print("Model Output (Probabilities):", output_probs)
    print("Predicted Class Index:", predicted_class_index)
    print("-" * 30)
else:
    print("No image selected.")