import streamlit as st
import pandas as pd
import joblib

crop_model = joblib.load("crop_model.pkl")
le_crop = joblib.load("crop_label_encoder.pkl")

st.title("🌱 Crop Recommendation")

# Inputs
Nitrogen = st.number_input("Nitrogen", 0.0, 200.0)
Phosphorus = st.number_input("Phosphorus", 0.0, 200.0)
Potassium = st.number_input("Potassium", 0.0, 200.0)
pH = st.number_input("pH", 0.0, 14.0)
Rainfall = st.number_input("Rainfall", 0.0, 150000.0)
Temperature = st.number_input("Temperature", 0.0, 60.0)
Soil_color = st.selectbox("Soil Color", ["Red", "Black", "Brown","Dark Brown","Reddish Brown"])  # أو data["Soil_color"].unique()

if st.button("Recommend Crop"):
    df = pd.DataFrame([{
        "Nitrogen": Nitrogen,
        "Phosphorus": Phosphorus,
        "Potassium": Potassium,
        "pH": pH,
        "Rainfall": Rainfall,
        "Temperature": Temperature,
        "Soil_color": Soil_color
    }])
    
    pred = crop_model.predict(df)
    crop = le_crop.inverse_transform(pred)[0]
    st.success(f"✅ Recommended Crop: {crop}")

fert_model = joblib.load("fertilizer_model_Xg.pkl")
le_fert = joblib.load("fertilizer_label_encoder.pkl")

st.title("🌱 Fertilizer Recommendation")
Crop = st.selectbox("Crop", ["Wheat", "Rice", "Maize"])

if st.button("Recommend Fertilizer"):
    df = pd.DataFrame([{
        "Nitrogen": Nitrogen,
        "Phosphorus": Phosphorus,
        "Potassium": Potassium,
        "pH": pH,
        "Rainfall": Rainfall,
        "Temperature": Temperature,
        "Soil_color": Soil_color,
        "Crop": Crop
    }])
    
    pred = fert_model.predict(df)
    fertilizer = le_fert.inverse_transform(pred)[0]
    
    st.success(f"✅ Recommended Fertilizer: {fertilizer}")
