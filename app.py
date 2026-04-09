import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from collections import Counter
import pandas as pd

# ==========================================
# 1. إعدادات الثوابت والخرائط (Configurations)
# ==========================================
CLASS_INDICES = {
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
IDX_TO_CLASS = {v: k for k, v in CLASS_INDICES.items()}

# ==========================================
# 2. محرك تحميل النماذج (Model Loader Engine)
# ==========================================
class AgriculturalAI:
    def __init__(self, models_dir=".."):
        self.models_dir = models_dir
        self.models = {}
        self.load_all_models()

    def load_all_models(self):
        print("--- Loading All Models ---")
        # تحميل نماذج الرؤية الحاسوبية (Deep Learning)
        dl_models = ["VGG19.h5", "VGG16.h5", "resnet101v2.h5", "InceptionV3.h5"]
        for m_name in dl_models:
            path = os.path.join(self.models_dir, m_name)
            try:
                self.models[m_name] = load_model(path, compile=False)
                print(f"Successfully loaded {m_name}")
            except Exception as e:
                print(f"Failed to load {m_name}: {e}")

        # تحميل نماذج البيانات الجدولية (Machine Learning)
        try:
            self.models['crop_model'] = joblib.load(os.path.join(self.models_dir, "crop_model.pkl"))
            self.models['le_crop'] = joblib.load(os.path.join(self.models_dir, "crop_label_encoder.pkl"))
            self.models['fert_model'] = joblib.load(os.path.join(self.models_dir, "fertilizer_model_Xg.pkl"))
            self.models['le_fert'] = joblib.load(os.path.join(self.models_dir, "fertilizer_label_encoder.pkl"))
            print("ML Models (Crop/Fertilizer) loaded successfully.")
        except Exception as e:
            print(f"ML Models loading error: {e}")

    # ==========================================
    # 3. وظائف المعالجة (Processing Logic)
    # ==========================================
    
    def predict_disease(self, img_path):
        """التعرف على مرض النبات باستخدام 4 موديلات (Ensemble)"""
        results = []
        confidences = []
        
        # أحجام المدخلات لكل موديل
        configs = {
            "VGG19.h5": 224, "VGG16.h5": 224, "resnet101v2.h5": 224, "InceptionV3_Fixed.h5": 299
        }

        for m_name, size in configs.items():
            if m_name in self.models:
                img = image.load_img(img_path, target_size=(size, size))
                x = image.img_to_array(img) / 255.0
                x = np.expand_dims(x, axis=0)
                preds = self.models[m_name].predict(x, verbose=0)[0]
                idx = np.argmax(preds)
                results.append(idx)
                confidences.append(preds[idx])

        # التصويت بالأغلبية
        occurence = Counter(results)
        most_common, count = occurence.most_common(1)[0]
        
        final_idx = most_common if count >= 2 else results[np.argmax(confidences)]
        return {
            "disease": IDX_TO_CLASS[final_idx],
            "confidence": float(confidences[results.index(final_idx)]),
            "agreement": f"{count}/4"
        }

    def recommend_crop_and_fert(self, env_data):
        """توصية المحصول والسماد بناءً على بيانات البيئة"""
        df = pd.DataFrame([env_data])
        
        # 1. توقع المحصول
        crop_pred = self.models['crop_model'].predict(df)
        crop_name = self.models['le_crop'].inverse_transform(crop_pred)[0]
        
        # 2. توقع السماد (إضافة المحصول المتوقع للبيانات)
        df['Crop'] = crop_name
        fert_pred = self.models['fert_model'].predict(df)
        fert_name = self.models['le_fert'].inverse_transform(fert_pred)[0]
        
        return {
            "recommended_crop": crop_name,
            "recommended_fertilizer": fert_name
        }

# ==========================================
# 4. نقطة الدخول الرئيسية (Application Entry)
# ==========================================
if __name__ == "__main__":
    # تشغيل النظام
    ai_system = AgriculturalAI(models_dir="..") # تأكد من أن النماذج في المجلد الأب

    print("\n" + "="*40)
    print("--- STEP 1: Disease Diagnosis ---")
    image_path = os.path.join("..", "data", "val_imgs", "PlantVillage", "val","Apple___Cedar_apple_rust","4e6676b6-154c-4f7d-a355-bcc00a397c3d___FREC_C.Rust 9853.jpg")   
    if os.path.exists(image_path):
        disease_result = ai_system.predict_disease(image_path)
        print(f"Result: {disease_result}")
    else:
        print("Image file not found, skipping diagnosis.")

    print("\n--- STEP 2: Crop & Fertilizer Recommendation ---")
    # محاكاة مدخلات المستخدم
    user_env_input = {
        "Nitrogen": 90,
        "Phosphorus": 42,
        "Potassium": 43,
        "pH": 6.5,
        "Rainfall": 200,
        "Temperature": 25,
        "Soil_color": "Black"
    }
    
    reco_result = ai_system.recommend_crop_and_fert(user_env_input)
    print(f"Recommendation: {reco_result}")
    print("="*40)