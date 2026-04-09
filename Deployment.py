import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# 1. إعداد المسارات (Path Setup)
# =========================
# الحصول على المسار الحالي للمجلد الذي يحتوي على Deployment.py
base_path = os.path.dirname(os.path.abspath(__file__))

# تحديد المسارات (نبحث في المجلد الحالي أولاً، ثم المجلد الأب كخيار احتياطي)
def get_model_path(filename):
    local_path = os.path.join(base_path, filename)
    parent_path = os.path.join(base_path, "..", filename)
    return local_path if os.path.exists(local_path) else parent_path

# =========================
# 2. تحميل النماذج (Loading Models)
# =========================
@st.cache_resource # استخدام التخزين المؤقت لتسريع التطبيق
def load_all_assets():
    try:
        crop_m = joblib.load(get_model_path("crop_model.pkl"))
        le_c = joblib.load(get_model_path("crop_label_encoder.pkl"))
        fert_m = joblib.load(get_model_path("fertilizer_model_Xg.pkl"))
        le_f = joblib.load(get_model_path("fertilizer_label_encoder.pkl"))
        return crop_m, le_c, fert_m, le_f
    except Exception as e:
        st.error(f"خطأ في تحميل الملفات: {e}")
        return None, None, None, None

crop_model, le_crop, fert_model, le_fert = load_all_assets()

# =========================
# 3. واجهة المستخدم (UI)
# =========================
st.set_page_config(page_title="GP Agricultural System", layout="wide")
st.title("🌱 Intelligent Agricultural Recommendation System")

# تقسيم الشاشة إلى مدخلات ومخرجات
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Environmental Data")
    Nitrogen = st.number_input("Nitrogen (N)", 0.0, 200.0, value=50.0)
    Phosphorus = st.number_input("Phosphorus (P)", 0.0, 200.0, value=50.0)
    Potassium = st.number_input("Potassium (K)", 0.0, 200.0, value=50.0)
    pH = st.number_input("Soil pH Level", 0.0, 14.0, value=6.5)
    Rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, value=100.0)
    Temperature = st.number_input("Temperature (°C)", 0.0, 60.0, value=25.0)
    Soil_color = st.selectbox("Soil Color", ["Red", "Black", "Brown", "Dark Brown", "Reddish Brown"])

with col2:
    st.header("Results & Recommendations")
    
    # الجزء الأول: توصية المحصول
    if st.button("Step 1: Recommend Crop"):
        if crop_model:
            df_crop = pd.DataFrame([{
                "Nitrogen": Nitrogen, "Phosphorus": Phosphorus, "Potassium": Potassium,
                "pH": pH, "Rainfall": Rainfall, "Temperature": Temperature, "Soil_color": Soil_color
            }])
            
            pred = crop_model.predict(df_crop)
            recommended_crop = le_crop.inverse_transform(pred)[0]
            st.session_state['recommended_crop'] = recommended_crop # حفظ النتيجة للمرحلة التالية
            st.success(f"✅ Recommended Crop: **{recommended_crop}**")
        else:
            st.error("Model files not found!")

    st.write("---")

    # الجزء الثاني: توصية السماد
    # نستخدم قائمة المحاصيل من الـ Label Encoder إذا كان متاحاً، وإلا نستخدم قائمة افتراضية
    crop_list = list(le_crop.classes_) if le_crop else ["Wheat", "Rice", "Maize"]
    selected_crop = st.selectbox("Select Crop for Fertilizer Analysis", crop_list)

    if st.button("Step 2: Recommend Fertilizer"):
        if fert_model:
            df_fert = pd.DataFrame([{
                "Nitrogen": Nitrogen, "Phosphorus": Phosphorus, "Potassium": Potassium,
                "pH": pH, "Rainfall": Rainfall, "Temperature": Temperature, 
                "Soil_color": Soil_color, "Crop": selected_crop
            }])
            
            pred_f = fert_model.predict(df_fert)
            fertilizer = le_fert.inverse_transform(pred_f)[0]
            st.success(f"🧪 Recommended Fertilizer: **{fertilizer}**")
        else:
            st.error("Fertilizer model not loaded!")