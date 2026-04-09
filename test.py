import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
import io

# 1. بناء الهيكل يدوياً (نفس اللي عملته في كولاب)
def build_structure():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(38, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=output)

# 2. محاولة استخراج الأوزان بطريقة إجبارية
def rescue_model(bad_model_path, save_name):
    # بناء موديل جديد نظيف
    new_model = build_structure()
    
    print(f"Attempting to rescue weights from {bad_model_path}...")
    
    try:
        # الحيلة هنا: نفتح الموديل مع تجاهل الأخطاء تماماً
        # ونمرر dict فارغ للـ custom_objects قد يساعد في تجاوز بعض القيود
        bad_model = load_model(bad_model_path, compile=False, safe_mode=False)
        
        # نقل الأوزان
        new_model.set_weights(bad_model.get_weights())
        
        # حفظ الموديل الجديد المتوافق مع جهازك
        new_model.save(save_name)
        print(f"✅ SUCCESS! Saved as {save_name}")
        return True
    except Exception as e:
        print(f"❌ Standard load failed: {e}")
        print("Trying alternative rescue method...")
        
        # محاولة أخيرة: تحميل الأوزان فقط من ملف الـ h5 (بافتراض أنه يحتوي عليها)
        try:
            new_model.load_weights(bad_model_path, by_name=True, skip_mismatch=True)
            new_model.save(save_name)
            print(f"✅ SUCCESS using load_weights! Saved as {save_name}")
            return True
        except Exception as e2:
            print(f"❌ All rescue attempts failed: {e2}")
            print("\nنصيحة: إذا فشل هذا الكود، فالحل الأخير هو تحديث TensorFlow عندك في الجهاز")
            print("pip install --upgrade tensorflow")
            return False

# شغل الكود (تأكد من المسار الصحيح للملف)
rescue_model("../inceptionv3.h5", "InceptionV3_Fixed.h5")