# -*- coding: utf-8 -*-

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d mohitsingh1804/plantvillage
!unzip plantvillage.zip

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_dir = "/content/PlantVillage/train"
val_dir   = "/content/PlantVillage/val"

# ==============================
# Data Generator
# ==============================
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input

img_size = (299, 299)
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print("Classes:", train_data.num_classes)

# ==============================
# MODEL
# ==============================
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam

base_model = InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(299,299,3)
)

# Freeze most layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Fine-tune last layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Head
x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(5e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==============================
# CALLBACKS (IMPROVED EARLY STOP)
# ==============================
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(
    "best_inceptionv3.keras",
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ==============================
# TRAIN
# ==============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[checkpoint, early_stop, reduce_lr]
)

# ==============================
# EVALUATION
# ==============================
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

best_model = load_model("best_inceptionv3.keras")

preds = best_model.predict(val_data)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=list(val_data.class_indices.keys())
))

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_data.class_indices.keys(),
            yticklabels=val_data.class_indices.keys())

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'])
plt.show()

print("Final Train Accuracy:", history.history['accuracy'][-1])
print("Final Val Accuracy:", history.history['val_accuracy'][-1])