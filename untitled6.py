#!/usr/bin/env python3
"""
Plant Disease Classification using ResNet101V2
+ Confusion Matrix + Precision/Recall/F1
"""

import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix


def setup_kaggle():
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

    if os.path.exists("kaggle.json"):
        subprocess.run(["cp", "kaggle.json", os.path.expanduser("~/.kaggle/")], check=True)
        subprocess.run(["chmod", "600", os.path.expanduser("~/.kaggle/kaggle.json")], check=True)
    else:
        print("Warning: kaggle.json not found.")
        return False

    subprocess.run(["kaggle", "datasets", "download", "-d", "mohitsingh1804/plantvillage"], check=True)
    subprocess.run(["unzip", "-q", "plantvillage.zip"], check=True)

    return True


def create_data_generators(train_dir, val_dir, img_size=(224, 224), batch_size=16):

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

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

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
        shuffle=False   # مهم جدًا للـ confusion matrix
    )

    return train_data, val_data


def build_resnet101v2_model(num_classes):

    base_model = ResNet101V2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model


def train_model(model, train_data, val_data, epochs=30):

    model.compile(
        optimizer=Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(
        "best_resnet101v2.keras",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=2,
        min_lr=1e-6
    )

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    return history


def evaluate_model(model, val_data):

    print("\nEvaluating model...")

    val_data.reset()

    predictions = model.predict(val_data)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_data.classes

    class_names = list(val_data.class_indices.keys())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)


def plot_training_history(history):

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.show()

    print("Final Train Accuracy:", history.history['accuracy'][-1])
    print("Final Val Accuracy:", history.history['val_accuracy'][-1])


def main():

    train_dir = "/content/PlantVillage/train"
    val_dir = "/content/PlantVillage/val"

    if not os.path.exists(train_dir):
        train_dir = "PlantVillage/train"
        val_dir = "PlantVillage/val"

    if not os.path.exists(train_dir):
        print("Dataset not found. Downloading...")
        setup_kaggle()

    print("Loading data...")
    train_data, val_data = create_data_generators(train_dir, val_dir)

    print(f"Classes: {train_data.num_classes}")

    print("\nBuilding ResNet101V2 model...")
    model = build_resnet101v2_model(train_data.num_classes)

    print("\nTraining...")
    history = train_model(model, train_data, val_data)

    best_model = load_model("best_resnet101v2.keras")
    print("\nBest model loaded")

    plot_training_history(history)

    # 🔥 التقييم الكامل
    evaluate_model(best_model, val_data)

    return model, history


if __name__ == "__main__":
    model, history = main()