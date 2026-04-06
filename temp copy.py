import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# GPU Setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Constants
BASE_DIR = "./content/Training"
BALANCED_DIR = "./content/Balanced"
CSV_PATH = "labels.csv"
resize_width, resize_height = 224, 224
BATCH_SIZE = 32
EPOCHS = 15
TARGET_SIZE_TUMOR = 155
TARGET_SIZE_STROKE = 961

# Data Augmentation
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=30,
    zoom_range=0.3,
    shear_range=0.2,
)

# Balance Dataset
def balance_class(src_class_dir, tgt_class_dir, target_size):
    os.makedirs(tgt_class_dir, exist_ok=True)
    files = os.listdir(src_class_dir)
    num_files = len(files)
    print(f"Class directory {src_class_dir} has {num_files} images.")

    for file in files:
        src_path = os.path.join(src_class_dir, file)
        tgt_path = os.path.join(tgt_class_dir, file)
        if not os.path.exists(tgt_path):
            os.rename(src_path, tgt_path)

    files = os.listdir(tgt_class_dir)
    if len(files) < target_size:
        for file in files:
            img = cv2.imread(os.path.join(tgt_class_dir, file))
            img = cv2.resize(img, (resize_width, resize_height))
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)

            for i in range(target_size - len(files)):
                if len(os.listdir(tgt_class_dir)) >= target_size:
                    break
                augmented_img = next(datagen.flow(img, batch_size=1))[0].astype(np.uint8)
                new_file = f"aug_{i}_{file}"
                augmented_file_path = os.path.join(tgt_class_dir, new_file)
                if not os.path.exists(augmented_file_path):
                    cv2.imwrite(augmented_file_path, augmented_img)

# Apply Dataset Balancing
balance_class(os.path.join(BASE_DIR, "Tumor", "yes"), os.path.join(BALANCED_DIR, "Tumor", "yes"), TARGET_SIZE_TUMOR)
balance_class(os.path.join(BASE_DIR, "Tumor", "no"), os.path.join(BALANCED_DIR, "Tumor", "no"), TARGET_SIZE_TUMOR)
balance_class(os.path.join(BASE_DIR, "Stroke", "yes"), os.path.join(BALANCED_DIR, "Stroke", "yes"), TARGET_SIZE_STROKE)
balance_class(os.path.join(BASE_DIR, "Stroke", "no"), os.path.join(BALANCED_DIR, "Stroke", "no"), TARGET_SIZE_STROKE)

# Load the Balanced Dataset into DataFrame
filenames, tumor_labels, stroke_labels = [], [], []
for condition in ["Tumor", "Stroke"]:
    for class_label in ["yes", "no"]:
        class_dir = os.path.join(BALANCED_DIR, condition, class_label)
        for file in os.listdir(class_dir):
            filenames.append(os.path.join(condition, class_label, file))
            tumor_labels.append(1 if condition == "Tumor" and class_label == "yes" else 0)
            stroke_labels.append(1 if condition == "Stroke" and class_label == "yes" else 0)

df = pd.DataFrame({"filename": filenames, "tumor": tumor_labels, "stroke": stroke_labels})
df.to_csv(CSV_PATH, index=False)

# Split Data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Custom Generator
def custom_generator(df, base_dir, batch_size, target_size):
    while True:
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            batch_images = []
            tumor_labels = []
            stroke_labels = []

            for _, row in batch_df.iterrows():
                img_path = os.path.join(base_dir, row["filename"])
                img = cv2.imread(img_path)
                img = cv2.resize(img, target_size)
                img = preprocess_input(img)
                batch_images.append(img)
                tumor_labels.append(row["tumor"])
                stroke_labels.append(row["stroke"])

            batch_images = np.array(batch_images)
            tumor_labels = np.array(tumor_labels)
            stroke_labels = np.array(stroke_labels)

            yield batch_images, {"tumor_output": tumor_labels, "stroke_output": stroke_labels}

train_data = custom_generator(train_df, BALANCED_DIR, BATCH_SIZE, (resize_width, resize_height))
val_data = custom_generator(val_df, BALANCED_DIR, BATCH_SIZE, (resize_width, resize_height))

# Steps Per Epoch
train_steps = len(train_df) // BATCH_SIZE
val_steps = len(val_df) // BATCH_SIZE

# Model Definition
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(resize_width, resize_height, 3))
for layer in base_model.layers[:-8]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.3)(x)
tumor_output = Dense(1, activation="sigmoid", name="tumor_output")(x)
stroke_output = Dense(1, activation="sigmoid", name="stroke_output")(x)

model = Model(inputs=base_model.input, outputs=[tumor_output, stroke_output])
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss={"tumor_output": "binary_crossentropy", "stroke_output": "binary_crossentropy"},
    metrics={"tumor_output": "accuracy", "stroke_output": "accuracy"}
)

# Callbacks
callbacks = [
    ModelCheckpoint("model/best_model.keras", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
]

# Train the Model
history = model.fit(
    train_data,
    validation_data=val_data,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save Model
model.save("model/final_model.keras")

# Plot Training History
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.show()
