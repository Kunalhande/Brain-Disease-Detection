# Imports
import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constants
BASE_DIR = "./content/Training"
BALANCED_DIR = "./content/Balanced"
CSV_PATH = "labels.csv"
resize_width, resize_height = 224, 224
BATCH_SIZE = 32
EPOCHS = 15

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
def balance_dataset(source_dir, target_dir, target_size):
    os.makedirs(target_dir, exist_ok=True)
    for class_label in ['yes', 'no']:
        src_class_dir = os.path.join(source_dir, class_label)
        tgt_class_dir = os.path.join(target_dir, class_label)
        os.makedirs(tgt_class_dir, exist_ok=True)
        
        # Copy existing images
        for file in os.listdir(src_class_dir):
            src_path = os.path.join(src_class_dir, file)
            tgt_path = os.path.join(tgt_class_dir, file)
            if not os.path.exists(tgt_path):
                os.rename(src_path, tgt_path)
        
        # Augment images if needed
        files = os.listdir(tgt_class_dir)
        if len(files) < target_size:
            for file in files:
                img = cv2.imread(os.path.join(tgt_class_dir, file))
                img = cv2.resize(img, (resize_width, resize_height))
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                
                for i in range(target_size - len(files)):
                    augmented_img = next(datagen.flow(img, batch_size=1))[0].astype(np.uint8)
                    new_file = f"aug_{i}_{file}"
                    cv2.imwrite(os.path.join(tgt_class_dir, new_file), augmented_img)

# Apply Balancing
balance_dataset(os.path.join(BASE_DIR, "Tumor"), os.path.join(BALANCED_DIR, "Tumor"), target_size=155)
balance_dataset(os.path.join(BASE_DIR, "Stroke"), os.path.join(BALANCED_DIR, "Stroke"), target_size=950)

# Load Balanced Dataset
filenames, tumor_labels, stroke_labels = [], [], []
for condition in ["Tumor", "Stroke"]:
    for class_label in ["yes", "no"]:
        class_dir = os.path.join(BALANCED_DIR, condition, class_label)
        for file in os.listdir(class_dir):
            filenames.append(os.path.join(condition, class_label, file))  # Include relative path
            tumor_labels.append(1 if condition == "Tumor" and class_label == "yes" else 0)
            stroke_labels.append(1 if condition == "Stroke" and class_label == "yes" else 0)

# Save CSV
df = pd.DataFrame({"filename": filenames, "tumor": tumor_labels, "stroke": stroke_labels})
df.to_csv(CSV_PATH, index=False)
# Split Data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Generators
train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_dataframe(
    train_df, BALANCED_DIR, x_col="filename", y_col=["tumor", "stroke"],
    target_size=(resize_width, resize_height), batch_size=BATCH_SIZE, class_mode="raw"
)

val_data = val_gen.flow_from_dataframe(
    val_df, BALANCED_DIR, x_col="filename", y_col=["tumor", "stroke"],
    target_size=(resize_width, resize_height), batch_size=BATCH_SIZE, class_mode="raw"
)

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
    loss=["binary_crossentropy", "binary_crossentropy"],
    metrics=["accuracy"]
)

# Callbacks
callbacks = [
    ModelCheckpoint("model/best_model.keras", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
]

# Train
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)

# Save History
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.show()

# Save Model
model.save("model/final_model.keras")
