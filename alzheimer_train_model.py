import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

print('Modules loaded')

# Check for GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Paths
data_dir = './content/Training/Alzheimer/Combined Dataset'

# Function to Load Data
def load_data(target_folder):
    filepath, labels = [], []
    all_folder_path = os.path.join(data_dir, target_folder)

    if os.path.isdir(all_folder_path):
        filelist = os.listdir(all_folder_path)
        for f in filelist:
            fpath = os.path.join(all_folder_path, f)
            if os.path.isdir(fpath):
                for image in os.listdir(fpath):
                    filepath.append(os.path.join(fpath, image))
                    labels.append(f)  # Folder name is the label

    return pd.DataFrame({"filepath": filepath, "labels": labels})

# Load Train & Test Data
train_df = load_data('train')
test_df = load_data('test')

# Data Distribution Visualization
def plot_data_distribution(df, title):
    count = df["labels"].value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=count.index, y=count.values, palette='viridis')
    plt.title(title)
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.show()

plot_data_distribution(train_df, 'Training Data Distribution')
plot_data_distribution(test_df, 'Testing Data Distribution')

# Train-Validation Split (Stratified)
train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['labels'])

print(f"Train Set: {train_df.shape}, Validation Set: {valid_df.shape}")

# Image Data Generator (with rescaling for normalization)
batch_size = 16
img_size = (224, 224)

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(train_df, x_col='filepath', y_col='labels',
                                              target_size=img_size, class_mode='categorical',
                                              batch_size=batch_size, shuffle=True, workers=4)

valid_gen = valid_datagen.flow_from_dataframe(valid_df, x_col='filepath', y_col='labels',
                                              target_size=img_size, class_mode='categorical',
                                              batch_size=batch_size, shuffle=True, workers=4)

test_gen = test_datagen.flow_from_dataframe(test_df, x_col='filepath', y_col='labels',
                                            target_size=img_size, class_mode='categorical',
                                            batch_size=batch_size, shuffle=False, workers=4)

# Display Sample Images
classes = list(train_gen.class_indices.keys())
images, labels = next(train_gen)

plt.figure(figsize=(12, 12))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(images[i])
    index = np.argmax(labels[i])
    plt.title(classes[index])
    plt.axis('off')
plt.show()

# Load EfficientNetB0 (Optimized for Speed)
base_model = keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze Base Layers

# Model Architecture
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(train_gen.class_indices), activation='softmax')
])

# Compile Model
model.compile(Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
epochs = 50  # Reduced from 100 to speed up training

steps_per_epoch = 100  # Added this line

history = model.fit(
    train_gen,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,  # Use the defined steps_per_epoch
    validation_data=valid_gen,
    callbacks=[early_stopping]
)

# Plot Training History
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

index_loss = np.argmin(val_loss)
val_lowest = val_loss[index_loss]
index_acc = np.argmax(val_acc)
acc_highest = val_acc[index_acc]

epochs_range = range(len(tr_acc))

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, tr_loss, 'r', label='Training Loss')
plt.plot(epochs_range, val_loss, 'g', label='Validation Loss')
plt.scatter(index_loss, val_lowest, s=150, c='blue', label=f'Best Epoch = {index_loss + 1}')
plt.title('Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, tr_acc, 'r', label='Training Accuracy')
plt.plot(epochs_range, val_acc, 'g', label='Validation Accuracy')
plt.scatter(index_acc, acc_highest, s=150, c='blue', label=f'Best Epoch = {index_acc + 1}')
plt.title('Training & Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Model Evaluation
test_steps = len(test_df) // batch_size
train_score = model.evaluate(train_gen, steps=test_steps)
valid_score = model.evaluate(valid_gen, steps=test_steps)
test_score = model.evaluate(test_gen, steps=test_steps)

print("Train Loss:", train_score[0], "Train Accuracy:", train_score[1])
print("Valid Loss:", valid_score[0], "Valid Accuracy:", valid_score[1])
print("Test Loss:", test_score[0], "Test Accuracy:", test_score[1])

# Predictions
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)

# Confusion Matrix
cm = confusion_matrix(test_gen.classes, y_pred)
labels = list(test_gen.class_indices.keys())

plt.figure(figsize=(10, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", linewidths=0.5)
plt.xlabel('\nPredicted Label')
plt.ylabel('Actual Label\n')
plt.show()

# Classification Report
print(classification_report(test_gen.classes, y_pred, target_names=labels))

# Ensure model save directory exists
model_save_path = 'model/updated_alzheimer_best_model.keras'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)

print(f"Model saved at {model_save_path}")
