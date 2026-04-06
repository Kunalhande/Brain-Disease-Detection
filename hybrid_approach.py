import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
DATASET_PATH = "./content/Training"

# Load and Preprocess Data with Augmentation
datagen = ImageDataGenerator(
    validation_split=0.2, 
    rescale=1.0 / 255.0,  # Normalize image pixels
    rotation_range=30,  # Random rotation
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Random shear
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill any missing pixels after transformation
)

# Training Tumor Data
train_gen_tumor = datagen.flow_from_directory(
    DATASET_PATH + "/Tumor",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    classes=["yes", "no"],
    seed=42,
)

# Validation Tumor Data
val_gen_tumor = datagen.flow_from_directory(
    DATASET_PATH + "/Tumor",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    classes=["yes", "no"],
    seed=42,
)

# Training Stroke Data
train_gen_stroke = datagen.flow_from_directory(
    DATASET_PATH + "/Stroke",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    classes=["yes", "no"],
    seed=42,
)

# Validation Stroke Data
val_gen_stroke = datagen.flow_from_directory(
    DATASET_PATH + "/Stroke",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    classes=["yes", "no"],
    seed=42,
)

# Compute class weights to handle imbalance
class_weights_tumor = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=train_gen_tumor.classes)
class_weights_stroke = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=train_gen_stroke.classes)

# Create Combined Generator
def combined_generator(tumor_gen, stroke_gen, class_weights_tumor, class_weights_stroke):
    while True:
        # Get the next batch from both tumor and stroke generators
        tumor_batch = next(tumor_gen)
        stroke_batch = next(stroke_gen)
        
        # Ensure batch sizes are the same
        batch_size = min(len(tumor_batch[0]), len(stroke_batch[0]))
        
        # Slice to ensure both batches are of equal size
        images = tumor_batch[0][:batch_size]  # Tumor images
        tumor_labels = tumor_batch[1][:batch_size]  # Tumor labels
        stroke_labels = stroke_batch[1][:batch_size]  # Stroke labels
        
        # Calculate class weights for each label
        tumor_class_weights = np.array([class_weights_tumor[int(label)] for label in tumor_labels])
        stroke_class_weights = np.array([class_weights_stroke[int(label)] for label in stroke_labels])
        
        # Return images and labels for both tasks with class weights
        yield images, {
            "tumor_output": tumor_labels,
            "stroke_output": stroke_labels
        }, {"tumor_output": tumor_class_weights, "stroke_output": stroke_class_weights}

# Ensure generators have the same batch size
train_gen_combined = combined_generator(train_gen_tumor, train_gen_stroke, class_weights_tumor, class_weights_stroke)
val_gen_combined = combined_generator(val_gen_tumor, val_gen_stroke, class_weights_tumor, class_weights_stroke)

# Build Hybrid Model using VGG16 as base
def create_hybrid_model():
    base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
    base_model.trainable = False  # Freeze the layers of VGG16

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)

    # Task-Specific Outputs
    tumor_output = layers.Dense(1, activation="sigmoid", name="tumor_output")(x)
    stroke_output = layers.Dense(1, activation="sigmoid", name="stroke_output")(x)

    model = models.Model(inputs=base_model.input, outputs=[tumor_output, stroke_output])
    return model

model = create_hybrid_model()
model.compile(
    optimizer="adam",
    loss={
        "tumor_output": "binary_crossentropy",
        "stroke_output": "binary_crossentropy",
    },
    metrics={
        "tumor_output": ["accuracy"],
        "stroke_output": ["accuracy"],
    },
)

model.summary()

# Callbacks
callbacks = [
    ModelCheckpoint("model/hybrid_model.keras", save_best_only=True, monitor="val_loss"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
]

# Train Model
history = model.fit(
    train_gen_combined,
    steps_per_epoch=len(train_gen_tumor),
    validation_data=val_gen_combined,
    validation_steps=len(val_gen_tumor),
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Plot Training History
def plot_history(history):
    plt.figure(figsize=(12, 6))
    # Tumor accuracy
    plt.plot(history.history["tumor_output_accuracy"], label="Tumor Train Accuracy")
    plt.plot(history.history["val_tumor_output_accuracy"], label="Tumor Validation Accuracy")
    # Stroke accuracy
    plt.plot(history.history["stroke_output_accuracy"], label="Stroke Train Accuracy")
    plt.plot(history.history["val_stroke_output_accuracy"], label="Stroke Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Loss curves
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Total Loss')
    plt.plot(history.history['tumor_output_loss'], label='Tumor Loss')
    plt.plot(history.history['stroke_output_loss'], label='Stroke Loss')
    plt.legend()
    plt.show()

plot_history(history)




# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt

# # Constants
# IMAGE_SIZE = (128, 128)
# BATCH_SIZE = 32
# EPOCHS = 15
# DATASET_PATH = "./content/Training"

# # Load and Preprocess Data
# datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0 / 255.0)

# # Training Tumor Data
# train_gen_tumor = datagen.flow_from_directory(
#     DATASET_PATH + "/Tumor",
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="binary",
#     subset="training",
#     classes=["yes", "no"],
#     seed=42,
# )

# # Validation Tumor Data
# val_gen_tumor = datagen.flow_from_directory(
#     DATASET_PATH + "/Tumor",
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="binary",
#     subset="validation",
#     classes=["yes", "no"],
#     seed=42,
# )

# # Training Stroke Data
# train_gen_stroke = datagen.flow_from_directory(
#     DATASET_PATH + "/Stroke",
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="binary",
#     subset="training",
#     classes=["yes", "no"],
#     seed=42,
# )

# # Validation Stroke Data
# val_gen_stroke = datagen.flow_from_directory(
#     DATASET_PATH + "/Stroke",
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode="binary",
#     subset="validation",
#     classes=["yes", "no"],
#     seed=42,
# )

# # Build Hybrid Model
# def create_hybrid_model():
#     input_layer = layers.Input(shape=(*IMAGE_SIZE, 3))

#     # Shared Layers
#     x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Flatten()(x)
#     shared_features = layers.Dense(256, activation="relu")(x)

#     # Task-Specific Outputs
#     tumor_output = layers.Dense(1, activation="sigmoid", name="tumor_output")(shared_features)
#     stroke_output = layers.Dense(1, activation="sigmoid", name="stroke_output")(shared_features)

#     model = models.Model(inputs=input_layer, outputs=[tumor_output, stroke_output])
#     return model

# model = create_hybrid_model()
# model.compile(
#     optimizer="adam",
#     loss={
#         "tumor_output": "binary_crossentropy",
#         "stroke_output": "binary_crossentropy",
#     },
#     metrics={
#         "tumor_output": ["accuracy"],
#         "stroke_output": ["accuracy"],
#     },
# )

# model.summary()

# # Callbacks
# callbacks = [
#     ModelCheckpoint("hybrid_model.keras", save_best_only=True, monitor="val_loss"),
#     ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
#     EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
# ]

# # Train Model
# history = model.fit(
#     x=train_gen_tumor,
#     validation_data=(val_gen_tumor, val_gen_stroke),
#     epochs=EPOCHS,
#     callbacks=callbacks,
# )

# # Plot Training History
# def plot_history(history):
#     plt.figure(figsize=(12, 6))
#     # Tumor accuracy
#     plt.plot(history.history["tumor_output_accuracy"], label="Tumor Train Accuracy")
#     plt.plot(history.history["val_tumor_output_accuracy"], label="Tumor Validation Accuracy")
#     # Stroke accuracy
#     plt.plot(history.history["stroke_output_accuracy"], label="Stroke Train Accuracy")
#     plt.plot(history.history["val_stroke_output_accuracy"], label="Stroke Validation Accuracy")
#     plt.title("Training and Validation Accuracy")
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.show()

# plot_history(history)

