from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Constants
TRAIN_DIR = "./content/Training"
RESIZE_WIDTH = 224
RESIZE_HEIGHT = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Function to load data generators
def create_generators(base_dir, validation_split=0.2):
    data_gen = ImageDataGenerator(
        validation_split=validation_split, preprocessing_function=preprocess_input
    )
    
    train_generator = data_gen.flow_from_directory(
        base_dir,
        target_size=(RESIZE_WIDTH, RESIZE_HEIGHT),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        shuffle=True,
        subset='training'
    )

    val_generator = data_gen.flow_from_directory(
        base_dir,
        target_size=(RESIZE_WIDTH, RESIZE_HEIGHT),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        shuffle=False,
        subset='validation'
    )
    
    return train_generator, val_generator

# Function to structure the model
def structure_model(input_shape, num_classes):
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

# Function to train and evaluate the model
def train_and_evaluate_model(dataset_name):
    print(f"\nTraining model for {dataset_name} detection")
    
    train_gen, val_gen = create_generators(os.path.join(TRAIN_DIR, dataset_name))
    model = structure_model((RESIZE_WIDTH, RESIZE_HEIGHT, 3), num_classes=2)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    
    callbacks = [
        ModelCheckpoint(f"model/{dataset_name.lower()}_best_model.keras", save_best_only=True, monitor="val_loss"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    predictions = model.predict(val_gen)
    true_labels = val_gen.classes
    predicted_labels = np.argmax(predictions, axis=1)
    
    cr = classification_report(true_labels, predicted_labels, target_names=list(train_gen.class_indices.keys()))
    print(f"{dataset_name} Classification Report:\n", cr)
    
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_gen.class_indices.keys(), yticklabels=train_gen.class_indices.keys())
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.show()
    
    return history

# Train and evaluate models for Stroke, Tumor, and Alzheimer
datasets = ["Stroke", "Tumor", "Alzheimer"]
histories = {dataset: train_and_evaluate_model(dataset) for dataset in datasets}
