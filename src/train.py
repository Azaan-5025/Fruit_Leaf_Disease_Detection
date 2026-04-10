import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
try:
    from src.model import build_model
except ImportError:
    from model import build_model

import argparse

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

def train(data_dir, output_model_path="best_model.h5"):
    """
    Trains the MobileNetV2 model on data found in data_dir.
    Structure of data_dir should be:
    data_dir/
       class_a/
       class_b/
       ...
    """
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    # Data Augmentation and generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        validation_split=0.2,
        fill_mode="nearest"
    )

    print(f"Loading data from {data_dir}...")
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Get number of classes found
    num_classes = train_generator.num_classes
    class_names = list(train_generator.class_indices.keys())
    print(f"Found {num_classes} classes: {class_names}")

    # Build Model
    model = build_model(num_classes=num_classes)

    # Callbacks
    checkpoint = ModelCheckpoint(
        output_model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train
    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    print(f"Training complete. Best model saved to {output_model_path}")
    
    # Save class names to a file for the app to read
    with open("class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print("Class names saved to class_names.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Plant Disease Detection Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    train(args.data_dir)
