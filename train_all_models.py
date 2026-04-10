"""
Unified training script for all models.
Trains CNN (MobileNetV2), ANN, ResNet50, and SVM on the fruit/leaf disease dataset.
Saves models and comparison metrics.

Usage:
    python train_all_models.py --data_dir data/training_data --epochs 5
"""

import os
import sys
import json
import time
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

from src.models import (
    build_cnn_model, build_ann_model, build_resnet50_model,
    build_feature_extractor, load_class_names
)

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
OUTPUT_DIR = "models"


from src.preprocessing import sakaguchi_tensor_conversion, resize_image

def create_data_generators(data_dir, class_names=None):
    """Create train and validation data generators with multi-input support."""
    def sakaguchi_preprocessor(image):
        """Applies Sakaguchi smoothing (Gaussian Blur) to training images."""
        if image is None:
            return image
        # Gaussian blur with (5,5) kernel as per src/preprocessing.py
        # THIS IS APPLIED AFTER RESIZING BY flow_from_directory
        smoothed = cv2.GaussianBlur(image, (5, 5), 0)
        return smoothed

    datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=sakaguchi_preprocessor,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        validation_split=0.2,
        fill_mode="nearest"
    )

    train_iterator = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        classes=class_names
    )

    val_iterator = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        classes=class_names
    )

    return MultiInputGeneratorWrapper(train_iterator), MultiInputGeneratorWrapper(val_iterator)


class MultiInputGeneratorWrapper(tf.keras.utils.Sequence):
    """Wraps a standard ImageDataGenerator as a tf.keras.utils.Sequence for multi-input datasets."""
    def __init__(self, generator):
        self.generator = generator
        self.num_classes = generator.num_classes
        self.class_indices = generator.class_indices
        self.samples = generator.samples
        self.batch_size = generator.batch_size

    def __len__(self):
        return len(self.generator)

    def on_epoch_end(self):
        self.generator.on_epoch_end()

    def __getitem__(self, index):
        img_batch, labels = self.generator[index]
        
        t8_batch = []
        t12_batch = []
        t16_batch = []
        
        for img in img_batch:
            # We assume img is already normalized [0, 1] due to rescale=1./255
            tensors = sakaguchi_tensor_conversion(img)
            t8_batch.append(tensors['8x8'])
            t12_batch.append(tensors['12x12'])
            t16_batch.append(tensors['16x16'])
            
        return {
            "img_input": np.array(img_batch, dtype='float32'),
            "t8_input": np.array(t8_batch, dtype='float32'),
            "t12_input": np.array(t12_batch, dtype='float32'),
            "t16_input": np.array(t16_batch, dtype='float32')
        }, labels


def train_keras_model(model, model_name, train_gen, val_gen, epochs, save_path):
    """Train a Keras model and return history + metrics."""
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}\n")

    checkpoint = ModelCheckpoint(
        save_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )

    start_time = time.time()

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )

    train_time = time.time() - start_time

    # Get best metrics
    best_val_acc = max(history.history['val_accuracy'])
    best_train_acc = max(history.history['accuracy'])
    final_val_loss = min(history.history['val_loss'])

    print(f"\n{model_name} Results:")
    print(f"  Best Train Accuracy: {best_train_acc:.4f}")
    print(f"  Best Val Accuracy:   {best_val_acc:.4f}")
    print(f"  Training Time:       {train_time:.1f}s")

    return {
        "model_name": model_name,
        "train_accuracy": float(best_train_acc),
        "val_accuracy": float(best_val_acc),
        "val_loss": float(final_val_loss),
        "training_time_seconds": round(train_time, 1),
        "epochs_trained": len(history.history['accuracy']),
        "history": {
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']],
            "loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']],
        }
    }


def train_svm(train_gen, val_gen, feature_extractor, save_path):
    """Train SVM using extracted features from MobileNetV2."""
    print(f"\n{'='*60}")
    print(f"  Training: SVM (with MobileNetV2 features)")
    print(f"{'='*60}\n")

    # Extract features from training data
    print("Extracting training features...")
    train_features = []
    train_labels = []
    n_batches = len(train_gen)

    for i in range(min(n_batches, 200)):  # Limit batches for speed
        batch_inputs, batch_y = train_gen[i]
        features = feature_extractor.predict(batch_inputs, verbose=0)
        train_features.append(features)
        train_labels.append(np.argmax(batch_y, axis=1))

        if (i + 1) % 50 == 0:
            print(f"  Extracted {i+1}/{min(n_batches, 200)} batches...")

    train_features = np.vstack(train_features)
    train_labels = np.concatenate(train_labels)
    print(f"  Training features shape: {train_features.shape}")

    # Extract features from validation data
    print("Extracting validation features...")
    val_features = []
    val_labels = []
    n_val_batches = len(val_gen)

    for i in range(n_val_batches):
        batch_inputs, batch_y = val_gen[i]
        features = feature_extractor.predict(batch_inputs, verbose=0)
        val_features.append(features)
        val_labels.append(np.argmax(batch_y, axis=1))

    val_features = np.vstack(val_features)
    val_labels = np.concatenate(val_labels)
    print(f"  Validation features shape: {val_features.shape}")

    # Train SVM
    print("Training SVM classifier...")
    start_time = time.time()

    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, verbose=False)
    svm.fit(train_features, train_labels)

    train_time = time.time() - start_time

    # Evaluate
    train_pred = svm.predict(train_features)
    val_pred = svm.predict(val_features)
    train_acc = accuracy_score(train_labels, train_pred)
    val_acc = accuracy_score(val_labels, val_pred)

    print(f"\nSVM Results:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Val Accuracy:   {val_acc:.4f}")
    print(f"  Training Time:  {train_time:.1f}s")

    # Save SVM model
    with open(save_path, 'wb') as f:
        pickle.dump(svm, f)
    print(f"  Saved to {save_path}")

    return {
        "model_name": "SVM",
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "val_loss": 0.0,
        "training_time_seconds": round(train_time, 1),
        "epochs_trained": 1,
        "history": {
            "accuracy": [float(train_acc)],
            "val_accuracy": [float(val_acc)],
            "loss": [0.0],
            "val_loss": [0.0],
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Train all models for comparison")
    parser.add_argument("--data_dir", type=str, default="data/training_data",
                        help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs for deep learning models")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated models to train: cnn,ann,resnet50,svm or 'all'")
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Determine which models to train
    if args.models == "all":
        models_to_train = ["cnn", "ann", "resnet50", "svm"]
    else:
        models_to_train = [m.strip().lower() for m in args.models.split(",")]

    # Determine class names and order
    target_classes = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    
    # Create data generators with explicit classes
    print("Loading dataset...")
    train_gen, val_gen = create_data_generators(args.data_dir, class_names=target_classes)
    num_classes = train_gen.num_classes
    class_names = list(train_gen.class_indices.keys())
    print(f"Found {num_classes} classes: {class_names[:5]}...")

    # Save class names for global consistency
    with open("class_names.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")

    # Results collection
    all_results = []

    # --- Train CNN (MobileNetV2) ---
    if "cnn" in models_to_train:
        model = build_cnn_model(num_classes=num_classes)
        result = train_keras_model(
            model, "CNN (MobileNetV2)", train_gen, val_gen,
            args.epochs, os.path.join(OUTPUT_DIR, "cnn_mobilenetv2.h5")
        )
        all_results.append(result)
        del model
        tf.keras.backend.clear_session()

    # --- Train ANN ---
    if "ann" in models_to_train:
        model = build_ann_model(num_classes=num_classes)
        result = train_keras_model(
            model, "ANN (MLP)", train_gen, val_gen,
            args.epochs, os.path.join(OUTPUT_DIR, "ann_model.h5")
        )
        all_results.append(result)
        del model
        tf.keras.backend.clear_session()

    # --- Train ResNet50 ---
    if "resnet50" in models_to_train:
        model = build_resnet50_model(num_classes=num_classes)
        result = train_keras_model(
            model, "ResNet50", train_gen, val_gen,
            args.epochs, os.path.join(OUTPUT_DIR, "resnet50_model.h5")
        )
        all_results.append(result)
        del model
        tf.keras.backend.clear_session()

    # --- Train SVM ---
    if "svm" in models_to_train:
        feature_extractor = build_feature_extractor()
        result = train_svm(
            train_gen, val_gen, feature_extractor,
            os.path.join(OUTPUT_DIR, "svm_model.pkl")
        )
        all_results.append(result)
        del feature_extractor
        tf.keras.backend.clear_session()

    # --- Save comparison results ---
    results_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # --- Print final comparison ---
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Train Acc':>10} {'Val Acc':>10} {'Time (s)':>10}")
    print(f"{'-'*55}")
    for r in all_results:
        print(f"{r['model_name']:<25} {r['train_accuracy']:>10.4f} {r['val_accuracy']:>10.4f} {r['training_time_seconds']:>10.1f}")

    print(f"\nResults saved to {results_path}")
    print("All model files saved in models/ directory.")


if __name__ == "__main__":
    main()
