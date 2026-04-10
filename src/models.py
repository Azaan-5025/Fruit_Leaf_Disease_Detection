"""
Multi-model architectures for Fruit & Leaf Disease Detection.
Contains: CNN (MobileNetV2), ANN (MLP), ResNet50, SVM (feature extractor + classifier).
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Flatten, 
    BatchNormalization, Input, Concatenate
)
from tensorflow.keras.models import Model
import os

def load_class_names(path="class_names.txt"):
    """Load class names from file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return [
        "Apple___Black_rot", "Apple___Healthy", "Corn___Common_rust", "Corn___Healthy",
        "Grape___Black_rot", "Grape___Healthy", "Tomato___Bacterial_spot", "Tomato___Healthy"
    ]


DEFAULT_CLASSES = load_class_names()


def get_multi_input_layers(img_shape=(224, 224, 3)):
    """Creates the 4 input layers and their initial feature extractions."""
    # 1. Main image input (224x224)
    img_input = Input(shape=img_shape, name="img_input")
    
    # 2. Sakaguchi Tensors (8x8, 12x12, 16x16)
    t8_input = Input(shape=(8, 8, 3), name="t8_input")
    t12_input = Input(shape=(12, 12, 3), name="t12_input")
    t16_input = Input(shape=(16, 16, 3), name="t16_input")
    
    # Simple feature extraction for the small tensors
    t8_feat = Flatten()(t8_input)
    t8_feat = Dense(32, activation='relu')(t8_feat)
    
    t12_feat = Flatten()(t12_input)
    t12_feat = Dense(64, activation='relu')(t12_feat)
    
    t16_feat = Flatten()(t16_input)
    t16_feat = Dense(128, activation='relu')(t16_feat)
    
    return [img_input, t8_input, t12_input, t16_input], [t8_feat, t12_feat, t16_feat]


# =============================================================================
# Model 1: Multi-Input CNN (MobileNetV2)
# =============================================================================
def build_cnn_model(num_classes=len(DEFAULT_CLASSES), input_shape=(224, 224, 3)):
    """MobileNetV2-based multi-input CNN."""
    inputs, aux_feats = get_multi_input_layers(input_shape)
    img_input = inputs[0]
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=img_input)
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Concatenate with Sakaguchi features
    combined = Concatenate()([x] + aux_feats)
    
    x = Dense(256, activation='relu')(combined)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# =============================================================================
# Model 2: Multi-Input ANN (MLP)
# =============================================================================
def build_ann_model(num_classes=len(DEFAULT_CLASSES), input_shape=(224, 224, 3)):
    """Stronger Multi-Input Artificial Neural Network."""
    inputs, aux_feats = get_multi_input_layers(input_shape)
    img_input = inputs[0]
    
    # Use Global Average Pooling even for ANN if possible, or flatten
    x = Flatten()(img_input)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Concatenate all features
    combined = Concatenate()([x] + aux_feats)
    
    x = Dense(256, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# =============================================================================
# Model 3: Multi-Input ResNet50
# =============================================================================
def build_resnet50_model(num_classes=len(DEFAULT_CLASSES), input_shape=(224, 224, 3)):
    """ResNet50-based multi-input model with deeper fine-tuning."""
    inputs, aux_feats = get_multi_input_layers(input_shape)
    img_input = inputs[0]
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=img_input)
    
    # Unfreeze the top 50 layers for better feature adaptation
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Concatenate all features
    combined = Concatenate()([x] + aux_feats)
    
    x = Dense(512, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# =============================================================================
# Model 4: Multi-Input Feature Extractor (for SVM)
# =============================================================================
def build_feature_extractor(input_shape=(224, 224, 3)):
    """Multi-Input Feature Extractor for SVM."""
    inputs, aux_feats = get_multi_input_layers(input_shape)
    img_input = inputs[0]
    
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=img_input)
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Concatenate all features
    combined = Concatenate()([x] + aux_feats)
    
    return Model(inputs=inputs, outputs=combined)


# =============================================================================
# Model builder registry
# =============================================================================
MODEL_REGISTRY = {
    "CNN (MobileNetV2)": build_cnn_model,
    "ANN (MLP)": build_ann_model,
    "ResNet50": build_resnet50_model,
}

MODEL_NAMES = list(MODEL_REGISTRY.keys()) + ["SVM"]

