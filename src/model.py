import tensorflow as tf
import numpy as np
import os
from src.models import (
    load_class_names, DEFAULT_CLASSES, 
    build_cnn_model, build_ann_model, build_resnet50_model
)

def load_trained_model(model_path, model_name=None):
    """
    Loads a trained Keras model from the given path.
    If the model_path doesn't exist, returns an untrained builder based on model_name.
    """
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Loaded {model_name or 'model'} from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
    
    # Fallback/Builder logic
    num_classes = len(DEFAULT_CLASSES)
    if model_name:
        if "ResNet" in model_name:
            return build_resnet50_model(num_classes=num_classes)
        elif "ANN" in model_name:
            return build_ann_model(num_classes=num_classes)
        else:
            return build_cnn_model(num_classes=num_classes)
    
    return build_cnn_model(num_classes=num_classes)

def predict_batch(model, images, tensors_batch, class_names=DEFAULT_CLASSES):
    """
    Predicts disease for a batch of images using multi-input architecture.
    images: numpy array of shape (N, 224, 224, 3)
    tensors_batch: list of dictionaries with '8x8', '12x12', '16x16' tensors
    """
    # Prepare auxiliary inputs as dictionary to match the names in src/models.py
    inputs = {
        "img_input": images.astype('float32'),
        "t8_input": np.array([t['8x8'] for t in tensors_batch], dtype='float32'),
        "t12_input": np.array([t['12x12'] for t in tensors_batch], dtype='float32'),
        "t16_input": np.array([t['16x16'] for t in tensors_batch], dtype='float32')
    }
    
    # Predict using multiple inputs
    predictions_raw = model.predict(inputs, verbose=0)
    results = []
    
    for preds in predictions_raw:
        top_index = int(np.argmax(preds))
        confidence = float(preds[top_index])
        predicted_label = class_names[top_index] if top_index < len(class_names) else "Unknown"
        
        # Format display name: "Banana___Formalin-mixed" -> "Banana (Formalin-mixed)"
        display_label = predicted_label.replace("___", " (") + ")" if "___" in predicted_label else predicted_label
        display_label = display_label.replace("_", " ") # Clean up remaining underscores
        
        status = "Healthy" if "healthy" in predicted_label.lower() or "fresh" in predicted_label.lower() else "Diseased"
        
        results.append({
            "disease": display_label,
            "confidence": confidence,
            "status": status,
            "raw_scores": preds # Full distribution for analysis
        })
        
    return results
