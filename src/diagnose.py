import os
import sys
import numpy as np
import tensorflow as tf

# Add current directory to path for src imports
sys.path.append(os.getcwd())

from src.preprocessing import preprocess_pipeline
from src.model import predict_batch, load_class_names
from src.models import build_cnn_model

def diagnose():
    print("--- Diagnostic Tool: Model Verification ---")
    
    # 1. Load Class Names
    class_names = load_class_names("class_names.txt")
    print(f"Loaded {len(class_names)} classes from class_names.txt")
    
    # 2. Check Directory Structure
    data_dir = "data/training_data"
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} not found")
        return
        
    actual_classes = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    print(f"Found {len(actual_classes)} actual class folders")
    
    if len(class_names) != len(actual_classes):
        print(f"WARNING: Count mismatch! {len(class_names)} vs {len(actual_classes)}")
    
    for i, (name1, name2) in enumerate(zip(class_names, actual_classes)):
        if name1 != name2:
            print(f"ERROR: Label shift at index {i}: '{name1}' vs '{name2}'")
            # We continue to see other shifts
            
    # 3. Load Model
    model_path = "models/cnn_mobilenetv2.h5"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 4. Test Sample Images (1 from each class)
    print("\nTesting 1 sample image from each class (first 5 and last 5 for brevity):")
    correct = 0
    total = 0
    
    test_indices = list(range(5)) + list(range(len(actual_classes)-5, len(actual_classes)))
    
    for idx in test_indices:
        class_name = actual_classes[idx]
        class_path = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            continue
            
        img_path = os.path.join(class_path, images[0])
        model_input, sakaguchi_tensors = preprocess_pipeline(img_path)
        
        # predict_batch expects a batch
        batch_images = np.array([model_input])
        batch_tensors = [sakaguchi_tensors]
        
        results = predict_batch(model, batch_images, batch_tensors, class_names=class_names)
        pred = results[0]
        
        is_correct = pred['disease'].replace(" ", "_").replace("(", "___").replace(")", "") == class_name
        # Simple heuristic check since the prediction display name 
        # is formatted: "Apple (Fresh)" vs folder "Apple___Fresh"
        
        # Better check: find index in class_names
        inputs = {
            "img_input": batch_images.astype('float32'),
            "t8_input": np.array([t['8x8'] for t in batch_tensors], dtype='float32'),
            "t12_input": np.array([t['12x12'] for t in batch_tensors], dtype='float32'),
            "t16_input": np.array([t['16x16'] for t in batch_tensors], dtype='float32')
        }
        raw_preds = model.predict(inputs, verbose=0)
        top_idx = np.argmax(raw_preds[0])
        pred_class = class_names[top_idx]
        
        success = (pred_class == class_name)
        if success: correct += 1
        total += 1
        
        status = "PASS" if success else "FAIL"
        print(f"[{status}] Folder: {class_name:<30} | Pred: {pred_class:<30} | Conf: {pred['confidence']:.2%}")

    print(f"\nFinal Diagnostic: {correct}/{total} correct on sample subset.")

if __name__ == "__main__":
    diagnose()
