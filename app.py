import streamlit as st
import numpy as np
import tensorflow as tf
import json
import os
import pickle
from src.preprocessing import preprocess_pipeline
from src.models import (
    build_cnn_model, build_ann_model, build_resnet50_model,
    build_feature_extractor, load_class_names, DEFAULT_CLASSES, MODEL_NAMES
)
from src.model import predict_batch, load_trained_model
import time

st.set_page_config(
    page_title="Fruit & Leaf Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# --- Custom CSS for Aesthetics ---
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50; 
        color: white; 
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
    }
    .card {
        background-color: white;
        color: #31333F;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .disease-red {
        color: #ff4b4b;
        font-weight: bold;
    }
    .healthy-green {
        color: #28a745;
        font-weight: bold;
    }
    .uncertain-gray {
        color: #ffc107;
        font-weight: bold;
    }
    .comparison-bar {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        padding: 8px 12px;
        border-radius: 6px;
        margin: 4px 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🌿 Multi-Input Fruit & Leaf Disease Detection")
st.markdown("### Automated Deep Learning System using Sakaguchi-based Preprocessing")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("System Config")

# Model selector
MODEL_OPTIONS = {
    "CNN (MobileNetV2)": {"file": "models/cnn_mobilenetv2.h5", "legacy": "best_model.h5", "type": "keras"},
    "ANN (MLP)": {"file": "models/ann_model.h5", "type": "keras"},
    "ResNet50": {"file": "models/resnet50_model.h5", "type": "keras"},
    "SVM": {"file": "models/svm_model.pkl", "type": "svm"},
}

# All models always available
all_model_names = list(MODEL_OPTIONS.keys())

selected_model = st.sidebar.selectbox(
    "🤖 Select Model",
    all_model_names,
    help="Choose which model to use for predictions."
)

# Show training status for selected model
info = MODEL_OPTIONS[selected_model]
has_weights = os.path.exists(info["file"]) or (info.get("legacy") and os.path.exists(info.get("legacy", "")))
if not has_weights and selected_model != "SVM":
    st.sidebar.warning(f"⚠️ {selected_model} not yet trained. Using ImageNet weights.")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.45,
    help="Minimum confidence to classify a disease."
)

st.sidebar.markdown("---")
st.sidebar.info("Supported: CNN, ANN, ResNet50, SVM")


# --- Model Loading (Cached) ---
@st.cache_resource
def load_keras_model(model_path, model_name):
    """Load a Keras/TF model using the unified loading logic."""
    return load_trained_model(model_path, model_name)


@st.cache_resource
def load_svm_model(model_path):
    """Load SVM model and feature extractor."""
    try:
        with open(model_path, 'rb') as f:
            svm = pickle.load(f)
        feature_extractor = build_feature_extractor()
        print(f"Loaded SVM from {model_path}")
        return svm, feature_extractor
    except Exception as e:
        st.error(f"Error loading SVM: {e}")
        return None, None


def load_all_available_models():
    """Load all trained models for consensus analysis."""
    loaded_models = {}
    for name, info in MODEL_OPTIONS.items():
        if info["type"] == "svm":
            if os.path.exists(info["file"]):
                loaded_models[name] = load_svm_model(info["file"])
        else:
            model_path = info["file"]
            if not os.path.exists(model_path) and info.get("legacy"):
                model_path = info["legacy"]
            if os.path.exists(model_path):
                loaded_models[name] = load_keras_model(model_path, name)
    return loaded_models


def predict_with_svm(svm, feature_extractor, images, tensors_batch, class_names=DEFAULT_CLASSES):
    """Make predictions using SVM model with multi-input feature extraction."""
    # Prepare auxiliary inputs
    t8_batch = np.array([t['8x8'] for t in tensors_batch])
    t12_batch = np.array([t['12x12'] for t in tensors_batch])
    t16_batch = np.array([t['16x16'] for t in tensors_batch])
    
    # Extract features using multi-input
    features = feature_extractor.predict([images, t8_batch, t12_batch, t16_batch], verbose=0)
    pred_indices = svm.predict(features)
    pred_probas = svm.predict_proba(features)

    results = []
    for idx, proba in zip(pred_indices, pred_probas):
        top_index = int(idx)
        confidence = float(proba[top_index])
        predicted_label = class_names[top_index] if top_index < len(class_names) else "Unknown"

        display_label = predicted_label.replace("___", " (") + ")" if "___" in predicted_label else predicted_label
        display_label = display_label.replace("_", " ")

        status = "Healthy" if "healthy" in predicted_label.lower() or "fresh" in predicted_label.lower() else "Diseased"

        results.append({
            "disease": display_label,
            "confidence": confidence,
            "status": status,
            "raw_scores": proba # Include probability distribution for analytics
        })

    return results


# Load the models
with st.spinner("Initializing models..."):
    all_models = load_all_available_models()

model_data = all_models.get(selected_model)
if selected_model == "SVM":
    svm_model, svm_extractor = model_data if isinstance(model_data, tuple) else (None, None)
    model_ready = svm_model is not None
else:
    model_ready = model_data is not None

if model_ready:
    st.sidebar.success(f"✅ {selected_model} loaded")
else:
    st.sidebar.error(f"❌ {selected_model} not available")

# --- Main Interface ---
uploaded_files = st.file_uploader(
    "Upload Fruit or Leaf Images (JPG, PNG)", 
    type=['jpg', 'png', 'jpeg'], 
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"Loaded {len(uploaded_files)} images.")
    
    if st.button("Analyze Images"):
        
        progress_bar = st.progress(0)
        
        # Prepare batch
        batch_images = []
        batch_filenames = []
        preprocessed_data = [] # Stores Sakaguchi Tensors
        
        for i, uploaded_file in enumerate(uploaded_files):
            model_input, sakaguchi_tensors = preprocess_pipeline(uploaded_file)
            batch_images.append(model_input)
            batch_filenames.append(uploaded_file.name)
            preprocessed_data.append(sakaguchi_tensors)
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        # Stack for batch prediction
        if batch_images:
            batch_stack = np.array(batch_images)
            
            # Prediction based on model type
            start_time = time.time()

            if selected_model == "SVM":
                if svm_model is not None:
                    predictions = predict_with_svm(svm_model, svm_extractor, batch_stack, preprocessed_data)
                else:
                    st.error("SVM model not trained. Please run training first.")
                    st.stop()
            else:
                predictions = predict_batch(model_data, batch_stack, preprocessed_data)

            processing_time = time.time() - start_time
            
            # --- Individual Model Prediction ---
            st.markdown(f"### 🤖 {selected_model} Predictions")
            
            if selected_model == "SVM":
                predictions = predict_with_svm(svm_model, svm_extractor, batch_stack, preprocessed_data)
            else:
                predictions = predict_batch(model_data, batch_stack, preprocessed_data)

            st.write(f"**Processing Time:** {processing_time:.2f}s")
            
            # Display Results
            cols = st.columns(3)
            
            for idx, (filename, pred, tensors) in enumerate(zip(batch_filenames, predictions, preprocessed_data)):
                col = cols[idx % 3]
                
                with col:
                    uploaded_files[idx].seek(0)
                    with st.container():
                        st.image(uploaded_files[idx], use_column_width=True, caption=filename)
                        
                        # Apply confidence threshold
                        if pred['confidence'] < confidence_threshold:
                            status_class = "uncertain-gray"
                            status_text = "Uncertain"
                            disease_text = "Low Confidence / Unknown"
                        else:
                            status_class = "healthy-green" if pred['status'] == "Healthy" else "disease-red"
                            status_text = pred['status']
                            disease_text = pred['disease']
                        
                        st.markdown(f"""
                        <div class="card">
                            <h4 class="{status_class}">{status_text}</h4>
                            <p><b>Disease:</b> {disease_text}</p>
                            <p><b>Confidence:</b> {pred['confidence']*100:.1f}%</p>
                            <p><b>Model Used:</b> {selected_model}</p>
                            <hr style="margin:10px 0">
                            <small>Sakaguchi Tensors:</small><br>
                            <code style="font-size: 8px">{list(tensors.keys())}</code>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # --- Technical Analysis Expander ---
                        with st.expander("🔍 Selected Model Analytics"):
                             if "raw_scores" in pred:
                                 st.write(f"**{selected_model} Confidence Breakdown:**")
                                 scores = pred['raw_scores']
                                 top_indices = np.argsort(scores)[::-1][:3]
                                 
                                 class_names = load_class_names()
                                 for rank, i in enumerate(top_indices):
                                     name = class_names[i].replace("___", " (").replace("_", " ")
                                     if "___" in class_names[i]: name += ")"
                                     score_val = float(scores[i])
                                     
                                     bar_color = "#28a745" if rank == 0 and score_val > 0.5 else "#ffc107" if rank == 0 else "#6c757d"
                                     st.markdown(f"""
                                     <div style="margin-bottom: 5px;">
                                         <div style="display: flex; justify-content: space-between; font-size: 11px;">
                                             <span>{name}</span>
                                             <span>{score_val*100:.1f}%</span>
                                         </div>
                                         <div style="background-color: #e9ecef; border-radius: 4px; height: 8px; width: 100%;">
                                             <div style="background-color: {bar_color}; width: {score_val*100}%; height: 100%; border-radius: 4px;"></div>
                                         </div>
                                     </div>
                                     """, unsafe_allow_html=True)
                             else:
                                 st.info("Additional analytics not available for this model type.")
else:
    st.info("Please upload images to begin analysis.")


# --- Model Comparison Section ---
st.markdown("---")
st.markdown("## 📊 Model Comparison")

comparison_path = "models/comparison_results.json"

if os.path.exists(comparison_path):
    with open(comparison_path, "r") as f:
        comparison_data = json.load(f)

    if comparison_data:
        # Comparison table
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### Accuracy Comparison")

            # Sort by validation accuracy
            sorted_data = sorted(comparison_data, key=lambda x: x['val_accuracy'], reverse=True)

            # Color palette for models
            colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

            for i, result in enumerate(sorted_data):
                name = result['model_name']
                val_acc = result['val_accuracy'] * 100
                train_acc = result['train_accuracy'] * 100
                color = colors[i % len(colors)]

                # Rank badge
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "

                st.markdown(f"""
                <div style="margin: 8px 0;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 20px;">{rank}</span>
                        <span style="font-weight: bold; min-width: 150px;">{name}</span>
                        <div style="flex-grow: 1; background: #e0e0e0; border-radius: 10px; height: 28px; position: relative;">
                            <div style="background: {color}; width: {val_acc}%; height: 100%; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                                <span style="color: white; font-weight: bold; font-size: 13px;">{val_acc:.1f}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Details")
            for result in sorted_data:
                with st.expander(result['model_name']):
                    st.metric("Validation Accuracy", f"{result['val_accuracy']*100:.1f}%")
                    st.metric("Training Accuracy", f"{result['train_accuracy']*100:.1f}%")
                    st.metric("Training Time", f"{result['training_time_seconds']:.0f}s")
                    st.metric("Epochs", result['epochs_trained'])

        # Best model highlight
        best = sorted_data[0]
        st.success(f"🏆 **Best Model: {best['model_name']}** with {best['val_accuracy']*100:.1f}% validation accuracy")

else:
    st.info("No comparison data available. Run `python train_all_models.py --data_dir data/training_data --epochs 5` to train all models and generate comparison.")
