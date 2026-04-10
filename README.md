# Multi-Input Fruit & Leaf Disease Detection System

An AI-based web application to detect diseases in fruits and leaves using a multi-input, multi-model deep learning architecture with Sakaguchi-based preprocessing.

**Student:** Syed Azaan Hussain (24BCE5025)  
**Guide:** Pattabhiraman V  
**School of Computer Science and Engineering, VIT**

## Features
- **Multi-Image Upload**: Analyze multiple fruit/leaf images at once.
- **Sakaguchi Preprocessing Pipeline**: Tensor Conversion (8x8, 12x12, 16x16), Noise Smoothing, Normalization.
- **4 Deep Learning Models**: CNN (MobileNetV2), ANN (MLP), ResNet50, SVM — compared and evaluated.
- **Interactive UI**: Built with Streamlit for real-time feedback.

## Datasets

- **PlantVillage Dataset**: 50,000+ images of healthy & diseased plant leaves (15 classes).
  - [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Fruits Disease Dataset**: 5 fruit categories × 3 conditions = 15 classes.
  - [Fruits Disease Dataset on Kaggle](https://www.kaggle.com/datasets/sriramr/fruits-diseases)
- **Combined**: 30 classes | 82,000+ images | 224×224 RGB

## Project Structure

```
├── src/
│   ├── preprocessing.py      # Sakaguchi pipeline & tensor conversions
│   ├── models.py              # CNN, ANN, ResNet50, SVM model definitions
│   ├── model.py               # Prediction logic
│   └── train.py               # Training utilities
├── app.py                     # Streamlit web application
├── train_all_models.py        # Unified training script for all models
├── consolidate_data.py        # Dataset consolidation script
├── eda_outputs/               # EDA visualizations
├── results/                   # Model results, charts, comparisons
├── Project_Report.docx        # Full project report
└── requirements.txt
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Azaan-5025/Fruit_Leaf_Disease_Detection.git
   cd Fruit_Leaf_Disease_Detection
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Datasets** from the Kaggle links above and place in `data/`.
4. **Train Models**:
   ```bash
   python train_all_models.py --data_dir data/training_data --epochs 20
   ```
5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## EDA Outputs

Exploratory Data Analysis visualizations are available in `eda_outputs/`:
- Class distribution across categories
- Healthy vs Diseased class breakdown
- Complete class listing
- Dataset statistics summary

## Methodology

The system employs a **multi-input architecture** where each model receives:
1. Preprocessed 224×224 image
2. Three Sakaguchi tensors (8×8, 12×12, 16×16)

Four models are trained and compared:
| Model | Architecture | Key Details |
|-------|-------------|-------------|
| CNN | MobileNetV2 | Transfer learning, frozen base |
| ANN | MLP | Dense(1024→512→256→128), BatchNorm |
| ResNet50 | ResNet50 | Fine-tuned top 50 layers |
| SVM | MobileNetV2 + SVM | Feature extraction + RBF kernel |

## Results

| Model | Train Acc (%) | Val Acc (%) | Training Time |
|-------|:---:|:---:|:---:|
| **CNN (MobileNetV2)** | **90.1** | **85.1** | 5580s |
| SVM | 84.2 | 82.3 | 120s |
| ANN (MLP) | 42.9 | 39.7 | 320s |
| ResNet50 | 29.1 | 25.1 | 950s |

Detailed results with training curves and prediction examples are in `results/`.

## Comparison with Existing Work

| Method | Dataset | Accuracy (%) |
|--------|---------|:---:|
| AlexNet (Mohanty 2016) | PlantVillage (38 cls) | 99.3 |
| VGG16 (Too et al. 2019) | PlantVillage (38 cls) | 99.5 |
| InceptionV3 (Brahimi 2017) | PlantVillage (38 cls) | 99.2 |
| **Our CNN (MobileNetV2)** | **PV + Fruits (30 cls)** | **85.1** |
| **Our SVM** | **PV + Fruits (30 cls)** | **82.3** |

> Note: Existing methods evaluate on PlantVillage alone. Our system tackles the harder problem of combined multi-source datasets with 30 diverse classes.

## Results and Discussion

- **CNN (MobileNetV2)** achieved the best validation accuracy of 85.1%, confirming that transfer learning with pre-trained features combined with multi-resolution Sakaguchi inputs provides robust classification.
- **SVM** offered a strong non-deep-learning alternative at 82.3% with significantly faster inference.
- **ANN** and **ResNet50** underperformed due to high parameter counts relative to the training data available.
- The Sakaguchi preprocessing pipeline effectively captures multi-scale spatial features that complement the primary image features.

## Conclusion

This project successfully developed a multi-input, multi-model disease detection system for fruits and vegetable leaves. The CNN model with MobileNetV2 backbone achieved the best performance at 85.1% validation accuracy across 30 disease classes. The Streamlit web application provides a practical tool for real-time agricultural disease diagnosis.

## Future Enhancement

- Expand the dataset to include more crop varieties and disease types
- Implement ensemble methods combining multiple model predictions
- Deploy as a mobile application for field use by farmers
- Add disease severity estimation (mild, moderate, severe)
- Implement attention mechanisms to highlight diseased regions
- Integrate with IoT sensors for automated field monitoring

## References

1. Mohanty, S. P., et al. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.
2. Too, E. C., et al. (2019). A comparative study of fine-tuning deep learning models for plant disease identification. *Computers and Electronics in Agriculture*, 161, 272-279.
3. Brahimi, M., et al. (2017). Deep learning for tomato diseases. *Journal of Control and Decision*, 4(3), 179-195.
4. Ferentinos, K. P. (2018). Deep learning models for plant disease detection and diagnosis. *Computers and Electronics in Agriculture*, 145, 311-318.
5. Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*.

An AI-based web application to detect diseases in fruits and leaves using deep learning. The system allows multiple image uploads, processes them using a Sakaguchi-based preprocessing pipeline, and predicts diseases using a CNN model.

## Features
- **Multi-Image Upload**: Analyze multiple fruit/leaf images at once.
- **Sakaguchi Preprocessing Pipeline**:
    - Image Resizing (Tensor Conversion: 8x8, 12x12, 16x16)
    - Intensity Normalization
    - Noise Smoothing
    - Feature Scaling
- **Deep Learning Model**: Evaluation using a CNN (MobileNetV2 architecture).
- **Interactive UI**: Built with Streamlit for real-time feedback.

## Datasets

This project utilizes the following datasets for training and evaluation:

- **PlantVillage Dataset**: A comprehensive dataset with over 50,000 images of healthy and diseased plant leaves across various species.
  - [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Fruits Disease Dataset**: This dataset includes labeled images from five fruit categories: Apple, Banana, Grape, Mango, and Orange, each further divided into healthy and diseased classes.
  - [Fruits Disease Dataset on Kaggle](https://www.kaggle.com/datasets/sriramr/fruits-diseases)

## Setup Instructions

1. **Clone the Project** (if not already local)
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `src/preprocessing.py`: Implements the Sakaguchi pipeline and tensor conversions.
- `src/model.py`: Defines the CNN architecture and prediction logic.
- `app.py`: The main web application entry point.

## Usage
1. Open the app in your browser (usually http://localhost:8501).
2. Upload one or more images (JPG/PNG).
3. Click **Analyze Images**.
4. View the disease status, confidence score, and Sakaguchi tensor details.
