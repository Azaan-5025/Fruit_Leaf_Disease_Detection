# Multi-Input Fruit and Leaf Disease Detection System

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
