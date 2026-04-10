import cv2
import numpy as np

def load_image(image_file, target_size=(224, 224)):
    """
    Loads an image from a file-like object (Streamlit upload) or path.
    Converts to RGB.
    """
    # If it's a file-like object (from Streamlit)
    if hasattr(image_file, 'read'):
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_file)
    
    if image is None:
        raise ValueError("Could not load image")
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def apply_noise_smoothing(image, kernel_size=(5, 5)):
    """
    Applies Gaussian Blur for noise smoothing.
    """
    return cv2.GaussianBlur(image, kernel_size, 0)

def normalize_image(image):
    """
    Normalizes image pixel values to [0, 1].
    """
    return image.astype(np.float32) / 255.0

def resize_image(image, size):
    """
    Resizes image to the given size.
    """
    return cv2.resize(image, size)

def sakaguchi_tensor_conversion(image):
    """
    Generates the specific tensor sizes mentioned in requirements:
    8x8, 12x12, 16x16.
    
    Returns a dictionary of these re-sized tensors.
    """
    tensors = {}
    sizes = [(8, 8), (12, 12), (16, 16)]
    
    for size in sizes:
        resized = resize_image(image, size)
        # Assuming the requirement "8x8x8" might imply 8x8 resize. 
        # If it implies channels, that requires feature extraction.
        # For now, we return the spatial resize.
        tensors[f"{size[0]}x{size[1]}"] = resized
        
    return tensors

def preprocess_pipeline(image_file, target_size=(224, 224)):
    """
    Full pipeline: Load -> Resize -> Smooth -> Normalize
    Matches the training behavior in train_all_models.py (ImageDataGenerator).
    """
    # 1. Load Original
    raw_image = load_image(image_file, target_size=None)
    
    # 2. Resize to target (e.g., 224x224) FIRST
    # This ensures a (5,5) blur kernel has the same relative effect as during training
    resized = resize_image(raw_image, target_size)
    
    # 3. Noise Smoothing (on 224x224)
    smoothed = apply_noise_smoothing(resized)
    
    # 4. Normalization [0, 1]
    model_input = normalize_image(smoothed)
    
    # 5. Sakaguchi Tensor Conversion
    # We generate them from the normalized, smoothed image
    sakaguchi_tensors = sakaguchi_tensor_conversion(model_input)
    
    return model_input, sakaguchi_tensors
