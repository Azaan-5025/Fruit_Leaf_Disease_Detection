import sys
import os
import numpy as np
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import sakaguchi_tensor_conversion, resize_image

def test_preprocessing():
    print("Testing Preprocessing...")
    # Create dummy image (100x100x3)
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test Resize
    resized = resize_image(dummy_img, (224, 224))
    assert resized.shape == (224, 224, 3), f"Resize failed: {resized.shape}"
    print("Resize: OK")
    
    # Test Sakaguchi Tensors
    tensors = sakaguchi_tensor_conversion(dummy_img)
    print(f"Sakaguchi Keys: {tensors.keys()}")
    
    assert '8x8' in tensors
    assert tensors['8x8'].shape == (8, 8, 3), f"8x8 shape wrong: {tensors['8x8'].shape}"
    assert '12x12' in tensors
    assert '16x16' in tensors
    print("Sakaguchi Tensors: OK")

if __name__ == "__main__":
    test_preprocessing()
