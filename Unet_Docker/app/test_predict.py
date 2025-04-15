import unittest
import os
import base64
import numpy as np
from predict import load_model, preprocess_base64_image, postprocess_output, check_tumor_presence

class TestPredict(unittest.TestCase):
    """
    Unit tests for the predict module
    """
    
    def test_load_model(self):
        """Test model loading"""
        # Skip if model file doesn't exist
        model_path = "./models/best_Attresunet_scripted.pt"
        if not os.path.exists(model_path):
            self.skipTest(f"Model file not found at {model_path}")
        
        # Load model
        model = load_model(model_path)
        
        # Check if model is loaded
        self.assertIsNotNone(model)
    
    def test_preprocess_base64_image(self):
        """Test image preprocessing"""
        # Create a simple test image (10x10 white square)
        img = np.ones((10, 10, 3), dtype=np.uint8) * 255
        
        # Convert to base64
        import cv2
        _, buffer = cv2.imencode('.png', img)
        base64_string = base64.b64encode(buffer).decode('utf-8')
        
        # Preprocess image
        tensor = preprocess_base64_image(base64_string, target_size=(10, 10))
        
        # Check tensor shape and values
        self.assertEqual(tensor.shape, (1, 1, 10, 10))
        self.assertTrue(np.allclose(tensor.numpy(), 1.0))
    
    def test_postprocess_output(self):
        """Test output postprocessing"""
        # Create a simple test output tensor (binary mask)
        import torch
        output = torch.zeros((1, 1, 10, 10))
        output[0, 0, 2:8, 2:8] = 1.0  # 6x6 white square in the middle
        
        # Postprocess output
        mask = postprocess_output(output)
        
        # Check mask shape and values
        self.assertEqual(mask.shape, (10, 10))
        self.assertEqual(mask[5, 5], 255)  # Center should be white
        self.assertEqual(mask[0, 0], 0)    # Corner should be black
    
    def test_check_tumor_presence_positive(self):
        """Test tumor detection with tumor present"""
        # Create a mask with some white pixels (tumor present)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 255  # Single white pixel
        
        # Check tumor presence
        result = check_tumor_presence(mask)
        
        # Should detect tumor
        self.assertEqual(result, "The image has a tumor in it.")
    
    def test_check_tumor_presence_negative(self):
        """Test tumor detection with no tumor"""
        # Create a mask with no white pixels (no tumor)
        mask = np.zeros((10, 10), dtype=np.uint8)
        
        # Check tumor presence
        result = check_tumor_presence(mask)
        
        # Should not detect tumor
        self.assertEqual(result, "The image does not have a tumor in it.")

if __name__ == "__main__":
    unittest.main()
