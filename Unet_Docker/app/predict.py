import torch
import numpy as np
import cv2
from utils.image_utils import decode_base64_to_image, encode_image_to_base64, resize_image, normalize_image

# Global variable to store the loaded model
model = None
device = torch.device("cpu")

def load_model(model_path):
    """Load the model from the specified path"""
    global model
    if model is None:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
    return model

def preprocess_base64_image(base64_string, target_size=(256, 256)):
    """
    Decode base64 string and preprocess the image for the model
    """
    # Decode base64 string to image
    image = decode_base64_to_image(base64_string)
    
    # Convert to grayscale (1 channel)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image
    image = resize_image(image, target_size)
    
    # Normalize the image
    image = normalize_image(image)
    
    # Convert to PyTorch tensor and add batch and channel dimensions
    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    return image_tensor

def postprocess_output(output_tensor):
    """
    Postprocess the model output
    """
    # Convert to numpy and remove batch dimension
    output = output_tensor.squeeze(0).detach().cpu().numpy()
    
    # If output has multiple channels, take the argmax
    if output.shape[0] > 1:
        output = np.argmax(output, axis=0)
    else:
        # For binary segmentation, threshold the output
        output = output.squeeze(0)
        output = (output > 0.5).astype(np.float32)
    
    # Scale to [0, 255] for image saving
    output = (output * 255).astype(np.uint8)
    
    return output

def encode_output_image(output_image):
    """
    Encode the output image to base64 string
    """
    return encode_image_to_base64(output_image)

def check_tumor_presence(segmentation_mask):
    """
    Check if segmentation mask has any white pixels
    """
    # If there are any white pixels (value > 0), a tumor is present
    has_tumor = np.any(segmentation_mask > 0)
    
    if has_tumor:
        return "The image has a tumor in it."
    else:
        return "The image does not have a tumor in it."

def predict(base64_image, model_path="./models/best_Attresunet_scripted.pt"):
    """
    Main prediction function that:
    1. Loads the model (if not already loaded)
    2. Preprocesses the input image
    3. Runs inference
    4. Postprocesses the output
    5. Checks for tumor presence
    6. Returns segmentation result and tumor detection statement
    """
    # Load model (will use cached model if already loaded)
    model = load_model(model_path)
    
    # Preprocess the image
    input_tensor = preprocess_base64_image(base64_image)
    input_tensor = input_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Postprocess the output
    segmentation_mask = postprocess_output(output_tensor)
    
    # Check for tumor presence
    tumor_statement = check_tumor_presence(segmentation_mask)
    
    # Encode the segmentation mask to base64
    encoded_image = encode_output_image(segmentation_mask)
    
    return {
        "segmentation_image": encoded_image,
        "tumor_detection": tumor_statement
    }
