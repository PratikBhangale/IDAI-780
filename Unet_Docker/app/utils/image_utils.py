import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

def decode_base64_to_image(base64_string):
    """
    Decode a base64 string to an image
    
    Args:
        base64_string (str): Base64 encoded image string
        
    Returns:
        numpy.ndarray: Decoded image as a numpy array
    """
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode the base64 image string")
    
    return image

def encode_image_to_base64(image):
    """
    Encode an image to a base64 string
    
    Args:
        image (numpy.ndarray): Image as a numpy array
        
    Returns:
        str: Base64 encoded image string
    """
    # Convert numpy array to PIL Image
    if len(image.shape) == 2:  # Grayscale
        pil_img = Image.fromarray(image)
    else:  # RGB
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Save to BytesIO object
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    
    # Encode to base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_str

def resize_image(image, target_size=(256, 256)):
    """
    Resize an image to the target size
    
    Args:
        image (numpy.ndarray): Image to resize
        target_size (tuple): Target size (width, height)
        
    Returns:
        numpy.ndarray: Resized image
    """
    return cv2.resize(image, target_size)

def normalize_image(image):
    """
    Normalize an image to [0, 1] range
    
    Args:
        image (numpy.ndarray): Image to normalize
        
    Returns:
        numpy.ndarray: Normalized image
    """
    return image.astype(np.float32) / 255.0
