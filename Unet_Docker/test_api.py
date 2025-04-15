import requests
import base64
import argparse
import os
import matplotlib.pyplot as plt
from PIL import Image
import io

def encode_image_to_base64(image_path):
    """
    Encode an image file to base64 string
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def decode_base64_to_image(base64_string):
    """
    Decode a base64 string to a PIL Image
    
    Args:
        base64_string (str): Base64 encoded image string
        
    Returns:
        PIL.Image: Decoded image
    """
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def main():
    parser = argparse.ArgumentParser(description='Test the Tumor Segmentation API')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--url', type=str, default='http://localhost:8000', help='API URL')
    parser.add_argument('--output', type=str, default='segmentation_result.png', help='Output image path')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.image):
        print(f"Error: Input image not found at {args.image}")
        return
    
    # Encode image to base64
    print(f"Encoding image: {args.image}")
    base64_image = encode_image_to_base64(args.image)
    
    # Make prediction request
    print(f"Sending request to {args.url}/predict")
    try:
        response = requests.post(
            f"{args.url}/predict",
            json={"image": base64_image},
            timeout=30
        )
        response.raise_for_status()  # Raise exception for HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return
    
    # Get results
    result = response.json()
    segmentation_image_base64 = result["segmentation_image"]
    tumor_detection = result["tumor_detection"]
    
    # Decode and save segmentation image
    segmentation_image = decode_base64_to_image(segmentation_image_base64)
    segmentation_image.save(args.output)
    print(f"Segmentation image saved to {args.output}")
    
    # Print tumor detection result
    print(f"Tumor detection result: {tumor_detection}")
    
    # Display the results
    plt.figure(figsize=(12, 6))
    
    # Display input image
    plt.subplot(1, 2, 1)
    input_image = Image.open(args.image)
    plt.imshow(input_image)
    plt.title("Input Image")
    plt.axis('off')
    
    # Display output image
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_image, cmap='gray')
    plt.title(f"Segmentation Result\n{tumor_detection}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("comparison_result.png")
    print("Comparison image saved to comparison_result.png")
    plt.show()

if __name__ == "__main__":
    main()
