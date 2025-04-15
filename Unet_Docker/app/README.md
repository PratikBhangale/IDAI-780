# Tumor Segmentation API

A FastAPI application for brain tumor segmentation using deep learning.

## Features

- Accepts base64-encoded images
- Returns segmentation mask as base64-encoded image
- Provides tumor detection statement
- Optimized for Docker deployment

## Project Structure

```
app/
├── Dockerfile
├── requirements.txt
├── main.py                 # FastAPI application entry point
├── predict.py              # Model loading and prediction logic
├── models/                 # Directory to store the model file
│   └── best_Attresunet_scripted.pt
└── utils/                  # Utility functions
    └── image_utils.py      # Image processing utilities
```

## Setup

### Prerequisites

- Python 3.9+
- PyTorch
- FastAPI
- Docker (for containerized deployment)

### Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Copy your model file to the models directory:
   ```
   cp /path/to/your/model.pt models/best_Attresunet_scripted.pt
   ```

3. Run the application:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Access the API documentation at http://localhost:8000/docs

### Model Setup

Before running the application, you need to set up the model:

```
python setup_model.py --model /path/to/your/best_Attresunet_scripted.pt
```

This script will copy the model file to the `models` directory.

### Docker Deployment

#### Using Docker Compose (Recommended)

1. Set up the model as described above

2. Run with docker-compose:
   ```
   docker-compose up -d
   ```

3. Access the API documentation at http://localhost:8000/docs

#### Using Docker Directly

1. Build the Docker image:
   ```
   docker build -t tumor-segmentation-api .
   ```

2. Run the Docker container:
   ```
   docker run -p 8000:8000 -v ./models:/app/models tumor-segmentation-api
   ```

3. Access the API documentation at http://localhost:8000/docs

## API Usage

### Predict Endpoint

**URL**: `/predict`

**Method**: `POST`

**Request Body**:
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response**:
```json
{
  "segmentation_image": "base64_encoded_segmentation_mask",
  "tumor_detection": "The image has a tumor in it."
}
```

### Health Check Endpoint

**URL**: `/health`

**Method**: `GET`

**Response**:
```json
{
  "status": "healthy"
}
```

## Testing the API

A test script is provided to help you test the API. The script takes an input image, sends it to the API, and displays the results.

### Usage

```
python test_api.py --image /path/to/your/image.jpg --url http://localhost:8000 --output segmentation_result.png
```

Arguments:
- `--image`: Path to the input image (required)
- `--url`: API URL (default: http://localhost:8000)
- `--output`: Output image path (default: segmentation_result.png)

The script will:
1. Encode the input image to base64
2. Send a request to the API
3. Decode the response
4. Save the segmentation result
5. Display the input image and segmentation result side by side
6. Print the tumor detection statement

## Example Usage (Python Code)

```python
import requests
import base64

# Read image file and encode to base64
with open("image.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Make prediction request
response = requests.post(
    "http://localhost:8000/predict",
    json={"image": encoded_string}
)

# Get results
result = response.json()
segmentation_image = result["segmentation_image"]
tumor_detection = result["tumor_detection"]

# Save segmentation image
with open("segmentation.png", "wb") as fh:
    fh.write(base64.b64decode(segmentation_image))

print(tumor_detection)
