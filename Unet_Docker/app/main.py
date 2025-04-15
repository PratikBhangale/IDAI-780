from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from predict import predict

app = FastAPI(
    title="Tumor Segmentation API",
    description="API for segmenting brain tumors in MRI images",
    version="1.0.0"
)

class ImageRequest(BaseModel):
    """Request model for the prediction endpoint"""
    image: str  # Base64 encoded image

class PredictionResponse(BaseModel):
    """Response model for the prediction endpoint"""
    segmentation_image: str  # Base64 encoded segmentation mask
    tumor_detection: str  # Tumor detection statement

@app.post("/predict", response_model=PredictionResponse)
async def get_prediction(request: ImageRequest):
    """
    Endpoint for getting tumor segmentation predictions
    """
    try:
        # Call prediction function
        result = predict(request.image)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    # Start the server - these settings are for local development
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
