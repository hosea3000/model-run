from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import logging
from typing import Dict, Any, Union
import os
import sys
from contextlib import asynccontextmanager

# Add parent directory to path to import ImageVectorExtractor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_vector_extractor import ImageVectorExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic model for request
class ImageEmbeddingRequest(BaseModel):
    image: str

# Pydantic model for error response
class ErrorResponse(BaseModel):
    error: Dict[str, str]

# Pydantic model for embedding response
class EmbeddingResponse(BaseModel):
    object: str
    data: list
    model: str

# Global extractor instance
extractor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the image vector extractor when the app starts"""
    global extractor
    try:
        logger.info("Initializing ImageVectorExtractor...")
        extractor = ImageVectorExtractor()
        logger.info("ImageVectorExtractor initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize ImageVectorExtractor: {str(e)}")
        raise
    finally:
        # Cleanup if needed
        logger.info("Shutting down application")

app = FastAPI(
    title="Image Embedding API",
    description="OpenAI-compatible API for image embeddings using CLIP model",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/v1/images/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: ImageEmbeddingRequest):
    """
    OpenAI-compatible endpoint for image embeddings.
    Endpoint: POST /v1/images/embeddings
    """
    try:
        image_input = request.image

        if not image_input:
            raise HTTPException(
                status_code=400,
                detail="The image field is required"
            )

        logger.info(f"Processing image input: {image_input}")

        if extractor is None:
            raise HTTPException(
                status_code=500,
                detail="ImageVectorExtractor not initialized"
            )

        vector_info = extractor.extract_vector_and_info(image_input)

        print(f"Image vector shape: {vector_info['vector'].shape}")
        # Convert numpy array to list for JSON serialization
        vector_list = vector_info['vector'].tolist()
        response = {
            'object': 'list',
            'data': [
                {
                    'object': 'embedding',
                    'embedding': vector_list,
                    'index': 0
                }
            ],
            'model': 'openai/clip-vit-base-patch32'
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in embeddings endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "image-embedding-api"}

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=400,
        content={"error": {"message": str(exc), "type": "invalid_request_error"}}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "type": "internal_error"}}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)