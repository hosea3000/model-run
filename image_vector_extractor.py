#!/usr/bin/env python3
"""
Image Vector Extractor using CLIP-ViT-Base-Patch32
Extract image vectors from base64, local path, or URL images
"""

import torch
import base64
import io
import argparse
import requests
from PIL import Image
import numpy as np
import json
import os
from transformers import AutoProcessor, AutoModel

class ImageVectorExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP model and processor

        Args:
            model_name: Name of the CLIP model to use
        """
        print(f"Loading model: {model_name}")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Set model to evaluation mode
        self.model.eval()

        # Get the dimension of the image embeddings
        self.vector_dim = self.model.config.vision_config.hidden_size

        print(f"Model loaded successfully! Vector dimension: {self.vector_dim}")

    def load_image_from_base64(self, base64_string):
        """
        Load image from base64 string

        Args:
            base64_string: Base64 encoded image string

        Returns:
            PIL Image object
        """
        try:
            # Remove data URL prefix if present
            if base64_string.startswith("data:image/"):
                base64_string = base64_string.split(",")[1]

            # Decode base64
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image
        except Exception as e:
            raise ValueError(f"Failed to decode base64 image: {str(e)}")

    def load_image_from_path(self, image_path):
        """
        Load image from local file path

        Args:
            image_path: Local path to the image file

        Returns:
            PIL Image object
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = Image.open(image_path)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from path: {str(e)}")

    def load_image_from_url(self, image_url):
        """
        Load image from URL

        Args:
            image_url: URL of the image

        Returns:
            PIL Image object
        """
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()

            image = Image.open(io.BytesIO(response.content))

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from URL: {str(e)}")

    def load_image(self, image_input):
        """
        Load image from various input types

        Args:
            image_input: Can be base64 string, file path, or URL

        Returns:
            PIL Image object
        """
        print(f"Loading image: {image_input}")
        if isinstance(image_input, str):
            # Check if it's a base64 string
            if image_input.startswith("data:image/") or (
                len(image_input) > 20 and
                all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
                    for c in image_input[:20])
            ):
                return self.load_image_from_base64(image_input)
            # Check if it's a URL
            elif image_input.startswith(("http://", "https://")):
                return self.load_image_from_url(image_input)
            # Assume it's a file path
            else:
                return self.load_image_from_path(image_input)
        else:
            print(f"Error: Image input must be a string, got {type(image_input)}")
            raise ValueError("Image input must be a string")

    def extract_image_vector(self, image_input, normalize=True):
        """
        Extract image vector from image input

        Args:
            image_input: Can be base64 string, file path, or URL
            normalize: Whether to normalize the vector

        Returns:
            torch.Tensor: Image vector (1, 512) for CLIP-ViT-Base-Patch32
        """
        # Load image
        image = self.load_image(image_input)

        # Process image
        inputs = self.processor(
            text=[""],  # Empty text, we only want image features
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Extract image features
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Get the image embeddings from the last hidden state
            image_embeddings = outputs.image_embeds

        # Normalize if requested
        if normalize:
            image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)

        return image_embeddings

    def extract_vector_and_info(self, image_input, normalize=True):
        """
        Extract image vector along with image information

        Args:
            image_input: Can be base64 string, file path, or URL
            normalize: Whether to normalize the vector

        Returns:
            dict: {
                'vector': numpy array of the image vector,
                'image_size': tuple of (width, height),
                'image_mode': str,
                'vector_dim': int
            }
        """
        # Load image
        image = self.load_image(image_input)

        # Extract vector (pass the loaded image to avoid double loading)
        vector = self._extract_vector_from_image(image, normalize)

        # Convert to numpy array
        vector_np = vector.cpu().numpy()

        # Return comprehensive information
        return {
            'vector': vector_np,
            'image_size': image.size,
            'image_mode': image.mode,
            'vector_dim': self.vector_dim
        }

    def _extract_vector_from_image(self, image, normalize=True):
        """
        Extract image vector from a PIL Image object

        Args:
            image: PIL Image object
            normalize: Whether to normalize the vector

        Returns:
            torch.Tensor: Image vector (1, 512) for CLIP-ViT-Base-Patch32
        """
        # Process image
        inputs = self.processor(
            text=[""],  # Empty text, we only want image features
            images=image,
            return_tensors="pt",
            padding=True
        )

        # Extract image features
        with torch.no_grad():
            outputs = self.model(**inputs)

            # Get the image embeddings from the last hidden state
            image_embeddings = outputs.image_embeds

        # Normalize if requested
        if normalize:
            image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)

        return image_embeddings

    def save_vector(self, vector, file_path):
        """
        Save vector to file

        Args:
            vector: Image vector
            file_path: Path to save the vector
        """
        # Convert to numpy if it's a tensor
        if isinstance(vector, torch.Tensor):
            vector = vector.cpu().numpy()

        # Save as numpy file
        if file_path.endswith('.npy'):
            np.save(file_path, vector)
        # Save as JSON
        elif file_path.endswith('.json'):
            # Convert to list for JSON serialization
            vector_list = vector.tolist()
            with open(file_path, 'w') as f:
                json.dump({
                    'vector': vector_list,
                    'shape': vector.shape
                }, f)
        else:
            raise ValueError("Unsupported file format. Use .npy or .json")


def main():
    try:
        # Initialize extractor
        extractor = ImageVectorExtractor()
        vector_info = extractor.extract_vector_and_info("http://images.cocodataset.org/val2017/000000039769.jpg")

        print(f"Image Size: {vector_info['image_size']}")
        print(f"Image Mode: {vector_info['image_mode']}")
        print(f"Vector Dimension: {vector_info['vector_dim']}")
        print(f"Vector Shape: {vector_info['vector']}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())