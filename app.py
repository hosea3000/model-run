from flask import Flask, request, jsonify
import base64
import requests
import numpy as np
from typing import Dict, Any, Union
import io
from PIL import Image
import json
import logging

from image_vector_extractor import ImageVectorExtractor

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/v1/images/embeddings', methods=['POST'])
def create_embeddings():
    """
    OpenAI-compatible endpoint for image embeddings.
    Endpoint: POST /v1/images/embeddings
    """
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'error': {
                    'message': 'The image field is required',
                    'type': 'invalid_request_error'
                }
            }), 400

        image_input = data['image']

        if not image_input:
            return jsonify({
                'error': {
                    'message': 'Image data cannot be empty',
                    'type': 'invalid_request_error'
                }
            }), 400

        logger.info(f"Processing image input: {image_input}")

        extractor = ImageVectorExtractor()
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
            'model': 'image-embedding-model'
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in embeddings endpoint: {str(e)}")
        return jsonify({
            'error': {
                'message': f'An error occurred: {str(e)}',
                'type': 'internal_error'
            }
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'image-embedding-api'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)