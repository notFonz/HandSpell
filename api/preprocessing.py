import io

import cv2
import numpy as np
from PIL import Image
'''
    HandSpell API image preprocessing module
    This module provides a function to preprocess images for the HandSpell model.
    The preprocessing steps include converting the image to grayscale,
    resizing it to 28x28 pixels, normalizing the pixel values, and reshaping
    it to match the input shape expected by the model.
'''

def preprocess_image(contents: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(contents)) as image:
        grayscale = image.convert("L")

    pixels = np.array(grayscale)
    resized = cv2.resize(pixels, (28, 28))
    normalized = resized.astype(np.float32) / 255.0

    return normalized.reshape(1, 28, 28, 1)
