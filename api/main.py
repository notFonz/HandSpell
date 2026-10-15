from fastapi import FastAPI, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import tensorflow as tf
import os
import aiofiles
from tempfile import TemporaryDirectory
from pathlib import Path
from PIL import Image

from handspell.data_preprocessing import load_model as load_preprocessing_model
from handspell.model import preprocess_and_predict_api, predict_image

"""
    HandSpell API main module
    This module provides the main FastAPI application for the HandSpell API.
    It includes the endpoint for predicting letters from uploaded images.
"""

HAND_DETECTION_MODEL_PATH = os.environ["HAND_DETECTION_MODEL_PATH"]
LETTER_CLASSIFICATION_MODEL_PATH = os.environ["LETTER_CLASSIFICATION_MODEL_PATH"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs before the API accepts requests
    print(f"Loading model from: {LETTER_CLASSIFICATION_MODEL_PATH}")

    app.state.model_letter_classification = tf.keras.models.load_model(
        LETTER_CLASSIFICATION_MODEL_PATH, compile=False
    )
    app.state.model_hand_detection = load_preprocessing_model(
        model_path=HAND_DETECTION_MODEL_PATH,
    )

    print("Model loaded successfully.")

    yield  # The API serves requests during this period

    # Runs when the API shuts down
    del app.state.model_letter_classification
    del app.state.model_hand_detection
    print("Model reference released.")


app = FastAPI(lifespan=lifespan)

# # Runs entirely in the worker thread
# def preprocess_and_predict(contents: bytes, model) -> dict:
#     image_tensor = preprocess_image(contents)
#     return predict_letter(model, image_tensor)


@app.get("/")
def root():
    return {"text": "Welcome to the HandSpell API"}


@app.post("/predict")
def predict(request: Request, file: UploadFile):
    # Kazim: Use a temporary directory to store the uploaded image before processing it.
    with TemporaryDirectory() as temporary_directory:
        image_path = (
            Path(temporary_directory) / "handspell_uploaded_temporary_image.png"
        )

        # Kazim: Open the uploaded image and convert it to RGB before saving it temporarily.
        with Image.open(file.file) as image:
            image.convert("RGB").save(image_path)

        # Kazim: Call the preprocess_and_predict_api function with the temporary image path and the loaded models.
        result = preprocess_and_predict_api(
            image_path=str(image_path),
            hand_detection_model=request.app.state.model_hand_detection,
            letter_classification_model=request.app.state.model_letter_classification,
        )

    # Kazim: After exiting the context manager, the temporary directory and image are automatically removed.
    return result
