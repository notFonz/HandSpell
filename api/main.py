from fastapi import FastAPI, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import tensorflow as tf
import os
import aiofiles

from handspell.data_preprocessing import load_model as load_preprocessing_model
from handspell.model import preprocess_and_predict_api


"""
    HandSpell API main module
    This module provides the main FastAPI application for the HandSpell API.
    It includes the endpoint for predicting letters from uploaded images.
"""

HAND_DETECTION_MODEL_PATH = os.environ["HAND_DETECTION_MODEL_PATH"]
LETTER_CLASSIFICATION_MODEL_PATH = os.environ["LETTER_CLASSIFICATION_MODEL_PATH"]
TEMPORARY_FOLDER = "temporary/"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs before the API accepts requests
    print(f"Loading model from: {LETTER_CLASSIFICATION_MODEL_PATH}")

    app.state.model_letter_classification = tf.keras.models.load_model(
        LETTER_CLASSIFICATION_MODEL_PATH,
        compile=False
    )
    app.state.model_hand_detection = load_preprocessing_model(
            model_path=HAND_DETECTION_MODEL_PATH,
    )

    if not os.path.exists(TEMPORARY_FOLDER):
        os.mkdir(TEMPORARY_FOLDER)

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
    return {"text": "welcome"}



@app.post("/predict")
async def predict(request: Request, file: UploadFile):
    async with aiofiles.open(f"{TEMPORARY_FOLDER}/{file.filename}", 'wb') as out_file:
        contents = await file.read()
        await out_file.write(contents)  # async write

    return await run_in_threadpool(
        preprocess_and_predict_api,
        f"{TEMPORARY_FOLDER}/{file.filename}",
        request.app.state.model_hand_detection,
        request.app.state.model_letter_classification
    )
