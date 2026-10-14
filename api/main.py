from fastapi import FastAPI, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import tensorflow as tf
import os
import aiofiles

from .preprocessing import preprocess_image
from .predict import predict_letter

# from handspell.model import predict


"""
    HandSpell API main module
    This module provides the main FastAPI application for the HandSpell API.
    It includes the endpoint for predicting letters from uploaded images.
"""

MODEL_PATH = os.environ["MODEL_PATH"]
TEMPORARY_FOLDER = "temporary/"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs before the API accepts requests
    print(f"Loading model from: {MODEL_PATH}")

    app.state.model_img28_to_letter = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    if not os.path.isdir(TEMPORARY_FOLDER):
        os.mkdir(TEMPORARY_FOLDER)

    print("Model loaded successfully.")


    yield  # The API serves requests during this period

    # Runs when the API shuts down
    del app.state.model_img28_to_letter
    print("Model reference released.")

app = FastAPI(lifespan=lifespan)

# Runs entirely in the worker thread
def preprocess_and_predict(contents: bytes, model) -> dict:
    image_tensor = preprocess_image(contents)
    return predict_letter(model, image_tensor)

@app.get("/")
def root():
    return {"text": "welcome"}



@app.post("/predict")
async def predict(request: Request, file: UploadFile):
    async with aiofiles.open(f"{TEMPORARY_FOLDER}/{file.filename}", 'wb') as out_file:
        contents = await file.read()
        await out_file.write(contents)  # async write

    return await run_in_threadpool(
        preprocess_and_predict,
        contents,
        request.app.state.model_img28_to_letter
    )
