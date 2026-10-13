from fastapi import FastAPI, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import tensorflow as tf
import os

from .preprocessing import preprocess_image
from .predict import predict_letter


"""
    HandSpell API main module
    This module provides the main FastAPI application for the HandSpell API.
    It includes the endpoint for predicting letters from uploaded images.
"""

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_baseline_model.keras")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs before the API accepts requests
    print(f"Loading model from: {MODEL_PATH}")

    app.state.model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")

    yield  # The API serves requests during this period

    # Runs when the API shuts down
    del app.state.model
    print("Model reference released.")

app = FastAPI(lifespan=lifespan)

# Runs entirely in the worker thread
def preprocess_and_predict(contents: bytes, model) -> dict:
    image_tensor = preprocess_image(contents)
    return predict_letter(model, image_tensor)


@app.post("/predict")
async def predict(request: Request, file: UploadFile):
    try:
        contents = await file.read()
    finally:
        await file.close()

    return await run_in_threadpool(
        preprocess_and_predict,
        contents,
        request.app.state.model
    )
