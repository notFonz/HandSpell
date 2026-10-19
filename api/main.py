# Import order matters: torch (pulled in by ultralytics) must be loaded BEFORE
# tensorflow. In the Linux Docker image, loading TensorFlow first makes YOLO
# inference crash with "free(): invalid pointer" (both bundle native libraries
# that clash). Keep this first and don't let an import sorter move it.
import torch  # noqa: F401  # isort: skip

from fastapi import FastAPI, Request, UploadFile, File
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import os
import aiofiles
from tempfile import TemporaryDirectory
from pathlib import Path
from PIL import Image
from shutil import rmtree

from handspell.data_preprocessing import load_model as load_preprocessing_model

from handspell.model import load_model, preprocess_and_predict_api

"""
    HandSpell API main module
    This module provides the main FastAPI application for the HandSpell API.
    It includes the endpoint for predicting letters from uploaded images.
"""

HAND_DETECTION_MODEL_PATH = os.environ["HAND_DETECTION_MODEL_PATH"]
LETTER_CLASSIFICATION_MODEL_PATH = os.environ["LETTER_CLASSIFICATION_MODEL_PATH"]
TEMPORARY_FOLDER = ".temporary/"

print("Hand detection model: ", HAND_DETECTION_MODEL_PATH)
print("ASL model: ", LETTER_CLASSIFICATION_MODEL_PATH)

def reset_temp_folder():
    temp_folder = Path(TEMPORARY_FOLDER)
    rmtree(temp_folder, ignore_errors=True)  # no error if it doesn't exist
    temp_folder.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs before the API accepts requests
    print(f"Loading model from: {LETTER_CLASSIFICATION_MODEL_PATH}")

    app.state.asl_model = load_model(
        LETTER_CLASSIFICATION_MODEL_PATH
    )
    app.state.hand_model = load_preprocessing_model(
        model_path=HAND_DETECTION_MODEL_PATH, download_if_needed=False
    )

    print("Models loaded successfully.")

    reset_temp_folder()

    yield  # The API serves requests during this period

    # Runs when the API shuts down
    del app.state.asl_model
    del app.state.hand_model
    print("Model reference released.")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"text": "Welcome to the HandSpell API"}


@app.post("/predict")
def predict(request: Request, file: UploadFile):
    # Use a temporary directory to store the uploaded image before processing it.
    with TemporaryDirectory() as temporary_directory:
        image_path = (
            Path(temporary_directory) / "handspell_uploaded_temporary_image.png"
        )

        # Open the uploaded image and convert it to RGB before saving it temporarily.
        with Image.open(file.file) as image:
            image.convert("RGB").save(image_path)

        # Call the preprocess_and_predict_api function with the temporary image path and the loaded models.
        result = preprocess_and_predict_api(
            image_path=str(image_path),
            hand_detection_model=request.app.state.hand_model,
            letter_classification_model=request.app.state.asl_model,
        )

    # After exiting the context manager, the temporary directory and image are automatically removed.
    return result


@app.post("/predict_and_save_temp")
async def predict_and_save(request: Request, file: UploadFile = File(...)):

    async with aiofiles.open(f"{TEMPORARY_FOLDER}/{file.filename}", 'wb') as out_file:
        contents = await file.read()
        await out_file.write(contents)  # async write

    return await run_in_threadpool(
        preprocess_and_predict_api,
        f"{TEMPORARY_FOLDER}/{file.filename}",
        request.app.state.hand_model,
        request.app.state.asl_model
    )

@app.delete("/clear_temp")
def clear_temp():
    reset_temp_folder()
    return "Temporary files trashed 🗑️✅"
