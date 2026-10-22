# import faulthandler, sys
# faulthandler.dump_traceback_later(20, exit=False, file=sys.stderr)
# Import order matters: torch (pulled in by ultralytics) must be loaded BEFORE
# tensorflow. In the Linux Docker image, loading TensorFlow first makes YOLO
# inference crash with "free(): invalid pointer" (both bundle native libraries
# that clash). Keep this first and don't let an import sorter move it.
# import torch  # noqa: F401  # isort: skip

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
import os
import aiofiles
from tempfile import TemporaryDirectory
from pathlib import Path
from PIL import Image
from shutil import rmtree, copyfileobj
import joblib
from xgboost import XGBClassifier
import threading

# from handspell.data_preprocessing import load_model as load_preprocessing_model

from handspell.model_landmarks import create_landmarker
from handspell.api_helper_functions import preprocess_and_predict_api
from handspell.text_to_image import text_to_image_dict, text_to_image_path, IMAGES_DIR

"""
    HandSpell API main module
    This module provides the main FastAPI application for the HandSpell API.
    It includes the endpoint for predicting letters from uploaded images.
"""

HAND_LANDMARKER_PATH = os.environ["HAND_LANDMARKER_PATH"]
ASL_MODEL_LANDMARKS_PATH = os.environ["ASL_MODEL_LANDMARKS_PATH"]
TEMPORARY_FOLDER = ".temporary/"

def reset_temp_folder():
    temp_folder = Path(TEMPORARY_FOLDER)
    rmtree(temp_folder, ignore_errors=True)  # no error if it doesn't exist
    temp_folder.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs before the API accepts requests

    print(f"Loading model from: {HAND_LANDMARKER_PATH}")
    app.state.hand_model = create_landmarker(
        HAND_LANDMARKER_PATH, download_if_needed=False
    )

    print(f"Loading ASL model from: {ASL_MODEL_LANDMARKS_PATH}")
    app.state.asl_model = joblib.load(ASL_MODEL_LANDMARKS_PATH)

    print("Models loaded successfully.")

    reset_temp_folder()

    app.state.letter_dict = text_to_image_dict(
        IMAGES_DIR
    )

    yield  # The API serves requests during this period

    # Runs when the API shuts down
    del app.state.asl_model
    del app.state.hand_model
    print("Model reference released.")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def root():
    return {"text": "Welcome to the HandSpell API"}

# Only one request at a time may use the MediaPipe HandLandmarker (it isn't thread-safe).
# FastAPI runs plain `def` endpoints in a threadpool, so a threading.Lock is needed here.
predict_lock = threading.Lock()

@app.post("/predict")
def predict(request: Request, file: UploadFile):
    # Use a temporary directory to store the uploaded image before processing it.
    with TemporaryDirectory() as temporary_directory:
        image_path = (
            Path(temporary_directory) / "handspell_uploaded_temporary_image.png"
        )

        with open(image_path, "wb") as destination:
            copyfileobj(file.file, destination)

        # Only the landmarker and classifier call needs to be serialized.
        with predict_lock:
            # Call the preprocess_and_predict_api function with the temporary image path and the loaded models.
            result = preprocess_and_predict_api(
                image_path=str(image_path),
                # image_path="data/raw_data/A05_notFonz_original/images/Afonso/H0.jpg",
                hand_landmarker_model=request.app.state.hand_model,
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


@app.get("/sign/{letter}")
def letter_to_image(request: Request, letter: str) -> FileResponse:
    # Gets letter as a variable on get and then on the function
    if len(letter) > 1:

        raise HTTPException(status_code=404,
                                    detail= f'Pumba does not recognize J and Z since these letters require motion, multiple letters or numbers :D')

    # Get the image path depending on the letter
    path_result = text_to_image_path(letter,
                                     mappingdict = request.app.state.letter_dict)

    # Handles the 500 error if path is none
    if path_result is None:

        raise HTTPException(status_code=404,
                            detail= f'Pumba does not recognize J and Z since these letters require motion, multiple letters or numbers :D')
    # Use FileResponse to get the image from the path, I don't know if this works on Docker
    else:
        image = FileResponse(path_result)

    return image
