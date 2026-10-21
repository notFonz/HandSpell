from pathlib import Path, PosixPath
from urllib.request import urlretrieve
import numpy as np
import os
# from PIL import Image, ImageOps
import joblib
from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier as sklearnXGBClassifier
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mediapipe import Image as mpImage, ImageFormat as mpImageFormat
from mediapipe import tasks
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker as PythonHandLandmarker

from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier as sklearnXGBClassifier




LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"
LABEL_MAP = {letter: index for index, letter in enumerate(LETTERS)}

HAND_MODEL = os.environ["HAND_LANDMARKER_PATH"]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

ASL_MODEL_LANDMARKS = os.environ["ASL_MODEL_LANDMARKS_PATH"]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (9, 10), (10, 11), (11, 12),             # middle
    (13, 14), (14, 15), (15, 16),            # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]

def create_landmarker(hand_model_path: str=HAND_MODEL,
                      num_hands: int=1,
                      min_hand_detection_confidence: float=0.5,
                      min_hand_presence_confidence: float=0.5,
                      running_mode: tasks.vision.RunningMode=tasks.vision.RunningMode.IMAGE,
                      download_if_needed: bool=True
                      ) -> PythonHandLandmarker:
    if download_if_needed:
        if not Path(hand_model_path).exists():
            urlretrieve(MODEL_URL, hand_model_path)

        print("MediaPipe model:", hand_model_path)

    # Configure MediaPipe HandLandmarker
    options = tasks.vision.HandLandmarkerOptions(
        base_options=tasks.BaseOptions(
            model_asset_path=hand_model_path
        ),
        running_mode=running_mode,
        num_hands=num_hands,
        min_hand_detection_confidence=min_hand_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
    )

    return tasks.vision.HandLandmarker.create_from_options(options)


# def read_rgb(path: str):
#     with Image.open(path) as original:
#         oriented = ImageOps.exif_transpose(original).convert("RGBA")
#         background = Image.new(
#             "RGBA", oriented.size, (255, 255, 255, 255)
#         )
#         image = Image.alpha_composite(
#             background, oriented
#         ).convert("RGB")

#         # Limit large images while preserving aspect ratio.
#         image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

#         return np.array(image)


# def get_image_rgb(image_path):
#     rgb = read_rgb(image_path)
#     image = mpImage(
#         image_format=mpImageFormat.SRGB,
#         data=np.ascontiguousarray(rgb),
#     )
#     return image, rgb


def get_features(image_path, landmarker: PythonHandLandmarker):
    # image, rgb_image = get_image_rgb(image_path)

    image = mpImage.create_from_file(image_path)
    rgb_image = image.numpy_view()

    result = landmarker.detect(image)

    if not result.hand_landmarks:
        return None, None

    points = np.array(
        [[point.x, point.y, point.z]
         for point in result.hand_landmarks[0]],
        dtype=np.float32,
    )

    # Express coordinates in consistent units.
    height, width = rgb_image.shape[:2]
    points *= np.array([width, height, width], dtype=np.float32)

    # Wrist becomes the origin.
    relative = points - points[0]

    # Normalize by wrist-to-middle-finger-knuckle distance.
    scale = np.linalg.norm(relative[9])

    if scale < 1e-6:
        return None, None

    features = (relative / scale).reshape(-1)

    # 21 points × 3 coordinates = 63 features.
    return features, result.hand_landmarks[0], rgb_image


def plot_landmarks(rgb_image: np.array, landmarks: list) -> plt.figure:
    # Plot
    img = rgb_image
    h, w = img.shape[:2]

    f, ax = plt.subplots()
    ax.imshow(img)
    hand = landmarks

    xs = [lm.x * w for lm in hand]
    ys = [lm.y * h for lm in hand]

    for a, b in HAND_CONNECTIONS:
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color="white", linewidth=2)
    ax.scatter(xs, ys, s=20, c="cyan", zorder=3)


    ax.set_axis_off()
    return f


def load_asl_model(model_path: str=ASL_MODEL_LANDMARKS):
    model = joblib.load(model_path)
    return model

def predict_letter(scores: np.ndarray) -> dict:

    predicted_idx = int(np.argmax(scores))

    return {
        "letter": LETTERS[predicted_idx],
        "confidence": float(scores[predicted_idx]),
        "scores": scores.tolist()
    }


def preprocess_and_predict_api(image_path: str,
                               hand_landmarker_model: PythonHandLandmarker,
                               letter_classification_model: sklearnXGBClassifier,
                           ) -> dict:
    """Predicts a ASL letter (not J or Z) from an image an image from a filepath.
    Accepts jpg, png or jpeg files of any size.
    Expects 'keras' model.

    Returns:
     - Dictionary with the most probable letter and its probability
     - Array with the probabilities for each of the letters
    """
    default_result = {
        "letter": "NO HAND/SIGN",
        "confidence": None,
        "scores": [],
        "letters": list(LETTERS),
        "landmarks": []
    }

    features, landmarks = get_features(read_rgb(image_path), hand_landmarker_model)
    if features is None:
        return default_result
    else:
        reshaped_features = np.expand_dims(features, 0)
        usable = np.isfinite(reshaped_features).all()
        if not usable:
            return default_result
        else:
            probas = letter_classification_model.predict_proba(
                reshaped_features
            )[0]
            r = predict_letter(probas)
            return {**r, "letters": list(LETTERS), "landmarks": landmarks}
