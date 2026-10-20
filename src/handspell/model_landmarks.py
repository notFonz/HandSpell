from pathlib import Path, PosixPath
from urllib.request import urlretrieve

import numpy as np
from mediapipe import Image as mpImage, ImageFormat as mpImageFormat
from mediapipe import tasks
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker as PythonHandLandmarker
import os
from PIL import Image, ImageOps
from tqdm.auto import tqdm

ROOT = '../data/dataset/D01_megadataset'

LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"
LABEL_MAP = {letter: index for index, letter in enumerate(LETTERS)}

HAND_MODEL = "../models/hand_landmarker.task"
os.makedirs("../models", exist_ok=True)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

def create_landmarker(hand_model_path: str=HAND_MODEL,
                      num_hands: int=1,
                      min_hand_detection_confidence: float=0.5,
                      min_hand_presence_confidence: float=0.5,
                      running_mode: tasks.vision.RunningMode=tasks.vision.RunningMode.IMAGE
                      ) -> PythonHandLandmarker:
    if not Path(HAND_MODEL).exists():
        urlretrieve(MODEL_URL, HAND_MODEL)

    print("MediaPipe model:", HAND_MODEL)

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


def find_images(root: str=ROOT):
    root = Path(root)

    if not root.is_dir():
        raise FileNotFoundError(root)

    paths, labels = [], []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        letter = path.parent.name.upper()

        if letter in LABEL_MAP:
            paths.append(path)
            labels.append(LABEL_MAP[letter])

    if not paths:
        raise ValueError(f"No images in supported letter folders: {root}")

    return paths, np.array(labels, dtype=np.int64)


def read_rgb(path: PosixPath):
    with Image.open(path) as original:
        oriented = ImageOps.exif_transpose(original).convert("RGBA")
        background = Image.new(
            "RGBA", oriented.size, (255, 255, 255, 255)
        )
        image = Image.alpha_composite(
            background, oriented
        ).convert("RGB")

        # Limit large images while preserving aspect ratio.
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

        return np.array(image)


def get_features(rgb: np.array, landmarker: PythonHandLandmarker):
    image = mpImage(
        image_format=mpImageFormat.SRGB,
        data=np.ascontiguousarray(rgb),
    )

    result = landmarker.detect(image)

    if not result.hand_landmarks:
        return None, None

    points = np.array(
        [[point.x, point.y, point.z]
         for point in result.hand_landmarks[0]],
        dtype=np.float32,
    )

    # Express coordinates in consistent units.
    height, width = rgb.shape[:2]
    points *= np.array([width, height, width], dtype=np.float32)

    # Wrist becomes the origin.
    relative = points - points[0]

    # Normalize by wrist-to-middle-finger-knuckle distance.
    scale = np.linalg.norm(relative[9])

    if scale < 1e-6:
        return None, None

    features = (relative / scale).reshape(-1)

    # 21 points × 3 coordinates = 63 features.
    return features, result.hand_landmarks[0]

def extract_features(paths):
    # NaN rows preserve failures in their original positions.
    features = np.full((len(paths), 63), np.nan, dtype=np.float32)
    failures = []

    with create_landmarker() as landmarker:
        for index, path in enumerate(tqdm(paths)):
            try:
                rgb = read_rgb(path)
                vector, _ = get_features(rgb, landmarker)

                if vector is None:
                    failures.append((str(path), "No usable hand landmarks"))
                else:
                    features[index] = vector

            except (OSError, ValueError, RuntimeError) as error:
                failures.append((str(path), str(error)))

    return features, failures
