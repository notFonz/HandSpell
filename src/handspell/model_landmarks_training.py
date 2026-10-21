from pathlib import Path, PosixPath
from urllib.request import urlretrieve
import numpy as np
from mediapipe import Image as mpImage, ImageFormat as mpImageFormat
from mediapipe import tasks
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker as PythonHandLandmarker
import os
import pandas as pd
from PIL import Image, ImageOps
from joblib import load
from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier as sklearnXGBClassifier
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report

from handspell.model_landmarks import create_landmarker, get_features, LETTERS, LABEL_MAP



ROOT = '../data/dataset/D01_megadataset'

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

def extract_features(paths):
    # NaN rows preserve failures in their original positions.
    features = np.full((len(paths), 63), np.nan, dtype=np.float32)
    failures = []

    with create_landmarker() as landmarker:
        for index, path in enumerate(tqdm(paths)):
            try:
                vector, _, _ = get_features(path, landmarker)

                if vector is None:
                    failures.append((str(path), "No usable hand landmarks"))
                else:
                    features[index] = vector

            except (OSError, ValueError, RuntimeError) as error:
                failures.append((str(path), str(error)))

    return features, failures

def get_features_df(features, labels, paths):
    features_df = pd.DataFrame(features)

    dataset_labels = []
    users = []
    for p in paths:
        splits = p.name.split('__')
        dataset_labels += [splits[0]]
        if len(splits)>2:
            users += [splits[1]]
        else:
            users += [""]

    # Add label
    features_df["letter_index"] = labels

    # Add Dataset label
    features_df["dataset"] = dataset_labels

    # Add User/Person
    features_df["user"] = users

    # Add image path
    features_df["image_path"] = [str(p) for p in paths]

    return features_df


def evaluate_pipeline(classifier, features, labels, name):
    usable = np.isfinite(features).all(axis=1)

    # -1 means no prediction; always incorrect for our letter labels.
    predictions = np.full(len(labels), -1, dtype=np.int64)

    if usable.any():
        predictions[usable] = classifier.predict(
            features[usable]
        ).astype(np.int64)

    print(f"\n{name}")
    print(f"Images: {len(labels)}")
    print(f"Usable hand detection: {usable.mean():.2%}")
    print(f"Overall accuracy: {(predictions == labels).mean():.2%}")

    if usable.any():
        accuracy = accuracy_score(
            labels[usable], predictions[usable]
        )
        print(f"Accuracy on detected hands: {accuracy:.2%}")

    return predictions


if __name__ == "__main__":

    landmarker = create_landmarker(
        hand_model_path="models/hand_landmarker_latest.task",
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.1
    )

    paths, labels = find_images("data/dataset/D01_megadataset")

    features, failures = extract_features(paths)

    features_df = get_features_df(features, labels, paths)

    features_df.to_csv("data/preprocessed/handlandmarks_d01megadataset_v2.csv")
