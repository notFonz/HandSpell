import numpy as np
import joblib
import os
# from handspell.model_landmarks import create_landmarker
from handspell.model_landmarks import LETTERS, PythonHandLandmarker, get_features

from xgboost import XGBClassifier
from xgboost.sklearn import XGBClassifier as sklearnXGBClassifier


ASL_MODEL = os.environ["ASL_MODEL_LANDMARKS_PATH"]

def load_asl_model(model_path: str=ASL_MODEL):
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

    features, landmarks, _ = get_features(str(image_path), hand_landmarker_model)
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
