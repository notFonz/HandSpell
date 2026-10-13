import numpy as np

'''
    HandSpell API prediction module
    This module provides a function to predict the letter represented by a preprocessed image tensor using the HandSpell model.
    The function returns a dictionary containing the predicted letter and the confidence score.
'''


def predict_letter(model, image_tensor: np.ndarray) -> dict:
    predictions = model.predict(image_tensor, verbose=0)

    scores = predictions[0]
    predicted_idx = int(np.argmax(scores))

    return {
        "letter": chr(predicted_idx + ord("A")),
        "confidence": float(scores[predicted_idx])
    }
