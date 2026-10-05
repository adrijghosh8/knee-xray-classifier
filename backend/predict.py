import numpy as np
from tensorflow.keras.models import load_model # type: ignore

from preprocessing import load_data


MODEL_PATH = r"models\model_knee_02.h5"

CLASS_NAMES = [
    "Normal",
    "Doubtful",
    "Mild",
    "Moderate",
    "Severe"
]


model = load_model(MODEL_PATH)


def predict_image(image):

    processed_image = load_data(image)

    predictions = model.predict(processed_image, verbose=0)

    probabilities = predictions[0]

    predicted_index = np.argmax(probabilities)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(probabilities[predicted_index])

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
    }