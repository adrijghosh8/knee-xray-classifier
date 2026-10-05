from PIL import Image
import numpy as np

IMG_SIZE = (200,200)

def load_data(image: Image.Image) -> np.ndarray:
    image = image.convert("L")

    image = image.resize(IMG_SIZE)

    X = np.array(image)

    X = X / 255.0

    X = np.expand_dims(X, axis=-1)

    X = np.expand_dims(X, axis=0)

    return X