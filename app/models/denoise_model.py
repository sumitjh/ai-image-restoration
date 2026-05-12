import numpy as np
import cv2


def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    if not 1 <= strength <= 30:
        raise ValueError("Strength must be between 1 and 30")
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
