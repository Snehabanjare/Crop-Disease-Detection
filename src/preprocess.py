"""Image loading and preprocessing utilities."""

from pathlib import Path

import cv2
import numpy as np


def preprocess_image(
    image: str | Path | np.ndarray,
    image_size: tuple[int, int] = (224, 224),
    denoise: bool = False,
) -> np.ndarray:
    """Return an RGB image resized to ``image_size`` and normalized to [0, 1].

    Array inputs are expected to use RGB or RGBA channel order. ``image_size``
    is specified as (height, width).
    """
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size dimensions must be positive.")

    if isinstance(image, (str, Path)):
        path = Path(image)
        loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if loaded is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        rgb_image = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb_image = image
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError("Array input must be a grayscale, RGB, or RGBA image.")
    else:
        raise TypeError("image must be a path or a NumPy array.")

    if denoise:
        rgb_image = cv2.GaussianBlur(rgb_image, (3, 3), sigmaX=0)

    resized = cv2.resize(
        rgb_image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )
    return resized.astype(np.float32) / 255.0
