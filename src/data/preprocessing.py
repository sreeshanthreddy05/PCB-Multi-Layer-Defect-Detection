"""
Image preprocessing.

Owner: Person 1 (Data & Preprocessing)
"""

from typing import Any, Tuple

import numpy as np
from PIL import Image


def preprocess_image(image: Any, target_size: Tuple[int, int] = (224, 224)) -> Any:
    """
    Preprocess a single PCB image: resize, normalize, and convert to
    the internal image format used by the rest of the pipeline.

    This function must produce the SAME output format regardless of
    whether the image came from the dataset or (in the future) a
    physical camera, per the Future Hardware Compatibility requirement.

    Args:
        image: Raw input image (e.g. numpy array or PIL Image).
        target_size: Desired (width, height) of the output image.

    Returns:
        Preprocessed image ready for model input.
    """
    image = resize_image(image, target_size)
    return normalize_image(image)


def normalize_image(image: Any) -> Any:
    """
    Normalize pixel values (e.g. scale to [0, 1] or standardize using
    dataset mean/std).

    Args:
        image: Input image array.

    Returns:
        Normalized image array.
    """
    return image.astype(np.float32) / 255.0


def resize_image(image: Any, size: Tuple[int, int]) -> Any:
    """
    Resize an image to the given (width, height).

    Args:
        image: Input image array.
        size: Target (width, height).

    Returns:
        Resized image array.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return np.array(image.resize(size, Image.BILINEAR))
