"""
Focus / sharpness scoring functions for microscopy images.

1. Variance of Laplacian (VoL) — second-derivative measure.
   Higher variance of the Laplacian response indicates more
   high-frequency detail, i.e. a sharper (better-focused) image.

2. Tenengrad — first-derivative (gradient) measure.
   Mean squared magnitude of Sobel gradients.  Sharper images
   produce stronger edge responses and higher scores.

"""

import cv2
import numpy as np


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Compute the Variance of the Laplacian.

    Args:
        gray: 2-D uint8 array (single-channel grayscale image).

    Returns:
        Scalar focus score (higher = sharper).
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def tenengrad(gray: np.ndarray, ksize) -> float:
    """Compute the Tenengrad focus score.

    The score is the mean of *Gx² + Gy²* where *Gx* and *Gy*
    are the Sobel gradient responses.

    Args:
        gray:  2-D uint8 array (single-channel grayscale image).
        ksize: Sobel kernel size (must be 1, 3, 5, or 7). Higher values are sometimes
               preferred for better edge detection but are also more sensitive to noise.

    Returns:
        Scalar focus score (higher = sharper).
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    gradient_magnitude_sq = gx ** 2 + gy ** 2
    return float(gradient_magnitude_sq.mean())
