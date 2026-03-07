"""
Focus / sharpness scoring functions for microscopy images.

1. Variance of Laplacian (VoL) — second-derivative measure.
   Higher variance of the Laplacian response indicates more
   high-frequency detail, i.e. a sharper (better-focused) image.

2. Tenengrad — first-derivative (gradient) measure.
   Mean squared magnitude of Sobel gradients.  Sharper images
   produce stronger edge responses and higher scores.

3. Percentile VoL — 75th percentile of absolute Laplacian responses.
   More robust than full variance when the image contains mixed
   in-focus and out-of-focus regions; the high-percentile tail
   better captures the sharpest local structures.

4. Percentile Tenengrad — 75th percentile of squared Sobel gradient
   magnitudes.  Same rationale as (3): the upper tail of the gradient
   distribution is a reliable indicator of focal sharpness even when
   large uniform regions would otherwise dilute a mean-based score.

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


def percentile_vol(gray: np.ndarray, p: float = 75.0) -> float:
    """Compute the p-th percentile of the absolute Laplacian response.

    Unlike`variance_of_laplacian`, which summarises the full
    distribution, this function looks at the upper tail of absolute
    second-derivative values.  The high-percentile tail is driven by
    the sharpest edges in the image and is therefore less affected by
    large flat / uniform areas that would dilute a global variance score.

    Args:
        gray: 2-D uint8 array (single-channel grayscale image).
        p:    Percentile in the range [0, 100].  Default is 75.

    Returns:
        Scalar focus score (higher = sharper).
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.percentile(np.abs(laplacian), p))


def percentile_tenengrad(gray: np.ndarray, ksize: int, p: float = 75.0) -> float:
    """Compute the p-th percentile of the squared Sobel gradient magnitudes.

    Unlike`tenengrad`, which uses the *mean* of *Gx² + Gy²*, this
    function reports the upper-tail percentile of that distribution.  In
    microscopy tiles that contain both background and specimen regions the
    mean can be suppressed by low-gradient background pixels; the
    high-percentile value isolates the sharpest specimen edges and is a
    more discriminative focal-plane selector.

    Args:
        gray:  2-D uint8 array (single-channel grayscale image).
        ksize: Sobel kernel size (must be 1, 3, 5, or 7).
        p:     Percentile in the range [0, 100].  Default is 75.

    Returns:
        Scalar focus score (higher = sharper).
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    gradient_magnitude_sq = gx ** 2 + gy ** 2
    return float(np.percentile(gradient_magnitude_sq, p))