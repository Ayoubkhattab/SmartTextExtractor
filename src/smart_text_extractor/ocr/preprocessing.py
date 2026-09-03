"""Pre-processing pipeline (§7.1): Deskew, Binarization/Contrast, Noise Removal.

Operates on numpy arrays (BGR or grayscale) so callers control I/O — this
module has no opinion about file formats.
"""
from __future__ import annotations

import cv2
import numpy as np


def denoise(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE (adaptive histogram equalization) — helps low-contrast/aged
    documents (the "د. أحمد" persona's old scanned papers, §7.1) without
    blowing out already-good scans the way global equalization would.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew(image: np.ndarray) -> np.ndarray:
    """Corrects page skew via minAreaRect over foreground (ink) pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    inverted = cv2.bitwise_not(gray)
    thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess(image: np.ndarray) -> np.ndarray:
    """The full §7.1 pipeline: denoise -> contrast -> deskew."""
    return deskew(enhance_contrast(denoise(image)))
